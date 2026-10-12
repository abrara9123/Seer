#include "visionassist/app/AppConfig.hpp"
#include "visionassist/vision/DepthEstimator.hpp"
#include "visionassist/vision/SceneSummary.hpp"
#include "visionassist/runtime/SpecialistWorker.hpp"
#include "visionassist/text/TextReader.hpp"
#include <atomic>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
Detection object(const std::string &name, cv::Rect box) {
    Detection result;
    result.rawLabel = result.label = name;
    result.clockDirection = "12 o'clock";
    result.box = box;
    return result;
}
void testRelativeDepthAndInvalidData() {
    cv::Mat depth(100, 200, CV_32F, cv::Scalar(8));
    const auto cup = object("cup", {10, 10, 60, 60});
    const auto chair = object("chair", {110, 10, 60, 60});
    depth(cup.box).setTo(2);
    require(nearestDetectedObject({cup, chair}, depth) == 0,
            "Closest-object comparison should use the matching object region");
    require(describeRelativeDepth({cup, chair}, depth).find("not a measured distance") !=
                std::string::npos,
            "Depth speech must retain uncertainty");
    depth(chair.box).setTo(2.1);
    require(!nearestDetectedObject({cup, chair}, depth),
            "Nearly equal depth estimates must not claim a reliable order");
    require(!nearestDetectedObject({cup}, depth),
            "A single object provides no comparative depth ranking");
    depth(cup.box).setTo(std::numeric_limits<float>::quiet_NaN());
    require(!DepthEstimator::sampleObject(depth, cup.box),
            "Invalid depth pixels must not become distance cues");
    require(!DepthEstimator::sampleObject(depth, {-100, -100, 5, 5}),
            "Out-of-frame boxes must not be sampled");
}
void testWorkerIsolationAndFrameBinding() {
    std::mutex mutex;
    std::vector<int> order;
    std::atomic<bool> started(false);
    SpecialistWorker worker(
        [&](const cv::Mat &image) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                order.push_back(image.at<unsigned char>(0, 0));
            }
            started = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            return cv::Mat(10, 10, CV_32F, cv::Scalar(2));
        },
        [&](const cv::Mat &) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                order.push_back(99);
            }
            return std::vector<TextLine>{{"CLEAR TEXT", 1, {0, 0, 10, 10}}};
        });
    const auto now = std::chrono::steady_clock::now();
    auto request = [&](int value) {
        return SpecialistWorker::Snapshot{
            cv::Mat(10, 10, CV_8U, cv::Scalar(value)), {}, static_cast<std::uint64_t>(value), now};
    };
    worker.submitDepth(request(1));
    const auto deadline = now + std::chrono::seconds(3);
    while (!started && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    require(started, "Depth job should start");
    worker.submitDepth(request(2));
    worker.submitDepth(request(3));
    cv::Mat textImage(10, 10, CV_8U, cv::Scalar(7));
    require(worker.requestText(textImage), "Text request should be admitted");
    require(!worker.requestText(textImage), "Held read key must not pile up OCR jobs");
    textImage.setTo(0);
    while (worker.depthResult().source.sceneSequence != 3 &&
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    worker.stop();
    require(order == std::vector<int>{1, 99, 3},
            "Text and depth should run independently, and old depth requests must be replaced");
    const auto depth = worker.depthResult();
    require(depth.isFreshFor(3, now + std::chrono::milliseconds(500)),
            "Depth result should bind to its source scene");
    require(!depth.isFreshFor(4, now + std::chrono::milliseconds(500)),
            "Depth from an old frame must not fuse with new object boxes");
    require(!depth.isFreshFor(3, now + std::chrono::seconds(4)), "Old depth must expire");
    require(worker.textResult().sourceFrame.at<unsigned char>(0, 0) == 7,
            "OCR request must own its immutable snapshot");
}
void testOfflineTextReading() {
    cv::Mat image(300, 1000, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::putText(image, "VISION ASSIST", {40, 110}, cv::FONT_HERSHEY_SIMPLEX, 2.4, {0, 0, 0}, 4,
                cv::LINE_AA);
    cv::putText(image, "CUP ON RIGHT", {40, 230}, cv::FONT_HERSHEY_SIMPLEX, 2.4, {0, 0, 0}, 4,
                cv::LINE_AA);
    const auto lines = TextReader().read(image);
    const auto spoken = TextReader::spokenText(lines);
    std::cout << "Offline OCR fixture: " << spoken << "\n";
    require(spoken.find("VISION ASSIST") != std::string::npos &&
                spoken.find("CUP ON RIGHT") != std::string::npos,
            "Apple Vision should read the local test fixture");
    require(TextReader::spokenText({{"uncertain", 0.1f, {}}}).find("couldn't read") !=
                std::string::npos,
            "Low-confidence text must not be spoken as fact");
}
int main() {
    try {
        testRelativeDepthAndInvalidData();
        testWorkerIsolationAndFrameBinding();
        testOfflineTextReading();
        std::cout << "PASS: specialist fusion, freshness, priority, depth uncertainty, and native "
                     "offline OCR\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
