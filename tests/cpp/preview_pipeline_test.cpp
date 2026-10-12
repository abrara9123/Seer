#include "visionassist/vision/ChangeMask.hpp"
#include "visionassist/vision/DepthEstimator.hpp"
#include "visionassist/vision/Detector.hpp"
#include "visionassist/vision/LocalDetectionWorker.hpp"
#include "visionassist/runtime/SpecialistWorker.hpp"
#include "visionassist/text/TextReader.hpp"
#include <chrono>
#include <iostream>
#include <thread>

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5)
        return 2;
    ObjectDetector detector;
    if (!detector.loadModel(argv[1], argv[2]))
        return 3;
    const cv::Mat source = cv::imread(argv[3]);
    if (source.empty())
        return 4;
    LocalDetectionWorker worker([&](const cv::Mat &image) { return detector.detect(image); },
                                std::chrono::milliseconds(1000));
    DepthEstimator depth;
    TextReader reader;
    const bool combined = argc == 5;
    if (combined && !depth.loadModel(argv[4]))
        return 7;
    SpecialistWorker specialists([&](const cv::Mat &image) { return depth.estimate(image); },
                                 [&](const cv::Mat &image) { return reader.read(image); });
    std::uint64_t submittedScene = 0;
    bool textSubmitted = false;
    cv::Mat textFixture(300, 1000, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::putText(textFixture, "VISION ASSIST", {40, 110}, cv::FONT_HERSHEY_SIMPLEX, 2.4, {0, 0, 0},
                4, cv::LINE_AA);
    ChangeMask mask;
    const auto started = std::chrono::steady_clock::now();
    auto nextFrame = started;
    int frames = 0, overBudget = 0;
    double maximumProcessingMs = 0;
    while (std::chrono::steady_clock::now() - started < std::chrono::seconds(6)) {
        const auto frameStarted = std::chrono::steady_clock::now();
        cv::Mat frame;
        const cv::Mat affine(cv::Matx23d(1, 0, std::sin(frames * 0.1) * 4, 0, 1, 0));
        cv::warpAffine(source, frame, affine, source.size(), cv::INTER_LINEAR,
                       cv::BORDER_REPLICATE);
        const auto changes = mask.update(frame);
        worker.submit(frame);
        const auto result = worker.result();
        if (combined && result.sequence != submittedScene && result.error.empty()) {
            submittedScene = result.sequence;
            specialists.submitDepth(
                {result.sourceFrame, result.detections, result.sequence, result.capturedAt});
        }
        if (combined && frames >= 30 && !textSubmitted) {
            textSubmitted = specialists.requestText(textFixture);
        }
        for (const auto &detection : result.detections) {
            cv::rectangle(frame, detection.box, {0, 255, 0}, 2);
            cv::putText(frame, detection.label,
                        {detection.box.x, std::max(20, detection.box.y - 5)},
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, {0, 255, 0}, 1);
        }
        cv::Mat preview;
        cv::cvtColor(changes.mask, preview, cv::COLOR_GRAY2BGR);
        const double processingMs = std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - frameStarted)
                                        .count();
        maximumProcessingMs = std::max(maximumProcessingMs, processingMs);
        if (processingMs > 33.33)
            ++overBudget;
        ++frames;
        nextFrame += std::chrono::microseconds(33333);
        if (nextFrame + std::chrono::microseconds(33333) < std::chrono::steady_clock::now()) {
            nextFrame = std::chrono::steady_clock::now();
        }
        std::this_thread::sleep_until(nextFrame);
    }
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    const auto result = worker.result();
    worker.stop();
    specialists.stop();
    if (combined) {
        const auto depthResult = specialists.depthResult();
        const auto textResult = specialists.textResult();
        std::cout << "Depth scene: " << depthResult.source.sceneSequence
                  << " | Depth: " << depthResult.milliseconds
                  << " ms | OCR: " << textResult.milliseconds << " ms\n";
        if (!depthResult.error.empty() || !textResult.error.empty() || depthResult.depth.empty() ||
            textResult.lines.empty()) {
            std::cerr << depthResult.error << " " << textResult.error << "\n";
            return 8;
        }
    }
    std::cout << "Preview processing FPS: " << frames / elapsed
              << " | Detection cycles: " << result.sequence
              << " | Latest inference including mask: " << result.inferenceMs << " ms\n"
              << "Preview frames over 33-ms budget: " << overBudget << "/" << frames
              << " | Slowest preview processing: " << maximumProcessingMs << " ms\n";
    if (!result.error.empty()) {
        std::cerr << result.error << "\n";
        return 5;
    }
    return result.sequence >= 5 && frames / elapsed >= 27 ? 0 : 6;
}
