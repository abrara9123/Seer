#include "visionassist/vision/Detector.hpp"
#include "visionassist/vision/LocalDetectionWorker.hpp"
#include "visionassist/vision/PresenceTracker.hpp"
#include "visionassist/audio/SpeechQueue.hpp"
#include <atomic>
#include <chrono>
#include <filesystem>
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

Detection object(const std::string &label, cv::Rect box) {
    Detection result;
    result.rawLabel = result.label = label;
    result.box = box;
    result.confidence = 0.9f;
    return result;
}

void testOutputFormatsAndCoordinates() {
    std::vector<std::string> names(365);
    for (int i = 0; i < 365; ++i)
        names[i] = "category " + std::to_string(i);
    names[10] = "Cup";
    names[364] = "Last Category";
    const int dimensions[] = {1, 369, 2};
    cv::Mat output(3, dimensions, CV_32F, cv::Scalar(0));
    const float rows[][5] = {{100, 180, 100, 120, 0.9f}, {500, 380, 100, 120, 0.8f}};
    for (int row = 0; row < 2; ++row) {
        for (int feature = 0; feature < 4; ++feature)
            output.ptr<float>()[feature * 2 + row] = rows[row][feature];
        output.ptr<float>()[(4 + (row == 0 ? 10 : 364)) * 2 + row] = rows[row][4];
    }
    auto results = decodeYoloDetections(output, {640, 480}, {640, 640}, names);
    require(results.size() == 2, "365-class raw output must produce both detections");
    require(results[0].label == "cup",
            "Category names should be normalized for speech and filtering");
    require(results[0].box == cv::Rect(50, 40, 100, 120),
            "Letterbox padding must be removed from coordinates");
    require(results[0].clockDirection == "10 o'clock", "Left object direction should be correct");
    require(results[1].rawLabel == "last category", "Final category ID must not be lost");
    require(results[1].clockDirection == "1 o'clock", "Right object direction should be correct");
    cv::Mat rawRows = cv::Mat(369, 2, CV_32F, output.ptr<float>()).t();
    auto rowResults = decodeYoloDetections(rawRows, {640, 480}, {640, 640}, names);
    require(rowResults.size() == 2 && rowResults[0].box == results[0].box,
            "Row-major raw scores should decode consistently");
    std::vector<std::string> incorrect(80, "wrong category");
    bool rejected = false;
    try {
        decodeYoloDetections(output, {640, 480}, {640, 640}, incorrect);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, "365 output with 80 labels must fail safely");

    const int endShape[] = {1, 5, 6};
    cv::Mat endOutput(3, endShape, CV_32F, cv::Scalar(0));
    const float endRows[][6] = {
        {50, 120, 150, 240, 0.9f, 10},
        {50, 120, 150, 240, 0.8f, 10},  // duplicate category
        {50, 120, 150, 240, 0.7f, 364}, // overlapping different category
        {-10, 60, 20, 110, 0.9f, 10},   // clipped frame edge
        {0, 0, 50, 50, 0.9f, 999}       // invalid category ID
    };
    for (int row = 0; row < 5; ++row)
        std::copy(endRows[row], endRows[row] + 6, endOutput.ptr<float>() + row * 6);
    results = decodeYoloDetections(endOutput, {640, 480}, {640, 640}, names);
    require(results.size() == 3,
            "Only duplicate same-category and invalid-ID boxes should be discarded");
    require(std::any_of(
                results.begin(), results.end(),
                [](const Detection &detection) { return detection.box == cv::Rect(0, 0, 20, 30); }),
            "Boxes must be clipped to the camera frame");
    endOutput.ptr<float>()[4] = std::numeric_limits<float>::quiet_NaN();
    endOutput.ptr<float>()[6] = std::numeric_limits<float>::infinity();
    require(decodeYoloDetections(endOutput, {640, 480}, {640, 640}, names).size() == 2,
            "Nonfinite confidence and coordinates must be discarded safely");

    const int cocoShape[] = {1, 84, 1};
    cv::Mat coco(3, cocoShape, CV_32F, cv::Scalar(0));
    coco.ptr<float>()[0] = 320;
    coco.ptr<float>()[1] = 320;
    coco.ptr<float>()[2] = 100;
    coco.ptr<float>()[3] = 100;
    coco.ptr<float>()[4 + 79] = 0.9f;
    std::vector<std::string> cocoNames(80, "coco object");
    require(decodeYoloDetections(coco, {640, 480}, {640, 640}, cocoNames).size() == 1,
            "Existing YOLOv8 fallback should still decode");
}

void testNewObjectAnnouncements() {
    PresenceTracker tracker;
    auto cup = object("cup", {10, 10, 100, 100});
    auto anotherCup = object("cup", {400, 10, 100, 100});
    require(tracker.update({cup}).size() == 1,
            "First visible object should be announced immediately");
    require(tracker.update({cup}).empty(), "Persistent object should not repeat every second");
    require(tracker.update({cup, anotherCup}).size() == 1,
            "Second object of same category must be announced");
    cup.box.x += 10;
    require(tracker.update({anotherCup, cup}).empty(),
            "Object motion and reordered predictions should not repeat speech");
    tracker.update({anotherCup});
    require(tracker.update({cup, anotherCup}).empty(),
            "One missed detection should not trigger duplicate speech");
    tracker.update({anotherCup});
    tracker.update({anotherCup});
    require(tracker.update({anotherCup, cup}).size() == 1,
            "An object returning after leaving should be announced");
    auto phone = object("phone", cup.box);
    require(tracker.update({anotherCup, cup, phone}).size() == 1,
            "New overlapping category must be announced");
}

void testOneSecondRefreshUsesNewestFrame() {
    using Clock = std::chrono::steady_clock;
    std::mutex mutex;
    std::vector<Clock::time_point> starts;
    std::vector<int> pixels;
    LocalDetectionWorker worker(
        [&](const cv::Mat &frame) {
            std::lock_guard<std::mutex> lock(mutex);
            starts.push_back(Clock::now());
            pixels.push_back(frame.at<unsigned char>(0, 0));
            return std::vector<Detection>{object("cup", {0, 0, 10, 10})};
        },
        std::chrono::milliseconds(1000));
    worker.submit(cv::Mat(10, 10, CV_8UC1, cv::Scalar(1)));
    const auto deadline = Clock::now() + std::chrono::seconds(2);
    while (worker.result().sequence == 0 && Clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    require(worker.result().sequence == 1,
            "First frame should be processed without a one-second startup delay");
    worker.submit(cv::Mat(10, 10, CV_8UC1, cv::Scalar(2)));
    worker.submit(cv::Mat(10, 10, CV_8UC1, cv::Scalar(3)));
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    require(worker.result().sequence == 1,
            "Submitting camera frames must not trigger continuous inference");
    while (worker.result().sequence < 2 && Clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    require(worker.result().sequence == 2, "Automatic refresh must happen without keyboard input");
    worker.stop();
    require(pixels.size() == 2 && pixels[1] == 3,
            "Refresh must use newest frame, not queued old frames");
    require(starts[1] - starts[0] >= std::chrono::milliseconds(995),
            "Refresh interval should be one second");
}

void testSpeechDoesNotOverlap() {
    char temporary[] = "/private/tmp/visionassist-speech-XXXXXX";
    const char *created = mkdtemp(temporary);
    require(created != nullptr, "Could not create speech test directory");
    const std::filesystem::path directory(created);
    const auto script = directory / "say";
    const auto log = directory / "speech.log";
    {
        std::ofstream file(script);
        file << "#!/bin/sh\nprintf 'start:%s\\n' \"$5\" >> \"$VOICE_TEST_LOG\"\n"
                "/bin/sleep 0.05\nprintf 'end:%s\\n' \"$5\" >> \"$VOICE_TEST_LOG\"\n";
    }
    std::filesystem::permissions(script, std::filesystem::perms::owner_all);
    const std::string oldPath = getenv("PATH") ? getenv("PATH") : "";
    const std::string testPath = directory.string() + ":" + oldPath;
    setenv("PATH", testPath.c_str(), 1);
    setenv("VOICE_TEST_LOG", log.c_str(), 1);
    std::vector<std::string> lines;
    {
        SpeechQueue speech;
        speech.speak("first ; echo unintended", "Samantha", 210);
        speech.speak("second", "Samantha", 210);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (std::chrono::steady_clock::now() < deadline) {
            lines.clear();
            std::ifstream file(log);
            std::string line;
            while (std::getline(file, line))
                lines.push_back(line);
            if (lines.size() == 4)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    setenv("PATH", oldPath.c_str(), 1);
    unsetenv("VOICE_TEST_LOG");
    std::filesystem::remove_all(directory);
    require(
        lines == std::vector<std::string>{"start:first ; echo unintended",
                                          "end:first ; echo unintended", "start:second",
                                          "end:second"},
        "Speech must finish each message before the next and pass text without shell evaluation");
}

int main() {
    try {
        testOutputFormatsAndCoordinates();
        testNewObjectAnnouncements();
        testOneSecondRefreshUsesNewestFrame();
        testSpeechDoesNotOverlap();
        std::cout << "PASS: output formats, 365 labels, box coordinates, new-object tracking, "
                     "automatic one-second refresh, and speech serialization\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
