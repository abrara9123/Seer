#include "visionassist/vision/ChangeMask.hpp"
#include "visionassist/vision/LocalDetectionWorker.hpp"
#include "visionassist/vision/PresenceTracker.hpp"
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

void check(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
cv::Mat texture() {
    cv::Mat image(240, 320, CV_8UC3);
    cv::RNG random(123456);
    random.fill(image, cv::RNG::UNIFORM, 40, 150);
    return image;
}
Detection object(std::string direction = "12 o'clock") {
    Detection detection;
    detection.rawLabel = detection.label = "cup";
    detection.clockDirection = std::move(direction);
    detection.box = {100, 80, 60, 60};
    detection.confidence = 0.9;
    return detection;
}

void testMaskBasicsAndLocalChanges() {
    ChangeMask mask;
    cv::Mat original = texture();
    auto first = mask.update(original);
    check(!first.referenceReady && cv::countNonZero(first.mask) == 0,
          "First frame should establish a quiet reference");
    check(mask.update(original).changedFraction == 0, "Unchanged scene should have an empty mask");
    cv::Mat changed = original.clone();
    cv::rectangle(changed, {100, 80, 60, 60}, {240, 240, 240}, cv::FILLED);
    auto result = mask.update(changed);
    check(result.fractionIn({100, 80, 60, 60}) > 0.6, "New object region should become white");
    check(result.fractionIn({220, 160, 50, 50}) < 0.05, "Unchanged background should stay black");
    check(!result.broadChange,
          "Small object changes should not be treated as entire-scene changes");
    check(result.mask.type() == CV_8U, "Change mask should be a single-channel binary image");
    check(mask.update(changed).changedFraction == 0,
          "A stationary object should not keep creating changes");
    check(mask.update(cv::Mat(480, 640, CV_8UC3, cv::Scalar(100, 100, 100))).referenceReady ==
              false,
          "Camera resolution changes must reset the comparison");
}

void testLightingAndCameraTranslation() {
    cv::Mat original = texture();
    ChangeMask exposure;
    exposure.update(original);
    cv::Mat brighter;
    original.convertTo(brighter, -1, 1, 20);
    const auto light = exposure.update(brighter);
    check(light.lightingChanged, "Uniform exposure change should be identified");
    check(light.changedFraction < 0.01, "Uniform exposure change should not turn the mask white");

    ChangeMask camera;
    camera.update(original);
    cv::Mat moved;
    cv::Mat transform(cv::Matx23d(1, 0, 8, 0, 1, 5));
    cv::warpAffine(original, moved, transform, original.size(), cv::INTER_LINEAR,
                   cv::BORDER_REPLICATE);
    const auto shift = camera.update(moved);
    check(shift.translationReliable && shift.cameraMoving,
          "A global camera shift should be identified");
    check(cv::norm(shift.cameraTranslation - cv::Point2f(8, 5)) < 1,
          "Camera translation should match image movement");
    check(shift.changedFraction < 0.04,
          "Camera translation should be compensated rather than filling the mask");
}

void testSpeechEventsNeedMeaningfulChanges() {
    PresenceTracker tracker;
    Detection cup = object();
    auto events = tracker.updateEvents({cup});
    check(events.size() == 1 && events[0].kind == PresenceTracker::Kind::Entered,
          "New objects should announce immediately");
    check(tracker.updateEvents({cup}).empty(), "Unchanged scene should remain silent");

    ChangeMaskResult localChange;
    localChange.frameSize = {320, 240};
    localChange.mask = cv::Mat::zeros(240, 320, CV_8U);
    cv::rectangle(localChange.mask, cup.box, cv::Scalar(255), cv::FILLED);
    localChange.referenceReady = true;
    Detection shifted = cup;
    shifted.clockDirection = "1 o'clock";
    check(tracker.updateEvents({shifted}, localChange).empty(),
          "One direction fluctuation should not speak");
    check(tracker.updateEvents({cup}, localChange).empty(),
          "A fluctuation returning to its original direction should stay silent");
    check(tracker.updateEvents({shifted}, localChange).empty(),
          "Direction change should be confirmed over two cycles");
    events = tracker.updateEvents({shifted});
    check(events.size() == 1 && events[0].kind == PresenceTracker::Kind::DirectionChanged,
          "A confirmed direction change with visual evidence should be announced");
    check(events[0].speech() == "cup now at 1 o'clock",
          "Speech must describe camera-relative location, not claim physical motion");
    check(tracker.updateEvents({shifted}, localChange).empty(),
          "An announced direction should not repeat");
    check(tracker.updateEvents({}).empty(),
          "One missed detection should not announce disappearance");
    events = tracker.updateEvents({});
    check(events.size() == 1 && events[0].kind == PresenceTracker::Kind::LeftView,
          "Two missing cycles should announce departure once");
    check(tracker.updateEvents({}).empty(), "Empty view should not repeat departures");
    check(tracker.updateEvents({cup}).size() == 1, "Returning object should be announced");

    PresenceTracker noisy;
    noisy.updateEvents({cup});
    noisy.updateEvents({shifted});
    check(noisy.updateEvents({shifted}).empty(),
          "Box-direction jitter without mask evidence must not speak");
    ChangeMaskResult broad = localChange;
    broad.broadChange = true;
    noisy.updateEvents({shifted}, broad);
    check(noisy.updateEvents({shifted}, broad).empty(),
          "Whole-frame changes alone should not trigger movement announcements");
}

void testCameraAlignmentPreservesObjectIdentity() {
    PresenceTracker tracker;
    auto cup = object();
    tracker.updateEvents({cup});
    ChangeMaskResult camera;
    camera.translationReliable = true;
    camera.cameraMoving = true;
    camera.cameraTranslation = {80, 0};
    cup.box.x += 80;
    check(tracker.updateEvents({cup}, camera).empty(),
          "Global camera translation must not turn the same object into a new object");
}

int main() {
    try {
        testMaskBasicsAndLocalChanges();
        testLightingAndCameraTranslation();
        testSpeechEventsNeedMeaningfulChanges();
        testCameraAlignmentPreservesObjectIdentity();
        ChangeMask performance;
        cv::Mat image = texture();
        performance.update(image);
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 90; ++i)
            performance.update(image);
        const double milliseconds =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count() /
            90;
        std::cout << "PASS: local changes, quiet static scenes, lighting/camera compensation, and "
                     "meaningful speech events\n"
                  << "Average mask processing: " << milliseconds << " ms\n";
        check(milliseconds < 33.3,
              "Change-mask processing should fit the 30-FPS display budget on this Mac");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
