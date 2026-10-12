#include "visionassist/app/VisionApplication.hpp"
#include "visionassist/ai/ChatGPTClient.hpp"
#include "visionassist/vision/DepthEstimator.hpp"
#include "visionassist/vision/Detector.hpp"
#include "visionassist/ui/DisplayRenderer.hpp"
#include "visionassist/vision/LocalDetectionWorker.hpp"
#include "visionassist/vision/PresenceTracker.hpp"
#include "visionassist/vision/SceneSummary.hpp"
#include "visionassist/runtime/SpecialistWorker.hpp"
#include "visionassist/audio/SpeechQueue.hpp"
#include "visionassist/text/TextReader.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

VisionApplication::VisionApplication(AppConfig config) : config_(std::move(config)) {}
int VisionApplication::run() {
    std::cout
        << "VisionAssist: local object detection, local depth, offline text reading, and ChatGPT\n"
        << "Objects: automatic every " << config_.detectionIntervalMs
        << " ms | Video target: " << config_.targetFps << " FPS\n"
        << "SPACE: ChatGPT | R: read text offline | D: describe relative depth | V: depth preview\n"
        << "M: change mask | L: ChatGPT sign in | ESC: quit\n";
    ObjectDetector detector;
    detector.setConfidenceThreshold(config_.confidence);
    detector.setInputSize(config_.inputSize, config_.inputSize);
    if (!detector.loadModel(config_.modelPath, config_.labelsPath, "assets/item_aliases.txt"))
        return 1;
    DepthEstimator depth;
    const bool depthReady =
        config_.depthEnabled && depth.loadModel(config_.depthModelPath, config_.depthInputSize);
    TextReader textReader(config_.ocrFast);
    SpeechQueue speech;
    ChatGPTClient chatgpt(speech, config_);
    LocalDetectionWorker objects([&](const cv::Mat &frame) { return detector.detect(frame); },
                                 std::chrono::milliseconds(config_.detectionIntervalMs),
                                 config_.changeThreshold);
    SpecialistWorker specialists(
        [&](const cv::Mat &frame) {
            if (!depthReady)
                throw std::runtime_error("Local depth is not enabled or its model is unavailable.");
            return depth.estimate(frame);
        },
        [&](const cv::Mat &frame) { return textReader.read(frame); });
    PresenceTracker presence;
    ChangeMask changeMask(config_.changeThreshold);
    FramePacer pacing(config_.targetFps);
    cv::VideoCapture camera(0);
    if (!camera.isOpened()) {
        std::cerr << "Could not open the camera.\n";
        return 1;
    }
    camera.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    camera.set(cv::CAP_PROP_FPS, config_.targetFps);
    speech.speak("Vision Assist active. Objects are checked automatically. Press R to read text, D "
                 "for a depth estimate, or Spacebar for ChatGPT.",
                 config_.voice, config_.speechRate);
    bool showMask = config_.showChangeMask, showDepth = false, depthRequested = false;
    bool depthWindowCreated = false;
    auto lastReadRequest = std::chrono::steady_clock::time_point{};
    auto lastDepthRequest = std::chrono::steady_clock::time_point{};
    std::uint64_t consumedObjects = 0, consumedText = 0;
    std::string banner;
    auto bannerAt = std::chrono::steady_clock::now();
    const std::string maskWindow = "VisionAssist Changes - white means changed";
    const std::string depthWindow = "Estimated Relative Depth - snapshot, not meters";
    const std::string textWindow = "Offline Text Reading - snapshot";
    auto say = [&](const std::string &message) {
        speech.speak(message, config_.voice, config_.speechRate);
    };
    auto notify = [&](const std::string &message) {
        banner = message;
        bannerAt = std::chrono::steady_clock::now();
    };

    while (true) {
        cv::Mat frame;
        camera >> frame;
        if (frame.empty())
            break;
        const auto change = changeMask.update(frame);
        objects.submit(frame);
        const auto scene = objects.result();
        if (scene.sequence != consumedObjects) {
            consumedObjects = scene.sequence;
            if (!scene.error.empty()) {
                notify("Local detection error");
                std::cerr << scene.error << "\n";
            } else {
                std::vector<Detection> audible;
                for (const auto &object : scene.detections) {
                    if (!config_.ignoredClasses.count(object.rawLabel) &&
                        !config_.ignoredClasses.count(object.label))
                        audible.push_back(object);
                }
                std::string changes;
                for (const auto &event : presence.updateEvents(audible, scene.change)) {
                    if (!changes.empty())
                        changes += ". ";
                    changes += event.speech();
                }
                if (!changes.empty()) {
                    std::cout << "[OBJECT CHANGE]: " << changes << "\n";
                    say(changes);
                }
                // Share the same immutable snapshot and scene ID: depth scores
                // are never fused with boxes from a different camera frame.
                if (depthReady)
                    specialists.submitDepth(
                        {scene.sourceFrame, audible, scene.sequence, scene.capturedAt});
            }
        }
        const auto reading = specialists.textResult();
        if (reading.sequence != consumedText) {
            consumedText = reading.sequence;
            say(reading.error.empty() ? TextReader::spokenText(reading.lines)
                                      : "Local text reading failed. Please try again.");
            if (reading.error.empty())
                cv::imshow(textWindow, DisplayRenderer::text(reading.sourceFrame, reading.lines));
        }
        const auto depthScene = specialists.depthResult();
        const bool freshDepth = depthScene.isFreshFor(scene.sequence);
        if (depthRequested && freshDepth) {
            say(depthScene.error.empty()
                    ? describeRelativeDepth(depthScene.source.objects, depthScene.depth)
                    : depthScene.error);
            depthRequested = false;
        }
        if (std::chrono::steady_clock::now() - bannerAt > std::chrono::seconds(3))
            banner.clear();
        cv::Mat display = DisplayRenderer::camera(frame, scene.detections, pacing.recordFrame(),
                                                  config_.detectionIntervalMs, banner);
        if (freshDepth && depthScene.error.empty()) {
            const auto nearest = nearestDetectedObject(depthScene.source.objects, depthScene.depth);
            if (nearest)
                cv::putText(
                    display, "Closest estimate: " + depthScene.source.objects[*nearest].label,
                    {15, display.rows - 15}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 200, 255}, 1);
        }
        cv::imshow(config_.windowTitle, display);
        if (showMask)
            cv::imshow(maskWindow, DisplayRenderer::changes(change));
        if (showDepth && freshDepth && !depthScene.depth.empty()) {
            cv::Mat preview = DepthEstimator::preview(depthScene.depth);
            cv::putText(preview, "Relative estimate from a snapshot", {10, 25},
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 200, 255}, 1);
            cv::imshow(depthWindow, preview);
            depthWindowCreated = true;
        }
        const int key = cv::waitKey(pacing.waitMilliseconds());
        if (key == 27)
            break;
        if (key == 32)
            notify(chatgpt.analyze(frame) ? "ChatGPT analyzing scene..."
                                          : "Please wait before another ChatGPT request.");
        else if (key == 'l' || key == 'L') {
            chatgpt.signIn();
            notify("Continue with ChatGPT in your browser.");
        } else if (key == 'r' || key == 'R') {
            const auto now = std::chrono::steady_clock::now();
            if (now - lastReadRequest >= std::chrono::seconds(2)) {
                if (specialists.requestText(frame)) {
                    lastReadRequest = now;
                    notify("Reading text locally...");
                } else
                    notify("Text reading is already running.");
            }
        } else if (key == 'd' || key == 'D') {
            const auto now = std::chrono::steady_clock::now();
            if (now - lastDepthRequest < std::chrono::seconds(2)) {
                pacing.finishFrame();
                continue;
            }
            lastDepthRequest = now;
            if (!depthReady)
                say("The local depth model is unavailable.");
            else if (freshDepth)
                say(depthScene.error.empty()
                        ? describeRelativeDepth(depthScene.source.objects, depthScene.depth)
                        : depthScene.error);
            else {
                depthRequested = true;
                notify("Waiting for a fresh depth estimate...");
            }
        } else if (key == 'v' || key == 'V') {
            if (!depthReady)
                say("The local depth model is unavailable.");
            else {
                showDepth = !showDepth;
                if (!showDepth && depthWindowCreated) {
                    cv::destroyWindow(depthWindow);
                    depthWindowCreated = false;
                }
            }
        } else if (key == 'm' || key == 'M') {
            showMask = !showMask;
            if (!showMask)
                cv::destroyWindow(maskWindow);
        }
        pacing.finishFrame();
    }
    objects.stop();
    specialists.stop();
    camera.release();
    cv::destroyAllWindows();
    return 0;
}
