#pragma once
#include "visionassist/vision/ChangeMask.hpp"
#include "visionassist/vision/Detector.hpp"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

// Keeps only the latest frame. Inference starts at most once per interval;
// slow inference never causes a backlog of old camera frames.
class LocalDetectionWorker {
  public:
    struct Result {
        std::vector<Detection> detections;
        std::uint64_t sequence = 0;
        double inferenceMs = 0;
        std::string error;
        ChangeMaskResult change;
        cv::Mat sourceFrame; // Immutable color snapshot used for this result.
        std::chrono::steady_clock::time_point capturedAt;
    };
    using Inference = std::function<std::vector<Detection>(const cv::Mat &)>;
    LocalDetectionWorker(Inference inference, std::chrono::milliseconds interval,
                         int changeThreshold = 20);
    ~LocalDetectionWorker();
    LocalDetectionWorker(const LocalDetectionWorker &) = delete;
    LocalDetectionWorker &operator=(const LocalDetectionWorker &) = delete;
    void submit(const cv::Mat &frame);
    Result result();
    void stop();

  private:
    void run();
    Inference inference_;
    std::chrono::milliseconds interval_;
    ChangeMask changeMask_;
    std::mutex mutex_;
    std::condition_variable ready_;
    cv::Mat latestFrame_;
    std::uint64_t frameSequence_ = 0;
    std::chrono::steady_clock::time_point latestCapturedAt_;
    Result result_;
    bool stopped_ = false;
    std::thread worker_;
};
