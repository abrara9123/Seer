#include "visionassist/vision/LocalDetectionWorker.hpp"
#include <algorithm>

LocalDetectionWorker::LocalDetectionWorker(Inference inference, std::chrono::milliseconds interval,
                                           int changeThreshold)
    : inference_(std::move(inference)), interval_(std::max(interval, std::chrono::milliseconds(1))),
      changeMask_(changeThreshold), worker_([this] { run(); }) {}

LocalDetectionWorker::~LocalDetectionWorker() {
    stop();
}

void LocalDetectionWorker::submit(const cv::Mat &frame) {
    if (frame.empty())
        return;
    std::lock_guard<std::mutex> lock(mutex_);
    latestFrame_ = frame.clone();
    latestCapturedAt_ = std::chrono::steady_clock::now();
    ++frameSequence_;
    ready_.notify_one();
}

LocalDetectionWorker::Result LocalDetectionWorker::result() {
    std::lock_guard<std::mutex> lock(mutex_);
    return result_;
}

void LocalDetectionWorker::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
    }
    ready_.notify_all();
    if (worker_.joinable())
        worker_.join();
}

void LocalDetectionWorker::run() {
    using Clock = std::chrono::steady_clock;
    auto nextDue = Clock::now();
    std::uint64_t processedFrame = 0;
    while (true) {
        cv::Mat frame;
        std::chrono::steady_clock::time_point capturedAt;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [&] { return stopped_ || frameSequence_ > processedFrame; });
            if (stopped_)
                return;
            ready_.wait_until(lock, nextDue, [&] { return stopped_; });
            if (stopped_)
                return;
            frame = latestFrame_; // submit replaces, rather than mutates, this buffer
            capturedAt = latestCapturedAt_;
            processedFrame = frameSequence_;
        }
        const auto started = Clock::now();
        Result completed;
        completed.sourceFrame = frame;
        completed.capturedAt = capturedAt;
        try {
            completed.change = changeMask_.update(frame);
            completed.detections = inference_(frame);
        } catch (const std::exception &error) {
            completed.error = error.what();
        }
        const auto finished = Clock::now();
        completed.inferenceMs =
            std::chrono::duration<double, std::milli>(finished - started).count();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            completed.sequence = result_.sequence + 1;
            result_ = std::move(completed);
        }
        nextDue = std::max(started + interval_, finished);
    }
}
