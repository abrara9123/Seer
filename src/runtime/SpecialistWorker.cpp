#include "visionassist/runtime/SpecialistWorker.hpp"
#include <stdexcept>

bool SpecialistWorker::DepthResult::isFreshFor(std::uint64_t sequence,
                                               std::chrono::steady_clock::time_point now) const {
    return sequence > 0 && source.sceneSequence == sequence && now >= source.capturedAt &&
           now - source.capturedAt < std::chrono::seconds(3);
}
SpecialistWorker::SpecialistWorker(DepthInference depth, TextInference text)
    : depth_(std::move(depth)), text_(std::move(text)), depthWorker_([this] { runDepth(); }),
      textWorker_([this] { runText(); }) {}
SpecialistWorker::~SpecialistWorker() {
    stop();
}
void SpecialistWorker::submitDepth(Snapshot snapshot) {
    if (snapshot.image.empty())
        return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped_)
        return;
    pendingDepth_ = std::move(snapshot);
    ready_.notify_all();
}
bool SpecialistWorker::requestText(const cv::Mat &image) {
    if (image.empty())
        return false;
    std::lock_guard<std::mutex> lock(mutex_);
    if (textBusy_ || stopped_)
        return false;
    pendingText_ = image.clone();
    textBusy_ = true;
    ready_.notify_all();
    return true;
}
SpecialistWorker::DepthResult SpecialistWorker::depthResult() {
    std::lock_guard<std::mutex> lock(mutex_);
    return depthResult_;
}
SpecialistWorker::TextResult SpecialistWorker::textResult() {
    std::lock_guard<std::mutex> lock(mutex_);
    return textResult_;
}
void SpecialistWorker::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        pendingDepth_.reset();
        pendingText_.release();
    }
    ready_.notify_all();
    if (depthWorker_.joinable())
        depthWorker_.join();
    if (textWorker_.joinable())
        textWorker_.join();
}
void SpecialistWorker::runDepth() {
    using Clock = std::chrono::steady_clock;
    while (true) {
        Snapshot snapshot;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [&] { return stopped_ || pendingDepth_.has_value(); });
            if (stopped_)
                return;
            snapshot = std::move(*pendingDepth_);
            pendingDepth_.reset();
        }
        DepthResult result;
        result.source = std::move(snapshot);
        const auto started = Clock::now();
        try {
            result.depth = depth_(result.source.image);
        } catch (const std::exception &error) {
            result.error = error.what();
        }
        result.milliseconds =
            std::chrono::duration<double, std::milli>(Clock::now() - started).count();
        std::lock_guard<std::mutex> lock(mutex_);
        depthResult_ = std::move(result);
    }
}
void SpecialistWorker::runText() {
    using Clock = std::chrono::steady_clock;
    while (true) {
        cv::Mat image;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [&] { return stopped_ || !pendingText_.empty(); });
            if (stopped_)
                return;
            image = std::move(pendingText_);
            pendingText_.release();
        }
        TextResult result;
        result.sourceFrame = image;
        const auto started = Clock::now();
        try {
            result.lines = text_(image);
        } catch (const std::exception &error) {
            result.error = error.what();
        }
        result.milliseconds =
            std::chrono::duration<double, std::milli>(Clock::now() - started).count();
        std::lock_guard<std::mutex> lock(mutex_);
        result.sequence = textResult_.sequence + 1;
        textResult_ = std::move(result);
        textBusy_ = false;
    }
}
