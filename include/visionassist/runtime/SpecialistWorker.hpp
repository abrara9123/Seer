#pragma once
#include "visionassist/vision/Detector.hpp"
#include "visionassist/text/TextReader.hpp"
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>

// Depth and OCR each have a worker so text recognition cannot stall depth.
// Keep only the latest pending depth frame and admit one text request at a time.
class SpecialistWorker {
  public:
    struct Snapshot {
        cv::Mat image;
        std::vector<Detection> objects;
        std::uint64_t sceneSequence = 0;
        std::chrono::steady_clock::time_point capturedAt;
    };
    struct DepthResult {
        Snapshot source;
        cv::Mat depth;
        double milliseconds = 0;
        std::string error;
        bool isFreshFor(std::uint64_t sceneSequence, std::chrono::steady_clock::time_point now =
                                                         std::chrono::steady_clock::now()) const;
    };
    struct TextResult {
        cv::Mat sourceFrame;
        std::vector<TextLine> lines;
        std::uint64_t sequence = 0;
        double milliseconds = 0;
        std::string error;
    };
    using DepthInference = std::function<cv::Mat(const cv::Mat &)>;
    using TextInference = std::function<std::vector<TextLine>(const cv::Mat &)>;
    SpecialistWorker(DepthInference depth, TextInference text);
    ~SpecialistWorker();
    void submitDepth(Snapshot snapshot);
    bool requestText(const cv::Mat &image);
    DepthResult depthResult();
    TextResult textResult();
    void stop();

  private:
    void runDepth();
    void runText();
    DepthInference depth_;
    TextInference text_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::optional<Snapshot> pendingDepth_;
    cv::Mat pendingText_;
    bool textBusy_ = false;
    bool stopped_ = false;
    DepthResult depthResult_;
    TextResult textResult_;
    std::thread depthWorker_;
    std::thread textWorker_;
};
