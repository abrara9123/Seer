#pragma once
#include "visionassist/vision/ChangeMask.hpp"
#include "visionassist/vision/Detector.hpp"
#include "visionassist/text/TextReader.hpp"
#include <chrono>

// Rendering and pacing are separate from all inference workers.
class FramePacer {
  public:
    explicit FramePacer(int targetFps);
    double recordFrame();
    int waitMilliseconds();
    void finishFrame();

  private:
    using Clock = std::chrono::steady_clock;
    Clock::duration period_;
    Clock::time_point next_;
    Clock::time_point fpsStarted_;
    int frames_ = 0;
    double fps_ = 0;
};
class DisplayRenderer {
  public:
    static cv::Mat camera(const cv::Mat &image, const std::vector<Detection> &objects, double fps,
                          int intervalMs, const std::string &banner);
    static cv::Mat changes(const ChangeMaskResult &change);
    static cv::Mat text(const cv::Mat &snapshot, const std::vector<TextLine> &lines);
};
