#pragma once
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>

// Runs the local depth model. Scores describe relative scene depth; they are
// deliberately not exposed as measured meters without camera calibration.
class DepthEstimator {
  public:
    bool loadModel(const std::string &path, int inputSize = 512);
    cv::Mat estimate(const cv::Mat &frame);
    static std::optional<float> sampleObject(const cv::Mat &depth, cv::Rect box);
    static cv::Mat preview(const cv::Mat &depth);

  private:
    cv::dnn::Net network_;
    cv::Size inputSize_{512, 512};
};
