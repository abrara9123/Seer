#pragma once
#include "visionassist/vision/Detector.hpp"
#include <optional>
#include <string>

// Fuses object boxes and the depth map from the exact same snapshot.
std::optional<std::size_t> nearestDetectedObject(const std::vector<Detection> &objects,
                                                 const cv::Mat &depth);
std::string describeRelativeDepth(const std::vector<Detection> &objects, const cv::Mat &depth);
