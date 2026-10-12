#pragma once
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <stdexcept>

struct ChangeMaskResult {
    cv::Mat mask; // 0 = unchanged; 255 = changed
    cv::Size frameSize;
    cv::Point2f cameraTranslation; // previous frame -> current frame, pixels
    bool translationReliable = false;
    bool cameraMoving = false;
    bool broadChange = false;
    bool lightingChanged = false;
    bool referenceReady = false;
    double changedFraction = 0;

    double fractionIn(cv::Rect region) const;
};

// A low-resolution grayscale comparison, never the input to the color-trained
// object detector. Exposure compensation and translation alignment reduce
// false changes caused by lighting and camera/head movement.
class ChangeMask {
  public:
    explicit ChangeMask(int pixelThreshold = 20);

    ChangeMaskResult update(const cv::Mat &frame);

  private:
    int threshold_;
    cv::Mat previous_;
    cv::Mat window_;
    cv::Size previousFrameSize_;
};
