#pragma once

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>

// Clean struct to hold information about one detected object
struct Detection {
    std::string label;          // Display / speech label (e.g. "smartphone")
    std::string rawLabel;       // Original model category (e.g. "cell phone")
    float confidence;           // e.g. 0.85 (85%)
    cv::Rect box;               // Bounding box [x, y, width, height]
    std::string clockDirection; // e.g. "12 o'clock", "2 o'clock"
    int centerX = 0;
    int centerY = 0;
};

// Decode raw class scores or end-to-end [x1,y1,x2,y2,score,id] output,
// mapping coordinates back from a letterboxed model input.
std::vector<Detection> decodeYoloDetections(const cv::Mat &output, cv::Size frameSize,
                                            cv::Size inputSize,
                                            const std::vector<std::string> &classNames,
                                            float confidenceThreshold = 0.45f,
                                            float nmsThreshold = 0.45f);

class ObjectDetector {
  public:
    ObjectDetector() = default;

    // Load ONNX network, labels, and optional aliases
    bool loadModel(const std::string &modelPath, const std::string &labelsPath,
                   const std::string &aliasesPath = "");

    void setConfidenceThreshold(float threshold);
    void setInputSize(int width, int height);
    std::size_t classCount() const {
        return classNames_.size();
    }

    // High-performance CPU inference
    std::vector<Detection> detect(const cv::Mat &frame);

  private:
    cv::dnn::Net net_;
    std::vector<std::string> classNames_;
    std::unordered_map<std::string, std::string> aliases_;

    float confThreshold_ = 0.45f;
    float nmsThreshold_ = 0.45f;
    cv::Size inputSize_ = cv::Size(640, 640);
};
