#include "visionassist/vision/Detector.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
std::string normalizedLabel(std::string value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos)
        return "";
    value = value.substr(begin, value.find_last_not_of(" \t\r\n") - begin + 1);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

struct Letterbox {
    float scale;
    int left;
    int top;
    cv::Size resized;
};
Letterbox letterboxGeometry(cv::Size frame, cv::Size input) {
    if (frame.width <= 0 || frame.height <= 0 || input.width <= 0 || input.height <= 0) {
        throw std::invalid_argument("Image dimensions must be positive.");
    }
    const float scale = std::min(static_cast<float>(input.width) / frame.width,
                                 static_cast<float>(input.height) / frame.height);
    const cv::Size resized(std::max(1, static_cast<int>(std::round(frame.width * scale))),
                           std::max(1, static_cast<int>(std::round(frame.height * scale))));
    return {scale, (input.width - resized.width) / 2, (input.height - resized.height) / 2, resized};
}
std::string clockDirection(int centerX, int width) {
    const float ratio = static_cast<float>(centerX) / width;
    if (ratio < 0.20f)
        return "10 o'clock";
    if (ratio < 0.40f)
        return "11 o'clock";
    if (ratio < 0.60f)
        return "12 o'clock";
    if (ratio < 0.80f)
        return "1 o'clock";
    return "2 o'clock";
}
} // namespace

std::vector<Detection> decodeYoloDetections(const cv::Mat &output, cv::Size frameSize,
                                            cv::Size inputSize,
                                            const std::vector<std::string> &classNames,
                                            float confidenceThreshold, float nmsThreshold) {
    if (output.empty())
        return {};
    if (classNames.empty() || output.type() != CV_32F || !output.isContinuous()) {
        throw std::runtime_error("Invalid YOLO output or missing class labels.");
    }
    const auto geometry = letterboxGeometry(frameSize, inputSize);
    const int expectedFeatures = 4 + static_cast<int>(classNames.size());
    cv::Mat predictions;
    bool endToEnd = false;
    if (output.dims == 3 && output.size[0] == 1) {
        if (output.size[1] == expectedFeatures) {
            predictions = cv::Mat(output.size[1], output.size[2], CV_32F,
                                  const_cast<float *>(output.ptr<float>()))
                              .t();
        } else if (output.size[2] == expectedFeatures || output.size[2] == 6) {
            predictions = cv::Mat(output.size[1], output.size[2], CV_32F,
                                  const_cast<float *>(output.ptr<float>()));
            endToEnd = output.size[2] == 6;
        }
    } else if (output.dims == 2 && (output.cols == expectedFeatures || output.cols == 6)) {
        predictions = output;
        endToEnd = output.cols == 6;
    }
    if (predictions.empty()) {
        throw std::runtime_error("YOLO output does not match the configured labels. Use the "
                                 "model's matching labels file.");
    }
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classes;
    for (int row = 0; row < predictions.rows; ++row) {
        const float *values = predictions.ptr<float>(row);
        int classId = 0;
        float score = 0;
        if (endToEnd) {
            if (!std::isfinite(values[5]) || values[5] < 0 || values[5] >= classNames.size() ||
                values[5] != std::floor(values[5]))
                continue;
            classId = static_cast<int>(values[5]);
            score = values[4];
        } else {
            score = values[4];
            for (int c = 1; c < static_cast<int>(classNames.size()); ++c) {
                if (values[4 + c] > score) {
                    score = values[4 + c];
                    classId = c;
                }
            }
        }
        if (!std::isfinite(score) || score < confidenceThreshold || score > 1)
            continue;
        if (!std::isfinite(values[0]) || !std::isfinite(values[1]) || !std::isfinite(values[2]) ||
            !std::isfinite(values[3]))
            continue;
        float x1, y1, x2, y2;
        if (endToEnd) {
            x1 = values[0];
            y1 = values[1];
            x2 = values[2];
            y2 = values[3];
        } else {
            if (values[2] <= 0 || values[3] <= 0)
                continue;
            x1 = values[0] - values[2] / 2;
            y1 = values[1] - values[3] / 2;
            x2 = values[0] + values[2] / 2;
            y2 = values[1] + values[3] / 2;
        }
        x1 = std::clamp((x1 - geometry.left) / geometry.scale, 0.0f,
                        static_cast<float>(frameSize.width));
        y1 = std::clamp((y1 - geometry.top) / geometry.scale, 0.0f,
                        static_cast<float>(frameSize.height));
        x2 = std::clamp((x2 - geometry.left) / geometry.scale, 0.0f,
                        static_cast<float>(frameSize.width));
        y2 = std::clamp((y2 - geometry.top) / geometry.scale, 0.0f,
                        static_cast<float>(frameSize.height));
        if (x2 <= x1 || y2 <= y1)
            continue;
        const int left = static_cast<int>(std::floor(x1));
        const int top = static_cast<int>(std::floor(y1));
        boxes.emplace_back(left, top, static_cast<int>(std::ceil(x2)) - left,
                           static_cast<int>(std::ceil(y2)) - top);
        scores.push_back(score);
        classes.push_back(classId);
    }
    std::vector<int> kept;
    // Different categories can legitimately overlap, e.g. a person and chair.
    std::unordered_map<int, std::vector<int>> byClass;
    for (int i = 0; i < static_cast<int>(boxes.size()); ++i)
        byClass[classes[i]].push_back(i);
    for (const auto &group : byClass) {
        std::vector<cv::Rect> classBoxes;
        std::vector<float> classScores;
        for (int i : group.second) {
            classBoxes.push_back(boxes[i]);
            classScores.push_back(scores[i]);
        }
        std::vector<int> indices;
        cv::dnn::NMSBoxes(classBoxes, classScores, confidenceThreshold, nmsThreshold, indices);
        for (int i : indices)
            kept.push_back(group.second[i]);
    }
    std::sort(kept.begin(), kept.end(), [&](int a, int b) { return scores[a] > scores[b]; });
    std::vector<Detection> results;
    for (int i : kept) {
        Detection detection;
        detection.rawLabel = normalizedLabel(classNames[classes[i]]);
        detection.label = detection.rawLabel;
        detection.confidence = scores[i];
        detection.box = boxes[i];
        detection.centerX = detection.box.x + detection.box.width / 2;
        detection.centerY = detection.box.y + detection.box.height / 2;
        detection.clockDirection = clockDirection(detection.centerX, frameSize.width);
        results.push_back(std::move(detection));
    }
    return results;
}

bool ObjectDetector::loadModel(const std::string &modelPath, const std::string &labelsPath,
                               const std::string &aliasesPath) {
    cv::setUseOptimized(true);
    std::ifstream labels(labelsPath);
    if (!labels) {
        std::cerr << "Failed to open labels file: " << labelsPath << "\n";
        return false;
    }
    classNames_.clear();
    std::string line;
    while (std::getline(labels, line)) {
        line = normalizedLabel(line);
        if (!line.empty())
            classNames_.push_back(line);
    }
    if (classNames_.empty())
        return false;
    aliases_.clear();
    std::ifstream aliasFile(aliasesPath);
    while (std::getline(aliasFile, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream parts(line);
        std::string original, friendly;
        if (std::getline(parts, original, '=') && std::getline(parts, friendly)) {
            aliases_[normalizedLabel(original)] = normalizedLabel(friendly);
        }
    }
    try {
        net_ = cv::dnn::readNetFromONNX(modelPath);
        if (net_.empty())
            return false;
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    } catch (const cv::Exception &error) {
        std::cerr << "Failed to load model: " << error.what() << "\n";
        return false;
    }
    std::cout << "[DETECTOR]: Loaded " << classNames_.size() << " categories.\n";
    return true;
}
void ObjectDetector::setConfidenceThreshold(float threshold) {
    confThreshold_ = threshold;
}
void ObjectDetector::setInputSize(int width, int height) {
    inputSize_ = {width, height};
}
std::vector<Detection> ObjectDetector::detect(const cv::Mat &frame) {
    if (frame.empty())
        return {};
    const auto geometry = letterboxGeometry(frame.size(), inputSize_);
    cv::Mat resized, padded;
    cv::resize(frame, resized, geometry.resized);
    cv::copyMakeBorder(resized, padded, geometry.top,
                       inputSize_.height - resized.rows - geometry.top, geometry.left,
                       inputSize_.width - resized.cols - geometry.left, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));
    net_.setInput(
        cv::dnn::blobFromImage(padded, 1.0 / 255.0, inputSize_, cv::Scalar(), true, false));
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());
    if (outputs.empty())
        throw std::runtime_error("Detector returned no output.");
    auto detections = decodeYoloDetections(outputs[0], frame.size(), inputSize_, classNames_,
                                           confThreshold_, nmsThreshold_);
    for (auto &detection : detections) {
        const auto alias = aliases_.find(detection.rawLabel);
        if (alias != aliases_.end())
            detection.label = alias->second;
    }
    return detections;
}
