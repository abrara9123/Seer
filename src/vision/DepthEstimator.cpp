#include "visionassist/vision/DepthEstimator.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

bool DepthEstimator::loadModel(const std::string &path, int inputSize) {
    try {
        network_ = cv::dnn::readNetFromONNX(path);
        inputSize_ = {inputSize, inputSize};
        return !network_.empty();
    } catch (const cv::Exception &error) {
        std::cerr << "Depth model could not be loaded: " << error.what() << "\n";
        return false;
    }
}

cv::Mat DepthEstimator::estimate(const cv::Mat &frame) {
    if (frame.empty())
        return {};
    const float scale = std::min(static_cast<float>(inputSize_.width) / frame.cols,
                                 static_cast<float>(inputSize_.height) / frame.rows);
    const cv::Size resizedSize(static_cast<int>(std::round(frame.cols * scale)),
                               static_cast<int>(std::round(frame.rows * scale)));
    const int left = (inputSize_.width - resizedSize.width) / 2;
    const int top = (inputSize_.height - resizedSize.height) / 2;
    cv::Mat resized, padded;
    cv::resize(frame, resized, resizedSize);
    cv::copyMakeBorder(resized, padded, top, inputSize_.height - resized.rows - top, left,
                       inputSize_.width - resized.cols - left, cv::BORDER_CONSTANT,
                       {114, 114, 114});
    network_.setInput(cv::dnn::blobFromImage(padded, 1.0 / 255.0, inputSize_, {}, true, false));
    cv::Mat output = network_.forward();
    if (output.type() != CV_32F || output.dims != 4 || output.size[0] != 1 || output.size[1] != 1) {
        throw std::runtime_error("Unsupported local depth output.");
    }
    cv::Mat raw(output.size[2], output.size[3], CV_32F, output.ptr<float>());
    cv::Mat modelSized, aligned;
    cv::resize(raw, modelSized, inputSize_, 0, 0, cv::INTER_LINEAR);
    cv::resize(modelSized(cv::Rect(left, top, resizedSize.width, resizedSize.height)), aligned,
               frame.size(), 0, 0, cv::INTER_LINEAR);
    return aligned;
}

std::optional<float> DepthEstimator::sampleObject(const cv::Mat &depth, cv::Rect box) {
    if (depth.empty() || depth.type() != CV_32F)
        return {};
    box &= cv::Rect(0, 0, depth.cols, depth.rows);
    if (box.width < 4 || box.height < 4)
        return {};
    // Sample the central patch to reduce background contamination at box edges.
    const cv::Rect center(box.x + box.width / 4, box.y + box.height / 4, std::max(1, box.width / 2),
                          std::max(1, box.height / 2));
    std::vector<float> values;
    for (int y = center.y; y < center.y + center.height; y += 2) {
        for (int x = center.x; x < center.x + center.width; x += 2) {
            const float value = depth.at<float>(y, x);
            if (std::isfinite(value) && value > 0 && value < 1000)
                values.push_back(value);
        }
    }
    const int expected = ((center.width + 1) / 2) * ((center.height + 1) / 2);
    if (values.size() < static_cast<std::size_t>(std::max(4, expected / 2)))
        return {};
    const auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    return *middle;
}

cv::Mat DepthEstimator::preview(const cv::Mat &depth) {
    if (depth.empty())
        return {};
    cv::Mat clipped = depth.clone();
    for (int y = 0; y < clipped.rows; ++y)
        for (int x = 0; x < clipped.cols; ++x) {
            float &value = clipped.at<float>(y, x);
            value = std::isfinite(value) && value > 0 ? std::log1p(value) : 0;
        }
    cv::Mat gray, display;
    cv::normalize(clipped, gray, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::subtract(cv::Scalar::all(255), gray, gray);
    cv::cvtColor(gray, display, cv::COLOR_GRAY2BGR);
    return display;
}
