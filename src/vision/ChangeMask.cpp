#include "visionassist/vision/ChangeMask.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

double ChangeMaskResult::fractionIn(cv::Rect region) const {
    if (mask.empty() || frameSize.empty())
        return 0;
    region &= cv::Rect(0, 0, frameSize.width, frameSize.height);
    if (region.empty())
        return 0;
    const double sx = static_cast<double>(mask.cols) / frameSize.width;
    const double sy = static_cast<double>(mask.rows) / frameSize.height;
    const int left = std::max(0, static_cast<int>(std::floor(region.x * sx)));
    const int top = std::max(0, static_cast<int>(std::floor(region.y * sy)));
    const int right =
        std::min(mask.cols, static_cast<int>(std::ceil((region.x + region.width) * sx)));
    const int bottom =
        std::min(mask.rows, static_cast<int>(std::ceil((region.y + region.height) * sy)));
    const cv::Rect roi(left, top, right - left, bottom - top);
    return roi.empty() ? 0 : static_cast<double>(cv::countNonZero(mask(roi))) / roi.area();
}

ChangeMask::ChangeMask(int pixelThreshold) : threshold_(std::clamp(pixelThreshold, 1, 255)) {}

ChangeMaskResult ChangeMask::update(const cv::Mat &frame) {
    ChangeMaskResult result;
    result.frameSize = frame.size();
    if (frame.empty()) {
        previous_.release();
        previousFrameSize_ = {};
        return result;
    }
    cv::Mat gray;
    if (frame.channels() == 3)
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    else if (frame.channels() == 4)
        cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);
    else if (frame.channels() == 1)
        gray = frame.clone();
    else
        throw std::invalid_argument("Unsupported camera image for the change mask.");
    const double scale = std::min(1.0, 320.0 / frame.cols);
    const cv::Size small(std::max(1, static_cast<int>(std::round(frame.cols * scale))),
                         std::max(1, static_cast<int>(std::round(frame.rows * scale))));
    cv::resize(gray, gray, small, 0, 0, cv::INTER_AREA);
    cv::GaussianBlur(gray, gray, {5, 5}, 0);
    cv::Mat current;
    gray.convertTo(current, CV_32F);
    result.mask = cv::Mat::zeros(small, CV_8U);
    if (previous_.size() != current.size() || previousFrameSize_ != frame.size()) {
        previous_ = current;
        previousFrameSize_ = frame.size();
        cv::createHanningWindow(window_, small, CV_32F);
        return result;
    }
    result.referenceReady = true;
    const double brightnessShift = cv::mean(current)[0] - cv::mean(previous_)[0];
    result.lightingChanged = std::abs(brightnessShift) > 12;
    cv::Mat previousCentered = previous_ - cv::mean(previous_)[0];
    cv::Mat currentCentered = current - cv::mean(current)[0];
    double response = 0;
    const cv::Point2d shift =
        cv::phaseCorrelate(previousCentered, currentCentered, window_, &response);
    result.translationReliable = std::isfinite(shift.x) && std::isfinite(shift.y) &&
                                 response >= 0.25 && std::abs(shift.x) <= small.width * 0.20 &&
                                 std::abs(shift.y) <= small.height * 0.20;
    cv::Mat aligned = previous_;
    cv::Mat valid(small, CV_8U, cv::Scalar(255));
    if (result.translationReliable) {
        result.cameraTranslation =
            cv::Point2f(static_cast<float>(shift.x / scale), static_cast<float>(shift.y / scale));
        result.cameraMoving = cv::norm(result.cameraTranslation) > std::max(4.0, frame.cols * 0.01);
        const cv::Mat affine(cv::Matx23d(1, 0, shift.x, 0, 1, shift.y));
        cv::warpAffine(previous_, aligned, affine, small, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        cv::warpAffine(valid, valid, affine, small, cv::INTER_NEAREST, cv::BORDER_CONSTANT);
        cv::erode(valid, valid, cv::getStructuringElement(cv::MORPH_RECT, {5, 5}));
    }
    cv::Mat difference;
    cv::absdiff(current, aligned + brightnessShift, difference);
    cv::threshold(difference, difference, threshold_, 255, cv::THRESH_BINARY);
    difference.convertTo(result.mask, CV_8U);
    cv::bitwise_and(result.mask, valid, result.mask);
    cv::morphologyEx(result.mask, result.mask, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_RECT, {3, 3}));
    // Remove isolated specks instead of treating sensor noise as new events.
    cv::Mat labels, statistics, centers;
    const int components =
        cv::connectedComponentsWithStats(result.mask, labels, statistics, centers);
    std::vector<bool> keep(components, false);
    for (int component = 1; component < components; ++component) {
        keep[component] = statistics.at<int>(component, cv::CC_STAT_AREA) >= 12;
    }
    for (int row = 0; row < result.mask.rows; ++row) {
        auto *pixels = result.mask.ptr<unsigned char>(row);
        const auto *ids = labels.ptr<int>(row);
        for (int column = 0; column < result.mask.cols; ++column) {
            if (!keep[ids[column]])
                pixels[column] = 0;
        }
    }
    result.changedFraction =
        static_cast<double>(cv::countNonZero(result.mask)) / result.mask.total();
    result.broadChange = result.changedFraction > 0.45;
    previous_ = current;
    return result;
}
