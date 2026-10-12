#include "visionassist/ui/DisplayRenderer.hpp"
#include <algorithm>
#include <cmath>
#include <thread>

FramePacer::FramePacer(int targetFps) {
    period_ = std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(1.0 / std::max(1, targetFps)));
    next_ = fpsStarted_ = Clock::now();
}
double FramePacer::recordFrame() {
    ++frames_;
    const double elapsed = std::chrono::duration<double>(Clock::now() - fpsStarted_).count();
    if (elapsed >= 0.5) {
        fps_ = frames_ / elapsed;
        frames_ = 0;
        fpsStarted_ = Clock::now();
    }
    return fps_;
}
int FramePacer::waitMilliseconds() {
    next_ += period_;
    return std::max(
        1,
        static_cast<int>(std::chrono::duration<double, std::milli>(next_ - Clock::now()).count()));
}
void FramePacer::finishFrame() {
    if (next_ + period_ < Clock::now())
        next_ = Clock::now();
    std::this_thread::sleep_until(next_);
}
cv::Mat DisplayRenderer::camera(const cv::Mat &image, const std::vector<Detection> &objects,
                                double fps, int intervalMs, const std::string &banner) {
    cv::Mat display = image.clone();
    for (const auto &object : objects) {
        cv::rectangle(display, object.box, {0, 255, 0}, 2);
        const std::string label = object.label + " (" + object.clockDirection + ") " +
                                  std::to_string(static_cast<int>(object.confidence * 100)) + "%";
        cv::putText(display, label, {object.box.x, std::max(20, object.box.y - 8)},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, {0, 255, 0}, 2);
    }
    const std::string status =
        "FPS: " + (fps > 0 ? std::to_string(static_cast<int>(std::round(fps))) : "--") +
        " | Local: " + std::to_string(intervalMs) + " ms | SPACE: ChatGPT";
    cv::putText(display, status, {15, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {0, 255, 255}, 2);
    if (!banner.empty())
        cv::putText(display, banner, {15, 65}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 165, 255}, 2);
    return display;
}
cv::Mat DisplayRenderer::changes(const ChangeMaskResult &change) {
    if (change.mask.empty())
        return {};
    cv::Mat preview;
    cv::cvtColor(change.mask, preview, cv::COLOR_GRAY2BGR);
    const std::string label =
        change.lightingChanged
            ? "Lighting changed"
            : (change.broadChange
                   ? "Broad scene change"
                   : (change.cameraMoving ? "Camera moved" : "White: changed pixels"));
    cv::putText(preview, label, {8, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.45, {0, 200, 255}, 1);
    return preview;
}
cv::Mat DisplayRenderer::text(const cv::Mat &snapshot, const std::vector<TextLine> &lines) {
    cv::Mat display = snapshot.clone();
    for (const auto &line : lines) {
        cv::rectangle(display, line.box, {0, 255, 255}, 2);
        cv::putText(display, line.text.substr(0, 60), {line.box.x, std::max(20, line.box.y - 4)},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 255}, 1);
    }
    cv::putText(display, "Text reading snapshot (offline)", {10, 25}, cv::FONT_HERSHEY_SIMPLEX,
                0.55, {0, 200, 255}, 2);
    return display;
}
