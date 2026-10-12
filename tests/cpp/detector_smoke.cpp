#include "visionassist/vision/Detector.hpp"
#include <chrono>
#include <iostream>
int main(int argc, char **argv) {
    if (argc != 4)
        return 2;
    ObjectDetector detector;
    if (!detector.loadModel(argv[1], argv[2], "assets/item_aliases.txt"))
        return 3;
    cv::Mat image = cv::imread(argv[3]);
    if (image.empty())
        return 4;
    std::vector<Detection> detections;
    double totalMs = 0, maximumMs = 0;
    try {
        for (int i = 0; i < 6; ++i) {
            const auto started = std::chrono::steady_clock::now();
            detections = detector.detect(image);
            const double milliseconds = std::chrono::duration<double, std::milli>(
                                            std::chrono::steady_clock::now() - started)
                                            .count();
            if (i > 0) {
                totalMs += milliseconds;
                maximumMs = std::max(maximumMs, milliseconds);
            }
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 5;
    }
    std::cout << "Categories: " << detector.classCount() << " | Average inference: " << totalMs / 5
              << " ms | Slowest warm inference: " << maximumMs << " ms\n";
    for (const auto &detection : detections) {
        std::cout << detection.rawLabel << " | " << detection.confidence << " | "
                  << detection.clockDirection << " | " << detection.box.x << "," << detection.box.y
                  << "," << detection.box.width << "," << detection.box.height << "\n";
    }
    return 0;
}
