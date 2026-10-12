#include "visionassist/vision/DepthEstimator.hpp"
#include <chrono>
#include <iostream>
int main(int argc, char **argv) {
    if (argc != 3)
        return 2;
    DepthEstimator estimator;
    if (!estimator.loadModel(argv[1]))
        return 3;
    const cv::Mat image = cv::imread(argv[2]);
    if (image.empty())
        return 4;
    try {
        const auto started = std::chrono::steady_clock::now();
        cv::Mat depth = estimator.estimate(image);
        if (depth.size() != image.size() || depth.type() != CV_32F ||
            !cv::checkRange(depth, true, nullptr, 0, 1000))
            return 5;
        std::cout << "Depth specialist: " << depth.cols << "x" << depth.rows << " aligned map, "
                  << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                               started)
                         .count()
                  << " ms. No calibrated distances exposed.\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << "\n";
        return 6;
    }
    return 0;
}
