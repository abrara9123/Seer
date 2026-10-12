#include "visionassist/vision/SceneSummary.hpp"
#include "visionassist/vision/DepthEstimator.hpp"
#include <algorithm>

std::optional<std::size_t> nearestDetectedObject(const std::vector<Detection> &objects,
                                                 const cv::Mat &depth) {
    std::vector<std::pair<float, std::size_t>> estimates;
    for (std::size_t index = 0; index < objects.size(); ++index) {
        if (const auto value = DepthEstimator::sampleObject(depth, objects[index].box))
            estimates.emplace_back(*value, index);
    }
    if (estimates.size() < 2)
        return {};
    std::sort(estimates.begin(), estimates.end());
    // Close estimates are not reliably distinguishable in a single-camera map.
    if (estimates[1].first < estimates[0].first * 1.2f)
        return {};
    return estimates.front().second;
}
std::string describeRelativeDepth(const std::vector<Detection> &objects, const cv::Mat &depth) {
    const auto nearest = nearestDetectedObject(objects, depth);
    if (!nearest)
        return "I couldn't reliably tell which detected object is closer. Distance is not "
               "measured.";
    const auto &object = objects[*nearest];
    return object.label + " at " + object.clockDirection +
           " appears closest among the detected objects. This is a depth estimate, not a measured "
           "distance.";
}
