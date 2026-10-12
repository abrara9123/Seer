#include "visionassist/app/AppConfig.hpp"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {
std::string trim(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return "";
    return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}
bool enabled(const std::string &text) {
    return text == "true" || text == "1";
}
} // namespace

AppConfig loadConfig(const std::string &path) {
    AppConfig config;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line.front() == '#')
            continue;
        const auto separator = line.find('=');
        if (separator == std::string::npos)
            continue;
        const auto key = trim(line.substr(0, separator));
        const auto value = trim(line.substr(separator + 1));
        try {
            if (key == "window_title")
                config.windowTitle = value;
            else if (key == "target_fps")
                config.targetFps = std::clamp(std::stoi(value), 1, 60);
            else if (key == "model")
                config.modelPath = value;
            else if (key == "labels")
                config.labelsPath = value;
            else if (key == "detection_interval_ms")
                config.detectionIntervalMs = std::max(100, std::stoi(value));
            else if (key == "change_threshold")
                config.changeThreshold = std::clamp(std::stoi(value), 1, 255);
            else if (key == "show_change_mask")
                config.showChangeMask = enabled(value);
            else if (key == "confidence")
                config.confidence = std::clamp(std::stof(value), 0.01f, 1.0f);
            else if (key == "voice")
                config.voice = value;
            else if (key == "rate")
                config.speechRate = std::clamp(std::stoi(value), 80, 400);
            else if (key == "input_size")
                config.inputSize = std::max(32, std::stoi(value));
            else if (key == "depth_model")
                config.depthModelPath = value;
            else if (key == "depth_input_size")
                config.depthInputSize = std::max(32, std::stoi(value));
            else if (key == "depth_enabled")
                config.depthEnabled = enabled(value);
            else if (key == "ocr_fast")
                config.ocrFast = enabled(value);
            else if (key == "ignore") {
                std::istringstream values(value);
                std::string label;
                while (std::getline(values, label, ',')) {
                    label = trim(label);
                    std::transform(label.begin(), label.end(), label.begin(), [](unsigned char c) {
                        return static_cast<char>(std::tolower(c));
                    });
                    if (!label.empty())
                        config.ignoredClasses.insert(label);
                }
            }
        } catch (const std::exception &) {
            throw std::runtime_error("Invalid configuration setting: " + key);
        }
    }
    return config;
}
