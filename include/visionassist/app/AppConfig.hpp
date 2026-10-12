#pragma once
#include <string>
#include <unordered_set>

// Settings shared by modules. The Python helper owns ChatGPT credentials;
// this C++ structure contains only camera, local-model, and audio settings.
struct AppConfig {
    std::string windowTitle = "VisionAssist Hybrid AI";
    int targetFps = 30;
    std::string modelPath = "models/objects365_detector.onnx";
    std::string labelsPath = "models/objects365.names";
    int detectionIntervalMs = 1000;
    int changeThreshold = 20;
    bool showChangeMask = true;
    float confidence = 0.45f;
    std::unordered_set<std::string> ignoredClasses;
    std::string voice = "Samantha";
    int speechRate = 210;
    int inputSize = 640;
    std::string depthModelPath = "models/depth_estimator.onnx";
    int depthInputSize = 512;
    bool depthEnabled = true;
    bool ocrFast = true;
};
AppConfig loadConfig(const std::string &path);
