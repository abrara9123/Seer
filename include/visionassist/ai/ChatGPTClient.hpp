#pragma once
#include "visionassist/app/AppConfig.hpp"
#include "visionassist/audio/SpeechQueue.hpp"
#include <atomic>
#include <chrono>
#include <opencv2/core.hpp>
#include <string>
#include <sys/types.h>
#include <thread>

// Owns the lifecycle of the existing Python ChatGPT helper. Only explicit
// Spacebar/sign-in actions reach this module; local workers never call it.
class ChatGPTClient {
  public:
    ChatGPTClient(SpeechQueue &speech, AppConfig config);
    ~ChatGPTClient();
    bool analyze(const cv::Mat &image);
    bool signIn();

  private:
    void run(bool login);
    SpeechQueue &speech_;
    AppConfig config_;
    std::atomic<bool> busy_{false};
    std::atomic<bool> stopping_{false};
    std::atomic<pid_t> child_{-1};
    std::chrono::steady_clock::time_point lastAnalysis_{};
    std::thread worker_;
};
