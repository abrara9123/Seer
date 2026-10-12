#include "visionassist/ai/ChatGPTClient.hpp"
#include <cerrno>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
extern char **environ;

ChatGPTClient::ChatGPTClient(SpeechQueue &speech, AppConfig config)
    : speech_(speech), config_(std::move(config)) {}
ChatGPTClient::~ChatGPTClient() {
    stopping_ = true;
    if (const pid_t pid = child_.load(); pid > 0)
        kill(pid, SIGTERM);
    if (worker_.joinable())
        worker_.join();
}
bool ChatGPTClient::analyze(const cv::Mat &image) {
    const auto now = std::chrono::steady_clock::now();
    if (now - lastAnalysis_ < std::chrono::seconds(5) || busy_.exchange(true))
        return false;
    try {
        std::filesystem::create_directories(".local");
        std::filesystem::permissions(".local", std::filesystem::perms::owner_all);
        if (image.empty() || !cv::imwrite(".local/snapshot.jpg", image)) {
            busy_ = false;
            return false;
        }
        std::filesystem::permissions(".local/snapshot.jpg",
                                     std::filesystem::perms::owner_read |
                                         std::filesystem::perms::owner_write);
    } catch (const std::exception &) {
        busy_ = false;
        return false;
    }
    lastAnalysis_ = now;
    if (worker_.joinable())
        worker_.join();
    speech_.speak("Analyzing scene...", config_.voice, config_.speechRate);
    worker_ = std::thread([this] { run(false); });
    return true;
}
bool ChatGPTClient::signIn() {
    if (busy_.exchange(true))
        return false;
    if (worker_.joinable())
        worker_.join();
    speech_.speak("Continue with ChatGPT. Complete sign in in your browser.", config_.voice,
                  config_.speechRate);
    worker_ = std::thread([this] { run(true); });
    return true;
}
void ChatGPTClient::run(bool login) {
    int outputPipe[2];
    if (pipe(outputPipe) != 0) {
        busy_ = false;
        return;
    }
    const std::string python =
        std::filesystem::exists(".venv/bin/python3") ? ".venv/bin/python3" : "python3";
    std::vector<std::string> arguments{python,
                                       login ? "python/chatgpt_auth.py" : "python/smart_vision.py",
                                       login ? "login" : ".local/snapshot.jpg"};
    std::vector<char *> args;
    for (auto &argument : arguments)
        args.push_back(argument.data());
    args.push_back(nullptr);
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, outputPipe[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&actions, outputPipe[0]);
    posix_spawn_file_actions_addclose(&actions, outputPipe[1]);
    pid_t process = -1;
    const int spawned =
        posix_spawnp(&process, python.c_str(), &actions, nullptr, args.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    close(outputPipe[1]);
    if (spawned != 0) {
        close(outputPipe[0]);
        busy_ = false;
        return;
    }
    child_ = process;
    if (stopping_)
        kill(process, SIGTERM);
    std::string output;
    char buffer[512];
    ssize_t count;
    while ((count = read(outputPipe[0], buffer, sizeof(buffer))) > 0) {
        if (output.size() < 16384)
            output.append(buffer, std::min<std::size_t>(count, 16384 - output.size()));
    }
    close(outputPipe[0]);
    int status = 0;
    while (waitpid(process, &status, 0) < 0 && errno == EINTR) {
    }
    child_ = -1;
    if (!stopping_) {
        while (!output.empty() && (output.back() == '\n' || output.back() == '\r'))
            output.pop_back();
        if (!output.empty())
            std::cout << "[CHATGPT]: " << output << "\n";
        if (login)
            speech_.speak(WIFEXITED(status) && WEXITSTATUS(status) == 0
                              ? "ChatGPT connected. Press Spacebar for scene analysis."
                              : "ChatGPT sign in did not finish. Please check the program window.",
                          config_.voice, config_.speechRate);
        else if (!output.empty())
            speech_.speak(output, config_.voice, config_.speechRate);
    }
    busy_ = false;
}
