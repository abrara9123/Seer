#include "visionassist/ai/ChatGPTClient.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <unistd.h>

int main() {
    namespace fs = std::filesystem;
    const auto original = fs::current_path();
    char pattern[] = "/private/tmp/visionassist-helper-XXXXXX";
    const char *created = mkdtemp(pattern);
    if (!created)
        return 1;
    const fs::path folder(created);
    const std::string originalPath = getenv("PATH") ? getenv("PATH") : "";
    try {
        fs::create_directories(folder / ".venv/bin");
        {
            std::ofstream helper(folder / ".venv/bin/python3");
            helper << "#!/usr/bin/env python3\nimport sys,time\nfrom pathlib import Path\n"
                      "Path('helper-started').write_text(sys.argv[1])\n"
                      "if Path(sys.argv[1]).name == 'chatgpt_auth.py': time.sleep(10)\n"
                      "else: print('A bottle at twelve oclock.', flush=True)\n";
            std::ofstream say(folder / "say");
            say << "#!/bin/sh\nexit 0\n";
        }
        fs::permissions(folder / ".venv/bin/python3", fs::perms::owner_all);
        fs::permissions(folder / "say", fs::perms::owner_all);
        setenv("PATH", (folder.string() + ":" + originalPath).c_str(), 1);
        fs::current_path(folder);
        const auto started = std::chrono::steady_clock::now();
        {
            SpeechQueue speech;
            ChatGPTClient client(speech, AppConfig{});
            cv::Mat image(10, 10, CV_8UC3, cv::Scalar(0, 0, 0));
            if (!client.analyze(image) || client.analyze(image))
                throw std::runtime_error("Spacebar admission/cooldown failed");
            const auto snapshot = folder / ".local/snapshot.jpg";
            if (!fs::exists(snapshot) || fs::status(snapshot).permissions() !=
                                             (fs::perms::owner_read | fs::perms::owner_write))
                throw std::runtime_error("Camera snapshot must be stored privately");
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
            if (!client.signIn())
                throw std::runtime_error("Helper did not return to idle");
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (std::chrono::steady_clock::now() < deadline) {
                std::ifstream marker("helper-started");
                std::string command;
                marker >> command;
                if (command == "python/chatgpt_auth.py")
                    break;
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
        if (std::chrono::steady_clock::now() - started > std::chrono::seconds(3))
            throw std::runtime_error("Closing the app must cancel its owned helper");
        fs::current_path(original);
        setenv("PATH", originalPath.c_str(), 1);
        fs::remove_all(folder);
        std::cout << "PASS: Spacebar-only helper admission, cooldown, and clean cancellation. No "
                     "real ChatGPT requests made.\n";
        return 0;
    } catch (const std::exception &error) {
        fs::current_path(original);
        setenv("PATH", originalPath.c_str(), 1);
        fs::remove_all(folder);
        std::cerr << error.what() << "\n";
        return 1;
    }
}
