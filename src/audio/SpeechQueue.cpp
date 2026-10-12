#include "visionassist/audio/SpeechQueue.hpp"
#include <cerrno>
#include <csignal>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

SpeechQueue::SpeechQueue() : worker_([this] { run(); }) {}

SpeechQueue::~SpeechQueue() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        pending_.clear();
        if (child_ > 0)
            kill(child_, SIGTERM);
    }
    ready_.notify_all();
    worker_.join();
}

void SpeechQueue::speak(std::string text, std::string voice, int rate) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!stopped_ && !text.empty()) {
        pending_.push_back({std::move(text), std::move(voice), std::to_string(rate)});
        ready_.notify_one();
    }
}

void SpeechQueue::run() {
    while (true) {
        Request request;
        pid_t process = -1;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [&] { return stopped_ || !pending_.empty(); });
            if (stopped_)
                return;
            request = std::move(pending_.front());
            pending_.pop_front();
            char *args[] = {const_cast<char *>("say"),
                            const_cast<char *>("-v"),
                            request.voice.data(),
                            const_cast<char *>("-r"),
                            request.rate.data(),
                            request.text.data(),
                            nullptr};
            if (posix_spawnp(&process, "say", nullptr, nullptr, args, environ) != 0)
                continue;
            child_ = process;
        }
        while (waitpid(process, nullptr, 0) < 0 && errno == EINTR) {
        }
        std::lock_guard<std::mutex> lock(mutex_);
        child_ = -1;
    }
}
