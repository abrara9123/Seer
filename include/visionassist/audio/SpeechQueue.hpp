#pragma once
#include <cerrno>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <mutex>
#include <spawn.h>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
extern char **environ;

// A single speech process prevents automatic callouts and requested AI
// descriptions from speaking over one another. Text is passed as an argument,
// never evaluated by a shell.
class SpeechQueue {
  public:
    SpeechQueue();
    ~SpeechQueue();
    void speak(std::string text, std::string voice, int rate);

  private:
    struct Request {
        std::string text;
        std::string voice;
        std::string rate;
    };
    void run();
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<Request> pending_;
    bool stopped_ = false;
    pid_t child_ = -1;
    std::thread worker_;
};
