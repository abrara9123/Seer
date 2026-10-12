#pragma once
#include "visionassist/app/AppConfig.hpp"

// Composition root: connects modules and handles camera frames and keyboard
// actions. Model inference implementations live in their own classes/files.
class VisionApplication {
  public:
    explicit VisionApplication(AppConfig config);
    int run();

  private:
    AppConfig config_;
};
