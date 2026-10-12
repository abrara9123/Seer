#include "visionassist/app/AppConfig.hpp"
#include "visionassist/app/VisionApplication.hpp"
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char **argv) {
    try {
        std::string configPath = "config/default.txt";
        if (argc != 1) {
            if (argc != 3 || std::string(argv[1]) != "--config") {
                std::cerr << "Usage: VisionAssist [--config settings-file]\n";
                return 2;
            }
            configPath = argv[2];
            if (!std::ifstream(configPath))
                throw std::runtime_error("Could not open the requested settings file.");
        }
        return VisionApplication(loadConfig(configPath)).run();
    } catch (const std::exception &error) {
        std::cerr << "VisionAssist could not start: " << error.what() << "\n";
        return 1;
    }
}
