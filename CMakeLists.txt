cmake_minimum_required(VERSION 3.20)
project(VisionAssist LANGUAGES CXX OBJCXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_OBJCXX_STANDARD 17)
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew" "/usr/local")
find_package(OpenCV REQUIRED)
find_library(VISION_FRAMEWORK Vision REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

# Core library: one responsibility per header/source pair. Tests use the same
# implementations as the camera app; no duplicate implementation is compiled.
add_library(visionassist_core
    src/app/AppConfig.cpp
    src/vision/Detector.cpp src/vision/ChangeMask.cpp src/vision/PresenceTracker.cpp
    src/vision/LocalDetectionWorker.cpp src/vision/DepthEstimator.cpp src/vision/SceneSummary.cpp
    src/audio/SpeechQueue.cpp src/text/TextReader.cpp src/text/TextReaderApple.mm
    src/runtime/SpecialistWorker.cpp src/ui/DisplayRenderer.cpp src/ai/ChatGPTClient.cpp)
target_compile_options(visionassist_core PRIVATE -O3)
set_source_files_properties(src/text/TextReaderApple.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
target_include_directories(visionassist_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(visionassist_core PUBLIC ${OpenCV_LIBS} ${VISION_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
add_executable(VisionAssist src/app/main.cpp src/app/VisionApplication.cpp)
target_link_libraries(VisionAssist PRIVATE visionassist_core)
target_compile_options(VisionAssist PRIVATE -O3)

include(CTest)
if(BUILD_TESTING)
    foreach(test_target local_detection_test change_mask_test detector_smoke preview_pipeline_test specialist_test depth_smoke chatgpt_client_test)
        add_executable(${test_target} tests/cpp/${test_target}.cpp)
        target_link_libraries(${test_target} PRIVATE visionassist_core)
        target_compile_options(${test_target} PRIVATE -O3)
    endforeach()
    add_test(NAME local_detection COMMAND local_detection_test)
    add_test(NAME change_mask COMMAND change_mask_test)
    add_test(NAME specialists COMMAND specialist_test)
    add_test(NAME chatgpt_client COMMAND chatgpt_client_test)
endif()
