#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct TextLine {
    std::string text;
    float confidence = 0;
    cv::Rect box;
    bool fastRecognition = false;
};
// Public C++ interface. The tiny macOS adapter calls Apple's on-device OCR.
class TextReader {
  public:
    explicit TextReader(bool fast = true);
    std::vector<TextLine> read(const cv::Mat &image) const;
    static std::string spokenText(const std::vector<TextLine> &lines);

  private:
    bool fast_;
};
