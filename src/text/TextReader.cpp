#include "visionassist/text/TextReader.hpp"
#include <algorithm>
#include <sstream>

std::vector<TextLine> recognizeTextWithAppleVision(const cv::Mat &image, bool fast);
TextReader::TextReader(bool fast) : fast_(fast) {}
std::vector<TextLine> TextReader::read(const cv::Mat &image) const {
    if (image.empty())
        return {};
    return recognizeTextWithAppleVision(image, fast_);
}
std::string TextReader::spokenText(const std::vector<TextLine> &lines) {
    std::string text;
    for (const auto &line : lines) {
        if (line.confidence < (line.fastRecognition ? 0.5f : 0.6f) || line.text.empty())
            continue;
        if (!text.empty())
            text += ". ";
        text += line.text;
        if (text.size() >= 500)
            break;
    }
    if (text.empty())
        return "I couldn't read clear text. Hold the camera steady and move the text closer.";
    std::istringstream words(text);
    std::string word, result;
    int count = 0;
    while (count < 80 && words >> word) {
        if (!result.empty())
            result += " ";
        result += word;
        ++count;
    }
    return "Text reads: " + result;
}
