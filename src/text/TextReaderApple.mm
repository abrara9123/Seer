// Objective-C++ is required only at the Apple framework boundary. The rest of
// the app, including TextReader's public API, remains ordinary C++.
#include "visionassist/text/TextReader.hpp"
#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
#include <algorithm>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <tuple>

std::vector<TextLine> recognizeTextWithAppleVision(const cv::Mat &image, bool fast) {
    @autoreleasepool {
        std::vector<unsigned char> encoded;
        if (!cv::imencode(".png", image, encoded))
            throw std::runtime_error("Could not prepare the text-reading image.");
        NSData *data = [NSData dataWithBytes:encoded.data() length:encoded.size()];
        VNRecognizeTextRequest *request = [[VNRecognizeTextRequest alloc] init];
        request.recognitionLevel =
            fast ? VNRequestTextRecognitionLevelFast : VNRequestTextRecognitionLevelAccurate;
        request.recognitionLanguages = @[ @"en-US" ];
        request.usesLanguageCorrection = fast ? NO : YES;
        request.minimumTextHeight = 0.015;
        VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithData:data
                                                                             options:@{}];
        NSError *error = nil;
        if (![handler performRequests:@[ request ] error:&error]) {
            const char *reason = error.localizedDescription.UTF8String;
            throw std::runtime_error(std::string("Local text recognition failed: ") +
                                     (reason ? reason : "unknown Vision error"));
        }
        std::vector<TextLine> lines;
        for (VNRecognizedTextObservation *observation in request.results) {
            VNRecognizedText *candidate = [[observation topCandidates:1] firstObject];
            // Apple's fast path uses a different confidence scale; correct
            // fast-path text commonly carries 0.5. Preserve that raw score.
            if (!candidate || candidate.confidence < (fast ? 0.5 : 0.6) ||
                candidate.string.length == 0)
                continue;
            const CGRect bounds = observation.boundingBox;
            cv::Rect box(static_cast<int>(bounds.origin.x * image.cols),
                         static_cast<int>((1 - bounds.origin.y - bounds.size.height) * image.rows),
                         static_cast<int>(bounds.size.width * image.cols),
                         static_cast<int>(bounds.size.height * image.rows));
            box &= cv::Rect(0, 0, image.cols, image.rows);
            lines.push_back(
                {std::string(candidate.string.UTF8String), candidate.confidence, box, fast});
        }
        std::sort(lines.begin(), lines.end(), [](const TextLine &a, const TextLine &b) {
            return std::tie(a.box.y, a.box.x) < std::tie(b.box.y, b.box.x);
        });
        return lines;
    }
}
