#include "visionassist/vision/PresenceTracker.hpp"
#include <algorithm>
#include <cmath>

std::string PresenceTracker::Event::speech() const {
    if (kind == Kind::LeftView)
        return detection.label + " no longer in view";
    return detection.label + (kind == Kind::DirectionChanged ? " now at " : " at ") +
           detection.clockDirection;
}

std::vector<PresenceTracker::Event>
PresenceTracker::updateEvents(const std::vector<Detection> &detections,
                              const ChangeMaskResult &change) {
    struct Match {
        float score;
        std::size_t track;
        std::size_t detection;
    };
    std::vector<Match> matches;
    std::vector<Event> events;
    for (auto &track : tracks_) {
        if (change.translationReliable) {
            track.detection.box.x += static_cast<int>(std::round(change.cameraTranslation.x));
            track.detection.box.y += static_cast<int>(std::round(change.cameraTranslation.y));
        }
    }
    for (std::size_t t = 0; t < tracks_.size(); ++t) {
        for (std::size_t d = 0; d < detections.size(); ++d) {
            if (tracks_[t].detection.rawLabel != detections[d].rawLabel)
                continue;
            const auto &a = tracks_[t].detection.box;
            const auto &b = detections[d].box;
            const float intersection = static_cast<float>((a & b).area());
            const float area = static_cast<float>(a.area()) + b.area() - intersection;
            const float overlap = area > 0 ? intersection / area : 0;
            if (overlap >= 0.15f)
                matches.push_back({1 + overlap, t, d});
            else {
                const cv::Point2f centerA(a.x + a.width / 2.0f, a.y + a.height / 2.0f);
                const cv::Point2f centerB(b.x + b.width / 2.0f, b.y + b.height / 2.0f);
                const double diagonal =
                    std::max(std::hypot(a.width, a.height), std::hypot(b.width, b.height));
                const double distance = cv::norm(centerA - centerB);
                const bool similarSize = a.area() > 0 && b.area() > 0 &&
                                         static_cast<double>(std::min(a.area(), b.area())) /
                                                 std::max(a.area(), b.area()) >=
                                             0.5;
                if (similarSize && diagonal > 0 && distance < diagonal * 0.8) {
                    matches.push_back({static_cast<float>(1 - distance / diagonal), t, d});
                }
            }
        }
    }
    std::sort(matches.begin(), matches.end(),
              [](const auto &a, const auto &b) { return a.score > b.score; });
    std::vector<bool> matchedTracks(tracks_.size(), false),
        matchedDetections(detections.size(), false);
    for (const auto &match : matches) {
        if (matchedTracks[match.track] || matchedDetections[match.detection])
            continue;
        auto &track = tracks_[match.track];
        const auto &detection = detections[match.detection];
        const double changed =
            std::max(change.fractionIn(track.detection.box), change.fractionIn(detection.box));
        if (!detection.clockDirection.empty() &&
            detection.clockDirection != track.announcedDirection) {
            if (track.pendingDirection == detection.clockDirection)
                ++track.directionCycles;
            else {
                track.pendingDirection = detection.clockDirection;
                track.directionCycles = 1;
                track.motionSeen = false;
            }
            if (!change.broadChange && changed >= 0.01)
                track.motionSeen = true;
            if (track.directionCycles >= 2 && track.motionSeen) {
                events.push_back({Kind::DirectionChanged, detection});
                track.announcedDirection = detection.clockDirection;
                track.pendingDirection.clear();
                track.directionCycles = 0;
                track.motionSeen = false;
            }
        } else {
            track.pendingDirection.clear();
            track.directionCycles = 0;
            track.motionSeen = false;
        }
        track.detection = detection;
        track.missedCycles = 0;
        matchedTracks[match.track] = matchedDetections[match.detection] = true;
    }
    for (std::size_t t = 0; t < tracks_.size(); ++t) {
        if (!matchedTracks[t] && ++tracks_[t].missedCycles >= 2) {
            events.push_back({Kind::LeftView, tracks_[t].detection});
        }
    }
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [](const Track &track) { return track.missedCycles >= 2; }),
                  tracks_.end());
    for (std::size_t d = 0; d < detections.size(); ++d) {
        if (!matchedDetections[d]) {
            tracks_.push_back({detections[d], detections[d].clockDirection, "", 0, 0, false});
            events.push_back({Kind::Entered, detections[d]});
        }
    }
    return events;
}

std::vector<Detection> PresenceTracker::update(const std::vector<Detection> &detections) {
    std::vector<Detection> entered;
    for (const auto &event : updateEvents(detections)) {
        if (event.kind == Kind::Entered)
            entered.push_back(event.detection);
    }
    return entered;
}
