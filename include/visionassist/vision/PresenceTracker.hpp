#pragma once
#include "visionassist/vision/ChangeMask.hpp"
#include "visionassist/vision/Detector.hpp"

// Keeps object identities and emits only meaningful scene-change events.
class PresenceTracker {
  public:
    enum class Kind { Entered, DirectionChanged, LeftView };
    struct Event {
        Kind kind;
        Detection detection;
        std::string speech() const;
    };
    std::vector<Event> updateEvents(const std::vector<Detection> &detections,
                                    const ChangeMaskResult &change = {});
    std::vector<Detection> update(const std::vector<Detection> &detections);

  private:
    struct Track {
        Detection detection;
        std::string announcedDirection;
        std::string pendingDirection;
        int missedCycles;
        int directionCycles;
        bool motionSeen;
    };
    std::vector<Track> tracks_;
};
