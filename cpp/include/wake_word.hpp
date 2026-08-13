#pragma once

#include <cstdint>
#include <string>

// reference WakeUp.axera wake-word detection on AX650 (fbank + 64-frame window + 3-frame trigger)
class WakeWordDetector {
public:
    WakeWordDetector(const std::string& model_path, float threshold = 0.615f, int det_win = 3);
    ~WakeWordDetector();

    WakeWordDetector(const WakeWordDetector&) = delete;
    WakeWordDetector& operator=(const WakeWordDetector&) = delete;

    // Feed one 32ms frame (512 int16 samples @16k). Returns {wake_score, window_sum, triggered}.
    struct FrameResult {
        float wake_score = 0.f;
        float window_sum = 0.f;
        bool triggered = false;
    };
    FrameResult ProcessFrame(const int16_t* samples);

    void Reset();

private:
    struct Impl;
    Impl* impl_;
};
