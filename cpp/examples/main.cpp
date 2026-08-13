// WakeUp.axera wake-word detection example: reads raw int16 16k mono PCM,
// feeds 512-sample frames, prints per-frame score and trigger events.
#include "wake_word.hpp"

#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.axmodel> <pcm_s16le_16k.raw> [threshold]\n", argv[0]);
        return 1;
    }
    const float threshold = argc >= 4 ? static_cast<float>(std::atof(argv[3])) : 0.615f;
    std::ifstream f(argv[2], std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "failed to open %s\n", argv[2]);
        return 1;
    }
    std::vector<int16_t> pcm((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
    if (pcm.size() % 2) pcm.pop_back();
    std::vector<int16_t> samples;
    samples.reserve(pcm.size() / 2);
    for (size_t i = 0; i < pcm.size(); i += 2) {
        samples.push_back(static_cast<int16_t>(pcm[i] | (pcm[i + 1] << 8)));
    }
    const int n_frames = (static_cast<int>(samples.size()) - 512) / 512 + 1;
    if (n_frames < 1) {
        std::fprintf(stderr, "PCM too short\n");
        return 1;
    }

    try {
        WakeWordDetector detector(argv[1], threshold, 3);
        int triggers = 0;
        float max_score = -1e9f, max_sum = -1e9f;
        int max_frame = 0;
        for (int i = 0; i < n_frames; ++i) {
            WakeWordDetector::FrameResult r = detector.ProcessFrame(&samples[i * 512]);
            if (r.wake_score > max_score) max_score = r.wake_score;
            if (r.window_sum > max_sum) {
                max_sum = r.window_sum;
                max_frame = i;
            }
            if (r.triggered) {
                ++triggers;
                std::printf("TRIGGER at frame %d (%.2fs) score=%.4f sum=%.4f\n",
                            i, i * 0.032, r.wake_score, r.window_sum);
            }
        }
        std::printf("frames=%d max_score=%.4f@frame%d(%.2fs) max_sum=%.4f triggers=%d threshold=%.3f\n",
                    n_frames, max_score, max_frame, max_frame * 0.032, max_sum, triggers, threshold);
        return 0;
    } catch (const std::exception& exc) {
        std::fprintf(stderr, "error: %s\n", exc.what());
        return 1;
    }
}
