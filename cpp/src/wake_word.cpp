#include "wake_word.hpp"

#include "model_runner.hpp"

#include <cmath>
#include <complex>
#include <cstring>
#include <utility>
#include <vector>

namespace {

constexpr int kNfft = 512;
constexpr int kFeatWidth = kNfft / 2 + 1;  // 257
constexpr int kNumMels = 26;
constexpr int kFrameLen = 512;
constexpr int kWindow = 64;
constexpr float kEps = 1.1920928955078125e-7f;  // reference log epsilon

// periodic-512 Hann (== reference window array)
std::vector<float> MakeHann() {
    std::vector<float> w(kFrameLen);
    for (int n = 0; n < kFrameLen; ++n) {
        w[n] = 0.5f * (1.0f - std::cos(2.0 * M_PI * n / kFrameLen));
    }
    return w;
}

float Hz2Mel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

struct MelBank {
    std::vector<float> coeff;
    std::vector<std::pair<int, int>> pos;  // (start, stop)
};

MelBank MakeMelBank(int nfilter, int low_freq, int high_freq, int samp_freq) {
    MelBank bank;
    const float lowmel = Hz2Mel(static_cast<float>(low_freq));
    const float highmel = Hz2Mel(static_cast<float>(high_freq));
    const float nyquist = samp_freq * 0.5f;
    std::vector<float> mel_points(nfilter + 2);
    for (int i = 0; i < nfilter + 2; ++i) {
        mel_points[i] = lowmel + i * (highmel - lowmel) / (nfilter + 1);
    }
    std::vector<float> bin_mels(kFeatWidth);
    for (int i = 0; i < kFeatWidth; ++i) {
        bin_mels[i] = Hz2Mel(i * nyquist / (kNfft / 2));
    }
    int off = 0;
    for (int i = 0; i < nfilter; ++i) {
        int start = -1, stop = -1;
        for (int j = 1; j < kFeatWidth; ++j) {  // bands_to_zero = 1
            const float lower =
                (bin_mels[j] - mel_points[i]) / (mel_points[i + 1] - mel_points[i]);
            const float upper =
                (mel_points[i + 2] - bin_mels[j]) / (mel_points[i + 2] - mel_points[i + 1]);
            const float temp = std::min(lower, upper);
            if (lower > 0 && start == -1) start = j;
            if (upper <= 0 && stop == -1) stop = j - 1;
            if (temp > 0.f) {
                bank.coeff.push_back(temp);
                ++off;
            }
        }
        bank.pos.emplace_back(start, stop);
    }
    (void)off;
    return bank;
}

// radix-2 iterative FFT (512 = 2^9)
void Fft(std::vector<std::complex<double>>& a) {
    const int n = static_cast<int>(a.size());
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        const double ang = -2.0 * M_PI / len;
        const std::complex<double> wlen(std::cos(ang), std::sin(ang));
        for (int i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (int k = 0; k < len / 2; ++k) {
                std::complex<double> u = a[i + k];
                std::complex<double> v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

}  // namespace

struct WakeWordDetector::Impl {
    ModelRunner runner;
    std::vector<float> hann = MakeHann();
    MelBank mel = MakeMelBank(kNumMels, 80, 7000, 16000);
    std::vector<float> window;  // (kWindow, 26) ring, oldest first
    std::vector<float> scores;
    float threshold;
    int det_win;
    int mel_off = 0;

    Impl(const std::string& model_path, float th, int dw)
        : runner(model_path, "wakeup_axera"), threshold(th), det_win(dw),
          window(kWindow * kNumMels, 0.f) {
        for (const auto& p : mel.pos) mel_off += p.second - p.first + 1;
    }

    float FrameFbank(const int16_t* samples) {
        std::vector<std::complex<double>> buf(kNfft, 0.0);
        for (int i = 0; i < kFrameLen; ++i) {
            buf[i] = static_cast<double>(samples[i]) / 32768.0 * hann[i];
        }
        Fft(buf);
        std::vector<float> spec(kFeatWidth);
        spec[0] = static_cast<float>(std::abs(buf[0]));
        for (int i = 1; i < kFeatWidth - 1; ++i) {
            spec[i] = static_cast<float>(std::abs(buf[i]));
        }
        spec[kFeatWidth - 1] = static_cast<float>(std::abs(buf[kNfft / 2]));

        float out[kNumMels];
        int off = 0;
        for (int f = 0; f < kNumMels; ++f) {
            const int s = mel.pos[f].first, e = mel.pos[f].second;
            double acc = 0.0;
            for (int j = s; j <= e; ++j) {
                acc += spec[j] * mel.coeff[off++];
            }
            out[f] = std::log(static_cast<float>(acc) + kEps);
        }
        // Q6.10 reference quantization
        for (int f = 0; f < kNumMels; ++f) {
            const float c = std::max(-32.f, std::min(32.f, out[f]));
            const float q = std::trunc(c * 1024.0f + 0.5f) / 1024.0f;
            window[(kWindow - 1) * kNumMels + f] = q;
        }
        // model input: (1,26,64) last window
        std::vector<float> feed(kNumMels * kWindow);
        for (int t = 0; t < kWindow; ++t) {
            for (int f = 0; f < kNumMels; ++f) {
                feed[f * kWindow + t] = window[t * kNumMels + f];
            }
        }
        std::vector<std::vector<float>> inputs;
        inputs.push_back(std::move(feed));
        std::vector<std::vector<float>> outs = runner.Run(inputs);  // (1,2,64)
        return outs[0][1 * kWindow + (kWindow - 1)];  // logits[1, -1]
    }
};

WakeWordDetector::WakeWordDetector(const std::string& model_path, float threshold, int det_win)
    : impl_(new Impl(model_path, threshold, det_win)) {}

WakeWordDetector::~WakeWordDetector() { delete impl_; }

WakeWordDetector::FrameResult WakeWordDetector::ProcessFrame(const int16_t* samples) {
    std::memmove(impl_->window.data(), impl_->window.data() + kNumMels,
                 (kWindow - 1) * kNumMels * sizeof(float));
    impl_->scores.push_back(impl_->FrameFbank(samples));
    if (static_cast<int>(impl_->scores.size()) > impl_->det_win) {
        impl_->scores.erase(impl_->scores.begin());
    }
    FrameResult r;
    r.wake_score = impl_->scores.back();
    for (float s : impl_->scores) r.window_sum += s;
    r.triggered = r.window_sum > impl_->threshold;
    return r;
}

void WakeWordDetector::Reset() {
    std::fill(impl_->window.begin(), impl_->window.end(), 0.f);
    impl_->scores.clear();
}
