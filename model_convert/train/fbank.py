"""Exact reference log-fbank (reconstructed from the reference implementation)."""
from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


EPS_F32 = struct.unpack("<f", struct.pack("<I", 0x33D6BF95))[0]  # ~1e-7


def hz2mel(hz):
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_filterbank(nfft, nfilter, low_freq, high_freq, samp_freq):
    """Exact replica of mel_filter_init: mel-space triangles, no normalization."""
    feat_width = nfft // 2 + 1
    lowmel = hz2mel(low_freq)
    if high_freq <= low_freq:
        high_freq = high_freq // 2
    highmel = hz2mel(high_freq)
    nyquist = samp_freq * 0.5
    mel_points = [lowmel + i * (highmel - lowmel) / (nfilter + 1) for i in range(nfilter + 2)]
    bin_mels = [hz2mel(i * nyquist / (nfft // 2)) for i in range(feat_width)]
    coeff, bank_pos = [], []
    for i in range(nfilter):
        start, stop = -1, -1
        for j in range(1, feat_width):  # bands_to_zero = 1
            lower = (bin_mels[j] - mel_points[i]) / (mel_points[i + 1] - mel_points[i])
            upper = (mel_points[i + 2] - bin_mels[j]) / (mel_points[i + 2] - mel_points[i + 1])
            temp = min(lower, upper)
            if lower > 0 and start == -1:
                start = j
            if upper <= 0 and stop == -1:
                stop = j - 1
            if temp > 0:
                coeff.append(temp)
        bank_pos.append((start, stop))
    return np.asarray(coeff, dtype=np.float32), bank_pos


def hann_periodic_512() -> np.ndarray:
    """window array from the reference implementation (torch hann_window(512, periodic=True))."""
    return torch.hann_window(512, periodic=True).numpy().astype(np.float32)


def fbank_frames(samples: np.ndarray, win: np.ndarray, coeff, bank_pos, n_mels=26) -> np.ndarray:
    """(T, 26) log-fbank from int16 samples, exact reference pipeline."""
    n_frames = (len(samples) - 512) // 512 + 1
    out = np.zeros((n_frames, n_mels), dtype=np.float32)
    offs = []
    off = 0
    for s, e in bank_pos:
        offs.append((s, e, off, off + e - s + 1))
        off += e - s + 1
    for t in range(n_frames):
        x = samples[t * 512 : t * 512 + 512].astype(np.float32) * (1.0 / 32768.0) * win
        spec = np.abs(np.fft.rfft(x, n=512))
        for f, (s, e, c0, c1) in enumerate(offs):
            out[t, f] = np.dot(spec[s : e + 1], coeff[c0:c1])
    return np.log(out + np.float32(EPS_F32))


def qmf_quantize(feats: np.ndarray) -> np.ndarray:
    """reference quantization: clip [-32, 32] then q = trunc(x*1024 + 0.5) / 1024."""
    q = np.trunc(np.clip(feats, -32, 32) * 1024.0 + 0.5).astype(np.int16)
    return q.astype(np.float32) / 1024.0


_FBANK_CACHE: dict = {}


def get_fbank(sr: int = 16000, n_mels: int = 26):
    """Cached (win, coeff, bank_pos) for the reference fbank."""
    key = (sr, n_mels)
    if key not in _FBANK_CACHE:
        _FBANK_CACHE[key] = (
            hann_periodic_512(),
            *mel_filterbank(512, n_mels, 80, 7000, sr),
        )
    return _FBANK_CACHE[key]


def wav_to_fbank(path: str | Path) -> tuple[np.ndarray, int]:
    """16k mono int16 wav -> (T, 26) qmf-quantized fbank."""
    data, sr = sf.read(str(path), dtype="int16")
    if sr != 16000:
        raise ValueError(f"only 16kHz supported, got {sr}")
    win, coeff, bank_pos = get_fbank()
    feats = fbank_frames(data, win, coeff, bank_pos)
    return qmf_quantize(feats), len(data)
