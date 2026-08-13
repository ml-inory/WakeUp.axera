"""WakeUpModel replica: logfbank + conv1/LIGLU/conv2/conv3, built from reference weights."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_loader import ReferenceModel, load_model


def mel_filterbank(n_mels: int, n_fft: int, sr: int, fmin: float, fmax: float) -> torch.Tensor:
    """Mel filterbank matrix (n_mels, n_fft//2+1); norm='slaney' or 'htk'."""
    return mel_filterbank_impl(n_mels, n_fft, sr, fmin, fmax, "slaney")


def mel_filterbank_impl(n_mels, n_fft, sr, fmin, fmax, norm: str = "slaney") -> torch.Tensor:
    n_freqs = n_fft // 2 + 1
    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    # reference: coeff[k] = hz2mel(k * samp_freq / (nfft/2+1)), triangular in mel space
    bin_mel = 2595.0 * np.log10(1.0 + (np.arange(n_freqs) * sr / (n_fft / 2 + 1)) / 700.0)
    filt = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = mel_points[i], mel_points[i + 1], mel_points[i + 2]
        left = (bin_mel - lo) / (mid - lo)
        right = (hi - bin_mel) / (hi - mid)
        filt[i] = np.clip(np.minimum(left, right), 0.0, 1.0)
    if norm == "slaney":
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
        filt *= enorm[:, None]
    return torch.from_numpy(filt)


class LogFbank(nn.Module):
    """32ms / 32ms log-fbank (26 bands), matching wakeup config."""

    def __init__(
        self,
        n_mels: int = 26,
        win_len: int = 512,
        win_step: int = 512,
        sr: int = 16000,
        fmin: float = 80.0,
        fmax: float = 7000.0,
        eps: float = 1.1920928955078125e-07,
        preemph: float = 0.0,
        power: bool = False,
        log_mode: int = 1,
        mel_norm: str = "slaney",
    ):
        super().__init__()
        n_fft = 512
        self.win_len = win_len
        self.win_step = win_step
        self.eps = eps
        self.preemph = preemph
        self.power = power
        self.log_mode = log_mode
        self.mel_norm = mel_norm
        win = torch.hann_window(win_len, periodic=False)
        self.register_buffer("window", win)
        self.register_buffer("mel", mel_filterbank_impl(n_mels, n_fft, sr, fmin, fmax, mel_norm))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N) int16 waveform -> float
        if self.preemph:
            x = torch.cat([x[:, :1], x[:, 1:] - self.preemph * x[:, :-1]], dim=1)
        x = x.unfold(1, self.win_len, self.win_step)  # (B, T, win)
        x = x * self.window
        spec = torch.fft.rfft(x, n=512, dim=2)
        spec = spec.abs() ** 2 if self.power else spec.abs()
        fbank = torch.matmul(spec, self.mel.T)  # (B, T, 26)
        if self.log_mode == 1:
            fbank = torch.log(fbank + self.eps)
        else:
            fbank = torch.log(torch.clamp(fbank, min=self.eps))
        return fbank.permute(0, 2, 1)  # (B, 26, T)


def _causal_pad(x: torch.Tensor, kernel: int, rate: int) -> torch.Tensor:
    pad = (kernel - 1) * rate
    return F.pad(x, (pad, 0))


class LIGLU(nn.Module):
    """Lightweight gated linear unit: 1x1 split -> depthwise dilated conv -> gate -> 1x1 -> LN."""

    def __init__(self, wg, bg, wd, bd, wo, bo, beta, gamma, rate: int, kernel: int, variant: str = "a", use_tanh: bool = False, ln_eps: float = 1e-5, act: str = "none"):
        super().__init__()
        self.rate = rate
        self.kernel = kernel
        self.variant = variant
        self.use_tanh = use_tanh
        self.act = act
        C = wg.shape[0] // 2
        self.wg = nn.Parameter(torch.tensor(wg, dtype=torch.float32).unsqueeze(-1), requires_grad=False)
        self.bg = nn.Parameter(torch.tensor(bg, dtype=torch.float32), requires_grad=False)
        # wd: 640 = C*K stored as (C, K) — reshape from flat (1, C*K) row
        self.wd = nn.Parameter(torch.tensor(wd.reshape(C, 1, kernel), dtype=torch.float32), requires_grad=False)
        self.bd = nn.Parameter(torch.tensor(bd, dtype=torch.float32), requires_grad=False)
        self.wo = nn.Parameter(torch.tensor(wo, dtype=torch.float32).unsqueeze(-1), requires_grad=False)
        self.bo = nn.Parameter(torch.tensor(bo, dtype=torch.float32), requires_grad=False)
        self.ln = nn.LayerNorm(C, eps=ln_eps)
        self.ln.weight.data = torch.tensor(beta, dtype=torch.float32)
        self.ln.bias.data = torch.tensor(gamma, dtype=torch.float32)
        for p in self.ln.parameters():
            p.requires_grad = False

    def _dw(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        x = _causal_pad(x, self.kernel, self.rate)
        return F.conv1d(x, self.wd, self.bd, groups=C, dilation=self.rate)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.act == "none":
            return x
        if self.act == "swish":
            return x * torch.sigmoid(x)
        if self.act == "doubleswish":
            return x * torch.sigmoid(x) + x
        raise ValueError(self.act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.conv1d(x, self.wg, self.bg)
        g, f = z[:, : z.shape[1] // 2], z[:, z.shape[1] // 2 :]
        if self.variant == "a":
            y = torch.sigmoid(g) * f
            if self.use_tanh:
                y = torch.tanh(y)
        elif self.variant == "c":
            y = torch.sigmoid(self._dw(g)) * f
        else:
            raise ValueError(self.variant)
        y = self._dw(y)
        y = self.ln(y.transpose(1, 2)).transpose(1, 2)
        y = self._act(y)
        return F.conv1d(y, self.wo, self.bo)


def quantize(x: torch.Tensor, mbit: int = 8, fbit: int = 8) -> torch.Tensor:
    """Simulate reference activation quantization: q = clip(round(x*2^fbit)) / 2^fbit."""
    lo, hi = -(2 ** (mbit - 1)), 2 ** (mbit - 1) - 1
    return torch.clamp(torch.round(x * (2**fbit)), lo, hi) / (2**fbit)


class WakeUpModel(nn.Module):
    def __init__(
        self,
        model: ReferenceModel,
        variant: str = "a",
        use_tanh: bool = False,
        fbank_kwargs: dict | None = None,
        with_fbank: bool = True,
        liglu_act: str = "none",
        use_residual: bool = True,
        use_l2norm: bool = True,
        input_gain: float = 1.0,
        ln_eps: float = 1e-5,
    ):
        super().__init__()
        cfg = model.config
        self.liglu_num = int(cfg.get("liglu_num", 5))
        self.liglu_act = liglu_act
        self.use_residual = use_residual
        self.use_l2norm = use_l2norm
        self.input_gain = input_gain
        self.ln_eps = ln_eps
        self.feat_dim = int(cfg.get("features_dim", 26))
        self.win_len = int(cfg.get("win_len", 32))
        self.win_step = int(cfg.get("win_step", 32))
        self.with_fbank = with_fbank
        if with_fbank:
            fk = dict(n_mels=self.feat_dim, win_len=self.win_len * 16, win_step=self.win_step * 16)
            fk.update(fbank_kwargs or {})
            self.fbank = LogFbank(**fk)
        w1 = model.weights("conv1_w")
        b1 = model.bias("conv1_b")
        self.conv1 = nn.Conv1d(w1.shape[1], w1.shape[0], 1)
        self.conv1.weight.data = torch.tensor(w1, dtype=torch.float32).unsqueeze(-1)
        self.conv1.bias.data = torch.tensor(b1, dtype=torch.float32)
        for p in self.conv1.parameters():
            p.requires_grad = False
        C = w1.shape[0]
        self.ln0 = nn.LayerNorm(C, eps=ln_eps)
        self.ln0.weight.data = torch.tensor(model.bias("conv1_beta"), dtype=torch.float32)
        self.ln0.bias.data = torch.tensor(model.bias("conv1_gamma"), dtype=torch.float32)
        for p in self.ln0.parameters():
            p.requires_grad = False
        self.liglus = nn.ModuleList()
        for i in range(1, self.liglu_num + 1):
            li = cfg.get(f"liglu{i}", {})
            self.liglus.append(
                LIGLU(
                    model.weights(f"liglu{i}_wg"),
                    model.bias(f"liglu{i}_bg"),
                    model.weights(f"liglu{i}_wd").reshape(-1),
                    model.bias(f"liglu{i}_bd"),
                    model.weights(f"liglu{i}_wo"),
                    model.bias(f"liglu{i}_bo"),
                    model.bias(f"liglu{i}_beta"),
                    model.bias(f"liglu{i}_gamma"),
                    rate=int(li.get("rate", 1)),
                    kernel=int(li.get("size", 5)),
                    variant=variant,
                    use_tanh=use_tanh,
                    ln_eps=ln_eps,
                )
            )
        w2, b2 = model.weights("conv2_w"), model.bias("conv2_b")
        w3 = model.weights("conv3_w")
        b3 = model.bias("conv3_b") if "conv3_b" in model.tensors else None
        self.conv2 = nn.Conv1d(w2.shape[1], w2.shape[0], 1)
        self.conv2.weight.data = torch.tensor(w2, dtype=torch.float32).unsqueeze(-1)
        self.conv2.bias.data = torch.tensor(b2, dtype=torch.float32)
        self.conv3 = nn.Conv1d(w3.shape[1], w3.shape[0], 1)
        self.conv3.weight.data = torch.tensor(w3, dtype=torch.float32).unsqueeze(-1)
        if b3 is not None:
            self.conv3.bias.data = torch.tensor(b3, dtype=torch.float32)
        else:
            self.conv3.bias.data.zero_()
        for p in list(self.conv2.parameters()) + list(self.conv3.parameters()):
            p.requires_grad = False

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.liglu_act == "none":
            return x
        if self.liglu_act == "swish":
            return x * torch.sigmoid(x)
        if self.liglu_act == "doubleswish":
            return x * torch.sigmoid(x) + x
        raise ValueError(self.liglu_act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_fbank:
            x = self.fbank(x) * self.input_gain
        x = self.conv1(x)
        h0 = self.ln0(x.transpose(1, 2)).transpose(1, 2)
        x = h0
        for liglu in self.liglus:
            x = liglu(x)
            x = self._act(x)
        if self.use_residual:
            x = x + h0
        x = self.conv2(x)
        if self.use_l2norm:
            x = x / torch.clamp(x.norm(dim=1, keepdim=True), min=1e-7)
        x = self.conv3(x)
        return F.log_softmax(x, dim=1)  # (B, 2, T)


def build_wakeup(model_dir: Path, **kwargs) -> WakeUpModel:
    m = load_model(model_dir)
    return WakeUpModel(m, **kwargs)


def p_wake(model: WakeUpModel, x: torch.Tensor) -> np.ndarray:
    """Probability of the wake-word class per frame."""
    with torch.no_grad():
        out = model(x)
    return out.exp()[0, 1].cpu().numpy()
