"""Trainable WakeNet9 replica for the "Hi,爱芯" wake word.

Architecture matches the shipped ONNX/AXMODEL (fbank in, (B,2,T) log-softmax out):
  conv1(26->128) -> LN -> LIGLU x5 -> residual -> conv2(128->64) -> L2norm
  -> conv3(64->2) -> log_softmax

Weights are initialized from the reference esp-sr wn9 model (conv3 x4 correction
kept, matching the validated float network), then ALL parameters are unfrozen
for fine-tuning.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_loader import load_model  # from WakeUp.axera/model_convert


EPS_F32 = np.float32(1.1920928955078125e-07)


class WakeNet9Train(nn.Module):
    def __init__(self, model_dir: Path | str, conv3_scale: float = 4.0, ln_eps: float = 1e-5):
        super().__init__()
        m = load_model(model_dir)
        self.cfg = m.config

        # conv1: 26 -> 128
        self.conv1 = nn.Conv1d(26, 128, 1)
        self.conv1.weight.data = torch.tensor(m.weights("conv1_w"), dtype=torch.float32).unsqueeze(-1)
        self.conv1.bias.data = torch.tensor(m.bias("conv1_b"), dtype=torch.float32)
        self.ln0 = nn.LayerNorm(128, eps=ln_eps)
        self.ln0.weight.data = torch.tensor(m.bias("conv1_gamma"), dtype=torch.float32)
        self.ln0.bias.data = torch.tensor(m.bias("conv1_beta"), dtype=torch.float32)

        # LIGLU x5
        self.liglus = nn.ModuleList()
        for i in range(1, 6):
            li = m.config[f"liglu{i}"]
            C = 128
            lg = nn.Module()
            lg.rate = int(li["rate"])
            lg.wg = nn.Conv1d(C, 2 * C, 1)
            lg.wg.weight.data = torch.tensor(m.weights(f"liglu{i}_wg"), dtype=torch.float32).unsqueeze(-1)
            lg.wg.bias.data = torch.tensor(m.bias(f"liglu{i}_bg"), dtype=torch.float32)
            lg.wd = nn.Conv1d(C, C, 5, groups=C, dilation=lg.rate)
            lg.wd.weight.data = torch.tensor(m.weights(f"liglu{i}_wd"), dtype=torch.float32).reshape(C, 1, 5)
            lg.wd.bias.data = torch.tensor(m.bias(f"liglu{i}_bd"), dtype=torch.float32)
            lg.ln = nn.LayerNorm(C, eps=ln_eps)
            lg.ln.weight.data = torch.tensor(m.bias(f"liglu{i}_gamma"), dtype=torch.float32)
            lg.ln.bias.data = torch.tensor(m.bias(f"liglu{i}_beta"), dtype=torch.float32)
            lg.wo = nn.Conv1d(C, C, 1)
            lg.wo.weight.data = torch.tensor(m.weights(f"liglu{i}_wo"), dtype=torch.float32).unsqueeze(-1)
            lg.wo.bias.data = torch.tensor(m.bias(f"liglu{i}_bo"), dtype=torch.float32)
            self.liglus.append(lg)

        # conv2 / conv3
        self.conv2 = nn.Conv1d(128, 64, 1)
        self.conv2.weight.data = torch.tensor(m.weights("conv2_w"), dtype=torch.float32).unsqueeze(-1)
        self.conv2.bias.data = torch.tensor(m.bias("conv2_b"), dtype=torch.float32)
        self.conv3 = nn.Conv1d(64, 2, 1)
        self.conv3.weight.data = torch.tensor(m.weights("conv3_w"), dtype=torch.float32).unsqueeze(-1) * conv3_scale
        self.conv3.bias.data = torch.zeros(2)

        # unfreeze everything for fine-tuning
        for p in self.parameters():
            p.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 26, T) log-fbank (Q6.10 real values) -> (B, 2, T) raw logits.

        Raw logits (not log-softmax) match the shipped ONNX/SDK: wake score =
        logits[1, frame], and the trigger threshold (0.615 for a 3-frame sum)
        is defined in this logit scale.
        """
        h = self.conv1(x)
        h0 = self.ln0(h.transpose(1, 2)).transpose(1, 2)
        y = h0
        for lg in self.liglus:
            z = lg.wg(y)
            C = z.shape[1] // 2
            gate, filt = z[:, :C], z[:, C:]
            prod = torch.sigmoid(gate) * filt
            pad = (5 - 1) * lg.rate
            y = lg.wd(F.pad(prod, (pad, 0)))
            y = lg.ln(y.transpose(1, 2)).transpose(1, 2)
            y = lg.wo(y)
        y = y + h0
        y = self.conv2(y)
        y = y / torch.clamp(y.norm(dim=1, keepdim=True), min=1e-7)
        y = self.conv3(y)
        return y


def detect_score(logits: np.ndarray, det_win: int = 3) -> np.ndarray:
    """(2, T) logits -> sliding-window sum of the wake channel (reference trigger logic)."""
    wake = logits[1]
    kernel = np.ones(det_win)
    return np.convolve(wake, kernel, mode="valid")


def model_thresholds(model_dir: Path) -> tuple[float, float]:
    """DET_MODE_90 / DET_MODE_95 from _MODEL_INFO_ (e.g. '..._3_0.608_0.615')."""
    info_path = Path(model_dir) / "_MODEL_INFO_"
    if not info_path.exists():
        return 0.608, 0.615
    nums = [float(m) for m in re.findall(r"-?\d+\.?\d*", info_path.read_text(errors="replace"))]
    if len(nums) < 2:
        return 0.608, 0.615
    return nums[-2], nums[-1]


def load_pt_checkpoint(model_dir: Path, ckpt: str | Path, conv3_scale: float = 4.0) -> WakeNet9Train:
    """Build model from reference weights, then overwrite with fine-tuned checkpoint."""
    net = WakeNet9Train(model_dir, conv3_scale=conv3_scale)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    net.load_state_dict(state)
    return net
