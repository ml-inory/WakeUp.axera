"""Export the fine-tuned "你好，爱芯" WakeNet9 to static ONNX (1,26,64) -> logits."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "WakeUp.axera/model_convert"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_code import WakeNet9Train, detect_score
from fbank import wav_to_fbank


ROOT = Path(__file__).resolve().parent
REF_MODEL = Path(__file__).resolve().parent / "ref_model"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/final_model.pt")
    ap.add_argument("--output", type=Path, default=ROOT / "onnx/wakeup_nihaoxin.onnx")
    ap.add_argument("--check-wav", type=Path, nargs="*", default=[])
    args = ap.parse_args()

    net = WakeNet9Train(REF_MODEL)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    net.load_state_dict(state)
    net.eval()

    # sanity: wake word triggers, negative stays silent
    for w in args.check_wav:
        feats, _ = wav_to_fbank(w)
        x = torch.from_numpy(feats.T).unsqueeze(0)
        with torch.no_grad():
            logits = net(x)[0].numpy()
        s = detect_score(logits)
        print(f"check {Path(w).name}: sum3_max={s.max():.3f} fire={(s > 0.615).any()}")

    W = 64
    dummy = torch.zeros(1, 26, W)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net,
        dummy,
        args.output,
        input_names=["fbank"],
        output_names=["logits"],
        dynamic_axes=None,
        opset_version=13,
    )
    print(f"exported {args.output}")

    # ONNX vs torch consistency on a real fbank window
    import onnxruntime as ort
    sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    feats, _ = wav_to_fbank(ROOT / "data/accept_nihao_aixin.wav")
    T = feats.shape[0]
    if T < W:
        feats = np.pad(feats, ((0, W - T), (0, 0)))
    x = feats[:W].T[None].astype(np.float32)
    with torch.no_grad():
        ref = net(torch.from_numpy(x)).numpy()
    onnx_out = sess.run(None, {"fbank": x})[0]
    diff = np.abs(ref - onnx_out).max()
    print(f"torch vs onnx max abs diff: {diff:.2e}")


if __name__ == "__main__":
    main()
