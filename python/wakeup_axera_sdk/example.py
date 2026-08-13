"""WakeUpModel 唤醒词检测示例：输入 int16 WAV，输出逐帧得分与触发结果。"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wakeup_axera_sdk.inference import ModelSession
from wakeup_axera_sdk.detector import detect_stream


def main():
    parser = argparse.ArgumentParser(description="wakeup_axera wake word detection example")
    parser.add_argument("--model", required=True, help="AXMODEL 路径")
    parser.add_argument("--input", required=True, help="int16 WAV (16k mono)")
    parser.add_argument("--threshold", type=float, default=0.615, help="DET_MODE_95=0.615 / DET_MODE_90=0.608")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    args = parser.parse_args()

    data, sr = sf.read(args.input, dtype="int16")
    if sr != 16000:
        raise SystemExit(f"仅支持 16kHz，got {sr}")
    session = ModelSession(args.model)
    result = detect_stream(session, data, det_win=3, threshold=args.threshold)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"backend: {session.backend}")
    print(f"frames: {len(result['frame_scores'])}")
    print(f"max wake score: {result['max_score']:.4f}")
    print(f"max 3-frame sum: {result['max_sum']:.4f}  (threshold {args.threshold})")
    print(f"TRIGGERED: {result['triggered']}")
    print(f"saved to: {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()
