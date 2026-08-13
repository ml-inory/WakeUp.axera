"""WakeUp.axera「你好，爱芯」唤醒词检测 Demo——复制即用。

在 AX650/AX630 板上运行：
    bash setup.sh && bash run.sh
输入为 16kHz 单声道 int16 PCM（512 样本/32ms 一帧），输出逐帧唤醒得分与触发结果。
"""
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import axengine  # noqa: F401
    AX_AVAILABLE = True
except Exception:
    AX_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "ax650" / "model.axmodel"
POS_WAV = ROOT / "models" / "sample_nihao_aixin.wav"
NEG_WAV = ROOT / "models" / "sample_nihao_qita.wav"
THRESHOLD = 0.615  # DET_MODE_95；DET_MODE_90 用 0.608


def main():
    ap = argparse.ArgumentParser(description="WakeUp.axera 唤醒词检测 Demo")
    ap.add_argument("--model", default=str(MODEL),
                    help="axmodel 路径（默认 models/ax650/model.axmodel；AX620E 用 models/ax620e/model.axmodel）")
    args = ap.parse_args()
    if not AX_AVAILABLE:
        print("当前主机没有 AX 芯片（pyaxengine 不可用），无法运行 NPU 推理。")
        print("本交付包为 NPU 专用版，请在 AX650/AX630/AX620E 板端执行：")
        print("  bash setup.sh && bash run.sh")
        return

    import soundfile as sf
    from wakeup_axera_sdk.detector import detect_stream
    from wakeup_axera_sdk.inference import ModelSession

    session = ModelSession(args.model)
    for wav, expect in ((POS_WAV, "应触发"), (NEG_WAV, "应静默")):
        data, sr = sf.read(str(wav), dtype="int16")
        res = detect_stream(session, data, det_win=3, threshold=THRESHOLD)
        status = "TRIGGERED" if res["triggered"] else "silent"
        print(f"{wav.name}: max_score={res['max_score']:.3f}  "
              f"max_sum3={res['max_sum']:.3f}  {status}（{expect}）")
    print(f"阈值: {THRESHOLD}（3 帧滑动和）  backend: {session.backend}")
    print("✅ WakeUpModel 唤醒词检测运行成功")


if __name__ == "__main__":
    main()
