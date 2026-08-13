"""Generate the "你好，爱芯" wake-word dataset with edge-tts.

Output: data/dataset.npz with entries per sample:
  feats  (T, 26) qmf-quantized log-fbank
  labels (T,)    0/1 frame labels (voice activity for positive samples, 0 for negatives)
  meta   list of dicts (kind, text, voice, label)
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

from fbank import fbank_frames, get_fbank, qmf_quantize


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

POS_TEXTS = [
    "你好，爱芯",
    "你好 爱芯",
    "你好,爱芯",
    "你好爱芯",
    "你好，Aixin",
    "你好，AIXIN",
    "你好! 爱芯",
    "你好，爱芯！",
]

RATES = [None, "+10%", "-10%"]

NEG_TEXTS = [
    "你好",
    "你好，小爱",
    "你好，小度",
    "你好，其他",
    "你好，再见",
    "你好，管家",
    "你好，天气",
    "你好，播放",
    "你好，打开",
    "你好，AI",
    "你好，AIX",
    "你好，爱星",
    "你好，艾心",
    "你好，爱信",
    "Hi，爱芯",
    "Hi, ESP",
    "爱芯你好",
    "打开空调",
    "好的，再见",
    "今天天气怎么样",
    "播放音乐",
    "我要睡觉了",
    "谢谢",
    "嗯，好的",
    "小度小度",
    "小爱同学",
    "天猫精灵",
    "设置闹钟",
    "打电话给妈妈",
    "导航去公司",
    "关闭灯光",
]

# voice split: train voices vs held-out validation voice
TRAIN_VOICES = [
    "zh-CN-XiaoxiaoNeural", "zh-CN-XiaoyiNeural", "zh-CN-YunjianNeural",
    "zh-CN-YunxiNeural", "zh-CN-YunxiaNeural", "zh-CN-XiaoxuanNeural",
    "zh-CN-liaoning-XiaobeiNeural", "zh-CN-shaanxi-XiaoniNeural",
    "en-US-JennyNeural", "en-US-GuyNeural", "en-US-AriaNeural",
    "en-US-AnaNeural", "en-US-ChristopherNeural", "en-US-EricNeural",
    "en-US-MichelleNeural", "en-US-RogerNeural",
]
VAL_VOICES = ["zh-CN-YunyangNeural"]
NEG_VOICES = [v for v in TRAIN_VOICES if v.startswith("zh-CN")] + VAL_VOICES


def tts_wav(text: str, voice: str, out: Path, rate: str | None = None) -> bool:
    cmd = ["edge-tts", "--voice", voice, "--text", text, "--write-media", str(out)]
    if rate:
        cmd += ["--rate", rate]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=90)
        return r.returncode == 0 and out.exists() and out.stat().st_size > 1000
    except Exception:
        return False


def resample16k(wav24k: Path, out16k: Path) -> bool:
    data, sr = sf.read(str(wav24k), dtype="int16")
    if sr == 16000:
        out16k.write_bytes(wav24k.read_bytes())
        return True
    if sr != 24000:
        raise ValueError(f"unexpected sr {sr}")
    x = data.astype(np.float32)
    n = int(len(x) * 16000 / sr)
    idx = np.linspace(0, len(x) - 1, n)
    y = np.interp(idx, np.arange(len(x)), x).astype(np.int16)
    sf.write(str(out16k), y, 16000, subtype="PCM_16")
    return True


def voice_activity_labels(samples: np.ndarray, n_frames: int) -> np.ndarray:
    """Frame-level VAD from int16 energy (relative threshold)."""
    rms = []
    for t in range(n_frames):
        blk = samples[t * 512 : t * 512 + 512].astype(np.float32)
        rms.append(float(np.sqrt((blk**2).mean())) if blk.size else 0.0)
    rms = np.asarray(rms)
    thr = max(100.0, rms.max() * 0.05)
    return (rms > thr).astype(np.int64)


def process_one(item: dict) -> dict | None:
    kind, text, voice = item["kind"], item["text"], item["voice"]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw = td / "raw.mp3"
        wav16 = td / "raw16.wav"
        if not tts_wav(text, voice, raw, rate=item.get("rate")):
            return None
        # edge-tts writes MP3 bytes even with a .wav name; decode with
        # soundfile (same decoder as wav_to_fbank at inference) so the fbank
        # gain scale matches training vs runtime. ffmpeg's mp3 decoder applies
        # a different gain and caused a ~100x fbank scale mismatch.
        data, sr = sf.read(str(raw), dtype="int16")
        if sr == 16000:
            samples = data
        else:
            wav24 = td / "raw24.wav"
            sf.write(str(wav24), data, sr, subtype="PCM_16")
            resample16k(wav24, wav16)
            samples, sr = sf.read(str(wav16), dtype="int16")
        win, coeff, bank_pos = get_fbank()
        feats = fbank_frames(samples, win, coeff, bank_pos)
        if feats.shape[0] < 20:
            return None
        feats = qmf_quantize(feats)
        labels = voice_activity_labels(samples, feats.shape[0]) if kind == "pos" else np.zeros(feats.shape[0], dtype=np.int64)
        # Keep the FULL audio (no trimming): at inference the wake word can
        # appear anywhere inside a 64-frame window, preceded by real silence.
        # Trimmed windows taught the model a shortcut ("window starts with
        # speech -> wake") that failed on full audio.
        return {
            "feats": feats.astype(np.float32),
            "labels": labels.astype(np.int64),
            "meta": item,
        }


def main():
    items = []
    for t in POS_TEXTS:
        for v in TRAIN_VOICES + VAL_VOICES:
            for r in RATES:
                items.append({"kind": "pos", "text": t, "voice": v, "rate": r})
    for t in NEG_TEXTS:
        for v in NEG_VOICES:
            items.append({"kind": "neg", "text": t, "voice": v, "rate": None})
    print(f"total items: {len(items)}")

    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(process_one, it) for it in items]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r:
                results.append(r)
            if i % 20 == 0:
                print(f"  {i}/{len(items)} done, {len(results)} ok")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA_DIR / "dataset.npz",
        feats=np.asarray([r["feats"] for r in results], dtype=object),
        labels=np.asarray([r["labels"] for r in results], dtype=object),
        meta=np.asarray([json.dumps(r["meta"]) for r in results], dtype=object),
    )
    pos = sum(1 for r in results if r["meta"]["kind"] == "pos")
    print(f"saved {len(results)} samples (pos={pos}, neg={len(results)-pos}) -> {DATA_DIR/'dataset.npz'}")


if __name__ == "__main__":
    main()
