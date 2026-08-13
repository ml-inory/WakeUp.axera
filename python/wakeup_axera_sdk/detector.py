"""流式唤醒词检测器：fbank -> 64 帧窗口 -> 模型 -> 3 帧滑动和触发。"""
from __future__ import annotations

import numpy as np

from .inference import ModelSession
from .preprocess import FbankStream
from .postprocess import wake_scores, detect, DET_THRESHOLD_95, DET_THRESHOLD_90


class WakeWordDetector:
    """对齐 reference 触发语义：每 32ms 帧喂入 512 个 int16 样本。

    process(frame) 返回 (wake_score, window_sum, triggered)。
    window_sum 为最近 det_win 帧 wake 得分之和；triggered = window_sum > threshold。
    """

    def __init__(self, model_path: str, det_win: int = 3,
                 threshold: float = DET_THRESHOLD_95, providers=None):
        self.session = ModelSession(model_path, providers=providers)
        self.fbank = FbankStream()
        self.det_win = det_win
        self.threshold = threshold
        self.scores: list[float] = []

    def process(self, frame: np.ndarray) -> dict:
        """frame: int16 (512,) 32ms PCM。返回 {wake_score, window_sum, triggered}。"""
        if frame.shape[0] != 512:
            raise ValueError(f"帧长必须为 512（32ms@16k），got {frame.shape[0]}")
        win = self.fbank.push(frame)
        logits = self.session.run_named([win])[0]
        score = float(wake_scores(logits).reshape(-1)[0])
        self.scores.append(score)
        if len(self.scores) > self.det_win:
            self.scores = self.scores[-self.det_win:]
        window_sum = float(np.sum(self.scores))
        return {
            "wake_score": score,
            "window_sum": window_sum,
            "triggered": bool(window_sum > self.threshold),
            "threshold": self.threshold,
        }

    def reset(self):
        self.fbank.reset()
        self.scores.clear()


def detect_stream(session, pcm: np.ndarray, det_win: int = 3,
                  threshold: float = DET_THRESHOLD_95) -> dict:
    """离线处理整段 int16 PCM：返回逐帧得分与触发结果。"""
    det = WakeWordDetector.__new__(WakeWordDetector)
    det.session = session
    det.fbank = FbankStream()
    det.det_win = det_win
    det.threshold = threshold
    det.scores = []
    pcm = np.asarray(pcm, dtype=np.int16).reshape(-1)
    frames = (pcm.shape[0] - 512) // 512 + 1
    per_frame = []
    for i in range(frames):
        r = det.process(pcm[i * 512 : i * 512 + 512])
        per_frame.append(r)
    scores = np.array([r["wake_score"] for r in per_frame])
    return detect(scores, det_win, threshold)
