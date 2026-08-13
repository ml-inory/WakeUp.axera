import numpy as np

DEFAULT_DET_WIN = 3
DET_THRESHOLD_90 = 0.608
DET_THRESHOLD_95 = 0.615


def wake_scores(logits: np.ndarray) -> np.ndarray:
    """(M,2,64) 或 (1,2,64) -> 每窗最后一帧 wake 通道得分 (M,)。"""
    a = np.asarray(logits)
    if a.ndim == 3:
        return a[:, 1, -1].astype(np.float64)
    if a.ndim == 4:
        return a[:, 0, 1, -1].astype(np.float64)
    raise ValueError(f'unexpected logits shape {a.shape}')


def detect(scores: np.ndarray, det_win: int = DEFAULT_DET_WIN, threshold: float = DET_THRESHOLD_95) -> dict:
    """reference 触发逻辑：det_win 帧滑动求和 vs 阈值。"""
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError('scores 应为 (M,) 每帧得分')
    kernel = np.ones(det_win)
    sums = np.convolve(s, kernel, mode='valid')
    return {
        'frame_scores': s.tolist(),
        'window_sums': sums.tolist(),
        'max_score': float(s.max()) if s.size else 0.0,
        'max_sum': float(sums.max()) if sums.size else 0.0,
        'triggered': bool((sums > threshold).any()),
        'threshold': threshold,
        'det_win': det_win,
    }


def postprocess(*arrays):
    """模型输出 (M,2,64) -> 检测结果 dict。"""
    if len(arrays) != 1:
        raise ValueError('wakeup 单输出')
    scores = wake_scores(arrays[0])
    return detect(scores)