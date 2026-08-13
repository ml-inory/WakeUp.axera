import numpy as np
import struct

EPS_F32 = struct.unpack('<f', struct.pack('<I', 0x33D6BF95))[0]


def _hz2mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_filterbank(nfft=512, nfilter=26, low=80, high=7000, sr=16000):
    feat_width = nfft // 2 + 1
    lowmel = _hz2mel(low)
    highmel = _hz2mel(high)
    nyquist = sr * 0.5
    mel_points = [lowmel + i * (highmel - lowmel) / (nfilter + 1) for i in range(nfilter + 2)]
    bin_mels = [_hz2mel(i * nyquist / (nfft // 2)) for i in range(feat_width)]
    coeff, bank_pos = [], []
    for i in range(nfilter):
        start, stop = -1, -1
        for j in range(1, feat_width):
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


_HANN = np.hanning(513)[:-1].astype(np.float32)  # periodic 512 hann == reference window array
_COEFF, _BANK_POS = _mel_filterbank()
_OFFS = []
_off = 0
for _s, _e in _BANK_POS:
    _OFFS.append((_s, _e, _off, _off + _e - _s + 1))
    _off += _e - _s + 1


def frame_fbank(samples: np.ndarray) -> np.ndarray:
    """单帧 (512,) int16 -> (26,) log-fbank (Q6.10 值，float)。"""
    x = samples.astype(np.float32) * (1.0 / 32768.0) * _HANN
    spec = np.abs(np.fft.rfft(x, n=512))
    out = np.zeros(26, dtype=np.float32)
    for f, (s, e, c0, c1) in enumerate(_OFFS):
        out[f] = np.dot(spec[s : e + 1], _COEFF[c0:c1])
    return np.log(out + EPS_F32)


class FbankStream:
    """流式 fbank：每帧 512 个 int16 样本 -> 维护 64 帧窗口，输出 (1,26,64)。"""

    def __init__(self, window: int = 64):
        self.window = window
        self.buf = np.zeros((window, 26), dtype=np.float32)

    def push(self, samples: np.ndarray) -> np.ndarray:
        """喂入一帧 int16 (512,)，返回当前窗口 (1,26,64) 模型输入。"""
        f = frame_fbank(samples)
        q = np.trunc(np.clip(f, -32, 32) * 1024.0 + 0.5) / 1024.0
        self.buf = np.roll(self.buf, -1, axis=0)
        self.buf[-1] = q
        return self.buf.T[np.newaxis].astype(np.float32)

    def reset(self):
        self.buf[:] = 0.0


def preprocess(*arrays):
    """兼容通用入口：输入 int16 PCM (N,) 或 (1,N)，返回全部滑窗 (M,1,26,64)。"""
    if len(arrays) != 1:
        raise ValueError('wakeup 需要单个 int16 PCM 数组')
    pcm = np.asarray(arrays[0]).reshape(-1).astype(np.int16)
    n_frames = (pcm.shape[0] - 512) // 512 + 1
    if n_frames < 1:
        raise ValueError('PCM 太短')
    fs = FbankStream()
    wins = []
    for i in range(n_frames):
        wins.append(fs.push(pcm[i * 512 : i * 512 + 512]))
    return [np.stack(wins)]