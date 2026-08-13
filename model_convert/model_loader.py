"""Parse reference model files (*_index + *_data) into structure config and dequantized tensors.

Format (verified against wakeup/mn7/nsnet2/vadnet1):
- index:  [optional 24B header: b"cJSON\\0" + 14B reserved + u32 json_len]
          repeated entries { char name[20]; u32 size_bytes }
- data:   [optional json header of json_len bytes (only when magic present)]
          repeated tensors, each { int32 w; int32 h; int32 stride; int32 flags; int32 exponent;
                                    w*h values as int8 | int16 | float32 }
- The json header (wakeup) or a tensor named "cJSON" (mn7) holds the network structure config.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Tensor:
    name: str
    w: int
    h: int
    stride: int
    flags: int
    exponent: int
    data: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return (self.w, self.h)

    def real(self) -> np.ndarray:
        """Dequantized float64/float32 values = data * 2**exponent."""
        return self.data.astype(np.float64) * (2.0**self.exponent)


@dataclass
class ReferenceModel:
    name: str
    info: dict
    config: dict
    tensors: dict[str, Tensor] = field(default_factory=dict)

    def t(self, name: str) -> Tensor:
        return self.tensors[name]

    def weights(self, name: str) -> np.ndarray:
        """Weight matrix as (out, in) float array (w=out, h=in)."""
        t = self.tensors[name]
        return t.real().reshape(t.w, t.h)

    def bias(self, name: str) -> np.ndarray:
        t = self.tensors[name]
        assert t.h == 1, f"{name} is not a bias (h={t.h})"
        return t.real().reshape(-1)


def _parse_info(text: str) -> dict:
    """Parse _MODEL_INFO_ lines like 'wakeup_v1h24_Hi,AXera_3_0.63_0.635'."""
    info: dict = {"raw": text}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p for p in line.replace("_", " ").split()]
        info["raw"] = line
        # arch identifiers: "wakeup"/"wakenet" etc. (compat with legacy model files)
        if len(parts) >= 2 and parts[0].lower().startswith(("wakeup", "wakenet", "mn", "nsnet", "vadnet")):
            info["arch"] = parts[0]
            info["version"] = parts[1] if len(parts) > 1 else ""
        num = []
        for p in parts:
            try:
                f = float(p)
                num.append(f)
            except ValueError:
                pass
        if num:
            info["nums"] = num
    return info


def _read_index(path: Path) -> tuple[int, list[tuple[str, int]]]:
    d = path.read_bytes()
    json_len = 0
    start = 0
    if d[:6] == b"cJSON\x00":
        json_len = struct.unpack_from("<I", d, 20)[0]
        start = 24
    ents: list[tuple[str, int]] = []
    off = start
    while off + 24 <= len(d):
        name = d[off : off + 20].split(b"\0")[0].decode(errors="replace")
        size = struct.unpack_from("<I", d, off + 20)[0]
        ents.append((name, size))
        off += 24
    return json_len, ents


def load_model(model_dir: str | Path) -> ReferenceModel:
    model_dir = Path(model_dir)
    index_path = next(model_dir.glob("*_index"))
    data_path = index_path.with_name(index_path.name.replace("_index", "_data"))
    info_path = model_dir / "_MODEL_INFO_"
    json_len, ents = _read_index(index_path)
    data = data_path.read_bytes()

    info = _parse_info(info_path.read_text(errors="replace")) if info_path.exists() else {"raw": ""}

    config: dict = {}
    if json_len:
        try:
            config = json.loads(data[:json_len])
        except json.JSONDecodeError:
            config = {}
    tensors: dict[str, Tensor] = {}
    off = json_len
    for name, size in ents:
        if name == "cJSON":
            raw = data[off : off + size].rstrip(b"\x00")
            try:
                config = json.loads(raw)
            except json.JSONDecodeError:
                config = {}
        elif off + 20 <= len(data):
            w, h, stride, flags, exponent = struct.unpack_from("<5i", data, off)
            n = w * h
            payload = data[off + 20 : off + size]
            if len(payload) == n * 4:
                arr = np.frombuffer(payload, dtype="<f4").copy()
            elif len(payload) == n * 2:
                arr = np.frombuffer(payload, dtype="<i2").copy().astype(np.float64)
            elif len(payload) == n:
                arr = np.frombuffer(payload, dtype="<i1").copy().astype(np.float64)
            else:
                raise ValueError(f"tensor {name}: size {size} != header w*h ({n}) for i8/i16/f32")
            tensors[name] = Tensor(name, w, h, stride, flags, exponent, arr)
        off += size

    name = model_dir.name
    return ReferenceModel(name=name, info=info, config=config, tensors=tensors)


def summary(model: ReferenceModel) -> str:
    lines = [f"== {model.name}  info={model.info.get('raw', '')[:80]}"]
    if model.config:
        lines.append(f"   config: {json.dumps(model.config)[:200]}")
    for name, t in model.tensors.items():
        dt = {1: "i8", 2: "i16", 4: "f32"}[t.data.dtype.itemsize if t.data.dtype != np.float64 else (2 if t.data.dtype == np.int16 else 4)]
        lines.append(f"   {name:22s} {t.w:6d} x {t.h:<6d} exp={t.exponent:5d} {dt}")
    return "\n".join(lines)
