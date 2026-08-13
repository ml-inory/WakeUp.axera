# 唤醒词扩展指南：自定义唤醒词训练、换词、多唤醒词

本仓库的唤醒词模型（当前唤醒词 **“你好，爱芯”**）基于 WakeNet9 架构，
使用 **edge-tts 合成数据微调**训练而来。每个模型只对应一个唤醒词；
想换词或新增唤醒词，按下面的流程走一遍即可。

## 目录

1. 训练自定义唤醒词（完整流程）
2. 只换唤醒词文本（复用脚本）
3. 多唤醒词并行支持
4. 阈值与触发逻辑

## 1. 训练自定义唤醒词（完整流程）

训练脚本在 `model_convert/train/`，依赖 `python3 + torch + numpy +
soundfile + edge-tts`（CPU 即可训练，模型只有 26 万参数）。

### 1.1 改唤醒词与负样本

编辑 `model_convert/train/make_dataset.py`：

- `POS_TEXTS`：唤醒词文本变体。例如唤醒词是“小爱同学”：
  ```python
  POS_TEXTS = ["小爱同学", "小爱 同学", "小爱同学！"]
  ```
- `NEG_TEXTS`：负样本文本，尽量覆盖容易混淆的词，尤其是
  **“前缀 + 其他核心词”**（如唤醒词“你好，爱芯”时，负样本必须包含
  “你好，小爱 / 你好，其他 / 你好，再见”），否则模型会学成“前缀即触发”。
- `TRAIN_VOICES / VAL_VOICES`：edge-tts 音色列表；验证集音色不参与训练，
  用于衡量泛化。

### 1.2 生成数据

```bash
cd model_convert/train
python make_dataset.py        # 输出 data/dataset.npz（feats + 帧级标签）
```

帧级标签由 `train.py` 加载时自动生成：唤醒词语音的最后一段（核心词）标为
正样本，前缀（如“你好”）与静音标为负样本，负样本全部为 0。

### 1.3 微调

```bash
python train.py               # 输出 checkpoints/best_model.pt + final_model.pt
```

训练从 `ref_model/` 参考权重初始化，全网络解冻微调（AdamW，40 epochs）。
日志输出训练/验证触发率；最后几个 epoch 的文件验收结果在
`logs/training_log.json` 里（正样本触发 / 负样本静默）。

### 1.4 导出 ONNX

```bash
python export.py --checkpoint checkpoints/final_model.pt \
  --check-wav data/accept_nihao_aixin.wav data/accept_nihao_qita.wav
# 输出 onnx/wakeup_nihaoxin.onnx，并打印 torch vs onnx 差异（应 < 1e-5）
```

### 1.5 编译 AXMODEL

```bash
cp onnx/wakeup_nihaoxin.onnx ../model.onnx
cd ..
bash compile_pulsar2.sh AX650    # NPU3（AX650 / AX630）
bash compile_pulsar2.sh AX620E   # NPU2
```

校准数据（`calib_data/input.tar.gz`）用训练集正样本 fbank 窗口生成
（30 个 1×26×64 npy），由 `train.py` 的数据集生成（见 1.2 的脚本逻辑）。

### 1.6 板端验证

```bash
# AX650：编译好的 model_example 直接验证
LD_LIBRARY_PATH=/soc/lib ./cpp/build/model_example models/ax650/model.axmodel \
  正样本.raw 0.615    # 应输出 TRIGGER
LD_LIBRARY_PATH=/soc/lib ./cpp/build/model_example models/ax650/model.axmodel \
  负样本.raw 0.615    # 应 triggers=0
```

## 2. 只换唤醒词文本（复用脚本）

只改 `make_dataset.py` 的 `POS_TEXTS / NEG_TEXTS`，重复 1.2–1.6 即可。
无需改模型结构、SDK 或编译配置。

## 3. 多唤醒词并行支持

每个唤醒词编译一个独立的 `model.axmodel`，SDK 里并行跑多个
`WakeWordDetector` 实例即可。AX650 板端实测单个模型每帧 1.83ms
（帧预算 32ms），并行 5 个模型仍实时：

```python
from wakeup_axera_sdk.detector import WakeWordDetector

dets = [
    WakeWordDetector("models/nihao_aixin/model.axmodel", threshold=0.615),
    WakeWordDetector("models/xiaoai/model.axmodel", threshold=0.620),
]
for det in dets:
    r = det.process(int16_512_samples)
    if r["triggered"]:
        print("唤醒词命中", det)
```

## 4. 阈值与触发逻辑

- 逐帧 wake score = `logits[1, frame]`（ONNX 输出第 2 通道）
- 触发判定：最近 3 帧 wake score 之和 > 阈值
- 阈值：`0.615`（DET_MODE_95）/ `0.608`（DET_MODE_90），按实际误触/漏报
  情况调整（调高更保守、调低更灵敏）

训练/验证时如负样本误触偏高，优先：
1. 扩充 `NEG_TEXTS`（尤其“前缀 + 近音核心词”）
2. 增加音色/语速多样性（`TRAIN_VOICES`、`RATES`）
3. 检查 `train.py` 里的标签切分逻辑（唤醒核心词是否标对）
