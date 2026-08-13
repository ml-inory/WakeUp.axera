# WakeUp.axera「你好，爱芯」唤醒词模型 AXMODEL（AX650 / AX620E）

基于 WakeNet9 架构自研训练的唤醒词模型（唤醒词 **“你好，爱芯”**），在
AX650/AX630（NPU3）与 AX620E（NPU2）芯片上预编译为 INT8 AXMODEL。
输入 16kHz 单声道 int16 PCM，每 32ms 一帧，输出逐帧唤醒得分与触发事件
（3 帧滑动和 > 阈值）。

模型使用 edge-tts 合成数据微调训练（`model_convert/train/` 提供完整可复现脚本），
并在 AX650 板端实测验证：**“你好，爱芯”触发、“你好，其他”静默**。

精度：AXMODEL 量化 cosine ≈ 0.999+ ｜ AX650 板端实测每帧推理 1.83ms
（帧预算 32ms，RTF≈0.057，可实时）｜ AX650 模型 0.5MB / AX620E 模型 0.33MB（INT8）

## 支持的芯片与模型

| 芯片 | 模型文件 | 说明 |
|------|----------|------|
| AX650 / AX630 | `models/ax650/model.axmodel` | NPU3，板端实测 1.83ms/帧，已验证 |
| AX620E | `models/ax620e/model.axmodel` | NPU2，Pulsar2 编译 + 内置仿真检查通过 |

## 快速开始（只需两步）

### 1. 安装环境

```bash
bash setup.sh
```

### 2. 跑推理

```bash
bash run.sh
```

运行后会看到 `models/sample_nihao_aixin.wav`（唤醒词正样本，应触发）和
`models/sample_nihao_qita.wav`（负样本，应静默）的检测结果。

## 在板子上跑

本包为 NPU 专用（无 CPU 回退），需要 AX650/AX630/AX620E 板子 + pyaxengine。
板端执行：

```bash
bash setup.sh
bash run.sh
```

## 怎么用自己的音频

```bash
# AX650 / AX630
python python/wakeup_axera_sdk/example.py --model models/ax650/model.axmodel \
  --input 你的音频.wav --threshold 0.615

# AX620E（NPU2）
python python/wakeup_axera_sdk/example.py --model models/ax620e/model.axmodel \
  --input 你的音频.wav --threshold 0.615
```

音频要求：16kHz、单声道、int16 WAV（PCM）。阈值默认 0.615（DET_MODE_95），
更灵敏可改用 0.608（DET_MODE_90）。

## 在自己的代码里用

```python
from wakeup_axera_sdk.detector import WakeWordDetector

det = WakeWordDetector("models/ax650/model.axmodel", threshold=0.615)
result = det.process(int16_512_samples)   # 32ms 一帧
print(result["wake_score"], result["window_sum"], result["triggered"])
```

输入为 512 个 int16 采样（32ms @ 16kHz），内部做 reference 同款 log-fbank
（26 mel 通道 × 64 帧窗口，Q6.10 量化）后交给 NPU 推理。

## 目录说明

| 目录 | 用途 |
|------|------|
| `models/` | ax650/（AX650 NPU3 模型）+ ax620e/（AX620E NPU2 模型）+ 示例音频 |
| `python/` | Python SDK（pyaxengine，NPU 专用）+ demo.py |
| `cpp/` | C++ SDK（CMake + AX Engine runtime 直接链接，板端实测通过，含交叉编译说明）|
| `model_convert/` | 可复现的导出 & 编译目录（含 ONNX、校准数据、Pulsar2 配置）|
| `model_convert/train/` | 唤醒词训练脚本（edge-tts 数据生成 + 微调 + 导出）|
| `docs/` | 唤醒词扩展指南（换词 / 自定义唤醒词 / 多唤醒词支持）|
| `reports/` | 导出/编译/仿真/板端验证报告 |
| `setup.sh` | 一键安装依赖 |
| `run.sh` | 一键运行检测 |

## 常见问题

**Q: import 报错找不到 pyaxengine？**
A: 在板端运行 `bash setup.sh`；本包为 NPU 专用，不支持 x86 CPU 回退。

**Q: 怎么训练/换其他唤醒词？**
A: 训练脚本在 `model_convert/train/`：改 `make_dataset.py` 里的
`POS_TEXTS`（唤醒词变体）与 `NEG_TEXTS`（负样本），跑 `make_dataset.py`
生成数据、`train.py` 微调、`export.py` 导出 ONNX，再按
`compile_pulsar2.sh` 编译 AXMODEL。完整流程见 `docs/wake_word_guide.md`。

**Q: 想支持多个唤醒词或自定义唤醒词？**
A: 每词编译一个 axmodel 并行推理即可（板端实测每帧 1.83ms，5 个模型并行仍实时）；
自定义唤醒词由训练后按同样流程部署。详见 `docs/wake_word_guide.md`。

**Q: 想自己重新编译 AXMODEL？**
A: 进入 `model_convert/`，执行 `bash compile_pulsar2.sh AX650`（默认）或
   `bash compile_pulsar2.sh AX620E`（Pulsar2 7.0，Docker，配置见 pulsar2_config*.json）。
