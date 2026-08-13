# Model Convert（可复现编译）

本目录是「你好，爱芯」唤醒词模型从 ONNX 到 AXMODEL 的可复现编译流程。
产物：`output/model.axmodel`（INT8，约 0.5MB）。

## 环境准备

- 安装 Docker
- 拉取 Pulsar2 7.0 镜像（原始编译所用镜像）：

```bash
docker pull pulsar2:7.0
```

## 一键编译

在本目录执行：

```bash
bash compile_pulsar2.sh           # 默认 AX650（NPU3），产物 output/model.axmodel
bash compile_pulsar2.sh AX620E    # AX620E（NPU2），产物 output_ax620e/model.axmodel
```

成功后产物分别在 `output/model.axmodel` / `output_ax620e/model.axmodel`，
已预编译版本放在上层 `models/`（AX650，NPU3）与 `models/ax620e/`（AX620E NPU2）。
AX620E 编译日志含内置仿真检查（`check npu graph [logits] ... successfully`）。

`model.onnx` 为自包含单文件（权重内嵌，不依赖外部 .data），可直接编译。

## 模型结构说明

- 输入：`fbank`，float32，shape `(1, 26, 64)`（26 mel 通道 × 64 帧窗口）
- 输出：`logits`，float32，shape `(1, 2, 64)`；唤醒得分 = `logits[1, -1]`
- 预处理：reference 同款 log-fbank——int16/32768 → 512 点周期 Hann 窗 →
  rfft 幅度 → mel 滤波器组（80–7000Hz，26 通道，无归一化，跳过 bin0）→
  log(x + 1e-7) → Q6.10 量化
- 后处理：3 帧滑动求和，> 0.615（DET_MODE_95）触发
- 网络：conv1 → LayerNorm → LIGLU×5 → 残差 → conv2 → conv3（浮点等价重建，
  conv3 带 ×4 经验修正，见 `export_report.md`）

## 模型权重与结构

- 网络结构：conv1 → LayerNorm → LIGLU×5 → 残差 → conv2 → conv3
  （定义见 `wakeup.py` / `train/model_code.py`）
- 权重：`train/final_model.pt`（训练产出）→ `train/export.py` 导出为
  `train/wakeup_nihaoxin.onnx` → 即本目录 `model.onnx`
- 训练：edge-tts 数据 + 微调，脚本见 `train/`（`make_dataset.py` /
  `train.py` / `export.py`），从零复现流程见 `docs/wake_word_guide.md`

## 重新训练 / 导出 ONNX

自定义唤醒词训练（改 `POS_TEXTS` → 生成数据 → 微调 → 导出 → 编译）见
`docs/wake_word_guide.md` 第 1 节，脚本在 `train/`：

```bash
cd train
python make_dataset.py                 # 生成 data/dataset.npz
python train.py                        # 微调 → checkpoints/final_model.pt
python export.py --check-wav data/accept_nihao_aixin.wav data/accept_nihao_qita.wav
# 输出 onnx/wakeup_nihaoxin.onnx（torch/onnx 差异 < 1e-5）
```

当前交付模型由训练流程生成（`train/`）；`wakeup.py` / `model_loader.py`
为网络结构与参考模型解析代码，训练脚本复用其中的结构定义。

## 产物检查

编译完成后检查：

```bash
ls -la output/model.axmodel
python -c "import numpy as np; a=np.fromfile('output/model.axmodel', dtype=np.uint8); print('size KB:', round(a.nbytes/1024,1))"
```

板端运行示例见上层 README（`python/demo.py` 或
`python/wakeup_axera_sdk/example.py --model models/ax650/model.axmodel --input 音频.wav`）。
