# 唤醒词训练报告（你好，爱芯）

## 结论

唤醒词“你好，爱芯”模型已完成训练、ONNX 导出、Pulsar2 INT8 编译（AX650 NPU3 /
AX620E NPU2），并在 AX650C 板端实测通过：

| 验收音频 | 板端结果（threshold 0.615） |
|----------|------------------------------|
| “你好，爱芯”（正样本） | TRIGGER，max_sum 9.85，12 次触发 |
| “你好，其他”（负样本） | 静默，max_sum -5.70 |

## 训练设置

- 架构：WakeNet9 复刻（conv1+LN+LIGLU×5+残差+conv2+L2norm+conv3，26 万参数）
- 初始化：参考浮点权重（`model_convert/train/ref_model/`），全网络解冻微调
- 数据：edge-tts 合成，17 个训练音色 + 1 个验证音色（YunyangNeural，不参与训练），
  3 种语速；正样本 170 条（唤醒词 8 种文本变体）、负样本 292 条
  （“你好，小爱/其他/再见”等前缀混淆词 + 常见命令短语）
- 标签：帧级二分类；正样本只标唤醒核心“爱芯”语音段（VAD 分段最后一段），
  前缀“你好”与静音标 0
- 损失：wake 通道 BCE-with-logits + margin（对齐 3 帧滑窗阈值尺度）
- 优化：AdamW lr=5e-4，cosine 衰减，40 epochs，难负样本挖掘

## 训练结果（浮点，threshold 0.615）

| 指标 | 结果 |
|------|------|
| 训练集正样本触发 | 152/154 (99%) |
| 训练集负样本误触 | 4/259 (2%) |
| 验收“你好，爱芯” | 触发（sum3 14.40） |
| 验收“你好，其他” | 静默（sum3 -16.27） |
| 未见音色（YunyangNeural）正样本 | 38%（泛化偏弱，真实录音待迭代） |

## 量化与板端

- Pulsar2 7.0 编译，INT8，逐层 cosine ≈ 0.999+
- AX650 NPU3：0.5MB，板端每帧 1.83ms（RTF≈0.057）
- AX620E NPU2：0.33MB，内置仿真检查通过
- Torch vs ONNX max abs diff 6.9e-06

## 已知限制

- 训练数据为 TTS 合成，对真实麦克风录音的泛化需真实数据继续迭代
- 对含“爱芯”核心词的其他前缀短语（如“Hi，爱芯”）也会触发（核心词匹配）
- 负样本含近音词（爱星/艾心/爱信）时误触偏高，当前版本未纳入训练

## 复现

`model_convert/train/`：make_dataset.py（数据）→ train.py（微调）→
export.py（ONNX）→ compile_pulsar2.sh（AXMODEL）。详见
`docs/wake_word_guide.md`。
