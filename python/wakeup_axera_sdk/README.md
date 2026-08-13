# wakeup_axera Python SDK

- 输入（与 model_meta.json 一致）: fbank[1, 26, 64]
- 输出（与 model_meta.json 一致）: logits[1, 2, 64]
- 预处理: reference log-fbank: int16/32768 -> hann512(periodic) -> rfft magnitude -> mel(26) -> log(x+1e-7) -> Q6.10；流式维护 64 帧窗口 (1,26,64)
- 后处理: wake score = logits[1,-1]；3 帧滑动求和 vs 阈值（DET_MODE_90=0.608 / DET_MODE_95=0.615）
- 示例输入: models/sample_nihao_aixin.wav（唤醒词“你好，爱芯”，应触发）


```bash
LD_LIBRARY_PATH=/soc/lib PYTHONPATH=$PWD/python python3 wakeup_axera_sdk/example.py           --model models/ax650/model.axmodel --input input.npy --output-dir output
```

> 发布版：端到端 NPU 验证已通过，SDK 仅依赖 pyaxengine（无 onnxruntime/torch/transformers 回退）。
