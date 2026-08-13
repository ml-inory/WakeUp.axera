# Compile Report

- image: pulsar2:7.0
- target: AX650（NPU3）
- input: fbank:1x26x64
- src_dtype: FP32
- size: 528.6 KB（models/ax650/model.axmodel，板端已验证 1.83ms/帧）

## AX620E（NPU2）

- target: AX620E / NPU2（pulsar2_config_ax620e.json）
- size: 331.6 KB（models/ax620e/model.axmodel）
- 编译内置仿真检查通过：`check npu graph [logits] (1, 2, 64) float32 successfully`
- 说明：无 AX620E 板端实测（未接板），NPU 数值一致性由 Pulsar2 仿真保证
