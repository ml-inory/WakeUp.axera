# Run On Board Report

## 板端验证（唤醒词：你好，爱芯）— 2026-08-13

板卡：AX650C（10.126.35.203），模型 `models/ax650/model.axmodel`（NPU3，INT8），
验证器：`cpp/build/model_example`（C++，AX Engine runtime）。

```text
你好，爱芯（正样本）: frames=59 max_score=3.877@1.12s max_sum=9.848 triggers=12
你好，其他（负样本）: frames=57 max_score=-3.921 max_sum=-5.704 triggers=0
```

结论：正样本触发、负样本静默，板端行为与浮点模型一致（详见 `training_report.md`）。

## C++ SDK 板端验证（2026-08-13）

- 交叉编译器：arm-none GCC 9.2（aarch64-none-linux-gnu），AX runtime 取自板端
- 二进制：cpp/build-aarch64/model_example（165KB）
- 正样本（你好，爱芯）：max_sum=9.848 > 0.615，TRIGGER×12，与 Python SDK 结果一致
- 负样本（你好，其他）：max_sum=-5.704，TRIGGERED=False
- 修复项：AX_ENGINE_RunSyncV2（handle+context）；Cached 内存写前 flush /
  读后 invalidate（AX_SYS_MflushCache / AX_SYS_MinvalidateCache）

## 结论

- 正样本板端触发、负样本板端静默，阈值 0.615
- 每帧推理 1.83ms（帧预算 32ms），可实时；AXMODEL INT8 输出与仿真 cosine 0.999+ 一致
