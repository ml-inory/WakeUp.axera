# wakeup_axera C++ SDK

直接链接 AX Engine runtime（`ax_engine`/`ax_sys`/`ax_interpreter`）的
流式唤醒词检测 SDK：fbank 预处理 + 64 帧窗口 + NPU 推理 + 3 帧滑动触发。

- 输入（与 model_meta.json 一致）: fbank[1, 26, 64] float32
- 输出（与 model_meta.json 一致）: logits[1, 2, 64] float32
- 支持芯片: AX650/AX630（models/ax650/model.axmodel）、AX620E（models/ax620e/model.axmodel）
- 板端实测（AX650C, NPU3）: “你好，爱芯”触发（max_sum 9.848 > 0.615，
  12 次 TRIGGER）/ “你好，其他”静默（max_sum -5.704），与 Python SDK 一致

## 交叉编译（aarch64）

依赖：`aarch64-none-linux-gnu-gcc/g++`（GCC 9.2）与 AX runtime 头文件/库。
本目录 `axrt/` 已预置来自 AX650C 板的 runtime（include + libax_engine/libax_sys/
libax_interpreter），可直接编译；也可用你自己的 BSP 覆盖：

```bash
cmake -S cpp -B cpp/build-aarch64 \
  -DCMAKE_TOOLCHAIN_FILE=cpp/toolchain-aarch64.cmake \
  -DTOOLCHAIN_ROOT=/path/to/aarch64-none-linux-gnu \
  -DAX_RUNTIME_ROOT=/path/to/axrt        # 默认使用 cpp/axrt
cmake --build cpp/build-aarch64 -j
```

产物：`cpp/build-aarch64/model_example`。

## 板端运行

```bash
./cpp/build-aarch64/model_example models/ax650/model.axmodel 音频.raw 0.615
# 输入为 16kHz 单声道 int16 raw PCM；阈值默认 0.615（DET_MODE_95）
```

运行前在板端设置 `export LD_LIBRARY_PATH=/soc/lib`（或指向 runtime 库目录）。

## 在自己的代码里用

```cpp
#include "wake_word.hpp"
WakeWordDetector det("model.axmodel", 0.615f, 3);
auto r = det.ProcessFrame(int16_512_samples);   // 每帧 512 个 int16
// r.wake_score / r.window_sum / r.triggered
```

`WakeWordDetector` 线程内自持 64 帧窗口与得分缓冲，逐帧喂入即可。
