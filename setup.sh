#!/usr/bin/env bash
set -euo pipefail

echo "=== 安装基础依赖（numpy + soundfile）==="
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
pip install --user -i "$PIP_INDEX_URL" -r python/requirements.txt

echo ""
echo "=== 安装 pyaxengine（AX 芯片推理引擎，仅 AX 板需要）==="
if [ "$(uname -m)" = "aarch64" ]; then
  pip install --user -i "$PIP_INDEX_URL" "pyaxengine @ git+https://gh-proxy.com/https://github.com/AXERA-TECH/pyaxengine.git"
  echo "✅ pyaxengine 安装成功"
else
  echo "当前主机非 aarch64（$(uname -m)），跳过 pyaxengine 安装。"
  echo "本交付包为 NPU 专用，请在 AX650/AX630/AX620E 板端（aarch64）运行。"
fi

echo ""
echo "C++ SDK: 请先安装 AX650 BSP SDK，然后："
echo "  export AX_RUNTIME_ROOT=/path/to/axruntime"
echo "  mkdir -p cpp/build && cd cpp/build"
echo "  cmake .. -DCMAKE_TOOLCHAIN_FILE=\${AX_RUNTIME_ROOT}/toolchain.cmake"
echo "  make -j\$(nproc)"

echo ""
echo "✅ 环境准备完成"
