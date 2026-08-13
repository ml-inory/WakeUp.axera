#!/usr/bin/env bash
set -euo pipefail

echo "=== 运行 WakeUp.axera 唤醒词检测 ==="
if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
"$PY" python/demo.py
# C++ 版（需先交叉编译）：./cpp/build/model_example models/ax650/model.axmodel
