#!/usr/bin/env bash
set -euo pipefail
# 用 Pulsar2 7.0 Docker 编译：model.onnx + calib_data -> output*/model.axmodel
# 用法：bash compile_pulsar2.sh [AX650|AX620E]   （默认 AX650，NPU1）
TARGET="${1:-AX650}"
case "$TARGET" in
  AX650)  CONFIG=pulsar2_config.json ;;
  AX620E) CONFIG=pulsar2_config_ax620e.json ;;
  *) echo "unknown target: $TARGET (支持 AX650 / AX620E)" >&2; exit 1 ;;
esac
echo "=== 编译目标：$TARGET（配置 $CONFIG）==="
docker run --rm -v "$(pwd)":/workspace pulsar2:7.0 \
  pulsar2 build --config "/workspace/$CONFIG"
