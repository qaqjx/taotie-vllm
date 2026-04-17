#!/bin/bash
# 兼容旧名字：转发到 run_full_methods_benchmark.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
exec "$SCRIPT_DIR/run_full_methods_benchmark.sh" "$@"
