#!/bin/bash
# 一键运行全方法 throughput 测试，并在结束后自动汇总结果

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

# METHODS=${METHODS:-"NONE,KIVI_2BIT,OURS,SVDQ"}
METHODS=${METHODS:-"KIVI_2BIT"}

RATES=${RATES:-"0.5,1,2,4,8"}
NUM_CONTEXTS=${NUM_CONTEXTS:-10}
NUM_REQUESTS=${NUM_REQUESTS:-30}
CTX_PER_REQ=${CTX_PER_REQ:-"10,10"}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-"full_methods"}

"$SCRIPT_DIR/run_throughput_benchmark.sh" \
  --methods "$METHODS" \
  --rates "$RATES" \
  --num-contexts "$NUM_CONTEXTS" \
  --num-requests "$NUM_REQUESTS" \
  --ctx-per-req "$CTX_PER_REQ" \
  --output-prefix "$OUTPUT_PREFIX" \
  "$@"
