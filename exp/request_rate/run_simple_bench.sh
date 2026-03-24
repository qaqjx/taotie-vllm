#!/bin/bash
# 启动服务器并运行 TTFT benchmark
# 用法: ./run_simple_bench.sh <COMPRESS_TYPE> [GPU]
# 例如: ./run_simple_bench.sh KIVI_2BIT 1
#       ./run_simple_bench.sh OURS 1

set -e
cd "$(dirname "$0")"

COMPRESS_TYPE=${1:-"KIVI_2BIT"}
GPU=${2:-1}
PORT=12345
MODEL="mistralai/Mistral-7B-Instruct-v0.3"

echo "=========================================="
echo "TTFT Benchmark: $COMPRESS_TYPE"
echo "=========================================="

# 杀掉旧进程
echo "Killing old processes..."
pkill -9 -f "vllm" 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true
sleep 3

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU
export HF_HUB_OFFLINE=1
export LMCACHE_COMPRESS_TYPE=$COMPRESS_TYPE
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_ENABLE_BLENDING=True
export LMCACHE_BLEND_SPECIAL_STR=" # # "
export LMCACHE_USE_LAYERWISE=True
export LMCACHE_BLEND_CHECK_LAYERS=1
export LMCACHE_BLEND_RECOMPUTE_RATIOS=0.15
export LMCACHE_BLEND_MIN_TOKENS=64
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=5

echo "Starting vLLM server with $COMPRESS_TYPE..."
vllm serve $MODEL \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
    --port $PORT \
    --gpu-memory-utilization 0.6 \
    --max-model-len 32000 \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --enforce-eager \
    -tp 1 &

SERVER_PID=$!

# 等待服务器启动
echo "Waiting for server to be ready..."
for i in {1..120}; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    sleep 2
done

# 检查服务器是否真的启动了
if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "Server failed to start!"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# 运行 benchmark
echo ""
echo "Running benchmark..."
python3 -u simple_bench.py \
    --model $MODEL \
    --port $PORT \
    --num-contexts 10 \
    --num-requests 20 \
    --contexts-per-request 3 \
    --rates "1,2,4,8"

# 清理
echo ""
echo "Cleaning up..."
kill $SERVER_PID 2>/dev/null || true
pkill -9 -f "vllm" 2>/dev/null || true

echo "Done!"
