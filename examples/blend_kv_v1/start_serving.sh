#!/bin/bash

# CacheBlend Serving启动脚本

# 设置LMCache环境变量
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_ENABLE_BLENDING=True
export LMCACHE_BLEND_SPECIAL_STR=" # # "
export LMCACHE_USE_LAYERWISE=True
export LMCACHE_BLEND_CHECK_LAYERS=1
export LMCACHE_BLEND_RECOMPUTE_RATIOS=0.3  # 提高到30%，增强out-of-order blend效果
export LMCACHE_BLEND_MIN_TOKENS=64  # 降低最小blend tokens，让小batch也能blend

# CPU后端配置
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=5

# 如果需要使用Disk后端，注释掉上面两行，取消注释下面三行
# export LMCACHE_LOCAL_CPU=False
# export LMCACHE_LOCAL_DISK=file://./local_disk/
# export LMCACHE_MAX_LOCAL_DISK_SIZE=10

# 模型配置
MODEL=${MODEL:-"mistralai/Mistral-7B-Instruct-v0.2"}
PORT=${PORT:-8000}
GPU_MEM=${GPU_MEM:-0.6}
MAX_LEN=${MAX_LEN:-32000}

echo "=========================================="
echo "启动LMCache CacheBlend Serving"
echo "=========================================="
echo "Model: $MODEL"
echo "Port: $PORT"
echo "GPU Memory Utilization: $GPU_MEM"
echo "Max Model Length: $MAX_LEN"
echo "Blending: Enabled"
echo "Backend: CPU (max ${LMCACHE_MAX_LOCAL_CPU_SIZE}GB)"
echo "Special String: '$LMCACHE_BLEND_SPECIAL_STR'"
echo "=========================================="

# 启动vLLM serving
vllm serve "$MODEL" \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-model-len "$MAX_LEN" \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --enforce-eager
