#!/bin/bash
# 统一 benchmark 脚本：测试不同压缩方法在不同 request rate 下的 TTFT

set -e

# ============== 配置参数 ==============
GPU=${GPU:-0}
PORT=${PORT:-12345}
MODEL="Qwen/Qwen2.5-14B-Instruct"
METHODS=${METHODS:-"NONE,KIVI_2BIT,OURS,SVDQ"}
RATES=${RATES:-"1,2,4,8,16"}
NUM_CONTEXTS=${NUM_CONTEXTS:-10}
NUM_REQUESTS=${NUM_REQUESTS:-20}
CTX_PER_REQ=${CTX_PER_REQ:-"2,3"}
TARGET_LENGTH=${TARGET_LENGTH:-""}
WARMUP_DELAY=${WARMUP_DELAY:-20}
ROUNDS=${ROUNDS:-1}
MATCH_RATE=${MATCH_RATE:-""}
OUTPUT_DIR="$(dirname $0)/results"
LOG_DIR="$(dirname $0)/server_logs"

# ============== 辅助函数 ==============
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

kill_server() {
    log "Killing existing server processes..."
    pkill -9 -f "vllm.*${PORT}" 2>/dev/null || true
    pkill -9 -f "EngineCore" 2>/dev/null || true
    sleep 3
}
start_server() {
    local method=$1
    log "Starting server with COMPRESS_TYPE=${method}..."

    export CUDA_VISIBLE_DEVICES=$GPU
    export HF_HUB_OFFLINE=1
    export LMCACHE_COMPRESS_TYPE=$method
    export LMCACHE_CHUNK_SIZE=256
    export LMCACHE_ENABLE_BLENDING=True
    export LMCACHE_BLEND_SPECIAL_STR=" # # "
    export LMCACHE_USE_LAYERWISE=True
    export LMCACHE_BLEND_CHECK_LAYERS=1
    export LMCACHE_BLEND_RECOMPUTE_RATIOS=0.15
    export LMCACHE_BLEND_MIN_TOKENS=64
    export LMCACHE_LOCAL_CPU=True
    export LMCACHE_MAX_LOCAL_CPU_SIZE=5

    mkdir -p "$LOG_DIR"

    vllm serve "$MODEL" \
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
        --port $PORT \
        --gpu-memory-utilization 0.8 \
        --max-model-len 32000 \
        --no-enable-prefix-caching \
        --no-enable-chunked-prefill \
        --enforce-eager \
        -tp 1 \
        > "$LOG_DIR/vllm_${method}_${PORT}.log" 2>&1 &

    SERVER_PID=$!
    log "Server PID: $SERVER_PID"
}

wait_server() {
    log "Waiting for server to be ready..."
    local max_wait=180
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if curl -s --max-time 2 "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            log "Server is ready! (waited ${waited}s)"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done
    log "ERROR: Server failed to start within ${max_wait}s"
    return 1
}

run_benchmark() {
    local method=$1
    log "Running benchmark for ${method}..."

    local extra_args=""
    if [ -n "$TARGET_LENGTH" ]; then
        extra_args="--target-length $TARGET_LENGTH"
    else
        extra_args="--ctx-per-req $CTX_PER_REQ"
    fi

    local match_rate_arg=""
    if [ -n "$MATCH_RATE" ]; then
        match_rate_arg="--match-rate"
    fi

    python3 "$(dirname $0)/bench_reuse.py" \
        --model "$MODEL" \
        --port $PORT \
        --methods "$method" \
        --rates "$RATES" \
        --num-contexts $NUM_CONTEXTS \
        --num-requests $NUM_REQUESTS \
        $extra_args \
        --warmup-delay $WARMUP_DELAY \
        --rounds $ROUNDS \
        $match_rate_arg \
        2>&1 | tee "$OUTPUT_DIR/${method}_benchmark.log"
}

# ============== 主程序 ==============
main() {
    log "=============================================="
    log "TTFT Benchmark: Multiple Compression Methods"
    log "=============================================="
    log "GPU: $GPU"
    log "Port: $PORT"
    log "Methods: $METHODS"
    log "Rates: $RATES"
    log "Contexts: $NUM_CONTEXTS"
    log "Requests per rate: $NUM_REQUESTS"
    log "Contexts per request: $CTX_PER_REQ"
    log "=============================================="

    mkdir -p "$OUTPUT_DIR"

    # 结果汇总文件
    SUMMARY_FILE="$OUTPUT_DIR/summary_$(date '+%Y%m%d_%H%M%S').csv"
    echo "method,rate,ttft_avg_ms,ttft_p50_ms,ttft_p90_ms,ttft_p99_ms,success_rate,timestamp" > "$SUMMARY_FILE"

    for method in ${METHODS//,/ }; do
        log ""
        log "=============================================="
        log "Testing: $method"
        log "=============================================="

        # 1. 杀掉旧服务器
        kill_server

        # 2. 启动新服务器
        start_server "$method"

        # 3. 等待服务器就绪
        if ! wait_server; then
            log "Skipping $method due to server startup failure"
            continue
        fi

        # 4. 运行 benchmark
        run_benchmark "$method"

        log "Completed testing: $method"
    done

    # 最终清理
    kill_server

    log ""
    log "=============================================="
    log "All benchmarks completed!"
    log "Results saved to: $OUTPUT_DIR"
    log "=============================================="
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --rates)
            RATES="$2"
            shift 2
            ;;
        --num-contexts)
            NUM_CONTEXTS="$2"
            shift 2
            ;;
        --num-requests)
            NUM_REQUESTS="$2"
            shift 2
            ;;
        --ctx-per-req)
            CTX_PER_REQ="$2"
            shift 2
            ;;
        --rounds)
            ROUNDS="$2"
            shift 2
            ;;
        --match-rate)
            MATCH_RATE="1"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --gpu GPU_ID         GPU to use (default: 1)"
            echo "  --port PORT          Server port (default: 12345)"
            echo "  --methods 'M1 M2'    Methods to test (default: 'OURS KIVI_2BIT NONE')"
            echo "  --rates '0.5,1,2,4,'    Request rates to test (default: '0.5,1,2,4')"
            echo "  --num-contexts N     Number of contexts for warmup (default: 10)"
            echo "  --num-requests N     Requests per rate (default: 10)"
            echo "  --ctx-per-req 'a,b'  Contexts per request range (default: '2,3')"
            echo "  --rounds N           Rounds per rate, take minimum (default: 1)"
            echo "  --match-rate         Set num_requests = rate (rate=4 sends 4 requests in 1s)"
            echo "  -h, --help           Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

main
