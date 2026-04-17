#!/bin/bash
# 一键执行不同 QPS 下的 request throughput benchmark

set -euo pipefail

GPU=${GPU:-0}
PORT=${PORT:-12345}
MODEL=${MODEL:-"mistralai/Mistral-7B-Instruct-v0.3"}
METHODS=${METHODS:-"NONE,KIVI_2BIT,OURS,SVDQ"}
RATES=${RATES:-"0.5,1,2,4,8"}
NUM_CONTEXTS=${NUM_CONTEXTS:-20}
NUM_REQUESTS=${NUM_REQUESTS:-20}
CTX_PER_REQ=${CTX_PER_REQ:-"10,10"}
TARGET_LENGTH=${TARGET_LENGTH:-""}
WARMUP_DELAY=${WARMUP_DELAY:-3}
DATASET=${DATASET:-"/home/xujie/TaoTie/dataset/needle_multi_rag.jsonl"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32000}
OUTPUT_DIR=${OUTPUT_DIR:-"$(dirname "$0")/results"}
LOG_DIR=${LOG_DIR:-"$(dirname "$0")/server_logs"}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-"throughput_summary"}
PYTHON_BIN=${PYTHON_BIN:-$(command -v python3)}
VLLM_EXE=${VLLM_EXE:-$(dirname "$PYTHON_BIN")/vllm}
SKIP_SERVER=${SKIP_SERVER:-0}
RESULT_PAIRS=()

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

kill_server() {
    if [ "${SKIP_SERVER}" = "1" ]; then
        return 0
    fi
    log "Killing existing server processes..."
    if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        unset SERVER_PID
    fi
    pkill -f "vllm.*${PORT}" 2>/dev/null || true
    pkill -f "EngineCore" 2>/dev/null || true
    sleep 2
    pkill -9 -f "vllm.*${PORT}" 2>/dev/null || true
    pkill -9 -f "EngineCore" 2>/dev/null || true
    sleep 3
}

start_server() {
    local method="$1"
    local log_file="$LOG_DIR/vllm_${method}_${PORT}.log"
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
    export LMCACHE_ENABLE_PROFILING=${LMCACHE_ENABLE_PROFILING:-1}

    mkdir -p "$LOG_DIR"

    "$VLLM_EXE" serve "$MODEL" \
        --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
        --port "$PORT" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --max-model-len "$MAX_MODEL_LEN" \
        --enforce-eager \
        -tp 1 \
        > "$log_file" 2>&1 &

    SERVER_PID=$!
    log "Server PID: $SERVER_PID"
    log "Server log: $log_file"
}

wait_server() {
    if [ "${SKIP_SERVER}" = "1" ]; then
        log "Skipping server startup; assuming server is already running."
        return 0
    fi

    log "Waiting for server to be ready..."
    local max_wait=180
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if [ -n "${SERVER_PID:-}" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log "ERROR: Server process exited before becoming ready"
            return 1
        fi
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
    local method="$1"
    local extra_args=()
    if [ -n "$TARGET_LENGTH" ]; then
        extra_args+=(--target-length "$TARGET_LENGTH")
    else
        extra_args+=(--ctx-per-req "$CTX_PER_REQ")
    fi

    mkdir -p "$OUTPUT_DIR"

    "$PYTHON_BIN" "$(dirname "$0")/run_throughput_benchmark.py" \
        --model "$MODEL" \
        --port "$PORT" \
        --dataset "$DATASET" \
        --server-log "$LOG_DIR/vllm_${method}_${PORT}.log" \
        --rates "$RATES" \
        --num-contexts "$NUM_CONTEXTS" \
        --num-requests "$NUM_REQUESTS" \
        --warmup-delay "$WARMUP_DELAY" \
        --output-dir "$OUTPUT_DIR" \
        --output-prefix "${OUTPUT_PREFIX}_${method}" \
        "${extra_args[@]}"
}

record_latest_result() {
    local method="$1"
    local latest_csv
    latest_csv=$(ls -t "$OUTPUT_DIR/${OUTPUT_PREFIX}_${method}"_*.csv 2>/dev/null | head -n 1 || true)
    if [ -n "$latest_csv" ]; then
        RESULT_PAIRS+=("${method}=${latest_csv}")
        log "Recorded result for ${method}: ${latest_csv}"
    else
        log "WARNING: Could not find result CSV for ${method}"
    fi
}

aggregate_results() {
    if [ "${#RESULT_PAIRS[@]}" -eq 0 ]; then
        log "No successful method results to aggregate."
        return 0
    fi

    local aggregate_args=()
    local pair
    for pair in "${RESULT_PAIRS[@]}"; do
        aggregate_args+=(--result "$pair")
    done

    "$PYTHON_BIN" "$(dirname "$0")/aggregate_throughput_results.py" \
        --output-dir "$OUTPUT_DIR" \
        --output-prefix "${OUTPUT_PREFIX}_combined" \
        "${aggregate_args[@]}"
}

cleanup() {
    if [ "${SKIP_SERVER}" != "1" ]; then
        kill_server
    fi
}

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --gpu GPU_ID                GPU to use (default: 0)"
    echo "  --port PORT                 Server port (default: 12345)"
    echo "  --model MODEL               Model name"
    echo "  --methods 'M1,M2'           Compression methods (default: NONE,KIVI_2BIT,OURS,SVDQ)"
    echo "  --method METHOD             Single compression method shorthand"
    echo "  --rates '1,2,4,8'           Target QPS values"
    echo "  --num-contexts N            Number of contexts for warmup"
    echo "  --num-requests N            Requests per QPS point"
    echo "  --ctx-per-req 'a,b'         Contexts per request range"
    echo "  --target-length TOKENS      Fixed target request length"
    echo "  --warmup-delay SECONDS      Delay after warmup"
    echo "  --dataset PATH              Dataset path"
    echo "  --output-dir PATH           Output directory"
    echo "  --output-prefix PREFIX      Output filename prefix"
    echo "  --skip-server               Reuse an already running server"
    echo "  -h, --help                  Show this help"
}

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
        --model)
            MODEL="$2"
            shift 2
            ;;
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --method)
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
        --target-length)
            TARGET_LENGTH="$2"
            shift 2
            ;;
        --warmup-delay)
            WARMUP_DELAY="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --output-prefix)
            OUTPUT_PREFIX="$2"
            shift 2
            ;;
        --skip-server)
            SKIP_SERVER=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

trap cleanup EXIT

log "=============================================="
log "Request Throughput Benchmark"
log "=============================================="
log "GPU: $GPU"
log "Port: $PORT"
log "Python: $PYTHON_BIN"
log "vLLM: $VLLM_EXE"
log "Model: $MODEL"
log "Methods: $METHODS"
log "Rates: $RATES"
log "Contexts: $NUM_CONTEXTS"
log "Requests per rate: $NUM_REQUESTS"
log "Contexts per request: $CTX_PER_REQ"
if [ -n "$TARGET_LENGTH" ]; then
    log "Target length: $TARGET_LENGTH"
fi
log "=============================================="

if [ "${SKIP_SERVER}" = "1" ]; then
    method_count=$(printf '%s' "$METHODS" | awk -F',' '{print NF}')
    if [ "$method_count" -gt 1 ]; then
        echo "ERROR: --skip-server only supports a single method in --methods/--method." >&2
        exit 1
    fi
fi

for method in ${METHODS//,/ }; do
    log ""
    log "=============================================="
    log "Testing method: $method"
    log "=============================================="

    if [ "${SKIP_SERVER}" != "1" ]; then
        kill_server
        start_server "$method"
    fi

    if ! wait_server; then
        log "Skipping $method due to server startup failure"
        continue
    fi

    if ! run_benchmark "$method"; then
        log "Skipping $method due to benchmark failure"
        continue
    fi

    record_latest_result "$method"
done

aggregate_results

log "Benchmark completed. Results saved to: $OUTPUT_DIR"
