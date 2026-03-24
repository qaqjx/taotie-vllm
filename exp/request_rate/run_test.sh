#!/bin/bash
# S3-only Benchmark test script (no local cache)

cd "$(dirname "$0")"

# Default parameters
MODEL="mistralai/Mistral-7B-Instruct-v0.3"
GPU=0
PORT=12345
NUM_CONTEXTS=10
NUM_REQUESTS=20
COMPRESS_METHODS="NONE,OURS"
REQUEST_RATES="1,2,4"
OUTPUT_PREFIX="s3_only_benchmark"

echo "=========================================="
echo "S3-Only Benchmark Test (No Local Cache)"
echo "=========================================="
echo "Model: $MODEL"
echo "GPU: $GPU"
echo "Port: $PORT"
echo "Compress methods: $COMPRESS_METHODS"
echo "Request rates: $REQUEST_RATES"
echo "Local cache: DISABLED"
echo "=========================================="

python3 -u benchmark_incremental.py \
    --model "$MODEL" \
    --gpu $GPU \
    --port $PORT \
    --num-contexts $NUM_CONTEXTS \
    --num-requests $NUM_REQUESTS \
    --compress-methods "$COMPRESS_METHODS" \
    --request-rates "$REQUEST_RATES" \
    --output-prefix "$OUTPUT_PREFIX" \
    --warmup-delay 100.0 \
    --tp 1 \
    --no-local-cache

echo "=========================================="
echo "Benchmark completed"
echo "=========================================="
