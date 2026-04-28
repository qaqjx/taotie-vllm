#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
RUNNER="${ROOT_DIR}/exp/request_rate/run_cpu_compress_dedup_experiment.py"

MODEL="${MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
DATASET="${DATASET:-${ROOT_DIR}/exp/data/wikimqa_s.jsonl}"
CUDA_VISIBLE_DEVICES_ARG="${CUDA_VISIBLE_DEVICES_ARG:-1}"
PORT="${PORT:-12345}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
NUM_REQUESTS="${NUM_REQUESTS:-200}"
REQUEST_RATE="${REQUEST_RATE:-2}"
SOURCE_LIMIT="${SOURCE_LIMIT:-200}"
SOURCE_OFFSET="${SOURCE_OFFSET:-0}"
MAX_TOKENS="${MAX_TOKENS:-32}"
TIMEOUT="${TIMEOUT:-1200}"
XJ_NUM_WORKERS="${XJ_NUM_WORKERS:-32}"
XJ_MAX_RSS_GIB="${XJ_MAX_RSS_GIB:-200}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[xj_cpucompress_wikimqa] missing python: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -f "${RUNNER}" ]]; then
  echo "[xj_cpucompress_wikimqa] missing runner: ${RUNNER}" >&2
  exit 1
fi

if [[ ! -f "${DATASET}" ]]; then
  echo "[xj_cpucompress_wikimqa] missing dataset: ${DATASET}" >&2
  exit 1
fi

cmd=(
  "${PYTHON_BIN}"
  "${RUNNER}"
  --model "${MODEL}"
  --dataset "${DATASET}"
  --port "${PORT}"
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES_ARG}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --num-requests "${NUM_REQUESTS}"
  --request-rate "${REQUEST_RATE}"
  --source-limit "${SOURCE_LIMIT}"
  --source-offset "${SOURCE_OFFSET}"
  --max-tokens "${MAX_TOKENS}"
  --timeout "${TIMEOUT}"
  --xj-num-workers "${XJ_NUM_WORKERS}"
  --xj-max-rss-gib "${XJ_MAX_RSS_GIB}"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
  cmd+=(--output-dir "${OUTPUT_DIR}")
fi

echo "[xj_cpucompress_wikimqa] root: ${ROOT_DIR}"
echo "[xj_cpucompress_wikimqa] model: ${MODEL}"
echo "[xj_cpucompress_wikimqa] dataset: ${DATASET}"
echo "[xj_cpucompress_wikimqa] num_requests=${NUM_REQUESTS} request_rate=${REQUEST_RATE} source_limit=${SOURCE_LIMIT} source_offset=${SOURCE_OFFSET}"
echo "[xj_cpucompress_wikimqa] xj_num_workers=${XJ_NUM_WORKERS} xj_max_rss_gib=${XJ_MAX_RSS_GIB}"
echo "[xj_cpucompress_wikimqa] command: ${cmd[*]}"

exec "${cmd[@]}"
