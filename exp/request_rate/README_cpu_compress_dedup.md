# CPU Compress Dedup Experiment

This experiment runner measures **xj CPU compression queue memory** while
`LMCACHE_USE_XJ_PROJECT=1` is enabled and the xj CPU-compress path uses a
**local dedup hash table** in `CPUMemoryStore`.

## What it measures

The runner records and exports:

- `remote_queue_bytes`: raw KV bytes waiting in the xj CPU-compress pipeline
- `remote_pending_count`: number of pending remote-upload items
- `rss_bytes`: process RSS sampled by the xj queue logger
- request-side latency/TTFT from the benchmark client

It writes:

- `client_results.json`
- `xj_queue.jsonl`
- `xj_cpu_memory_backlog.png`
- `xj_cpu_memory_backlog.csv`
- `summary.json`

## Dedup semantics

The dedup table lives inside `xj_project/src/cpu_memory_store.cu`.

- Once a `path` has entered the xj remote-upload / CPU-compress path,
  subsequent identical `path` submissions are dropped **before**
  GPU→CPU copy, local CPU store, compression, and queue insertion.
- This is useful for repeat-dataset experiments where the same chunk keys
  are submitted again and again.

## Typical usage

Run from the repo root:

```bash
cd /home/xujie/catkv-vllm
.venv/bin/python exp/request_rate/run_cpu_compress_dedup_experiment.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --dataset exp/data/wikimqa_s.jsonl \
  --cuda-visible-devices 1 \
  --num-requests 120 \
  --request-rate 4 \
  --source-limit 20
```

Notes:

- `--source-limit 20` + repeat mode makes the runner cycle over the first 20
  dataset rows, which is the easiest way to surface dedup behavior.
- The runner always uses:
  - `LMCACHE_USE_XJ_PROJECT=1`
  - `LMCACHE_XJ_STORE=1`
  - `LMCACHE_XJ_PREFETCH=1`
  - `LMCACHE_COMPRESS_TYPE=OURS`

## Outputs

By default, outputs are written under:

```text
exp/request_rate/results/xj_cpucompress_dedup_<timestamp>/
```

The runner prints the final output directory path on success.

## Interpreting the plot

- Rising `remote_queue_bytes` means xj CPU compression is falling behind request ingress.
- Falling `remote_queue_bytes` means workers are draining the queue.
- `max_pending_count` is often easier to compare across runs than raw bytes.
- If dedup is effective, repeated workloads should show less queue growth than
  a non-dedup baseline because duplicate keys never enter the CPU-compress path.
