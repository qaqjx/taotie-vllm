# Throughput Benchmarks

This subdirectory contains throughput-oriented benchmark helpers for measuring achieved request throughput (`req/s`) under different target QPS loads.

## Start Here

Recommended entrypoint:

- `./run_full_methods_benchmark.sh`: one-click full-method benchmark with automatic aggregation
- `./run_throughput_benchmark.sh`: one-click flow that starts the server and runs the throughput sweep
- `python3 run_throughput_benchmark.py`: sweep one or more target QPS values and report achieved throughput

## What It Measures

For each configured target QPS, the benchmark reports:

- attempted requests
- successful completed requests
- elapsed wall-clock seconds
- achieved throughput in `req/s`
- success rate

The workload intentionally reuses the same dataset loading, context warmup, separator-token convention, and prompt construction style as `exp/request_rate/bench_reuse.py` so the results stay comparable to the existing request-rate experiments.

## Typical Usage

```bash
cd exp/throughput
./run_full_methods_benchmark.sh
```

This wrapper defaults to:

- methods: `NONE,KIVI_2BIT,OURS,SVDQ`
- rates: `0.5,1,2,4,8`
- contexts: `10`
- requests per rate: `6`

Customized one-click run:

```bash
./run_full_methods_benchmark.sh \
  --rates "0.5,1,2,4,8,12" \
  --num-requests 10
```

General benchmark entrypoint:

```bash
cd exp/throughput
./run_throughput_benchmark.sh \
  --methods "OURS,KIVI_2BIT,NONE" \
  --rates "1,2,4,8" \
  --num-contexts 20 \
  --num-requests 20 \
  --ctx-per-req "2,3"
```

If the server is already running, use:

```bash
./run_throughput_benchmark.sh \
  --skip-server \
  --method OURS \
  --rates "1,2,4,8"
```

Direct Python entrypoint:

```bash
python3 run_throughput_benchmark.py \
  --rates "1,2,4,8" \
  --num-contexts 20 \
  --num-requests 20 \
  --ctx-per-req "2,3"
```

Example with a target request length:

```bash
python3 run_throughput_benchmark.py \
  --rates "2,4,8,12" \
  --num-contexts 30 \
  --num-requests 24 \
  --target-length 16000
```

## Outputs

By default, results are written under `exp/throughput/results/`:

- `throughput_summary_<timestamp>.csv`: one row per target QPS
- `throughput_summary_<timestamp>.json`: benchmark metadata plus per-rate summaries and request records
- `throughput_summary_combined_<timestamp>.csv`: combined long-format results across methods
- `throughput_summary_combined_<timestamp>_wide.csv`: single wide CSV with all methods on one row per QPS
- `throughput_summary_combined_<timestamp>.md`: combined markdown comparison tables across methods

## Manual Aggregation

If you already have per-method CSV files and want to aggregate them manually:

```bash
cd exp/throughput
python3 aggregate_throughput_results.py \
  --output-dir ./results \
  --output-prefix manual_compare \
  --result OURS=./results/throughput_summary_OURS_YYYYMMDD_HHMMSS.csv \
  --result KIVI_2BIT=./results/throughput_summary_KIVI_2BIT_YYYYMMDD_HHMMSS.csv \
  --result NONE=./results/throughput_summary_NONE_YYYYMMDD_HHMMSS.csv
```

This generates:

- a combined long-format CSV
- a combined wide CSV
- a markdown comparison table

## Notes

- Start with a small sweep before running large overload experiments.
- The script assumes a compatible vLLM server is already running and reachable on the configured port.
- Larger `--num-requests` values usually produce more stable throughput numbers.
