# Request-Rate Benchmarks

This subdirectory contains throughput, TTFT, and cache-reuse benchmark helpers for `blend_kv_v1`.

## Start Here

Choose the runner by intent:

| Goal | Recommended entrypoint | Notes |
| --- | --- | --- |
| Normal multi-method benchmark | `./run_benchmark.sh` | Preferred top-level runner |
| Quick smoke benchmark | `./run_simple_bench.sh` or `python3 test_single.py` | Lowest-overhead path |
| S3-only benchmark flow | `python3 run_s3_only_benchmark.py` | Preferred S3-only path |
| Broader sweep across models/methods | `python3 run_full_benchmark.py` | Advanced and invasive |
| Direct reuse benchmark implementation | `python3 bench_reuse.py` | Main Python implementation behind the shell runner |

## Script Groups

### Preferred entrypoints

- `run_benchmark.sh`: primary shell entrypoint for multi-method request-rate benchmarks
- `bench_reuse.py`: preferred Python implementation for reuse and TTFT measurement
- `run_s3_only_benchmark.py`: safer S3-only orchestrator
- `run_simple_bench.sh`: fastest way to run a smaller smoke benchmark
- `test_single.py`: single-run debug/smoke helper

### Alternate or advanced runners

- `run_benchmark.py`: alternate Python orchestrator with overlapping responsibilities
- `run_full_benchmark.py`: full sweep runner across models, compression methods, and rates
- `benchmark_incremental.py`: incremental / checkpoint-oriented benchmark flow
- `benchmark_reuse.py`: older or alternate reuse implementation
- `ttft_bench.py`: TTFT-specific runner
- `run_test.sh`: thin wrapper around a narrower benchmark path

### Analysis-only scripts

- `eval.py`: result aggregation helper
- `plot_s3_only_comparison.py`: plotting/analysis helper, not a benchmark entrypoint

## Important Notes

- `run_benchmark.sh` is the best default if you are unsure where to start.
- `bench_reuse.py` and `benchmark_reuse.py` overlap; prefer `bench_reuse.py` unless you need the alternate flow.
- `run_full_benchmark.py` is higher risk than the other runners.
  It edits `lmcache/v1/compute/blend/kvmanager.py` to switch compression mode, so treat it as an invasive experiment runner rather than a normal benchmark wrapper.

## Generated Outputs

Most files in this directory are experiment outputs, not source:

- `*.log`
- `results/`
- `*_checkpoint.json`
- `*_incremental.csv`
- `ttft_results.json`
- ad-hoc result JSON files such as `test_*`, `fix_*`, and `ttft_*`

These are ignored via local ignore rules so the runnable scripts remain easy to find.

## Typical Usage

```bash
cd exp/request_rate
bash run_benchmark.sh --methods "OURS,KIVI_2BIT,NONE" --rates "1,2,4"
```

Run the same benchmark on `cuda:1` with fixed-interval scheduling for sanity checking:

```bash
cd exp/request_rate
GPU=1 SCHEDULE=fixed bash run_benchmark.sh --methods "NONE,KIVI_2BIT,OURS,SVDQ"
```

The primary runner will:

1. Start a vLLM server with LMCache blend env vars.
2. Wait for the health endpoint.
3. Run the selected benchmark flow.
4. Write logs and summaries under the benchmark output locations.

`bench_reuse.py` supports two arrival modes:

- `--schedule poisson`: randomized Poisson arrivals, closer to bursty online traffic
- `--schedule fixed`: strict `1 / rate` spacing between requests, useful for sanity checks

The shell wrapper forwards `SCHEDULE`/`--schedule` to `bench_reuse.py`.

## Practical Advice

- Start with `run_simple_bench.sh` or `test_single.py` before large sweeps.
- Use `run_benchmark.sh` as the default shared workflow for reproducible experiments.
- Keep long-lived outputs under `results/` if you want to preserve them.
- If you need a result committed to git, move the selected output to a tracked path first instead of committing the whole experiment clutter.
