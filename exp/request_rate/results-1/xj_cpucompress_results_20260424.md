# xj CPU Compress Queue Results (2026-04-24 UTC)

## Completed Runs

### 1) Single-request validation
- Output dir: `/home/xujie/catkv-vllm/exp/request_rate/results/xj_cpucompress_final_20260423T1320Z`
- Requests: `1`
- Request rate: `1 RPS`
- Source limit: `1`
- Max tokens: `1`
- Peak queue: `30.36 MiB`
- Peak pending: `52`
- Max RSS: `45.20 GiB`
- Notes: first successful end-to-end xj CPU compress queue run after fixing config propagation, `PYTHONPATH`, and xj prefetch miss handling.

### 2) Shell wrapper validation
- Output dir: `/home/xujie/catkv-vllm/exp/request_rate/results/xj_cpucompress_shell_20260423T1326Z`
- Requests: `2`
- Request rate: `2 RPS`
- Source limit: `2`
- Max tokens: `1`
- Peak queue: `36.40 MiB`
- Peak pending: `68` (queue log peak), summary file reports `68`; raw queue log shows transient `73`
- Max RSS: `45.33 GiB`
- Notes: validates `exp/request_rate/run_xj_cpucompress_wikimqa.sh`.

### 3) Medium pressure run
- Output dir: `/home/xujie/catkv-vllm/exp/request_rate/results/xj_cpucompress_dedup_20260424T022956Z`
- Requests: `120`
- Request rate: `4 RPS`
- Source limit: `20`
- Max tokens: `16`
- Peak queue: `2012.37 MiB`
- Peak pending: `797`
- Max RSS: `49.14 GiB`
- Total submit volume: about `83.09 GiB`
- Peak / cumulative submit ratio: about `6.8%`
- Notes: this run demonstrates that queue peak is instantaneous backlog, not total processed bytes.

### 4) High pressure run
- Output dir: `/home/xujie/catkv-vllm/exp/request_rate/results/xj_cpucompress_dedup_20260424T024345Z`
- Requests: `200`
- Request rate: `2 RPS`
- Source limit: `200`
- Max tokens: `32`
- Peak queue: `5783.31 MiB` (`5.65 GiB`)
- Peak pending: `2287`
- Max RSS: `55.24 GiB`
- Offload events: `33760`
- Unique offloaded paths: `33760`
- Average bytes per pending item at peak: about `2.53 MiB`
- Notes: queue peak is consistent with `2287 * ~2.53 MiB` item size.

## In Progress
- Output dir: `/home/xujie/catkv-vllm/exp/request_rate/results/xj_cpucompress_backlog_push_20260424T1330Z`
- Goal: push backlog higher with more aggressive settings (`REQUEST_RATE=4`, `XJ_NUM_WORKERS=8`)
- Status: still running; latest observed state at 2026-04-24 03:08 UTC -> queue log has only `init/tick`, `max_queue_gib=0.0`, while client requests are still completing slowly.


### 5) Aggressive backlog push run
- Output dir: `/home/xujie/catkv-vllm/exp/request_rate/results/xj_cpucompress_backlog_push_20260424T1330Z`
- Requests: `200`
- Request rate: `4 RPS`
- Source limit: `200`
- Max tokens: `32`
- XJ workers: `8`
- Success count: `90/200`
- Peak queue: `0.0 MiB`
- Peak pending: `0`
- Max RSS: `45.07 GiB`
- Notes: this run did **not** enter the xj CPU-compress queue. The queue log contains only `init/tick`, and many client requests timed out at `300s`. This is not a valid backlog-push result and should be treated as a failed run / wrong test condition.


### 6) Aggressive backlog push run (timeout opened)
- Output dir: `/home/xujie/catkv-vllm/exp/request_rate/results/xj_cpucompress_backlog_push_timeout1800_20260424T0345Z`
- Requests: `200`
- Request rate: `4 RPS`
- Source limit: `200`
- Max tokens: `32`
- XJ workers: `8`
- Timeout: `1800s`
- Success count: `200/200`
- Peak queue: `55018.46 MiB` (`53.73 GiB`)
- Peak pending: `21877`
- Max RSS: `131.03 GiB`
- Notes: after removing remote cache and lifting the 300s timeout, the same high-pressure case entered xj CPU-compress correctly and backlog rose continuously past 50 GiB before completion.


### 7) Prewarm first 20 rows, then test at 2 RPS
- Output dir: `/home/xujie/catkv-vllm/exp/request_rate/results/xj_cpucompress_prewarm20_rate2_20260424T0435Z`
- Warmup: first `20` dataset rows, sequentially, waited until each generation finished
- Warmup result file: `/home/xujie/catkv-vllm/exp/request_rate/results/xj_cpucompress_prewarm20_rate2_20260424T0435Z/warmup_results.json`
- Warmup success: `20/20`
- Formal test requests: `200`
- Request rate: `2 RPS`
- Source limit: `200`
- Max tokens: `32`
- Success count: `200/200`
- Peak queue: `4688.65 MiB` (`4.58 GiB`)
- Peak pending: `1858`
- Max RSS: `53.55 GiB`
- Notes: compared with the non-prewarm `200 req / 2 RPS / 200 source / 32 tokens` run, warmup of the first 20 rows reduced the observed queue peak from about `5.65 GiB` to about `4.58 GiB`.


### 8) 32-worker no-prewarm high-rate run
- Output dir: `/home/xujie/catkv-vllm/exp/request_rate/results/xj_cpucompress_50g_target_w32_20260424T0720Z`
- Requests: `200`
- Request rate: `8 RPS`
- Source limit: `200`
- Max tokens: `32`
- XJ workers: `32`
- Timeout: `1800s`
- Success count: `200/200`
- Peak queue: `3227.52 MiB` (`3.15 GiB`)
- Peak pending: `1250`
- Max RSS: `51.19 GiB`
- Notes: with the same 200 unique requests but `32` CPU-compress workers, queue backlog stays around `3 GiB`; this is much lower than the earlier `53.73 GiB` run with `8` workers.
