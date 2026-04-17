# KIVI Warmup Compression Fix Report

## Summary

This report documents the investigation and repair for the issue where running
`./run_full_method_benchmark.sh` with `KIVI_2BIT` did not route warmup KV-cache
storage through the KIVI compression function.

## Repair Plan

1. Trace the warmup storage path from the throughput benchmark into the blend
   KV storage code.
2. Identify where `KIVI_2BIT` diverges from the expected compressor path.
3. Patch the storage/offload logic so compressed methods always execute their
   compressor implementation.
4. Validate by running a focused KIVI benchmark and checking logs for the KIVI
   compression profile output.

## Root Cause

The throughput benchmark itself correctly exported:

- `LMCACHE_COMPRESS_TYPE=KIVI_2BIT`

However, warmup storage did **not** always call `KVCacheManager.store_data()`.
Instead, the layer prefill path used:

- `ContextManager.prefill()`
- `KVCacheManager.offload_layer_data()`

Inside `KVCacheManager.offload_layer_data()`, the code allowed
`CompressType.KIVI_2BIT` to go through `_supports_stream_prefill_offload()`.
That branch copied raw key/value tensors to pinned CPU memory and called
`db.store_data()` directly, which bypassed:

- `self.compressor.compress(...)`

As a result, the KIVI-specific compression function in
`lmcache/v1/compute/blend/compress/kivi.py` was skipped during warmup storage.

## Code Changes

### 1. Restrict raw stream offload to `NONE`

File:

- `lmcache/v1/compute/blend/kvmanager.py`

Change:

- `_supports_stream_prefill_offload()` now returns `True` only for
  `CompressType.NONE`.

Why:

- Raw pinned-memory offload is only semantically correct for uncompressed KV.
- Any compressed format must execute `self.compressor.compress(...)`.

### 2. Make `offload_compress_data()` compression-aware

File:

- `lmcache/v1/compute/blend/kvmanager.py`

Change:

- Added `layer_idx` argument.
- If `self.compress_type != CompressType.NONE`, the method now falls back to
  `self.store_data(...)` so the active compressor is used.

Why:

- This prevents future chunk/offload paths from bypassing compression for
  `KIVI_2BIT`, `OURS`, or `SVDQ`.

### 3. Propagate `layer_idx` at the caller

File:

- `lmcache/v1/compute/blend/context_manager.py`

Change:

- `store_chunks_kv()` now passes `layer_idx` into
  `self.kv_manager.offload_compress_data(...)`.

Why:

- Keeps the compression-aware fallback path fully informed and consistent with
  the normal `store_data()` interface.

## Files Modified

- `lmcache/v1/compute/blend/kvmanager.py`
- `lmcache/v1/compute/blend/context_manager.py`
- `lmcache/v1/compute/blend/compress/kivi.py`
- `exp/throughput/KIVI_WARMUP_FIX_REPORT_2026-04-16.md`

## Expected Behavior After Fix

When the benchmark runs with:

- `LMCACHE_COMPRESS_TYPE=KIVI_2BIT`

warmup KV storage should now flow through:

- `KVCacheManager.store_data()`
- `self.compressor.compress(...)`
- `Kivi2Bit.compress(...)`

instead of the raw pinned-memory `db.store_data()` bypass path.

## Additional Issue Found During Validation

After the warmup/offload fix, validation showed that the KIVI path was finally
reaching `Kivi2Bit.compress(...)`, but the request still failed because the
compressor called:

- `profile_log(..., flush=True)`

while `profile_log()` only accepted a single positional argument. That caused:

- `TypeError: profile_log() got an unexpected keyword argument 'flush'`

### Fix

Files:

- `lmcache/v1/compute/blend/kvmanager.py`
- `lmcache/v1/compute/blend/compress/kivi.py`

Change:

- Updated `profile_log()` helpers to accept `*args, **kwargs` and ignore extra logging
  options.

Why:

- Keeps profiling helper calls backward-compatible with existing call sites.
- Avoids crashing the actual compression path just because a debug flag passes
  through a `flush` keyword.

## Validation Steps

Recommended validation command:

```bash
cd exp/throughput
./run_full_method_benchmark.sh --method KIVI_2BIT --rates "0.5,1" --num-contexts 4 --num-requests 2 --ctx-per-req "2,2" --output-prefix kivi_fix_check
```

Then inspect:

```bash
rg -n "\[KIVI compress\]" exp/throughput/server_logs/vllm_KIVI_2BIT_12345.log
```

If the fix is working, the server log should include KIVI compression profile
lines during warmup/store requests.

## Notes

- The benchmark wrapper previously also had local default overrides that forced
  only `KIVI_2BIT`; those defaults were already corrected earlier so the runner
  supports all compression methods again.
- This report focuses on the actual compressor-bypass bug in the warmup storage
  path.

## Validation Result

Validation was run with:

```bash
cd exp/throughput
./run_full_method_benchmark.sh --method KIVI_2BIT --rates "0.5" --num-contexts 2 --num-requests 1 --ctx-per-req "2,2" --output-prefix kivi_fix_check3
```

Result:

- Warmup completed for 2/2 contexts.
- Benchmark completed with 1/1 successful request.
- Combined CSV / wide CSV / Markdown outputs were generated.
- Server log contains `[KIVI compress]` entries, confirming warmup storage now enters `Kivi2Bit.compress(...)`.

Output files:

- `exp/throughput/results/kivi_fix_check3_KIVI_2BIT_20260416_165036.csv`
- `exp/throughput/results/kivi_fix_check3_KIVI_2BIT_20260416_165036.json`
- `exp/throughput/results/kivi_fix_check3_combined_20260416_165037.csv`
- `exp/throughput/results/kivi_fix_check3_combined_20260416_165037_wide.csv`
- `exp/throughput/results/kivi_fix_check3_combined_20260416_165037.md`
