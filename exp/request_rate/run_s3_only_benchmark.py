#!/usr/bin/env python3
"""
S3-only Benchmark (robust version)

Fixes common "hang" issues:
1) DO NOT use stdout=PIPE without reading it -> redirect server logs to file.
2) Add aiohttp timeouts + per-request timeout to avoid waiting forever.
3) Add prompt length check (skip too-long prompts) to avoid server-side stalls/OOM.
4) Optional: warmup a "max-shape" request to reduce first-request compilation stall.
5) Safer server stop: SIGTERM -> wait -> SIGKILL.

Requires:
- vllm serve available in PATH
- test_large_prompt.py provides:
  - helpers.stream_completion(session, prompt, request_name) -> dict
  - helpers.send_requests_with_rate_limit(entries, rate, burstiness) -> async generator
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple, Optional

import aiohttp
from transformers import AutoTokenizer

CURRENT_DIR = Path(__file__).resolve().parent
BLEND_DIR = CURRENT_DIR.parent
if str(BLEND_DIR) not in sys.path:
    sys.path.insert(0, str(BLEND_DIR))

import test_large_prompt as helpers  # noqa: E402


DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_COMPRESS_METHODS = [ "OURS", "SVDQ","KIVI_2BIT", "NONE"]
DEFAULT_REQUEST_RATES = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0]
DEFAULT_DATASET = "/home/xujie/TaoTie/dataset/needle_multi_rag.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S3-only benchmark (robust)")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    p.add_argument("--num-contexts", type=int, default=20)
    p.add_argument("--num-requests", type=int, default=30)
    p.add_argument("--contexts-per-request", type=str, default="2,4")
    p.add_argument("--request-rates", type=str, default=None)
    p.add_argument("--compress-methods", type=str, default=None)

    p.add_argument("--port", type=int, default=12345)
    p.add_argument("--gpu", type=int, default=1)

    p.add_argument("--warmup-delay", type=float, default=5.0)
    p.add_argument("--warmup-max-shape", action="store_true",
                   help="After storing contexts, send 1 max-shape request to precompile hottest path.")

    p.add_argument("--burstiness", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--output-dir", type=str, default=".")
    p.add_argument("--output-csv", type=str, default="s3_only_benchmark.csv")
    p.add_argument("--output-prefix", type=str, default="benchmark")

    # Robustness knobs
    p.add_argument("--max-model-len", type=int, default=32000,
                   help="Must match vLLM --max-model-len; used for prompt length guard.")
    p.add_argument("--client-total-timeout", type=float, default=600.0,
                   help="aiohttp total timeout (seconds).")
    p.add_argument("--client-read-timeout", type=float, default=600.0,
                   help="aiohttp sock_read timeout (seconds).")
    p.add_argument("--client-connect-timeout", type=float, default=30.0,
                   help="aiohttp sock_connect timeout (seconds).")
    p.add_argument("--per-request-timeout", type=float, default=600.0,
                   help="Timeout for a single request coroutine (seconds).")
    p.add_argument("--server-start-timeout", type=int, default=180,
                   help="Wait-for-server timeout (seconds).")
    p.add_argument("--server-stop-timeout", type=float, default=20.0,
                   help="How long to wait for SIGTERM before SIGKILL (seconds).")

    return p.parse_args()


def load_dataset(path: str) -> List[dict]:
    samples: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def extract_contexts_and_questions(samples: List[dict], limit: int) -> Tuple[List[str], List[str]]:
    contexts: List[str] = []
    seen = set()
    questions: List[str] = []

    for sample in samples:
        for ctx in sample.get("contexts", []):
            if ctx not in seen:
                seen.add(ctx)
                contexts.append(ctx)
                if len(contexts) >= limit:
                    break
        q = sample.get("question", "").strip()
        if q:
            questions.append(q)
        if len(contexts) >= limit:
            break

    return contexts, questions


def kill_server():
    # Keep your original logic; simplest and effective in shared machines
    subprocess.run(["pkill", "-9", "-f", "EngineCore"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "APIServer"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "vllm serve"], capture_output=True)
    time.sleep(3)


def start_server(model: str, port: int, gpu: int, compress_type: str, log_dir: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["HF_HUB_OFFLINE"] = "1"
    env["LMCACHE_COMPRESS_TYPE"] = compress_type

    # LMCache configs
    env["LMCACHE_CHUNK_SIZE"] = "256"
    env["LMCACHE_ENABLE_BLENDING"] = "True"
    env["LMCACHE_BLEND_SPECIAL_STR"] = " # # "
    env["LMCACHE_USE_LAYERWISE"] = "True"
    env["LMCACHE_BLEND_CHECK_LAYERS"] = "1"
    env["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = "0.15"
    env["LMCACHE_BLEND_MIN_TOKENS"] = "64"
    env["LMCACHE_LOCAL_CPU"] = "True"
    env["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"

    cmd = [
        "vllm", "serve", model,
        "--kv-transfer-config", '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}',
        "--port", str(port),
        "--gpu-memory-utilization", "0.6",
        "--max-model-len", "32000",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--enforce-eager",
        "-tp", "1",
    ]

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"vllm_{compress_type}_{port}.log"
    log_f = open(log_path, "a", buffering=1)

    print(f"[Server] Starting with LMCACHE_COMPRESS_TYPE={compress_type}")
    print(f"[Server] Log -> {log_path}")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_f,              # <-- critical: avoid PIPE hang
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )
    # attach for cleanup
    setattr(proc, "_log_f", log_f)
    setattr(proc, "_log_path", str(log_path))
    return proc


def wait_for_server(port: int, timeout: int = 180) -> bool:
    import urllib.request
    import urllib.error

    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    print(f"[Server] Ready on port {port}")
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
            pass
        time.sleep(2)
    print(f"[Server] Timeout waiting for server on port {port}")
    return False


def stop_server(proc: Optional[subprocess.Popen], stop_timeout: float):
    if proc is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except Exception:
        pass

    # Wait a bit
    t0 = time.time()
    while time.time() - t0 < stop_timeout:
        if proc.poll() is not None:
            break
        time.sleep(0.2)

    # Force kill if still alive
    if proc.poll() is None:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            pass

    # Close log file handle
    try:
        log_f = getattr(proc, "_log_f", None)
        if log_f:
            log_f.close()
    except Exception:
        pass


async def warmup_contexts(
    session: aiohttp.ClientSession,
    contexts: List[str],
    tokenizer,
    sep_tokens: List[int],
    per_request_timeout: float
) -> Tuple[int, int]:
    successes = 0

    
    failures = 0
    total = len(contexts)

    print(f"\n{'=' * 60}")
    print(f"Warmup: storing {total} unique contexts")
    print(f"{'=' * 60}")

    for idx, ctx in enumerate(contexts, start=1):
        tokens = tokenizer.encode(ctx, add_special_tokens=False)
        full_tokens = sep_tokens + tokens
        request_name = f"[Warmup {idx}/{total}]"

        print(f"{request_name} start (tokens={len(full_tokens)})")
        try:
            await asyncio.wait_for(
                helpers.stream_completion(session=session, prompt=full_tokens, request_name=request_name),
                timeout=per_request_timeout
            )
            successes += 1
            print(f"{request_name} done")
        except Exception as e:
            print(f"{request_name} failed: {type(e).__name__}: {e}")
            failures += 1

        await asyncio.sleep(0.3)

    print(f"{'-' * 60}")
    print(f"Warmup completed. Successes: {successes}, Failures: {failures}")
    print(f"{'-' * 60}")
    return successes, failures


def _build_prompt_from_contexts(
    selected_contexts: List[str],
    question: str,
    tokenizer,
    sep_tokens: List[int]
) -> List[int]:
    tokens: List[int] = []
    for ctx in selected_contexts:
        tokens.extend(sep_tokens)
        tokens.extend(tokenizer.encode(ctx, add_special_tokens=False))
    tokens.extend(sep_tokens)
    tokens.extend(tokenizer.encode(question, add_special_tokens=False))
    return tokens


async def warmup_max_shape_once(
    session: aiohttp.ClientSession,
    contexts: List[str],
    questions: List[str],
    ctx_max: int,
    tokenizer,
    sep_tokens: List[int],
    per_request_timeout: float,
    max_model_len: int
):
    """Send one 'largest' request to trigger compilation/allocations early."""
    if not contexts:
        return

    # pick longest contexts by token length
    ctx_lens = [(len(tokenizer.encode(c, add_special_tokens=False)), c) for c in contexts]
    ctx_lens.sort(key=lambda x: x[0], reverse=True)
    selected = [c for _, c in ctx_lens[:max(1, min(ctx_max, len(ctx_lens)))]]

    q = questions[0] if questions else "Answer the question."
    prompt = _build_prompt_from_contexts(selected, q, tokenizer, sep_tokens)

    if len(prompt) > max_model_len:
        # If too long, truncate selected contexts until fits
        while len(prompt) > max_model_len and len(selected) > 1:
            selected.pop()  # drop one context
            prompt = _build_prompt_from_contexts(selected, q, tokenizer, sep_tokens)

    request_name = f"[Warmup MaxShape] ctx={len(selected)} tokens={len(prompt)}"
    print(f"{request_name} start")
    try:
        await asyncio.wait_for(
            helpers.stream_completion(session=session, prompt=prompt, request_name=request_name),
            timeout=per_request_timeout
        )
        print(f"{request_name} done")
    except Exception as e:
        print(f"{request_name} failed: {type(e).__name__}: {e}")


def prepare_requests(
    contexts: List[str],
    questions: List[str],
    num_requests: int,
    contexts_range: Tuple[int, int],
    tokenizer,
    sep_tokens: List[int],
    seed: int
) -> List[List[int]]:
    rng = random.Random(seed)
    prompts: List[List[int]] = []
    min_ctx, max_ctx = contexts_range

    for _ in range(num_requests):
        n_ctx = rng.randint(min_ctx, max_ctx)
        selected = rng.sample(contexts, min(n_ctx, len(contexts)))
        question = rng.choice(questions) if questions else "Answer the question."
        prompts.append(_build_prompt_from_contexts(selected, question, tokenizer, sep_tokens))

    return prompts


async def run_benchmark_rate(
    session: aiohttp.ClientSession,
    prompts: List[List[int]],
    rate: float,
    burstiness: float,
    per_request_timeout: float,
    max_model_len: int
) -> Dict:
    rate_str = f"{rate}" if rate != float("inf") else "inf"
    print(f"\n{'=' * 60}")
    print(f"Running benchmark for rate={rate_str} RPS")
    print(f"{'=' * 60}")

    # Build entries for helpers.send_requests_with_rate_limit
    entries = []
    skipped = 0
    for i, prompt in enumerate(prompts, start=1):
        if len(prompt) > max_model_len:
            skipped += 1
            print(f"[Skip] req={i} prompt too long: {len(prompt)} > {max_model_len}")
            continue
        request_name = f"[rate={rate_str}] req={i}"
        entries.append((request_name, prompt, {"ordinal": i}))

    if not entries:
        return {
            "ttft_avg": None,
            "latency_avg": None,
            "itl_avg": None,
            "success_rate": 0.0,
            "successes": 0,
            "total": 0,
            "skipped": skipped
        }

    tasks = []
    async for request_name, prompt, _ in helpers.send_requests_with_rate_limit(entries, rate, burstiness):
        coro = helpers.stream_completion(session=session, prompt=prompt, request_name=request_name)
        task = asyncio.create_task(asyncio.wait_for(coro, timeout=per_request_timeout))
        tasks.append((request_name, task))

    gathered = await asyncio.gather(*(t for _, t in tasks), return_exceptions=True)

    ttfts, latencies, itls = [], [], []
    successes = 0

    for (request_name, _), result in zip(tasks, gathered):
        if isinstance(result, Exception):
            print(f"{request_name} errored: {type(result).__name__}: {result}")
        else:
            if result.get("success"):
                successes += 1
                if result.get("ttft") is not None:
                    ttfts.append(result["ttft"])
                if result.get("latency") is not None:
                    latencies.append(result["latency"])
                if result.get("itl"):
                    itls.extend(result["itl"])

    total_sent = len(entries)
    return {
        "ttft_avg": mean(ttfts) * 1000 if ttfts else None,
        "latency_avg": mean(latencies) * 1000 if latencies else None,
        "itl_avg": mean(itls) * 1000 if itls else None,
        "success_rate": successes / total_sent * 100 if total_sent else 0,
        "successes": successes,
        "total": total_sent,
        "skipped": skipped
    }


async def run_single_method_benchmark(
    model: str,
    port: int,
    contexts: List[str],
    questions: List[str],
    request_rates: List[float],
    num_requests: int,
    contexts_range: Tuple[int, int],
    warmup_delay: float,
    warmup_max_shape: bool,
    burstiness: float,
    seed: int,
    client_timeout_total: float,
    client_timeout_read: float,
    client_timeout_connect: float,
    per_request_timeout: float,
    max_model_len: int
) -> List[Dict]:
    results: List[Dict] = []

    tokenizer = AutoTokenizer.from_pretrained(model)
    sep_tokens = tokenizer.encode(" # # ", add_special_tokens=False)

    helpers.api_base = f"http://localhost:{port}"
    helpers.model = model

    timeout = aiohttp.ClientTimeout(
        total=client_timeout_total,
        sock_read=client_timeout_read,
        sock_connect=client_timeout_connect
    )

    async with aiohttp.ClientSession(timeout=timeout) as session:
        await warmup_contexts(session, contexts, tokenizer, sep_tokens, per_request_timeout)

        # Optional max-shape warmup to reduce first benchmark request stall
        if warmup_max_shape:
            ctx_min, ctx_max = contexts_range
            await warmup_max_shape_once(
                session=session,
                contexts=contexts,
                questions=questions,
                ctx_max=ctx_max,
                tokenizer=tokenizer,
                sep_tokens=sep_tokens,
                per_request_timeout=per_request_timeout,
                max_model_len=max_model_len
            )

        print(f"Waiting {warmup_delay} seconds after warmup...")
        await asyncio.sleep(warmup_delay)

        for rate in request_rates:
            prompts = prepare_requests(
                contexts, questions, num_requests, contexts_range,
                tokenizer, sep_tokens, seed
            )
            metrics = await run_benchmark_rate(
                session=session,
                prompts=prompts,
                rate=rate,
                burstiness=burstiness,
                per_request_timeout=per_request_timeout,
                max_model_len=max_model_len
            )
            results.append({"rate": rate, **metrics})

    return results


def print_summary_table(all_results: Dict[str, List[Dict]]):
    print(f"\n{'=' * 88}")
    print("Summary Table (TTFT in ms) | (Skipped) | (Success%)")
    print(f"{'=' * 88}")

    methods = list(all_results.keys())
    rates = sorted(set(r["rate"] for rs in all_results.values() for r in rs))

    header = f"{'Rate(RPS)':>10} "
    for m in methods:
        header += f"| {m:>18} "
    print(header)
    print("-" * 88)

    for rate in rates:
        rate_str = f"{rate}" if rate != float("inf") else "inf"
        row = f"{rate_str:>10} "
        for m in methods:
            rec = next((x for x in all_results[m] if x["rate"] == rate), None)
            if not rec:
                row += f"| {'N/A':>18} "
                continue
            ttft = rec["ttft_avg"]
            succ = rec.get("success_rate", 0.0)
            skipped = rec.get("skipped", 0)
            if ttft is None:
                row += f"| {'N/A':>7}ms sk={skipped:<3} s={succ:>5.1f}% "
            else:
                row += f"| {ttft:>7.1f}ms sk={skipped:<3} s={succ:>5.1f}% "
        print(row)

    print("-" * 88)


def write_results(all_results: Dict[str, List[Dict]], output_dir: Path, output_csv: str, output_prefix: str, model: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON per method
    for method, results in all_results.items():
        json_path = output_dir / f"{output_prefix}_{method}_s3_only.json"
        with open(json_path, "w") as f:
            json.dump({"model": model, "method": method, "results": results}, f, indent=2)
        print(f"[Write] {json_path}")

    # Combined CSV
    csv_path = output_dir / output_csv
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model", "compress_method", "rate_rps",
            "ttft_avg_ms", "latency_avg_ms", "itl_avg_ms",
            "success_rate", "successes", "total", "skipped",
            "timestamp"
        ])
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        for method, results in all_results.items():
            for r in results:
                w.writerow([
                    model,
                    method,
                    r["rate"],
                    f"{r['ttft_avg']:.2f}" if r["ttft_avg"] is not None else "N/A",
                    f"{r['latency_avg']:.2f}" if r["latency_avg"] is not None else "N/A",
                    f"{r['itl_avg']:.2f}" if r["itl_avg"] is not None else "N/A",
                    f"{r.get('success_rate', 0.0):.1f}%",
                    r.get("successes", 0),
                    r.get("total", 0),
                    r.get("skipped", 0),
                    now
                ])
    print(f"[Write] {csv_path}")


def main():
    args = parse_args()

    compress_methods = args.compress_methods.split(",") if args.compress_methods else DEFAULT_COMPRESS_METHODS

    request_rates = DEFAULT_REQUEST_RATES
    if args.request_rates:
        request_rates = []
        for r in args.request_rates.split(","):
            r = r.strip()
            if r.lower() == "inf":
                request_rates.append(float("inf"))
            else:
                request_rates.append(float(r))

    ctx_min, ctx_max = map(int, args.contexts_per_request.split(","))
    output_dir = Path(args.output_dir)
    log_dir = output_dir / "server_logs"

    print("=" * 80)
    print("S3-Only Benchmark (robust)")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Compression methods: {compress_methods}")
    print(f"Request rates: {request_rates}")
    print(f"Num contexts: {args.num_contexts}")
    print(f"Requests per rate: {args.num_requests}")
    print(f"Contexts per request: {ctx_min}-{ctx_max}")
    print(f"GPU: {args.gpu}")
    print(f"Output dir: {output_dir}")
    print(f"Warmup max-shape: {args.warmup_max_shape}")
    print(f"Timeouts: client_total={args.client_total_timeout}s, per_req={args.per_request_timeout}s")
    print("=" * 80)

    print("\n[Dataset] Loading...")
    samples = load_dataset(args.dataset)
    contexts, questions = extract_contexts_and_questions(samples, args.num_contexts)
    print(f"[Dataset] {len(contexts)} unique contexts, {len(questions)} questions")

    all_results: Dict[str, List[Dict]] = {}

    for method in compress_methods:
        print(f"\n{'#' * 80}")
        print(f"# Testing: {method}")
        print(f"{'#' * 80}")

        print("[Server] Killing existing processes...")
        kill_server()

        proc = start_server(args.model, args.port, args.gpu, method, log_dir)

        try:
            if not wait_for_server(args.port, timeout=args.server_start_timeout):
                print(f"[Error] Server failed to start for {method}. Check log: {getattr(proc, '_log_path', 'N/A')}")
                continue

            results = asyncio.run(run_single_method_benchmark(
                model=args.model,
                port=args.port,
                contexts=contexts,
                questions=questions,
                request_rates=request_rates,
                num_requests=args.num_requests,
                contexts_range=(ctx_min, ctx_max),
                warmup_delay=args.warmup_delay,
                warmup_max_shape=args.warmup_max_shape,
                burstiness=args.burstiness,
                seed=args.seed,
                client_timeout_total=args.client_total_timeout,
                client_timeout_read=args.client_read_timeout,
                client_timeout_connect=args.client_connect_timeout,
                per_request_timeout=args.per_request_timeout,
                max_model_len=args.max_model_len
            ))

            all_results[method] = results

            print(f"\n{method} Results:")
            print(f"{'Rate':>10} {'TTFT(avg)':>12} {'Latency(avg)':>14} {'ITL(avg)':>12} {'Succ%':>8} {'Skip':>6}")
            print("-" * 72)
            for r in results:
                rate_str = f"{r['rate']}" if r["rate"] != float("inf") else "inf"
                ttft = f"{r['ttft_avg']:.2f}ms" if r["ttft_avg"] is not None else "N/A"
                lat = f"{r['latency_avg']:.2f}ms" if r["latency_avg"] is not None else "N/A"
                itl = f"{r['itl_avg']:.2f}ms" if r["itl_avg"] is not None else "N/A"
                succ = f"{r.get('success_rate', 0.0):.1f}%"
                sk = r.get("skipped", 0)
                print(f"{rate_str:>10} {ttft:>12} {lat:>14} {itl:>12} {succ:>8} {sk:>6}")

        except Exception as e:
            print(f"[Error] Benchmark failed for {method}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            print(f"[Hint] Check server log: {getattr(proc, '_log_path', 'N/A')}")
        finally:
            print("[Server] Stopping...")
            stop_server(proc, stop_timeout=args.server_stop_timeout)
            kill_server()

    if all_results:
        print_summary_table(all_results)
        write_results(all_results, output_dir, args.output_csv, args.output_prefix, args.model)

    print("\n[Done] Benchmark complete.")


if __name__ == "__main__":
    main()
