#!/usr/bin/env python3
"""
S3-only Benchmark with incremental saves after each step.
Supports resume from checkpoint if interrupted.
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
from typing import Dict, List, Tuple

import aiohttp
from transformers import AutoTokenizer

CURRENT_DIR = Path(__file__).resolve().parent
BLEND_DIR = CURRENT_DIR.parent
if str(BLEND_DIR) not in sys.path:
    sys.path.insert(0, str(BLEND_DIR))

import test_large_prompt as helpers

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_COMPRESS_METHODS = ["KIVI_2BIT", "OURS", "SVDQ", "NONE"]
DEFAULT_REQUEST_RATES = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0]
DEFAULT_DATASET = "/home/xujie/TaoTie/dataset/needle_multi_rag.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S3-only benchmark with incremental saves"
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--num-contexts", type=int, default=20)
    parser.add_argument("--num-requests", type=int, default=30)
    parser.add_argument("--contexts-per-request", type=str, default="2,4")
    parser.add_argument("--request-rates", type=str, default=None)
    parser.add_argument("--compress-methods", type=str, default=None)
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--warmup-delay", type=float, default=5.0)
    parser.add_argument("--burstiness", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--output-prefix", type=str, default="benchmark")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--no-local-cache", action="store_true", help="Disable local CPU cache, S3 only")
    return parser.parse_args()


def load_dataset(path: str) -> List[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def extract_contexts_and_questions(samples: List[dict], limit: int) -> Tuple[List[str], List[str]]:
    contexts = []
    seen = set()
    questions = []

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


def kill_server(port: int = 12345):
    # Kill by process name
    subprocess.run(["pkill", "-9", "-f", "EngineCore"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "APIServer"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "vllm serve"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "vllm"], capture_output=True)
    time.sleep(2)

    # Kill process holding the port
    try:
        result = subprocess.run(
            f"ss -tlnp | grep :{port} | grep -oP 'pid=\\K[0-9]+'",
            shell=True, capture_output=True, text=True
        )
        if result.stdout.strip():
            for pid in result.stdout.strip().split('\n'):
                subprocess.run(["kill", "-9", pid], capture_output=True)
    except Exception:
        pass
    time.sleep(3)


def start_server(model: str, port: int, gpu: int, compress_type: str, tp: int = 1, use_local_cache: bool = True) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu) if tp == 1 else ",".join(str(i) for i in range(gpu, gpu + tp))
    env["HF_HUB_OFFLINE"] = "1"
    env["LMCACHE_COMPRESS_TYPE"] = compress_type
    env["LMCACHE_CHUNK_SIZE"] = "256"
    env["LMCACHE_ENABLE_BLENDING"] = "True"
    env["LMCACHE_BLEND_SPECIAL_STR"] = " # # "
    env["LMCACHE_USE_LAYERWISE"] = "True"
    env["LMCACHE_BLEND_CHECK_LAYERS"] = "1"
    env["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = "0.15"
    env["LMCACHE_BLEND_MIN_TOKENS"] = "64"

    # 本地缓存配置
    if use_local_cache:
        env["LMCACHE_LOCAL_CPU"] = "True"
        env["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"
    else:
        # 使用较小的本地缓存(1GB)，确保大部分数据走 S3
        # 注意：完全禁用或太小的本地缓存会导致 LMCache 崩溃
        # 1GB 足够存储压缩后的数据，但未压缩数据会溢出到 S3
        env["LMCACHE_LOCAL_CPU"] = "True"
        env["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "1"

    cmd = [
        "vllm", "serve", model,
        "--kv-transfer-config", '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}',
        "--port", str(port),
        "--gpu-memory-utilization", "0.6",
        "--max-model-len", "32000",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--enforce-eager",
        "-tp", str(tp)
    ]

    cache_mode = "local+S3" if use_local_cache else "S3 only"
    print(f"[Server] Starting with LMCACHE_COMPRESS_TYPE={compress_type}, tp={tp}, cache={cache_mode}")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )
    return proc


def wait_for_server(port: int, timeout: int = 300) -> bool:
    import urllib.request
    import urllib.error

    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    print(f"[Server] Ready on port {port}")
                    return True
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
            pass
        time.sleep(2)
    print(f"[Server] Timeout waiting for server on port {port}")
    return False


async def warmup_contexts(
    session: aiohttp.ClientSession,
    contexts: List[str],
    tokenizer,
    sep_tokens: List[int]
) -> Tuple[int, int]:
    successes = 0
    failures = 0
    total = len(contexts)

    print(f"\n{'='*60}")
    print(f"Warmup: storing {total} unique contexts")
    print(f"{'='*60}")

    for idx, ctx in enumerate(contexts, start=1):
        # Send context WITHOUT sep_tokens prefix to match blend's extraction
        # Blend's _fast_split_by_subtensor extracts tokens AFTER separator, not including it
        tokens = tokenizer.encode(ctx, add_special_tokens=False)
        request_name = f"[Warmup {idx}/{total}]"

        print(f"{request_name} start (tokens={len(tokens)})")

        try:
            await helpers.stream_completion(
                session=session,
                prompt=tokens,  # Only context tokens, no sep_tokens prefix
                request_name=request_name
            )
            successes += 1
            print(f"{request_name} done")
        except Exception as e:
            print(f"{request_name} failed: {e}")
            failures += 1

        await asyncio.sleep(0.3)

    print(f"{'-'*60}")
    print(f"Warmup completed. Successes: {successes}, Failures: {failures}")
    print(f"{'-'*60}")
    return successes, failures


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
    prompts = []
    min_ctx, max_ctx = contexts_range

    for _ in range(num_requests):
        n_ctx = rng.randint(min_ctx, max_ctx)
        selected = rng.sample(contexts, min(n_ctx, len(contexts)))
        question = rng.choice(questions) if questions else "Answer the question."

        tokens = []
        for ctx in selected:
            tokens.extend(sep_tokens)
            tokens.extend(tokenizer.encode(ctx, add_special_tokens=False))
        tokens.extend(sep_tokens)
        tokens.extend(tokenizer.encode(question, add_special_tokens=False))

        prompts.append(tokens)

    return prompts


async def run_benchmark_rate(
    session: aiohttp.ClientSession,
    prompts: List[List[int]],
    rate: float,
    burstiness: float
) -> Dict:
    rate_str = f"{rate}" if rate != float("inf") else "inf"
    print(f"\n{'='*60}")
    print(f"Running benchmark for rate={rate_str} RPS (sequential mode)")
    print(f"{'='*60}")

    # Sequential execution: wait for each request to complete before sending next
    # This is necessary because vLLM blend requests block the inference loop
    ttfts = []
    latencies = []
    itls = []
    successes = 0
    total = len(prompts)

    for idx, prompt in enumerate(prompts, start=1):
        request_name = f"[rate={rate_str}] req={idx}/{total}"
        try:
            result = await helpers.stream_completion(
                session=session, prompt=prompt, request_name=request_name
            )
            if result.get("success"):
                successes += 1
                if result.get("ttft") is not None:
                    ttfts.append(result["ttft"])
                if result.get("latency") is not None:
                    latencies.append(result["latency"])
                if result.get("itl"):
                    itls.extend(result["itl"])
                print(f"{request_name} TTFT={result['ttft']*1000:.1f}ms")
            else:
                print(f"{request_name} failed (no success flag)")
        except Exception as e:
            print(f"{request_name} errored with {type(e).__name__}: {e}")

    return {
        "ttft_avg": mean(ttfts) * 1000 if ttfts else None,
        "latency_avg": mean(latencies) * 1000 if latencies else None,
        "itl_avg": mean(itls) * 1000 if itls else None,
        "success_rate": successes / total * 100 if total else 0,
        "successes": successes,
        "total": total
    }


def save_checkpoint(checkpoint_path: Path, all_results: Dict, current_method: str, current_rate_idx: int):
    checkpoint = {
        "all_results": all_results,
        "current_method": current_method,
        "current_rate_idx": current_rate_idx
    }
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=2)
    print(f"[Checkpoint] Saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path: Path) -> Dict:
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return None


def save_incremental_csv(csv_path: Path, method: str, rate: float, metrics: Dict, model: str):
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "model", "compress_method", "rate_rps", "ttft_avg_ms", "latency_avg_ms",
                "itl_avg_ms", "success_rate", "timestamp"
            ])
        writer.writerow([
            model,
            method,
            rate,
            f"{metrics['ttft_avg']:.2f}" if metrics['ttft_avg'] else "N/A",
            f"{metrics['latency_avg']:.2f}" if metrics['latency_avg'] else "N/A",
            f"{metrics['itl_avg']:.2f}" if metrics['itl_avg'] else "N/A",
            f"{metrics['success_rate']:.1f}%",
            time.strftime("%Y-%m-%d %H:%M:%S")
        ])
    print(f"[Save] Appended result to {csv_path}")


async def run_single_method_benchmark(
    model: str,
    port: int,
    contexts: List[str],
    questions: List[str],
    request_rates: List[float],
    num_requests: int,
    contexts_range: Tuple[int, int],
    warmup_delay: float,
    burstiness: float,
    seed: int,
    output_dir: Path,
    output_prefix: str,
    method: str,
    start_rate_idx: int = 0
) -> List[Dict]:
    results = []

    tokenizer = AutoTokenizer.from_pretrained(model)
    sep_tokens = tokenizer.encode(" # # ", add_special_tokens=False)

    helpers.api_base = f"http://localhost:{port}"
    helpers.model = model

    csv_path = output_dir / f"{output_prefix}_incremental.csv"
    checkpoint_path = output_dir / f"{output_prefix}_checkpoint.json"

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        await warmup_contexts(session, contexts, tokenizer, sep_tokens)

        print(f"Waiting {warmup_delay} seconds after warmup...")
        await asyncio.sleep(warmup_delay)

        for idx, rate in enumerate(request_rates):
            if idx < start_rate_idx:
                print(f"[Skip] rate={rate} (already completed)")
                continue

            prompts = prepare_requests(
                contexts, questions, num_requests, contexts_range,
                tokenizer, sep_tokens, seed
            )

            metrics = await run_benchmark_rate(session, prompts, rate, burstiness)
            result = {"rate": rate, **metrics}
            results.append(result)

            save_incremental_csv(csv_path, method, rate, metrics, model)

            rate_str = f"{rate}" if rate != float("inf") else "inf"
            ttft = f"{metrics['ttft_avg']:.2f} ms" if metrics['ttft_avg'] else "N/A"
            lat = f"{metrics['latency_avg']:.2f} ms" if metrics['latency_avg'] else "N/A"
            itl = f"{metrics['itl_avg']:.2f} ms" if metrics['itl_avg'] else "N/A"
            print(f"\n[Result] {method} @ {rate_str} RPS: TTFT={ttft}, Latency={lat}, ITL={itl}, Success={metrics['success_rate']:.1f}%")

    return results


def print_summary_table(all_results: Dict[str, List[Dict]]):
    print(f"\n{'='*72}")
    print("Summary Table (TTFT in ms)")
    print(f"{'='*72}")
    print(f"{'Rate (RPS)':>12}  ", end="")
    for method in all_results.keys():
        print(f"{method:>15}", end="")
    print()
    print("-" * 72)

    rates = sorted(set(r["rate"] for results in all_results.values() for r in results))

    for rate in rates:
        rate_str = f"{rate}" if rate != float("inf") else "inf"
        print(f"{rate_str:>12}  ", end="")
        for method, results in all_results.items():
            ttft = next((r["ttft_avg"] for r in results if r["rate"] == rate), None)
            if ttft is not None:
                print(f"{ttft:>12.2f} ms", end="")
            else:
                print(f"{'N/A':>15}", end="")
        print()
    print("-" * 72)


def write_final_results(all_results: Dict[str, List[Dict]], output_dir: Path, output_prefix: str, model: str):
    for method, results in all_results.items():
        json_path = output_dir / f"{output_prefix}_{method}.json"
        with open(json_path, "w") as f:
            json.dump({"model": model, "method": method, "results": results}, f, indent=2)
        print(f"Results written to {json_path}")


def main():
    args = parse_args()

    compress_methods = (
        args.compress_methods.split(",") if args.compress_methods
        else DEFAULT_COMPRESS_METHODS
    )

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
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / f"{args.output_prefix}_checkpoint.json"
    checkpoint = None
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            print(f"[Resume] Found checkpoint, resuming from {checkpoint['current_method']}")

    print("=" * 72)
    print("S3-Only Benchmark with Incremental Saves")
    print("=" * 72)
    print(f"Model: {args.model}")
    print(f"Compression methods: {compress_methods}")
    print(f"Request rates: {request_rates}")
    print(f"Num contexts: {args.num_contexts}")
    print(f"Requests per rate: {args.num_requests}")
    print(f"Contexts per request: {ctx_min}-{ctx_max}")
    print(f"GPU: {args.gpu}, TP: {args.tp}")
    print(f"Output: {output_dir / args.output_prefix}*")
    print("=" * 72)

    print("\n[Dataset] Loading...")
    samples = load_dataset(args.dataset)
    contexts, questions = extract_contexts_and_questions(samples, args.num_contexts)
    print(f"[Dataset] {len(contexts)} unique contexts, {len(questions)} questions")

    all_results = checkpoint["all_results"] if checkpoint else {}
    start_method_idx = 0
    start_rate_idx = 0

    if checkpoint:
        for i, method in enumerate(compress_methods):
            if method == checkpoint["current_method"]:
                start_method_idx = i
                start_rate_idx = checkpoint["current_rate_idx"]
                break

    for method_idx, method in enumerate(compress_methods):
        if method_idx < start_method_idx:
            print(f"\n[Skip] {method} (already completed)")
            continue

        print(f"\n{'#'*72}")
        print(f"# Testing: {method}")
        print(f"{'#'*72}")

        print("[Server] Killing existing processes...")
        kill_server()

        proc = start_server(args.model, args.port, args.gpu, method, args.tp, use_local_cache=not args.no_local_cache)

        try:
            if not wait_for_server(args.port):
                print(f"[Error] Server failed to start for {method}")
                continue

            current_start_rate = start_rate_idx if method_idx == start_method_idx else 0

            results = asyncio.run(run_single_method_benchmark(
                model=args.model,
                port=args.port,
                contexts=contexts,
                questions=questions,
                request_rates=request_rates,
                num_requests=args.num_requests,
                contexts_range=(ctx_min, ctx_max),
                warmup_delay=args.warmup_delay,
                burstiness=args.burstiness,
                seed=args.seed,
                output_dir=output_dir,
                output_prefix=args.output_prefix,
                method=method,
                start_rate_idx=current_start_rate
            ))

            if method in all_results:
                all_results[method].extend(results)
            else:
                all_results[method] = results

            save_checkpoint(checkpoint_path, all_results, method, len(request_rates))

            print(f"\n{method} Results:")
            print(f"{'Rate':>10} {'TTFT(avg)':>15} {'Latency(avg)':>15} {'ITL(avg)':>12} {'Success%':>10}")
            print("-" * 65)
            for r in results:
                rate_str = f"{r['rate']}" if r['rate'] != float("inf") else "inf"
                ttft = f"{r['ttft_avg']:.2f} ms" if r['ttft_avg'] else "N/A"
                lat = f"{r['latency_avg']:.2f} ms" if r['latency_avg'] else "N/A"
                itl = f"{r['itl_avg']:.2f} ms" if r['itl_avg'] else "N/A"
                print(f"{rate_str:>10} {ttft:>15} {lat:>15} {itl:>12} {r['success_rate']:>9.1f}%")

        except Exception as e:
            print(f"[Error] Benchmark failed for {method}: {e}")
            import traceback
            traceback.print_exc()
            save_checkpoint(checkpoint_path, all_results, method, 0)
        finally:
            print("[Server] Stopping...")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except:
                pass
            kill_server()

    if all_results:
        print_summary_table(all_results)
        write_final_results(all_results, output_dir, args.output_prefix, args.model)

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("[Cleanup] Removed checkpoint file")

    print("\n[Done] Benchmark complete.")


if __name__ == "__main__":
    main()
