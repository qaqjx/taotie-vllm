#!/usr/bin/env python3
"""
Full benchmark: multiple models x multiple compression methods x multiple request rates.
Results are written to a CSV file.
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
from typing import Dict, List, Sequence, Tuple

import aiohttp
from transformers import AutoTokenizer

CURRENT_DIR = Path(__file__).resolve().parent
BLEND_DIR = CURRENT_DIR.parent
if str(BLEND_DIR) not in sys.path:
    sys.path.insert(0, str(BLEND_DIR))

import test_large_prompt as helpers

# Configuration
MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

COMPRESS_METHODS = ["OURS", "KIVI_2BIT", "SVDQ", "NONE"]

REQUEST_RATES = [1, 2, 4, 8, float("inf")]

KVMANAGER_PATH = Path("/home/xujie/lmcache-v1/lmcache/v1/compute/blend/kvmanager.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full benchmark across models and compression methods")
    parser.add_argument("--dataset", type=str,
                        default="/home/xujie/TaoTie/dataset/needle_multi_rag.jsonl",
                        help="Path to dataset")
    parser.add_argument("--num-contexts", type=int, default=100,
                        help="Number of unique contexts to use from dataset")
    parser.add_argument("--num-requests", type=int, default=20,
                        help="Number of requests per rate")
    parser.add_argument("--min-prompt-tokens", type=int, default=10000,
                        help="Minimum prompt length in tokens")
    parser.add_argument("--warmup-delay", type=float, default=15.0,
                        help="Seconds to wait after warmup for compression to complete")
    parser.add_argument("--port", type=int, default=12345,
                        help="Server port")
    parser.add_argument("--output-csv", type=str, default="benchmark_results.csv",
                        help="Output CSV file path")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated list of models to test (default: all)")
    parser.add_argument("--compress-methods", type=str, default=None,
                        help="Comma-separated list of compression methods (default: all)")
    parser.add_argument("--request-rates", type=str, default=None,
                        help="Comma-separated list of request rates (default: 1,2,4,8,inf)")
    return parser.parse_args()


def load_dataset(path: str) -> List[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def extract_unique_contexts(samples: List[dict], limit: int) -> List[str]:
    """Extract unique contexts from dataset, up to limit."""
    contexts = []
    seen = set()
    for sample in samples:
        for ctx in sample.get("contexts", []):
            if ctx not in seen:
                seen.add(ctx)
                contexts.append(ctx)
                if len(contexts) >= limit:
                    return contexts
    return contexts


def extract_questions(samples: List[dict]) -> List[str]:
    """Extract questions from dataset."""
    questions = []
    for sample in samples:
        q = sample.get("question", "").strip()
        if q:
            questions.append(q)
    return questions


def set_compression_type(compress_type: str) -> None:
    """Modify kvmanager.py to use specified compression type."""
    content = KVMANAGER_PATH.read_text()

    # Find and replace the compress_type default parameter
    import re
    pattern = r'compress_type=CompressType\.\w+'
    replacement = f'compress_type=CompressType.{compress_type}'

    new_content = re.sub(pattern, replacement, content)
    KVMANAGER_PATH.write_text(new_content)
    print(f"[Config] Set compression type to: {compress_type}")


def kill_server():
    """Kill any running vLLM server processes."""
    subprocess.run(["pkill", "-9", "-f", "EngineCore"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "APIServer"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "vllm serve"], capture_output=True)
    time.sleep(3)


def start_server(model: str, port: int) -> subprocess.Popen:
    """Start vLLM server with specified model."""
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["MODEL"] = model
    env["PORT"] = str(port)
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
        "-tp", "1"
    ]

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )
    return proc


def wait_for_server(port: int, timeout: int = 180) -> bool:
    """Wait for server to be ready."""
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


def build_prompt_tokens(
    contexts: Sequence[str],
    question: str,
    tokenizer,
    sep_tokens: List[int]
) -> List[int]:
    """Build prompt tokens from contexts and question."""
    tokens = []
    for ctx in contexts:
        tokens.extend(sep_tokens)
        tokens.extend(tokenizer.encode(ctx, add_special_tokens=False))
    tokens.extend(sep_tokens)
    tokens.extend(tokenizer.encode(question, add_special_tokens=False))
    return tokens


def select_contexts_for_prompt(
    all_contexts: List[str],
    tokenizer,
    sep_tokens: List[int],
    min_tokens: int,
    rng: random.Random
) -> List[str]:
    """Select contexts to reach minimum token count."""
    selected = []
    total_tokens = 0

    # Shuffle contexts for randomness
    shuffled = all_contexts.copy()
    rng.shuffle(shuffled)

    for ctx in shuffled:
        ctx_tokens = len(tokenizer.encode(ctx, add_special_tokens=False)) + len(sep_tokens)
        selected.append(ctx)
        total_tokens += ctx_tokens
        if total_tokens >= min_tokens:
            break

    # If we still don't have enough, repeat contexts
    while total_tokens < min_tokens and all_contexts:
        ctx = rng.choice(all_contexts)
        ctx_tokens = len(tokenizer.encode(ctx, add_special_tokens=False)) + len(sep_tokens)
        selected.append(ctx)
        total_tokens += ctx_tokens

    return selected


async def warmup_contexts(
    session: aiohttp.ClientSession,
    contexts: List[str],
    tokenizer,
    sep_tokens: List[int]
) -> Tuple[int, int]:
    """Warmup by storing all contexts individually with delays."""
    successes = 0
    failures = 0
    total = len(contexts)
    max_retries = 3

    print(f"[Warmup] Storing {total} unique contexts...")

    for idx, ctx in enumerate(contexts, start=1):
        tokens = tokenizer.encode(ctx, add_special_tokens=False)
        # Add separator at beginning to match how it will be used
        full_tokens = sep_tokens + tokens
        request_name = f"[Warmup {idx}/{total}]"

        for attempt in range(max_retries):
            try:
                await helpers.stream_completion(
                    session=session,
                    prompt=full_tokens,
                    request_name=request_name
                )
                successes += 1
                if idx % 10 == 0 or idx == total:
                    print(f"[Warmup] Progress: {idx}/{total}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[Warmup] {request_name} retry {attempt+1}/{max_retries}")
                    await asyncio.sleep(2)
                else:
                    print(f"[Warmup] {request_name} failed after {max_retries} attempts: {e}")
                    failures += 1

        # Small delay between warmup requests to avoid overloading
        await asyncio.sleep(0.5)

    print(f"[Warmup] Completed. Successes: {successes}, Failures: {failures}")
    return successes, failures


async def run_benchmark_requests(
    session: aiohttp.ClientSession,
    prompts: List[List[int]],
    request_rate: float,
    burstiness: float = 1.0
) -> Dict:
    """Run benchmark requests at specified rate."""
    entries = []
    for i, prompt in enumerate(prompts, start=1):
        rate_str = "inf" if request_rate == float("inf") else f"{request_rate}"
        request_name = f"[rate={rate_str}] req={i}"
        entries.append((request_name, prompt, {"ordinal": i}))

    tasks = []
    async for request_name, prompt, data in helpers.send_requests_with_rate_limit(
        entries, request_rate, burstiness
    ):
        task = asyncio.create_task(
            helpers.stream_completion(session=session, prompt=prompt, request_name=request_name)
        )
        tasks.append((request_name, task))

    gathered = await asyncio.gather(*(t for _, t in tasks), return_exceptions=True)

    ttfts = []
    latencies = []
    itls = []
    successes = 0

    for (request_name, _), result in zip(tasks, gathered):
        if isinstance(result, Exception):
            print(f"[Benchmark] {request_name} error: {result}")
        else:
            if result.get("success"):
                successes += 1
                if result.get("ttft") is not None:
                    ttfts.append(result["ttft"])
                if result.get("latency") is not None:
                    latencies.append(result["latency"])
                if result.get("itl"):
                    itls.extend(result["itl"])

    return {
        "ttft_avg": mean(ttfts) if ttfts else None,
        "ttft_p50": sorted(ttfts)[len(ttfts)//2] if ttfts else None,
        "ttft_p99": sorted(ttfts)[int(len(ttfts)*0.99)] if len(ttfts) > 1 else (ttfts[0] if ttfts else None),
        "latency_avg": mean(latencies) if latencies else None,
        "itl_avg": mean(itls) if itls else None,
        "success_rate": successes / len(prompts) if prompts else 0,
        "total_requests": len(prompts),
        "successful_requests": successes,
    }


async def run_single_benchmark(
    model: str,
    compress_method: str,
    port: int,
    contexts: List[str],
    questions: List[str],
    request_rates: List[float],
    num_requests: int,
    min_prompt_tokens: int,
    warmup_delay: float,
    seed: int,
) -> List[Dict]:
    """Run benchmark for a single model+compression combination."""

    results = []

    # Load tokenizer
    print(f"[Benchmark] Loading tokenizer for {model}...")
    tokenizer = AutoTokenizer.from_pretrained(model)
    sep_tokens = tokenizer.encode(" # # ", add_special_tokens=False)

    # Setup helpers
    helpers.api_base = f"http://localhost:{port}"
    helpers.model = model

    async with aiohttp.ClientSession() as session:
        # Warmup
        await warmup_contexts(session, contexts, tokenizer, sep_tokens)

        # Wait for compression to complete
        print(f"[Benchmark] Waiting {warmup_delay}s for compression to complete...")
        await asyncio.sleep(warmup_delay)

        # Run benchmarks for each request rate
        for rate in request_rates:
            rate_str = "inf" if rate == float("inf") else f"{rate}"
            print(f"\n[Benchmark] Running rate={rate_str} RPS")

            # Prepare prompts
            rng = random.Random(seed)
            prompts = []
            prompt_lengths = []

            for _ in range(num_requests):
                selected_ctxs = select_contexts_for_prompt(
                    contexts, tokenizer, sep_tokens, min_prompt_tokens, rng
                )
                question = rng.choice(questions) if questions else "Please answer the question based on the context."
                prompt = build_prompt_tokens(selected_ctxs, question, tokenizer, sep_tokens)
                prompts.append(prompt)
                prompt_lengths.append(len(prompt))

            avg_prompt_len = mean(prompt_lengths)
            print(f"[Benchmark] Average prompt length: {avg_prompt_len:.0f} tokens")

            # Run benchmark
            metrics = await run_benchmark_requests(session, prompts, rate)

            results.append({
                "model": model,
                "compress_method": compress_method,
                "request_rate": rate_str,
                "ttft_avg_ms": metrics["ttft_avg"] * 1000 if metrics["ttft_avg"] else None,
                "ttft_p50_ms": metrics["ttft_p50"] * 1000 if metrics["ttft_p50"] else None,
                "ttft_p99_ms": metrics["ttft_p99"] * 1000 if metrics["ttft_p99"] else None,
                "latency_avg_ms": metrics["latency_avg"] * 1000 if metrics["latency_avg"] else None,
                "itl_avg_ms": metrics["itl_avg"] * 1000 if metrics["itl_avg"] else None,
                "success_rate": metrics["success_rate"],
                "avg_prompt_tokens": avg_prompt_len,
                "num_requests": num_requests,
            })

            print(f"[Benchmark] rate={rate_str}: TTFT_avg={metrics['ttft_avg']*1000:.2f}ms" if metrics['ttft_avg'] else f"[Benchmark] rate={rate_str}: TTFT_avg=N/A")

    return results


def write_csv(results: List[Dict], output_path: str):
    """Write results to CSV file."""
    if not results:
        print("[CSV] No results to write")
        return

    fieldnames = [
        "model", "compress_method", "request_rate",
        "ttft_avg_ms", "ttft_p50_ms", "ttft_p99_ms",
        "latency_avg_ms", "itl_avg_ms", "success_rate",
        "avg_prompt_tokens", "num_requests"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"[CSV] Results written to {output_path}")


def main():
    args = parse_args()

    # Parse command line overrides
    models = args.models.split(",") if args.models else MODELS
    compress_methods = args.compress_methods.split(",") if args.compress_methods else COMPRESS_METHODS

    if args.request_rates:
        request_rates = []
        for r in args.request_rates.split(","):
            r = r.strip()
            if r.lower() == "inf":
                request_rates.append(float("inf"))
            else:
                request_rates.append(float(r))
    else:
        request_rates = REQUEST_RATES

    print("=" * 70)
    print("Full Benchmark: Models x Compression Methods x Request Rates")
    print("=" * 70)
    print(f"Models: {models}")
    print(f"Compression methods: {compress_methods}")
    print(f"Request rates: {request_rates}")
    print(f"Dataset: {args.dataset}")
    print(f"Num contexts: {args.num_contexts}")
    print(f"Num requests per rate: {args.num_requests}")
    print(f"Min prompt tokens: {args.min_prompt_tokens}")
    print(f"Output CSV: {args.output_csv}")
    print("=" * 70)

    # Load dataset
    print("\n[Dataset] Loading...")
    samples = load_dataset(args.dataset)
    contexts = extract_unique_contexts(samples, args.num_contexts)
    questions = extract_questions(samples)
    print(f"[Dataset] Loaded {len(samples)} samples, {len(contexts)} unique contexts, {len(questions)} questions")

    all_results = []

    # Run benchmarks for each combination
    total_combinations = len(models) * len(compress_methods)
    current = 0

    for model in models:
        for compress_method in compress_methods:
            current += 1
            print("\n" + "=" * 70)
            print(f"[{current}/{total_combinations}] Model: {model}, Compression: {compress_method}")
            print("=" * 70)

            # Set compression type
            set_compression_type(compress_method)

            # Kill any existing server
            print("[Server] Killing existing processes...")
            kill_server()

            # Start server
            print(f"[Server] Starting with model {model}...")
            proc = start_server(model, args.port)

            try:
                # Wait for server to be ready
                if not wait_for_server(args.port):
                    print(f"[Error] Server failed to start for {model}")
                    continue

                # Run benchmark
                results = asyncio.run(run_single_benchmark(
                    model=model,
                    compress_method=compress_method,
                    port=args.port,
                    contexts=contexts,
                    questions=questions,
                    request_rates=request_rates,
                    num_requests=args.num_requests,
                    min_prompt_tokens=args.min_prompt_tokens,
                    warmup_delay=args.warmup_delay,
                    seed=args.seed,
                ))

                all_results.extend(results)

                # Write intermediate results
                write_csv(all_results, args.output_csv)

            except Exception as e:
                print(f"[Error] Benchmark failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Stop server
                print("[Server] Stopping...")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except:
                    pass
                kill_server()

    # Final summary
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Total results: {len(all_results)}")
    print(f"Output file: {args.output_csv}")

    # Print summary table
    if all_results:
        print("\nSummary (TTFT avg in ms):")
        print("-" * 70)
        header = f"{'Model':<40} {'Compress':<10} {'Rate':<8} {'TTFT_avg':>10}"
        print(header)
        print("-" * 70)
        for r in all_results:
            model_short = r['model'].split('/')[-1][:35]
            ttft = f"{r['ttft_avg_ms']:.1f}" if r['ttft_avg_ms'] else "N/A"
            print(f"{model_short:<40} {r['compress_method']:<10} {r['request_rate']:<8} {ttft:>10}")


if __name__ == "__main__":
    main()
