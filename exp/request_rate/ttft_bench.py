#!/usr/bin/env python3
"""
TTFT Benchmark: 测试 KIVI_2BIT 和 OURS 方法在不同 request rate 下的 TTFT
"""
import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, stdev

import aiohttp
import numpy as np
from transformers import AutoTokenizer

# 配置
MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
PORT = 12345
DATASET = "/home/xujie/TaoTie/dataset/needle_multi_rag.jsonl"
SEP_STR = " # # "


def load_contexts(path: str, limit: int = 10):
    """加载数据集中的 context"""
    contexts = []
    seen = set()
    with open(path) as f:
        for line in f:
            sample = json.loads(line.strip())
            for ctx in sample.get("contexts", []):
                if ctx not in seen:
                    seen.add(ctx)
                    contexts.append(ctx)
                    if len(contexts) >= limit:
                        return contexts
    return contexts


async def send_request(session, prompt, timeout=120):
    """发送单个请求，返回 TTFT"""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 64,
        "temperature": 0,
        "stream": True,
    }
    start = time.perf_counter()
    try:
        async with session.post(
            f"http://localhost:{PORT}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as resp:
            if resp.status != 200:
                return None, f"HTTP {resp.status}"
            async for chunk in resp.content:
                if chunk and b'"choices"' in chunk:
                    return time.perf_counter() - start, None
    except Exception as e:
        return None, str(e)
    return None, "no response"


async def warmup(session, contexts, tokenizer, sep_tokens):
    """预热：存储所有 context"""
    print(f"\n[Warmup] Storing {len(contexts)} contexts...")
    success = 0
    for i, ctx in enumerate(contexts):
        tokens = sep_tokens + tokenizer.encode(ctx, add_special_tokens=False)
        ttft, err = await send_request(session, tokens)
        if ttft:
            success += 1
            print(f"  [{i+1}/{len(contexts)}] OK ({ttft*1000:.0f}ms)")
        else:
            print(f"  [{i+1}/{len(contexts)}] FAIL: {err}")
        await asyncio.sleep(0.3)
    print(f"[Warmup] Done: {success}/{len(contexts)} succeeded\n")
    return success == len(contexts)


async def benchmark_rate(session, prompts, rate: float):
    """测试指定 rate 下的 TTFT"""
    tasks = []

    async def schedule():
        for i, prompt in enumerate(prompts):
            task = asyncio.create_task(send_request(session, prompt))
            tasks.append(task)
            if rate < float('inf') and i < len(prompts) - 1:
                await asyncio.sleep(np.random.exponential(1.0 / rate))

    await schedule()
    results = await asyncio.gather(*tasks)

    ttfts = [r[0] for r in results if r[0] is not None]
    errors = [r[1] for r in results if r[1] is not None]

    return ttfts, errors


def build_prompts(contexts, tokenizer, sep_tokens, num_requests=20, ctx_per_req=3, seed=42):
    """构建测试 prompt"""
    rng = np.random.default_rng(seed)
    prompts = []
    question = tokenizer.encode("Answer the question.", add_special_tokens=False)

    for _ in range(num_requests):
        selected = rng.choice(contexts, size=min(ctx_per_req, len(contexts)), replace=False)
        tokens = []
        for ctx in selected:
            tokens.extend(sep_tokens)
            tokens.extend(tokenizer.encode(ctx, add_special_tokens=False))
        tokens.extend(sep_tokens)
        tokens.extend(question)
        prompts.append(tokens)

    return prompts


def kill_server():
    """杀掉旧服务器"""
    subprocess.run(["pkill", "-9", "-f", "vllm"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "EngineCore"], capture_output=True)
    time.sleep(3)


def start_server(compress_type: str, gpu: int = 1):
    """启动服务器"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["HF_HUB_OFFLINE"] = "1"
    env["LMCACHE_COMPRESS_TYPE"] = compress_type
    env["LMCACHE_CHUNK_SIZE"] = "256"
    env["LMCACHE_ENABLE_BLENDING"] = "True"
    env["LMCACHE_BLEND_SPECIAL_STR"] = SEP_STR
    env["LMCACHE_USE_LAYERWISE"] = "True"
    env["LMCACHE_BLEND_CHECK_LAYERS"] = "1"
    env["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = "0.15"
    env["LMCACHE_BLEND_MIN_TOKENS"] = "64"
    env["LMCACHE_LOCAL_CPU"] = "True"
    env["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"

    cmd = [
        "vllm", "serve", MODEL,
        "--kv-transfer-config", '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}',
        "--port", str(PORT),
        "--gpu-memory-utilization", "0.6",
        "--max-model-len", "32000",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--enforce-eager",
        "-tp", "1"
    ]

    log_dir = Path(__file__).parent / "server_logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"vllm_{compress_type}_{PORT}.log"

    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)

    print(f"[Server] Starting with {compress_type}, log: {log_file}")
    return proc


def wait_server(timeout=180):
    """等待服务器就绪"""
    import urllib.request
    import urllib.error

    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    print("[Server] Ready!")
                    return True
        except:
            pass
        time.sleep(2)
    print("[Server] Timeout!")
    return False


async def run_benchmark(compress_type: str, rates: list, num_contexts=10, num_requests=20, ctx_per_req=3, gpu=1):
    """运行单个压缩方法的 benchmark"""
    print(f"\n{'='*60}")
    print(f"Testing: {compress_type}")
    print(f"{'='*60}")

    # 启动服务器
    kill_server()
    proc = start_server(compress_type, gpu)

    try:
        if not wait_server():
            return None

        # 加载数据
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        sep_tokens = tokenizer.encode(SEP_STR, add_special_tokens=False)
        contexts = load_contexts(DATASET, num_contexts)
        print(f"[Data] Loaded {len(contexts)} contexts")

        # 构建 prompt
        prompts = build_prompts(contexts, tokenizer, sep_tokens, num_requests, ctx_per_req)
        print(f"[Data] Built {len(prompts)} prompts (avg tokens: {np.mean([len(p) for p in prompts]):.0f})")

        connector = aiohttp.TCPConnector(limit=100)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Warmup
            if not await warmup(session, contexts, tokenizer, sep_tokens):
                print("[Error] Warmup failed!")
                return None

            await asyncio.sleep(5)  # 等待缓存稳定

            # 测试各个 rate
            results = {}
            for rate in rates:
                print(f"\n[Test] Rate = {rate} RPS...")
                ttfts, errors = await benchmark_rate(session, prompts, rate)

                if ttfts:
                    avg = mean(ttfts) * 1000
                    std = stdev(ttfts) * 1000 if len(ttfts) > 1 else 0
                    results[rate] = {"avg": avg, "std": std, "success": len(ttfts), "total": len(prompts)}
                    print(f"  TTFT: {avg:.1f} ± {std:.1f} ms, Success: {len(ttfts)}/{len(prompts)}")
                else:
                    results[rate] = {"avg": None, "success": 0, "total": len(prompts)}
                    print(f"  All failed! Errors: {errors[:3]}")

                await asyncio.sleep(2)

            return results

    finally:
        # 清理
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except:
            pass
        kill_server()


def print_comparison(all_results: dict):
    """打印对比结果"""
    print(f"\n{'='*70}")
    print("COMPARISON TABLE (TTFT in ms)")
    print(f"{'='*70}")

    # Header
    methods = list(all_results.keys())
    print(f"{'Rate':>8}", end="")
    for m in methods:
        print(f"{m:>20}", end="")
    print()
    print("-" * 70)

    # 获取所有 rate
    rates = set()
    for results in all_results.values():
        if results:
            rates.update(results.keys())

    for rate in sorted(rates):
        print(f"{rate:>8}", end="")
        for m in methods:
            if all_results[m] and rate in all_results[m]:
                r = all_results[m][rate]
                if r["avg"]:
                    print(f"{r['avg']:>15.1f} ms", end="")
                else:
                    print(f"{'FAIL':>18}", end="")
            else:
                print(f"{'N/A':>18}", end="")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="KIVI_2BIT,OURS", help="Compression methods to test")
    parser.add_argument("--rates", default="1,2,4,8", help="Request rates to test")
    parser.add_argument("--num-contexts", type=int, default=10)
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--ctx-per-req", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    methods = args.methods.split(",")
    rates = [float(r) for r in args.rates.split(",")]

    print("="*60)
    print("TTFT Benchmark: KIVI vs OURS")
    print("="*60)
    print(f"Methods: {methods}")
    print(f"Rates: {rates}")
    print(f"Contexts: {args.num_contexts}, Requests: {args.num_requests}, Ctx/Req: {args.ctx_per_req}")

    all_results = {}
    for method in methods:
        results = asyncio.run(run_benchmark(
            method, rates,
            num_contexts=args.num_contexts,
            num_requests=args.num_requests,
            ctx_per_req=args.ctx_per_req,
            gpu=args.gpu
        ))
        all_results[method] = results

    print_comparison(all_results)

    # 保存结果
    output_file = Path(__file__).parent / "ttft_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
