#!/usr/bin/env python3
"""
简单的 TTFT benchmark 脚本
测试单个压缩方法在不同 request rate 下的 TTFT
"""
import argparse
import asyncio
import json
import time
from statistics import mean

import aiohttp
import numpy as np
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--dataset", default="/home/xujie/TaoTie/dataset/needle_multi_rag.jsonl")
    parser.add_argument("--num-contexts", type=int, default=10)
    parser.add_argument("--num-requests", type=int, default=20)
    parser.add_argument("--contexts-per-request", type=int, default=3)
    parser.add_argument("--rates", default="1,2,4,8")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_contexts(path: str, limit: int):
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


async def send_request(session, url, model, prompt, timeout=120):
    """发送单个请求并返回 TTFT"""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 32,
        "temperature": 0,
        "stream": True,
    }
    start = time.perf_counter()
    try:
        async with session.post(
            f"{url}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as resp:
            resp.raise_for_status()
            async for chunk in resp.content:
                if not chunk:
                    continue
                text = chunk.decode()
                if '"choices"' in text:
                    ttft = time.perf_counter() - start
                    return ttft
    except Exception as e:
        print(f"  Error: {e}")
        return None
    return None


async def warmup(session, url, model, contexts, tokenizer, sep_tokens):
    """预热：存储所有 context 到缓存"""
    print(f"Warmup: storing {len(contexts)} contexts...")
    for i, ctx in enumerate(contexts):
        tokens = sep_tokens + tokenizer.encode(ctx, add_special_tokens=False)
        ttft = await send_request(session, url, model, tokens)
        status = f"{ttft*1000:.0f}ms" if ttft else "failed"
        print(f"  [{i+1}/{len(contexts)}] {status}")
        await asyncio.sleep(0.2)
    print("Warmup done.\n")


async def benchmark_rate(session, url, model, prompts, rate: float):
    """在指定 rate 下发送请求并收集 TTFT"""
    tasks = []

    async def schedule():
        for i, prompt in enumerate(prompts):
            task = asyncio.create_task(send_request(session, url, model, prompt))
            tasks.append(task)
            if rate < float('inf') and i < len(prompts) - 1:
                interval = np.random.exponential(1.0 / rate)
                await asyncio.sleep(interval)

    await schedule()
    results = await asyncio.gather(*tasks)

    ttfts = [r for r in results if r is not None]
    return ttfts


def build_prompts(contexts, tokenizer, sep_tokens, num_requests, ctx_per_req, seed):
    """构建测试请求的 prompt"""
    rng = np.random.default_rng(seed)
    prompts = []
    question_tokens = tokenizer.encode("Answer the question.", add_special_tokens=False)

    for _ in range(num_requests):
        selected = rng.choice(contexts, size=min(ctx_per_req, len(contexts)), replace=False)
        tokens = []
        for ctx in selected:
            tokens.extend(sep_tokens)
            tokens.extend(tokenizer.encode(ctx, add_special_tokens=False))
        tokens.extend(sep_tokens)
        tokens.extend(question_tokens)
        prompts.append(tokens)

    return prompts


async def main():
    args = parse_args()
    url = f"http://localhost:{args.port}"
    rates = [float(r) for r in args.rates.split(",")]

    print("=" * 60)
    print("Simple TTFT Benchmark")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Port: {args.port}")
    print(f"Rates: {rates}")
    print(f"Requests per rate: {args.num_requests}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sep_tokens = tokenizer.encode(" # # ", add_special_tokens=False)

    contexts = load_contexts(args.dataset, args.num_contexts)
    print(f"Loaded {len(contexts)} contexts\n")

    prompts = build_prompts(
        contexts, tokenizer, sep_tokens,
        args.num_requests, args.contexts_per_request, args.seed
    )

    connector = aiohttp.TCPConnector(limit=100)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup
        await warmup(session, url, args.model, contexts, tokenizer, sep_tokens)
        await asyncio.sleep(3)

        # Benchmark each rate
        results = {}
        for rate in rates:
            print(f"Testing rate={rate} RPS...")
            ttfts = await benchmark_rate(session, url, args.model, prompts, rate)

            if ttfts:
                avg_ttft = mean(ttfts) * 1000
                success = len(ttfts)
                print(f"  TTFT: {avg_ttft:.1f}ms, Success: {success}/{args.num_requests}")
                results[rate] = avg_ttft
            else:
                print(f"  All requests failed!")
                results[rate] = None

            await asyncio.sleep(2)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Rate (RPS)':<12} {'Avg TTFT (ms)':<15}")
    print("-" * 30)
    for rate, ttft in results.items():
        ttft_str = f"{ttft:.1f}" if ttft else "N/A"
        print(f"{rate:<12} {ttft_str:<15}")


if __name__ == "__main__":
    asyncio.run(main())
