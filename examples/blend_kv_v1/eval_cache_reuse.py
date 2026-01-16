#!/usr/bin/env python3
"""Evaluate KV cache reuse across dataset samples."""
import argparse
import asyncio
import json
import math
import sys
from pathlib import Path
from statistics import mean, median

import aiohttp
from transformers import AutoTokenizer

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import test_large_prompt as helpers  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval KV cache reuse efficiency on a dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model name for the backend.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/wikimqa_s.jsonl",
        help="Path to JSONL dataset.",
    )
    parser.add_argument(
        "--precompute-contexts",
        action="store_true",
        help="Send unique contexts ahead of time for warmup.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Request rate (RPS). Use inf for unlimited.",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor for rate limiting.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to process. Default uses all samples.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:8000",
        help="Base URL for the API service.",
    )
    return parser.parse_args()


def load_dataset(path, limit=None):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if limit is not None and len(samples) >= limit:
                break
    return samples


def build_prompt(sample, tokenizer, sep_tokens):
    """Build tokenized prompt with separator tokens between contexts."""
    contexts = sample.get("contexts") or []
    question = sample.get("question", "")
    # 格式: sep + ctx1 + sep + ctx2 + sep + ... + sep + question
    tokens = []
    for ctx in contexts:
        tokens.extend(sep_tokens)
        tokens.extend(tokenizer.encode(ctx, add_special_tokens=False))
    tokens.extend(sep_tokens)
    tokens.extend(tokenizer.encode(question, add_special_tokens=False))
    return tokens


def unique_contexts(samples):
    ordered = []
    seen = set()
    for sample in samples:
        for ctx in sample.get("contexts") or []:
            if ctx not in seen:
                seen.add(ctx)
                ordered.append(ctx)
    return ordered


def percentile(sorted_values, pct):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * pct
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return sorted_values[int(k)]
    weight = k - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize(values):
    if not values:
        return {"avg": None, "median": None, "p50": None, "p90": None, "p99": None}
    sorted_values = sorted(values)
    return {
        "avg": mean(sorted_values),
        "median": median(sorted_values),
        "p50": percentile(sorted_values, 0.5),
        "p90": percentile(sorted_values, 0.9),
        "p99": percentile(sorted_values, 0.99),
    }


def print_summary(title, stats, formatter=str):
    print(f"{title}:")
    print(
        f"  Avg: {formatter(stats['avg'])} | Median: {formatter(stats['median'])} | "
        f"P50: {formatter(stats['p50'])} | P90: {formatter(stats['p90'])} | "
        f"P99: {formatter(stats['p99'])}"
    )


def format_count(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.2f}"
    return str(value)


def print_request_metrics(name, sample, metrics):
    print("-" * 60)
    print(
        f"{name} | id={sample.get('id')} | "
        f"contexts={len(sample.get('contexts') or [])}"
    )
    print(f"Question: {sample.get('question', '')}")
    print(f"Answer: {sample.get('answer', [])}")
    print(f"生成内容: {metrics['generated_text']}")
    if not metrics["success"]:
        print("警告: 未收到任何含choices的chunk，以下指标可能无效。")
    ttft = helpers._format_ms(metrics["ttft"])
    latency = helpers._format_ms(metrics["latency"])
    if metrics["itl"]:
        mean_itl = helpers._format_ms(mean(metrics["itl"]))
        median_itl = helpers._format_ms(median(metrics["itl"]))
    else:
        mean_itl = helpers._format_ms(None)
        median_itl = helpers._format_ms(None)
    print(f"TTFT: {ttft} | Latency: {latency}")
    print(f"Mean ITL: {mean_itl} | Median ITL: {median_itl}")
    prompt_tokens = metrics.get("prompt_tokens")
    completion_tokens = metrics.get("completion_tokens")
    print(
        f"Prompt tokens: {prompt_tokens if prompt_tokens is not None else 'N/A'} | "
        f"Completion tokens: "
        f"{completion_tokens if completion_tokens is not None else 'N/A'}"
    )


async def precompute_contexts(session, contexts, tokenizer, sep_tokens):
    """预热：每个 context 用纯内容 token 发送（不带 separator）。

    原因：SegmentTokenDatabase 分段时返回的是不含分隔符的纯 context，
    taotie_blender 计算 hash 时也是基于不含分隔符的 tokens[start:end]。
    因此 warmup 存储时也必须用纯 context 内容，hash 才能匹配。
    """
    total = len(contexts)
    if total == 0:
        print("无需预计算：无可用上下文。")
        return
    print("=" * 60)
    print(f"开始预计算contexts，总计{total}条unique contexts")
    print("=" * 60)
    for idx, ctx in enumerate(contexts, start=1):
        # 只发送纯 context 内容（不含分隔符），与检索时的 hash 计算方式一致
        tokens = tokenizer.encode(ctx, add_special_tokens=False)

        request_name = f"[预热 {idx}/{total}]"
        print(f"{request_name} 开始 (tokens={len(tokens)})")
        await helpers.stream_completion(session=session, prompt=tokens, request_name=request_name)
        print(f"{request_name} 完成")
    print("预计算完成，等待2秒确保服务稳定...")
    await asyncio.sleep(2)


async def run_requests(session, samples, request_rate, burstiness, tokenizer, sep_tokens):
    entries = []
    for idx, sample in enumerate(samples, start=1):
        prompt = build_prompt(sample, tokenizer, sep_tokens)
        name = f"[样本 {idx}]"
        entries.append((name, prompt, sample))

    tasks = []
    async for item in helpers.send_requests_with_rate_limit(
        entries, request_rate, burstiness
    ):
        name, prompt, sample = item
        print(f"{name} 发送请求")
        task = asyncio.create_task(
            helpers.stream_completion(
                session=session, prompt=prompt, request_name=name
            )
        )
        tasks.append((name, sample, task))

    metrics_list = await asyncio.gather(*(t for _, _, t in tasks))
    results = []
    for (name, sample, _), metrics in zip(tasks, metrics_list):
        results.append((name, sample, metrics))
    return results


def aggregate_results(results):
    ttfts = []
    latencies = []
    itls = []
    prompt_tokens = []
    completion_tokens = []
    for _, _, metrics in results:
        if metrics.get("ttft") is not None:
            ttfts.append(metrics["ttft"])
        if metrics.get("latency") is not None:
            latencies.append(metrics["latency"])
        itls.extend(metrics.get("itl") or [])
        if metrics.get("prompt_tokens") is not None:
            prompt_tokens.append(metrics["prompt_tokens"])
        if metrics.get("completion_tokens") is not None:
            completion_tokens.append(metrics["completion_tokens"])
    return {
        "ttft": summarize(ttfts),
        "latency": summarize(latencies),
        "itl": summarize(itls),
        "prompt_tokens": summarize(prompt_tokens),
        "completion_tokens": summarize(completion_tokens),
    }


async def main():
    args = parse_args()
    helpers.api_base = args.api_base
    helpers.model = args.model

    # Initialize tokenizer and separator tokens
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sep_tokens = tokenizer.encode(" # # ", add_special_tokens=False)
    print(f"Separator tokens: {sep_tokens}")

    samples = load_dataset(args.dataset, args.num_samples)
    if not samples:
        print("未加载到任何样本，退出。")
        return

    print("=" * 60)
    print("Blend KV Cache Reuse Eval")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Loaded samples: {len(samples)}")
    print(f"Request rate (RPS): {args.request_rate}")
    print(f"Burstiness: {args.burstiness}")
    print(f"API Base: {args.api_base}")
    print(f"预计算contexts: {args.precompute_contexts}")
    print()

    async with aiohttp.ClientSession() as session:
        if args.precompute_contexts:
            contexts = unique_contexts(samples)
            await precompute_contexts(session, contexts, tokenizer, sep_tokens)

        print("=" * 60)
        print("开始正式测试")
        print("=" * 60)
        results = await run_requests(
            session, samples, args.request_rate, args.burstiness, tokenizer, sep_tokens
        )

    print()
    for name, sample, metrics in results:
        print_request_metrics(name, sample, metrics)

    print()
    print("=" * 60)
    print("汇总统计")
    print("=" * 60)
    summary = aggregate_results(results)
    print_summary("TTFT", summary["ttft"], helpers._format_ms)
    print_summary("Latency", summary["latency"], helpers._format_ms)
    print_summary("ITL", summary["itl"], helpers._format_ms)
    print_summary("Prompt tokens", summary["prompt_tokens"], format_count)
    print_summary("Completion tokens", summary["completion_tokens"], format_count)


if __name__ == "__main__":
    asyncio.run(main())
