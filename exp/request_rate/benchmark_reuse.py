#!/usr/bin/env python3
"""Benchmark cache reuse sensitivity to request rate via blended contexts."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Sequence, Tuple

import aiohttp
from transformers import AutoTokenizer

CURRENT_DIR = Path(__file__).resolve().parent
BLEND_DIR = CURRENT_DIR.parent
if str(BLEND_DIR) not in sys.path:
    sys.path.insert(0, str(BLEND_DIR))

import test_large_prompt as helpers  # noqa: E402


def parse_args() -> argparse.Namespace:
    default_dataset = str(BLEND_DIR / "data" / "wikimqa_s.jsonl")
    parser = argparse.ArgumentParser(
        description="Benchmark KV cache reuse when contexts are randomly blended."
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000")
    parser.add_argument(
        "--dataset",
        type=str,
        default=default_dataset,
        help="Path to wikimqa-style JSONL dataset.",
    )
    parser.add_argument(
        "--num-contexts",
        type=int,
        default=10,
        help="Number of unique contexts to pre-store in cache.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=50,
        help="Requests to send per request rate.",
    )
    parser.add_argument(
        "--contexts-per-request",
        type=str,
        default="2,4",
        help="Range 'min,max' for contexts blended per request.",
    )
    parser.add_argument(
        "--request-rates",
        type=str,
        default="1,2,4,8,16",
        help="Comma separated list of request rates (RPS).",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Gamma-shape burstiness factor for helpers.send_requests_with_rate_limit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for sampling contexts/questions.",
    )
    parser.add_argument(
        "--warmup-delay",
        type=float,
        default=2.0,
        help="Seconds to wait after warmup to let cache settle.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip warmup requests (cache must already contain contexts).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="benchmark_reuse_results.json",
        help="Where to store aggregated metrics as JSON.",
    )
    return parser.parse_args()


def parse_request_rates(raw: str) -> List[float]:
    rates: List[float] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.lower() == "inf":
            rates.append(float("inf"))
        else:
            rates.append(float(chunk))
    if not rates:
        raise ValueError("At least one request rate must be specified.")
    return rates


def parse_context_range(raw: str) -> Tuple[int, int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("contexts-per-request must be in 'min,max' format.")
    try:
        lo, hi = (int(parts[0]), int(parts[1]))
    except ValueError as exc:  # pragma: no cover - sanity guard
        raise ValueError("contexts-per-request values must be integers") from exc
    if lo <= 0 or hi <= 0:
        raise ValueError("contexts-per-request bounds must be > 0")
    if lo > hi:
        raise ValueError("contexts-per-request lower bound cannot exceed upper bound")
    return lo, hi


def load_dataset(path: str, limit: int | None = None) -> List[dict]:
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


def unique_contexts(samples: Iterable[dict]) -> List[str]:
    ordered = []
    seen = set()
    for sample in samples:
        for ctx in sample.get("contexts") or []:
            if ctx not in seen:
                seen.add(ctx)
                ordered.append(ctx)
    return ordered


def collect_questions(samples: Iterable[dict]) -> List[str]:
    questions = []
    for sample in samples:
        question = (sample.get("question") or "").strip()
        if question:
            questions.append(question)
    return questions


def build_prompt(contexts: Sequence[str], question: str, tokenizer, sep_tokens: List[int]) -> List[int]:
    tokens: List[int] = []
    for ctx in contexts:
        tokens.extend(sep_tokens)
        tokens.extend(tokenizer.encode(ctx, add_special_tokens=False))
    tokens.extend(sep_tokens)
    tokens.extend(tokenizer.encode(question, add_special_tokens=False))
    return tokens


def percentile(sorted_values: List[float], pct: float) -> float | None:
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


def summarize(values: List[float]) -> dict:
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


def aggregate_records(records: List[dict]) -> dict:
    ttfts = [r["ttft"] for r in records if r["ttft"] is not None]
    latencies = [r["latency"] for r in records if r["latency"] is not None]
    itls: List[float] = []
    for r in records:
        itls.extend(r.get("itl") or [])
    prompt_tokens = [r["prompt_tokens"] for r in records if r["prompt_tokens"] is not None]
    completion_tokens = [
        r["completion_tokens"] for r in records if r["completion_tokens"] is not None
    ]
    return {
        "request_count": len(records),
        "ttft": summarize(ttfts),
        "latency": summarize(latencies),
        "itl": summarize(itls),
        "prompt_tokens": summarize(prompt_tokens),
        "completion_tokens": summarize(completion_tokens),
        "success_rate": (
            sum(1 for r in records if r["success"]) / len(records) if records else None
        ),
    }


def sample_contexts(contexts: Sequence[str], count: int, rng: random.Random) -> List[str]:
    if not contexts or count <= 0:
        return []
    if len(contexts) >= count:
        return rng.sample(list(contexts), count)
    return [rng.choice(list(contexts)) for _ in range(count)]


def prepare_requests_for_rate(
    *,
    num_requests: int,
    contexts: Sequence[str],
    question_pool: Sequence[str],
    contexts_range: Tuple[int, int],
    tokenizer,
    sep_tokens: List[int],
    rng: random.Random,
) -> List[dict]:
    if not question_pool:
        question_pool = ["请回答问题，并引用相关上下文。"]
    prepared: List[dict] = []
    lo, hi = contexts_range
    for ordinal in range(1, num_requests + 1):
        ctx_count = rng.randint(lo, hi)
        blended_contexts = sample_contexts(contexts, ctx_count, rng)
        question = rng.choice(list(question_pool))
        prompt = build_prompt(blended_contexts, question, tokenizer, sep_tokens)
        prepared.append(
            {
                "ordinal": ordinal,
                "contexts": blended_contexts,
                "question": question,
                "prompt": prompt,
            }
        )
    return prepared


async def warmup_contexts(session, contexts: Sequence[str], tokenizer) -> None:
    total = len(contexts)
    if total == 0:
        print("没有可供预热的contexts，跳过warmup。")
        return
    max_retries = 3
    retry_delay = 1.0
    successes = 0
    failures = 0
    print("=" * 60)
    print(f"Warmup: storing {total} unique contexts")
    print("=" * 60)
    for idx, ctx in enumerate(contexts, start=1):
        tokens = tokenizer.encode(ctx, add_special_tokens=False)
        request_name = f"[Warmup {idx}/{total}]"
        print(f"{request_name} start (tokens={len(tokens)})")
        attempt = 1
        while attempt <= max_retries:
            try:
                await helpers.stream_completion(
                    session=session, prompt=tokens, request_name=request_name
                )
                print(f"{request_name} done")
                successes += 1
                break
            except aiohttp.ClientError as exc:
                print(
                    f"{request_name} failed with {exc.__class__.__name__}: {exc} "
                    f"(attempt {attempt}/{max_retries})"
                )
                if attempt >= max_retries:
                    failures += 1
                    print(f"{request_name} giving up after {max_retries} attempts.")
                    break
                attempt += 1
                print(f"Retrying {request_name} in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
    print("-" * 60)
    print(f"Warmup completed. Successes: {successes}, Failures: {failures}")
    print("-" * 60)


def _format_rate(rate_value: float) -> str:
    return "inf" if math.isinf(rate_value) else f"{rate_value:g}"


async def run_requests_for_rate(
    session,
    prepared_requests: List[dict],
    request_rate: float,
    burstiness: float,
):
    entries = []
    for req in prepared_requests:
        request_name = f"[rate={_format_rate(request_rate)}] req={req['ordinal']}"
        entries.append((request_name, req["prompt"], req))

    tasks = []
    async for request_name, prompt, data in helpers.send_requests_with_rate_limit(
        entries, request_rate, burstiness
    ):
        task = asyncio.create_task(
            helpers.stream_completion(session=session, prompt=prompt, request_name=request_name)
        )
        tasks.append((request_name, data, task))

    gathered = await asyncio.gather(*(t for _, _, t in tasks), return_exceptions=True)
    records = []
    for (request_name, data, _), metrics in zip(tasks, gathered):
        base_record = {
            "request_name": request_name,
            "num_contexts": len(data.get("contexts") or []),
            "question": data.get("question"),
        }
        if isinstance(metrics, Exception):
            print(
                f"{request_name} errored with {metrics.__class__.__name__}: {metrics}",
                file=sys.stderr,
            )
            records.append(
                {
                    **base_record,
                    "ttft": None,
                    "latency": None,
                    "itl": [],
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "success": False,
                    "error": str(metrics),
                }
            )
        else:
            records.append(
                {
                    **base_record,
                    "ttft": metrics.get("ttft"),
                    "latency": metrics.get("latency"),
                    "itl": metrics.get("itl") or [],
                    "prompt_tokens": metrics.get("prompt_tokens"),
                    "completion_tokens": metrics.get("completion_tokens"),
                    "success": metrics.get("success", False),
                    "error": None,
                }
            )
    summary = aggregate_records(records)
    return {"records": records, "summary": summary}


def ms_str(value: float | None) -> str:
    return helpers._format_ms(value)


def print_summary_table(results: Dict[float, dict], request_rates: List[float]):
    header = f"{'Rate (RPS)':>12}{'TTFT(avg)':>16}{'Latency(avg)':>16}{'ITL(avg)':>16}{'Success%':>12}"
    print()
    print("=" * len(header))
    print("Summary Table")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for rate in request_rates:
        data = results.get(rate)
        if not data:
            continue
        summary = data["summary"]
        success_rate = summary["success_rate"]
        ttft_avg = ms_str(summary["ttft"]["avg"])
        latency_avg = ms_str(summary["latency"]["avg"])
        itl_avg = ms_str(summary["itl"]["avg"])
        success_pct = f"{success_rate * 100:.1f}%" if success_rate is not None else "N/A"
        rate_str = _format_rate(rate)
        print(
            f"{rate_str:>12}{ttft_avg:>16}{latency_avg:>16}{itl_avg:>16}{success_pct:>12}"
        )
    print("-" * len(header))


async def main_async() -> None:
    args = parse_args()
    helpers.api_base = args.api_base
    helpers.model = args.model
    random.seed(args.seed)

    request_rates = parse_request_rates(args.request_rates)
    contexts_range = parse_context_range(args.contexts_per_request)
    dataset_path = Path(args.dataset).expanduser()
    samples = load_dataset(str(dataset_path))
    if not samples:
        raise RuntimeError("Dataset is empty, cannot run benchmark.")

    contexts = unique_contexts(samples)
    questions = collect_questions(samples)
    if not contexts:
        raise RuntimeError("Dataset does not contain any contexts.")
    warmup_contexts_list = contexts[: args.num_contexts] if args.num_contexts else contexts
    if not warmup_contexts_list:
        raise RuntimeError("No contexts selected for warmup.")
    if not questions:
        print("警告: 数据集中缺少问题，将使用占位问题。")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sep_tokens = tokenizer.encode(" # # ", add_special_tokens=False)

    print("=" * 60)
    print("Blend KV Cache Reuse Request-Rate Benchmark")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"API Base: {args.api_base}")
    print(f"Dataset: {dataset_path}")
    print(f"Loaded samples: {len(samples)}")
    print(f"Unique contexts: {len(contexts)}")
    print(f"Warmup contexts used: {len(warmup_contexts_list)}")
    print(f"Questions available: {len(questions)}")
    print(f"Contexts per request range: {contexts_range}")
    print(f"Requests per rate: {args.num_requests}")
    print(f"Request rates: {request_rates}")
    print(f"Burstiness: {args.burstiness}")
    print()

    async with aiohttp.ClientSession() as session:
        if args.skip_warmup:
            print("Skipping warmup as requested; assuming cache is already primed.")
        else:
            await warmup_contexts(session, warmup_contexts_list, tokenizer)
            if args.warmup_delay > 0:
                print(f"Waiting {args.warmup_delay} seconds after warmup...")
                await asyncio.sleep(args.warmup_delay)

        results: Dict[float, dict] = {}
        for idx, rate in enumerate(request_rates):
            rng = random.Random(args.seed + idx)
            prepared = prepare_requests_for_rate(
                num_requests=args.num_requests,
                contexts=warmup_contexts_list,
                question_pool=questions,
                contexts_range=contexts_range,
                tokenizer=tokenizer,
                sep_tokens=sep_tokens,
                rng=rng,
            )
            print("=" * 60)
            print(f"Running benchmark for rate={_format_rate(rate)} RPS")
            print("=" * 60)
            results[rate] = await run_requests_for_rate(
                session=session,
                prepared_requests=prepared,
                request_rate=rate,
                burstiness=args.burstiness,
            )

    print_summary_table(results, request_rates)

    output_payload = {
        "config": {
            "model": args.model,
            "api_base": args.api_base,
            "dataset": str(dataset_path),
            "request_rates": request_rates,
            "num_requests": args.num_requests,
            "num_contexts": len(warmup_contexts_list),
            "contexts_per_request": contexts_range,
            "burstiness": args.burstiness,
            "seed": args.seed,
        },
        "results": {
            (_format_rate(rate)): data for rate, data in results.items()
        },
    }

    output_path = Path(args.output_json).expanduser()
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"Results written to {output_path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
