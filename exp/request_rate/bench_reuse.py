#!/usr/bin/env python3
"""Benchmark cache reuse at different request rates for multiple compression methods."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Sequence, Tuple

import aiohttp
import numpy as np
from transformers import AutoTokenizer

CURRENT_DIR = Path(__file__).resolve().parent
BLEND_DIR = CURRENT_DIR.parent
if str(BLEND_DIR) not in sys.path:
    sys.path.insert(0, str(BLEND_DIR))

import test_large_prompt as helpers  # noqa: E402

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_PORT = 12345
DEFAULT_DATASET = "/home/xujie/TaoTie/dataset/needle_multi_rag.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark cache reuse (TTFT) for multiple compression methods under different request rates."
        )
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument(
        "--server-log",
        type=str,
        default=os.environ.get("LMCACHE_SERVER_PROFILE_LOG"),
        help="Optional vLLM server log path used to extract server-side profile metrics.",
    )
    parser.add_argument("--methods", type=str, default="OURS,KIVI_2BIT")
    parser.add_argument("--num-contexts", type=int, default=10)
    parser.add_argument(
        "--num-requests",
        type=int,
        default=30,
        help="Number of benchmark requests per rate (default: 30).",
    )
    parser.add_argument(
        "--ctx-per-req",
        type=str,
        default="2,3",
        help="Range 'min,max' of contexts to include per request.",
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=None,
        help="Target average request length (tokens). Overrides --ctx-per-req when set.",
    )
    parser.add_argument(
        "--rates",
        type=str,
        default="0.5,1,2,4",
        help="Comma separated request rates (RPS). Use 'inf' for unlimited.",
    )
    parser.add_argument("--warmup-delay", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reuse-requests-across-rates",
        action="store_true",
        help="Reuse the same prepared request set across rates within a method/round.",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        choices=("poisson", "fixed"),
        default="poisson",
        help="Request arrival schedule: 'poisson' or fixed interval 'fixed'.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of rounds to run per rate, take minimum TTFT (default: 1).",
    )
    parser.add_argument(
        "--match-rate",
        action="store_true",
        help="Set num_requests = rate for each rate (e.g., rate=4 sends 4 requests in 1 second).",
    )
    return parser.parse_args()


def parse_methods(raw: str) -> List[str]:
    methods = [chunk.strip().upper() for chunk in (raw or "").split(",") if chunk.strip()]
    if not methods:
        raise ValueError("At least one compression method must be specified via --methods.")
    return methods


def parse_rates(raw: str) -> List[float]:
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
        raise ValueError("At least one request rate must be provided.")
    return rates


def parse_ctx_range(raw: str) -> Tuple[int, int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("--ctx-per-req must be in 'min,max' format.")
    lo, hi = int(parts[0]), int(parts[1])
    if lo <= 0 or hi <= 0:
        raise ValueError("Context counts must be positive.")
    if lo > hi:
        raise ValueError("Range lower bound cannot exceed upper bound.")
    return lo, hi


def load_dataset(path: str) -> List[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def unique_contexts(samples: Iterable[dict]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for sample in samples:
        for ctx in sample.get("contexts") or []:
            if ctx not in seen:
                seen.add(ctx)
                ordered.append(ctx)
    return ordered


def collect_questions(samples: Iterable[dict]) -> List[str]:
    questions: List[str] = []
    for sample in samples:
        q = (sample.get("question") or "").strip()
        if q:
            questions.append(q)
    return questions


def encode_contexts(contexts: Sequence[str], tokenizer) -> List[dict]:
    encoded = []
    for ctx in contexts:
        tokens = tokenizer.encode(ctx, add_special_tokens=False)
        encoded.append({"text": ctx, "tokens": tokens})
    return encoded


def sample_context_entries(
    entries: Sequence[dict], count: int, rng: random.Random
) -> List[dict]:
    if not entries or count <= 0:
        return []
    if len(entries) >= count:
        return rng.sample(list(entries), count)
    return [rng.choice(list(entries)) for _ in range(count)]


def build_prompt_from_tokens(
    context_entries: Sequence[dict],
    question_tokens: Sequence[int],
    sep_tokens: Sequence[int],
) -> List[int]:
    tokens: List[int] = []
    for entry in context_entries:
        tokens.extend(sep_tokens)
        tokens.extend(entry["tokens"])
    tokens.extend(sep_tokens)
    tokens.extend(question_tokens)
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
    headers_latencies = [
        r["response_headers_latency"]
        for r in records
        if r.get("response_headers_latency") is not None
    ]
    first_chunk_latencies = [
        r["first_chunk_latency"]
        for r in records
        if r.get("first_chunk_latency") is not None
    ]
    first_choice_latencies = [
        r["first_choice_latency"]
        for r in records
        if r.get("first_choice_latency") is not None
    ]
    ttfts = [r["ttft"] for r in records if r["ttft"] is not None]
    latencies = [r["latency"] for r in records if r["latency"] is not None]
    server_blend_envelope = [
        (r["server_profile"]["server_blend_envelope_ms"] / 1000.0)
        for r in records
        if r.get("server_profile")
        and r["server_profile"].get("server_blend_envelope_ms") is not None
    ]
    server_true_ttft = [
        (r["server_profile"]["server_true_ttft_ms"] / 1000.0)
        for r in records
        if r.get("server_profile")
        and r["server_profile"].get("server_true_ttft_ms") is not None
    ]
    server_blender_done = [
        (r["server_profile"]["server_blender_done_ms"] / 1000.0)
        for r in records
        if r.get("server_profile")
        and r["server_profile"].get("server_blender_done_ms") is not None
    ]
    server_blend_core = [
        (r["server_profile"]["server_blend_core_ms"] / 1000.0)
        for r in records
        if r.get("server_profile")
        and r["server_profile"].get("server_blend_core_ms") is not None
    ]
    itls: List[float] = []
    for r in records:
        itls.extend(r.get("itl") or [])
    prompt_tokens = [r["prompt_tokens"] for r in records if r["prompt_tokens"] is not None]
    completion_tokens = [
        r["completion_tokens"] for r in records if r["completion_tokens"] is not None
    ]
    success_count = sum(1 for r in records if r["success"])
    request_count = len(records)
    return {
        "request_count": request_count,
        "headers_latency": summarize(headers_latencies),
        "first_chunk_latency": summarize(first_chunk_latencies),
        "first_choice_latency": summarize(first_choice_latencies),
        "ttft": summarize(ttfts),
        "latency": summarize(latencies),
        "server_true_ttft": summarize(server_true_ttft),
        "server_blender_done": summarize(server_blender_done),
        "server_blend_envelope": summarize(server_blend_envelope),
        "server_blend_core": summarize(server_blend_core),
        "itl": summarize(itls),
        "prompt_tokens": summarize(prompt_tokens),
        "completion_tokens": summarize(completion_tokens),
        "success_rate": (success_count / request_count) if request_count else None,
    }


def _format_rate(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value:g}"


def format_ms(value: float | None) -> str:
    return helpers._format_ms(value)


async def warmup_contexts(
    session: aiohttp.ClientSession,
    context_entries: Sequence[dict],
    sep_tokens: Sequence[int],
) -> None:
    """预热：每个 context 用纯内容 token 发送（不带 separator）。

    原因：SegmentTokenDatabase 分段时返回的是不含分隔符的纯 context，
    taotie_blender 计算 hash 时也是基于不含分隔符的 tokens[start:end]。
    因此 warmup 存储时也必须用纯 context 内容，hash 才能匹配。
    """
    total = len(context_entries)
    if total == 0:
        print("No contexts available for warmup, skipping.")
        return
    print("=" * 60)
    print(f"Warmup stage: sending {total} unique contexts (pure context tokens, no separator)")
    print("=" * 60)
    for idx, entry in enumerate(context_entries, start=1):
        request_name = f"[Warmup {idx}/{total}]"
        # 只发送纯 context 内容（不含分隔符），与检索时的 hash 计算方式一致
        warmup_tokens = list(entry["tokens"])
        try:
            await helpers.stream_completion(
                session=session,
                prompt=warmup_tokens,
                request_name=request_name,
            )
            print(f"{request_name} done (tokens={len(warmup_tokens)})")
        except Exception as exc:  # pragma: no cover - network errors
            print(f"{request_name} failed: {exc}")
    print("Warmup complete.")


def prepare_requests(
    *,
    num_requests: int,
    context_entries: Sequence[dict],
    question_pool: Sequence[str],
    ctx_range: Tuple[int, int],
    tokenizer,
    sep_tokens: Sequence[int],
    rng: random.Random,
) -> List[dict]:
    lo, hi = ctx_range
    prompts: List[dict] = []
    fallback_question = "Please answer the question based on above contexts."
    for ordinal in range(1, num_requests + 1):
        ctx_count = rng.randint(lo, hi)
        selected = sample_context_entries(context_entries, ctx_count, rng)
        question = rng.choice(question_pool) if question_pool else fallback_question
        question_tokens = tokenizer.encode(question, add_special_tokens=False)
        prompt = build_prompt_from_tokens(selected, question_tokens, sep_tokens)
        prompts.append(
            {
                "ordinal": ordinal,
                "prompt": prompt,
                "num_contexts": len(selected),
                "question": question,
            }
        )
    return prompts


def select_shared_requests(shared_requests: Sequence[dict], num_requests: int) -> List[dict]:
    selected: List[dict] = []
    for ordinal, request in enumerate(shared_requests[:num_requests], start=1):
        selected.append(
            {
                "ordinal": ordinal,
                "prompt": request["prompt"],
                "num_contexts": request["num_contexts"],
                "question": request["question"],
            }
        )
    return selected


async def poisson_schedule(
    entries: Sequence[tuple],
    request_rate: float,
    rng: np.random.Generator,
):
    total = len(entries)
    if total == 0:
        return
    if math.isinf(request_rate) or request_rate <= 0:
        for entry in entries:
            yield entry
        return
    interval = 1.0 / request_rate
    for idx, entry in enumerate(entries):
        yield entry
        if idx == total - 1:
            break
        delay = float(rng.exponential(interval))
        if delay > 0:
            await asyncio.sleep(delay)


async def fixed_interval_schedule(
    entries: Sequence[tuple],
    request_rate: float,
):
    total = len(entries)
    if total == 0:
        return
    if math.isinf(request_rate) or request_rate <= 0:
        for entry in entries:
            yield entry
        return
    interval = 1.0 / request_rate
    for idx, entry in enumerate(entries):
        yield entry
        if idx == total - 1:
            break
        await asyncio.sleep(interval)


async def run_requests_for_rate(
    session: aiohttp.ClientSession,
    prepared_requests: Sequence[dict],
    request_rate: float,
    rng: np.random.Generator,
    schedule_mode: str = "poisson",
):
    entries = []
    for req in prepared_requests:
        name = f"[rate={_format_rate(request_rate)}] req={req['ordinal']}"
        entries.append((name, req["prompt"], req))

    tasks = []
    if schedule_mode == "fixed":
        schedule_iter = fixed_interval_schedule(entries, request_rate)
    else:
        schedule_iter = poisson_schedule(entries, request_rate, rng)

    async for entry in schedule_iter:
        request_name, prompt, meta = entry
        task = asyncio.create_task(
            helpers.stream_completion(
                session=session,
                prompt=prompt,
                request_name=request_name,
            )
        )
        tasks.append((request_name, meta, task))

    gathered = await asyncio.gather(*(t for _, _, t in tasks), return_exceptions=True)
    records: List[dict] = []
    for (request_name, meta, _), result in zip(tasks, gathered):
        base = {
            "request_name": request_name,
            "num_contexts": meta.get("num_contexts"),
            "question": meta.get("question"),
        }
        if isinstance(result, Exception):
            print(f"{request_name} error: {result}", file=sys.stderr)
            records.append(
                {
                    **base,
                    "ttft": None,
                    "response_headers_latency": None,
                    "first_chunk_latency": None,
                    "first_choice_latency": None,
                    "latency": None,
                    "itl": [],
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "server_profile": None,
                    "success": False,
                }
            )
        else:
            records.append(
                {
                    **base,
                    "ttft": result.get("ttft"),
                    "response_headers_latency": result.get("response_headers_latency"),
                    "first_chunk_latency": result.get("first_chunk_latency"),
                    "first_choice_latency": result.get("first_choice_latency"),
                    "latency": result.get("latency"),
                    "itl": result.get("itl") or [],
                    "prompt_tokens": result.get("prompt_tokens"),
                    "completion_tokens": result.get("completion_tokens"),
                    "server_profile": result.get("server_profile"),
                    "success": result.get("success", False),
                }
            )
    summary = aggregate_records(records)
    return {"records": records, "summary": summary}


def print_rate_summary(method: str, rate: float, summary: dict) -> None:
    rate_str = _format_rate(rate)
    success_rate = summary.get("success_rate")
    req_count = summary.get("request_count", 0)
    success_pct = f"{success_rate * 100:.1f}%" if success_rate is not None else "N/A"
    headers_stats = summary.get("headers_latency", {})
    first_chunk_stats = summary.get("first_chunk_latency", {})
    first_choice_stats = summary.get("first_choice_latency", {})
    server_true_ttft = summary.get("server_true_ttft", {})
    server_blender_done = summary.get("server_blender_done", {})
    server_blend_envelope = summary.get("server_blend_envelope", {})
    server_blend_core = summary.get("server_blend_core", {})
    print(f"[{method}] Rate={rate_str} RPS over {req_count} requests:")
    print(
        f"  Headers avg={format_ms(headers_stats.get('avg'))}, "
        f"FirstChunk avg={format_ms(first_chunk_stats.get('avg'))}, "
        f"FirstChoice avg={format_ms(first_choice_stats.get('avg'))}"
    )
    if server_blend_envelope.get("avg") is not None:
        print(
            f"  Server trueTTFT avg={format_ms(server_true_ttft.get('avg'))}, "
            f"arrival->blender avg={format_ms(server_blender_done.get('avg'))}, "
            f"blend envelope avg={format_ms(server_blend_envelope.get('avg'))}, "
            f"blend core avg={format_ms(server_blend_core.get('avg'))}"
        )
    print(f"  Success rate: {success_pct}")


def build_summary_rows(
    all_results: Dict[str, Dict[float, dict]]
) -> List[dict]:
    rows: List[dict] = []
    for method in sorted(all_results.keys()):
        rate_dict = all_results[method]
        for rate in sorted(rate_dict.keys(), key=lambda x: float("inf") if math.isinf(x) else x):
            data = rate_dict[rate]
            summary = data["summary"]
            first_choice = summary["first_choice_latency"]
            headers = summary["headers_latency"]
            first_chunk = summary["first_chunk_latency"]
            server_true_ttft = summary["server_true_ttft"]
            server_blender_done = summary["server_blender_done"]
            server_blend_core = summary["server_blend_core"]
            rows.append(
                {
                    "method": method,
                    "rate": _format_rate(rate),
                    "headers_avg": headers["avg"],
                    "first_chunk_avg": first_chunk["avg"],
                    "avg": first_choice["avg"],
                    "p50": first_choice["p50"],
                    "p90": first_choice["p90"],
                    "p99": first_choice["p99"],
                    "server_true_ttft_avg": server_true_ttft["avg"],
                    "server_blender_done_avg": server_blender_done["avg"],
                    "server_blend_core_avg": server_blend_core["avg"],
                    "success_rate": summary.get("success_rate"),
                }
            )
    return rows


def print_final_table(rows: List[dict]) -> None:
    if not rows:
        return
    header = (
        f"{'Method':<12}{'Rate':>8}{'Headers':>12}{'1stChunk':>12}"
        f"{'1stChoice':>12}{'1stP50':>12}{'1stP90':>12}"
        f"{'SrvTTFT':>12}{'->Blend':>12}{'Success':>12}"
    )
    print("\n" + "=" * len(header))
    print("Final Summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in rows:
        success = row["success_rate"]
        success_str = f"{success * 100:.1f}%" if success is not None else "N/A"
        print(
            f"{row['method']:<12}"
            f"{row['rate']:>8}"
            f"{format_ms(row['headers_avg']):>12}"
            f"{format_ms(row['first_chunk_avg']):>12}"
            f"{format_ms(row['avg']):>12}"
            f"{format_ms(row['p50']):>12}"
            f"{format_ms(row['p90']):>12}"
            f"{format_ms(row['server_true_ttft_avg']):>12}"
            f"{format_ms(row['server_blender_done_avg']):>12}"
            f"{success_str:>12}"
        )
    print("-" * len(header))


async def run_method_benchmark(
    *,
    method: str,
    session: aiohttp.ClientSession,
    context_entries: Sequence[dict],
    question_pool: Sequence[str],
    tokenizer,
    sep_tokens: Sequence[int],
    ctx_range: Tuple[int, int],
    rates: Sequence[float],
    num_requests: int,
    warmup_delay: float,
    base_seed: int,
    rounds: int = 1,
    match_rate: bool = False,
    schedule_mode: str = "poisson",
    reuse_requests_across_rates: bool = False,
) -> Dict[float, dict]:
    print("\n" + "=" * 60)
    print(f"Testing compression method: {method}")
    print("=" * 60)

    os.environ["LMCACHE_COMPRESS_TYPE"] = method
    await warmup_contexts(session, context_entries, sep_tokens)
    if warmup_delay > 0:
        print(f"Waiting {warmup_delay}s for cache to settle...")
        await asyncio.sleep(warmup_delay)

    method_results: Dict[float, dict] = {}
    effective_num_requests_by_rate = [
        max(1, int(rate)) if match_rate else num_requests
        for rate in rates
    ]
    max_num_requests = max(effective_num_requests_by_rate, default=num_requests)
    shared_requests_by_round: Dict[int, List[dict]] = {}

    if reuse_requests_across_rates:
        for round_num in range(1, rounds + 1):
            rng_seed = base_seed * 1000 + round_num
            req_rng = random.Random(rng_seed)
            shared_requests_by_round[round_num] = prepare_requests(
                num_requests=max_num_requests,
                context_entries=context_entries,
                question_pool=question_pool,
                ctx_range=ctx_range,
                tokenizer=tokenizer,
                sep_tokens=sep_tokens,
                rng=req_rng,
            )

    for rate_idx, rate in enumerate(rates):
        # If match_rate is True, num_requests = rate (e.g., rate=4 sends 4 requests in 1 second)
        effective_num_requests = effective_num_requests_by_rate[rate_idx]
        print("-" * 60)
        print(
            f"Running rate={_format_rate(rate)} RPS "
            f"({effective_num_requests} requests, schedule={schedule_mode}, {rounds} round(s))"
        )

        best_result = None
        best_ttft_avg = float("inf")

        for round_num in range(1, rounds + 1):
            rng_seed = base_seed * 1000 + rate_idx * 100 + round_num
            if reuse_requests_across_rates:
                prepared = select_shared_requests(
                    shared_requests_by_round[round_num], effective_num_requests
                )
            else:
                req_rng = random.Random(rng_seed)
                prepared = prepare_requests(
                    num_requests=effective_num_requests,
                    context_entries=context_entries,
                    question_pool=question_pool,
                    ctx_range=ctx_range,
                    tokenizer=tokenizer,
                    sep_tokens=sep_tokens,
                    rng=req_rng,
                )
            poisson_rng = np.random.default_rng(rng_seed + 12345)

            if rounds > 1:
                print(f"  Round {round_num}/{rounds}...")

            result = await run_requests_for_rate(
                session=session,
                prepared_requests=prepared,
                request_rate=rate,
                rng=poisson_rng,
                schedule_mode=schedule_mode,
            )

            ttft_avg = result["summary"]["ttft"].get("avg")
            if rounds > 1:
                print(f"    TTFT avg={format_ms(ttft_avg)}")

            # Keep result with minimum TTFT avg
            if ttft_avg is not None and ttft_avg < best_ttft_avg:
                best_ttft_avg = ttft_avg
                best_result = result
            elif best_result is None:
                best_result = result

        method_results[rate] = best_result
        if rounds > 1:
            print(f"  Best round TTFT avg={format_ms(best_ttft_avg)}")
        print_rate_summary(method, rate, best_result["summary"])
    return method_results


async def main_async() -> None:
    args = parse_args()
    helpers.model = args.model
    helpers.api_base = f"http://localhost:{args.port}"
    helpers.server_profile_log_path = args.server_log

    methods = parse_methods(args.methods)
    rates = parse_rates(args.rates)
    ctx_range = parse_ctx_range(args.ctx_per_req)

    dataset_path = Path(args.dataset).expanduser()
    samples = load_dataset(str(dataset_path))
    if not samples:
        raise RuntimeError(f"Dataset {dataset_path} is empty or missing.")

    contexts = unique_contexts(samples)
    if not contexts:
        raise RuntimeError("Dataset has no contexts to cache.")
    if args.num_contexts > len(contexts):
        print(
            f"Requested {args.num_contexts} contexts but dataset only has {len(contexts)}; using all."
        )
    warmup_contexts_list = contexts[: args.num_contexts] if args.num_contexts else contexts
    questions = collect_questions(samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # IMPORTANT: Must match LMCache's separator encoding: encode()[1:] to skip first token
    # LMCache (token_database.py:313): self.sep_tokens = self.tokenizer.encode(config.blend_special_str)[1:]
    sep_tokens = tokenizer.encode(" # # ")[1:]
    context_entries = encode_contexts(warmup_contexts_list, tokenizer)
    avg_ctx_len: float | None = None
    estimated_request_len: float | None = None
    if args.target_length is not None:
        if not context_entries:
            raise RuntimeError("Cannot compute target length override without contexts.")
        avg_ctx_len = sum(len(entry["tokens"]) for entry in context_entries) / len(context_entries)
        if avg_ctx_len <= 0:
            print("Average context length is zero; defaulting to 1 context per request.")
            ctx_count = 1
            estimated_request_len = None
        else:
            ctx_count = max(1, int(round(args.target_length / avg_ctx_len)))
            estimated_request_len = ctx_count * avg_ctx_len
        ctx_range = (ctx_count, ctx_count)

    print("=" * 60)
    print("Cache Reuse Benchmark")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Server: http://localhost:{args.port}")
    print(f"Dataset: {dataset_path}")
    print(f"Unique contexts loaded: {len(contexts)}")
    print(f"Contexts used for warmup: {len(warmup_contexts_list)}")
    print(f"Questions available: {len(questions)}")
    print(f"Context per request range: {ctx_range}")
    if args.target_length is not None:
        approx_tokens = (
            int(round(estimated_request_len)) if estimated_request_len is not None else "unknown"
        )
        avg_ctx_str = f"{avg_ctx_len:.1f}" if avg_ctx_len is not None else "N/A"
        print(
            f"Target length {args.target_length} tokens -> using {ctx_range[0]} contexts "
            f"per request (~{approx_tokens} tokens, avg ctx {avg_ctx_str} tokens)"
        )
    print(f"Request rates: {', '.join(_format_rate(r) for r in rates)}")
    print(f"Requests per rate: {args.num_requests}")
    print(f"Reuse requests across rates: {args.reuse_requests_across_rates}")
    print(f"Schedule mode: {args.schedule}")
    print(f"Rounds per rate: {args.rounds}")
    print(f"Compression methods: {methods}")
    print("=" * 60)

    all_results: Dict[str, Dict[float, dict]] = {}
    base_seed = args.seed

    connector = aiohttp.TCPConnector(limit=256)
    async with aiohttp.ClientSession(connector=connector) as session:
        for idx, method in enumerate(methods):
            seed_offset = base_seed + idx * 100
            method_results = await run_method_benchmark(
                method=method,
                session=session,
                context_entries=context_entries,
                question_pool=questions,
                tokenizer=tokenizer,
                sep_tokens=sep_tokens,
                ctx_range=ctx_range,
                rates=rates,
                num_requests=args.num_requests,
                warmup_delay=args.warmup_delay,
                base_seed=seed_offset,
                rounds=args.rounds,
                match_rate=args.match_rate,
                schedule_mode=args.schedule,
                reuse_requests_across_rates=args.reuse_requests_across_rates,
            )
            all_results[method] = method_results

    summary_rows = build_summary_rows(all_results)
    print_final_table(summary_rows)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
