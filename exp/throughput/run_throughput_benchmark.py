#!/usr/bin/env python3
"""Benchmark achieved request throughput under target QPS sweeps."""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import aiohttp
import numpy as np
from transformers import AutoTokenizer

CURRENT_DIR = Path(__file__).resolve().parent
EXP_DIR = CURRENT_DIR.parent
REQUEST_RATE_DIR = EXP_DIR / "request_rate"
if str(REQUEST_RATE_DIR) not in sys.path:
    sys.path.insert(0, str(REQUEST_RATE_DIR))

import bench_reuse as request_rate_helpers  # noqa: E402

DEFAULT_MODEL = request_rate_helpers.DEFAULT_MODEL
DEFAULT_PORT = request_rate_helpers.DEFAULT_PORT
DEFAULT_DATASET = request_rate_helpers.DEFAULT_DATASET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep target QPS values and measure achieved request throughput (req/s)."
        )
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument(
        "--server-log",
        type=str,
        default=None,
        help="Optional vLLM server log path for server-side profiling reuse.",
    )
    parser.add_argument(
        "--rates",
        type=str,
        default="1,2,4,8",
        help="Comma separated target QPS values. Use 'inf' for unlimited.",
    )
    parser.add_argument("--num-contexts", type=int, default=20)
    parser.add_argument("--num-requests", type=int, default=20)
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
    parser.add_argument("--warmup-delay", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(CURRENT_DIR / "results"),
        help="Directory for summary CSV/JSON outputs.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="throughput_summary",
        help="Prefix for generated output filenames.",
    )
    return parser.parse_args()


def _format_rate(rate: float) -> str:
    return request_rate_helpers._format_rate(rate)


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _latency_stats(records: Sequence[dict]) -> dict:
    latencies = [r["latency"] for r in records if r.get("latency") is not None]
    return request_rate_helpers.summarize(latencies)


async def warmup_contexts_or_fail(
    session: aiohttp.ClientSession,
    context_entries: Sequence[dict],
    sep_tokens: Sequence[int],
) -> None:
    total = len(context_entries)
    if total == 0:
        raise RuntimeError("No contexts available for warmup.")

    successes = 0
    failures = 0
    print("=" * 60)
    print(f"Warmup stage: sending {total} unique contexts (pure context tokens, no separator)")
    print("=" * 60)
    for idx, entry in enumerate(context_entries, start=1):
        request_name = f"[Warmup {idx}/{total}]"
        warmup_tokens = list(entry["tokens"])
        try:
            await request_rate_helpers.helpers.stream_completion(
                session=session,
                prompt=warmup_tokens,
                request_name=request_name,
            )
            successes += 1
            print(f"{request_name} done (tokens={len(warmup_tokens)})")
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            failures += 1
            print(f"{request_name} failed: {exc}")

    print("Warmup complete.")
    if failures:
        raise RuntimeError(
            f"Warmup failed for {failures}/{total} contexts (successes={successes}). "
            "Aborting this method benchmark."
        )


async def _run_single_request(
    *,
    session: aiohttp.ClientSession,
    prompt: Sequence[int],
    request_name: str,
    metadata: dict,
) -> dict:
    started_at = time.perf_counter()
    try:
        result = await request_rate_helpers.helpers.stream_completion(
            session=session,
            prompt=prompt,
            request_name=request_name,
        )
        finished_at = time.perf_counter()
        return {
            **metadata,
            "request_name": request_name,
            "started_at": started_at,
            "finished_at": finished_at,
            "ttft": result.get("ttft"),
            "latency": result.get("latency"),
            "prompt_tokens": result.get("prompt_tokens"),
            "completion_tokens": result.get("completion_tokens"),
            "success": result.get("success", False),
            "request_id": result.get("request_id"),
        }
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        finished_at = time.perf_counter()
        return {
            **metadata,
            "request_name": request_name,
            "started_at": started_at,
            "finished_at": finished_at,
            "ttft": None,
            "latency": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "success": False,
            "request_id": None,
            "error": str(exc),
        }


async def run_requests_for_rate(
    *,
    session: aiohttp.ClientSession,
    prepared_requests: Sequence[dict],
    request_rate: float,
    rng: np.random.Generator,
) -> dict:
    scheduled_entries: List[Tuple[str, Sequence[int], dict]] = []
    for req in prepared_requests:
        request_name = f"[qps={_format_rate(request_rate)}] req={req['ordinal']}"
        metadata = {
            "ordinal": req["ordinal"],
            "num_contexts": req.get("num_contexts"),
            "question": req.get("question"),
        }
        scheduled_entries.append((request_name, req["prompt"], metadata))

    dispatch_started_at = None
    tasks = []
    async for request_name, prompt, metadata in request_rate_helpers.poisson_schedule(
        scheduled_entries, request_rate, rng
    ):
        if dispatch_started_at is None:
            dispatch_started_at = time.perf_counter()
        task = asyncio.create_task(
            _run_single_request(
                session=session,
                prompt=prompt,
                request_name=request_name,
                metadata=metadata,
            )
        )
        tasks.append(task)

    if not tasks:
        return {
            "records": [],
            "summary": {
                "target_qps": _format_rate(request_rate),
                "attempted_requests": 0,
                "successful_requests": 0,
                "elapsed_seconds": 0.0,
                "achieved_throughput_rps": None,
                "success_rate": None,
                "latency_avg_seconds": None,
                "latency_p50_seconds": None,
                "latency_p90_seconds": None,
            },
        }

    records = await asyncio.gather(*tasks)
    finished_at = max(record["finished_at"] for record in records)
    if dispatch_started_at is None:
        dispatch_started_at = min(record["started_at"] for record in records)
    elapsed_seconds = finished_at - dispatch_started_at
    attempted_requests = len(records)
    successful_requests = sum(1 for record in records if record.get("success"))
    success_rate = _safe_divide(successful_requests, attempted_requests)
    throughput_rps = _safe_divide(successful_requests, elapsed_seconds)
    latency_stats = _latency_stats(records)

    return {
        "records": records,
        "summary": {
            "target_qps": _format_rate(request_rate),
            "attempted_requests": attempted_requests,
            "successful_requests": successful_requests,
            "elapsed_seconds": elapsed_seconds,
            "achieved_throughput_rps": throughput_rps,
            "success_rate": success_rate,
            "latency_avg_seconds": latency_stats.get("avg"),
            "latency_p50_seconds": latency_stats.get("p50"),
            "latency_p90_seconds": latency_stats.get("p90"),
        },
    }


def build_summary_rows(results_by_rate: Dict[float, dict]) -> List[dict]:
    rows: List[dict] = []
    for rate in sorted(
        results_by_rate.keys(),
        key=lambda value: float("inf") if math.isinf(value) else value,
    ):
        summary = results_by_rate[rate]["summary"]
        rows.append(summary)
    return rows


def print_rate_summary(rate: float, summary: dict) -> None:
    throughput = summary["achieved_throughput_rps"]
    throughput_str = f"{throughput:.2f} req/s" if throughput is not None else "N/A"
    print(
        f"[QPS={_format_rate(rate)}] "
        f"attempted={summary['attempted_requests']}, "
        f"success={summary['successful_requests']}, "
        f"elapsed={summary['elapsed_seconds']:.2f}s, "
        f"throughput={throughput_str}"
    )
    print(
        f"  Success rate={_format_ratio(summary['success_rate'])}, "
        f"Latency avg={request_rate_helpers.format_ms(summary['latency_avg_seconds'])}, "
        f"p50={request_rate_helpers.format_ms(summary['latency_p50_seconds'])}, "
        f"p90={request_rate_helpers.format_ms(summary['latency_p90_seconds'])}"
    )


def print_final_table(rows: Sequence[dict]) -> None:
    if not rows:
        return
    header = (
        f"{'TargetQPS':>10}{'Attempted':>12}{'Success':>10}"
        f"{'Elapsed(s)':>12}{'Req/s':>12}{'SuccessRate':>14}{'LatencyAvg':>14}"
    )
    print("\n" + "=" * len(header))
    print("Throughput Summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for row in rows:
        throughput = row["achieved_throughput_rps"]
        throughput_str = f"{throughput:.2f}" if throughput is not None else "N/A"
        latency_avg = request_rate_helpers.format_ms(row["latency_avg_seconds"])
        print(
            f"{row['target_qps']:>10}"
            f"{row['attempted_requests']:>12}"
            f"{row['successful_requests']:>10}"
            f"{row['elapsed_seconds']:>12.2f}"
            f"{throughput_str:>12}"
            f"{_format_ratio(row['success_rate']):>14}"
            f"{latency_avg:>14}"
        )
    print("-" * len(header))


def save_results(
    *,
    output_dir: Path,
    output_prefix: str,
    args: argparse.Namespace,
    rows: Sequence[dict],
    results_by_rate: Dict[float, dict],
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"{output_prefix}_{timestamp}.csv"
    json_path = output_dir / f"{output_prefix}_{timestamp}.json"

    fieldnames = [
        "target_qps",
        "attempted_requests",
        "successful_requests",
        "elapsed_seconds",
        "achieved_throughput_rps",
        "success_rate",
        "latency_avg_seconds",
        "latency_p50_seconds",
        "latency_p90_seconds",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": args.model,
            "port": args.port,
            "dataset": str(Path(args.dataset).expanduser()),
            "rates": args.rates,
            "num_contexts": args.num_contexts,
            "num_requests": args.num_requests,
            "ctx_per_req": args.ctx_per_req,
            "target_length": args.target_length,
            "warmup_delay": args.warmup_delay,
            "seed": args.seed,
        },
        "summary_rows": list(rows),
        "per_rate": {
            _format_rate(rate): data
            for rate, data in sorted(
                results_by_rate.items(),
                key=lambda item: float("inf") if math.isinf(item[0]) else item[0],
            )
        },
    }
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2)

    return csv_path, json_path


async def main_async() -> None:
    args = parse_args()
    request_rate_helpers.helpers.model = args.model
    request_rate_helpers.helpers.api_base = f"http://localhost:{args.port}"
    request_rate_helpers.helpers.server_profile_log_path = args.server_log

    rates = request_rate_helpers.parse_rates(args.rates)
    ctx_range = request_rate_helpers.parse_ctx_range(args.ctx_per_req)

    dataset_path = Path(args.dataset).expanduser()
    samples = request_rate_helpers.load_dataset(str(dataset_path))
    if not samples:
        raise RuntimeError(f"Dataset {dataset_path} is empty or missing.")

    contexts = request_rate_helpers.unique_contexts(samples)
    if not contexts:
        raise RuntimeError("Dataset has no contexts to cache.")
    if args.num_contexts > len(contexts):
        print(
            f"Requested {args.num_contexts} contexts but dataset only has {len(contexts)}; using all."
        )
    warmup_contexts_list = contexts[: args.num_contexts] if args.num_contexts else contexts
    questions = request_rate_helpers.collect_questions(samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sep_tokens = tokenizer.encode(" # # ")[1:]
    context_entries = request_rate_helpers.encode_contexts(warmup_contexts_list, tokenizer)

    avg_ctx_len: float | None = None
    estimated_request_len: float | None = None
    if args.target_length is not None:
        if not context_entries:
            raise RuntimeError("Cannot compute target length override without contexts.")
        avg_ctx_len = sum(len(entry["tokens"]) for entry in context_entries) / len(context_entries)
        if avg_ctx_len <= 0:
            ctx_count = 1
        else:
            ctx_count = max(1, int(round(args.target_length / avg_ctx_len)))
            estimated_request_len = ctx_count * avg_ctx_len
        ctx_range = (ctx_count, ctx_count)

    print("=" * 60)
    print("Request Throughput Benchmark")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Server: http://localhost:{args.port}")
    print(f"Dataset: {dataset_path}")
    print(f"Unique contexts loaded: {len(contexts)}")
    print(f"Contexts used for warmup: {len(warmup_contexts_list)}")
    print(f"Questions available: {len(questions)}")
    print(f"Contexts per request: {ctx_range}")
    if args.target_length is not None:
        approx_tokens = (
            int(round(estimated_request_len)) if estimated_request_len is not None else "unknown"
        )
        avg_ctx_str = f"{avg_ctx_len:.1f}" if avg_ctx_len is not None else "N/A"
        print(
            f"Target length {args.target_length} tokens -> using {ctx_range[0]} contexts "
            f"per request (~{approx_tokens} tokens, avg ctx {avg_ctx_str} tokens)"
        )
    print(f"Target QPS values: {', '.join(_format_rate(rate) for rate in rates)}")
    print(f"Requests per rate: {args.num_requests}")
    print("=" * 60)

    connector = aiohttp.TCPConnector(limit=256)
    results_by_rate: Dict[float, dict] = {}
    async with aiohttp.ClientSession(connector=connector) as session:
        await warmup_contexts_or_fail(session, context_entries, sep_tokens)
        if args.warmup_delay > 0:
            print(f"Waiting {args.warmup_delay}s for cache to settle...")
            await asyncio.sleep(args.warmup_delay)

        for rate_index, rate in enumerate(rates):
            print("-" * 60)
            print(f"Running target QPS={_format_rate(rate)}")
            req_rng = random.Random(args.seed * 1000 + rate_index)
            prepared_requests = request_rate_helpers.prepare_requests(
                num_requests=args.num_requests,
                context_entries=context_entries,
                question_pool=questions,
                ctx_range=ctx_range,
                tokenizer=tokenizer,
                sep_tokens=sep_tokens,
                rng=req_rng,
            )
            poisson_rng = np.random.default_rng(args.seed * 1000 + rate_index + 12345)
            result = await run_requests_for_rate(
                session=session,
                prepared_requests=prepared_requests,
                request_rate=rate,
                rng=poisson_rng,
            )
            results_by_rate[rate] = result
            print_rate_summary(rate, result["summary"])

    rows = build_summary_rows(results_by_rate)
    print_final_table(rows)

    csv_path, json_path = save_results(
        output_dir=Path(args.output_dir).expanduser(),
        output_prefix=args.output_prefix,
        args=args,
        rows=rows,
        results_by_rate=results_by_rate,
    )
    print(f"Saved CSV summary to: {csv_path}")
    print(f"Saved JSON summary to: {json_path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
