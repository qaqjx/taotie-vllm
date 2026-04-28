#!/usr/bin/env python3
"""Send wikimqa_s requests without cache warmup to stress xj_project CPU compression."""
from __future__ import annotations

import argparse
import asyncio
import bisect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import test_large_prompt as profile_helpers  # noqa: E402


def build_prompt_token_ids(
    contexts: list[str],
    question: str,
    *,
    tokenizer,
    sep_tokens: list[int],
) -> list[int]:
    tokens: list[int] = []
    for context in contexts:
        tokens.extend(sep_tokens)
        tokens.extend(tokenizer.encode(context, add_special_tokens=False))
    tokens.extend(sep_tokens)
    tokens.extend(tokenizer.encode(question, add_special_tokens=False))
    return tokens


def build_completion_payload(
    *,
    model: str,
    sample: dict[str, Any],
    max_tokens: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    if sample.get("prompt_token_ids") is not None:
        payload["prompt"] = sample["prompt_token_ids"]
    else:
        payload["prompt"] = sample["prompt"]
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--api-base", default="http://127.0.0.1:12345")
    parser.add_argument("--num-requests", type=int, default=24)
    parser.add_argument(
        "--source-limit",
        type=int,
        default=None,
        help="Limit how many dataset rows are used as the repeating source pool.",
    )
    parser.add_argument(
        "--source-offset",
        type=int,
        default=0,
        help="Skip this many dataset rows before building the source pool.",
    )
    parser.add_argument("--request-rate", type=float, default=4.0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--sep", default=" # # ")
    parser.add_argument(
        "--repeat-dataset",
        action="store_true",
        help="Reuse dataset entries cyclically until --num-requests is reached.",
    )
    parser.add_argument(
        "--queue-log",
        default=None,
        help="Optional xj queue JSONL used to annotate queue memory at request send time.",
    )
    parser.add_argument(
        "--server-log",
        default=os.environ.get("LMCACHE_SERVER_PROFILE_LOG"),
        help="Optional vLLM server log used to extract per-request server-side timings.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_prompts(
    path: str,
    limit: int,
    sep: str,
    repeat_dataset: bool = False,
    source_limit: int | None = None,
    source_offset: int = 0,
    tokenizer=None,
    sep_tokens: list[int] | None = None,
) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    base_samples: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as dataset:
        for line_idx, line in enumerate(dataset):
            if line_idx < source_offset:
                continue
            sample = json.loads(line)
            base_samples.append(sample)
            max_source = source_limit if source_limit is not None else limit
            if max_source is not None and len(base_samples) >= max_source:
                break

    if not base_samples:
        return prompts

    while len(prompts) < limit:
        sample = base_samples[len(prompts) % len(base_samples)]
        contexts = sample.get("contexts") or []
        question = sample.get("question") or ""
        prompt = sep + sep.join(contexts) + sep + question
        prompt_token_ids = None
        if tokenizer is not None and sep_tokens is not None:
            prompt_token_ids = build_prompt_token_ids(
                list(contexts),
                question,
                tokenizer=tokenizer,
                sep_tokens=sep_tokens,
            )
        prompts.append(
            {
                "id": sample.get("id", len(prompts)),
                "prompt": prompt,
                "prompt_token_ids": prompt_token_ids,
                "num_contexts": len(contexts),
            }
        )
        if not repeat_dataset and len(prompts) >= len(base_samples):
            break
    return prompts


async def send_one(
    session: aiohttp.ClientSession,
    *,
    api_url: str,
    model: str,
    sample: dict[str, Any],
    max_tokens: int,
    server_profile_log_path: str | None = None,
) -> dict[str, Any]:
    payload = build_completion_payload(
        model=model,
        sample=sample,
        max_tokens=max_tokens,
    )
    sent_at_wall = time.time()
    start = time.perf_counter()
    first_chunk = None
    text_chunks: list[str] = []
    error = None
    status = None
    request_id = None

    try:
        async with session.post(api_url, json=payload) as response:
            status = response.status
            request_id = response.headers.get("x-request-id")
            async for raw_line in response.content:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data: "):
                    continue
                if first_chunk is None:
                    first_chunk = time.perf_counter()
                data = line[len("data: ") :]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                request_id = chunk.get("id") or request_id
                choices = chunk.get("choices") or []
                if choices:
                    text_chunks.append(choices[0].get("text") or "")
    except Exception as exc:
        error = repr(exc)

    end = time.perf_counter()
    result = {
        "id": sample["id"],
        "status": status,
        "success": error is None and status == 200,
        "error": error,
        "num_contexts": sample["num_contexts"],
        "prompt_chars": len(sample["prompt"]),
        "sent_at_wall": sent_at_wall,
        "ttft_s": None if first_chunk is None else first_chunk - start,
        "latency_s": end - start,
        "generated_text": "".join(text_chunks),
        "output_chars": sum(len(chunk) for chunk in text_chunks),
    }
    if server_profile_log_path:
        profile_helpers.server_profile_log_path = server_profile_log_path
        server_profile = await profile_helpers.maybe_collect_server_profile(
            request_id
        )
        merge_server_profile(result, server_profile)
    return result


def merge_server_profile(
    result: dict[str, Any],
    server_profile: dict[str, Any] | None,
) -> None:
    if not server_profile:
        return
    result["server_profile"] = server_profile
    for key in (
        "server_true_ttft_ms",
        "server_blender_done_ms",
        "server_queue_ms",
        "server_prefill_ms",
        "server_blend_envelope_ms",
        "server_load_path_ms",
        "server_blend_core_ms",
    ):
        if key in server_profile:
            result[key] = server_profile[key]
    true_ttft = server_profile.get("server_true_ttft_ms")
    queue_ms = server_profile.get("server_queue_ms")
    if true_ttft is not None and queue_ms is not None:
        result["server_ttft_no_queue_ms"] = true_ttft - queue_ms
    scheduled_to_first_token_ms = server_profile.get(
        "server_scheduled_to_first_token_ms",
        server_profile.get("server_prefill_ms"),
    )
    if scheduled_to_first_token_ms is not None:
        result["server_scheduled_to_first_token_ms"] = (
            scheduled_to_first_token_ms
        )
    load_path_ms = server_profile.get("server_load_path_ms")
    if load_path_ms is not None:
        result["server_prefill_path_ms"] = load_path_ms
        result["server_load_path_ms"] = load_path_ms
    total_load_path_ms = server_profile.get("server_total_load_path_ms")
    if total_load_path_ms is not None:
        result["server_total_load_path_ms"] = total_load_path_ms
    blend_core_ms = server_profile.get("server_blend_core_ms")
    if blend_core_ms is not None:
        result["server_prefill_compute_ms"] = blend_core_ms


def load_queue_records(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    queue_path = Path(path)
    if not queue_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with queue_path.open("r", encoding="utf-8") as log_file:
        for line in log_file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def annotate_requests_with_queue_memory(
    results: list[dict[str, Any]],
    queue_records: list[dict[str, Any]],
) -> None:
    if not results or not queue_records:
        return
    queue_timestamps = [float(record.get("ts", 0.0)) for record in queue_records]
    for result in results:
        sent_at_wall = result.get("sent_at_wall")
        if sent_at_wall is None:
            continue
        record_idx = bisect.bisect_right(queue_timestamps, float(sent_at_wall)) - 1
        if record_idx < 0:
            continue
        record = queue_records[record_idx]
        queue_bytes = int(record.get("remote_queue_bytes", 0))
        result["arrival_queue_ts"] = record.get("ts")
        result["arrival_queue_bytes"] = queue_bytes
        result["arrival_queue_mib"] = queue_bytes / 1024 / 1024
        result["arrival_pending_count"] = int(record.get("remote_pending_count", 0))
        result["arrival_rss_bytes"] = record.get("rss_bytes")


async def main_async() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sep_tokens = tokenizer.encode(args.sep)[1:]
    prompts = load_prompts(
        args.dataset,
        args.num_requests,
        args.sep,
        repeat_dataset=args.repeat_dataset,
        source_limit=args.source_limit,
        source_offset=args.source_offset,
        tokenizer=tokenizer,
        sep_tokens=sep_tokens,
    )
    if not prompts:
        raise RuntimeError(f"No prompts loaded from {args.dataset}")

    api_url = f"{args.api_base.rstrip('/')}/v1/completions"
    timeout = aiohttp.ClientTimeout(total=args.timeout, sock_read=args.timeout)
    connector = aiohttp.TCPConnector(limit=0)
    results: list[dict[str, Any]] = []

    print(
        f"Sending {len(prompts)} wikimqa_s requests without warmup to {api_url} "
        f"at {args.request_rate} RPS"
    )
    profile_helpers.server_profile_log_path = args.server_log
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = []
        start_wall = time.time()
        for idx, sample in enumerate(prompts):
            tasks.append(
                asyncio.create_task(
                    send_one(
                        session,
                        api_url=api_url,
                        model=args.model,
                        sample=sample,
                        max_tokens=args.max_tokens,
                        server_profile_log_path=args.server_log,
                    )
                )
            )
            if idx != len(prompts) - 1 and args.request_rate > 0:
                await asyncio.sleep(1.0 / args.request_rate)

        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            print(
                f"request {result['id']} success={result['success']} "
                f"ttft={result['ttft_s']} latency={result['latency_s']:.3f}s",
                flush=True,
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    queue_records = load_queue_records(args.queue_log)
    annotate_requests_with_queue_memory(results, queue_records)
    summary = {
        "started_at": start_wall,
        "finished_at": time.time(),
        "api_url": api_url,
        "model": args.model,
        "dataset": args.dataset,
        "num_requests": len(prompts),
        "request_rate": args.request_rate,
        "queue_log": args.queue_log,
        "server_log": args.server_log,
        "success_count": sum(1 for result in results if result["success"]),
        "results": sorted(results, key=lambda item: str(item["id"])),
    }
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {output}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
