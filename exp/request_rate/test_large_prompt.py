#!/usr/bin/env python3
"""大prompt测试 - 看真实加速比"""
import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from statistics import mean, median

import aiohttp
import numpy as np
from transformers import AutoTokenizer


api_base = "http://localhost:12345"
model = "mistralai/Mistral-7B-Instruct-v0.3"
server_profile_log_path = os.environ.get("LMCACHE_SERVER_PROFILE_LOG")

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse_args():
    parser = argparse.ArgumentParser(description="Blend KV large prompt test")
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
        help="Burstiness factor. 1.0 corresponds to a Poisson process.",
    )
    parser.add_argument(
        "--server-log",
        type=str,
        default=server_profile_log_path,
        help="Optional vLLM server log path used to extract server-side profile metrics.",
    )
    return parser.parse_args()


def _format_ms(value):
    return f"{value * 1000:.2f} ms" if value is not None else "N/A"


def _sum_profile_metric(lines, pattern):
    total = 0.0
    for line in lines:
        match = re.search(pattern, line)
        if match:
            total += float(match.group(1))
    return total


def _last_profile_metric(lines, pattern):
    value = None
    for line in lines:
        match = re.search(pattern, line)
        if match:
            value = float(match.group(1))
    return value


def _extract_server_profile(lines, request_id: str):
    matching_indices = [
        index
        for index, line in enumerate(lines)
        if "[PROFILE]" in line and request_id in line
    ]
    if not matching_indices:
        return None

    start_index = matching_indices[0]
    end_index = None
    for index in range(start_index, len(lines)):
        line = lines[index]
        if "[PROFILE]" not in line:
            continue
        if index > start_index and "scheduler.lookup:" in line and request_id not in line:
            end_index = index
            break
        if "start_load_kv: all requests total took" in line:
            end_index = index + 1
            break

    block = lines[start_index:end_index]
    if not block:
        return None

    retrieve_db_ms = _sum_profile_metric(
        block, r"retrieve_by_task_id: db\.retrive_by_task took ([0-9.]+)ms"
    )
    retrieve_transfer_ms = _sum_profile_metric(
        block, r"retrieve_by_task_id: transfer \d+ items took ([0-9.]+)ms"
    )
    decompress_ms = 0.0
    copy_ms = 0.0
    for line in block:
        match = re.search(
            r"get_reuse_kv: layer \d+ decompressed \d+ chunks in ([0-9.]+)ms "
            r"and copied to buffers in ([0-9.]+)ms",
            line,
        )
        if match:
            decompress_ms += float(match.group(1))
            copy_ms += float(match.group(2))
    blend_forward_ms = _sum_profile_metric(
        block,
        r"prefill_select_token: layer=\d+ get_reuse_kv=[0-9.]+ms "
        r"blend_forward=([0-9.]+)ms",
    )

    def _last_profile_metric_for_request(pattern):
        value = None
        for line in lines:
            if request_id not in line:
                continue
            match = re.search(pattern, line)
            if match:
                value = float(match.group(1))
        return value

    return {
        "request_id": request_id,
        "server_true_ttft_ms": _last_profile_metric_for_request(
            r"true_server_ttft: req=.* arrival_to_first_token=([0-9.]+)ms",
        ),
        "server_blender_done_ms": _last_profile_metric_for_request(
            r"arrival_to_first_blender_done: req=.* ([0-9.]+)ms",
        ),
        "server_queue_ms": _last_profile_metric_for_request(
            r"true_server_ttft: req=.* queue=([0-9.]+)ms",
        ),
        "server_prefill_ms": _last_profile_metric_for_request(
            r"true_server_ttft: req=.* prefill=([0-9.]+)ms",
        ),
        "scheduler_lookup_ms": _last_profile_metric(
            block, r"scheduler\.lookup: .* took ([0-9.]+)ms"
        ),
        "build_connector_meta_ms": _last_profile_metric(
            block, r"build_connector_meta: .* took ([0-9.]+)ms"
        ),
        "process_tokens_ms": _last_profile_metric(
            block,
            r"start_load_kv: req=.* process_tokens .* took ([0-9.]+)ms",
        ),
        "server_blend_envelope_ms": _last_profile_metric(
            block,
            r"start_load_kv: req=.* blender\.blend total took ([0-9.]+)ms",
        ),
        "server_load_path_ms": _last_profile_metric(
            block,
            r"start_load_kv: req=.* total load-path took ([0-9.]+)ms",
        ),
        "server_blend_core_ms": (
            retrieve_db_ms + retrieve_transfer_ms + decompress_ms + copy_ms + blend_forward_ms
        ),
        "retrieve_by_task_db_ms": retrieve_db_ms,
        "retrieve_by_task_transfer_ms": retrieve_transfer_ms,
        "decompress_ms": decompress_ms,
        "copy_ms": copy_ms,
        "blend_forward_ms": blend_forward_ms,
    }


async def maybe_collect_server_profile(request_id: str | None, wait_timeout: float = 3.0):
    if not request_id or not server_profile_log_path:
        return None

    log_path = Path(server_profile_log_path)
    deadline = time.perf_counter() + wait_timeout
    while time.perf_counter() < deadline:
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [_ANSI_ESCAPE_RE.sub("", line.rstrip()) for line in f]
            profile = _extract_server_profile(lines, request_id)
            if profile is not None:
                return profile
        await asyncio.sleep(0.1)
    return None


async def send_requests_with_rate_limit(prompts, request_rate, burstiness):
    prompts = list(prompts)
    if not prompts:
        return
    if request_rate == float("inf"):
        for prompt in prompts:
            yield prompt
        return
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}."
    )
    theta = 1.0 / (request_rate * burstiness)
    total = len(prompts)
    for idx, prompt in enumerate(prompts):
        yield prompt
        if idx == total - 1:
            break
        interval = np.random.gamma(shape=burstiness, scale=theta)
        await asyncio.sleep(float(interval))


async def stream_completion(session, prompt, request_name):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 64,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start_time = time.perf_counter()
    # Set both total timeout and per-read timeout to avoid hanging on stuck connections
    # Use longer sock_read for concurrent blend requests that may take time to process
    timeout = aiohttp.ClientTimeout(total=300, sock_read=120)
    print(f"{request_name} Sending request...")
    async with session.post(
        f"{api_base}/v1/completions", json=payload, timeout=timeout
    ) as response:
        response.raise_for_status()
        print(f"{request_name} Response received, streaming...")
        response_headers_time = time.perf_counter()
        first_chunk_time = None
        first_token_time = None
        most_recent_timestamp = start_time
        generated_chunks = []
        itl = []
        prompt_tokens = None
        completion_tokens = None
        success = False
        request_id = response.headers.get("x-request-id")
        buffer = ""
        done = False
        async for raw_chunk in response.content:
            if not raw_chunk:
                continue
            buffer += raw_chunk.decode("utf-8")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                if first_chunk_time is None:
                    first_chunk_time = time.perf_counter()
                request_id = chunk.get("id") or request_id
                choices = chunk.get("choices")
                usage = chunk.get("usage")
                if choices:
                    timestamp = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = timestamp
                        success = True
                    else:
                        itl.append(timestamp - most_recent_timestamp)
                    most_recent_timestamp = timestamp
                    text = choices[0].get("text") or ""
                    generated_chunks.append(text)
                elif usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get(
                        "completion_tokens", completion_tokens
                    )
            if done:
                break
    response_headers_latency = (
        response_headers_time - start_time if response_headers_time is not None else None
    )
    first_chunk_latency = (
        first_chunk_time - start_time if first_chunk_time is not None else None
    )
    first_choice_latency = (
        first_token_time - start_time if first_token_time is not None else None
    )
    latency = most_recent_timestamp - start_time
    server_profile = await maybe_collect_server_profile(request_id)
    metrics = {
        "generated_text": "".join(generated_chunks),
        "ttft": first_choice_latency,
        "response_headers_latency": response_headers_latency,
        "first_chunk_latency": first_chunk_latency,
        "first_choice_latency": first_choice_latency,
        "itl": itl,
        "latency": latency,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "success": success,
        "request_id": request_id,
        "server_profile": server_profile,
    }
    return metrics


def print_metrics(metrics):
    print(metrics["generated_text"])
    if not metrics["success"]:
        print("警告: 未收到任何含choices的chunk，以下指标可能无效。")
    print(f"Headers latency: {_format_ms(metrics.get('response_headers_latency'))}")
    print(f"First chunk latency: {_format_ms(metrics.get('first_chunk_latency'))}")
    print(f"First choice latency: {_format_ms(metrics.get('first_choice_latency'))}")
    if metrics["itl"]:
        mean_itl = mean(metrics["itl"])
        median_itl = median(metrics["itl"])
    else:
        mean_itl = None
        median_itl = None
    print(f"Mean ITL: {_format_ms(mean_itl)}, Median ITL: {_format_ms(median_itl)}")
    print(f"总延迟: {_format_ms(metrics['latency'])}")
    prompt_tokens = metrics["prompt_tokens"]
    completion_tokens = metrics["completion_tokens"]
    prompt_tokens_str = prompt_tokens if prompt_tokens is not None else "N/A"
    completion_tokens_str = (
        completion_tokens if completion_tokens is not None else "N/A"
    )
    print(
        f"Prompt tokens: {prompt_tokens_str} | Completion tokens: {completion_tokens_str}"
    )
    server_profile = metrics.get("server_profile")
    if server_profile:
        print(
            "Server profile: "
            f"true TTFT={server_profile.get('server_true_ttft_ms', 0.0):.2f} ms, "
            f"arrival->blender={server_profile.get('server_blender_done_ms', 0.0):.2f} ms, "
            f"blend envelope={server_profile.get('server_blend_envelope_ms', 0.0):.2f} ms, "
            f"blend core={server_profile.get('server_blend_core_ms', 0.0):.2f} ms"
        )


async def main():
    args = parse_args()
    global server_profile_log_path
    server_profile_log_path = args.server_log
    tokenizer = AutoTokenizer.from_pretrained(model)

    # 更大的chunks - 每个约2000 tokens
    sys_prompt = tokenizer.encode("You are a very helpful assistant.")
    chunk1 = tokenizer.encode("Hello, how are you doing today? " * 300)[1:]
    chunk2 = tokenizer.encode("What's your favorite color? " * 300)[1:]
    chunk3 = tokenizer.encode("Tell me about yourself. " * 300)[1:]
    sep = tokenizer.encode(" # # ")[1:]
    question1 = tokenizer.encode("Please help me")[1:]
    question2 = tokenizer.encode("What do you think?")[1:]
    question3 = tokenizer.encode("Can you assist?")[1:]

    # prompt1 = chunk1
    # prompt2 = chunk2
    # prompt3 = chunk3

    prompt1 = sys_prompt  + chunk1  + chunk2  + chunk3  + question1
    prompt2 = sys_prompt + sep + chunk2 + sep + chunk1 + sep + chunk3 + sep + question2
    prompt3 = sys_prompt + sep + chunk3 + sep + chunk1 + sep + chunk2 + sep + question3

    request_info = [
        ("[第1次请求] 冷启动", prompt1),
        ("[第2次请求] 不同顺序（应该复用cache）", prompt2),
        ("[第3次请求] 再次不同顺序", prompt3),
    ]

    print("=" * 60)
    print("大Prompt测试（每个约6K+ tokens）")
    print("=" * 60)
    print(f"Prompt1总长度: {len(prompt1)} tokens")
    print(f"Request rate (RPS): {args.request_rate}")
    print(f"Burstiness: {args.burstiness}")
    print()

    async with aiohttp.ClientSession() as session:
        # 预热阶段：先store cache，等待系统稳定
        print("=" * 60)
        print("预热阶段：存储cache...")
        print("=" * 60)
        warmup_metrics = await stream_completion(
            session=session,
            prompt=prompt1,
            request_name="[预热-Store]"
        )
        print("[预热-Store] 完成")
        print_metrics(warmup_metrics)
        print()

        print("等待3秒后开始正式测试（测试retrieve性能）...")
        await asyncio.sleep(3)
        print()

        print("=" * 60)
        print("开始正式测试（只测试prompt2和prompt3的retrieve性能）")
        print("=" * 60)

        # 只测试prompt2和prompt3（它们会retrieve cache）
        prompts = [info[1] for info in request_info[1:]]  # 跳过prompt1
        tasks = []
        idx = 1  # 从request_info[1]开始
        async for prompt in send_requests_with_rate_limit(
            prompts, args.request_rate, args.burstiness
        ):
            request_name = request_info[idx][0]
            tasks.append(
                asyncio.create_task(
                    stream_completion(
                        session=session, prompt=prompt, request_name=request_name
                    )
                )
            )
            idx += 1

        metrics_list = await asyncio.gather(*tasks)

    print()
    print("=" * 60)
    print("正式测试结果")
    print("=" * 60)

    metrics2, metrics3 = metrics_list

    print(request_info[1][0])
    print_metrics(metrics2)
    if warmup_metrics["ttft"] is not None and metrics2["ttft"] is not None:
        print(f"相比预热加速比: {warmup_metrics['ttft']/metrics2['ttft']:.2f}x")
    else:
        print("加速比: 无法计算（缺少TTFT数据）")
    print()

    print(request_info[2][0])
    print_metrics(metrics3)
    if warmup_metrics["ttft"] is not None and metrics3["ttft"] is not None:
        print(f"相比预热加速比: {warmup_metrics['ttft']/metrics3['ttft']:.2f}x")
    else:
        print("加速比: 无法计算（缺少TTFT数据）")
    print()

    print("=" * 60)
    print("期望：更大的prompt应该看到更明显的加速")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
