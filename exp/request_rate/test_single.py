#!/usr/bin/env python3
"""
单次测试脚本：启动服务器，发送几个请求，查看 debug 输出
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

import aiohttp
from transformers import AutoTokenizer

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
PORT = 12345
DATASET = "/home/xujie/TaoTie/dataset/needle_multi_rag.jsonl"
SEP_STR = " # # "


def load_contexts(path: str, limit: int = 5):
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
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 32,
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


def kill_server():
    subprocess.run(["pkill", "-9", "-f", "vllm"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "EngineCore"], capture_output=True)
    time.sleep(3)


def start_server(compress_type: str, gpu: int = 1):
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
    log_file = log_dir / f"test_{compress_type}_{PORT}.log"

    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)

    print(f"[Server] Starting with {compress_type}, log: {log_file}")
    return proc, log_file


def wait_server(timeout=180):
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


async def run_test(compress_type: str, gpu: int = 1):
    print(f"\n{'='*60}")
    print(f"Testing: {compress_type}")
    print(f"{'='*60}")

    kill_server()
    proc, log_file = start_server(compress_type, gpu)

    try:
        if not wait_server():
            return

        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        sep_tokens = tokenizer.encode(SEP_STR, add_special_tokens=False)
        contexts = load_contexts(DATASET, 5)
        print(f"[Data] Loaded {len(contexts)} contexts")

        connector = aiohttp.TCPConnector(limit=100)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Pre-encode all contexts (without sep_tokens prefix for hash consistency)
            ctx_tokens_list = [tokenizer.encode(ctx, add_special_tokens=False) for ctx in contexts]

            # Warmup: store contexts - send ONLY context tokens (no sep prefix)
            # This matches what blend will extract when splitting by sep_tokens
            print("\n[Warmup] Storing contexts...")
            for i, ctx_tokens in enumerate(ctx_tokens_list):
                # Send context WITHOUT sep_tokens prefix to match blend's extraction
                ttft, err = await send_request(session, ctx_tokens)
                status = f"{ttft*1000:.0f}ms" if ttft else f"FAIL: {err}"
                print(f"  [{i+1}/{len(ctx_tokens_list)}] tokens={len(ctx_tokens)}, {status}")
                await asyncio.sleep(0.3)

            await asyncio.sleep(3)

            # Test: send a blending request (3 contexts)
            print("\n[Test] Sending blend request with 3 contexts...")
            question = tokenizer.encode("Answer the question.", add_special_tokens=False)
            tokens = []
            for ctx_tokens in ctx_tokens_list[:3]:
                tokens.extend(sep_tokens)
                tokens.extend(ctx_tokens)
            tokens.extend(sep_tokens)
            tokens.extend(question)

            print(f"  Prompt tokens: {len(tokens)}")
            ttft, err = await send_request(session, tokens)
            if ttft:
                print(f"  TTFT: {ttft*1000:.1f}ms")
            else:
                print(f"  FAILED: {err}")

    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except:
            pass
        kill_server()

        # Print last 50 lines of log
        print(f"\n[Log] Last 50 lines of {log_file}:")
        try:
            with open(log_file) as f:
                lines = f.readlines()
                for line in lines[-50:]:
                    print(line.rstrip())
        except:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="KIVI_2BIT", help="Compression method")
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()

    asyncio.run(run_test(args.method, args.gpu))


if __name__ == "__main__":
    main()
