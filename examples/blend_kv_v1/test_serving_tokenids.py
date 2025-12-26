#!/usr/bin/env python3
"""
CacheBlend Serving测试客户端 - 使用vLLM Python API

注意：这个脚本需要在serving启动后运行，使用vLLM的Python SDK
"""

import os
import time
import requests
from transformers import AutoTokenizer


def test_with_openai_api():
    """使用OpenAI API测试（通过HTTP）"""
    api_base = "http://localhost:8000"
    model = "mistralai/Mistral-7B-Instruct-v0.2"

    print("=" * 60)
    print("方法1: 通过vLLM HTTP API直接发送token IDs")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model)

    # 准备token IDs（跳过BOS token）
    sys_prompt = tokenizer.encode("You are a very helpful assistant.")
    chunk1 = tokenizer.encode("Hello, how are you?" * 500)[1:]
    chunk2 = tokenizer.encode("Hello, what's up?" * 500)[1:]
    chunk3 = tokenizer.encode("Hi, what are you up to?" * 500)[1:]
    sep = tokenizer.encode(" # # ")[1:]  # 跳过BOS
    question1 = tokenizer.encode("Hello, my name is")[1:]
    question2 = tokenizer.encode("Hello, how are you?")[1:]
    question3 = tokenizer.encode("Hello, what's up?")[1:]

    # 构建三个不同顺序的prompts
    prompt1 = sys_prompt + sep + chunk1 + sep + chunk2 + sep + chunk3 + sep + question1
    prompt2 = sys_prompt + sep + chunk2 + sep + chunk1 + sep + chunk3 + sep + question2
    prompt3 = sys_prompt + sep + chunk3 + sep + chunk1 + sep + chunk2 + sep + question3

    print(f"Prompt 1 长度: {len(prompt1)} tokens")
    print(f"Prompt 2 长度: {len(prompt2)} tokens")
    print(f"Prompt 3 长度: {len(prompt3)} tokens")
    print(f"分隔符 tokens: {sep}")
    print()

    # 第一次请求
    print("[第1次请求] 顺序: sys + chunk1 + chunk2 + chunk3")
    start = time.time()
    response = requests.post(
        f"{api_base}/v1/completions",
        json={
            "model": model,
            "prompt": prompt1,
            "max_tokens": 10,
            "temperature": 0,
        },
        timeout=60
    )
    ttft1 = time.time() - start
    result1 = response.json()
    print(f"生成文本: {result1['choices'][0]['text']}")
    print(f"TTFT: {ttft1:.2f}秒")
    print()

    time.sleep(1)

    # 第二次请求
    print("[第2次请求] 顺序: sys + chunk2 + chunk1 + chunk3 (应该复用cache)")
    start = time.time()
    response = requests.post(
        f"{api_base}/v1/completions",
        json={
            "model": model,
            "prompt": prompt2,
            "max_tokens": 10,
            "temperature": 0,
        },
        timeout=60
    )
    ttft2 = time.time() - start
    result2 = response.json()
    print(f"生成文本: {result2['choices'][0]['text']}")
    print(f"TTFT: {ttft2:.2f}秒")
    print(f"加速比: {ttft1/ttft2:.2f}x")
    print()

    time.sleep(1)

    # 第三次请求
    print("[第3次请求] 顺序: sys + chunk3 + chunk1 + chunk2 (应该复用cache)")
    start = time.time()
    response = requests.post(
        f"{api_base}/v1/completions",
        json={
            "model": model,
            "prompt": prompt3,
            "max_tokens": 10,
            "temperature": 0,
        },
        timeout=60
    )
    ttft3 = time.time() - start
    result3 = response.json()
    print(f"生成文本: {result3['choices'][0]['text']}")
    print(f"TTFT: {ttft3:.2f}秒")
    print(f"加速比: {ttft1/ttft3:.2f}x")
    print()

    print("=" * 60)
    print("测试完成!")
    print(f"如果第2、3次请求明显加速（>2x），说明CacheBlend工作正常")
    print("=" * 60)


if __name__ == "__main__":
    test_with_openai_api()
