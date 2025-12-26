#!/usr/bin/env python3
"""
CacheBlend测试 - 更小的chunk size
"""

import time
import requests
from transformers import AutoTokenizer


def main():
    api_base = "http://localhost:8000"
    model = "mistralai/Mistral-7B-Instruct-v0.2"

    print("=" * 60)
    print("CacheBlend测试 - 调整chunk大小")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model)

    # 减少重复次数：从500改成50
    sys_prompt = tokenizer.encode("You are a very helpful assistant.")
    chunk1 = tokenizer.encode("Hello, how are you?" * 50)[1:]  # 减少到50次
    chunk2 = tokenizer.encode("Hello, what's up?" * 50)[1:]
    chunk3 = tokenizer.encode("Hi, what are you up to?" * 50)[1:]
    sep = tokenizer.encode(" # # ")[1:]
    question1 = tokenizer.encode("Hello, my name is")[1:]
    question2 = tokenizer.encode("Hello, how are you?")[1:]
    question3 = tokenizer.encode("Hello, what's up?")[1:]

    prompt1 = sys_prompt + sep + chunk1 + sep + chunk2 + sep + chunk3 + sep + question1
    prompt2 = sys_prompt + sep + chunk2 + sep + chunk1 + sep + chunk3 + sep + question2
    prompt3 = sys_prompt + sep + chunk3 + sep + chunk1 + sep + chunk2 + sep + question3

    print(f"Chunk1长度: {len(chunk1)} tokens")
    print(f"Chunk2长度: {len(chunk2)} tokens")
    print(f"Chunk3长度: {len(chunk3)} tokens")
    print(f"Prompt1总长度: {len(prompt1)} tokens")
    print()

    # 请求1
    print("[第1次请求] 冷启动")
    start = time.time()
    r1 = requests.post(f"{api_base}/v1/completions", json={
        "model": model, "prompt": prompt1, "max_tokens": 10, "temperature": 0
    }, timeout=60)
    ttft1 = time.time() - start
    print(f"生成: {r1.json()['choices'][0]['text']}")
    print(f"TTFT: {ttft1:.2f}秒\n")

    time.sleep(1)

    # 请求2
    print("[第2次请求] 不同顺序")
    start = time.time()
    r2 = requests.post(f"{api_base}/v1/completions", json={
        "model": model, "prompt": prompt2, "max_tokens": 10, "temperature": 0
    }, timeout=60)
    ttft2 = time.time() - start
    print(f"result r2 :{r2.json()}")
    print(f"生成: {r2.json()['choices'][0]['text']}")
    print(f"TTFT: {ttft2:.2f}秒")
    print(f"加速比: {ttft1/ttft2:.2f}x\n")

    time.sleep(1)

    # 请求3
    print("[第3次请求] 再次不同顺序")
    start = time.time()
    r3 = requests.post(f"{api_base}/v1/completions", json={
        "model": model, "prompt": prompt3, "max_tokens": 10, "temperature": 0
    }, timeout=60)
    ttft3 = time.time() - start
    print(f"生成: {r3.json()['choices'][0]['text']}")
    print(f"TTFT: {ttft3:.2f}秒")
    print(f"加速比: {ttft1/ttft3:.2f}x\n")

    print("=" * 60)
    print(f"期望：第2、3次请求有明显加速")
    print("=" * 60)


if __name__ == "__main__":
    main()
