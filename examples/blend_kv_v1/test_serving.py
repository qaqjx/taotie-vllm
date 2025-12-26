#!/usr/bin/env python3
"""
CacheBlend Serving测试客户端

演示如何使用OpenAI API调用LMCache CacheBlend serving
"""

import os
import time
from openai import OpenAI


def main():
    # 配置
    api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
    model = os.getenv("MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

    # 特殊分隔符（必须与serving端的LMCACHE_BLEND_SPECIAL_STR一致）
    sep = " # # "

    # 初始化客户端
    client = OpenAI(
        api_key="EMPTY",
        base_url=api_base
    )

    print("=" * 60)
    print("LMCache CacheBlend Serving 测试")
    print("=" * 60)
    print(f"API Base: {api_base}")
    print(f"Model: {model}")
    print(f"Separator: '{sep}'")
    print("=" * 60)

    # 定义共享的chunks
    sys_prompt = "You are a very helpful assistant."
    chunk1 = "Hello, how are you?" * 500
    chunk2 = "Hello, what's up?" * 500
    chunk3 = "Hi, what are you up to?" * 500

    # 第一次请求
    print("\n[第1次请求] 顺序: sys + chunk1 + chunk2 + chunk3")
    prompt1 = f"{sys_prompt}{sep}{chunk1}{sep}{chunk2}{sep}{chunk3}{sep}Hello, my name is"
    start = time.time()
    response1 = client.completions.create(
        model=model,
        prompt=prompt1,
        max_tokens=10,
        temperature=0
    )
    ttft1 = time.time() - start
    print(f"生成文本: {response1.choices[0].text}")
    print(f"TTFT: {ttft1:.2f}秒")

    time.sleep(1)

    # 第二次请求（不同顺序，应该复用cache）
    print("\n[第2次请求] 顺序: sys + chunk2 + chunk1 + chunk3 (应该复用cache)")
    prompt2 = f"{sys_prompt}{sep}{chunk2}{sep}{chunk1}{sep}{chunk3}{sep}Hello, how are you?"
    start = time.time()
    response2 = client.completions.create(
        model=model,
        prompt=prompt2,
        max_tokens=10,
        temperature=0
    )
    ttft2 = time.time() - start
    print(f"生成文本: {response2.choices[0].text}")
    print(f"TTFT: {ttft2:.2f}秒")
    print(f"加速比: {ttft1/ttft2:.2f}x")

    time.sleep(1)

    # 第三次请求（再次不同顺序）
    print("\n[第3次请求] 顺序: sys + chunk3 + chunk1 + chunk2 (应该复用cache)")
    prompt3 = f"{sys_prompt}{sep}{chunk3}{sep}{chunk1}{sep}{chunk2}{sep}Hello, what's up?"
    start = time.time()
    response3 = client.completions.create(
        model=model,
        prompt=prompt3,
        max_tokens=10,
        temperature=0
    )
    ttft3 = time.time() - start
    print(f"生成文本: {response3.choices[0].text}")
    print(f"TTFT: {ttft3:.2f}秒")
    print(f"加速比: {ttft1/ttft3:.2f}x")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("如果TTFT明显降低，说明CacheBlend正常工作")
    print("=" * 60)


if __name__ == "__main__":
    main()
