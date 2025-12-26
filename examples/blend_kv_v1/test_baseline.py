#!/usr/bin/env python3
"""对比测试：有LMCache vs 没有LMCache"""
import time
import requests

api_base = "http://localhost:8000"
model = "mistralai/Mistral-7B-Instruct-v0.2"

# 简单的重复prompt（模拟cache场景）
prompt = "Hello, how are you? " * 200  # 约1200 tokens

print("=" * 60)
print("Baseline测试：连续3次相同请求")
print("=" * 60)

times = []
for i in range(3):
    print(f"\n[请求{i+1}]")
    start = time.time()
    r = requests.post(f"{api_base}/v1/completions", json={
        "model": model,
        "prompt": prompt,
        "max_tokens": 10,
        "temperature": 0
    }, timeout=60)
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"耗时: {elapsed:.2f}秒")
    time.sleep(0.5)

print("\n" + "=" * 60)
print(f"请求1: {times[0]:.2f}秒 (冷启动)")
print(f"请求2: {times[1]:.2f}秒 (加速比: {times[0]/times[1]:.2f}x)")
print(f"请求3: {times[2]:.2f}秒 (加速比: {times[0]/times[2]:.2f}x)")
print("=" * 60)
