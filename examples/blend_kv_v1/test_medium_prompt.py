#!/usr/bin/env python3
"""中等大小prompt测试 - 避免崩溃"""
import time
import requests
from transformers import AutoTokenizer

api_base = "http://localhost:8000"
model = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model)

# 中等大小的chunks - 每个约800 tokens（安全范围）
sys_prompt = tokenizer.encode("You are a helpful assistant.")
chunk1 = tokenizer.encode("Hello, how are you? " * 120)[1:]  # ~800 tokens
chunk2 = tokenizer.encode("What's up? " * 120)[1:]
chunk3 = tokenizer.encode("How can I help? " * 120)[1:]
sep = tokenizer.encode(" # # ")[1:]
q1 = tokenizer.encode("Please help me")[1:]
q2 = tokenizer.encode("Thank you")[1:]
q3 = tokenizer.encode("Great")[1:]

prompt1 = sys_prompt + sep + chunk1 + sep + chunk2 + sep + chunk3 + sep + q1
prompt2 = sys_prompt + sep + chunk2 + sep + chunk1 + sep + chunk3 + sep + q2
prompt3 = sys_prompt + sep + chunk3 + sep + chunk1 + sep + chunk2 + sep + q3

print("=" * 60)
print("中等Prompt测试（约2.5K tokens，安全范围）")
print("=" * 60)
print(f"Prompt1长度: {len(prompt1)} tokens")
print()

times = []

# 请求1
print("[第1次] 冷启动")
start = time.time()
r1 = requests.post(f"{api_base}/v1/completions", json={
    "model": model, "prompt": prompt1, "max_tokens": 5, "temperature": 0
}, timeout=60)
t1 = time.time() - start
times.append(t1)
print(f"TTFT: {t1:.3f}秒\n")
time.sleep(1)

# 请求2
print("[第2次] 不同顺序")
start = time.time()
r2 = requests.post(f"{api_base}/v1/completions", json={
    "model": model, "prompt": prompt2, "max_tokens": 5, "temperature": 0
}, timeout=60)
t2 = time.time() - start
times.append(t2)
print(f"TTFT: {t2:.3f}秒")
print(f"加速: {t1/t2:.2f}x\n")
time.sleep(1)

# 请求3
print("[第3次] 再次不同顺序")
start = time.time()
r3 = requests.post(f"{api_base}/v1/completions", json={
    "model": model, "prompt": prompt3, "max_tokens": 5, "temperature": 0
}, timeout=60)
t3 = time.time() - start
times.append(t3)
print(f"TTFT: {t3:.3f}秒")
print(f"加速: {t1/t3:.2f}x\n")

print("=" * 60)
print("总结：")
print(f"  冷启动: {t1:.3f}秒")
print(f"  平均热启动: {(t2+t3)/2:.3f}秒")
print(f"  平均加速: {t1/((t2+t3)/2):.2f}x")
print("=" * 60)
