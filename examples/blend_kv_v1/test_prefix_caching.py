#!/usr/bin/env python3
"""测试前缀复用场景（相同位置）"""
import time
import requests
from transformers import AutoTokenizer

api_base = "http://localhost:8000"
model = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model)

# 共同前缀 + 不同后缀（相同位置测试）
prefix = tokenizer.encode("You are a helpful assistant. " * 100)  # ~2000 tokens
sep = tokenizer.encode(" # # ")[1:]
suffix1 = tokenizer.encode("Question 1: What is AI?" * 100)[1:]
suffix2 = tokenizer.encode("Question 2: What is ML?" * 100)[1:]
suffix3 = tokenizer.encode("Question 3: What is DL?" * 100)[1:]

prompt1 = prefix + sep + suffix1
prompt2 = prefix + sep + suffix2
prompt3 = prefix + sep + suffix3

print("=" * 60)
print("前缀复用测试（相同位置，不同内容）")
print("=" * 60)
print(f"共同前缀: {len(prefix)} tokens")
print(f"Prompt1总长度: {len(prompt1)} tokens\n")

times = []

# 请求1
print("[第1次] 冷启动")
start = time.time()
r1 = requests.post(f"{api_base}/v1/completions", json={
    "model": model, "prompt": prompt1, "max_tokens": 5, "temperature": 0
}, timeout=120)
t1 = time.time() - start
times.append(t1)
print(f"TTFT: {t1:.3f}秒\n")
time.sleep(1)

# 请求2 - 相同前缀
print("[第2次] 相同前缀，不同后缀")
start = time.time()
r2 = requests.post(f"{api_base}/v1/completions", json={
    "model": model, "prompt": prompt2, "max_tokens": 5, "temperature": 0
}, timeout=120)
t2 = time.time() - start
times.append(t2)
print(f"TTFT: {t2:.3f}秒")
print(f"加速: {t1/t2:.2f}x\n")
time.sleep(1)

# 请求3 - 相同前缀
print("[第3次] 相同前缀，再次不同后缀")
start = time.time()
r3 = requests.post(f"{api_base}/v1/completions", json={
    "model": model, "prompt": prompt3, "max_tokens": 5, "temperature": 0
}, timeout=120)
t3 = time.time() - start
times.append(t3)
print(f"TTFT: {t3:.3f}秒")
print(f"加速: {t1/t3:.2f}x\n")

print("=" * 60)
print("总结：")
print(f"  冷启动: {t1:.3f}秒")
print(f"  平均热启动: {(t2+t3)/2:.3f}秒")
print(f"  平均加速: {t1/((t2+t3)/2):.2f}x")
print("期望：前缀复用应该有显著加速")
print("=" * 60)
