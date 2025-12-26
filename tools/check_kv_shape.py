#!/usr/bin/env python3
"""直接检查vLLM KV cache shape"""
import os
import sys

# 设置CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 最小化配置
os.environ["LMCACHE_CHUNK_SIZE"] = "256"
os.environ["LMCACHE_ENABLE_BLENDING"] = "False"  # 关闭blending简化测试
os.environ["LMCACHE_USE_LAYERWISE"] = "False"  # 关闭layerwise简化测试
os.environ["LMCACHE_LOCAL_CPU"] = "True"

from transformers import AutoConfig
print("=" * 80)
print("1. Mistral-7B-Instruct-v0.2 模型配置")
print("=" * 80)

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
config = AutoConfig.from_pretrained(model_name)

print(f"num_attention_heads:     {config.num_attention_heads}")
print(f"num_key_value_heads:     {config.num_key_value_heads}")
print(f"hidden_size:             {config.hidden_size}")
print(f"head_dim:                {config.hidden_size // config.num_attention_heads}")
print(f"num_hidden_layers:       {config.num_hidden_layers}")

print("\n" + "=" * 80)
print("2. LMCache配置 (从vLLM v1 adapter推断)")
print("=" * 80)

# 模拟vLLM的计算
num_kv_head = config.num_key_value_heads  # 应该是8
head_size = config.hidden_size // config.num_attention_heads  # 应该是128
chunk_size = 256
num_layer = config.num_hidden_layers  # 32

print(f"LMCache kv_shape: ({num_layer}, 2, {chunk_size}, {num_kv_head}, {head_size})")
print(f"  - num_layer:   {num_layer}")
print(f"  - K and V:     2")
print(f"  - chunk_size:  {chunk_size}")
print(f"  - num_kv_head: {num_kv_head}")
print(f"  - head_size:   {head_size}")

print("\n" + "=" * 80)
print("3. vLLM v1 Paged KV Cache格式（推测）")
print("=" * 80)

print("""
vLLM v1的paged KV cache可能使用以下格式之一：

格式A (NHD - Num heads, Head dim):
  [2, num_blocks, num_kv_heads, block_size, head_size]

格式B (HND - Head dim, Num heads):
  [2, num_blocks, head_size, num_kv_heads, block_size]

如果你看到的shape是 [2, 12038, 16, 8, 128]：
  - 2:      K和V
  - 12038:  GPU上分配的总block数
  - 16:     可能是 num_kv_heads (但配置显示应该是8) 或 block_size的两倍
  - 8:      可能是 block_size 或 num_kv_heads
  - 128:    head_size (确定)

可能的解释：
1. block_size = 16，num_kv_heads = 8 (HND格式)
2. num_kv_heads = 16由于某些并行配置 (NHD格式)
3. 某种reshaping或合并后的格式
""")

print("\n" + "=" * 80)
print("4. 结论")
print("=" * 80)

print("""
维度16的来源可能是：
1. vLLM的block_size设置（默认通常是16）
2. 如果使用了某种KV cache布局变换

要确认：请查看blend.py的完整输出中的这一行：
  INFO ... [kv_cache_utils.py] Connectors do not specify a kv cache layout, defaulting to NHD.

这表明vLLM使用NHD (Num heads, Head dim) 布局。

在NHD布局下，shape应该是：
  [2, num_blocks, num_kv_heads, block_size, head_size]
  [2, 12038,      8,             16,         128]

但你看到的是 [2, 12038, 16, 8, 128]，这表明可能：
- 某处有维度顺序错误
- 或者 block_size 和 num_kv_heads 被交换了
""")
