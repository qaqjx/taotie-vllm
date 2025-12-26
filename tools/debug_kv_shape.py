#!/usr/bin/env python3
"""调试脚本：查看KV cache shape的来源"""
# SPDX-License-Identifier: Apache-2.0
import os
import sys

# 设置环境变量
os.environ["LMCACHE_CHUNK_SIZE"] = "256"
os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
os.environ["LMCACHE_BLEND_SPECIAL_STR"] = "# #"
os.environ["LMCACHE_USE_LAYERWISE"] = "True"
os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = "1"
os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = "0.15"
os.environ["LMCACHE_LOCAL_CPU"] = "True"
os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"

# 指定使用cuda:1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs
from dataclasses import asdict

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

print("=" * 80)
print("1. 查看模型配置")
print("=" * 80)
config = AutoConfig.from_pretrained(model_name)
print(f"模型: {model_name}")
print(f"  num_attention_heads: {config.num_attention_heads}")
print(f"  num_key_value_heads: {config.num_key_value_heads}")
print(f"  hidden_size: {config.hidden_size}")
print(f"  head_dim: {config.hidden_size // config.num_attention_heads}")

print("\n" + "=" * 80)
print("2. 初始化vLLM并检查KV cache shape")
print("=" * 80)

lmcache_connector = "LMCacheConnectorV1"
ktc = KVTransferConfig(
    kv_connector=lmcache_connector,
    kv_role="kv_both",
)

llm_args = EngineArgs(
    model=model_name,
    kv_transfer_config=ktc,
    max_model_len=8192,
    gpu_memory_utilization=0.5,
    enable_prefix_caching=False,
    enforce_eager=True,
)

print(f"正在初始化LLM...")
llm = LLM(**asdict(llm_args))

# 获取cache config信息
cache_config = llm.llm_engine.model_config
print(f"\nvLLM cache配置:")
print(f"  block_size: {llm.llm_engine.cache_config.block_size}")
print(f"  num_gpu_blocks: {llm.llm_engine.cache_config.num_gpu_blocks}")

print("\n" + "=" * 80)
print("3. 执行简单推理并检查KV cache")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = tokenizer.encode("Hello world" * 10)[1:]
sampling_params = SamplingParams(temperature=0, max_tokens=1)

print(f"提示词长度: {len(prompt)} tokens")
print("正在生成...")

outputs = llm.generate(
    prompts={"prompt_token_ids": prompt},
    sampling_params=sampling_params
)

print("生成完成")
print("\n推理过程中的KV cache shape已在adapter中打印")

print("\n" + "=" * 80)
print("解释:")
print("=" * 80)
print("""
vLLM v1的paged KV cache格式是:
  [2, num_blocks, num_kv_heads, block_size, head_size]

其中:
  - 第1维度(2): 分别存储K和V
  - 第2维度(num_blocks): GPU上分配的KV cache块总数
  - 第3维度(num_kv_heads): KV头的数量（对于Mistral-7B是8，但使用GQA时可能不同）
  - 第4维度(block_size): 每个块的token数（通常是16）
  - 第5维度(head_size): 每个头的维度（128）

如果你看到的shape是 [2, 12038, 16, 8, 128]，说明:
  - 2: K和V两个缓存
  - 12038: 总共分配了12038个blocks
  - 16: num_kv_heads = 16 (这可能是因为tensor并行或配置问题)
  - 8: block_size = 8
  - 128: head_size = 128

注意: LMCache的kv_shape和vLLM的paged KV cache shape是不同的！
  - LMCache kv_shape: (num_layer, 2, chunk_size, num_kv_head, head_size)
  - vLLM paged KV: (2, num_blocks, num_kv_head, block_size, head_size)
""")
