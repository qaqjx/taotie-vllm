# vLLM v1 KV Cache Shape 分析报告

## 问题描述

在执行 `examples/blend_kv_v1/blend.py` 时，`lmcache/integration/vllm/vllm_v1_adapter.py:770` 打印的 `kvcaches[0].size()` 返回：

```
torch.Size([2, 12038, 16, 8, 128])
```

疑问：**维度16从哪里来？**

---

## 答案总结

**维度16是 vLLM 的 `block_size`（每个paged block可以存储的token数量），不是 num_kv_heads！**

---

## 详细分析

### 1. Shape的每个维度含义

`torch.Size([2, 12038, 16, 8, 128])` 的完整含义：

| 维度位置 | 值    | 含义                           | 说明                               |
|---------|-------|--------------------------------|-----------------------------------|
| dim 0   | 2     | K和V                           | Key和Value两个cache               |
| dim 1   | 12038 | num_blocks                     | GPU上分配的paged cache block总数   |
| dim 2   | **16** | **block_size**                | **每个block可以存储的token数量**    |
| dim 3   | 8     | num_kv_heads                   | Mistral-7B的KV头数量（GQA）        |
| dim 4   | 128   | head_size                      | 每个attention head的维度           |

### 2. vLLM v1的KV Cache布局

#### 2.1 NHD vs HND

vLLM v1使用 **NHD布局**（N=num_tokens, H=num_heads, D=head_dim）：

- **NHD布局**：`[2, num_blocks, block_size, num_kv_heads, head_size]`
- **HND布局**：`[2, num_blocks, num_kv_heads, block_size, head_size]`（未使用）

你在日志中看到的这一行确认了使用NHD：
```
INFO ... [utils.py:114] Connectors do not specify a kv cache layout, defaulting to NHD.
```

#### 2.2 代码位置

Shape定义在 `vllm/v1/attention/backends/flash_attn.py:103-112`：

```python
def get_kv_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
) -> Tuple[int, int, int, int, int]:
    return (2, num_blocks, block_size, num_kv_heads, head_size)
```

### 3. Mistral-7B-Instruct-v0.2的配置

从模型config确认：

```python
num_attention_heads:     32   # Query heads
num_key_value_heads:     8    # KV heads (GQA - Grouped Query Attention)
hidden_size:             4096
head_dim:                128  # 4096 / 32 = 128
num_hidden_layers:       32
```

### 4. block_size的来源

`block_size` 是vLLM配置的参数（默认通常是16），定义在 `vllm/config/cache.py:42-48`：

```python
class CacheConfig:
    block_size: int = 16  # 每个paged block存储的token数量
    ...
```

你可以通过vLLM的参数修改block_size，但默认值是16。

### 5. KV Cache的初始化流程

在 `lmcache/integration/vllm/vllm_v1_adapter.py:675-685`：

```python
def _init_kv_caches_from_forward_context(self, forward_context: "ForwardContext"):
    for layer_name in forward_context.no_compile_layers:
        attn_layer = forward_context.no_compile_layers[layer_name]
        if not hasattr(attn_layer, "kv_cache"):
            continue

        if layer_name not in self.kv_caches:
            # 从vLLM的attention layer获取已分配的KV cache
            self.kv_caches[layer_name] = attn_layer.kv_cache[
                forward_context.virtual_engine
            ]
```

vLLM在worker初始化时已经分配了这些KV cache tensors：

1. `_allocate_kv_cache` (`vllm/v1/worker/gpu/attn_utils.py:69-87`)
   - 分配原始的paged buffer

2. `_reshape_kv_cache` (`vllm/v1/worker/gpu/attn_utils.py:89-127`)
   - 将buffer reshape成 `[2, num_blocks, block_size, num_kv_heads, head_size]`

3. `bind_kv_cache` (`vllm/v1/worker/utils.py:306-364`)
   - 绑定到attention layers

### 6. 为什么混淆了？

容易混淆的原因：

1. **HND布局预期**：如果使用HND布局，shape应该是：
   ```
   [2, 12038, 8, 16, 128]
   ```
   此时 dim 2 = num_kv_heads(8)，dim 3 = block_size(16)

2. **实际NHD布局**：vLLM实际使用NHD布局：
   ```
   [2, 12038, 16, 8, 128]
   ```
   此时 dim 2 = block_size(16)，dim 3 = num_kv_heads(8)

3. **误读**：如果按HND的顺序理解NHD的tensor，就会把block_size(16)误认为是num_kv_heads。

---

## 总结

### 关键结论

1. **维度16 = block_size**，不是 num_kv_heads
2. **维度8 = num_kv_heads**，是Mistral-7B使用GQA的KV头数量
3. vLLM v1使用 **NHD布局**：`[2, num_blocks, block_size, num_kv_heads, head_size]`
4. LMCache的KV shape定义（在adapter中）与vLLM的paged KV cache shape是不同的：
   - **LMCache**: `(num_layer, 2, chunk_size, num_kv_head, head_size)` = `(32, 2, 256, 8, 128)`
   - **vLLM paged**: `(2, num_blocks, block_size, num_kv_heads, head_size)` = `(2, 12038, 16, 8, 128)`

### 验证

总token容量：`12038 blocks × 16 tokens/block = 192608 tokens`

这与vLLM日志中的KV cache大小一致：
```
INFO ... [kv_cache_utils.py:1087] GPU KV cache size: 192,608 tokens
```

---

## 代码位置参考

| 功能 | 文件路径 |
|------|---------|
| Shape定义 | `vllm/v1/attention/backends/flash_attn.py:103-112` |
| Cache分配 | `vllm/v1/worker/gpu/attn_utils.py:69-87` |
| Cache reshape | `vllm/v1/worker/gpu/attn_utils.py:89-127` |
| Layout决策 | `vllm/v1/attention/backends/utils.py:458-485` |
| LMCache获取KV | `lmcache/integration/vllm/vllm_v1_adapter.py:675-685` |
| 打印语句 | `lmcache/integration/vllm/vllm_v1_adapter.py:770` |

---

**生成时间**: 2025-12-24
**测试环境**: CUDA device 1, Mistral-7B-Instruct-v0.2
**vLLM版本**: v0.11.0
**LMCache版本**: v0.3.6.dev62-ge34074df4
