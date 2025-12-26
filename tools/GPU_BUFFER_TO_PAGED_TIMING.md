# GPU Buffer 到 Paged Memory 的拷贝时序

## 关键代码位置

`lmcache/v1/gpu_connector.py:500-566` 的 `batched_to_gpu` 方法

## 流水线设计

循环: `for layer_id in range(self.num_layers + 2)`  （假设32层模型，循环34次）

每次迭代执行**三个阶段**：

### 阶段1: 写入 Paged Memory (layer_id > 1)
```python
if layer_id > 1:
    lmc_ops.single_layer_kv_transfer(
        self.buffer_mapping[layer_id - 2].tensor,  # 从buffer读取
        self.kvcaches[layer_id - 2],               # 写入paged memory
        slot_mapping_full,
        ...
    )
    del self.buffer_mapping[layer_id - 2]
```
**写入 layer (layer_id - 2)**

### 阶段2: RoPE处理 (layer_id > 0 and layer_id <= num_layers)
```python
if layer_id > 0 and layer_id <= self.num_layers:
    torch.cuda.synchronize()

    # ping-pong交换
    compute_gpu_buffer_obj, load_gpu_buffer_obj = (
        load_gpu_buffer_obj, compute_gpu_buffer_obj
    )

    # RoPE恢复位置编码
    compute_gpu_buffer_obj.tensor[0] = self.fused_rotary_emb(...)

    # Gap置零
    compute_gpu_buffer_obj.tensor[:, self.current_gap_positions] = 0.0

    # 存入buffer_mapping供Blender使用
    self.buffer_mapping[layer_id - 1] = compute_gpu_buffer_obj
```
**处理 layer (layer_id - 1)**

### 阶段3: 加载CPU数据 (layer_id < num_layers)
```python
if layer_id < self.num_layers:
    memory_objs_layer = yield  # 等待外部传入CPU数据

    with torch.cuda.stream(self.load_stream):
        # 异步拷贝 CPU -> load_gpu_buffer
        load_gpu_buffer_obj.tensor[0][...].copy_(
            memory_obj.tensor[0], non_blocking=True
        )
```
**加载 layer (layer_id) 的CPU数据**

---

## 完整时序表

假设有32层（layer 0-31），循环34次（layer_id 0-33）：

| layer_id | 写入Paged (阶段1) | RoPE处理 (阶段2) | 加载CPU (阶段3) | 说明 |
|----------|------------------|-----------------|-----------------|------|
| **0**    | ❌ (layer_id≤1)  | ❌ (layer_id=0) | ✅ 加载 layer 0 | 启动流水线 |
| **1**    | ❌ (layer_id≤1)  | ✅ 处理 layer 0 | ✅ 加载 layer 1 | |
| **2**    | ✅ **写入 layer 0** | ✅ 处理 layer 1 | ✅ 加载 layer 2 | 第一次写入paged! |
| **3**    | ✅ **写入 layer 1** | ✅ 处理 layer 2 | ✅ 加载 layer 3 | |
| **4**    | ✅ **写入 layer 2** | ✅ 处理 layer 3 | ✅ 加载 layer 4 | |
| **...**  | ... | ... | ... | ... |
| **31**   | ✅ **写入 layer 29** | ✅ 处理 layer 30 | ✅ 加载 layer 31 | |
| **32**   | ✅ **写入 layer 30** | ✅ 处理 layer 31 | ❌ (layer_id≥32) | 停止加载 |
| **33**   | ✅ **写入 layer 31** | ❌ (layer_id>32) | ❌ (layer_id≥32) | 最后一次写入 |

---

## 关键时机

### 1. **第一次写入paged memory: layer_id = 2**
```
当处理到 layer 2 时:
  - 写入 layer 0 (已经过RoPE处理)
  - 处理 layer 1 (RoPE)
  - 加载 layer 2 (CPU -> GPU)
```

### 2. **滞后两层 (Layer lag = 2)**
```
当前处理 layer N:
  - 写入到paged的是 layer (N-2)
  - 这保证了RoPE等处理完成后再写入
```

### 3. **为什么需要滞后？**

因为流水线需要确保：
1. CPU数据先到 `load_buffer`
2. 交换buffer后在 `compute_buffer` 上做RoPE/Blending
3. 处理完成后才能写入paged memory

---

## 可视化流程（以layer 5为例）

```
迭代顺序：

layer_id=5: 加载layer 5的CPU数据
     ↓
layer_id=6: 处理layer 5 (RoPE + 放入buffer_mapping)
     ↓
layer_id=7: 写入layer 5到paged memory
            ↑
            这里！single_layer_kv_transfer被调用
```

---

## Blending的特殊处理

在blend模式下，当buffer在 `buffer_mapping` 中时：

```python
# Blender可以通过 gpu_connector.get_kv(layer_id) 访问
# 在 lmcache/v1/compute/blend/blender.py:70
old_k, old_v = self.gpu_connector.get_kv(layer_id)

# Blender在layer_id对应的buffer上做差分计算
# 然后更新会反映到buffer中

# 两个迭代后，这个buffer被写入paged memory
# 此时包含了blending的修改
```

---

## 总结

**拷贝到paged memory的时机：**

✅ **在 `layer_id = N+2` 时写入 layer N**
✅ **使用 `lmc_ops.single_layer_kv_transfer` 函数**
✅ **发生在 RoPE恢复和Blending处理之后**
✅ **每层写入一次，写入后立即从buffer_mapping删除**

这种设计实现了：
- **内存效率**：只需要2个buffer（load + compute）处理所有层
- **流水线并行**：加载、处理、写入三个阶段并行进行
- **正确性**：确保所有处理（RoPE/Blending）完成后再写入paged memory
