# LMCache CacheBlend Serving 使用指南

## 环境检查

✅ **环境已就绪**:
- vLLM版本: 0.11.0
- LMCache: 已安装
- vLLM源码: 已修改（gpu_worker.py已配置好LMCache集成）

## 快速启动

### 1. 启动Serving

```bash
cd /home/xujie/lmcache-v1/examples/blend_kv_v1
./start_serving.sh
```

**可选参数**（通过环境变量配置）:
```bash
# 使用不同的模型
MODEL="meta-llama/Llama-3.1-8B-Instruct" ./start_serving.sh

# 修改端口
PORT=8001 ./start_serving.sh

# 调整GPU内存使用率
GPU_MEM=0.9 ./start_serving.sh

# 调整最大长度
MAX_LEN=8192 ./start_serving.sh
```

### 2. 测试Serving（另开一个终端）

```bash
cd /home/xujie/lmcache-v1/examples/blend_kv_v1
python3 test_serving.py
```

**预期结果**:
- 第1次请求: TTFT较慢（冷启动）
- 第2、3次请求: TTFT明显降低（CacheBlend生效）
- 加速比应该 >1.5x

### 3. 使用cURL测试

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.2",
    "prompt": "Hello # # World",
    "max_tokens": 10,
    "temperature": 0
  }'
```

## 配置说明

### 核心环境变量

当前配置（在start_serving.sh中）:
```bash
LMCACHE_CHUNK_SIZE=256                    # Chunk大小
LMCACHE_ENABLE_BLENDING=True              # 启用Blending
LMCACHE_BLEND_SPECIAL_STR=" # # "        # Chunk分隔符
LMCACHE_USE_LAYERWISE=True                # 逐层处理
LMCACHE_BLEND_CHECK_LAYERS=1              # 检查层数
LMCACHE_BLEND_RECOMPUTE_RATIOS=0.15       # 重计算比例
LMCACHE_LOCAL_CPU=True                    # 使用CPU后端
LMCACHE_MAX_LOCAL_CPU_SIZE=5              # CPU缓存上限(GB)
```

### 切换到Disk后端

编辑 `start_serving.sh`，注释掉CPU配置，取消注释Disk配置:
```bash
# export LMCACHE_LOCAL_CPU=True
# export LMCACHE_MAX_LOCAL_CPU_SIZE=5

export LMCACHE_LOCAL_CPU=False
export LMCACHE_LOCAL_DISK=file://./local_disk/
export LMCACHE_MAX_LOCAL_DISK_SIZE=10
```

## 关键点

1. **分隔符一致性**: prompt中使用的分隔符必须与`LMCACHE_BLEND_SPECIAL_STR`完全一致
2. **Chunk组织**: 把可重用的context组织成chunks，用分隔符连接
3. **不同顺序复用**: 即使chunks顺序不同，也能复用cache（这是CacheBlend的核心优势）

## 客户端集成示例

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
sep = " # # "  # 必须与服务端一致

# 构建prompt
prompt = f"{sys_prompt}{sep}{chunk1}{sep}{chunk2}{sep}{question}"

response = client.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    prompt=prompt,
    max_tokens=100,
    temperature=0
)
print(response.choices[0].text)
```

### Chat Completions API

```python
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    messages=[
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"{chunk1}{sep}{chunk2}{sep}{question}"}
    ],
    temperature=0
)
print(response.choices[0].message.content)
```

## 性能调优

### 提高缓存命中率
- 增加CPU/Disk后端容量
- 合理设置chunk_size（默认256已是最佳实践）
- 尽量复用相同的context chunks

### 降低内存占用
- 减少`LMCACHE_MAX_LOCAL_CPU_SIZE`
- 使用Disk后端替代CPU后端
- 降低`gpu_memory_utilization`

### 加速推理
- 调整`LMCACHE_BLEND_RECOMPUTE_RATIOS`（0.1-0.2之间）
- 使用FlashInfer backend: `export VLLM_ATTENTION_BACKEND=FLASHINFER`

## 故障排查

### 启动失败
- 检查GPU是否被占用: `nvidia-smi`
- 检查端口是否被占用: `lsof -i :8000`
- 查看vLLM日志中的LMCache初始化信息

### CacheBlend不生效
- 确认分隔符配置一致
- 检查日志中是否有"CacheBlend enabled"
- 验证chunk是否被正确识别（查看日志）

### OOM错误
- 降低`gpu_memory_utilization` (0.7 或 0.6)
- 减小`max_model_len`
- 减少`LMCACHE_MAX_LOCAL_CPU_SIZE`

## 文件说明

- `start_serving.sh`: Serving启动脚本
- `test_serving.py`: 自动化测试脚本
- `blend.py`: 原始的离线推理示例
- `README.md`: vLLM源码修改说明（已完成）

## 生产部署建议

1. **多实例部署**: 使用P2P sharing实现跨实例cache共享
2. **监控**: 监控cache命中率、TTFT、吞吐量
3. **容量规划**: 根据业务场景合理配置cache容量
4. **日志**: 保留vLLM和LMCache的详细日志用于调试

## 下一步

如需要高级功能:
- **P2P Cache Sharing**: 参考 `examples/kv_cache_reuse/share_across_instances/p2p_sharing/`
- **Disaggregated Prefill**: 参考 `examples/disagg_prefill/`
- **自定义配置**: 使用YAML配置文件替代环境变量
