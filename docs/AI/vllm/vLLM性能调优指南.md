# vLLM 性能调优指南

## 性能指标

在优化 vLLM 性能之前，需要了解关键的性能指标：

### 1. 吞吐量（Throughput）

- **定义**：单位时间内处理的请求数或 token 数
- **单位**：requests/s 或 tokens/s
- **重要性**：衡量系统整体处理能力
- **优化目标**：最大化吞吐量

### 2. 延迟（Latency）

- **首 token 延迟（TTFT）**：从请求到第一个 token 的时间
- **每 token 延迟（TPOT）**：生成每个 token 的平均时间
- **总延迟**：完成整个请求的时间
- **优化目标**：最小化延迟

### 3. 显存利用率

- **定义**：GPU 显存的使用比例
- **范围**：0-100%
- **最佳值**：80-95%
- **优化目标**：提高利用率，避免浪费

### 4. GPU 利用率

- **定义**：GPU 计算资源的使用比例
- **范围**：0-100%
- **最佳值**：大于90%
- **优化目标**：充分利用 GPU 计算能力

## 核心参数调优

### 1. GPU 内存利用率 (`--gpu-memory-utilization`)

**作用**：控制 vLLM 使用的 GPU 显存比例

**默认值**：0.9 (90%)

**调优策略**：

```bash
# 保守配置（稳定性优先）
--gpu-memory-utilization 0.8

# 平衡配置（推荐）
--gpu-memory-utilization 0.9

# 激进配置（性能优先）
--gpu-memory-utilization 0.95
```

**调优步骤**：

1. 从 0.95 开始测试
2. 如果出现 OOM（Out of Memory），降低到 0.9
3. 继续降低直到稳定运行
4. 在稳定的基础上，逐步提高以获得最佳性能

**注意事项**：

- 设置过高可能导致 OOM
- 设置过低会浪费显存，降低并发能力
- 不同模型和硬件需要不同的配置

### 2. 最大模型长度 (`--max-model-len`)

**作用**：限制模型处理的最大上下文长度

**默认值**：模型配置的最大长度

**调优策略**：

```bash
# 短文本场景
--max-model-len 2048

# 中等文本场景（推荐）
--max-model-len 4096

# 长文本场景
--max-model-len 8192

# 超长文本场景
--max-model-len 32768
```

**影响**：

- 越大：支持更长的上下文，但占用更多显存
- 越小：节省显存，可支持更多并发，但限制了输入长度

**建议**：

- 根据实际业务需求设置
- 不要设置过大，避免浪费显存
- 可以通过监控实际请求长度来优化

### 3. 最大并发序列数 (`--max-num-seqs`)

**作用**：限制同时处理的最大序列数

**默认值**：256

**调优策略**：

```bash
# 低并发场景
--max-num-seqs 64

# 中等并发场景（推荐）
--max-num-seqs 256

# 高并发场景
--max-num-seqs 512
```

**影响**：

- 越大：可处理更多并发请求，但每个请求可能等待更久
- 越小：每个请求响应更快，但总吞吐量降低

**调优方法**：

1. 监控实际并发请求数
2. 设置为实际并发数的 1.5-2 倍
3. 观察 GPU 利用率和延迟
4. 逐步调整到最佳值

### 4. 最大批处理 Token 数 (`--max-num-batched-tokens`)

**作用**：限制单次批处理的最大 token 数

**默认值**：与 `--max-model-len` 相同

**调优策略**：

```bash
# 小批量（低延迟优先）
--max-num-batched-tokens 2048

# 中批量（平衡）
--max-num-batched-tokens 4096

# 大批量（吞吐量优先）
--max-num-batched-tokens 8192
```

**影响**：

- 越大：吞吐量越高，但首 token 延迟可能增加
- 越小：延迟更低，但吞吐量降低

### 5. 张量并行大小 (`--tensor-parallel-size`)

**作用**：将模型分割到多个 GPU 上并行计算

**默认值**：1

**调优策略**：

```bash
# 单卡
--tensor-parallel-size 1

# 双卡
--tensor-parallel-size 2

# 四卡
--tensor-parallel-size 4

# 八卡
--tensor-parallel-size 8
```

**建议**：

- 设置为 GPU 数量
- 确保模型大小需要多卡才能运行
- 注意通信开销，不是越多越好

### 6. 数据类型 (`--dtype`)

**作用**：指定模型推理的数据类型

**选项**：

```bash
# 自动选择（推荐）
--dtype auto

# 半精度（推荐，平衡性能和精度）
--dtype bfloat16

# 半精度（兼容性好）
--dtype float16

# 全精度（精度最高，但慢）
--dtype float32
```

**对比**：

| 数据类型 | 显存占用 | 速度 | 精度 | 推荐场景 |
|---------|---------|------|------|---------|
| float32 | 100% | 慢 | 最高 | 研究、对精度要求极高 |
| float16 | 50% | 快 | 高 | 通用场景 |
| bfloat16 | 50% | 快 | 高 | 推荐，兼容性好 |

**建议**：

- 优先使用 `bfloat16`
- 如果 GPU 不支持，使用 `float16`
- 避免使用 `float32`，除非有特殊需求

## 高级优化技巧

### 1. 启用分块预填充 (`--enable-chunked-prefill`)

**作用**：将大型预填充操作拆分成更小的块，与解码请求一起批处理

**启用方式**：

```bash
vllm serve /data/models/Qwen2.5-7B-Instruct \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192
```

**优势**：

- 减少首 token 延迟
- 提高 GPU 利用率
- 更好的批处理效率

**适用场景**：

- 长文本输入
- 混合长短请求
- 需要低延迟的场景

### 2. 禁用日志请求 (`--disable-log-requests`)

**作用**：禁用请求日志，减少 I/O 开销

**启用方式**：

```bash
vllm serve /data/models/Qwen2.5-7B-Instruct \
  --disable-log-requests
```

**优势**：

- 减少 I/O 开销
- 提升性能（约 5-10%）

**注意**：

- 生产环境建议启用
- 调试时建议禁用

### 3. 使用量化模型

**作用**：通过量化减少显存占用，提升推理速度

**支持的量化方法**：

- **AWQ**：4-bit 量化，精度损失小
- **GPTQ**：4-bit 量化，兼容性好
- **SqueezeLLM**：3-bit 量化，压缩率高

**使用方式**：

```bash
# AWQ 量化
vllm serve /data/models/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq

# GPTQ 量化
vllm serve /data/models/Qwen2.5-7B-Instruct-GPTQ \
  --quantization gptq
```

**性能对比**：

| 量化方法 | 显存占用 | 速度 | 精度损失 |
|---------|---------|------|---------|
| 无量化 | 100% | 基准 | 0% |
| AWQ | 25-30% | 1.5-2x | 小于1% |
| GPTQ | 25-30% | 1.5-2x | 小于2% |

### 4. 调整采样参数

**作用**：优化生成质量和速度

**关键参数**：

```python
sampling_params = SamplingParams(
    temperature=0.7,          # 控制随机性，0-1
    top_p=0.9,               # 核采样，优先调节
    top_k=50,                # Top-K 采样
    repetition_penalty=1.1,  # 重复惩罚，大于1
    max_tokens=1000,         # 最大生成 token 数
    presence_penalty=0.0,    # 存在惩罚
    frequency_penalty=0.0    # 频率惩罚
)
```

**调优建议**：

- **temperature**：
  - 0.1-0.3：确定性强，适合事实性任务
  - 0.7-0.9：平衡，适合通用对话
  - 1.0+：创造性强，适合创作任务

- **top_p**：
  - 优先调节此参数，比 top_k 更有效
  - 0.9-0.95：推荐值
  - 越小越确定，越大越多样

- **repetition_penalty**：
  - 1.0：无惩罚
  - 1.1-1.2：轻度惩罚（推荐）
  - 1.5+：强惩罚，可能影响流畅性

## 性能监控

### 1. 使用内置监控

```bash
# 启用监控
vllm serve /data/models/Qwen2.5-7B-Instruct \
  --enable-metrics \
  --metrics-port 9090
```

访问 `http://localhost:9090/metrics` 查看指标。

### 2. 关键监控指标

- `vllm:num_requests_running`：正在运行的请求数
- `vllm:num_requests_waiting`：等待中的请求数
- `vllm:gpu_cache_usage_perc`：GPU 缓存使用率
- `vllm:time_to_first_token_seconds`：首 token 延迟
- `vllm:time_per_output_token_seconds`：每 token 延迟

### 3. 使用 Prometheus + Grafana

**Prometheus 配置** (`prometheus.yml`)：

```yaml
scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['localhost:9090']
```

**Grafana 仪表板**：

- 导入 vLLM 官方仪表板
- 监控吞吐量、延迟、GPU 使用率等

## 性能测试

### 1. 使用 vLLM 内置基准测试

```bash
# 吞吐量测试
python -m vllm.entrypoints.openai.api_server \
  --model /data/models/Qwen2.5-7B-Instruct \
  --benchmark

# 自定义测试
python benchmarks/benchmark_throughput.py \
  --model /data/models/Qwen2.5-7B-Instruct \
  --num-prompts 1000 \
  --input-len 128 \
  --output-len 128
```

### 2. 使用 wrk 压测

```bash
# 安装 wrk
sudo apt-get install wrk

# 压测
wrk -t 4 -c 100 -d 60s \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  --script post.lua \
  http://localhost:8000/v1/completions
```

**post.lua**：

```lua
wrk.method = "POST"
wrk.body = '{"model": "/data/models/Qwen2.5-7B-Instruct", "prompt": "你好", "max_tokens": 100}'
wrk.headers["Content-Type"] = "application/json"
```

## 调优流程

### 1. 基准测试

```bash
# 记录默认配置的性能
vllm serve /data/models/Qwen2.5-7B-Instruct
# 运行压测，记录吞吐量和延迟
```

### 2. 逐步调优

**步骤 1：优化显存利用率**

```bash
# 测试不同的 gpu-memory-utilization
for util in 0.8 0.85 0.9 0.95; do
  vllm serve /data/models/Qwen2.5-7B-Instruct \
    --gpu-memory-utilization $util
  # 运行压测，记录结果
done
```

**步骤 2：优化并发数**

```bash
# 测试不同的 max-num-seqs
for seqs in 64 128 256 512; do
  vllm serve /data/models/Qwen2.5-7B-Instruct \
    --max-num-seqs $seqs
  # 运行压测，记录结果
done
```

**步骤 3：优化批处理**

```bash
# 测试不同的 max-num-batched-tokens
for tokens in 2048 4096 8192; do
  vllm serve /data/models/Qwen2.5-7B-Instruct \
    --max-num-batched-tokens $tokens
  # 运行压测，记录结果
done
```

### 3. 综合优化

```bash
# 应用最佳配置
vllm serve /data/models/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --disable-log-requests
```

## 常见性能问题

### 1. 吞吐量低

**可能原因**：

- GPU 利用率低
- 并发数不足
- 批处理大小过小

**解决方案**：

- 增加 `--max-num-seqs`
- 增加 `--max-num-batched-tokens`
- 启用 `--enable-chunked-prefill`

### 2. 延迟高

**可能原因**：

- 批处理大小过大
- 并发数过高
- 模型过大

**解决方案**：

- 减小 `--max-num-batched-tokens`
- 减小 `--max-num-seqs`
- 使用量化模型

### 3. OOM（显存不足）

**可能原因**：

- `--gpu-memory-utilization` 过高
- `--max-model-len` 过大
- 并发数过高

**解决方案**：

- 降低 `--gpu-memory-utilization`
- 减小 `--max-model-len`
- 减小 `--max-num-seqs`
- 使用量化模型
- 使用多卡部署

## 总结

vLLM 性能调优是一个系统工程，需要：

1. **了解业务需求**：吞吐量优先还是延迟优先
2. **监控关键指标**：GPU 利用率、显存使用、吞吐量、延迟
3. **逐步调优**：从默认配置开始，逐步优化各个参数
4. **持续监控**：部署后持续监控，根据实际情况调整

**推荐配置**（通用场景）：

```bash
vllm serve /data/models/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --disable-log-requests \
  --enable-metrics
```

