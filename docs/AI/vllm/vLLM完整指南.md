# vLLM 完整指南

## 什么是 vLLM？

vLLM（Vectorized Large Language Model Serving System）是由加州大学伯克利分校 LMSYS 团队开发的**高性能大语言模型推理引擎**。它专注于通过创新的内存管理和计算优化技术，实现高吞吐、低延迟、低成本的模型服务。

### 核心特点

- **高性能推理**：支持分布式推理，能高效利用多机多卡资源
- **显存优化**：采用 PagedAttention 内存管理技术，显著提升 GPU 显存利用率
- **多场景适配**：无论是低延迟的在线服务，还是资源受限的边缘部署，vLLM 都能提供卓越的性能表现
- **OpenAI 兼容**：提供与 OpenAI API 兼容的接口，方便迁移和集成

### 官方资源

- **中文站点**：https://vllm.hyper.ai/docs/
- **英文站点**：https://docs.vllm.ai/en/latest/index.html
- **GitHub**：https://github.com/vllm-project/vllm

## vLLM vs Ollama：对比分析

在 LLM 推理引擎的选择上，vLLM 和 Ollama 是两个常见的选项。

| 对比维度 | Ollama | vLLM | 备注 |
|---------|--------|------|------|
| **量化与压缩策略** | 默认采用 4-bit/8-bit 量化，显存占用降至 25%-50% | 默认使用 FP16/BF16 精度，保留完整参数精度 | Ollama 牺牲精度换显存，vLLM 牺牲显存换计算效率 |
| **优化目标** | 轻量化和本地部署，动态加载模型分块，按需使用显存 | 高吞吐量、低延迟，预加载完整模型到显存，支持高并发 | Ollama 适合单任务，vLLM 适合批量推理 |
| **显存管理机制** | 分块加载 + 动态缓存，仅保留必要参数和激活值 | PagedAttention + 全量预加载，保留完整参数和中间激活值 | vLLM 显存占用为 Ollama 的 2-5 倍 |
| **硬件适配** | 针对消费级 GPU（如 RTX 3060）优化，显存需求低 | 依赖专业级 GPU（如 A100/H100），需多卡并行或分布式部署 | Ollama 可在 24GB 显存运行 32B 模型，vLLM 需至少 64GB |
| **性能与资源平衡** | 显存占用低，但推理速度较慢（适合轻量级应用） | 显存占用高，但吞吐量高（适合企业级服务） | 量化后 Ollama 速度可提升，但仍低于 vLLM |
| **适用场景** | 个人开发、本地测试、轻量级应用 | 企业级 API 服务、高并发推理、大规模部署 | 根据显存和性能需求选择框架 |

### DeepSeek-R1-Distill-Qwen-32B 模型对比

| 指标 | Ollama (4-bit) | vLLM (FP16) | 说明 |
|------|----------------|-------------|------|
| **显存占用** | 19-24 GB | 64-96 GB | Ollama 通过 4-bit 量化压缩参数，vLLM 需保留完整 FP16 参数和激活值 |
| **存储空间** | 20 GB | 64 GB | Ollama 存储量化后模型，vLLM 存储原始 FP16 精度模型 |
| **推理速度** | 较低（5-15 tokens/s） | 中高（30-60 tokens/s） | Ollama 因量化计算效率降低，vLLM 通过批处理和并行优化提升吞吐量 |
| **硬件门槛** | 高端消费级 GPU（≥24GB） | 多卡专业级 GPU（如 2×A100 80GB） | Ollama 勉强单卡运行，vLLM 需多卡并行或分布式部署 |

**总结**：Ollama 更适合个人开发和轻量级应用，而 vLLM 则更适合企业级服务和高并发场景。

## 核心技术：PagedAttention

PagedAttention 是 vLLM 最核心的技术创新，它解决了大型语言模型推理过程中的内存管理难题。

### 传统 KV Cache 的问题

在 LLM 推理时，KV Cache（存储注意力机制的 Key-Value 对）会占用大量显存，且由于请求长度不一，容易造成：

1. **显存占用增长快**：KV Cache 占用迅速增长，极易耗尽 GPU 内存
2. **内存碎片严重**：不同长度的请求导致内存碎片化
3. **缓存难以复用**：无法有效复用已计算的 KV Cache

### PagedAttention 的解决方案

PagedAttention 的设计灵感来自操作系统的虚拟内存分页管理技术：

- **分页管理**：将 KV Cache 分割成固定大小的块（pages），类似操作系统的内存页
- **按需分配**：只在需要时分配内存页，避免预先分配大块连续内存
- **高效复用**：不同请求可以共享相同的 KV Cache 页，提高内存利用率
- **减少碎片**：通过页表管理，减少内存碎片

### 性能提升

- **吞吐量提升**：相比 HuggingFace Transformers 提升高达 24 倍
- **显存利用率**：显著提高 GPU 显存利用率，可在相同硬件上处理更多并发请求
- **延迟降低**：通过优化内存管理，降低推理延迟

## 安装部署

### 环境要求

- **操作系统**：Linux（推荐 Ubuntu 20.04+）
- **Python**：3.8-3.11
- **GPU**：NVIDIA GPU with CUDA 11.8+
- **显存**：根据模型大小而定（建议 ≥24GB）

### 使用 pip 安装

```bash
# 安装 vLLM
pip install vllm

# 或者从源码安装最新版本
pip install git+https://github.com/vllm-project/vllm.git
```

### 使用 Docker 部署

#### 1. 安装 NVIDIA Container Toolkit

```bash
# 更新软件包列表并安装 NVIDIA 容器工具包
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# 配置 NVIDIA 容器运行时
sudo nvidia-ctk runtime configure --runtime=docker

# 重加载系统服务并重启 Docker
sudo systemctl daemon-reload
sudo systemctl restart docker
```

#### 2. 拉取 vLLM 镜像

```bash
docker pull vllm/vllm-openai:latest
```

#### 3. 启动 vLLM 容器

```bash
docker run -itd --restart=always --name vllm_service \
  -v /path/to/models:/models \
  -p 8000:8000 \
  --gpus all \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model /models/your-model \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --api-key your-api-key
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--restart=always` | 容器退出后自动重启 |
| `-v /path/to/models:/models` | 挂载模型目录 |
| `-p 8000:8000` | 端口映射 |
| `--gpus all` | 使用所有 GPU |
| `--ipc=host` | 共享主机 IPC，提升并行性能 |
| `--model` | 模型路径 |
| `--dtype` | 数据类型（auto, half, float16, bfloat16, float, float32） |
| `--gpu-memory-utilization` | GPU 内存使用率（0.7-0.95） |
| `--tensor-parallel-size` | 张量并行大小（GPU 数量） |
| `--max-model-len` | 最大上下文长度 |
| `--api-key` | API 密钥 |

## 使用方法

### Python API

```python
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    max_model_len=4096
)

# 设置采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=1000
)

# 生成文本
prompts = ["北京的著名景点有哪些？"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

### OpenAI 兼容 API

启动 OpenAI 兼容服务器：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --api-key your-api-key
```

使用 curl 调用：

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "prompt": "北京的著名景点有哪些？",
    "max_tokens": 1000,
    "temperature": 0.7
  }'
```

使用 Python OpenAI 客户端：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key"
)

response = client.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    prompt="北京的著名景点有哪些？",
    max_tokens=1000,
    temperature=0.7
)

print(response.choices[0].text)
```

## 性能优化

### 关键参数调优

#### 1. GPU 内存利用率 (`--gpu-memory-utilization`)

- **范围**：0.7-0.95
- **建议**：从 0.95 开始降低，直到效果达到最佳
- **说明**：限制模型使用的 GPU 内存占比，避免因内存不足导致服务崩溃

#### 2. 张量并行 (`--tensor-parallel-size`)

- **说明**：将模型分割到多个 GPU 上进行并行计算
- **建议**：设置为 GPU 数量
- **示例**：8 卡 A100 设置为 8

#### 3. 数据类型 (`--dtype`)

- **auto**：根据模型类型自动选择精度
- **bfloat16**：推荐，平衡精度和性能
- **float16**：半精度，节省显存
- **float32**：全精度，占用显存大

#### 4. 最大上下文长度 (`--max-model-len`)

- **说明**：限制模型在一次推理中能处理的最大输入长度
- **建议**：根据实际需求设置，避免过长输入导致性能问题

#### 5. 采样参数

```python
sampling_params = SamplingParams(
    temperature=0.7,      # 控制随机性，0-1 之间
    top_p=0.9,           # 核采样，优先调节此参数
    top_k=50,            # Top-K 采样
    repetition_penalty=1.1,  # 重复惩罚，大于 1
    max_tokens=1000      # 最大生成 token 数
)
```

### 最佳实践

1. **先测试后部署**：在实际部署前进行充分测试，确定最佳配置
2. **监控资源使用**：使用 Prometheus + Grafana 监控 GPU 使用率、吞吐量等指标
3. **分块预填充**：启用 Chunked Prefill，将大型预填充操作拆分成更小的块
4. **批处理优化**：通过 `max_num_batched_tokens` 参数优化批处理性能
5. **长文本优化**：对于长文本场景，使用 YaRN 等长度外推技术

## 常见模型显存占用参考

| 模型 | Base Model | Ollama | vLLM |
|------|-----------|--------|------|
| DeepSeek-R1-Distill-Qwen-1.5B | Qwen2.5-Math-1.5B | 1.1GB | 3-6 GB |
| DeepSeek-R1-Distill-Qwen-7B | Qwen2.5-Math-7B | 4.7GB | 14-21 GB |
| DeepSeek-R1-Distill-Llama-8B | Llama-3.1-8B | 4.9GB | 16-24 GB |
| DeepSeek-R1-Distill-Qwen-14B | Qwen2.5-14B | 9.0GB | 28-42 GB |
| DeepSeek-R1-Distill-Qwen-32B | Qwen2.5-32B | 20GB | 64-96 GB |
| DeepSeek-R1-Distill-Llama-70B | Llama-3.3-70B-Instruct | 43GB | 140-210 GB |
| DeepSeek-R1-671B | DeepSeek-R1-671B | 404GB | 1342-2013 GB |

## 技术栈关系

vLLM 的运行依赖于以下技术栈：

```
应用层：vLLM
    ↓
加速库层：cuDNN（深度学习加速库）
    ↓
计算平台层：CUDA（GPU 计算平台）
    ↓
驱动层：NVIDIA 驱动
    ↓
硬件层：NVIDIA GPU
```

- **vLLM**：应用层，调用 cuDNN 和 CUDA 提供的接口来加速计算
- **cuDNN**：加速库层，依赖于 CUDA 提供的 GPU 计算能力，优化了深度学习任务
- **CUDA**：计算平台层，依赖于 NVIDIA 驱动与 GPU 硬件通信，提供了通用的 GPU 计算接口
- **NVIDIA 驱动**：驱动层，管理着 NVIDIA GPU 的硬件资源，允许上层软件与 GPU 进行交互
- **NVIDIA GPU**：硬件层，执行实际的计算任务，提供了强大的并行计算能力

## 总结

vLLM 是一个强大的大语言模型推理引擎，通过 PagedAttention 等创新技术，显著提升了推理性能和显存利用率。它适合企业级服务和高并发场景，是构建生产级 LLM 应用的理想选择。

### 何时选择 vLLM？

- ✅ 需要高吞吐量和低延迟
- ✅ 有充足的 GPU 资源（专业级 GPU）
- ✅ 需要处理高并发请求
- ✅ 需要 OpenAI 兼容的 API 接口
- ✅ 企业级生产环境

### 何时选择 Ollama？

- ✅ 个人开发和测试
- ✅ 消费级 GPU（显存有限）
- ✅ 单用户或低并发场景
- ✅ 快速原型开发
- ✅ 本地部署和轻量级应用

