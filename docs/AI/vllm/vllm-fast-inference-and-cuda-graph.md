# vLLM 快速推理与 CUDA Graph 详解

本文档详细介绍 vLLM 的快速推理机制、CUDA Graph 模式以及相关配置。

---

## 1. 什么是快速推理 (Fast Inference)

### 1.1 定义

**快速推理 (Fast Inference)** 是指使用优化技术来加速大语言模型 (LLM) 的推理过程。在 Unsloth 和 vLLM 的上下文中，快速推理主要指：

- **vLLM 集成**：使用 vLLM 库来优化推理性能
- **内存优化**：通过 PagedAttention 等技术提高内存利用率
- **批处理优化**：高效处理多个请求
- **CUDA Graph 加速**：减少 CPU-GPU 通信开销

### 1.2 vLLM 的核心优化技术

#### PagedAttention
- 将注意力机制的 KV Cache 分页管理
- 类似操作系统的虚拟内存管理
- 减少内存碎片，提高利用率

#### 连续批处理 (Continuous Batching)
- 动态调整批次大小
- 不等待整个批次完成
- 提高 GPU 利用率

#### CUDA Graph
- 预先捕获 CUDA 操作序列
- 减少 CPU 开销
- 加速重复执行的计算图

### 1.3 在 Unsloth 中的应用

在 Unsloth 中，`fast_inference=True` 参数会：

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Base",
    max_seq_length=2048,
    load_in_4bit=False,
    fast_inference=True,  # 启用 vLLM 快速推理
    max_lora_rank=32,
)
```

**作用：**
- 启用 vLLM 推理引擎
- 自动优化推理性能
- 适用于推理和生成任务

**注意：**
- 主要用于**推理阶段**，不是训练阶段
- 训练时通常设置为 `False`
- GRPO 等训练任务不需要此优化

---

## 2. CUDA Graph 详解

### 2.1 什么是 CUDA Graph

**CUDA Graph** 是 NVIDIA CUDA 提供的一种优化技术，用于减少 CPU 开销和提高 GPU 利用率。

#### 传统 CUDA 执行流程
```
CPU: 准备参数 → 启动 Kernel 1 → 等待 → 启动 Kernel 2 → 等待 → ...
GPU:              执行 Kernel 1         执行 Kernel 2
```

**问题：**
- 每次 Kernel 启动都需要 CPU-GPU 通信
- CPU 开销大
- 延迟高

#### CUDA Graph 执行流程
```
CPU: 捕获图 (一次) → 重放图 → 重放图 → 重放图 → ...
GPU:                 执行整个图  执行整个图  执行整个图
```

**优势：**
- 一次捕获，多次重放
- 减少 CPU 开销（可降低 50% 以上）
- 降低延迟
- 提高吞吐量

### 2.2 CUDA Graph 的工作原理

#### 捕获阶段 (Capture)
```python
# PyTorch 示例
with torch.cuda.graph(cuda_graph):
    # 执行一次前向传播
    output = model(input)
```

- 记录所有 CUDA 操作
- 构建计算图
- 保存参数和依赖关系

#### 重放阶段 (Replay)
```python
# 重放图
cuda_graph.replay()
```

- 直接执行预先捕获的操作序列
- 无需 CPU 参与
- 极低延迟

### 2.3 CUDA Graph 的限制

**不支持的操作：**
- 动态形状（shape 必须固定）
- 动态控制流（if/while 等）
- CPU-GPU 同步操作
- 某些特殊的 Attention 实现

**适用场景：**
- 批次大小固定
- 模型结构固定
- 重复执行相同操作

---

## 3. vLLM 的 CUDA Graph 模式

### 3.1 三种模式详解

vLLM 通过 `VLLM_CUDAGRAPH_MODE` 环境变量控制 CUDA Graph 的使用方式。

#### 模式 1: NEVER (从不使用)

```bash
export VLLM_CUDAGRAPH_MODE=NEVER
```

**特点：**
- 完全禁用 CUDA Graph
- 所有操作在 Eager 模式下执行
- 最灵活，支持动态形状

**适用场景：**
- 调试和开发
- 使用不兼容的 Attention 后端
- 需要动态批次大小

**性能：**
- 最慢
- CPU 开销最大

#### 模式 2: PIECEWISE (分段捕获)

```bash
export VLLM_CUDAGRAPH_MODE=PIECEWISE
```

**特点：**
- 将计算图分段捕获
- 只对兼容的部分使用 CUDA Graph
- Attention 操作在 Eager 模式下执行
- 其他操作（FFN、LayerNorm 等）使用 CUDA Graph

**工作原理：**
```
[CUDA Graph] → [Eager Attention] → [CUDA Graph] → [Eager Attention] → ...
   FFN/Norm        Attention           FFN/Norm        Attention
```

**适用场景：**
- 使用 FlexAttention 等特殊 Attention 实现
- 需要灵活性和性能的平衡
- **vLLM V1 架构的默认模式**

**性能：**
- 中等
- 平衡灵活性和速度

#### 模式 3: FULL (完全捕获)

```bash
export VLLM_CUDAGRAPH_MODE=FULL
```

**特点：**
- 捕获整个计算图
- 包括 Attention 操作
- 最大化性能

**要求：**
- Attention 后端必须支持 CUDA Graph
- 批次大小固定
- 所有操作必须兼容

**适用场景：**
- 使用兼容的 Attention 后端（如 FlashAttention）
- 批次大小固定
- 追求极致性能

**性能：**
- 最快
- CPU 开销最小

### 3.2 模式对比表

| 特性 | NEVER | PIECEWISE | FULL |
|------|-------|-----------|------|
| **性能** | 慢 | 中等 | 快 |
| **灵活性** | 高 | 中等 | 低 |
| **Attention 支持** | 所有 | 所有 | 仅兼容的 |
| **动态批次** | 支持 | 部分支持 | 不支持 |
| **CPU 开销** | 高 | 中等 | 低 |
| **推荐场景** | 调试/开发 | 生产环境（默认） | 高性能推理 |

### 3.3 vLLM V1 架构的 Piecewise 实现

vLLM V1 使用 **Piecewise CUDA Graph** 作为默认模式：

```
计算图分割：
┌─────────────────────────────────────────────────┐
│ Layer 1                                         │
│  ├─ [CUDA Graph] Input Processing               │
│  ├─ [Eager] Attention Operation                 │
│  └─ [CUDA Graph] FFN + LayerNorm                │
├─────────────────────────────────────────────────┤
│ Layer 2                                         │
│  ├─ [CUDA Graph] (复用 Layer 1 的图)            │
│  ├─ [Eager] Attention Operation                 │
│  └─ [CUDA Graph] (复用 Layer 1 的图)            │
├─────────────────────────────────────────────────┤
│ ...                                             │
└─────────────────────────────────────────────────┘
```

**优势：**
- 保持 Attention 的灵活性
- 优化其他计算密集型操作
- 自动管理中间缓冲区

---

## 4. 常见错误及解决方案

### 4.1 FlexAttentionMetadataBuilder 错误

#### 错误信息
```
ValueError: CUDAGraphMode.FULL is not supported with FlexAttentionMetadataBuilder 
backend (support: AttentionCGSupport.NEVER); please try cudagraph_mode=PIECEWISE
```

#### 原因
- 使用了 FlexAttention 后端
- FlexAttention 不支持 FULL 模式的 CUDA Graph
- vLLM 默认尝试使用 FULL 模式

#### 解决方案

**方案 1：设置环境变量（推荐）**
```python
import os
os.environ["VLLM_CUDAGRAPH_MODE"] = "PIECEWISE"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Base",
    fast_inference=True,
    # ... 其他参数
)
```

**方案 2：禁用快速推理（训练时）**
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Base",
    fast_inference=False,  # 训练时不需要 vLLM
    # ... 其他参数
)
```

**方案 3：使用兼容的 Attention 后端**
```python
# 在 vLLM 配置中指定 Attention 后端
# 例如使用 FlashAttention
```

### 4.2 训练 vs 推理的配置

#### 训练阶段（GRPO、LoRA 等）
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Base",
    max_seq_length=2048,
    load_in_4bit=False,
    fast_inference=False,  # ❌ 训练时禁用
    max_lora_rank=32,
)
```

**原因：**
- 训练不需要 vLLM 的推理优化
- 避免 CUDA Graph 相关问题
- 简化训练流程

#### 推理阶段
```python
import os
os.environ["VLLM_CUDAGRAPH_MODE"] = "PIECEWISE"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Base",
    max_seq_length=2048,
    load_in_4bit=False,
    fast_inference=True,  # ✅ 推理时启用
    max_lora_rank=32,
)
```

---

## 5. 性能优化建议

### 5.1 选择合适的 CUDA Graph 模式

```python
# 开发/调试
os.environ["VLLM_CUDAGRAPH_MODE"] = "NEVER"

# 生产环境（推荐）
os.environ["VLLM_CUDAGRAPH_MODE"] = "PIECEWISE"

# 极致性能（需要兼容的后端）
os.environ["VLLM_CUDAGRAPH_MODE"] = "FULL"
```

### 5.2 编译特定形状

vLLM 支持为特定批次大小编译优化的 Kernel：

```bash
vllm serve meta-llama/Llama-3.2-1B \
  --compilation-config '{"compile_sizes": [1, 2, 4, 8]}'
```

**效果：**
- 为常用批次大小生成优化代码
- 自动调优 Kernel 参数
- 显著提升性能

### 5.3 缓存编译结果

vLLM 会缓存编译结果到：
```
~/.cache/vllm/torch_compile_cache/
```

**建议：**
- 在部署时复制缓存目录
- 避免重复编译
- 加快启动速度

---

## 6. 参考资源

### 官方文档
- [vLLM 官方文档](https://docs.vllm.ai/)
- [vLLM torch.compile 集成](https://docs.vllm.ai/en/latest/design/torch_compile.html)
- [PyTorch CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
- [Unsloth 文档](https://docs.unsloth.ai/)

### 相关 Issues
- [vLLM #24943 - CUDAGraphMode PIECEWISE](https://github.com/vllm-project/vllm/issues/24943)
- [vLLM #23261 - Piecewise Graph Splitting](https://github.com/vllm-project/vllm/issues/23261)
- [Unsloth #2846 - vLLM Fast Inference](https://github.com/unslothai/unsloth/issues/2846)

### 博客文章
- [vLLM V1 Alpha Release](https://developers.redhat.com/articles/2025/01/28/vllm-v1-a-major-upgrade-vllms-core-architecture)
- [Unsloth GRPO Guide](https://unsloth.ai/blog/r1-reasoning)

---

## 7. 快速参考

### 环境变量速查

```bash
# CUDA Graph 模式
export VLLM_CUDAGRAPH_MODE=NEVER      # 禁用
export VLLM_CUDAGRAPH_MODE=PIECEWISE  # 分段（默认）
export VLLM_CUDAGRAPH_MODE=FULL       # 完全

# 禁用编译缓存（调试用）
export VLLM_DISABLE_COMPILE_CACHE=1

# 日志级别
export VLLM_LOGGING_LEVEL=DEBUG
```

### 代码模板

```python
import os

# 训练配置
def setup_for_training():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B-Base",
        max_seq_length=2048,
        load_in_4bit=False,
        fast_inference=False,  # 训练时禁用
        max_lora_rank=32,
    )
    return model, tokenizer

# 推理配置
def setup_for_inference():
    os.environ["VLLM_CUDAGRAPH_MODE"] = "PIECEWISE"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B-Base",
        max_seq_length=2048,
        load_in_4bit=False,
        fast_inference=True,  # 推理时启用
        max_lora_rank=32,
    )
    return model, tokenizer
```

---

## 更新日志

- 2025-10-10: 初始版本，整理 vLLM 快速推理和 CUDA Graph 相关知识

