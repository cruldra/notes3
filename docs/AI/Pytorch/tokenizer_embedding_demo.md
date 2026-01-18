---
title: Tokenizer Embedding Demo
marimo-version: 0.19.2
width: medium
---

```python {.marimo}
import marimo as mo
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from pathlib import Path
```

# 🎯 分词器与词嵌入完整流程演示

本笔记本演示从**原始文本**到**词嵌入向量**的完整转换过程。

## 📊 流程概览

```python {.marimo hide_code="true"}
mo.mermaid(r"""
    graph TD
        A[原始文本<br/>'小红帽去森林'] -->|分词| B[Tokenizer]
        B -->|切分词语| C[Tokens<br/>小红帽,去,森林]
        C -->|查词典| D[Token IDs<br/>1,453,234,789,2]
        D -->|索引查找| E[Embedding Table<br/>6400×512矩阵]
        E -->|获取向量| F[Embedding Vectors<br/>5×512矩阵]
""")
```

## 🔧 步骤1: 加载预训练分词器

我们使用 HuggingFace 的 `AutoTokenizer` 加载预训练的分词器。
这里使用 Qwen2.5-7B-Instruct 的分词器作为示例。

```python {.marimo}
# 加载预训练分词器
# 可以替换为其他模型: "Qwen/Qwen2.5-1.5B-Instruct", "meta-llama/Llama-3.1-8B-Instruct" 等
tokenizer_path = "Qwen/Qwen2.5-7B-Instruct"

print(f"📥 正在加载分词器: {tokenizer_path}")
print("⏳ 首次运行会从 HuggingFace 下载,请稍候...")

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path,
    trust_remote_code=True,  # Qwen 模型需要此参数
)

print(f"✅ 分词器加载成功!")
print(f"📊 词汇表大小: {tokenizer.vocab_size:,}")
print(f"🔤 特殊 Token:")
print(f"   BOS (开始): '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
print(f"   EOS (结束): '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
print(f"   PAD (填充): '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
```

## 🔤 步骤2: 文本分词 (Tokenization)
<!---->
### 💡 分词器已就绪

`AutoTokenizer.from_pretrained()` 自动加载了:
- **词汇表 (vocab.json)**: 所有支持的 Token 及其 ID
- **合并规则 (merges.txt/tokenizer.json)**: BPE/WordPiece 分词规则
- **特殊 Token 配置**: BOS/EOS/PAD 等标记
- **分词逻辑**: 完整的编码/解码功能
<!---->
### 📝 实际演示: 对 "小红帽去森林" 进行分词

```python {.marimo}
# 原始文本
text = "小红帽去森林"

# 步骤1: 分词
tokens = tokenizer.tokenize(text)
print(f"🔤 原始文本: {text}")
print(f"📋 Tokens: {tokens}")

# 步骤2: 转换为 IDs
token_ids = tokenizer.encode(text, add_special_tokens=False)
print(f"🔢 Token IDs: {token_ids}")

# 显示对应关系
print("\n📊 Token <-> ID 映射:")
for token, tid in zip(tokens, token_ids):
    print(f"  '{token}' -> {tid}")

# 验证解码
decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
print(f"\n🔄 解码结果: {decoded_text}")
print(f"✅ 解码正确: {decoded_text == text}")
```

## 🎨 步骤3: 词嵌入 (Embedding)
<!---->
### 核心概念

**Embedding Layer** = 一个查找表 (Lookup Table)

- **输入**: Token ID (整数, 例如 453)
- **输出**: 密集向量 (例如 512 维的浮点数向量)
- **本质**: 从 `vocab_size × hidden_size` 的大矩阵中,根据 ID 提取对应的行

```python {.marimo}
# 创建 Embedding 层 - 使用真实的词汇表大小
vocab_size = tokenizer.vocab_size  # 使用分词器的实际词汇表大小
hidden_size = 512  # 嵌入维度 (可以调整,实际模型可能是 2048, 4096 等)

embedding_layer = nn.Embedding(vocab_size, hidden_size)

# 初始化为小随机数 (实际训练中会学习到语义信息)
nn.init.normal_(embedding_layer.weight, mean=0.0, std=0.02)

print(f"📐 Embedding 层形状: {embedding_layer.weight.shape}")
print(f"   = {vocab_size:,} 个词 × {hidden_size} 维向量")
print(f"   = 总共 {vocab_size * hidden_size:,} 个参数")
print(f"   ≈ {vocab_size * hidden_size * 4 / 1024 / 1024:.2f} MB (float32)")
```

### 🔍 演示: 将 Token IDs 转换为嵌入向量

```python {.marimo}
# 转换为 Tensor
input_ids = torch.tensor([token_ids])  # shape: [1, seq_len]

# 通过 Embedding 层
embeddings = embedding_layer(input_ids)

print(f"📥 输入 Token IDs: {input_ids.shape} = [batch_size, seq_len]")
print(f"   实际值: {input_ids.tolist()}")
print(
    f"\n📤 输出 Embeddings: {embeddings.shape} = [batch_size, seq_len, hidden_size]"
)
print(f"\n🔍 第一个 Token ('{tokens[0]}', ID={token_ids[0]}) 的嵌入向量 (前10维):")
print(f"   {embeddings[0, 0, :10].detach().numpy()}")
```

## 📊 可视化: Embedding 查找过程

以 Token "小红帽" (ID=453) 为例:

```python {.marimo}
# 单独查询一个 Token 的嵌入
test_text = "小红帽"
token_id_test = tokenizer.encode(test_text, add_special_tokens=False)[0]
embedding_test = embedding_layer(torch.tensor([token_id_test]))

print(f"🎯 Token: '{test_text}'")
print(f"🔢 ID: {token_id_test}")
print(f"📐 嵌入向量形状: {embedding_test.shape}")
print(f"\n🔍 向量内容 (前20维):")
print(embedding_test[0, :20].detach().numpy())
```

## 🧮 步骤4: 完整流程串联

模拟完整的 **文本 -> Transformer** 输入准备过程

```python {.marimo}
def text_to_embeddings(text, tokenizer, embedding_layer):
    """完整流程: 文本 -> Embeddings"""
    # 1. 分词
    tokens_1 = tokenizer.tokenize(text)

    # 2. 转 IDs (不添加特殊 token,仅为演示)
    ids = tokenizer.encode(text, add_special_tokens=False)

    # 3. 转 Tensor
    input_tensor = torch.tensor([ids])

    # 4. 获取 Embeddings
    embeddings_1 = embedding_layer(input_tensor)

    return {
        "text": text,
        "tokens": tokens_1,
        "ids": ids,
        "input_shape": input_tensor.shape,
        "embedding_shape": embeddings_1.shape,
        "embeddings": embeddings_1,
    }

# 演示
result = text_to_embeddings(text, tokenizer, embedding_layer)

print("=" * 60)
print("🎯 完整流程演示")
print("=" * 60)
print(f"📝 原始文本: {result['text']}")
print(f"🔤 Tokens: {result['tokens']}")
print(f"🔢 Token IDs: {result['ids']}")
print(f"📥 输入形状: {result['input_shape']}")
print(f"📤 嵌入形状: {result['embedding_shape']}")
print(f"\n✅ 此时嵌入向量已准备好送入 Transformer 进行处理!")
```

## 🎓 关键要点总结

### 1️⃣ 为什么需要 Embedding?

| 表示方式 | 优点 | 缺点 |
|---------|------|------|
| **One-Hot** (独热编码) | 简单直观 | 维度爆炸 (6400维), 无语义信息 |
| **Embedding** (密集向量) | 低维 (512维), 包含语义 | 需要训练学习 |

### 2️⃣ Embedding 如何学习语义?

- **初始化**: 随机小数值
- **训练过程**: 通过反向传播,根据上下文自动调整
- **结果**: 语义相近的词向量距离更近

例如训练后:
```
similarity("国王", "王后") > similarity("国王", "苹果")
```

### 3️⃣ 实际代码对应

```python
# 来自 从头开始训练自己的大模型.py

# 定义 Embedding 层 (py:1160)
self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

# 前向传播 (py:1393)
h = self.embedding(idx)  # [batch, seq_len] -> [batch, seq_len, hidden_size]
```

### 4️⃣ 下一步

嵌入向量会送入 **Transformer Blocks** 进行多层处理:
- 位置编码 (RoPE)
- 自注意力机制 (Attention)
- 前馈网络 (FeedForward)
- 层归一化 (RMSNorm)

最终输出预测下一个词的概率分布。
<!---->
## 🔬 扩展实验: 多个句子的批处理

实际训练时,我们会同时处理多个句子 (batch)

```python {.marimo}
# 多个句子
sentences = [
    "小红帽去森林",
    "小红帽爱学习",
]

# 使用 tokenizer 自带的批处理功能
# padding=True 会自动补齐到最长序列
# return_tensors="pt" 返回 PyTorch tensor
batch_encoding = tokenizer(
    sentences, padding=True, return_tensors="pt", add_special_tokens=False
)

batch_tensor = batch_encoding["input_ids"]  # shape: [batch_size, max_len]
attention_mask = batch_encoding[
    "attention_mask"
]  # 0 表示 padding, 1 表示真实 token

# 获取 Embeddings
batch_embeddings = embedding_layer(batch_tensor)

print(f"📚 批处理 {len(sentences)} 个句子")
print(f"📥 输入形状: {batch_tensor.shape} = [batch_size, max_seq_len]")
print(
    f"📤 输出形状: {batch_embeddings.shape} = [batch_size, max_seq_len, hidden_size]"
)
print(f"\n🔍 每个句子的 IDs (已自动 Padding):")
for i, sent in enumerate(sentences):
    print(f"  [{i}] '{sent}'")
    print(f"      IDs: {batch_tensor[i].tolist()}")
    print(f"      Mask: {attention_mask[i].tolist()}")
```

---

## 📚 参考资料

- 完整代码: `从头开始训练自己的大模型.py`
- 详细解释: `从头开始训练自己的大模型.md`
- Transformer 论文: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)