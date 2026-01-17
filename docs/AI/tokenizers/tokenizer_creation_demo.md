---
title: Tokenizer Creation Demo
marimo-version: 0.19.2
---

```python {.marimo}
import marimo as mo
```

# Tokenizer Creation Demo

这两个文件 (`tokenizer.json` 和 `tokenizer_config.json`) 通常是由 Hugging Face 的 `tokenizers` 库和 `transformers` 库协同生成的。

- **`tokenizer.json`**: 包含了分词器的核心数据结构（词表 Vocabulary、合并规则 Merges）、预处理逻辑（Normalizer, Pre-tokenizer）和解码逻辑（Decoder）。这是由 Rust 编写的 `tokenizers` 库直接管理的底层序列化文件。
- **`tokenizer_config.json`**: 包含了 Hugging Face `transformers` 库在加载分词器时需要的高级配置，例如特殊 token 的映射（BOS, EOS, PAD）、聊天模板 (`chat_template`)、最大长度限制等。

下面演示如何从头训练一个分词器并生成这两个文件。

```python {.marimo}
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast
import json
import os
```

```python {.marimo}
# 1. 初始化一个 BPE Tokenizer
# 你的 tokenizer.json 显示使用了 ByteLevel 的 pre-tokenizer 和 decoder，以及 BPE 模型。
tokenizer = Tokenizer(models.BPE())

# 配置 Pre-tokenizer (负责将文本切分为单词/子词的初步单元)
# ByteLevel 能够处理 utf-8 字节，适合多语言和代码
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# 配置 Decoder (负责将 token ID 解码回文本)
tokenizer.decoder = decoders.ByteLevel()
```

```python {.marimo}
# 2. 训练 Tokenizer
# 我们需要一些文本数据来训练。这里使用一个简单的语料库列表。
corpus = [
    "Hello world!",
    "This is a test sentence.",
    "Tokenizers are awesome.",
    "How to create tokenizer.json and tokenizer_config.json?",
    "I love AI and Pytorch.",
    "Deep learning involves neural networks.",
    "The quick brown fox jumps over the lazy dog.",
]

# 定义特殊 Token
# 注意：这些 token 需要与 tokenizer_config.json 中的 map 对应
special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

# 初始化训练器
trainer = trainers.BpeTrainer(
    vocab_size=1000,  # 演示用，设小一点
    special_tokens=special_tokens,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)

# 在语料上训练
tokenizer.train_from_iterator(corpus, trainer=trainer)

print("Tokenizer 训练完成！")
```

```python {.marimo}
# 3. 保存原始的 tokenizer.json
# 这步生成的只是底层的 tokenizer 数据
output_dir = "my_tokenizer_demo"
os.makedirs(output_dir, exist_ok=True)

# 此时主要生成 tokenizer.json
# 注意：这时候还没有 tokenizer_config.json
tokenizer.save(os.path.join(output_dir, "tokenizer.json"))

print(f"底层 tokenizer.json 已保存到 {output_dir}")
```

```python {.marimo}
# 4. 使用 Transformers 包装并添加高级配置
# 这一步是生成 tokenizer_config.json 的关键

# 从刚才保存的 tokenizer.json 加载
fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=os.path.join(output_dir, "tokenizer.json"),
    # 下面这些参数会进入 tokenizer_config.json
    bos_token=special_tokens[1],  # <|im_start|>
    eos_token=special_tokens[2],  # <|im_end|>
    unk_token=special_tokens[0],  # <|endoftext|>
    pad_token=special_tokens[0],  # <|endoftext|>
    clean_up_tokenization_spaces=False,
)

# 添加聊天模板 (Chat Template)
# 这是一个 Jinja2 模板，用于将对话列表转换为 prompt 字符串
fast_tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{{ '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}{% elif message['role'] == 'assistant' %}{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}{% endif %}{% endfor %}"

print("Transformer Tokenizer 包装完成，已配置特殊 token 和聊天模板。")
```

```python {.marimo}
# 5. 保存完整的分词器
# save_pretrained 会同时生成/更新 tokenizer.json 和生成 tokenizer_config.json
fast_tokenizer.save_pretrained(output_dir)

print(f"最终文件已保存到 {output_dir} 目录：")
```

```python {.marimo}
# 查看生成的文件
files = os.listdir(output_dir)
print("生成的文件列表:", files)

# 打印 tokenizer_config.json 的内容
config_path = os.path.join(output_dir, "tokenizer_config.json")
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        print("\n生成的 tokenizer_config.json 内容摘要:")
        print(json.dumps(config, indent=2, ensure_ascii=False))
```