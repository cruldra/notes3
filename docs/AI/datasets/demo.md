---
title: Demo
marimo-version: 0.19.2
width: medium
---

```python {.marimo}
import marimo as mo
```

# Hugging Face Datasets API 演示

本笔记本演示 Hugging Face `datasets` 库的核心 API 用法。

```python {.marimo}
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import numpy as np
```

## 1. 从 Hub 加载数据集

使用 `load_dataset` 函数从 Hugging Face Hub 加载数据集。

```python {.marimo}
# 加载 GLUE 基准中的 MRPC 数据集
dataset = load_dataset("glue", "mrpc")

# 查看数据集结构
dataset
```

```python {.marimo}
# DatasetDict 包含多个 split
splits = dataset.keys()
splits
```

```python {.marimo}
# 访问 train split
train_dataset = dataset["train"]
train_dataset
```

```python {.marimo}
# 查看数据集特征
train_dataset.features
```

```python {.marimo}
# 查看数据集信息
train_dataset.info
```

```python {.marimo}
# 数据集长度
len(train_dataset)
```

## 2. 数据访问与切片

```python {.marimo}
# 随机访问
example = train_dataset[0]
example
```

```python {.marimo}
# 切片访问
examples = train_dataset[10:15]
examples
```

```python {.marimo}
# 访问单列
sentence1 = train_dataset["sentence1"]
sentence1[:5]  # 显示前5个
```

```python {.marimo}
# 访问多列
sentence1_col = train_dataset["sentence1"]
sentence2_col = train_dataset["sentence2"]
# 显示前5个
{"sentence1": sentence1_col[:5], "sentence2": sentence2_col[:5]}
```

## 3. 数据变换 - map 方法

`map` 方法可以对数据集进行批量变换，支持多进程处理和缓存机制。

```python {.marimo name="*add_length"}
# 定义一个简单的预处理函数
def add_length(examples):
    return {
        "sentence1_length": len(examples["sentence1"]),
        "sentence2_length": len(examples["sentence2"]),
    }
```

```python {.marimo}
# 应用 map 方法
mapped_dataset = train_dataset.map(add_length)

# 查看结果
mapped_dataset[0]
```

```python {.marimo name="*add_length_batched"}
# 使用 batched=True 提高效率
def add_length_batched(examples):
    return {
        "sentence1_length_batched": [len(s) for s in examples["sentence1"]],
        "sentence2_length_batched": [len(s) for s in examples["sentence2"]],
    }
```

```python {.marimo}
# 批量应用
batched_mapped = train_dataset.map(add_length_batched, batched=True)
batched_mapped[0]
```

## 4. 数据过滤 - filter 方法

使用 `filter` 方法根据条件过滤数据。

```python {.marimo name="*filter_short_sentences"}
# 定义过滤函数
def filter_short_sentences(examples):
    return len(examples["sentence1"]) < 50 and len(examples["sentence2"]) < 50
```

```python {.marimo}
# 过滤数据
short_sentences = train_dataset.filter(filter_short_sentences)
print(f"原始数据集大小: {len(train_dataset)}")
print(f"过滤后数据集大小: {len(short_sentences)}")
```

```python {.marimo}
# 查看过滤后的数据
short_sentences[:5]
```

## 5. 排序 - sort 方法

使用 `sort` 方法对数据集进行排序。

```python {.marimo}
# 按 label 排序
sorted_dataset = train_dataset.sort("label")
sorted_dataset[:5]
```

```python {.marimo}
# 降序排序
sorted_desc = train_dataset.sort("label", reverse=True)
sorted_desc[:5]
```

## 6. 打乱数据 - shuffle 方法

使用 `shuffle` 方法随机打乱数据集。

```python {.marimo}
# 打乱数据
shuffled_dataset = train_dataset.shuffle(seed=42)
shuffled_dataset[:5]
```

```python {.marimo}
# 打乱并只取一部分
small_dataset = train_dataset.shuffle(seed=42).select(range(10))
small_dataset
```

## 7. 数据格式转换 - set_format

使用 `set_format` 方法零拷贝地转换数据格式。

```python {.marimo}
# 转换为 numpy 格式
train_dataset.set_format("numpy")
```

```python {.marimo}
# 现在访问数据会返回 numpy 数组
label_array = train_dataset["label"]
label_array[:5], type(label_array), label_array.dtype
```

```python {.marimo}
# 重置为默认格式
train_dataset.reset_format()
```

## 8. 保存与加载本地数据集

```python {.marimo}
# 从 Pandas DataFrame 创建 Dataset
df = pd.DataFrame(
    {
        "text": ["hello world", "hello marimo", "marimo is awesome"],
        "label": [0, 1, 1],
    }
)
df
```

```python {.marimo}
# 创建 Dataset
custom_dataset = Dataset.from_pandas(df)
custom_dataset
```

```python {.marimo}
# 从字典创建 Dataset
dict_data = {
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "city": ["Beijing", "Shanghai", "Guangzhou"],
}
from_dict_dataset = Dataset.from_dict(dict_data)
from_dict_dataset
```

```python {.marimo}
# 保存到磁盘
from_dict_dataset.save_to_disk("my_custom_dataset")
```

```python {.marimo}
# 从磁盘加载
loaded_dataset = Dataset.load_from_disk("my_custom_dataset")
loaded_dataset
```

```python {.marimo}
# 保存为 JSON 格式
from_dict_dataset.to_json("my_dataset.json")
```

```python {.marimo}
# 从 JSON 加载
from_json_dataset = Dataset.from_json("my_dataset.json")
from_json_dataset
```

## 9. DatasetDict 操作

DatasetDict 用于管理数据集的不同划分。

```python {.marimo}
# 创建 DatasetDict
my_dataset_dict = DatasetDict(
    {
        "train": Dataset.from_dict({"x": [1, 2, 3], "y": [4, 5, 6]}),
        "validation": Dataset.from_dict({"x": [7, 8], "y": [9, 10]}),
        "test": Dataset.from_dict({"x": [11, 12], "y": [13, 14]}),
    }
)
my_dataset_dict
```

```python {.marimo}
# 访问各个 split
my_dataset_dict["train"]
```

```python {.marimo}
# 对所有 split 应用相同的变换
def multiply_by_2(batch):
    return {"x": [i * 2 for i in batch["x"]]}

transformed_dict = my_dataset_dict.map(multiply_by_2, batched=True)
transformed_dict
```

## 10. 列操作

```python {.marimo}
# 添加新列
def add_features(examples):
    return {
        "sentence1_upper": [s.upper() for s in examples["sentence1"]],
        "sentence2_upper": [s.upper() for s in examples["sentence2"]],
    }

dataset_with_new_columns = train_dataset.map(add_features, batched=True)
dataset_with_new_columns[0]
```

```python {.marimo}
# 重命名列
renamed_dataset = train_dataset.rename_column("sentence1", "text1")
renamed_dataset.features
```

```python {.marimo}
# 删除列
dataset_removed = train_dataset.remove_columns(["idx"])
dataset_removed.features
```

## 11. 流式数据集 - IterableDataset

对于超大数据集，使用 `streaming=True` 可以避免下载完整数据。

```python {.marimo}
# 加载流式数据集
streaming_dataset = load_dataset("imdb", streaming=True)
streaming_dataset
```

```python {.marimo}
# IterableDataset 只能迭代，不能随机访问
train_iterable = streaming_dataset["train"]
type(train_iterable)
```

```python {.marimo}
# 迭代访问数据
iterator = iter(train_iterable)
for i in range(5):
    example = next(iterator)
    print(f"Example {i}: {example['text'][:50]}...")
```

## 12. 数据集分片 - sharding

使用 `shard` 方法将数据集分成多个分片，便于分布式处理。

```python {.marimo}
# 将数据集分成 4 个分片，取第 0 个
shard0 = train_dataset.shard(num_shards=4, index=0)
len(shard0)
```

```python {.marimo}
# 获取第 1 个分片
shard1 = train_dataset.shard(num_shards=4, index=1)
len(shard1)
```

```python {.marimo}
# 验证所有分片的总和
total_shards = sum(
    len(train_dataset.shard(num_shards=4, index=i)) for i in range(4)
)
print(f"原始数据集大小: {len(train_dataset)}")
print(f"所有分片总和: {total_shards}")
```

## 13. 数据集连接与拆分

```python {.marimo}
# 创建两个小数据集
ds1 = Dataset.from_dict({"a": [1, 2], "b": [3, 4]})
ds2 = Dataset.from_dict({"a": [5, 6], "b": [7, 8]})

# 拼接数据集
concatenated = Dataset.from_dict(
    {"a": ds1["a"] + ds2["a"], "b": ds1["b"] + ds2["b"]}
)
concatenated
```

```python {.marimo}
# 拆分数据集
original = Dataset.from_dict({"x": list(range(10)), "y": list(range(10, 20))})

# 按 80/20 拆分
split_ds = original.train_test_split(test_size=0.2)
split_ds
```

```python {.marimo}
# 访问训练集和测试集
print(f"训练集大小: {len(split_ds['train'])}")
print(f"测试集大小: {len(split_ds['test'])}")
```

## 14. 使用本地文件加载数据

可以从各种本地文件格式加载数据。

```python {.marimo}
# 创建一个临时的 CSV 文件
import csv

with open("temp_data.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    writer.writerow(["hello world", "0"])
    writer.writerow(["hello marimo", "1"])
    writer.writerow(["marimo is awesome", "1"])
```

```python {.marimo}
# 从 CSV 文件加载
csv_dataset = load_dataset("csv", data_files="temp_data.csv")
csv_dataset
```

```python {.marimo}
# 查看数据
csv_dataset["train"][:]
```

## 15. 与 Transformers 集成

演示如何使用 Transformers 的 tokenizer 处理数据集。

```python {.marimo name="*simple_tokenizer"}
# 模拟一个简单的 tokenizer 函数
def simple_tokenizer(text, max_length=10):
    # 简单的按空格分词
    tokens = text.split()[:max_length]
    # 简单的词汇表映射
    vocab = {"hello": 0, "world": 1, "marimo": 2, "is": 3, "awesome": 4}
    input_ids = [vocab.get(token, 999) for token in tokens]  # 999 表示未知词
    return input_ids
```

```python {.marimo}
# 创建示例数据集
text_data = {
    "text": [
        "hello world",
        "hello marimo",
        "marimo is awesome",
        "hello hello world",
    ]
}
text_dataset = Dataset.from_dict(text_data)
text_dataset
```

```python {.marimo name="*tokenize_function"}
# 定义 tokenize 函数
def tokenize_function(examples):
    return {"input_ids": [simple_tokenizer(text) for text in examples["text"]]}
```

```python {.marimo}
# 应用 tokenize
tokenized_dataset = text_dataset.map(tokenize_function, batched=True)
tokenized_dataset
```

```python {.marimo}
# 查看结果
tokenized_dataset[:]
```

## 16. 性能提示

1. **使用 `batched=True`**: 在 map 方法中启用批处理可以显著提高速度
2. **设置 `num_proc`**: 利用多核 CPU 加速处理
3. **保持 Arrow 格式**: 不要将大型 Dataset 转换为列表，直接使用 Arrow 格式
4. **使用缓存**: map 方法会自动缓存结果，避免重复计算
5. **流式处理超大数据**: 对于 TB 级数据，使用 `streaming=True`

```python {.marimo}
# 示例: 使用多进程处理
def slow_function(examples):
    import time

    time.sleep(0.01)  # 模拟耗时操作
    return {"processed": [1] * len(examples["sentence1"])}

# 注意: 实际运行时取消 num_proc 参数以使用多进程
# 这里设置 num_proc=1 以避免演示时出现问题
result = train_dataset.map(slow_function, batched=True, num_proc=1)
result
```