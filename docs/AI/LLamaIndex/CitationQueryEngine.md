---
title: Citationqueryengine
marimo-version: 0.19.2
header: |-
  # /// script
  # dependencies = [
  #     "llama-index-embeddings-openai",
  #     "llama-index-embeddings-openai-like",
  #     "llama-index-llms-openai",
  #     "llama-index-llms-openai-like",
  #     "llama-index",
  #     "marimo>=0.19.2",
  #     "pydantic-ai==1.41.0",
  # ]
  # ///
---

# LlamaIndex CitationQueryEngine 演示

本笔记本演示如何使用 LlamaIndex 的 CitationQueryEngine 来生成带有引用的回答。

CitationQueryEngine 可与任何现有索引一起使用，并提供引用溯源功能。
<!---->
## 安装依赖

如果您在 colab 上打开此笔记本，您可能需要安装 LlamaIndex 🦙。
<!---->
## 导入必要的库

```python {.marimo}
import os
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
```

```python {.marimo}
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.core import Settings

# 配置全局设置
Settings.llm = OpenAILike(
    model="openai/gpt-4o",
    api_base="https://openrouter.ai/api/v1",
    api_key="",
    is_chat_model=True
)
Settings.embed_model = OpenAILikeEmbedding(
    model_name="openai/text-embedding-3-small",
    api_key="",
    api_base="https://openrouter.ai/api/v1"
)
```

## 下载数据

我们将使用 Paul Graham 的文章作为示例数据。

```python {.marimo}
if not os.path.exists("data/paul_graham/"):
    os.makedirs("data/paul_graham/")

if not os.path.exists("data/paul_graham/paul_graham_essay.txt"):
    import urllib.request

    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt",
        "data/paul_graham/paul_graham_essay.txt",
    )
```

## 创建或加载索引

```python {.marimo}
if not os.path.exists("./citation"):
    # 加载文档
    documents = SimpleDirectoryReader("./data/paul_graham").load_data()

    # 创建向量索引
    index = VectorStoreIndex.from_documents(
        documents,
    )

    # 持久化索引
    index.storage_context.persist(persist_dir="./citation")
    print("索引已创建并保存到 ./citation")
else:
    # 加载已存在的索引
    index = load_index_from_storage(
        StorageContext.from_defaults(persist_dir="./citation"),
    )
    print("从 ./citation 加载已有索引")
```

## 使用默认参数创建 CitationQueryEngine

```python {.marimo}
query_engine = CitationQueryEngine.from_args(
    index,
    similarity_top_k=3,
    # 这里我们可以控制引用来源的粒度，默认是512
    citation_chunk_size=512,
)
```

## 进行查询

```python {.marimo}
response = query_engine.query("What did the author do growing up?")
```

```python {.marimo}
print(response)
```

```python {.marimo}
# 原始1024大小的节点被分割成更细粒度的节点
print(f"来源节点数量: {len(response.source_nodes)}")
```

### 检查实际来源

来源从 1 开始计数，但 Python 数组从 0 开始计数！

让我们确认一下来源是否合理。

```python {.marimo}
print("=== 来源 1 ===")
print(response.source_nodes[0].node.get_text()[:500] + "...")
```

```python {.marimo}
print("=== 来源 2 ===")
print(response.source_nodes[1].node.get_text()[:500] + "...")
```

## 调整设置

请注意，将 chunk size（块大小）设置为大于节点的原始 chunk size 将不会产生任何效果。

默认的节点块大小是 1024，因此在这里，我们并没有让引用节点变得更细粒度。

```python {.marimo}
query_engine_large_chunk = CitationQueryEngine.from_args(
    index,
    # 增加引用块大小！
    citation_chunk_size=1024,
    similarity_top_k=3,
)
```

```python {.marimo}
response_large = query_engine_large_chunk.query("What did the author do growing up?")
```

```python {.marimo}
print(response_large)
```

```python {.marimo}
# 现在应该有更少的来源节点！
print(f"来源节点数量: {len(response_large.source_nodes)}")
```

### 检查实际来源

```python {.marimo}
print("=== 来源 1 (大块) ===")
print(response_large.source_nodes[0].node.get_text()[:800] + "...")
```

## 更多查询示例

```python {.marimo}
# 使用默认的查询引擎
query = "What influenced the author to study AI?"
response2 = query_engine.query(query)

print(f"问题: {query}")
print(f"回答: {response2}")
print(f"来源数量: {len(response2.source_nodes)}")

# 显示第一个来源的部分内容
if response2.source_nodes:
    print("\n部分来源内容:")
    print(response2.source_nodes[0].node.get_text()[:300] + "...")
```

## 总结

CitationQueryEngine 提供了以下功能：

1. **引用溯源**：回答中包含了引用标记（如 [1], [2]）
2. **粒度控制**：通过 `citation_chunk_size` 参数控制引用块的粒度
3. **来源访问**：可以通过 `response.source_nodes` 访问原始的来源节点
4. **灵活性**：可与任何 LlamaIndex 索引一起使用

通过调整 `citation_chunk_size`，您可以控制引用粒度：
- 较小的值：更细粒度的引用，但可能更分散
- 较大的值：更大块的引用，提供更完整的上下文

```python {.marimo}
import marimo as mo
```