如果你还没做过，请在阅读本文之前安装 LlamaIndex 并完成入门教程。这将有助于你结合实际经验来理解这些步骤。

LLM（大语言模型）是在海量数据上训练的，但它们并没有在 **你的** 数据上进行训练。检索增强生成（Retrieval-Augmented Generation，简称 RAG）通过将你的数据添加到 LLM 已有访问权限的数据中来解决这个问题。你会在本文档中经常看到 RAG 的相关引用。查询引擎、聊天引擎和 Agent（智能体）通常使用 RAG 来完成它们的任务。

在 RAG 中，你的数据被加载并为查询做好准备，即“索引化”（indexed）。用户的查询作用于索引，索引将你的数据过滤为最相关的上下文。然后，这个上下文和你的查询连同提示词（prompt）一起发送给 LLM，LLM 随后提供回答。

即使你正在构建的是聊天机器人或 Agent，你也会想了解将数据导入应用的 RAG 技术。

![](https://developers.llamaindex.ai/python/_astro/basic_rag.sdlwNwWz_Z1yQWLG.webp)

### RAG 的各个阶段

RAG 包含五个关键阶段，它们也将是你构建的大多数大型应用程序的一部分。这些阶段包括：

* **Loading（加载）**：这指的是将你的数据从其存储位置——无论是文本文件、PDF、其他网站、数据库还是 API——导入到你的工作流中。[LlamaHub](https://llamahub.ai/) 提供了数百种连接器供你选择。
* **Indexing（索引）**：这意味着创建一个允许查询数据的数据结构。对于 LLM 来说，这几乎总是意味着创建 `vector embeddings`（向量嵌入），即数据含义的数值表示，以及许多其他元数据策略，以便轻松准确地找到上下文相关的数据。
* **Storing（存储）**：一旦你的数据被索引，你几乎总是希望存储你的索引以及其他元数据，以避免必须重新索引它。
* **Querying（查询）**：对于任何给定的索引策略，都有许多方式利用 LLM 和 LlamaIndex 数据结构进行查询，包括子查询、多步查询和混合策略。
* **Evaluation（评估）**：任何流程中的一个关键步骤是检查它相对于其他策略的有效性，或者当你通过更改时进行检查。评估提供了关于你的查询响应的准确性、忠实度和速度的客观衡量标准。

![](https://developers.llamaindex.ai/python/_astro/stages.B-QMnT9I_1uEetk.webp)

### RAG 中的重要概念

你还会遇到一些术语，它们指的是这些阶段中的具体步骤。

#### Loading（加载）阶段

**Nodes and Documents（节点和文档）**：`Document` 是任何数据源的容器——例如 PDF、API 输出或从数据库检索的数据。`Node` 是 LlamaIndex 中数据的原子单位，代表源 `Document` 的一个“块”（chunk）。节点具有将其与所在文档以及其他节点相关联的元数据。

**Connectors（连接器）**：
数据连接器（通常称为 `Reader`）将来自不同数据源和数据格式的数据摄取为 `Documents` 和 `Nodes`。

#### Indexing（索引）阶段

**Indexes（索引）**：
一旦你摄取了数据，LlamaIndex 将帮助你将数据索引为易于检索的结构。这通常涉及生成 `vector embeddings`，它们存储在称为 `vector store` 的专用数据库中。索引还可以存储关于你的数据的各种元数据。

**Embeddings（嵌入）**：LLM 生成称为 `embeddings` 的数据的数值表示。在过滤数据的相关性时，LlamaIndex 将把查询转换为嵌入，你的向量存储将查找与查询的嵌入在数值上相似的数据。

#### Querying（查询）阶段

**Retrievers（检索器）**：
检索器定义了在给定查询时如何从索引中高效地检索相关上下文。你的检索策略是检索数据的相关性以及检索效率的关键。

**Routers（路由器）**：
路由器确定将使用哪个检索器从知识库中检索相关上下文。更具体地说，`RouterRetriever` 类负责选择一个或多个候选检索器来执行查询。它们使用选择器根据每个候选者的元数据和查询来选择最佳选项。

**Node Postprocessors（节点后处理器）**：
节点后处理器接收一组检索到的节点，并对它们应用转换、过滤或重排序逻辑。

**Response Synthesizers（响应合成器）**：
响应合成器利用用户查询和给定的一组检索到的文本块，通过 LLM 生成响应。
