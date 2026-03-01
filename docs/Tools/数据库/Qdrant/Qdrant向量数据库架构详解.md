# Qdrant 向量数据库架构深度解析 - 以智能销售系统为例

> **文档说明**：本文档结合 smart-sales 项目代码，深入讲解 Qdrant 向量数据库的架构设计、工作原理及在 RAG 系统中的应用。

## 目录

1. [Qdrant 简介与定位](#一qdrant-简介与定位)
2. [核心架构组件](#二核心架构组件)
3. [向量检索原理](#三向量检索原理)
4. [项目中的 Qdrant 实践](#四项目中的-qdrant-实践)
5. [Collection 设计](#五collection-设计)
6. [性能优化与最佳实践](#六性能优化与最佳实践)

---

## 一、Qdrant 简介与定位

### 1.1 什么是向量数据库

向量数据库是专门用于存储和检索高维向量的数据库系统。与传统数据库不同，它通过**相似度搜索**而非精确匹配来查询数据。

```
传统数据库查询: "查找 title = '产品价格' 的文档"
                ↓
         精确匹配 (B-Tree 索引)

向量数据库查询: "查找与'训练营多少钱'语义相似的文档"  
                ↓
         向量相似度搜索 (HNSW/余弦相似度)
```

### 1.2 Qdrant 在系统中的定位

在 smart-sales 智能销售系统中，Qdrant 作为 **RAG (Retrieval-Augmented Generation)** 基础设施：

```
┌─────────────────────────────────────────────────────────────┐
│                   RAG 架构全景图                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   用户问题: "你们训练营多少钱？"                             │
│                    ↓                                        │
│   ┌──────────────────────────────┐                         │
│   │   Embedding 模型              │                         │
│   │   (VoyageAI voyage-4-large)   │                         │
│   │   1024 维向量                 │                         │
│   └──────────────┬───────────────┘                         │
│                  ↓                                          │
│   ┌──────────────────────────────┐                         │
│   │   Qdrant 向量检索             │ ◄── 本文重点             │
│   │   - 原子事实库                │                         │
│   │   - 成功案例库                │                         │
│   └──────────────┬───────────────┘                         │
│                  ↓                                          │
│   ┌──────────────────────────────┐                         │
│   │   LLM (Claude/GPT-4o)         │                         │
│   │   生成回答: "我们的训练营      │                         │
│   │   根据年龄段分为多个价位..."   │                         │
│   └──────────────────────────────┘                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Qdrant 核心特性

| 特性 | 说明 | 项目中应用 |
|------|------|-----------|
| **HNSW 索引** | 近似最近邻搜索，毫秒级响应 | 实时语义检索 |
| **元数据过滤** | 向量搜索 + 属性过滤组合 | 按分类/评级筛选 |
| **多向量支持** | 同一集合多种向量类型 | - |
| **分布式** | 内置分片和复制 | 生产扩展 |
| **混合搜索** | 向量相似度 + 关键词 BM25 | 精确匹配增强 |

---

## 二、核心架构组件

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                   Qdrant 架构层次                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  API 层 (REST/gRPC)                  │   │
│  │  POST /collections/{name}/points/search              │   │
│  └─────────────────────┬───────────────────────────────┘   │
│                        ↓                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  查询处理层                          │   │
│  │  - 查询向量化                                        │   │
│  │  - 过滤器解析                                        │   │
│  │  - 检索策略选择                                      │   │
│  └─────────────────────┬───────────────────────────────┘   │
│                        ↓                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  索引层 (HNSW)                       │   │
│  │  - 图索引结构                                        │   │
│  │  - 分层导航                                          │   │
│  │  - 近似最近邻搜索                                    │   │
│  └─────────────────────┬───────────────────────────────┘   │
│                        ↓                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  存储层                              │   │
│  │  - 向量数据 (memmap/内存)                            │   │
│  │  - Payload 数据 (JSONB)                              │   │
│  │  - WAL 日志                                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 HNSW 索引算法详解

HNSW (Hierarchical Navigable Small World) 是 Qdrant 的核心检索算法：

```
┌─────────────────────────────────────────────────────────────┐
│                   HNSW 分层结构示意                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Layer 2 (稀疏层): ○────────○                               │
│                        ↓                                    │
│   Layer 1 (中层):   ○──○──○──○──○                            │
│                      ↓ ↓ ↓ ↓ ↓                              │
│   Layer 0 (稠密层): ○○○○○○○○○○○○○○  ← 全部数据点             │
│                                                             │
│   查询流程:                                                 │
│   1. 从顶层随机入口点开始                                    │
│   2. 贪心搜索找到当前层最近邻                                 │
│   3. 下降到下一层，以上一层结果为新入口                       │
│   4. 重复直到第 0 层，返回最近邻                              │
│                                                             │
│   时间复杂度: O(log N) 近似搜索                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**HNSW 关键参数**（创建 Collection 时配置）：

| 参数 | 说明 | 默认值 | 调优建议 |
|------|------|--------|---------|
| `m` | 每层最大连接数 | 16 | 增大提高召回率，但增加内存 |
| `ef_construct` | 构建时的搜索深度 | 100 | 增大提高索引质量，构建变慢 |
| `ef` | 查询时的搜索深度 | - | 查询时动态设置，越大越准越慢 |

### 2.3 存储引擎

Qdrant 支持多种存储模式：

```
┌─────────────────────────────────────────┐
│           Qdrant 存储模式               │
├─────────────────────────────────────────┤
│                                         │
│  1. 内存模式 (In-Memory)                 │
│     └── 向量全量加载到 RAM               │
│     └── 最快，但容量受限于内存           │
│                                         │
│  2. Mmap 模式 (推荐)                     │
│     └── 向量存储在磁盘，按需加载         │
│     └── 性能接近内存，容量受限于磁盘     │
│                                         │
│  3. 磁盘模式                             │
│     └── 适合超大索引，速度较慢           │
│                                         │
└─────────────────────────────────────────┘
```

---

## 三、向量检索原理

### 3.1 相似度度量方法

Qdrant 支持多种距离/相似度计算方式：

| 距离类型 | 公式 | 适用场景 | 项目中使用 |
|---------|------|---------|-----------|
| **Cosine** | 1 - (A·B)/(\|A\|\|B\|) | 语义相似度（归一化向量） | ✅ 默认使用 |
| **Euclidean** | \|A-B\| | 绝对距离 | - |
| **Dot Product** | A·B | 推荐系统 | - |
| **Manhattan** | Σ\|Ai-Bi\| | 稀疏向量 | - |

**余弦相似度直观理解**：
```
向量夹角越小 → 方向越接近 → 语义越相似 → 得分越高

    ↑                 ↗
    |               ↗
  A ●             ● B
    |           ↗
    └─────────→
    夹角 30° = 相似度 0.87
```

### 3.2 检索流程详解

```
┌─────────────────────────────────────────────────────────────┐
│                Qdrant 向量检索完整流程                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 查询向量化                                               │
│     查询文本: "训练营多少钱"                                  │
│              ↓                                              │
│     Embedding 模型 → [0.023, -0.156, 0.089, ...] (1024维)    │
│                                                             │
│  2. 可选: 元数据过滤条件                                      │
│     category = "price" AND status = "active"                │
│                                                             │
│  3. HNSW 近似搜索                                            │
│     - 从入口点开始贪心遍历                                    │
│     - ef=100 控制搜索深度                                     │
│     - 找到 top_k 个最近邻                                     │
│                                                             │
│  4. 相似度计算                                               │
│     - 精确计算候选集与查询向量的余弦相似度                     │
│     - 按得分降序排列                                          │
│                                                             │
│  5. 阈值过滤                                                 │
│     - 过滤 score < threshold 的结果                          │
│     - 项目中默认 threshold = 0.35                            │
│                                                             │
│  6. 返回结果                                                 │
│     [{id, score, payload}, ...]                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 过滤机制

Qdrant 支持在向量搜索前应用过滤条件，减少搜索空间：

```python
# 项目中的过滤示例 (backend/src/smart_sales/decision/rag/retriever.py)
from qdrant_client.models import FieldCondition, Filter, MatchValue

# 构建过滤条件: category = "price"
query_filter = Filter(
    must=[FieldCondition(key="category", match=MatchValue(value="price"))]
)

# 组合检索
response = await client.query_points(
    collection_name="atomic_facts",
    query=vector,
    query_filter=query_filter,  # 先过滤，再向量搜索
    limit=5,
    score_threshold=0.35,
)
```

**支持的过滤条件类型**：

| 条件类型 | 说明 | 示例 |
|---------|------|------|
| `MatchValue` | 精确匹配 | `category = "faq"` |
| `MatchAny` | 多值匹配 | `tags IN ["促销", "新客"]` |
| `Range` | 范围匹配 | `created_at > 2024-01-01` |
| `GeoBoundingBox` | 地理围栏 | 位置范围 |
| `IsEmpty` | 空值判断 | `description IS NULL` |

---

## 四、项目中的 Qdrant 实践

### 4.1 连接管理

项目使用 `AsyncQdrantClient` 进行异步连接：

```python
# backend/src/smart_sales/core/database.py
from qdrant_client import AsyncQdrantClient

_qdrant_client: AsyncQdrantClient | None = None

def get_qdrant() -> AsyncQdrantClient:
    """获取或创建 Qdrant 异步客户端。"""
    global _qdrant_client
    if _qdrant_client is None:
        settings = get_settings()
        _qdrant_client = AsyncQdrantClient(url=settings.QDRANT_URL)
    return _qdrant_client
```

**配置项**：

```python
# backend/src/smart_sales/core/config.py
class Settings(BaseSettings):
    # Qdrant 连接配置
    QDRANT_URL: str = Field(
        default="http://localhost:6333",
        description="Qdrant 向量数据库 URL",
    )
    
    # RAG 集合配置
    RAG_ATOMIC_FACTS_COLLECTION: str = "atomic_facts"
    RAG_SUCCESS_CASES_COLLECTION: str = "success_cases"
    RAG_EMBEDDING_DIM: int = 1024  # VoyageAI 向量维度
    RAG_TOP_K: int = 5
    RAG_SCORE_THRESHOLD: float = 0.35
```

### 4.2 Collection 初始化

项目在启动时幂等创建所需集合：

```python
# backend/src/smart_sales/decision/rag/collections.py
from qdrant_client.models import Distance, VectorParams

async def ensure_collections() -> None:
    """确保 RAG 所需的 Qdrant collection 存在，不存在则创建。"""
    settings = get_settings()
    client = get_qdrant()
    dim = settings.RAG_EMBEDDING_DIM

    collections = [
        settings.RAG_ATOMIC_FACTS_COLLECTION,
        settings.RAG_SUCCESS_CASES_COLLECTION,
    ]

    for name in collections:
        exists = await client.collection_exists(name)
        if exists:
            logger.debug(f"Qdrant collection 已存在，跳过创建: {name}")
            continue

        await client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info(f"Qdrant collection 已创建: {name} (dim={dim})")
```

### 4.3 文档入库 (Indexer)

将原子事实向量化并写入 Qdrant：

```python
# backend/src/smart_sales/decision/rag/indexer.py
from qdrant_client.models import PointStruct

async def index_atomic_fact(
    *,
    doc_id: str,
    title: str,
    content: str,
    category: str,
    tags: list[str] | None = None,
    status: str = "active",
) -> None:
    """将单条原子事实向量化并 upsert 到 Qdrant。"""
    settings = get_settings()
    client = get_qdrant()
    
    # 1. 合并标题和内容进行向量化
    text = f"{title}\n{content}"
    [vector] = await embed_documents([text])

    # 2. 将字符串 ID 哈希为 uint64 (Qdrant 要求)
    point_id = int(hashlib.md5(doc_id.encode()).hexdigest(), 16) % (2**63)
    
    # 3. 构建 Point
    point = PointStruct(
        id=point_id,
        vector=vector,
        payload={
            "doc_id": doc_id,
            "title": title,
            "content": content,
            "category": category,
            "tags": tags or [],
            "status": status,
        },
    )
    
    # 4. Upsert (存在则更新，不存在则插入)
    await client.upsert(
        collection_name=settings.RAG_ATOMIC_FACTS_COLLECTION,
        points=[point],
    )
```

### 4.4 语义检索 (Retriever)

```python
# backend/src/smart_sales/decision/rag/retriever.py
from dataclasses import dataclass

@dataclass
class RAGResult:
    """单条检索结果。"""
    doc_id: str
    title: str
    content: str
    score: float
    payload: dict

async def search_atomic_facts(
    query: str,
    *,
    category: str | None = None,
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> list[RAGResult]:
    """在原子事实库中语义检索。"""
    settings = get_settings()
    client = get_qdrant()
    
    # 1. 查询向量化
    vector = await embed_query(query)
    
    # 2. 构建过滤条件
    query_filter: Filter | None = None
    if category:
        query_filter = Filter(
            must=[FieldCondition(key="category", match=MatchValue(value=category))]
        ]
    )
    
    # 3. 执行检索
    response = await client.query_points(
        collection_name=settings.RAG_ATOMIC_FACTS_COLLECTION,
        query=vector,
        limit=top_k or settings.RAG_TOP_K,
        score_threshold=score_threshold or settings.RAG_SCORE_THRESHOLD,
        query_filter=query_filter,
        with_payload=True,
    )
    
    # 4. 转换为业务对象
    return [
        RAGResult(
            doc_id=h.payload.get("doc_id", ""),
            title=h.payload.get("title", ""),
            content=h.payload.get("content", ""),
            score=h.score,
            payload=h.payload,
        )
        for h in response.points
    ]
```

### 4.5 双写策略 (PG + Qdrant)

项目采用 **PostgreSQL 作为 Source of Truth，Qdrant 作为检索索引** 的双写架构：

```
┌─────────────────────────────────────────────────────────────┐
│                    双写架构数据流                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   创建/更新原子事实                                          │
│          ↓                                                  │
│   ┌──────────────┐     ┌──────────────┐                    │
│   │  PostgreSQL  │────▶│   Qdrant     │                    │
│   │  (主存储)    │     │  (检索索引)   │                    │
│   │  - 完整数据  │     │  - 向量      │                    │
│   │  - 事务支持  │     │  - 元数据    │                    │
│   │  - 软删除    │     │  - 点删除    │                    │
│   └──────────────┘     └──────────────┘                    │
│          ↑                          ↑                       │
│     Source of Truth           仅用于检索                    │
│                                                             │
│   失败策略:                                                 │
│   - PG 失败 → 整体回滚                                      │
│   - Qdrant 失败 → 只记录 warning，不影响 PG                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**服务层实现**：

```python
# backend/src/smart_sales/services/knowledge_service.py
class KnowledgeService:
    async def create_atomic_fact(self, db: AsyncSession, *, data: dict) -> AtomicFact:
        """创建原子事实（双写 PG + Qdrant）。"""
        # 1. 写入 PG（事务内）
        fact = await self._repo.create(db, obj_in=data)
        
        # 2. 同步到 Qdrant（异步，失败不影响 PG）
        await self._sync_to_qdrant(fact)
        return fact

    async def _sync_to_qdrant(self, fact: AtomicFact) -> None:
        """将原子事实同步到 Qdrant（失败只 warning）。"""
        try:
            from smart_sales.decision.rag.indexer import index_atomic_fact
            await index_atomic_fact(
                doc_id=fact.id,
                title=fact.title,
                content=fact.content,
                category=fact.category,
                tags=list(fact.tags) if fact.tags else [],
                status="archived" if fact.is_archived else "active",
            )
        except Exception as exc:
            logger.warning(f"原子事实 {fact.id} 同步 Qdrant 失败（不影响 PG）: {exc}")

    async def archive_atomic_fact(self, db: AsyncSession, *, fact_id: str) -> Optional[AtomicFact]:
        """归档原子事实（PG 软删 + Qdrant 物理删除）。"""
        # 1. PG 软删除
        db_obj = await self._repo.get(db, id=fact_id)
        fact = await self._repo.update(db, db_obj=db_obj, obj_in={"is_archived": True})
        
        # 2. Qdrant 物理删除
        await self._delete_from_qdrant(fact_id)
        return fact
```

### 4.6 Embedding 服务

项目使用 VoyageAI 进行文本向量化：

```python
# backend/src/smart_sales/decision/rag/embeddings.py
import voyageai

_client: voyageai.AsyncClient | None = None

async def embed_query(text: str) -> list[float]:
    """将查询文本转为向量（input_type='query'）。"""
    settings = get_settings()
    client = _get_client()
    result = await client.embed(
        [text],
        model=settings.VOYAGE_EMBEDDING_MODEL,  # voyage-4-large
        input_type="query",  # 查询类型优化
    )
    return result.embeddings[0]  # 1024 维

async def embed_documents(texts: list[str], batch_size: int = 128) -> list[list[float]]:
    """将文档列表批量转为向量（input_type='document'）。"""
    all_embeddings: list[list[float]] = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = await _get_client().embed(
            batch,
            model=settings.VOYAGE_EMBEDDING_MODEL,
            input_type="document",  # 文档类型优化
        )
        all_embeddings.extend(result.embeddings)
    
    return all_embeddings
```

**input_type 的作用**：
- `query`: 针对短查询优化，强调相关性
- `document`: 针对长文档优化，强调语义完整性

---

## 五、Collection 设计

### 5.1 原子事实库 (atomic_facts)

```yaml
Collection: atomic_facts
  Vector Parameters:
    size: 1024          # VoyageAI 向量维度
    distance: COSINE    # 余弦相似度
  
  Payload Schema:
    doc_id: string      # 业务侧唯一 ID
    title: string       # 标题
    content: string     # 内容
    category: string    # 分类 (product/price/faq/policy)
    tags: [string]      # 标签列表
    status: string      # 状态 (active/archived)
```

### 5.2 成功案例库 (success_cases)

```yaml
Collection: success_cases
  Vector Parameters:
    size: 1024
    distance: COSINE
  
  Payload Schema:
    doc_id: string      # 业务侧唯一 ID
    title: string       # 案例标题
    content: string     # 案例正文
    grade: string       # 案例评级 (S/A/B/C)
    industry: string    # 行业标签
    tags: [string]      # 标签列表
```

### 5.3 数据模型对比

| 特性 | PostgreSQL (主存储) | Qdrant (检索索引) |
|------|-------------------|------------------|
| 数据完整性 | 完整业务数据 | 向量化内容 + 关键元数据 |
| 查询方式 | 精确查询、范围查询 | 语义相似度搜索 |
| 事务支持 | ✅ ACID | ❌ 最终一致性 |
| 软删除 | ✅ is_archived 字段 | 物理删除 Point |
| 适用场景 | 数据管理、CRUD | RAG 检索、推荐 |

---

## 六、性能优化与最佳实践

### 6.1 索引优化

```python
# HNSW 参数调优建议
# 在创建 collection 时通过 hnsw_config 参数配置

from qdrant_client.models import HnswConfigDiff

await client.create_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    hnsw_config=HnswConfigDiff(
        m=16,                # 增大提高召回率
        ef_construct=100,    # 增大提高索引质量
    ),
)
```

### 6.2 批量操作

```python
# 批量 upsert 提高写入性能
points = [
    PointStruct(id=1, vector=[...], payload={...}),
    PointStruct(id=2, vector=[...], payload={...}),
    # ...
]

await client.upsert(
    collection_name="atomic_facts",
    points=points,
    batch_size=100,  # 分批写入
)
```

### 6.3 查询优化

```python
# 1. 使用 score_threshold 过滤低质量结果
response = await client.query_points(
    collection_name="atomic_facts",
    query=vector,
    score_threshold=0.35,  # 低于此分数的结果被过滤
    limit=10,
)

# 2. 限制返回 payload 字段
response = await client.query_points(
    collection_name="atomic_facts",
    query=vector,
    with_payload=["title", "content"],  # 只返回需要的字段
)

# 3. 使用滚动搜索避免 deep paging
# 不推荐: offset=10000
# 推荐: 使用 scroll API 或 last_seen_id
```

### 6.4 监控指标

| 指标 | 说明 | 健康阈值 |
|------|------|---------|
| 检索延迟 (p99) | 向量搜索响应时间 | < 100ms |
| 召回率 | 返回的相关结果比例 | > 90% |
| 索引大小 | 向量索引内存占用 | < 80% RAM |
| QPS | 每秒查询数 | 视硬件而定 |

### 6.5 Docker 部署配置

```yaml
# docker-compose.yml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"  # REST API
      - "6334:6334"  # gRPC API
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__LOG_LEVEL=INFO
    restart: unless-stopped

volumes:
  qdrant_data:
```

---

## 参考资料

- [Qdrant 官方文档](https://qdrant.tech/documentation/)
- [HNSW 论文](https://arxiv.org/abs/1603.09320)
- [VoyageAI Embedding 模型](https://docs.voyageai.com/)
- [smart-sales 项目源码](../../backend/src/smart_sales/decision/rag/)

---

*文档生成时间：2026-03-02*  
*基于 Qdrant 1.12+*
