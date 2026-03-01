# MongoDB 架构深度解析 - 以智能销售系统为例

> **文档说明**：本文档结合 smart-sales 项目代码，深入讲解 MongoDB 的架构设计、工作流程及核心组件。

## 目录

1. [架构概览](#一架构概览)
2. [核心组件](#二核心组件)
3. [数据模型设计](#三数据模型设计)
4. [工作流程](#四工作流程)
5. [项目中的 MongoDB 实践](#五项目中的-mongodb-实践)
6. [高级特性](#六高级特性)

---

## 一、架构概览

### 1.1 MongoDB 在系统中的定位

在 smart-sales 智能销售系统中，MongoDB 与 PostgreSQL、Redis、Qdrant 构成异构存储架构：

```
┌─────────────────────────────────────────────────────────────┐
│                    智能销售系统存储架构                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PostgreSQL ◄── 核心业务数据 (客户、用户、订单)                │
│       │                                                    │
│  MongoDB    ◄── 聊天记录 / 行为轨迹 (非结构化、高写入)          │
│       │                                                    │
│  Redis      ◄── 缓存 / 会话 / 任务队列                        │
│       │                                                    │
│  Qdrant     ◄── 向量知识库 (RAG 检索增强)                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**选择 MongoDB 存储聊天记录的原因**：
- 聊天记录数据量大、结构多变
- 需要灵活的模式设计（每条消息可能有不同字段）
- 高写入吞吐量（实时消息接入）
- 支持文本搜索和时间范围查询

### 1.2 MongoDB 部署架构

```
┌─────────────────────────────────────────────────────────────┐
│                   MongoDB 部署架构                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐                                            │
│  │  mongod     │  ◄── 主数据服务进程                         │
│  │  (单节点)   │      监听 27017 端口                        │
│  └──────┬──────┘                                            │
│         │                                                   │
│         ▼                                                   │
│  ┌───────────────────────────────────────┐                 │
│  │         WiredTiger 存储引擎            │                 │
│  │  ┌─────────┐  ┌─────────┐  ┌────────┐ │                 │
│  │  │ 内存    │  │ 缓存    │  │ 压缩   │ │                 │
│  │  │映射存储 │  │ 淘汰    │  │ 算法   │ │                 │
│  │  └─────────┘  └─────────┘  └────────┘ │                 │
│  └───────────────────────────────────────┘                 │
│                                                             │
│  ┌───────────────────────────────────────┐                 │
│  │            数据文件层                  │                 │
│  │  /data/db/                            │                 │
│  │    ├── collection-*.wt                │                 │
│  │    ├── index-*.wt                     │                 │
│  │    └── journal/                       │                 │
│  └───────────────────────────────────────┘                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、核心组件

### 2.1 mongod - 数据库服务进程

`mongod` 是 MongoDB 的核心守护进程，负责：
- 管理数据库连接
- 执行数据读写操作
- 维护索引
- 处理复制和分片

**项目中的连接配置**：

```python
# backend/src/smart_sales/core/config.py
class Settings(BaseSettings):
    # MongoDB 连接 URI
    MONGODB_URL: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB 连接 URI",
    )
    MONGODB_DATABASE: str = "smart_sales"
```

**URI 格式解析**：
```
mongodb://[用户名:密码@]主机[:端口][/数据库][?选项]
```

**常见连接选项**：
- `?replicaSet=rs0` - 指定复制集
- `?authSource=admin` - 认证数据库
- `?ssl=true` - 启用 SSL/TLS

### 2.2 存储引擎 - WiredTiger

MongoDB 3.2+ 的默认存储引擎，提供：

| 特性 | 说明 |
|------|------|
| 文档级并发控制 | 读写操作不阻塞，支持 MVCC |
| 压缩 | 默认 Snappy 压缩，可选 zlib/zstd |
| 缓存 | 使用 50% RAM - 1GB 作为缓存 |
| 日志 (Journal) | 保证崩溃恢复能力 |

**存储结构**：
```
数据文件 (*.wt)
├── 集合数据 (Collections)
│   └── chat_messages.wt  # 聊天记录集合
├── 索引 (Indexes)
│   └── index-{id}.wt     # B-Tree 索引
└── 元数据
    └── _mdb_catalog.wt   # 目录文件
```

### 2.3 内存管理

```
┌─────────────────────────────────────────┐
│            MongoDB 内存布局              │
├─────────────────────────────────────────┤
│  预留内存 (~1GB)                        │
├─────────────────────────────────────────┤
│  WiredTiger Cache (50% RAM - 1GB)       │
│  ├── 热数据 (最常访问)                   │
│  ├── 温数据 (偶尔访问)                   │
│  └── 脏页 (待写入磁盘)                   │
├─────────────────────────────────────────┤
│  文件系统缓存 (剩余内存)                 │
│  └── 压缩后的数据页                      │
└─────────────────────────────────────────┘
```

---

## 三、数据模型设计

### 3.1 BSON 文档格式

MongoDB 使用 BSON（Binary JSON）存储数据，相比 JSON 的优势：
- 支持更多数据类型（Date、ObjectId、Binary 等）
- 遍历速度更快
- 支持内嵌文档和数组

**项目中的文档示例**（聊天记录）：

```javascript
// chat_messages 集合中的文档
{
  "_id": ObjectId("65a1b2c3d4e5f6a7b8c9d0e1"),
  "customer_id": "customer_a1b2c3",
  "sales_id": "sales_1234",
  "content": "请问训练营多少钱？",
  "msg_type": "text",
  "timestamp": ISODate("2024-01-15T08:30:00Z"),
  "sender": "customer",
  "ai_categories": [
    {"category": "价格", "confidence": 0.95}
  ],
  "metadata": {
    "source": "wecom",
    "message_id": "msg_xxx"
  }
}
```

**BSON 数据类型对照表**：

| BSON 类型 | Python 类型 | 说明 |
|-----------|-------------|------|
| ObjectId | `bson.ObjectId` | 12字节唯一ID |
| String | `str` | UTF-8 字符串 |
| Date | `datetime.datetime` | 时间戳 |
| Int32 | `int` | 32位整数 |
| Int64 | `int` | 64位整数 |
| Double | `float` | 浮点数 |
| Boolean | `bool` | 布尔值 |
| Array | `list` | 数组 |
| Document | `dict` | 内嵌文档 |
| Binary | `bytes` | 二进制数据 |

### 3.2 集合（Collection）

集合是 MongoDB 中存储文档的容器，类似于关系数据库的表，但：
- 无需预定义模式（Schema-less）
- 同一集合的文档可以有不同的字段
- 自动创建 `_id` 字段作为主键

**项目中的集合设计**：

```python
# backend/src/smart_sales/repositories/chat.py
class ChatRepository:
    """聊天记录数据访问层（MongoDB）。"""
    
    COLLECTION_NAME = "chat_messages"
    
    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db
        self._collection = db[self.COLLECTION_NAME]
```

### 3.3 数据库（Database）

数据库是集合的逻辑分组：

```python
# backend/src/smart_sales/core/config.py
MONGODB_DATABASE: str = "smart_sales"

# 生成的数据库结构
smart_sales/
├── chat_messages/          # 聊天记录集合
│   ├── 文档1
│   ├── 文档2
│   └── ...
├── customer_events/        # 客户事件（可选）
└── system.indexes/         # 系统索引集合
```

---

## 四、工作流程

### 4.1 写入流程

```
┌─────────────────────────────────────────────────────────────┐
│                   MongoDB 写入流程                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 应用层                                                    │
│     └─> insert_one(document)                                │
│                                                             │
│  2. 驱动层 (Motor/PyMongo)                                    │
│     └─> 序列化为 BSON                                        │
│                                                             │
│  3. 网络传输                                                  │
│     └─> TCP 发送到 mongod:27017                             │
│                                                             │
│  4. 存储引擎层 (WiredTiger)                                   │
│     ├─> 写入 Journal (WAL) ──┐                              │
│     ├─> 更新内存中的 B-Tree   │  崩溃恢复                     │
│     └─> 标记脏页             ◄─┘                              │
│                                                             │
│  5. 后台刷盘                                                  │
│     └─> 检查点 (Checkpoint) 将脏页写入 .wt 文件              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**项目代码示例**：

```python
# backend/src/smart_sales/repositories/chat.py
async def insert(self, record: dict[str, Any]) -> str:
    """插入一条聊天记录。"""
    result = await self._collection.insert_one(record)
    return str(result.inserted_id)
```

**写入优化策略**：
- 批量插入 (`insert_many`) 比单条插入更高效
- 使用无序写入 (`ordered=False`) 提高并发性能
- 合理设置 WriteConcern（权衡一致性与性能）

### 4.2 查询流程

```
┌─────────────────────────────────────────────────────────────┐
│                   MongoDB 查询流程                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 查询解析                                                  │
│     └─> 解析查询条件、投影、排序、限制等                       │
│                                                             │
│  2. 查询优化器                                                │
│     ├─> 评估可用的索引                                        │
│     └─> 选择最优查询计划 (Query Plan)                         │
│                                                             │
│  3. 索引检索 (如有)                                           │
│     └─> B-Tree 索引查找                                       │
│                                                             │
│  4. 文档获取                                                  │
│     ├─> 从 WiredTiger 缓存读取（命中）                        │
│     └─> 或从磁盘加载（未命中）                                │
│                                                             │
│  5. 结果处理                                                  │
│     ├─> 应用投影 (Projection)                                 │
│     ├─> 排序 (Sort)                                          │
│     └─> 限制/跳过 (Limit/Skip)                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**项目代码示例**（带过滤的查询）：

```python
# backend/src/smart_sales/repositories/chat.py
async def get_history(
    self,
    *,
    customer_id: str,
    sales_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """获取客户聊天历史。"""
    query: dict[str, Any] = {"customer_id": customer_id}
    if sales_id is not None:
        query["sales_id"] = sales_id

    cursor = (
        self._collection.find(query, {"_id": 0})  # 排除 _id 字段
        .sort("timestamp", -1)                     # 按时间倒序
        .skip(offset)
        .limit(limit)
    )
    return await cursor.to_list(length=limit)
```

### 4.3 索引机制

MongoDB 默认在 `_id` 字段创建唯一索引，支持多种索引类型：

| 索引类型 | 适用场景 | 项目使用 |
|---------|---------|---------|
| 单字段索引 | 等值查询、范围查询 | `customer_id` |
| 复合索引 | 多条件组合查询 | `customer_id + timestamp` |
| 文本索引 | 全文搜索 | `content`（正则查询）|
| TTL 索引 | 自动过期删除 | - |
| 哈希索引 | 等值查询、分片键 | - |

**项目中的查询优化建议**：

```javascript
// 为常见查询创建复合索引
db.chat_messages.createIndex(
  { "customer_id": 1, "timestamp": -1 },
  { name: "idx_customer_time" }
)

// 为文本搜索创建文本索引
db.chat_messages.createIndex(
  { "content": "text" },
  { name: "idx_content_text" }
)
```

---

## 五、项目中的 MongoDB 实践

### 5.1 连接管理

项目使用 **Motor** 作为异步 MongoDB 驱动：

```python
# backend/src/smart_sales/core/database.py
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

# 模块级单例（进程内复用）
_mongo_client: AsyncIOMotorClient | None = None

def get_mongo_client() -> AsyncIOMotorClient:
    """获取或创建 Motor 异步客户端。"""
    global _mongo_client
    if _mongo_client is None:
        settings = get_settings()
        _mongo_client = AsyncIOMotorClient(settings.MONGODB_URL)
    return _mongo_client

def get_mongo_db() -> AsyncIOMotorDatabase:
    """获取默认 MongoDB 数据库实例。"""
    settings = get_settings()
    return get_mongo_client()[settings.MONGODB_DATABASE]

# 应用关闭时清理
async def close_all_connections() -> None:
    if _mongo_client is not None:
        _mongo_client.close()
        _mongo_client = None
```

**设计要点**：
- 使用模块级单例避免重复创建连接
- 利用 Motor 的连接池管理（默认最大 100 连接）
- 在 FastAPI lifespan 中统一关闭连接

### 5.2 Repository 模式

项目采用 Repository 模式封装 MongoDB 操作：

```python
# backend/src/smart_sales/repositories/chat.py
class ChatRepository:
    """聊天记录数据访问层（MongoDB）。"""

    COLLECTION_NAME = "chat_messages"

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db
        self._collection = db[self.COLLECTION_NAME]

    async def get_history(...) -> list[dict[str, Any]]:
        """获取聊天历史。"""
        ...

    async def search(...) -> list[dict[str, Any]]:
        """全文搜索聊天记录。"""
        ...

    async def insert(self, record: dict[str, Any]) -> str:
        """插入聊天记录。"""
        ...
```

**API 层使用**：

```python
# backend/src/smart_sales/api/v1/chats.py
def _get_chat_repo() -> ChatRepository:
    """获取 ChatRepository 实例（依赖 MongoDB）。"""
    return ChatRepository(get_mongo_db())

@router.get("/{customer_id}/history")
async def get_chat_history(
    customer_id: str,
    chat_repo: ChatRepository = Depends(_get_chat_repo),
) -> dict[str, Any]:
    messages = await chat_repo.get_history(customer_id=customer_id)
    return {"items": messages, "customer_id": customer_id}
```

### 5.3 数据模式设计

**聊天记录 Schema**（Pydantic 定义）：

```python
# backend/src/smart_sales/schemas/chat.py
class ChatRecordResponse(BaseSchema):
    """聊天记录响应。"""
    id: str = Field(description="MongoDB ObjectId")
    customer_id: str = Field(description="客户 ID")
    sales_id: str = Field(description="销售 ID")
    content: str = Field(description="聊天内容或文件 URL")
    msg_type: str = Field(description="消息类型: text/image/file")
    timestamp: datetime = Field(description="消息时间戳")
    ai_categories: Optional[list[dict[str, Any]]] = Field(
        default=None, description="AI 提取的话题分类"
    )
```

**设计特点**：
- 使用内嵌数组存储 AI 分类结果
- 时间戳使用 `datetime` 类型便于范围查询
- 保留 `_id` 但 API 层映射为 `id` 字符串

### 5.4 测试实践

项目使用 pytest 进行 MongoDB 集成测试：

```python
# backend/tests/test_core/test_mongodb.py
@pytest.fixture
def mongo_collection():
    """创建独立测试集合，测试后清理。"""
    settings = get_settings()
    client = MongoClient(settings.MONGODB_URL)
    db = client[settings.MONGODB_DATABASE]
    collection_name = f"test_collection_{uuid4().hex}"
    collection = db[collection_name]
    try:
        yield collection
    finally:
        db.drop_collection(collection_name)
        client.close()

@pytest.mark.integration
class TestMongoDB:
    def test_insert_and_find(self, mongo_collection) -> None:
        doc = {"name": "Test 1", "timestamp": datetime.now()}
        result = mongo_collection.insert_one(doc)
        assert result.inserted_id is not None
```

### 5.5 Docker 部署配置

```yaml
# docker-compose.yml
services:
  mongodb:
    image: mongo:7
    profiles: ["infra"]
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
    networks:
      - smart-sales-net
    restart: unless-stopped

volumes:
  mongodb_data:
```

---

## 六、高级特性

### 6.1 聚合管道（Aggregation Pipeline）

MongoDB 强大的数据处理框架，支持复杂的数据分析：

```javascript
// 统计每日消息量
db.chat_messages.aggregate([
  { $match: { customer_id: "customer_123" } },
  { 
    $group: {
      _id: { $dateToString: { format: "%Y-%m-%d", date: "$timestamp" } },
      count: { $sum: 1 }
    }
  },
  { $sort: { _id: 1 } }
])
```

**常用管道阶段**：
| 阶段 | 功能 |
|------|------|
| `$match` | 过滤文档 |
| `$group` | 分组聚合 |
| `$sort` | 排序 |
| `$project` | 投影字段 |
| `$lookup` | 关联查询 |
| `$unwind` | 展开数组 |

### 6.2 复制集（Replica Set）

生产环境推荐架构，提供高可用和数据冗余：

```
┌─────────────────────────────────────────┐
│           Replica Set 架构              │
│              (3 节点)                   │
├─────────────────────────────────────────┤
│                                         │
│   ┌──────────┐                         │
│   │ Primary  │ ◄── 读写操作            │
│   │ 主节点    │                         │
│   └────┬─────┘                         │
│        │                                │
│        │ Oplog 同步                     │
│        ▼                                │
│   ┌──────────┐    ┌──────────┐         │
│   │Secondary │◄──►│Secondary │         │
│   │ 从节点    │    │ 从节点    │         │
│   └──────────┘    └──────────┘         │
│                                         │
│   Arbiter (可选) - 仅参与选举，不存数据   │
│                                         │
└─────────────────────────────────────────┘
```

**故障转移**：
- Primary 宕机时，Secondary 自动选举为新 Primary
- 应用层通过 MongoDB URI 配置多个节点自动重连

### 6.3 分片（Sharding）

用于水平扩展存储容量和读写性能：

```
┌─────────────────────────────────────────┐
│           Sharded Cluster               │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────┐                           │
│  │  mongos  │ ◄── 查询路由              │
│  │  路由器   │                           │
│  └────┬─────┘                           │
│       │                                 │
│       ▼                                 │
│  ┌──────────────┐                       │
│  │ Config Server │ 存储分片元数据        │
│  └──────────────┘                       │
│                                         │
│       ┌──────────────┐                  │
│       │ Shard 1      │                  │
│       │ (副本集)     │                  │
│       └──────────────┘                  │
│                                         │
│       ┌──────────────┐                  │
│       │ Shard 2      │                  │
│       │ (副本集)     │                  │
│       └──────────────┘                  │
│                                         │
└─────────────────────────────────────────┘
```

**分片键选择原则**：
- 高基数（Cardinality）字段
- 写操作分布均匀
- 支持常见查询模式

---

## 七、最佳实践总结

### 7.1 项目中的 MongoDB 使用规范

1. **连接管理**
   - 使用单例模式管理客户端连接
   - 配置合理的连接池大小
   - 应用关闭时显式释放连接

2. **文档设计**
   - 优先内嵌，谨慎引用
   - 避免文档过大（< 16MB）
   - 关键字段添加索引

3. **查询优化**
   - 使用投影减少返回字段
   - 避免大范围 skip，改用范围查询
   - 监控慢查询日志

4. **数据安全**
   - 启用身份验证
   - 定期备份（mongodump/mongorestore）
   - 生产环境使用复制集

### 7.2 性能调优建议

```javascript
// 1. 创建合适的索引
db.chat_messages.createIndex({ customer_id: 1, timestamp: -1 })

// 2. 限制返回结果
db.chat_messages.find().limit(100)

// 3. 使用投影减少数据传输
db.chat_messages.find({}, { content: 1, timestamp: 1 })

// 4. 批量操作
const bulk = db.chat_messages.initializeUnorderedBulkOp()
bulk.insert({ ... })
bulk.insert({ ... })
bulk.execute()
```

---

## 参考资料

- [MongoDB 官方文档](https://docs.mongodb.com/)
- [Motor 异步驱动文档](https://motor.readthedocs.io/)
- [WiredTiger 存储引擎](https://source.wiredtiger.com/)
- smart-sales 项目源码: `../../backend/src/smart_sales/core/database.py`

---

*文档生成时间：2026-03-02*  
*基于 MongoDB 7.0 和 Motor 3.7*
