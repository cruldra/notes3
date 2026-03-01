# Celery 分布式任务队列详解 - 基于 Smart Sales 项目实践

> 本文档结合智能销售助手系统（Smart Sales）的实际代码，深入讲解 Celery 的架构设计、工作流程和核心组件。

---

## 📋 目录

1. [什么是 Celery](#什么是-celery)
2. [系统架构概览](#系统架构概览)
3. [核心组件详解](#核心组件详解)
4. [工作流程](#工作流程)
5. [项目中的实践](#项目中的实践)
6. [配置详解](#配置详解)
7. [最佳实践与注意事项](#最佳实践与注意事项)

---

## 什么是 Celery

**Celery** 是一个简单、灵活且可靠的分布式任务队列系统，基于 Python 开发。它主要用于：

- **异步任务处理**：将耗时操作（如发送邮件、图像处理、AI 计算）从主请求中剥离
- **定时任务调度**：通过 Celery Beat 实现类似 cron 的定时任务
- **分布式计算**：支持多 Worker 横向扩展，处理海量任务

### 为什么选择 Celery？

| 特性 | 说明 |
|------|------|
| **高可用** | 支持多个 Broker（Redis、RabbitMQ、Amazon SQS） |
| **可扩展** | 水平扩展 Worker，支持分布式部署 |
| **灵活性** | 支持任务重试、延迟执行、任务链、组任务等 |
| **监控** | Flower 提供 Web UI 监控任务状态 |
| **生态丰富** | 与 Django、Flask、FastAPI 等框架无缝集成 |

---

## 系统架构概览

### Celery 在 Smart Sales 中的部署架构

```
┌─────────────────────────────────────────────────────────────────┐
│                          Smart Sales 系统                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│   │   API 服务    │      │  Celery      │      │  Celery      │  │
│   │  (FastAPI)   │──────▶│   Worker     │      │    Beat      │  │
│   └──────────────┘      └──────┬───────┘      └──────────────┘  │
│          │                     │                                │
│          │                     │                                │
│          ▼                     ▼                                │
│   ┌──────────────────────────────────────┐                     │
│   │            Redis 服务                 │                     │
│   │  ┌─────────────┐ ┌────────────────┐  │                     │
│   │  │   Broker    │ │ Result Backend │  │                     │
│   │  │   (DB 1)    │ │    (DB 2)      │  │                     │
│   │  └─────────────┘ └────────────────┘  │                     │
│   └──────────────────────────────────────┘                     │
│                                                                  │
│   ┌──────────────────────────────────────┐                     │
│   │         Flower 监控面板               │                     │
│   │         (Port: 5555)                 │                     │
│   └──────────────────────────────────────┘                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 项目中的 Celery 服务拆分

根据 `docker-compose.yml` 配置，Celery 被拆分为三个独立服务：

| 服务 | 容器 | 职责 | 启动命令 |
|------|------|------|----------|
| **Worker** | worker | 执行实际任务 | `celery -A smart_sales.celery_app worker` |
| **Beat** | beat | 定时任务调度 | `celery -A smart_sales.celery_app beat` |
| **Flower** | flower | Web 监控面板 | `celery -A smart_sales.celery_app flower` |

---

## 核心组件详解

### 1. Celery App（应用实例）

**文件位置**：`backend/src/smart_sales/celery_app.py`

```python
from celery import Celery
from smart_sales.core.config import get_settings

settings = get_settings()

# 创建 Celery 应用实例
celery_app = Celery(
    "smart_sales",                          # 应用名称
    broker=settings.CELERY_BROKER_URL,      # 消息队列地址
    backend=settings.CELERY_RESULT_BACKEND, # 结果存储地址
)
```

**核心概念**：
- **Celery 实例**：每个项目应只有一个 Celery 应用实例
- **Broker**：任务队列的存储后端（本项目使用 Redis）
- **Backend**：任务执行结果的存储后端（本项目使用 Redis）

### 2. Broker（消息中间件）

**项目配置**（`backend/src/smart_sales/core/config.py`）：

```python
CELERY_BROKER_URL: str = Field(
    default="redis://localhost:6379/1",  # Redis DB 1 作为 Broker
    description="Celery broker URL",
)
```

**作用**：
- 接收生产者（API）发送的任务消息
- 将消息存储到队列中
- Worker 从 Broker 拉取任务执行

**常用 Broker 对比**：

| Broker | 优点 | 缺点 | 适用场景 |
|--------|------|------|----------|
| **Redis** | 配置简单、性能高、支持优先级队列 | 非持久化（默认）、消息可能丢失 | 中小型项目、开发环境 |
| **RabbitMQ** | 企业级、支持复杂路由、高可靠 | 配置复杂、资源占用高 | 生产环境、复杂业务 |
| **SQS** | 托管服务、无需运维 | 延迟较高、成本 | AWS 云环境 |

### 3. Worker（任务执行器）

**Docker 配置**：

```yaml
worker:
  build:
    context: .
    dockerfile: backend/docker/Dockerfile.celery
  command: >
    uv run celery -A smart_sales.celery_app worker
    --loglevel=info
    --concurrency=4  # 并发工作进程数
```

**Worker 参数说明**：

| 参数 | 说明 | 项目配置 |
|------|------|----------|
| `-A` / `--app` | 指定 Celery 应用模块 | `smart_sales.celery_app` |
| `--loglevel` | 日志级别 | `info` |
| `--concurrency` | 并发 Worker 进程数 | `4`（根据 CPU 核心数调整） |
| `-Q` | 指定监听的队列 | 默认监听所有队列 |
| `-n` | Worker 节点名称 | 用于分布式识别 |

### 4. Beat（定时任务调度器）

**Docker 配置**：

```yaml
beat:
  build:
    context: .
    dockerfile: backend/docker/Dockerfile.celery
  command: >
    uv run celery -A smart_sales.celery_app beat
    --loglevel=info
```

**特点**：
- Beat 本身**不执行任务**，只负责在指定时间触发任务
- 需要与 Worker 配合使用（Beat 发消息，Worker 执行）
- 支持 crontab 表达式和间隔调度

### 5. Task（任务）

任务是 Celery 的核心单元，使用 `@celery_app.task` 装饰器定义。

**基本结构**（来自 `ai_tasks.py`）：

```python
@celery_app.task(
    name="smart_sales.tasks.run_scoring",  # 任务唯一标识
    bind=True,                              # 绑定 self 参数
    max_retries=2,                          # 最大重试次数
    default_retry_delay=120,                # 重试间隔（秒）
)
def run_scoring(self, *, customer_batch_record_id: str) -> dict[str, Any]:
    """对指定客户批次记录执行意向度评分."""
    logger.info(f"run_scoring: customer_batch_record_id={customer_batch_record_id}")
    # 任务逻辑...
```

**装饰器参数详解**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | str | 任务全局唯一名称，用于调用和监控 |
| `bind` | bool | 是否绑定任务实例，允许使用 `self.retry()` |
| `max_retries` | int | 失败后的最大重试次数 |
| `default_retry_delay` | int | 默认重试间隔（秒） |
| `queue` | str | 指定任务进入的队列 |
| `time_limit` | int | 硬超时限制（秒） |
| `soft_time_limit` | int | 软超时限制（秒），触发 SoftTimeLimitExceeded |

### 6. Queue（队列）

**项目中的队列设计**（`schedules.py`）：

```python
CELERY_BEAT_SCHEDULE: dict = {
    "sync-chat-records": {
        "task": "smart_sales.tasks.sync_chat_records",
        "schedule": crontab(minute="*/5"),
        "options": {"queue": "sync"},  # 指定 sync 队列
    },
    "ai-scoring-hourly": {
        "task": "smart_sales.tasks.scan_all_pending_scoring",
        "schedule": crontab(minute=0, hour="*"),
        "options": {"queue": "ai"},    # 指定 ai 队列
    },
}
```

**队列策略**：

| 队列 | 用途 | 任务类型 |
|------|------|----------|
| `default` | 通用任务 | 提醒发送、无互动检测 |
| `sync` | 数据同步 | 聊天记录同步、客户信息同步 |
| `ai` | AI 计算 | 客户评分、画像更新 |

**多队列的优势**：
1. **资源隔离**：AI 任务（CPU 密集型）与同步任务（IO 密集型）分开
2. **优先级控制**：重要队列可分配更多 Worker
3. **故障隔离**：一个队列堵塞不影响其他队列

---

## 工作流程

### 1. 异步任务执行流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│   Broker    │────▶│   Worker    │────▶│   Backend   │
│  (API/Fast) │     │   (Redis)   │     │  (Process)  │     │   (Redis)   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                                              │              │
      │ 1. 调用 .delay() 或 .apply_async()           │              │
      │─────────────────────────────────────────────▶│              │
      │                                              │              │
      │                                              │ 2. 执行      │
      │                                              │─────────────▶│
      │                                              │              │
      │ 3. 返回 AsyncResult                          │              │
      │◀─────────────────────────────────────────────│              │
      │                                              │              │
      │ 4. 查询结果                                  │              │
      │◀────────────────────────────────────────────────────────────│
```

**代码示例**：

```python
# 方式1：简单调用（推荐）
run_scoring.delay(customer_batch_record_id="record_123")

# 方式2：高级调用（支持更多参数）
run_scoring.apply_async(
    kwargs={"customer_batch_record_id": "record_123"},
    queue="ai",
    countdown=60,  # 延迟 60 秒执行
    retry=True,
    retry_policy={
        'max_retries': 3,
        'interval_start': 0,
        'interval_step': 0.2,
        'interval_max': 0.2,
    }
)
```

### 2. 定时任务执行流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    Beat     │────▶│   Broker    │────▶│   Worker    │────▶│   Backend   │
│  (Scheduler)│     │   (Redis)   │     │  (Process)  │     │   (Redis)   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                                              │              │
      │ 1. 检查调度表（每分钟）                      │              │
      │─────────────────────────────────────────────▶│              │
      │                                              │              │
      │ 2. 到期任务投递                              │              │
      │─────────────────────────────────────────────▶│              │
      │                                              │              │
      │                                              │ 3. 执行      │
      │                                              │─────────────▶│
```

### 3. 任务状态流转

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  PENDING │───▶│ RECEIVED │───▶│  STARTED │───▶│  SUCCESS │───▶│  RETRY   │
│  (等待)  │    │ (已接收) │    │ (执行中) │    │ (成功)   │    │ (重试)   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └────┬─────┘
                                                                      │
                                                                      ▼
                                                                 ┌──────────┐
                                                                 │  FAILURE │
                                                                 │ (失败)   │
                                                                 └──────────┘
```

---

## 项目中的实践

### 1. 任务分类组织

**文件结构**：

```
backend/src/smart_sales/tasks/
├── __init__.py       # 包初始化，导出任务模块
├── schedules.py      # Beat 定时任务调度表
├── ai_tasks.py       # AI 相关任务（评分、画像）
├── sync_tasks.py     # 数据同步任务
└── remind_tasks.py   # 提醒调度任务
```

**任务分类原则**：
- **按业务域划分**：AI、同步、提醒各自独立
- **按执行特性划分**：CPU 密集型（AI）与 IO 密集型（同步）分开
- **按调度频率划分**：高频（5分钟）、中频（1小时）、低频（每日）

### 2. AI 任务实现（`ai_tasks.py`）

**核心特点**：

```python
# 在 Celery 任务中运行异步代码的特殊处理
_task_loop: asyncio.AbstractEventLoop | None = None

def _run_in_task_loop(coro: Coroutine[Any, Any, dict[str, Any]]) -> dict[str, Any]:
    """在 Celery Worker 中运行异步代码的辅助函数."""
    global _task_loop
    if _task_loop is None or _task_loop.is_closed():
        _task_loop = asyncio.new_event_loop()
    return _task_loop.run_until_complete(coro)

@celery_app.task(bind=True, max_retries=2, default_retry_delay=120)
def run_scoring(self, *, customer_batch_record_id: str) -> dict[str, Any]:
    async def _run() -> dict[str, Any]:
        agent = ScoringAgent()
        result = await agent.run(customer_batch_record_id)
        return result
    
    return _run_in_task_loop(_run())
```

**设计要点**：
1. **异步兼容**：Celery Worker 是同步的，需要手动管理事件循环
2. **分布式锁**：使用 Redis 锁避免并发重复评分
3. **链式调用**：评分后自动触发画像更新
4. **错误处理**：捕获异常并返回结构化结果

### 3. 定时任务调度表（`schedules.py`）

**调度策略设计**：

```python
CELERY_BEAT_SCHEDULE: dict = {
    # ══ 高频任务（每 5 分钟）══
    "sync-chat-records": {
        "task": "smart_sales.tasks.sync_chat_records",
        "schedule": crontab(minute="*/5"),
        "options": {"queue": "sync"},
    },
    "send-reminders": {
        "task": "smart_sales.tasks.send_reminders",
        "schedule": crontab(minute="*/5"),
        "options": {"queue": "default"},
    },
    
    # ══ 整点任务（每小时）══
    "sync-customer-info": {
        "task": "smart_sales.tasks.sync_customer_info",
        "schedule": crontab(minute=0, hour="*"),
        "options": {"queue": "sync"},
    },
    "ai-scoring-hourly": {
        "task": "smart_sales.tasks.scan_all_pending_scoring",
        "schedule": crontab(minute=0, hour="*"),
        "options": {"queue": "ai"},
    },
    
    # ══ 每日任务（00:00）══
    "batch-sync-history": {
        "task": "smart_sales.tasks.batch_sync_history",
        "schedule": crontab(minute=0, hour=0),
        "options": {"queue": "sync"},
    },
    "update-profiles-daily": {
        "task": "smart_sales.tasks.scan_all_pending_profiles",
        "schedule": crontab(minute=0, hour=0),
        "options": {"queue": "ai"},
    },
    
    # ══ 每周任务（周日 00:00）══
    "weekly-pool-transfer": {
        "task": "smart_sales.tasks.batch_sync_history",
        "schedule": crontab(minute=0, hour=0, day_of_week="sunday"),
        "kwargs": {"days": 7},  # 传递参数
        "options": {"queue": "sync"},
    },
}
```

**调度规则（crontab）语法**：

```python
from celery.schedules import crontab

# 每分钟执行
crontab(minute="*")

# 每 5 分钟执行
crontab(minute="*/5")

# 每小时的第 0 分钟执行
crontab(minute=0, hour="*")

# 每天 00:00 执行
crontab(minute=0, hour=0)

# 每周日 00:00 执行
crontab(minute=0, hour=0, day_of_week="sunday")

# 每月 1 号 00:00 执行
crontab(minute=0, hour=0, day_of_month=1)
```

### 4. 自动发现任务

**配置**（`celery_app.py`）：

```python
celery_app.autodiscover_tasks(
    [
        "smart_sales.tasks",      # 显式加载 tasks 包
        "smart_sales.perception", # 感知层可能包含任务
        "smart_sales.decision",   # 决策层可能包含任务
        "smart_sales.execution",  # 执行层可能包含任务
    ]
)
```

**自动发现机制**：
- Celery 会扫描指定包下的所有 `tasks.py` 文件
- 自动注册带有 `@shared_task` 或 `@celery_app.task` 装饰器的函数
- 无需手动导入每个任务函数

---

## 配置详解

### 1. 序列化配置

```python
celery_app.conf.update(
    task_serializer="json",        # 任务消息序列化格式
    accept_content=["json"],        # 接受的序列化格式
    result_serializer="json",       # 结果序列化格式
)
```

**可选序列化器**：
- `json`：人类可读，跨语言支持好（推荐）
- `pickle`：Python 专用，支持任意对象（安全风险）
- `msgpack`：二进制格式，性能更高
- `yaml`：人类可读，适合调试

### 2. 时区配置

```python
celery_app.conf.update(
    timezone="Asia/Shanghai",      # 本地时区
    enable_utc=True,                # 启用 UTC 时间戳
)
```

**注意**：
- 生产环境建议统一使用 UTC，前端转换本地时间
- Beat 调度会根据 `timezone` 配置进行本地时间调度

### 3. 任务行为配置

```python
celery_app.conf.update(
    task_track_started=True,        # 追踪任务开始状态（STARTED）
    task_acks_late=True,            # 任务完成后才确认（防丢消息）
    worker_prefetch_multiplier=1,   # 每个 Worker 预取任务数
)
```

**关键参数说明**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `task_track_started` | False | 启用后 Worker 会在开始执行时更新状态为 STARTED |
| `task_acks_late` | False | 启用后消息在任务完成后才确认，防止 Worker 崩溃导致任务丢失 |
| `worker_prefetch_multiplier` | 4 | 每个 Worker 预取的任务数，设为 1 实现公平调度 |

### 4. 结果过期配置

```python
celery_app.conf.update(
    result_expires=3600,  # 结果保存 1 小时后自动删除
)
```

**建议**：
- 对于大量任务，应设置合理的过期时间，避免 Redis 内存无限增长
- 如需长期保存结果，应持久化到数据库

---

## 最佳实践与注意事项

### ✅ 最佳实践

#### 1. 任务设计原则

```python
# ✅ 好的实践：任务幂等性
@celery_app.task(bind=True, max_retries=3)
def process_payment(self, order_id: str) -> dict:
    """处理订单支付（幂等设计）."""
    # 1. 检查是否已处理
    if is_order_processed(order_id):
        return {"status": "already_processed"}
    
    # 2. 获取分布式锁
    lock = acquire_lock(f"payment:{order_id}", timeout=300)
    if not lock:
        raise self.retry(countdown=10)  # 未获取到锁，稍后重试
    
    try:
        # 3. 执行业务逻辑
        result = do_payment(order_id)
        return result
    finally:
        release_lock(lock)
```

#### 2. 异步任务中的数据库操作

```python
# ✅ 好的实践：每个任务独立获取 Session
@celery_app.task(bind=True)
def update_customer(self, customer_id: str) -> dict:
    async def _run():
        session_factory = get_pg_session_factory()
        async with session_factory() as session:
            repo = CustomerRepository()
            customer = await repo.get(session, id=customer_id)
            # 处理逻辑...
            await session.commit()
    
    return _run_in_task_loop(_run())
```

#### 3. 任务链与工作流

```python
from celery import chain

# 定义任务链：下载 → 处理 → 通知
workflow = chain(
    download_file.s(url),
    process_image.s(),
    notify_user.s(user_id)
)

# 执行任务链
workflow.delay()
```

### ⚠️ 常见陷阱

#### 1. 不要在任务中传递 ORM 对象

```python
# ❌ 错误：传递 ORM 对象
customer = await get_customer(session, id=1)
process_customer.delay(customer)  # 无法序列化

# ✅ 正确：传递 ID
customer_id = 1
process_customer.delay(customer_id=customer_id)
```

#### 2. 注意循环导入

```python
# ❌ 错误：在 models 中导入 tasks
# models/customer.py
from smart_sales.tasks import notify_user  # 循环导入！

# ✅ 正确：延迟导入或在应用层调用
# api/customer.py
from smart_sales.tasks import notify_user  # 安全
```

#### 3. 避免在任务中阻塞过长时间

```python
# ❌ 错误：长时间阻塞
@celery_app.task
def bad_task():
    time.sleep(3600)  # 阻塞 1 小时，Worker 无法处理其他任务

# ✅ 正确：分段处理或使用更细粒度的任务
@celery_app.task
def good_task():
    for batch in get_batches():
        process_batch.delay(batch)  # 分发子任务
```

#### 4. 正确处理异步代码

```python
# ❌ 错误：直接在同步任务中 await
@celery_app.task
def bad_task():
    result = await async_function()  # SyntaxError

# ✅ 正确：使用 run_until_complete
@celery_app.task
def good_task():
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(async_function())
```

### 🔍 监控与调试

#### 1. Flower 监控面板

```bash
# 启动 Flower
uv run celery -A smart_sales.celery_app flower --port=5555

# 访问地址
http://localhost:5555/flower
```

**Flower 功能**：
- 实时查看任务状态
- 查看 Worker 节点状态
- 手动重试失败任务
- 任务速率统计
- 队列深度监控

#### 2. 日志监控

```python
# 在任务中添加详细日志
@celery_app.task(bind=True)
def my_task(self, data):
    logger.info(f"Task {self.request.id} started with data: {data}")
    
    try:
        result = process(data)
        logger.info(f"Task {self.request.id} completed successfully")
        return result
    except Exception as e:
        logger.error(f"Task {self.request.id} failed: {e}")
        raise
```

#### 3. 任务结果查询

```python
# 获取任务结果
result = my_task.delay(data)

# 阻塞等待结果（不推荐在生产环境使用）
result.get(timeout=10)

# 非阻塞检查状态
if result.ready():
    print(result.result)

# 获取任务状态
print(result.state)  # PENDING, STARTED, SUCCESS, FAILURE, RETRY
```

---

## 总结

### Celery 在 Smart Sales 中的价值

| 场景 | 不使用 Celery | 使用 Celery |
|------|--------------|-------------|
| AI 评分 | API 响应 10s+ | 立即返回，后台评分 |
| 数据同步 | 手动触发 | 定时自动执行 |
| 消息推送 | 同步发送，可能超时 | 异步发送，失败重试 |
| 批量处理 | 单线程处理慢 | 分布式并行处理 |

### 核心要点回顾

1. **Broker** 负责消息存储，**Worker** 负责执行，**Beat** 负责调度
2. 任务应该是**幂等的**（多次执行结果一致）
3. 使用**多队列**隔离不同特性的任务
4. 配置合理的**超时**和**重试**策略
5. 使用 **Flower** 监控任务执行情况
6. 注意**异步代码**在 Celery 中的特殊处理

---

## 参考资源

- [Celery 官方文档](https://docs.celeryq.dev/)
- [Redis 作为 Celery Broker](https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/redis.html)
- [Flower 监控工具](https://flower.readthedocs.io/)
- [项目 Celery 配置](backend/src/smart_sales/celery_app.py)
- [项目任务实现](backend/src/smart_sales/tasks/)
