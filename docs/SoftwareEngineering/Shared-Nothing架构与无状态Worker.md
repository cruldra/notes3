# Shared-Nothing 架构与无状态 Worker

> 单 worker 扩多 worker / 单进程扩多进程时的核心架构问题. 12-factor app "Processes" 原则的工程化展开. 适用于 FastAPI/Flask/Django 等 web 框架, Celery worker, 任何打算横向扩进程的服务.

---

## 一、核心命题: 区分误区与正解

一个常见的总结:

> "单 worker 改多 worker, 把进程内状态外移到 Redis 就行了."

**方向对, 但精确性不够, 会指导出错误设计.**

精确的提法:

> **让 worker 无状态化 (stateless worker / shared-nothing architecture).**
> Redis 只是外移的**手段之一**, 不是**目标本身**.

### 为什么这个区分重要

口号决定团队下意识的选择:

| 口号 | 团队会怎么做 |
|---|---|
| "外移到 Redis" | 不分青红皂白把状态塞 Redis. 订单状态机塞 Redis (数据一致性失控), 连接池也想塞 Redis (根本塞不进去), 业务感觉性能快了实际埋雷 |
| "让 worker 无状态" | 先问 "这个状态本来该归谁管?" 答案可能是 PG、Redis、Celery、或者直接消除 |

---

## 二、判断标准: 哪些状态该外移 vs 该保留

**判断准则**: 这个状态有没有 "在 A 请求里写、在 B 请求里读, 且 B 请求可能落到别的 worker"?

是 → 必须外移; 否 → 保留进程内是对的.

### 该保留进程内 (强行外移反而错)

| 状态 | 原因 |
|---|---|
| **PG 连接池** (`_pg_engine`) | TCP 连接对象本就是进程内的, 每 worker 各开一份是正确的; 只需调连接数预算避免撞 PG 上限 |
| **Redis 客户端** (`_redis_client`) | 同上, client 实例本身是进程级 |
| **HTTP SDK 客户端** (MinIO/OSS/billing client) | 每 worker 一份连接池合理 |
| **LRU 编译缓存** (e.g. AgentPool, compiled graph cache) | 缓存内容含闭包/不可序列化对象, 序列化代价 > 命中收益; 多 worker 重复占内存但不出错 |
| **`@lru_cache` 配置/secret** | 启动一次性 load, 每 worker 各持完全合理 |
| **ContextVar** | 协程上下文本来就是 task-local, 与 worker 无关 |
| **运维诊断的 in-memory 监控** | 每 worker 一份, 日志聚合时按 worker 分桶即可 |

### 必须外移 (跨请求/跨 worker 协同)

| 状态 | 典型例子 | 外移到哪儿 |
|---|---|---|
| **任务/会话取消注册表** | `cancel_registry: dict[session_id, asyncio.Task]` | Redis Pub/Sub channel |
| **分布式 Semaphore / 限流计数** | `global_semaphore = asyncio.Semaphore(N)` | Redis INCR + Lua TTL |
| **事件总线** | 进程内 EventBus | Redis Pub/Sub / Streams 或 Celery |
| **per-key 分布式锁** | `locks: dict[key, asyncio.Lock]` | Redis SETNX + EXPIRE / Redlock |
| **后台 polling 的 `asyncio.Event`** | "等部署完成" 语义跨 worker 必须共享 | PG 状态轮询 + 客户端 poll |

---

## 三、外移目标按场景选型 (不是只有 Redis)

不同状态类型该去不同存储:

| 状态类型 | 推荐 backing service | 例子 |
|---|---|---|
| **持久业务数据** | **PG / MySQL** | 用户、订单、会话历史 |
| **临时锁 / 计数器 / 限流 / 短 TTL 缓存** | **Redis** | cancel registry, rate limit, hot cache |
| **跨进程异步消息** | **Celery (Redis broker) / Kafka** | 事件总线、后台任务 |
| **大文件 / 二进制** | **S3 / MinIO** | 附件、模型权重 |
| **长期协调状态机** | **PG (with row locking / advisory lock)** | 部署进度、工作流引擎 |
| **配置 / Secret** | **配置中心 / Vault / 文件 + 启动 cache** | 不需要分布式协调, 各 worker 启动时 load |

### Redis 适合什么

- 高频读写 (~10万 QPS 起)
- 小数据 (单 key < 1MB)
- 接受最终一致
- 有自然 TTL (能过期就别永久保留)

### Redis 不适合什么

- 强事务一致性 (订单状态机)
- 关系查询、聚合
- 大对象存储
- 长期持久化 (除非配置 AOF + RDB 双备份, 仍弱于 PG)

**反模式**: 把 PG 该干的活硬塞 Redis. 性能确实快, 但数据一致性、复杂查询、备份恢复全部退化.

---

## 四、12-Factor App: 第六条 Processes

工业界对 stateless worker 的经典论述, [12factor.net VI. Processes](https://12factor.net/processes):

> The twelve-factor app is executed in the **execution environment as one or more processes**.
>
> Twelve-factor processes are **stateless** and **share-nothing**. Any data that needs to persist must be stored in a stateful backing service, typically a database.

核心三句话:

1. **Processes are stateless**: 进程内不保留请求间状态.
2. **Share-nothing**: 进程之间不共享内存.
3. **Persist via backing services**: 持久化交给数据库等 backing service.

这是 stateless worker 的"宪法". 后续所有的 Redis/PG/Kafka 选型都是它的工程化展开.

---

## 五、另一条路: Sticky Session (不是无状态化, 是状态局部化)

不是所有团队都有条件一上来做完整的无状态化改造. **过渡方案**: 通过 load balancer 让同一会话黏到同一 worker.

### 实现方式

- Nginx: `ip_hash` (按客户端 IP 哈希)
- Caddy: `lb_policy ip_hash` 或 `lb_policy cookie`
- HAProxy: `balance source` 或 `cookie SERVERID`
- K8s Ingress: `session-affinity: ClientIP`

### 对比表

| 维度 | Stateless + 外部存储 | Sticky Session |
|---|---|---|
| 改造成本 | **高** (改 3-5 个核心类, 补测试) | **极低** (改一行 LB 配置) |
| 横向扩展 | 任意扩 worker / 跨机房 | 受单 worker / 单节点容量限制 |
| 节点宕机 | 自动 failover, session 透明迁移 | 黏的会话丢失 |
| client IP 变化 | 不受影响 | 同一用户切网络会换 worker, 偶发状态丢失 |
| 适合阶段 | 中长期、规模化 | 短期、过渡、小规模 |
| 适合场景 | 任意请求 | 主要是 SSE/WebSocket 长连接相关流程 |

### 经验法则

- **不要直接跳到大重构**. 先 sticky session 把痛点压下去, 业务跑稳了再做无状态化.
- **sticky 不是终态**. 它解决了"同一会话同一 worker", 但**没解决全局协调** (比如全局限流、跨用户限流) — 这些仍需外部存储.
- **混合使用很常见**: SSE 黏住 worker + 限流走 Redis + 持久化走 PG. 现实生产配置往往是三者叠加.

---

## 六、实战改造流程

1. **审计 in-memory state**:
   - 模块级 `_singleton`、`_cache`、`_lock`、`_registry`
   - `app.state.X = ...` (FastAPI)、`g.X` (Flask)
   - `asyncio.Lock/Semaphore/Queue/Event`
   - `@lru_cache / @cache` 装饰器
   - `asyncio.create_task` 启动的后台任务
   - SSE/WebSocket 路由依赖的进程内 state

2. **分类**:
   - 保留 (process-local resource, 见 §2)
   - 必移 (cross-worker 业务状态)
   - 短期可暂留 (sticky session 能盖住的)

3. **选型**:
   - 持久 → PG
   - 高频小数据 → Redis
   - 跨进程消息 → Celery / Streams
   - 大文件 → S3/MinIO

4. **改造**:
   - 一次改一个状态类, 分别 PR
   - 每个类配对应的多 worker e2e 测试
   - 保留 backward-compatible 接口 (避免大爆炸)

5. **验证**:
   - 多 worker 启动 ≥ 2 实例
   - 跑场景测试 (cancel、限流、事件协同)
   - 压测对比 (单 worker baseline vs 多 worker)

---

## 七、反模式 (Anti-Patterns)

| 反模式 | 后果 |
|---|---|
| **"全外移到 Redis"一刀切** | 业务状态丢一致性, 连接池/SDK 客户端塞不进去, 设计错乱 |
| **用 Redis 当持久存储** (没配 AOF + RDB) | 进程重启或机器故障数据全丢 |
| **跳过 sticky session 直接做大重构** | 周期长、风险高、业务等不起 |
| **不识别 process-local resource** | 把 `_pg_engine` 也想"外移", 浪费时间 |
| **改造时不补测试** | 多 worker 下 race condition 是潜伏 bug, 单 worker 测试盖不到 |
| **多 worker 上线前没算 PG 连接预算** | `pool_size × N + 其他服务` 撞 PG `max_connections`, 上线即雪崩 |

---

## 八、案例: FastAPI 后端从 1 worker 扩到 N

典型审计清单 (基于 luffy-agent-platform 项目 2026-05-29 实际审计):

### P0 真 Bug — 加 worker 必修
- Session 取消注册表 (cancel registry): 多 worker 下 cancel 50% 失败
- 全局限流 (`ConcurrencyManager`): per-worker semaphore, 全局上限失真
- 进程内事件总线 (EventBus): 跨 worker listener 互不可见

### P1 偶发 Bug — 业务可能踩坑
- per-session 锁 map (`_locks: dict[id, asyncio.Lock]`): 跨 worker 并发可冲突
- 后台 polling 的 `asyncio.Event`: 等待"完成"语义跨 worker 失效

### P2 资源失真 — 不致 Bug 但要算账
- PG 连接池 × N worker: 总连接数撞 `max_connections` 上限
- 监控/诊断任务: 每 worker 一份, 日志重复
- 各种 SDK 客户端实例: 内存占用 × N

### P3 无影响
- ContextVar (协程上下文)
- 启动一次的配置 cache (`@lru_cache(get_settings)`)
- Celery 专用单例 (uvicorn worker 用不到)

---

## 九、一句话精炼

> **核心思想是让 worker 无状态化 (shared-nothing).**
>
> 跨请求/跨 worker 的**业务状态**必须外移到合适的 backing service:
> - 持久数据 → PG
> - 锁 / 限流 / 计数 → Redis
> - 事件协同 → Celery / Streams
>
> 但**进程本身的资源** (连接池、SDK 客户端、LRU 缓存) **该留进程内就留** — 它们不是跨 worker 业务状态, 强行外移反而错.
>
> 短期没条件做完无状态化时, **sticky session** 是合法的过渡方案, 不是 "二等公民".

---

## 延伸阅读

- [12-factor app: VI. Processes](https://12factor.net/processes)
- [Heroku: Concurrency](https://devcenter.heroku.com/articles/concurrency-and-database-connections) — 多进程 + DB 连接预算的工程实践
- 本笔记库相关:
  - [并发性能：观测、评估、优化](./软件测试/并发性能-观测评估与优化.md)
  - [压测术语速查](./软件测试/压测术语速查.md)
  - [Fan-out-Fan-in 并发模式](./Fan-out-Fan-in并发模式.md)
  - [Semaphore 信号量详解](./Semaphore信号量详解.md)
  - [PostgreSQL 连接耗尽问题总结](../Tools/数据库/Postgres/PostgreSQL%20连接耗尽问题总结.md)
  - [PgBouncer 连接池中间件](../Tools/数据库/Postgres/PgBouncer%20连接池中间件.md)
