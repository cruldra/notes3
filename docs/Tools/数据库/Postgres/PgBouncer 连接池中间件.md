### PgBouncer 是什么

PgBouncer 是 PostgreSQL 前面的一层**连接池代理**。应用连 PgBouncer，PgBouncer 连 PG。它把"应用持有的连接数"和"PG 实际占用的连接数"解耦。

为什么需要：PG 每个连接对应一个 backend 进程，吃 5-15 MB 内存。应用层池子（SQLAlchemy / HikariCP 等）里有大量 `open but idle` 的连接，浪费 PG 资源。多进程（多容器、prefork worker、多副本）各自维护一套池，叠加之下连接数会爆掉 `max_connections`。

### 不加 vs 加的对比

```
不加 PgBouncer:
┌─api┐ ┌worker×4┐ ┌beat┐ ┌billing┐
│池30│ │池×4=120│ │池30│ │池30   │   每个进程独立池
└─┬──┘ └──┬─────┘ └─┬──┘ └─┬─────┘
  └───────┴─────────┴──────┘
                 │  最多 ~210 个 TCP 连接
                 ▼
          ┌──────────┐
          │PG max=100│  ← 撞天花板
          └──────────┘

加 PgBouncer (transaction mode):
┌─api┐ ┌worker×4┐ ┌beat┐ ┌billing┐
│池1k│ │池×4=4k │ │池1k│ │池1k   │   应用层池子可以巨大
└─┬──┘ └──┬─────┘ └─┬──┘ └─┬─────┘
  └───────┴─────────┴──────┘
                 │  几千个轻量 TCP
                 ▼
          ┌──────────┐
          │PgBouncer │   到 PG 的小池, 跨连接复用
          │ 池 20-50 │
          └────┬─────┘
               │  20-50 个真实 PG 连接
               ▼
          ┌──────────┐
          │PG max=100│  ← 用不到一半
          └──────────┘
```

关键机制：**transaction pooling** —— 应用 `BEGIN` 时 PgBouncer 临时借出一个 PG 连接，`COMMIT` / `ROLLBACK` 时立刻归还。100 个 idle 应用连接 + 5 个并发活跃事务，PG 端只需要 5 个真实连接。

### 三种 pooling 模式

| Mode | 何时归还 PG 连接 | 复用力 | 限制 |
|---|---|---|---|
| **session** | 应用 disconnect 时 | 最弱 | 没限制，但等于没复用 |
| **transaction**（最常用） | 事务结束（COMMIT / ROLLBACK） | 强 | **不支持** prepared statement（自动模式除外）、`SET`（除 `SET LOCAL`）、`LISTEN` / `NOTIFY`、事务外的 advisory lock、临时表、`WITH HOLD` cursor |
| **statement** | 每条 SQL 结束 | 最强 | 不支持事务，基本不实用 |

绝大多数生产环境用 **transaction mode**。

### 决策树：什么时候该上

| 触发条件 | 是否该上 |
|---|---|
| 应用进程数 × pool_size > PG max_connections × 2 | **该上**（过设计，撑不住） |
| 多容器叠 prefork 子进程（worker × N） | **该上** |
| 应用用 prepared statement / `LISTEN-NOTIFY` / 事务外 advisory lock | 谨慎上 transaction mode，先小流量灰度 |
| 单节点测试机 / 内部工具 | 不必上，over-engineering |
| 即将做万级并发 | **必须上** |

### 最小可用配置

`pgbouncer.ini`：

```ini
[databases]
mydb = host=postgres port=5432 dbname=mydb

[pgbouncer]
listen_addr = 0.0.0.0
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt

pool_mode = transaction
max_client_conn = 1000        ; 应用侧最多 1000 个连接
default_pool_size = 25        ; 每个 (user, db) 到 PG 的真实连接数
reserve_pool_size = 5         ; 紧急储备
reserve_pool_timeout = 3      ; 储备连接被占用 3s 后才启用

server_idle_timeout = 600     ; 真实 PG 连接 idle 10 分钟后关闭
server_lifetime = 3600        ; 真实 PG 连接最长存活 1 小时
```

`userlist.txt`（生成 md5 hash）：

```text
"appuser" "md5<md5(passwordappuser)>"
```

Docker 跑：

```yaml
pgbouncer:
  image: edoburu/pgbouncer:latest
  environment:
    DB_HOST: postgres
    DB_NAME: mydb
    DB_USER: appuser
    DB_PASSWORD: ${DB_PASSWORD}
    POOL_MODE: transaction
    MAX_CLIENT_CONN: "1000"
    DEFAULT_POOL_SIZE: "25"
  ports:
    - "6432:6432"
```

应用侧只需把 DSN 端口从 `5432` 改成 `6432`，其他不变。

### Python / SQLAlchemy + asyncpg + transaction mode 的坑

asyncpg 默认会用 prepared statement（每个 SQL 都自动 prepare），跟 transaction mode 冲突。SQLAlchemy 2.x 的 asyncpg 方言要显式关闭：

```python
engine = create_async_engine(
    POSTGRES_DSN,
    pool_pre_ping=True,
    # asyncpg 关闭自动 prepare, 兼容 PgBouncer transaction mode
    connect_args={
        "statement_cache_size": 0,
        "prepared_statement_cache_size": 0,
    },
)
```

不关掉会报：

```text
asyncpg.exceptions.DuplicatePreparedStatementError:
prepared statement "__asyncpg_stmt_1__" already exists
```

或者 PG 14+ 走 [PgBouncer 1.21+ 的自动 prepared statement 支持](https://www.pgbouncer.org/config.html#max_prepared_statements)（`max_prepared_statements = 100`），可以保留 prepare 但要确认 PgBouncer 版本。

### PgBouncer 的可观测性

进入 PgBouncer 自己的管理库 `pgbouncer`：

```bash
psql -h pgbouncer-host -p 6432 -U pgbouncer pgbouncer
```

关键命令：

```sql
SHOW POOLS;       -- 每个 (db, user) 池的 cl_active/cl_waiting/sv_active/sv_idle
SHOW CLIENTS;     -- 当前应用连接
SHOW SERVERS;     -- 当前到 PG 的真实连接
SHOW STATS;       -- 累计查询数 / 平均等待时间 / 平均事务耗时
SHOW CONFIG;      -- 当前生效配置
RELOAD;           -- 不重启情况下重载配置
```

排查首选 `SHOW POOLS`：

| 字段 | 含义 | 异常信号 |
|---|---|---|
| `cl_active` | 活跃的应用连接 | 跟应用预期一致即可 |
| `cl_waiting` | 等待 PG 连接的应用连接 | **> 0 持续存在 = 池太小** |
| `sv_active` | 借出去的真实 PG 连接 | 接近 `default_pool_size` 时考虑扩 |
| `sv_idle` | 空闲的真实 PG 连接 | 长期 0 = 复用率很低 |
| `maxwait` | 最长等待时间（秒） | **> 1s = 严重瓶颈** |

### 与「PostgreSQL 连接耗尽问题总结」的关系

那篇笔记里讲的问题（asyncio.run 一次性 loop 把 asyncpg connection 绑死 → idle in transaction 累积 → too many clients）**不是 PgBouncer 能直接解决的**——根因在应用层的事务没正常结束，PgBouncer 即使在前面也只会把卡死的事务一直占着不还。

PgBouncer 解决的是另一类问题：

| 问题 | 该用哪个方案 |
|---|---|
| 应用进程 × pool 数远超 PG max_connections | **PgBouncer** |
| 应用层事务漏 commit/rollback 累积 idle_in_transaction | **修代码 + PG 端 `idle_in_transaction_session_timeout` 兜底**，PgBouncer 帮不上 |
| 短连接频繁建立断开（如脚本任务）开销大 | **PgBouncer** session/transaction mode 都能复用底层连接 |
| 多租户 SaaS 想限流单租户连接数 | **PgBouncer** 的 per-db / per-user pool_size |

### 何时不要用

- 应用强依赖 prepared statement 又不想改：先试 PgBouncer 1.21+ 的 max_prepared_statements，不行就维持单纯加大 PG `max_connections`
- 用 `LISTEN/NOTIFY` 做事件驱动：transaction mode 会破坏长连接语义
- 团队没人懂 PgBouncer 的运维：多一个中间件就多一个故障点
- 单节点小机器（< 8 GB / < 4 核）：直接加大 max_connections 到 200 更简单

### 一句话结论

**应用层连接数撑不下 PG max_connections 时，PgBouncer transaction mode 是首选解药；但应用层自己漏连接，PgBouncer 救不了，先修代码。**
