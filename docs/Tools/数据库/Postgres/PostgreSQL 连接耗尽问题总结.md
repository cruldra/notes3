### 背景

测试环境运行一段时间后，部署脚本和数据库初始化脚本会报错：

```text
FATAL: sorry, too many clients already
```

最初怀疑点有两个：

1. PostgreSQL `max_connections` 太小。
2. 应用侧数据库连接没有被正确释放。

后续排查证明，真正的问题在应用侧，且主要集中在 Celery Worker 的异步任务执行方式。

### 先前的历史问题

项目最初使用的是：

- 异步 PostgreSQL engine 进程级单例
- Celery 同步任务里通过 `asyncio.run(...)` 调用异步逻辑

这种组合会触发一个旧问题：

1. 第一次任务执行时，`asyncio.run(...)` 创建一个临时 event loop。
2. 进程级单例 `AsyncEngine` 绑定到这个 loop。
3. 任务结束后，这个 loop 被关闭。
4. 第二次任务执行又会创建一个新的 loop。
5. 但代码仍复用第一次那个 `AsyncEngine`。
6. 最终报错：

```text
event loop is closed
```

为绕过这个问题，后来把异步 PG 工厂改成了“按 loop 缓存”。

### 按 loop 缓存为什么又出问题

按 loop 缓存解决了“复用已关闭 loop 的 engine”问题，但引入了新的更严重问题：

1. Celery 任务每次执行都调用 `asyncio.run(...)`。
2. `asyncio.run(...)` 每次都会新建一个 loop。
3. 由于 engine 按 loop 缓存，每个新 loop 都会新建一套 `AsyncEngine` 和连接池。
4. 这些历史 loop 对应的连接池不会在 Worker 生命周期中及时清掉。
5. 任务跑久了以后，连接池会持续累积。
6. 最终 PostgreSQL 连接数被打满。

这就是当前测试环境里 `too many clients already` 的结构性原因。

### 真实验证过程

这次没有停留在推断层面，而是做了两类真实验证。

#### 1. 真实 PG 验证连接累积问题

新增测试文件：

- `backend/tests/test_pg_real_connection_lifecycle.py`

验证内容：

1. 使用真实 PostgreSQL DSN 建立连接。
2. 模拟 Celery 风格的反复执行。
3. 验证按 loop 缓存时，是否会累计多套连接。

结论：会。

#### 2. 真实 PG 验证最终方案是否可行

继续用真实 PostgreSQL 验证两件事：

1. 仅恢复进程级单例，但继续保留 `asyncio.run(...)`，会再次复现旧问题。
2. 进程级单例配合固定后台 loop，则可以稳定复用同一套 engine。

结论：

- `进程级单例 + 每次 asyncio.run(...)` 不可行。
- `进程级单例 + 固定后台 loop` 可行。

### 线上测试环境的直接证据

后续查看测试环境容器状态，确认：

1. Worker 实际上处于运行状态。
2. Beat 也在持续投递任务。
3. Worker 日志里直接出现：

```text
customer/pool/tasks.py
return asyncio.run(_expire_activation())
asyncpg.exceptions.TooManyConnectionsError: sorry, too many clients already
```

这与前面的真实验证完全吻合。

### 最终修复方案

最终采用的核心思路不是单独改一个点，而是成对修改：

### 方案核心

**进程级单例 async engine/session + 进程级固定后台 event loop runner**

具体包含两部分：

#### 1. 恢复异步 PG 工厂为进程级单例

在 `backend/src/smart_sales/core/db/__init__.py` 中恢复为：

- `_pg_engine`
- `_pg_session_factory`

不再按 loop 缓存。

#### 2. Celery 同步任务不再使用 `asyncio.run(...)`

新增：

- `backend/src/smart_sales/core/async_runner.py`

这个模块做的事很简单：

1. 在后台线程中启动一个常驻 event loop。
2. 提供 `run_async(coro)`。
3. Celery 的同步任务统一把协程提交到这个固定 loop 上执行。

这样就保证：

- 同一个 Worker 进程中始终只有一个异步 loop 被 Celery 任务使用。
- 同一个 Worker 进程中始终只复用一套 `AsyncEngine` 和连接池。

### 这次改动覆盖的代码

#### 新增文件

- `backend/src/smart_sales/core/async_runner.py`

#### 修改文件

- `backend/src/smart_sales/core/db/__init__.py`
- `backend/src/smart_sales/camp/tasks.py`
- `backend/src/smart_sales/customer/pool/tasks.py`
- `backend/src/smart_sales/customer/tasks.py`
- `backend/src/smart_sales/customer/profile/tasks.py`
- `backend/src/smart_sales/wecom/tasks.py`

#### 修改方式

把所有 Celery 任务入口从：

```python
return asyncio.run(coro())
```

统一改成：

```python
return run_async(coro())
```

### 为什么这个方案是正确的

这个方案同时避开了之前的两个坑：

#### 坑 1：进程级单例 + 每次 asyncio.run

会复用到已经绑定旧 loop 的 engine，触发 `event loop is closed`。

现在通过固定后台 loop，避免了这个问题。

#### 坑 2：按 loop 缓存 + 每次 asyncio.run

会不断创建新的 loop、新的 engine、新的 pool，最终把 PG 打满。

现在通过恢复进程级单例，避免了这个问题。

### 测试结果

关键测试已通过：

```bash
cd backend
uv run pytest tests/test_pg_session_factory_loop_isolation.py tests/test_pg_real_connection_lifecycle.py tests/test_profile_tasks.py tests/test_chat_archive_task.py tests/test_pool_tasks.py -q
```

结果：

```text
49 passed, 1 warning
```

真实 PG 验证也通过：

```bash
uv run pytest tests/test_pg_real_connection_lifecycle.py -q
```

结果：

```text
4 passed, 1 warning
```

### 当前仍需注意的点

这次修的是根因，但测试环境还建议继续做两件事：

#### 1. 下调连接池参数

当前默认值偏大：

- async pool: `pool_size=10`
- async overflow: `max_overflow=20`
- sync pool: `pool_size=5`
- sync overflow: `max_overflow=10`

对小机器测试环境来说不够保守。

建议下一步把这些值做成环境变量，并给测试环境更小的默认值。

#### 2. 给 engine 增加 `application_name`

这样以后在 `pg_stat_activity` 里可以快速区分：

- `api`
- `worker`
- `beat`

排查会更直接。

### 一句话结论

这次问题的核心不是单纯“PostgreSQL 连接数不够”，而是：

**Celery 同步任务使用 `asyncio.run(...)`，而异步 PG 工厂又按 loop 缓存，最终导致 Worker 持续累积新的连接池。**

最终正确方案是：

**进程级单例 async engine/session + 进程级固定后台 event loop runner。**
