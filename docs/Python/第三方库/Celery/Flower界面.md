# Flower 界面说明

Flower 是一个用于 Celery 任务队列的实时监控和管理工具。

## Dashboard (仪表台)

登录成功后，默认进入 **Workers** 页面，这里展示了所有已注册的 Worker 节点及其运行状态。

### 界面概览

![image](https://github.com/cruldra/picx-images-hosting/raw/master/image.mm81gztt.png)

### 1. 顶部导航菜单

| 菜单项 | 说明 |
| :--- | :--- |
| **Workers** | 查看所有 Worker 节点及其统计信息（当前页面）。 |
| **Tasks** | 查看任务历史记录、当前执行中的任务详情。 |
| **Broker** | 查看消息中间件（如 Redis/RabbitMQ）的状态和队列长度。 |
| **Documentation** | 跳转至 Flower 官方文档。 |

### 2. Worker 统计列表

表格展示了每个 Worker 节点的实时数据：

| 列名 | 说明 |
| :--- | :--- |
| **Worker** | Worker 节点的名称。点击名称可进入该 Worker 的详情页。 |
| **Status** | 当前状态（`Online` 表示在线运行中，`Offline` 表示已掉线）。 |
| **Active** | 当前正在执行的任务数量。 |
| **Processed** | 该节点自启动以来处理过的总任务数。 |
| **Failed** | 处理失败的任务数。 |
| **Succeeded** | 处理成功的任务数。 |
| **Retried** | 重试过的任务数。 |
| **Load Average** | 节点的系统负载情况（分别对应 1、5、15 分钟）。 |

### 3. 操作与筛选

- **Show [N] workers**: 设置每页显示的 Worker 数量。
- **Search**: 按名称快速搜索特定的 Worker。
- **Total 行**: 统计所有节点汇总的任务数据。

### 操作提示
- 点击 Worker 名称可以进入详细页面进行更多管理操作（如：查看池大小、自动扩缩容配置、甚至是远程关闭 Worker）。

## Worker 详情页

在 Worker 列表中点击特定的 Worker 名称（如 `celery@3562b4d064ad`）即可进入其详情页面。

### 界面概览

![image](https://github.com/cruldra/picx-images-hosting/raw/master/image.7ehbgt5arw.png)

### 1. 详情页签 (Tabs)
页面顶部提供了该 Worker 的多个维度管理页签：
- **Pool**: 查看和控制 Worker 池的运行状态（默认展示项）。
- **Broker**: 该 Worker 连接的消息中间件信息。
- **Queues**: 该 Worker 当前监听的队列列表。
- **Tasks**: 该 Worker 处理过的任务历史。
- **Limits**: 查看和设置速率限制（Rate Limits）。
- **Config**: 查看该 Worker 的实时配置参数。
- **System**: 查看操作系统层面的详细统计信息（负载、内存、CPU 等）。
- **Other**: 其它杂项信息。

### 2. Worker 池选项 (Worker pool options)
展示了 Worker 内部池的配置与运行状态：
- **Implementation**: 并发池的实现方式（如 `prefork`, `eventlet` 等）。
- **Max concurrency**: 最大并发数。
- **Processes**: 当前运行的子进程 PID 列表。
- **Worker PID**: Worker 主进程的 PID。
- **Prefetch Count**: 预取数量（Worker 一次从队列中取出的任务数）。

### 3. 池大小控制 (Pool size control)
支持在不重启服务的情况下，动态调整 Worker 的并发能力：
- **Pool size**: 输入数值并点击 **Grow** (增加) 或 **Shrink** (减少) 来实时改变并发进程数。
- **Auto scale**: 输入 `Min` 和 `Max` 值并点击 **Apply**，配置该 Worker 的自动扩缩容策略。

### 4. 远程控制操作 (Refresh 下拉菜单)
右上角的绿色按钮提供了对 Worker 的直接控制命令：
- **Shut Down**: 远程关闭该 Worker 进程。
- **Restart Pool**: 重启该 Worker 的进程池（不关闭主进程）。
- **Refresh / Refresh All**: 手动刷新当前或所有 Worker 的实时统计数据。

### 5. Broker 详情 (Broker options)

在详情页切换至 **Broker** 页签，可以查看该 Worker 连接的消息中间件配置：

![image](https://github.com/cruldra/picx-images-hosting/raw/master/image.2yywbjyas8.png)

该表格列出了 Worker 与消息中间件之间的连接参数：
- **Hostname / Port**: 中间件的主机地址和端口（如 `192.168.1.4:6379`）。
- **Virtual host**: 使用的虚拟主机（Redis 中通常对应数据库索引，如 `1`）。
- **Transport**: 传输协议类型（如 `redis`, `amqp` 等）。
- **Heartbeat**: 心跳间隔时间（秒），用于维持连接活性。
- **SSL**: 是否启用了加密传输。
- **Connect timeout**: 连接超时设置（秒）。
- **Failover strategy**: 故障转移策略（如 `round-robin`）。
- **Userid**: 登录使用的用户名（若有）。

### 6. Queues (队列详情)

展示该 Worker 当前监听的所有队列信息：

![image](https://github.com/cruldra/picx-images-hosting/raw/master/image.4clffljlww.png)

- **Active queues**: 列出了当前正在监听的队列（如 `celery`）。包含是否持久化 (Durable)、路由键 (Routing key)、自动删除 (Auto delete) 等参数。
- **Cancel Consumer**: 点击可停止 Worker 对该队列的监听。
- **New consumer**: 支持手动输入队列名称并点击 **Add**，让该 Worker 开始监听新队列。

### 7. Tasks (任务统计)
该页签分类展示了不同状态的任务：

![image](https://github.com/cruldra/picx-images-hosting/raw/master/image.et1yx92wj.png)

- **Processed tasks**: 已处理完成的任务列表。
- **Active tasks**: 当前正在执行的任务。
- **Scheduled / Reserved tasks**: 调度中或已预留（等待执行）的任务。
- **Revoked tasks**: 已撤销的任务。
- **字段说明**: 包括任务名称、UUID、确认状态 (Ack)、执行进程 (PID) 以及参数 (args/kwargs)。

### 8. Limits (任务限流)
支持针对具体任务类型设置频率限制和超时时间：

![image](https://github.com/cruldra/picx-images-hosting/raw/master/image.2rvog4n21j.png)

- **Rate limit**: 设置每秒/分钟处理任务的最大频率（例如 `10/m`），点击 **Apply** 生效。
- **Timeouts**: 设置任务的 **Soft**（软超时）和 **Hard**（硬超时）限制。到达软超时可捕获异常处理，硬超时则直接终止任务进程。

### 9. Config (配置信息)

![image](https://github.com/cruldra/picx-images-hosting/raw/master/image.2rvog4n8oc.png)

列出了该 Worker 启动时的完整配置项，包括：
- **broker_url / result_backend**: 消息中间件和结果存储的连接地址。
- **beat_schedule**: 如果该 Worker 启用了周期性任务（Beat），这里会展示完整的调度计划。
- **timezone**: 使用的时区配置。
- **task_serializer**: 任务序列化格式（如 `json`）。
- **worker_prefetch_multiplier**: 预取乘数。

### 10. System (系统状态)

![image](https://github.com/cruldra/picx-images-hosting/raw/master/image.9gx44vkjta.png)

提供 Worker 进程的底层系统资源消耗统计：
- **utime / stime**: 用户态和内核态的 CPU 时间。
- **maxrss**: 进程占用的最大常驻内存大小。
- **nvcsw / nivcsw**: 自愿与非自愿的上下文切换次数。
- **inblock / oublock**: 磁盘输入/输出块操作数。