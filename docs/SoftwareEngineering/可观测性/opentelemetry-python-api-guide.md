# OpenTelemetry Python API 完全指南：从概念到实践

## 引言

OpenTelemetry 是一个开源的可观测性标准和工具集，提供了一套统一的 API 来收集 distributed traces、metrics 和 logs。本文将深入介绍 OpenTelemetry Python SDK 的核心 API，从基础概念到具体用法，帮助你快速掌握如何在 Python 项目中使用 OpenTelemetry。

## 核心概念

### Trace（追踪）

**Trace** 是一个请求在分布式系统中的完整执行路径。它由多个 Span 组成，形成一个树状结构。

- **1 个 Trace = 1 个请求的完整生命周期**
- Trace ID：全局唯一标识符，32 位十六进制字符串
- 一个 Trace 包含多个 Span，形成父子关系

### Span（跨度）

**Span** 是 Trace 中的基本工作单元，代表一个具体的操作或步骤。

- **1 个 Span = 1 个具体的操作**
- Span ID：16 位十六进制字符串，在 Trace 内唯一
- Span 包含：
  - 操作名称
  - 开始和结束时间
  - 属性（Attributes）
  - 事件（Events）
  - 状态（Status）
  - 父 Span 引用

### Context（上下文）

**Context** 用于在进程内传播 Trace 和 Span 信息。

- 通过 `contextvars` 自动传播到异步任务
- 包含当前 Trace ID 和 Span ID
- 支持跨线程、跨异步任务传递

### Tracer（追踪器）

**Tracer** 是创建 Span 的工厂。

- 每个 Tracer 有一个名称（通常是模块名）
- 提供 `start_span` 和 `start_as_current_span` 方法
- 全局通过 `trace.get_tracer()` 获取

## Trace API

### 获取 Tracer

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)
```

**参数说明**：
- `__name__`：推荐使用当前模块名，便于识别 span 来源
- 也可以使用自定义名称，如 `tracer = trace.get_tracer("my-app")`

### 创建 Span

有两种创建方式：

#### 1. `start_as_current_span` - 推荐

自动设置当前上下文，span 结束后自动恢复父上下文：

```python
tracer = get_tracer(__name__)

# 使用 context manager（推荐）
with tracer.start_as_current_span("operation_name") as span:
    # 业务逻辑
    do_something()
    # span 结束时自动关闭
```

**优点**：
- 自动管理上下文
- 异常时仍能正确关闭 span
- 代码简洁

#### 2. `start_span` - 手动管理

需要手动设置上下文和结束 span：

```python
tracer = get_tracer(__name__)

span = tracer.start_span("operation_name")
try:
    with trace.use_span(span):
        # 业务逻辑
        do_something()
finally:
    span.end()  # 必须手动结束
```

**适用场景**：
- 需要在 span 外部管理上下文
- 需要精确控制 span 的生命周期

### 创建嵌套 Span

在父 span 内创建子 span：

```python
tracer = get_tracer(__name__)

with tracer.start_as_current_span("parent_operation") as parent_span:
    # 父 span 逻辑
    
    # 创建子 span
    with tracer.start_as_current_span("child_operation") as child_span:
        # 子 span 逻辑
        child_span.set_attribute("child.key", "child.value")
    
    # 继续父 span 逻辑
    parent_span.set_attribute("parent.key", "parent.value")
```

**实际案例**（来自 smart-sales 项目）：

```python
tracer = get_tracer(__name__)

with tracer.start_as_current_span("customer.handle_wecom_added") as span:
    span.set_attribute("wecom.external_userid", external_userid)
    
    # 子 span 1：客户匹配
    with tracer.start_as_current_span("customer.match_by_external_contact") as match_span:
        customer = await match_customer(...)
        match_span.set_attribute("customer.id", customer.id)
    
    # 子 span 2：解析配置
    with tracer.start_as_current_span("customer.resolve_touch_config") as config_span:
        config = await resolve_config(...)
        config_span.set_attribute("customer.has_touch_config", bool(config))
    
    # 子 span 3：发送欢迎语
    with tracer.start_as_current_span("customer.send_welcome") as welcome_span:
        await send_welcome(...)
        welcome_span.set_attribute("customer.has_welcome_text", True)
```

生成的 trace 结构：

```
customer.handle_wecom_added
├── customer.match_by_external_contact
├── customer.resolve_touch_config
└── customer.send_welcome
```

### 设置 Span 属性

使用 `set_attribute` 方法：

```python
with tracer.start_as_current_span("http.request") as span:
    span.set_attribute("http.method", "POST")
    span.set_attribute("http.url", "https://api.example.com/users")
    span.set_attribute("http.status_code", 200)
    span.set_attribute("user.id", 12345)
    span.set_attribute("request.size", 1024)
```

**支持的属性类型**：

| 类型 | 示例 |
|------|------|
| 字符串 | `"value"` |
| 整数 | `42` |
| 浮点数 | `3.14` |
| 布尔值 | `True`, `False` |
| 字符串数组 | `["a", "b", "c"]` |
| 整数数组 | `[1, 2, 3]` |
| 浮点数组 | `[1.1, 2.2]` |
| 布尔数组 | `[True, False]` |

**语义约定（Semantic Conventions）**：

OpenTelemetry 定义了标准的属性命名约定：

```python
# HTTP 相关
span.set_attribute("http.method", "GET")
span.set_attribute("http.url", "https://...")
span.set_attribute("http.status_code", 200)
span.set_attribute("http.response_content_length", 1024)

# 数据库相关
span.set_attribute("db.system", "postgresql")
span.set_attribute("db.statement", "SELECT * FROM users")
span.set_attribute("db.operation", "SELECT")

# 自定义业务属性
span.set_attribute("wecom.userid", "user123")
span.set_attribute("customer.id", 456)
span.set_attribute("order.total", 99.99)
```

### 添加 Span 事件

使用 `add_event` 记录 span 内的重要时刻：

```python
with tracer.start_as_current_span("long_operation") as span:
    span.add_event("开始处理", {"step": 1})
    
    process_step1()
    span.add_event("步骤 1 完成", {"duration_ms": 100})
    
    process_step2()
    span.add_event("步骤 2 完成", {"duration_ms": 200})
    
    span.add_event("处理完成")
```

**参数**：
- `name`：事件名称
- `attributes`：事件属性（可选）
- `timestamp`：时间戳（可选，默认为当前时间）

**用途**：
- 记录 span 内的关键节点
- 标记特定时刻发生的事件
- 辅助定位问题

## Status API

### 设置 Span 状态

使用 `Status` 和 `StatusCode` 表示操作结果：

```python
from opentelemetry.trace import Status, StatusCode

with tracer.start_as_current_span("operation") as span:
    try:
        result = do_something()
        span.set_status(Status(StatusCode.OK))  # 成功
    except Exception as exc:
        span.set_status(Status(StatusCode.ERROR, str(exc)))  # 失败
```

**StatusCode 枚举值**：

| 状态 | 含义 | 使用场景 |
|------|------|----------|
| `StatusCode.UNSET` | 默认状态 | span 未明确设置状态 |
| `StatusCode.OK` | 操作成功 | 操作正常完成 |
| `StatusCode.ERROR` | 操作失败 | 操作出现错误 |

**最佳实践**：

```python
from opentelemetry.trace import Status, StatusCode

tracer = get_tracer(__name__)

async def handle_request(request_id: str):
    """处理请求示例。"""
    with tracer.start_as_current_span("handle_request") as span:
        span.set_attribute("request.id", request_id)
        
        try:
            # 验证参数
            validate_request(request_id)
            
            # 处理业务
            result = await process_request(request_id)
            span.set_attribute("request.result_size", len(result))
            
            # 设置成功状态
            span.set_status(Status(StatusCode.OK))
            return result
            
        except ValidationError as exc:
            # 设置错误状态
            span.set_status(Status(StatusCode.ERROR, "参数验证失败"))
            span.record_exception(exc)
            raise
            
        except TimeoutError as exc:
            span.set_status(Status(StatusCode.ERROR, "请求超时"))
            span.record_exception(exc)
            raise
```

**实际案例**（来自 smart-sales webhook）：

```python
tracer = get_tracer(__name__)

async def receive_callback(request: Request, ...):
    """接收企微回调。"""
    with tracer.start_as_current_span("wecom.receive_callback") as root_span:
        try:
            # 解密消息
            xml_message = crypto.decrypt_message(...)
            
            # 解析事件
            event = parse_event(xml_message)
            
            # 成功状态
            root_span.set_status(Status(StatusCode.OK))
            return {"errcode": 0}
            
        except ValueError as exc:
            # 记录异常并设置错误状态
            root_span.record_exception(exc)
            root_span.set_status(Status(StatusCode.ERROR, str(exc)))
            logger.exception("解密失败", error_type=exc.__class__.__name__)
            raise HTTPException(status_code=403)
            
        except ET.ParseError as exc:
            root_span.record_exception(exc)
            root_span.set_status(Status(StatusCode.ERROR, str(exc)))
            logger.exception("XML 解析失败", error_type=exc.__class__.__name__)
            raise HTTPException(status_code=400)
```

## Exception API

### 记录异常

使用 `record_exception` 将异常信息附加到 span：

```python
from opentelemetry.trace import Status, StatusCode

with tracer.start_as_current_span("operation") as span:
    try:
        do_something()
    except Exception as exc:
        # 记录异常
        span.record_exception(exc)
        # 设置错误状态
        span.set_status(Status(StatusCode.ERROR, str(exc)))
        raise
```

**record_exception 参数**：

```python
span.record_exception(
    exception,              # 异常对象
    attributes={            # 附加属性（可选）
        "error.type": exc.__class__.__name__,
        "error.message": str(exc),
    },
    timestamp=None,         # 时间戳（可选）
    escaped=False           # 是否逃逸异常（可选）
)
```

**实际案例**（来自 smart-sales services）：

```python
tracer = get_tracer(__name__)

async def auto_tag_customer(external_userid: str, tag_names: list[str]):
    """自动为客户打标签。"""
    with tracer.start_as_current_span("customer.auto_tag") as tag_span:
        try:
            result = await wecom_api.tag_customer(external_userid, tag_names)
            tag_span.set_attribute("customer.tag_count", len(tag_names))
            tag_span.set_attribute("wecom.api_success", True)
            
        except WeComAPIError as exc:
            # 记录异常详情
            tag_span.record_exception(exc)
            tag_span.set_status(Status(StatusCode.ERROR, str(exc)))
            
            # 附加业务属性
            tag_span.set_attribute("wecom.api_error_code", exc.error_code)
            tag_span.set_attribute("wecom.api_error_msg", exc.message)
            
            logger.exception(
                "打标签失败",
                external_userid=external_userid,
                tag_names=tag_names,
                error_type=exc.__class__.__name__,
            )
            raise
```

## Context API

### 获取当前 Span

```python
from opentelemetry import trace

# 获取当前活跃的 span
current_span = trace.get_current_span()

# 判断是否有效
if current_span.get_span_context().is_valid:
    trace_id = format(current_span.get_span_context().trace_id, "032x")
    span_id = format(current_span.get_span_context().span_id, "016x")
    print(f"Trace ID: {trace_id}, Span ID: {span_id}")
else:
    print("No active span")
```

### 获取 Trace Context

```python
from opentelemetry import trace

def get_current_trace_context():
    """返回当前 trace/span 标识。"""
    span = trace.get_current_span()
    context = span.get_span_context()
    
    if not context.is_valid:
        return {"trace_id": None, "span_id": None}
    
    return {
        "trace_id": format(context.trace_id, "032x"),
        "span_id": format(context.span_id, "016x"),
    }

# 使用示例
context = get_current_trace_context()
print(f"Trace: {context['trace_id']}, Span: {context['span_id']}")
```

**实际应用**（来自 smart-sales logging）：

```python
def _add_trace_context(_logger, _name: str, event_dict: dict) -> dict:
    """将 trace context 注入日志。"""
    span = trace.get_current_span()
    context = span.get_span_context()
    
    if context.is_valid:
        trace_id = format(context.trace_id, "032x")
        span_id = format(context.span_id, "016x")
        
        event_dict.setdefault("trace_id", trace_id)
        event_dict.setdefault("span_id", span_id)
    
    return event_dict
```

日志输出示例：

```
[2026-04-10 10:30:45] [info] [smart_sales.customer.services] 
  (trace_id=abc123def456789 span_id=1234567890abcdef customer_id=456)
  客户匹配成功
```

### Context 传播

#### 同步代码

Context 在同步代码中自动传播：

```python
tracer = get_tracer(__name__)

def process_request():
    with tracer.start_as_current_span("process_request") as parent_span:
        # 父 span context 自动传播
        
        validate_data()  # 内部可以获取父 span
        save_to_db()     # 内部可以获取父 span
        send_notification()  # 内部可以获取父 span
```

#### 异步代码（asyncio）

通过 `contextvars` 自动传播到 async task：

```python
tracer = get_tracer(__name__)

async def handle_request():
    with tracer.start_as_current_span("handle_request") as parent_span:
        # 创建的 async task 会继承当前 context
        task1 = asyncio.create_task(process_data())
        task2 = asyncio.create_task(validate_data())
        
        await asyncio.gather(task1, task2)

async def process_data():
    # 可以获取到父 span
    current_span = trace.get_current_span()
    # trace_id 与父 span 相同
```

**实际案例**（来自 smart-sales signals）：

```python
def dispatch_signal_handler(userid: str, external_userid: str, ...):
    """分发 signal 到异步任务。"""
    tracer = get_tracer(__name__)
    
    with tracer.start_as_current_span("dispatch_signal") as parent_span:
        parent_span.set_attribute("wecom.userid", userid)
        
        # asyncio.create_task 会自动继承当前 trace context
        asyncio.create_task(
            handle_wecom_added(userid, external_userid, ...)
        )

async def handle_wecom_added(userid: str, external_userid: str, ...):
    """异步任务处理。"""
    tracer = get_tracer(__name__)
    
    # 当前 trace context 来自父任务
    with tracer.start_as_current_span("customer.handle_wecom_added") as span:
        span.set_attribute("wecom.userid", userid)
        
        # 创建的子 span 会形成正确的父子关系
        await process_customer(...)
```

#### 手动传播 Context

跨线程或特殊场景需要手动传播：

```python
import contextvars
from opentelemetry import trace

# 保存当前 context
current_context = contextvars.copy_context()

def worker_thread():
    # 在新线程中恢复 context
    current_context.run(do_work)

def do_work():
    # 可以获取到原来的 span
    span = trace.get_current_span()
    span.add_event("thread_work_start")
```

## TracerProvider API

### 初始化 TracerProvider

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# 创建资源（服务标识）
resource = Resource.create({
    "service.name": "my-service",
    "service.version": "1.0.0",
    "deployment.environment": "production",
})

# 创建 TracerProvider
provider = TracerProvider(resource=resource)

# 创建导出器
exporter = OTLPSpanExporter(
    endpoint="http://localhost:4318/v1/traces",
)

# 添加 Span 处理器
provider.add_span_processor(BatchSpanProcessor(exporter))

# 注册为全局 TracerProvider
trace.set_tracer_provider(provider)
```

**实际封装**（来自 smart-sales otel.py）：

```python
def init_otel() -> bool:
    """初始化 OpenTelemetry。"""
    settings = get_settings()
    
    # 创建资源
    resource = Resource.create({
        "service.name": settings.OTEL_SERVICE_NAME,
    })
    
    # 创建 TracerProvider
    provider = TracerProvider(resource=resource)
    
    # 创建导出器
    exporter = OTLPSpanExporter(
        endpoint=f"{settings.OTEL_EXPORTER_OTLP_ENDPOINT}/v1/traces",
    )
    
    # 添加批量处理器
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    
    # 注册全局
    trace.set_tracer_provider(provider)
    
    return True
```

### SpanProcessor 类型

| 处理器 | 说明 | 使用场景 |
|--------|------|----------|
| `SimpleSpanProcessor` | 每个 span 立即导出 | 开发调试 |
| `BatchSpanProcessor` | 批量导出（推荐） | 生产环境 |
| `MultiSpanProcessor` | 多处理器组合 | 多导出目标 |

**BatchSpanProcessor 参数**：

```python
BatchSpanProcessor(
    exporter,
    max_queue_size=2048,          # 队列最大容量
    schedule_delay_millis=5000,   # 导出间隔（毫秒）
    max_export_batch_size=512,    # 单次导出最大数量
    export_timeout_millis=30000,  # 导出超时（毫秒）
)
```

## Logs API

### 初始化 LoggerProvider

```python
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

# 创建 LoggerProvider
provider = LoggerProvider(
    resource=Resource.create({"service.name": "my-service"})
)

# 创建导出器
exporter = OTLPLogExporter(
    endpoint="http://localhost:4318/v1/logs",
)

# 添加处理器
provider.add_log_record_processor(BatchLogRecordProcessor(exporter))

# 注册全局
set_logger_provider(provider)

# 创建 LoggingHandler
handler = LoggingHandler(level=logging.INFO, logger_provider=provider)

# 添加到 Python logger
logger = logging.getLogger("my_app")
logger.addHandler(handler)
```

### 日志自动关联 Trace

```python
import logging
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler

# 初始化 OTel logs
provider = LoggerProvider(...)
set_logger_provider(provider)
handler = LoggingHandler(level=logging.INFO, logger_provider=provider)

# 添加到应用 logger
app_logger = logging.getLogger("smart_sales")
app_logger.addHandler(handler)

# 在 span 内记录日志
tracer = get_tracer(__name__)

with tracer.start_as_current_span("operation") as span:
    # 日志会自动携带 trace_id 和 span_id
    logger.info("操作开始")
    
    do_something()
    
    logger.info("操作完成")
```

**实际集成**（来自 smart-sales logging）：

```python
def setup_logging() -> None:
    """配置应用日志系统。"""
    # 创建标准日志 handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    # 获取 OTel log handler
    otel_handler = get_otel_log_handler()
    
    # 配置应用 logger
    app_logger = logging.getLogger("smart_sales")
    app_logger.addHandler(handler)  # 控制台输出
    if otel_handler:
        app_logger.addHandler(otel_handler)  # 导出到 SigNoz
    
    # 日志处理器自动注入 trace_id/span_id
    def _add_trace_context(_logger, _name, event_dict):
        span = trace.get_current_span()
        context = span.get_span_context()
        if context.is_valid:
            event_dict.setdefault("trace_id", format(context.trace_id, "032x"))
            event_dict.setdefault("span_id", format(context.span_id, "016x"))
        return event_dict
```

## Resource API

### 定义服务资源

```python
from opentelemetry.sdk.resources import Resource

resource = Resource.create({
    # 必需字段
    "service.name": "smart-sales-backend",
    
    # 可选字段
    "service.version": "1.0.0",
    "service.instance.id": "instance-123",
    "deployment.environment": "production",
    
    # 主机信息
    "host.name": "server-01",
    "host.type": "virtual_machine",
    
    # 进程信息
    "process.pid": 12345,
    "process.executable.name": "python",
    
    # 自定义属性
    "team.name": "backend",
    "region": "cn-east",
})
```

**语义约定**：

| 属性 | 说明 |
|------|------|
| `service.name` | 服务名称（必需） |
| `service.version` | 服务版本 |
| `service.instance.id` | 实例标识 |
| `deployment.environment` | 部署环境 |
| `host.name` | 主机名 |
| `process.pid` | 进程 ID |

## Exporter API

### OTLP HTTP Exporter

```python
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

# Span Exporter
span_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4318/v1/traces",
    timeout=10,  # 超时时间（秒）
)

# Log Exporter
log_exporter = OTLPLogExporter(
    endpoint="http://localhost:4318/v1/logs",
    timeout=10,
)
```

### OTLP gRPC Exporter

```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

exporter = OTLPSpanExporter(
    endpoint="localhost:4317",
    timeout=10,
)
```

### Console Exporter（开发调试）

```python
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

exporter = ConsoleSpanExporter()
provider.add_span_processor(SimpleSpanProcessor(exporter))
```

输出示例：

```json
{
    "name": "operation",
    "context": {
        "trace_id": "0xabc123def456789",
        "span_id": "0x1234567890abcdef",
    },
    "start_time": "2026-04-10T10:30:45Z",
    "end_time": "2026-04-10T10:30:46Z",
    "attributes": {
        "key": "value"
    },
    "status": {
        "status_code": "OK"
    }
}
```

## 最佳实践

### 1. Span 命名约定

使用统一的命名格式：

```python
# 推荐：领域.操作
tracer.start_as_current_span("customer.match")
tracer.start_as_current_span("order.create")
tracer.start_as_current_span("payment.process")

# 不推荐：过于笼统
tracer.start_as_current_span("process")
tracer.start_as_current_span("handle")
```

### 2. 属性设计原则

只记录定位问题需要的字段：

```python
# 推荐：关键业务字段
span.set_attribute("customer.id", customer_id)
span.set_attribute("order.status", "paid")
span.set_attribute("payment.amount", 99.99)

# 不推荐：大 payload 或敏感信息
span.set_attribute("request.body", huge_json)  # 太大
span.set_attribute("user.password", password)  # 敏感
```

### 3. 错误处理模式

统一的异常处理流程：

```python
with tracer.start_as_current_span("operation") as span:
    try:
        result = do_something()
        span.set_status(Status(StatusCode.OK))
        return result
    except Exception as exc:
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, str(exc)))
        logger.exception("操作失败", error_type=exc.__class__.__name__)
        raise
```

### 4. 异步任务传播

利用 Python contextvars 自动传播：

```python
async def handle_request():
    with tracer.start_as_current_span("request") as parent_span:
        # 自动传播到 async task
        tasks = [
            asyncio.create_task(process_step1()),
            asyncio.create_task(process_step2()),
        ]
        await asyncio.gather(*tasks)
```

### 5. 配置灵活降级

初始化失败不影响业务：

```python
def init_otel() -> bool:
    """初始化 OTel，失败时静默降级。"""
    try:
        provider = TracerProvider(...)
        exporter = OTLPSpanExporter(...)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        return True
    except Exception:
        logger.warning("OTel 初始化失败，将降级运行")
        return False

# 使用时检查
if init_otel():
    logger.info("OTel 已启用")
else:
    logger.info("OTel 未启用，继续运行")
```

### 6. 日志与 Trace 关联

自动注入 trace context：

```python
def _add_trace_context(_logger, _name, event_dict):
    span = trace.get_current_span()
    context = span.get_span_context()
    if context.is_valid:
        event_dict["trace_id"] = format(context.trace_id, "032x")
        event_dict["span_id"] = format(context.span_id, "016x")
    return event_dict

# 添加到 structlog 处理器链
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        _add_trace_context,  # 自动注入
        structlog.processors.JSONRenderer(),
    ],
)
```

### 7. 批量导出优化

生产环境使用 BatchSpanProcessor：

```python
# 推荐：批量导出
provider.add_span_processor(
    BatchSpanProcessor(
        exporter,
        max_queue_size=2048,
        schedule_delay_millis=5000,
    )
)

# 不推荐：立即导出（性能开销大）
provider.add_span_processor(SimpleSpanProcessor(exporter))
```

### 8. 避免过度追踪

只追踪关键链路：

```python
# 推荐：只追踪业务关键操作
tracer.start_as_current_span("customer.match")
tracer.start_as_current_span("payment.process")

# 不推荐：追踪所有函数
tracer.start_as_current_span("util.format_string")
tracer.start_as_current_span("helper.calculate")
```

## 完整示例

### Webhook 集成示例

```python
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from smart_sales.core.observability.otel import get_tracer

tracer = get_tracer(__name__)

async def receive_callback(request: Request, timestamp: str, nonce: str, msg_signature: str):
    """接收企微事件回调。"""
    crypto = _get_crypto()
    body = await request.body()
    
    # 创建根 span
    with tracer.start_as_current_span(
        "wecom.receive_callback",
        attributes={
            "wecom.timestamp": timestamp,
            "wecom.nonce": nonce,
            "wecom.msg_signature": msg_signature,
        },
    ) as root_span:
        try:
            # 解密子 span
            with tracer.start_as_current_span("wecom.decrypt_callback"):
                xml_message = crypto.decrypt_message(
                    encrypt=encrypt,
                    signature=msg_signature,
                    timestamp=timestamp,
                    nonce=nonce,
                )
            
            # 解析子 span
            with tracer.start_as_current_span("wecom.parse_callback_event") as parse_span:
                event = _parse_event(xml_message)
                parse_span.set_attribute("wecom.event_type", event.__class__.__name__)
                parse_span.set_attribute("wecom.userid", event.UserID)
            
            # 处理事件
            if isinstance(event, AddExternalContact):
                with tracer.start_as_current_span(
                    "wecom.handle_add_contact",
                    attributes={
                        "wecom.external_userid": event.ExternalUserID,
                        "wecom.state": event.State,
                    },
                ):
                    await handle_add_contact(event)
            
            # 成功状态
            root_span.set_status(Status(StatusCode.OK))
            return {"errcode": 0}
            
        except ValueError as exc:
            # 记录异常
            root_span.record_exception(exc)
            root_span.set_status(Status(StatusCode.ERROR, str(exc)))
            logger.exception("解密失败", error_type=exc.__class__.__name__)
            raise HTTPException(status_code=403)
```

### Service 层集成示例

```python
from opentelemetry.trace import Status, StatusCode
from smart_sales.core.observability.otel import get_tracer

tracer = get_tracer(__name__)

async def handle_customer_added(external_userid: str, sales_userid: str):
    """处理客户添加事件。"""
    with tracer.start_as_current_span(
        "customer.handle_wecom_added",
        attributes={
            "wecom.external_userid": external_userid,
            "wecom.sales_userid": sales_userid,
        },
    ) as parent_span:
        logger.info("开始处理客户添加", external_userid=external_userid)
        
        # 客户匹配
        with tracer.start_as_current_span(
            "customer.match_by_external_contact"
        ) as match_span:
            customer = await match_customer(external_userid)
            if customer:
                match_span.set_attribute("customer.id", customer.id)
                match_span.set_attribute("customer.matched", True)
            else:
                match_span.set_attribute("customer.matched", False)
        
        # 解析触达配置
        with tracer.start_as_current_span(
            "customer.resolve_touch_config"
        ) as config_span:
            config = await resolve_touch_config(customer.id)
            config_span.set_attribute("customer.has_touch_config", bool(config))
        
        # 自动打标签
        if config and config.auto_tags:
            with tracer.start_as_current_span("customer.auto_tag") as tag_span:
                try:
                    await auto_tag(external_userid, config.auto_tags)
                    tag_span.set_attribute("customer.tag_count", len(config.auto_tags))
                except Exception as exc:
                    tag_span.record_exception(exc)
                    tag_span.set_status(Status(StatusCode.ERROR, str(exc)))
                    logger.exception("打标签失败", error_type=exc.__class__.__name__)
        
        # 发送欢迎语
        if config and config.welcome_text:
            with tracer.start_as_current_span("customer.send_welcome") as welcome_span:
                await send_welcome(external_userid, config.welcome_text)
                welcome_span.set_attribute("customer.has_welcome_text", True)
        
        logger.info("客户添加处理完成", external_userid=external_userid)
        parent_span.set_status(Status(StatusCode.OK))
```

## 总结

OpenTelemetry Python SDK 提供了一套完整的 API 来构建可观测性能力：

**核心 API**：
- **Trace API**：`get_tracer()`, `start_as_current_span()`, `set_attribute()`, `add_event()`
- **Status API**：`Status`, `StatusCode`, `set_status()`
- **Exception API**：`record_exception()`
- **Context API**：`get_current_span()`, `get_span_context()`
- **TracerProvider API**：TracerProvider 初始化和配置
- **Logs API**：LoggerProvider, LoggingHandler
- **Resource API**：服务标识和元数据
- **Exporter API**：OTLP exporters

**关键设计原则**：
1. 使用 context manager 自动管理 span 生命周期
2. 统一的命名约定和语义属性
3. 完善的异常处理和状态设置
4. 利用 contextvars 自动传播上下文
5. 批量导出优化性能
6. 静默降级保障业务稳定性
7. 日志自动关联 trace

通过合理使用这些 API，可以快速在 Python 项目中建立起完整的可观测性体系，为故障定位、性能优化和业务监控提供强大支撑。

## 参考资料

- [OpenTelemetry Python SDK Documentation](https://opentelemetry-python.readthedocs.io/)
- [OpenTelemetry API Specification](https://opentelemetry.io/docs/specs/otel-api/)
- [OpenTelemetry Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/)
- [OTLP Specification](https://opentelemetry.io/docs/specs/otlp/)
- [Python ContextVars](https://docs.python.org/3/library/contextvars.html)