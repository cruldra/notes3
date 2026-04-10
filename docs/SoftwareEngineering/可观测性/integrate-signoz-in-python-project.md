# 如何在 Python 项目中集成 SigNoz 可观测性

## 引言

在微服务架构中，可观测性（Observability）是保障系统稳定性的关键能力。传统日志虽然能记录业务流程，但存在两个明显问题：

1. **无法直接看到完整的调用链树状结构** - 只能按时间和字段手动拼接
2. **难以快速定位瓶颈和失败点** - 不知道哪一步最慢、失败发生在哪个阶段

SigNoz 是一个开源的可观测性平台，支持 OpenTelemetry 标准，能提供类似 Langfuse 的 trace 树状体验。本文将介绍如何在 Python FastAPI 项目中集成 SigNoz，实现 traces 和 logs 的导出。

## 方案选择

### 为什么选择 OpenTelemetry

在集成方案选择上，我们对比了三种方案：

| 方案 | 优点 | 缺点 |
|------|------|------|
| **方案 A：只强化结构化日志** | 改动最小 | 无法形成真正的 trace/span 树 |
| **方案 B：自研轻量 trace 上下文** | 少引入依赖 | 需要造轮子，SigNoz 无法识别为标准 trace |
| **方案 C：引入 OpenTelemetry（推荐）** | SigNoz 原生支持，日志可关联 trace | 需要新增少量依赖 |

我们选择方案 C，原因是：
- SigNoz 可以原生展示 trace 树
- 日志可以通过 `trace_id` / `span_id` 关联
- 覆盖范围明确，风险可控

## 设计原则

### 薄基础设施 + 业务显式 span

采用"薄基础设施层 + 业务链路显式 span"的设计：

- **基础设施层**：负责 OTel 初始化、导出器配置、tracer 获取和上下文读取
- **业务层**：在关键节点手动创建 span，记录业务属性

不引入复杂的 manager、adapter 或注册表，保持 tracing 代码直接、可读和易验证。

### 最小化覆盖范围

本期只覆盖扫码主链路，不做全站自动埋点：
- webhook 接收企微事件
- signal 异步任务处理
- service 业务逻辑（客户匹配、自动触达等）

不自动跟踪数据库、HTTP 客户端、Celery 任务，trace 粒度只到业务步骤。

### 静默降级策略

tracing 初始化或上报失败时：
- 只记录错误日志
- 不阻断应用启动
- `OTEL_ENABLED=false` 时业务行为与现状保持一致

## 实现步骤

### 1. 安装依赖

在 `pyproject.toml` 中添加 OpenTelemetry 相关依赖：

```toml
[project.dependencies]
opentelemetry-api = ">=1.20.0"
opentelemetry-sdk = ">=1.20.0"
opentelemetry-exporter-otlp = ">=1.20.0"
```

### 2. 配置环境变量

在 `.env.example` 中新增配置项：

```bash
# ========== OpenTelemetry ==========
OTEL_ENABLED=false
OTEL_SERVICE_NAME=smart-sales-backend
OTEL_EXPORTER_OTLP_ENDPOINT=http://192.168.1.4:4318
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
```

在 `Settings` schema 中定义配置字段：

```python
class Settings(BaseSettings):
    OTEL_ENABLED: bool = Field(
        default=False,
        description="是否启用 OpenTelemetry tracing",
    )
    OTEL_SERVICE_NAME: str = Field(
        default="smart-sales-backend",
        description="服务名称，用于标识 trace 来源",
    )
    OTEL_EXPORTER_OTLP_ENDPOINT: str = Field(
        default="",
        description="OTLP exporter 地址",
    )
    OTEL_EXPORTER_OTLP_PROTOCOL: str = Field(
        default="http/protobuf",
        description="OTLP 协议，支持 http/protobuf 和 grpc",
    )
```

### 3. 创建 OTel 最小封装

新增文件 `backend/src/smart_sales/core/observability/otel.py`：

```python
"""OpenTelemetry 最小封装。"""

from __future__ import annotations

from collections.abc import Mapping
import logging

from opentelemetry._logs import set_logger_provider
from opentelemetry import trace
from opentelemetry.trace import Tracer
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from smart_sales.core.setting import get_settings

_INITIALIZED = False
_OTEL_LOG_HANDLER: logging.Handler | None = None
_OTEL_LOGS_INITIALIZED = False


def _build_otlp_signal_endpoint(endpoint: str, signal_path: str) -> str:
    """把 OTLP 基础地址规范成指定信号的导出地址。"""
    normalized = endpoint.rstrip("/")
    normalized_signal_path = f"/v1/{signal_path}"
    if normalized.endswith(normalized_signal_path):
        return normalized
    return f"{normalized}{normalized_signal_path}"


def _build_resource(service_name: str) -> Resource:
    """构建 OTel 资源描述。"""
    return Resource.create({"service.name": service_name})


def _is_otel_config_enabled() -> bool:
    """判断当前配置是否允许初始化 OTel。"""
    settings = get_settings()
    if not settings.OTEL_ENABLED or not settings.OTEL_EXPORTER_OTLP_ENDPOINT:
        return False
    return settings.OTEL_EXPORTER_OTLP_PROTOCOL == "http/protobuf"


def init_otel() -> bool:
    """初始化 OpenTelemetry tracing。"""
    global _INITIALIZED

    settings = get_settings()
    if _INITIALIZED:
        return True
    if not _is_otel_config_enabled():
        return False

    try:
        provider = TracerProvider(
            resource=_build_resource(settings.OTEL_SERVICE_NAME),
        )
        exporter = OTLPSpanExporter(
            endpoint=_build_otlp_signal_endpoint(
                settings.OTEL_EXPORTER_OTLP_ENDPOINT, "traces"
            ),
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
    except Exception:
        return False

    _INITIALIZED = True
    return True


def get_tracer(name: str) -> Tracer:
    """返回 tracer。"""
    return trace.get_tracer(name)


def get_otel_log_handler() -> logging.Handler | None:
    """返回 OTel 日志导出 handler，不可用时返回 None。"""
    global _OTEL_LOGS_INITIALIZED
    global _OTEL_LOG_HANDLER

    if _OTEL_LOG_HANDLER is not None:
        return _OTEL_LOG_HANDLER
    if _OTEL_LOGS_INITIALIZED or not _is_otel_config_enabled():
        return None

    settings = get_settings()
    try:
        provider = LoggerProvider(
            resource=_build_resource(settings.OTEL_SERVICE_NAME)
        )
        exporter = OTLPLogExporter(
            endpoint=_build_otlp_signal_endpoint(
                settings.OTEL_EXPORTER_OTLP_ENDPOINT, "logs"
            ),
        )
        provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
        set_logger_provider(provider)
        _OTEL_LOG_HANDLER = LoggingHandler(
            level=logging.INFO, logger_provider=provider
        )
    except Exception:
        _OTEL_LOGS_INITIALIZED = True
        return None

    _OTEL_LOGS_INITIALIZED = True
    return _OTEL_LOG_HANDLER


def get_current_trace_context() -> Mapping[str, str | None]:
    """返回当前 trace/span 标识。"""
    span = trace.get_current_span()
    context = span.get_span_context()
    if not context.is_valid:
        return {"trace_id": None, "span_id": None}
    return {
        "trace_id": format(context.trace_id, "032x"),
        "span_id": format(context.span_id, "016x"),
    }
```

**关键设计点**：

1. **endpoint 规范化** - 自动将基础地址转换为 `/v1/traces` 和 `/v1/logs`
2. **配置判断** - 只有同时满足 enabled、endpoint、protocol 三个条件才初始化
3. **失败降级** - 异常时返回 False/None，不抛异常
4. **单例保证** - 使用全局变量防止重复初始化

### 4. 应用启动时初始化

在 `main.py` 的 lifespan 中初始化：

```python
@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期管理。"""
    settings = get_settings()
    
    setup_logging()
    logger = get_logger(__name__)
    
    try:
        otel_initialized = init_otel()
    except Exception as exc:
        logger.exception(
            "OpenTelemetry 初始化失败",
            error_type=exc.__class__.__name__
        )
    else:
        if otel_initialized:
            logger.info(
                "OpenTelemetry 初始化完成",
                service_name=settings.OTEL_SERVICE_NAME
            )
        else:
            logger.info("OpenTelemetry 未启用或初始化失败")
    
    logger.info("应用启动", app_name=settings.APP_NAME)
    setup_wecom_signal_handlers()
    
    yield
    
    logger.info("应用关闭", app_name=settings.APP_NAME)
    await close_all_connections()
```

### 5. 日志自动注入 trace_id/span_id

在 `logging/__init__.py` 中添加处理器：

```python
from smart_sales.core.observability.otel import (
    get_current_trace_context,
    get_otel_log_handler
)


def _add_trace_context(_logger, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """把当前 trace/span 标识注入日志上下文。"""
    trace_context = get_current_trace_context()
    if trace_context["trace_id"] is not None:
        event_dict.setdefault("trace_id", trace_context["trace_id"])
    if trace_context["span_id"] is not None:
        event_dict.setdefault("span_id", trace_context["span_id"])
    return event_dict


def _shared_processors() -> list[Any]:
    """返回 structlog 的公共处理器链。"""
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_trace_context,  # 新增的处理器
        structlog.processors.StackInfoRenderer(),
    ]


def setup_logging(*, stream: IO[str] | None = None) -> None:
    """配置应用日志系统。"""
    # ... 前面的处理器配置 ...
    
    handler = logging.StreamHandler(target_stream)
    handler.setFormatter(formatter)
    otel_handler = get_otel_log_handler()  # 获取 OTel 日志 handler
    
    app_logger = logging.getLogger(_APP_LOGGER_NAME)
    app_logger.handlers.clear()
    app_logger.addHandler(handler)
    if otel_handler is not None:
        app_logger.addHandler(otel_handler)  # 添加 OTel handler
    app_logger.setLevel(logging.INFO)
    app_logger.propagate = False
    
    # uvicorn logger 也添加 OTel handler
    for logger_name in ("uvicorn", "uvicorn.error"):
        stdlib_logger = logging.getLogger(logger_name)
        stdlib_logger.handlers.clear()
        stdlib_logger.addHandler(handler)
        if otel_handler is not None:
            stdlib_logger.addHandler(otel_handler)
        stdlib_logger.setLevel(logging.INFO)
        stdlib_logger.propagate = False
```

**关键点**：
- `_add_trace_context` 处理器自动从当前 span 读取 trace_id/span_id
- 日志同时输出到控制台和 SigNoz
- 保持 structlog 渲染链不变

### 6. 在业务代码中创建 span

#### webhook 层创建根 span

```python
from opentelemetry.trace import Status, StatusCode
from smart_sales.core.observability.otel import get_tracer


async def receive_callback(
    request: Request,
    timestamp: str,
    nonce: str,
    msg_signature: str,
    db: AsyncSession = Depends(get_db),
) -> PlainTextResponse:
    """接收企微事件回调。"""
    tracer = get_tracer(__name__)
    
    with tracer.start_as_current_span(
        "wecom.add_external_contact",
        attributes={
            "wecom.event_type": "callback",
            "wecom.timestamp": timestamp,
        },
    ) as root_span:
        try:
            # 解密消息
            with tracer.start_as_current_span("wecom.decrypt_callback"):
                crypto = _get_crypto()
                body = await request.body()
                xml_message = crypto.decrypt_message(...)
            
            # 解析事件
            with tracer.start_as_current_span(
                "wecom.parse_callback_event",
                attributes={
                    "wecom.userid": event.UserID,
                    "wecom.external_userid": event.ExternalUserID,
                    "wecom.state": event.State,
                },
            ):
                event = _adapt_to_wecom_event(data)
            
            # 分发信号
            if isinstance(event, AddExternalContact):
                with tracer.start_as_current_span(
                    "wecom.dispatch_friend_added_signal"
                ):
                    salesperson_student_wecom_friend_added_signal.send(...)
        
        except Exception as exc:
            root_span.set_status(Status(StatusCode.ERROR))
            root_span.record_exception(exc)
            raise
```

#### service 层创建子 span

```python
async def handle_wecom_added_event(
    userid: str,
    external_userid: str,
    welcome_code: str | None,
    state: str,
) -> None:
    """处理企微客户添加事件。"""
    tracer = get_tracer(__name__)
    
    with tracer.start_as_current_span("customer.handle_wecom_added"):
        # 客户匹配
        with tracer.start_as_current_span(
            "customer.match_by_external_contact",
            attributes={
                "wecom.userid": userid,
                "wecom.external_userid": external_userid,
            },
        ) as match_span:
            customer = await match_customer(...)
            match_span.set_attribute("customer.id", customer.id)
        
        # 解析触达配置
        with tracer.start_as_current_span(
            "customer.resolve_touch_config",
            attributes={"customer.qr_code_id": qr_code_id},
        ) as config_span:
            touch_config = await resolve_touch_config(...)
            config_span.set_attribute(
                "customer.has_touch_config",
                bool(touch_config)
            )
        
        # 自动打标签
        with tracer.start_as_current_span("customer.auto_tag") as tag_span:
            tags = await auto_tag_customer(...)
            tag_span.set_attribute("customer.tag_count", len(tags))
        
        # 发送欢迎语
        with tracer.start_as_current_span("customer.send_welcome") as welcome_span:
            await send_welcome_message(...)
            welcome_span.set_attribute(
                "customer.has_welcome_text",
                bool(welcome_text)
            )
```

### 7. 异步任务上下文传播

signal 通过 `asyncio.create_task` 启动后台任务时，Python 的 `contextvars` 机制会自动复制当前 trace 上下文：

```python
def dispatch_signal_handler(userid: str, external_userid: str, ...):
    """分发 signal 到异步任务。"""
    # 当前 trace context 会自动传播到 async task
    asyncio.create_task(
        handle_wecom_added_event(userid, external_userid, ...)
    )
```

不需要额外的 trace carrier 设计。

## Trace 设计

### Trace 粒度

- **1 条 trace = 1 次企微 AddExternalContact 事件处理尝试**

### Span 层级结构

```
wecom.add_external_contact (根 span)
├── wecom.decrypt_callback
├── wecom.parse_callback_event
├── wecom.dispatch_friend_added_signal
└── customer.handle_wecom_added
    ├── customer.match_by_external_contact
    ├── customer.resolve_touch_config
    ├── customer.auto_tag
    ├── customer.auto_remark
    ├── customer.send_welcome
    └── customer.update_lead_pool
```

### Span 属性设计

只记录定位问题需要的关键字段，不记录大 payload：

```python
# webhook 层属性
"wecom.event_type": "AddExternalContact"
"wecom.userid": userid
"wecom.external_userid": external_userid
"wecom.state": state
"wecom.has_welcome_code": bool(welcome_code)

# service 层属性
"customer.id": customer_id
"customer.qr_code_id": qr_code_id
"customer.has_touch_config": bool(touch_config)
"customer.tag_count": len(tags)
"customer.has_remark": bool(remark)
"customer.has_welcome_text": bool(welcome_text)
"lead_pool.updated_rows": updated_count
```

### 错误处理

```python
try:
    # 业务逻辑
    ...
except Exception as exc:
    span.set_status(Status(StatusCode.ERROR))
    span.record_exception(exc)
    logger.exception("业务异常", error_type=exc.__class__.__name__)
    raise
```

## 日志设计

### 保留的日志

保留有业务语义的日志：
- webhook 验签或解密失败
- 事件解析失败
- `state` 为空或找不到二维码
- 匹配成功时的 `customer_id`、`qr_code_id`
- 触达配置是否存在
- 自动打标签、备注、欢迎语被跳过的原因
- 外部系统调用失败及其关键参数

### 收缩的日志

删除与 span 重复的流水账日志：
- 单纯的"开始执行 X"
- 单纯的"执行完成 X"

### 日志字段增强

所有日志自动携带：
- `trace_id`：32 位十六进制字符串
- `span_id`：16 位十六进制字符串

## 验证方案

### 代码层验证

```python
def test_init_otel_returns_false_when_disabled(monkeypatch):
    """OTEL_ENABLED=false 时不应初始化。"""
    from smart_sales.core.observability.otel import init_otel
    
    monkeypatch.setattr(
        "smart_sales.core.observability.otel.get_settings",
        lambda: type("S", (), {
            "OTEL_ENABLED": False,
            "OTEL_SERVICE_NAME": "smart-sales-backend",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "",
            "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
        })(),
    )
    
    assert init_otel() is False
    assert get_current_trace_context() == {
        "trace_id": None,
        "span_id": None
    }


def test_init_otel_appends_traces_path_to_base_endpoint(monkeypatch):
    """endpoint 应自动追加 /v1/traces。"""
    # ... 测试 endpoint 规范化逻辑 ...
```

### 本地运行验证

1. 设置 `OTEL_ENABLED=true`
2. 启动应用，触发扫码主链路
3. 检查控制台日志是否出现 `trace_id` 和 `span_id`

### SigNoz 层验证

1. 在 SigNoz 中检索服务 `smart-sales-backend`
2. 找到 `wecom.add_external_contact` trace
3. 检查 span 树状结构是否完整
4. 检查错误步骤是否显示 error 状态
5. 通过 `Related Signals > Logs` 查看关联日志
6. 在日志中通过 `trace_id` 反查 trace

## 性能策略

为了控制开销：

1. **只覆盖关键链路** - 不做全站自动埋点
2. **使用批量导出器** - `BatchSpanProcessor` 和 `BatchLogRecordProcessor` 异步上报
3. **不记录大 payload** - 只记录最小必要的业务属性
4. **收缩重复日志** - 删除与 span 完全重复的流水账日志

预期额外开销是低个位数百分比级别，适合本地开发和测试环境。

## 部署 SigNoz

### 本地开发环境

推荐使用 Docker Compose 部署：

```yaml
version: '3.8'
services:
  signoz:
    image: signoz/signoz:latest
    ports:
      - "3301:3301"  # 前端
      - "4318:4318"  # OTLP HTTP receiver
      - "4317:4317"  # OTLP gRPC receiver
    environment:
      - CLICKHOUSE_URL=clickhouse://clickhouse:9000
```

### 配置应用连接

```bash
OTEL_ENABLED=true
OTEL_SERVICE_NAME=smart-sales-backend
OTEL_EXPORTER_OTLP_ENDPOINT=http://192.168.1.4:4318
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
```

## 总结

通过本次集成，我们实现了：

1. **完整的调用链可视化** - SigNoz 中能看到树状 trace 结构
2. **日志与 trace 关联** - 通过 `trace_id` 和 `span_id` 从日志反查 trace
3. **瓶颈定位能力** - 每个步骤的耗时、状态、异常一目了然
4. **开发友好** - 配置灵活，失败降级，不阻断业务
5. **架构简洁** - 薄基础设施层，业务代码显式控制 span

关键设计原则：

- **最小化** - 只覆盖关键链路，不做过度抽象
- **实用性** - 优先解决实际问题，避免过早优化
- **稳定性** - 失败降级，不影响业务功能
- **可维护性** - 代码直接可读，无复杂注册机制

这种集成方式适合中小型项目快速建立可观测性能力，为后续扩展到更多链路、接入告警、建设 Dashboard 奠定基础。

## 参考资料

- [OpenTelemetry Python SDK](https://opentelemetry.io/docs/instrumentation/python/)
- [SigNoz Documentation](https://signoz.io/docs/)
- [OTLP Specification](https://opentelemetry.io/docs/specs/otlp/)
- [Python ContextVars](https://docs.python.org/3/library/contextvars.html)