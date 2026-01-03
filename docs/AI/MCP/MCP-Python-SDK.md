# MCP Python SDK

**模型上下文协议(MCP)的Python实现** [![PyPI][pypi-badge]][pypi-url] [![MIT licensed][mit-badge]][mit-url] [![Python Version][python-badge]][python-url] [![Documentation][docs-badge]][docs-url] [![Specification][spec-badge]][spec-url] [![GitHub Discussions][discussions-badge]][discussions-url]

## 目录
- 概述
- 安装
- 将MCP添加到你的Python项目
- 运行独立的MCP开发工具
- 快速入门
- 什么是MCP？
- 核心概念
- 服务器
- 资源
- 工具
- 提示
- 图像
- 上下文
- 运行你的服务器
- 开发模式
- Claude Desktop集成
- 直接执行
- 挂载到现有的ASGI服务器
- 示例
- 回显服务器
- SQLite浏览器
- 高级用法
- 低级服务器
- 编写MCP客户端
- MCP原语
- 服务器能力
- 文档
- 贡献
- 许可证

[pypi-badge]: https://img.shields.io/pypi/v/mcp.svg
[pypi-url]: https://pypi.org/project/mcp/
[mit-badge]: https://img.shields.io/pypi/l/mcp.svg
[mit-url]: https://github.com/modelcontextprotocol/python-sdk/blob/main/LICENSE
[python-badge]: https://img.shields.io/pypi/pyversions/mcp.svg
[python-url]: https://www.python.org/downloads/
[docs-badge]: https://img.shields.io/badge/docs-modelcontextprotocol.io-blue.svg
[docs-url]: https://modelcontextprotocol.io
[spec-badge]: https://img.shields.io/badge/spec-spec.modelcontextprotocol.io-blue.svg
[spec-url]: https://spec.modelcontextprotocol.io
[discussions-badge]: https://img.shields.io/github/discussions/modelcontextprotocol/python-sdk
[discussions-url]: https://github.com/modelcontextprotocol/python-sdk/discussions

## 概述

模型上下文协议允许应用程序以标准化的方式为LLMs提供上下文，将提供上下文的关注点与实际的LLM交互分离。这个Python SDK实现了完整的MCP规范，使以下操作变得简单：

- 构建可以连接到任何MCP服务器的MCP客户端
- 创建公开资源、提示和工具的MCP服务器
- 使用标准传输方式如stdio和SSE
- 处理所有MCP协议消息和生命周期事件

## 安装

### 将MCP添加到你的Python项目

我们推荐使用[uv](https://docs.astral.sh/uv/)来管理你的Python项目。如果你还没有创建一个uv管理的项目，请创建一个：

```bash
uv init mcp-server-demo
cd mcp-server-demo
```

然后将MCP添加到你的项目依赖中：

```bash
uv add "mcp[cli]"
```

或者，对于使用pip管理依赖的项目：

```bash
pip install "mcp[cli]"
```

### 运行独立的MCP开发工具

使用uv运行mcp命令：

```bash
uv run mcp
```

## 快速入门

让我们创建一个简单的MCP服务器，它公开一个计算器工具和一些数据：

```python
# server.py
from mcp.server.fastmcp import FastMCP

# 创建一个MCP服务器
mcp = FastMCP("Demo")

# 添加一个加法工具
@mcp.tool()
def add(a: int, b: int) -> int:
    """将两个数字相加"""
    return a + b

# 添加一个动态问候资源
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """获取个性化问候"""
    return f"Hello, {name}!"
```

你可以在[Claude Desktop](https://claude.ai/download)中安装这个服务器并立即与之交互，只需运行：

```bash
mcp install server.py
```

或者，你可以使用MCP检查器进行测试：

```bash
mcp dev server.py
```

## 什么是MCP？

[模型上下文协议(MCP)](https://modelcontextprotocol.io)让你能够构建服务器，以安全、标准化的方式向LLM应用程序公开数据和功能。可以将其视为Web API，但专为LLM交互设计。MCP服务器可以：

- 通过**资源**公开数据（可以将其视为类似于GET端点；它们用于将信息加载到LLM的上下文中）
- 通过**工具**提供功能（类似于POST端点；它们用于执行代码或产生副作用）
- 通过**提示**定义交互模式（LLM交互的可重用模板）
- 以及更多！

## 核心概念

### 服务器

FastMCP服务器是你与MCP协议的核心接口。它处理连接管理、协议合规性和消息路由：

```python
# 添加具有强类型的启动/关闭生命周期支持
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from fake_database import Database  # 替换为你实际的数据库类型
from mcp.server.fastmcp import Context, FastMCP

# 创建一个命名服务器
mcp = FastMCP("My App")

# 指定部署和开发的依赖项
mcp = FastMCP("My App", dependencies=["pandas", "numpy"])

@dataclass
class AppContext:
    db: Database

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """管理具有类型安全上下文的应用程序生命周期"""
    # 启动时初始化
    db = await Database.connect()
    try:
        yield AppContext(db=db)
    finally:
        # 关闭时清理
        await db.disconnect()

# 将生命周期传递给服务器
mcp = FastMCP("My App", lifespan=app_lifespan)

# 在工具中访问类型安全的生命周期上下文
@mcp.tool()
def query_db(ctx: Context) -> str:
    """使用初始化资源的工具"""
    db = ctx.request_context.lifespan_context.db
    return db.query()
```

### 资源

资源是你向LLMs公开数据的方式。它们类似于REST API中的GET端点 - 它们提供数据，但不应执行重要的计算或产生副作用：

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My App")

@mcp.resource("config://app")
def get_config() -> str:
    """静态配置数据"""
    return "App configuration here"

@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: str) -> str:
    """动态用户数据"""
    return f"Profile data for user {user_id}"
```

### 工具

工具让LLMs通过你的服务器采取行动。与资源不同，工具预期会执行计算并产生副作用：

```python
import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My App")

@mcp.tool()
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """计算BMI，给定体重（公斤）和身高（米）"""
    return weight_kg / (height_m**2)

@mcp.tool()
async def fetch_weather(city: str) -> str:
    """获取城市的当前天气"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.weather.com/{city}")
        return response.text
```

### 提示

提示是可重用的模板，帮助LLMs有效地与你的服务器交互：

```python
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base

mcp = FastMCP("My App")

@mcp.prompt()
def review_code(code: str) -> str:
    return f"Please review this code:\n\n{code}"

@mcp.prompt()
def debug_error(error: str) -> list[base.Message]:
    return [
        base.UserMessage("I'm seeing this error:"),
        base.UserMessage(error),
        base.AssistantMessage("I'll help debug that. What have you tried so far?"),
    ]
```

### 图像

FastMCP提供了一个`Image`类，自动处理图像数据：

```python
from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage

mcp = FastMCP("My App")

@mcp.tool()
def create_thumbnail(image_path: str) -> Image:
    """从图像创建缩略图"""
    img = PILImage.open(image_path)
    img.thumbnail((100, 100))
    return Image(data=img.tobytes(), format="png")
```

### 上下文

Context对象让你的工具和资源访问MCP功能：

```python
from mcp.server.fastmcp import FastMCP, Context

mcp = FastMCP("My App")

@mcp.tool()
async def long_task(files: list[str], ctx: Context) -> str:
    """处理多个文件并跟踪进度"""
    for i, file in enumerate(files):
        ctx.info(f"Processing {file}")
        await ctx.report_progress(i, len(files))
        data, mime_type = await ctx.read_resource(f"file://{file}")
    return "Processing complete"
```

## 运行你的服务器

### 开发模式

测试和调试服务器的最快方法是使用MCP检查器：

```bash
mcp dev server.py
# 添加依赖
mcp dev server.py --with pandas --with numpy
# 挂载本地代码
mcp dev server.py --with-editable .
```

### Claude Desktop集成

一旦你的服务器准备就绪，将其安装在Claude Desktop中：

```bash
mcp install server.py
# 自定义名称
mcp install server.py --name "My Analytics Server"
# 环境变量
mcp install server.py -v API_KEY=abc123 -v DB_URL=postgres://...
mcp install server.py -f .env
```

### 直接执行

对于高级场景，如自定义部署：

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My App")

if __name__ == "__main__":
    mcp.run()
```

运行它：

```bash
python server.py
# 或
mcp run server.py
```

### 挂载到现有的ASGI服务器

你可以使用`sse_app`方法将SSE服务器挂载到现有的ASGI服务器。这允许你将SSE服务器与其他ASGI应用程序集成。

```python
from starlette.applications import Starlette
from starlette.routing import Mount, Host
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My App")

# 将SSE服务器挂载到现有的ASGI服务器
app = Starlette(
    routes=[
        Mount('/', app=mcp.sse_app()),
    ]
)

# 或动态挂载为主机
app.router.routes.append(Host('mcp.acme.corp', app=mcp.sse_app()))
```

有关在Starlette中挂载应用程序的更多信息，请参阅[Starlette文档](https://www.starlette.io/routing/#submounting-routes)。

## 示例

### 回显服务器

一个演示资源、工具和提示的简单服务器：

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Echo")

@mcp.resource("echo://{message}")
def echo_resource(message: str) -> str:
    """将消息作为资源回显"""
    return f"Resource echo: {message}"

@mcp.tool()
def echo_tool(message: str) -> str:
    """将消息作为工具回显"""
    return f"Tool echo: {message}"

@mcp.prompt()
def echo_prompt(message: str) -> str:
    """创建回显提示"""
    return f"Please process this message: {message}"
```

### SQLite浏览器

一个展示数据库集成的更复杂示例：

```python
import sqlite3
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("SQLite Explorer")

@mcp.resource("schema://main")
def get_schema() -> str:
    """提供数据库模式作为资源"""
    conn = sqlite3.connect("database.db")
    schema = conn.execute("SELECT sql FROM sqlite_master WHERE type='table'").fetchall()
    return "\n".join(sql[0] for sql in schema if sql[0])

@mcp.tool()
def query_data(sql: str) -> str:
    """安全执行SQL查询"""
    conn = sqlite3.connect("database.db")
    try:
        result = conn.execute(sql).fetchall()
        return "\n".join(str(row) for row in result)
    except Exception as e:
        return f"Error: {str(e)}"
```

## 高级用法

### 低级服务器

为了获得更多控制，你可以直接使用低级服务器实现。这给你提供了对协议的完全访问权限，并允许你自定义服务器的每个方面，包括通过生命周期API进行生命周期管理：

```python
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from fake_database import Database  # 替换为你实际的数据库类型
from mcp.server import Server

@asynccontextmanager
async def server_lifespan(server: Server) -> AsyncIterator[dict]:
    """管理服务器启动和关闭生命周期。"""
    # 启动时初始化资源
    db = await Database.connect()
    try:
        yield {"db": db}
    finally:
        # 关闭时清理
        await db.disconnect()

# 将生命周期传递给服务器
server = Server("example-server", lifespan=server_lifespan)

# 在处理程序中访问生命周期上下文
@server.call_tool()
async def query_db(name: str, arguments: dict) -> list:
    ctx = server.request_context
    db = ctx.lifespan_context["db"]
    return await db.query(arguments["query"])
```

生命周期API提供：
- 一种在服务器启动时初始化资源并在停止时清理它们的方法
- 通过处理程序中的请求上下文访问初始化的资源
- 在生命周期和请求处理程序之间进行类型安全的上下文传递

```python
import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions

# 创建服务器实例
server = Server("example-server")

@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="example-prompt",
            description="An example prompt template",
            arguments=[
                types.PromptArgument(
                    name="arg1",
                    description="Example argument",
                    required=True
                )
            ],
        )
    ]

@server.get_prompt()
async def handle_get_prompt(
    name: str,
    arguments: dict[str, str] | None
) -> types.GetPromptResult:
    if name != "example-prompt":
        raise ValueError(f"Unknown prompt: {name}")
    return types.GetPromptResult(
        description="Example prompt",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(type="text", text="Example prompt text"),
            )
        ],
    )

async def run():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="example",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
```

### 编写MCP客户端

SDK提供了一个高级客户端接口，用于连接到MCP服务器：

```python
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

# 为stdio连接创建服务器参数
server_params = StdioServerParameters(
    command="python",  # 可执行文件
    args=["example_server.py"],  # 可选命令行参数
    env=None,  # 可选环境变量
)

# 可选：创建采样回调
async def handle_sampling_message(
    message: types.CreateMessageRequestParams,
) -> types.CreateMessageResult:
    return types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(
            type="text",
            text="Hello, world! from model",
        ),
        model="gpt-3.5-turbo",
        stopReason="endTurn",
    )

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read,
            write,
            sampling_callback=handle_sampling_message
        ) as session:
            # 初始化连接
            await session.initialize()
            
            # 列出可用提示
            prompts = await session.list_prompts()
            
            # 获取提示
            prompt = await session.get_prompt(
                "example-prompt",
                arguments={"arg1": "value"}
            )
            
            # 列出可用资源
            resources = await session.list_resources()
            
            # 列出可用工具
            tools = await session.list_tools()
            
            # 读取资源
            content, mime_type = await session.read_resource("file://some/path")
            
            # 调用工具
            result = await session.call_tool("tool-name", arguments={"arg1": "value"})

if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
```

### MCP原语

MCP协议定义了服务器可以实现的三个核心原语：

| 原语 | 控制 | 描述 | 使用示例 |
|------------|----------------------|-----------------------------------------------------|------------------------------|
| 提示 | 用户控制 | 由用户选择调用的交互模板 | 斜杠命令、菜单选项 |
| 资源 | 应用程序控制 | 由客户端应用程序管理的上下文数据 | 文件内容、API响应 |
| 工具 | 模型控制 | 公开给LLM以采取行动的函数 | API调用、数据更新 |

### 服务器能力

MCP服务器在初始化期间声明能力：

| 能力 | 功能标志 | 描述 |
|-------------|------------------------------|-----------------------------------|
| `prompts` | `listChanged` | 提示模板管理 |
| `resources` | `subscribe`<br/>`listChanged`| 资源公开和更新 |
| `tools` | `listChanged` | 工具发现和执行 |
| `logging` | - | 服务器日志配置 |
| `completion`| - | 参数完成建议 |

## 文档

- [模型上下文协议文档](https://modelcontextprotocol.io)
- [模型上下文协议规范](https://spec.modelcontextprotocol.io)
- [官方支持的服务器](https://github.com/modelcontextprotocol/servers)

## 贡献

我们热衷于支持各种经验水平的贡献者，并希望看到你参与到项目中。请参阅[贡献指南](https://github.com/modelcontextprotocol/python-sdk/blob/main/CONTRIBUTING.md)开始。

## 许可证

本项目采用MIT许可证 - 详情请参阅LICENSE文件。
