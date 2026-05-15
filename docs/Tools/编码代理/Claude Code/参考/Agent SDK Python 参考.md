---
sidebar_position: 9
---

> Python Agent SDK 的完整 API 参考，包括所有函数、类型和类。

## 安装

```bash theme={null}
pip install claude-agent-sdk
```

## `query()` 与 `ClaudeSDKClient` 的选择

Python SDK 提供两种与 Claude Code 交互的方式：

### 快速对比

| 特性                 | `query()`                   | `ClaudeSDKClient`                |
| :------------------- | :-------------------------- | :------------------------------- |
| **会话**             | 每次调用创建新会话          | 复用同一会话                     |
| **对话**             | 单次交互                    | 同一上下文中的多次交互           |
| **连接**             | 自动管理                    | 手动控制                         |
| **流式输入**         | ✅ 支持                     | ✅ 支持                          |
| **中断**             | ❌ 不支持                   | ✅ 支持                          |
| **Hooks**            | ✅ 支持                     | ✅ 支持                          |
| **自定义工具**       | ✅ 支持                     | ✅ 支持                          |
| **继续对话**         | ❌ 每次都是新会话            | ✅ 保持对话上下文                 |
| **适用场景**         | 一次性任务                  | 持续对话                         |

### 何时使用 `query()`（每次新建会话）

**适用于：**

* 不需要对话历史的一次性问题
* 独立任务，不需要之前交互的上下文
* 简单的自动化脚本
* 需要每次从零开始的场景

### 何时使用 `ClaudeSDKClient`（持续对话）

**适用于：**

* **继续对话** — 需要 Claude 记住上下文
* **追问** — 基于之前的回答继续提问
* **交互式应用** — 聊天界面、REPL
* **响应驱动的逻辑** — 下一步操作取决于 Claude 的响应
* **会话控制** — 显式管理对话生命周期

## 函数

### `query()`

每次与 Claude Code 交互时创建新会话。返回一个异步迭代器，在消息到达时逐条产出。每次调用 `query()` 都从零开始，不记忆之前的交互。

```python theme={null}
async def query(
    *,
    prompt: str | AsyncIterable[dict[str, Any]],
    options: ClaudeAgentOptions | None = None,
    transport: Transport | None = None
) -> AsyncIterator[Message]
```

#### 参数

| 参数        | 类型                           | 描述                                                   |
| :---------- | :----------------------------- | :----------------------------------------------------- |
| `prompt`    | `str \| AsyncIterable[dict]`   | 输入提示，可以是字符串或异步可迭代对象（流式模式）       |
| `options`   | `ClaudeAgentOptions \| None`   | 可选的配置对象（默认值为 `ClaudeAgentOptions()`）        |
| `transport` | `Transport \| None`            | 可选的自定义传输层，用于与 CLI 进程通信                  |

#### 返回值

返回 `AsyncIterator[Message]`，产出对话中的消息。

#### 示例 — 带选项

```python theme={null}
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions


async def main():
    options = ClaudeAgentOptions(
        system_prompt="你是一位 Python 专家开发者",
        permission_mode="acceptEdits",
        cwd="/home/user/project",
    )

    async for message in query(prompt="创建一个 Python web 服务器", options=options):
        print(message)


asyncio.run(main())
```

### `tool()`

用于定义具有类型安全的 MCP 工具的装饰器。

```python theme={null}
def tool(
    name: str,
    description: str,
    input_schema: type | dict[str, Any],
    annotations: ToolAnnotations | None = None
) -> Callable[[Callable[[Any], Awaitable[dict[str, Any]]]], SdkMcpTool[Any]]
```

#### 参数

| 参数            | 类型                                              | 描述                                              |
| :-------------- | :------------------------------------------------ | :------------------------------------------------ |
| `name`          | `str`                                             | 工具的唯一标识符                                   |
| `description`   | `str`                                             | 工具功能的人类可读描述                             |
| `input_schema`  | `type \| dict[str, Any]`                          | 定义工具输入参数的 schema（见下方）                 |
| `annotations`   | [`ToolAnnotations`](#toolannotations)` \| None`   | 可选的 MCP 工具注解，向客户端提供行为提示            |

#### 输入 schema 选项

1. **简单类型映射**（推荐）：

   ```python theme={null}
   {"text": str, "count": int, "enabled": bool}
   ```

2. **JSON Schema 格式**（用于复杂验证）：
   ```python theme={null}
   {
       "type": "object",
       "properties": {
           "text": {"type": "string"},
           "count": {"type": "integer", "minimum": 0},
       },
       "required": ["text"],
   }
   ```

#### 返回值

返回一个装饰器函数，包装工具实现并返回 `SdkMcpTool` 实例。

#### 示例

```python theme={null}
from claude_agent_sdk import tool
from typing import Any


@tool("greet", "问候用户", {"name": str})
async def greet(args: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": f"你好, {args['name']}!"}]}
```

#### `ToolAnnotations`

从 `mcp.types` 重新导出（也可以通过 `from claude_agent_sdk import ToolAnnotations` 导入）。所有字段均为可选提示；客户端不应依赖它们来做安全决策。

| 字段               | 类型            | 默认值   | 描述                                                                                             |
| :----------------- | :-------------- | :------- | :----------------------------------------------------------------------------------------------- |
| `title`            | `str \| None`   | `None`   | 工具的人类可读标题                                                                                |
| `readOnlyHint`     | `bool \| None`  | `False`  | 若为 `True`，表示该工具不会修改其环境                                                              |
| `destructiveHint`  | `bool \| None`  | `True`   | 若为 `True`，表示该工具可能执行破坏性更新（仅在 `readOnlyHint` 为 `False` 时有意义）                 |
| `idempotentHint`   | `bool \| None`  | `False`  | 若为 `True`，表示使用相同参数重复调用不会产生额外效果（仅在 `readOnlyHint` 为 `False` 时有意义）       |
| `openWorldHint`    | `bool \| None`  | `True`   | 若为 `True`，表示该工具与外部实体交互（例如网页搜索）。若为 `False`，则该工具的领域是封闭的（例如内存工具） |

```python theme={null}
from claude_agent_sdk import tool, ToolAnnotations
from typing import Any


@tool(
    "search",
    "搜索网页",
    {"query": str},
    annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True),
)
async def search(args: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": f"搜索结果: {args['query']}"}]}
```

### `create_sdk_mcp_server()`

创建在 Python 应用内运行的进程内 MCP 服务器。

```python theme={null}
def create_sdk_mcp_server(
    name: str,
    version: str = "1.0.0",
    tools: list[SdkMcpTool[Any]] | None = None
) -> McpSdkServerConfig
```

#### 参数

| 参数      | 类型                              | 默认值     | 描述                                      |
| :-------- | :-------------------------------- | :--------- | :---------------------------------------- |
| `name`    | `str`                             | -          | 服务器的唯一标识符                          |
| `version` | `str`                             | `"1.0.0"`  | 服务器版本字符串                            |
| `tools`   | `list[SdkMcpTool[Any]] \| None`   | `None`     | 使用 `@tool` 装饰器创建的工具函数列表         |

#### 返回值

返回一个 `McpSdkServerConfig` 对象，可传递给 `ClaudeAgentOptions.mcp_servers`。

#### 示例

```python theme={null}
from claude_agent_sdk import tool, create_sdk_mcp_server


@tool("add", "两数相加", {"a": float, "b": float})
async def add(args):
    return {"content": [{"type": "text", "text": f"和: {args['a'] + args['b']}"}]}


@tool("multiply", "两数相乘", {"a": float, "b": float})
async def multiply(args):
    return {"content": [{"type": "text", "text": f"积: {args['a'] * args['b']}"}]}


calculator = create_sdk_mcp_server(
    name="calculator",
    version="2.0.0",
    tools=[add, multiply],  # 传入装饰后的函数
)

# 与 Claude 一起使用
options = ClaudeAgentOptions(
    mcp_servers={"calc": calculator},
    allowed_tools=["mcp__calc__add", "mcp__calc__multiply"],
)
```

### `list_sessions()`

列出历史会话及其元数据。可按项目目录过滤或跨所有项目列出。同步函数，立即返回。

```python theme={null}
def list_sessions(
    directory: str | None = None,
    limit: int | None = None,
    include_worktrees: bool = True
) -> list[SDKSessionInfo]
```

#### 参数

| 参数                 | 类型            | 默认值   | 描述                                                       |
| :------------------- | :-------------- | :------- | :--------------------------------------------------------- |
| `directory`          | `str \| None`   | `None`   | 要列出会话的目录。省略时返回所有项目的会话                    |
| `limit`              | `int \| None`   | `None`   | 返回的最大会话数                                            |
| `include_worktrees`  | `bool`          | `True`   | 当 `directory` 在 git 仓库内时，包括所有 worktree 路径的会话   |

#### 返回类型：`SDKSessionInfo`

| 属性             | 类型           | 描述                                           |
| :--------------- | :------------- | :--------------------------------------------- |
| `session_id`     | `str`          | 唯一会话标识符                                  |
| `summary`        | `str`          | 显示标题：自定义标题、自动生成的摘要或第一条提示    |
| `last_modified`  | `int`          | 最后修改时间，自 epoch 以来的毫秒数               |
| `file_size`      | `int \| None`  | 会话文件大小（字节），远程存储后端为 `None`        |
| `custom_title`   | `str \| None`  | 用户设置的会话标题                               |
| `first_prompt`   | `str \| None`  | 会话中的第一条有意义用户提示                      |
| `git_branch`     | `str \| None`  | 会话结束时的 git 分支                            |
| `cwd`            | `str \| None`  | 会话的工作目录                                   |
| `tag`            | `str \| None`  | 用户设置的会话标签（参见 [`tag_session()`](#tag_session)） |
| `created_at`     | `int \| None`  | 会话创建时间，自 epoch 以来的毫秒数               |

#### 示例

打印某个项目最近 10 个会话。结果按 `last_modified` 降序排列，因此第一项是最新的。省略 `directory` 可跨所有项目搜索。

```python theme={null}
from claude_agent_sdk import list_sessions

for session in list_sessions(directory="/path/to/project", limit=10):
    print(f"{session.summary} ({session.session_id})")
```

### `get_session_messages()`

从历史会话中获取消息。同步函数，立即返回。

```python theme={null}
def get_session_messages(
    session_id: str,
    directory: str | None = None,
    limit: int | None = None,
    offset: int = 0
) -> list[SessionMessage]
```

#### 参数

| 参数         | 类型            | 默认值     | 描述                                  |
| :----------- | :-------------- | :--------- | :------------------------------------ |
| `session_id` | `str`           | 必填       | 要获取消息的会话 ID                     |
| `directory`  | `str \| None`   | `None`     | 要查找的项目目录。省略时搜索所有项目     |
| `limit`      | `int \| None`   | `None`     | 返回的最大消息数                        |
| `offset`     | `int`           | `0`        | 从开头跳过的消息数                      |

#### 返回类型：`SessionMessage`

| 属性                 | 类型                              | 描述               |
| :------------------- | :-------------------------------- | :----------------- |
| `type`               | `Literal["user", "assistant"]`    | 消息角色            |
| `uuid`               | `str`                             | 唯一消息标识符       |
| `session_id`         | `str`                             | 会话标识符          |
| `message`            | `Any`                             | 原始消息内容         |
| `parent_tool_use_id` | `None`                            | 保留供将来使用       |

#### 示例

```python theme={null}
from claude_agent_sdk import list_sessions, get_session_messages

sessions = list_sessions(limit=1)
if sessions:
    messages = get_session_messages(sessions[0].session_id)
    for msg in messages:
        print(f"[{msg.type}] {msg.uuid}")
```

### `get_session_info()`

按 ID 读取单个会话的元数据，无需扫描整个项目目录。同步函数，立即返回。

```python theme={null}
def get_session_info(
    session_id: str,
    directory: str | None = None,
) -> SDKSessionInfo | None
```

#### 参数

| 参数         | 类型            | 默认值   | 描述                                     |
| :----------- | :-------------- | :------- | :--------------------------------------- |
| `session_id` | `str`           | 必填     | 要查找的会话 UUID                          |
| `directory`  | `str \| None`   | `None`   | 项目目录路径。省略时搜索所有项目目录         |

返回 [`SDKSessionInfo`](#返回类型sdksessioninfo)，若未找到会话则返回 `None`。

#### 示例

查找单个会话的元数据，无需扫描项目目录。当你已经有了之前运行中的会话 ID 时非常有用。

```python theme={null}
from claude_agent_sdk import get_session_info

info = get_session_info("550e8400-e29b-41d4-a716-446655440000")
if info:
    print(f"{info.summary} (分支: {info.git_branch}, 标签: {info.tag})")
```

### `rename_session()`

通过追加自定义标题条目来重命名会话。重复调用是安全的；最近的标题会生效。同步函数。

```python theme={null}
def rename_session(
    session_id: str,
    title: str,
    directory: str | None = None,
) -> None
```

#### 参数

| 参数         | 类型            | 默认值   | 描述                                                   |
| :----------- | :-------------- | :------- | :----------------------------------------------------- |
| `session_id` | `str`           | 必填     | 要重命名的会话 UUID                                      |
| `title`      | `str`           | 必填     | 新标题。去除空白后必须非空                                |
| `directory`  | `str \| None`   | `None`   | 项目目录路径。省略时搜索所有项目目录                       |

如果 `session_id` 不是有效的 UUID 或 `title` 为空，抛出 `ValueError`；如果找不到会话，抛出 `FileNotFoundError`。

#### 示例

重命名最近的会话以便日后查找。新标题会出现在后续读取的 [`SDKSessionInfo.custom_title`](#返回类型sdksessioninfo) 中。

```python theme={null}
from claude_agent_sdk import list_sessions, rename_session

sessions = list_sessions(directory="/path/to/project", limit=1)
if sessions:
    rename_session(sessions[0].session_id, "重构认证模块")
```

### `tag_session()`

给会话打标签。传入 `None` 可清除标签。重复调用是安全的；最近的标签会生效。同步函数。

```python theme={null}
def tag_session(
    session_id: str,
    tag: str | None,
    directory: str | None = None,
) -> None
```

#### 参数

| 参数         | 类型            | 默认值   | 描述                                                |
| :----------- | :-------------- | :------- | :-------------------------------------------------- |
| `session_id` | `str`           | 必填     | 要打标签的会话 UUID                                   |
| `tag`        | `str \| None`   | 必填     | 标签字符串，或 `None` 以清除。存储前经过 Unicode 清理   |
| `directory`  | `str \| None`   | `None`   | 项目目录路径。省略时搜索所有项目目录                    |

如果 `session_id` 不是有效的 UUID 或 `tag` 清理后为空，抛出 `ValueError`；如果找不到会话，抛出 `FileNotFoundError`。

#### 示例

给会话打标签，之后可按该标签过滤。传入 `None` 可清除已有标签。

```python theme={null}
from claude_agent_sdk import list_sessions, tag_session

# 给会话打标签
tag_session("550e8400-e29b-41d4-a716-446655440000", "待审查")

# 之后：查找所有带该标签的会话
for session in list_sessions(directory="/path/to/project"):
    if session.tag == "待审查":
        print(session.summary)
```

## 类

### `ClaudeSDKClient`

**跨多次交互维护一个对话会话。** 这是 TypeScript SDK 的 `query()` 函数内部工作方式的 Python 等价物——它创建一个可以继续对话的客户端对象。

#### 关键特性

* **会话连续性**：跨多次 `query()` 调用维护对话上下文
* **同一对话**：会话保留先前的消息
* **中断支持**：可以中途停止执行
* **显式生命周期**：你控制会话何时开始和结束
* **响应驱动的流程**：可以对响应做出反应并发送跟进消息
* **自定义工具和 Hooks**：支持自定义工具（通过 `@tool` 装饰器创建）和 hooks

```python theme={null}
class ClaudeSDKClient:
    def __init__(self, options: ClaudeAgentOptions | None = None, transport: Transport | None = None)
    async def connect(self, prompt: str | AsyncIterable[dict] | None = None) -> None
    async def query(self, prompt: str | AsyncIterable[dict], session_id: str = "default") -> None
    async def receive_messages(self) -> AsyncIterator[Message]
    async def receive_response(self) -> AsyncIterator[Message]
    async def interrupt(self) -> None
    async def set_permission_mode(self, mode: str) -> None
    async def set_model(self, model: str | None = None) -> None
    async def rewind_files(self, user_message_id: str) -> None
    async def get_mcp_status(self) -> McpStatusResponse
    async def reconnect_mcp_server(self, server_name: str) -> None
    async def toggle_mcp_server(self, server_name: str, enabled: bool) -> None
    async def stop_task(self, task_id: str) -> None
    async def get_server_info(self) -> dict[str, Any] | None
    async def disconnect(self) -> None
```

#### 方法

| 方法                                      | 描述                                                                                                                      |
| :---------------------------------------- | :------------------------------------------------------------------------------------------------------------------------ |
| `__init__(options)`                       | 使用可选配置初始化客户端                                                                                                   |
| `connect(prompt)`                         | 连接到 Claude，可附带可选的初始提示或消息流                                                                                |
| `query(prompt, session_id)`               | 以流式模式发送新请求                                                                                                       |
| `receive_messages()`                      | 以异步迭代器的形式接收来自 Claude 的所有消息                                                                                |
| `receive_response()`                      | 接收消息直到并包括一个 ResultMessage                                                                                       |
| `interrupt()`                             | 发送中断信号（仅在流式模式下有效）                                                                                         |
| `set_permission_mode(mode)`               | 更改当前会话的权限模式                                                                                                     |
| `set_model(model)`                        | 更改当前会话的模型。传入 `None` 重置为默认值                                                                                |
| `rewind_files(user_message_id)`           | 将文件恢复到指定用户消息时的状态。需要 `enable_file_checkpointing=True`。参见[文件检查点](/en/agent-sdk/file-checkpointing) |
| `get_mcp_status()`                        | 获取所有已配置 MCP 服务器的状态。返回 [`McpStatusResponse`](#mcpstatusresponse)                                             |
| `reconnect_mcp_server(server_name)`       | 重新连接失败或断开连接的 MCP 服务器                                                                                        |
| `toggle_mcp_server(server_name, enabled)` | 在会话中途启用或禁用 MCP 服务器。禁用会移除其工具                                                                            |
| `stop_task(task_id)`                      | 停止正在运行的后台任务。随后会在消息流中产生一个状态为 `"stopped"` 的 [`TaskNotificationMessage`](#tasknotificationmessage) |
| `get_server_info()`                       | 获取服务器信息，包括会话 ID 和能力                                                                                          |
| `disconnect()`                            | 断开与 Claude 的连接                                                                                                       |

#### 上下文管理器支持

客户端可作为异步上下文管理器使用，实现自动连接管理：

```python theme={null}
async with ClaudeSDKClient() as client:
    await client.query("你好 Claude")
    async for message in client.receive_response():
        print(message)
```

> **重要：** 迭代消息时，避免使用 `break` 提前退出，因为这可能导致 asyncio 清理问题。应让迭代自然完成，或使用标志来跟踪是否已找到需要的内容。

#### 示例 — 继续对话

```python theme={null}
import asyncio
from claude_agent_sdk import ClaudeSDKClient, AssistantMessage, TextBlock, ResultMessage


async def main():
    async with ClaudeSDKClient() as client:
        # 第一个问题
        await client.query("法国的首都是什么？")

        # 处理响应
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Claude: {block.text}")

        # 追问 — 会话保留了之前的上下文
        await client.query("那个城市的人口是多少？")

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Claude: {block.text}")

        # 再次追问 — 仍在同一对话中
        await client.query("那里有哪些著名地标？")

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Claude: {block.text}")


asyncio.run(main())
```

#### 示例 — ClaudeSDKClient 流式输入

```python theme={null}
import asyncio
from claude_agent_sdk import ClaudeSDKClient


async def message_stream():
    """动态生成消息。"""
    yield {
        "type": "user",
        "message": {"role": "user", "content": "分析以下数据："},
    }
    await asyncio.sleep(0.5)
    yield {
        "type": "user",
        "message": {"role": "user", "content": "温度: 25°C, 湿度: 60%"},
    }
    await asyncio.sleep(0.5)
    yield {
        "type": "user",
        "message": {"role": "user", "content": "你看到了什么规律？"},
    }


async def main():
    async with ClaudeSDKClient() as client:
        # 流式输入给 Claude
        await client.query(message_stream())

        # 处理响应
        async for message in client.receive_response():
            print(message)

        # 在同一会话中追问
        await client.query("我们应该关注这些读数吗？")

        async for message in client.receive_response():
            print(message)


asyncio.run(main())
```

#### 示例 — 使用中断

```python theme={null}
import asyncio
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, ResultMessage


async def interruptible_task():
    options = ClaudeAgentOptions(allowed_tools=["Bash"], permission_mode="acceptEdits")

    async with ClaudeSDKClient(options=options) as client:
        # 启动一个长时间运行的任务
        await client.query("使用 bash sleep 命令从 1 慢慢数到 100")

        # 让它运行一会儿
        await asyncio.sleep(2)

        # 中断任务
        await client.interrupt()
        print("任务已中断！")

        # 排空被中断任务的消息（包括其 ResultMessage）
        async for message in client.receive_response():
            if isinstance(message, ResultMessage):
                print(f"被中断的任务结束，subtype={message.subtype!r}")
                # subtype 为 "error_during_execution" 表示被中断的任务

        # 发送新命令
        await client.query("直接说你好就行了")

        # 现在接收新的响应
        async for message in client.receive_response():
            if isinstance(message, ResultMessage) and message.subtype == "success":
                print(f"新结果: {message.result}")


asyncio.run(interruptible_task())
```

> **注意：** **中断后的缓冲区行为：** `interrupt()` 发送停止信号但不清除消息缓冲区。被中断任务已产生的消息（包括其 `ResultMessage`，`subtype="error_during_execution"`）仍保留在流中。你必须在读取新查询的响应之前用 `receive_response()` 排空它们。如果你在 `interrupt()` 之后立即发送新查询并只调用一次 `receive_response()`，你将收到被中断任务的消息，而不是新查询的响应。

#### 示例 — 高级权限控制

```python theme={null}
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from claude_agent_sdk.types import (
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)


async def custom_permission_handler(
    tool_name: str, input_data: dict, context: ToolPermissionContext
) -> PermissionResultAllow | PermissionResultDeny:
    """自定义工具权限逻辑。"""

    # 阻止写入系统目录
    if tool_name == "Write" and input_data.get("file_path", "").startswith("/system/"):
        return PermissionResultDeny(
            message="不允许写入系统目录", interrupt=True
        )

    # 重定向敏感文件操作
    if tool_name in ["Write", "Edit"] and "config" in input_data.get("file_path", ""):
        safe_path = f"./sandbox/{input_data['file_path']}"
        return PermissionResultAllow(
            updated_input={**input_data, "file_path": safe_path}
        )

    # 允许其他所有操作
    return PermissionResultAllow(updated_input=input_data)


async def main():
    options = ClaudeAgentOptions(
        can_use_tool=custom_permission_handler, allowed_tools=["Read", "Write", "Edit"]
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query("更新系统配置文件")

        async for message in client.receive_response():
            # 会使用沙箱路径代替
            print(message)


asyncio.run(main())
```

## 类型

> **注意：** **`@dataclass` vs `TypedDict`：** 此 SDK 使用两种类型。用 `@dataclass` 装饰的类（如 `ResultMessage`、`AgentDefinition`、`TextBlock`）在运行时是对象实例，支持属性访问：`msg.result`。用 `TypedDict` 定义的类（如 `ThinkingConfigEnabled`、`McpStdioServerConfig`、`SyncHookJSONOutput`）在运行时是**普通字典**，需要键访问：`config["budget_tokens"]`，而不是 `config.budget_tokens`。两种都支持 `ClassName(field=value)` 调用语法，但只有 dataclass 生成带属性的对象。

### `SdkMcpTool`

使用 `@tool` 装饰器创建的 SDK MCP 工具定义。

```python theme={null}
@dataclass
class SdkMcpTool(Generic[T]):
    name: str
    description: str
    input_schema: type[T] | dict[str, Any]
    handler: Callable[[T], Awaitable[dict[str, Any]]]
    annotations: ToolAnnotations | None = None
```

| 属性            | 类型                                         | 描述                                                              |
| :-------------- | :------------------------------------------- | :---------------------------------------------------------------- |
| `name`          | `str`                                        | 工具的唯一标识符                                                    |
| `description`   | `str`                                        | 人类可读的描述                                                      |
| `input_schema`  | `type[T] \| dict[str, Any]`                  | 输入验证的 schema                                                   |
| `handler`       | `Callable[[T], Awaitable[dict[str, Any]]]`   | 处理工具执行的异步函数                                              |
| `annotations`   | `ToolAnnotations \| None`                    | 可选的 MCP 工具注解（如 `readOnlyHint`、`destructiveHint`、`openWorldHint`）。来自 `mcp.types` |

### `Transport`

自定义传输层实现的抽象基类。用于通过自定义通道（例如远程连接而非本地子进程）与 Claude 进程通信。

> **警告：** 这是低级内部 API。接口可能在未来的版本中更改。自定义实现必须更新以匹配任何接口更改。

```python theme={null}
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class Transport(ABC):
    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def write(self, data: str) -> None: ...

    @abstractmethod
    def read_messages(self) -> AsyncIterator[dict[str, Any]]: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    def is_ready(self) -> bool: ...

    @abstractmethod
    async def end_input(self) -> None: ...
```

| 方法               | 描述                                                  |
| :----------------- | :---------------------------------------------------- |
| `connect()`        | 连接传输层并准备通信                                    |
| `write(data)`      | 将原始数据（JSON + 换行符）写入传输层                    |
| `read_messages()`  | 异步迭代器，产出解析后的 JSON 消息                       |
| `close()`          | 关闭连接并清理资源                                       |
| `is_ready()`       | 如果传输层可以发送和接收数据，返回 `True`                  |
| `end_input()`      | 关闭输入流（例如，对子进程传输层关闭 stdin）               |

导入方式：`from claude_agent_sdk import Transport`

### `ClaudeAgentOptions`

Claude Code 查询的配置 dataclass。

```python theme={null}
@dataclass
class ClaudeAgentOptions:
    tools: list[str] | ToolsPreset | None = None
    allowed_tools: list[str] = field(default_factory=list)
    system_prompt: str | SystemPromptPreset | None = None
    mcp_servers: dict[str, McpServerConfig] | str | Path = field(default_factory=dict)
    strict_mcp_config: bool = False
    permission_mode: PermissionMode | None = None
    continue_conversation: bool = False
    resume: str | None = None
    max_turns: int | None = None
    max_budget_usd: float | None = None
    disallowed_tools: list[str] = field(default_factory=list)
    model: str | None = None
    fallback_model: str | None = None
    betas: list[SdkBeta] = field(default_factory=list)
    output_format: dict[str, Any] | None = None
    permission_prompt_tool_name: str | None = None
    cwd: str | Path | None = None
    cli_path: str | Path | None = None
    settings: str | None = None
    add_dirs: list[str | Path] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    extra_args: dict[str, str | None] = field(default_factory=dict)
    max_buffer_size: int | None = None
    debug_stderr: Any = sys.stderr  # 已弃用
    stderr: Callable[[str], None] | None = None
    can_use_tool: CanUseTool | None = None
    hooks: dict[HookEvent, list[HookMatcher]] | None = None
    user: str | None = None
    include_partial_messages: bool = False
    include_hook_events: bool = False
    fork_session: bool = False
    agents: dict[str, AgentDefinition] | None = None
    setting_sources: list[SettingSource] | None = None
    sandbox: SandboxSettings | None = None
    plugins: list[SdkPluginConfig] = field(default_factory=list)
    max_thinking_tokens: int | None = None  # 已弃用：请使用 thinking
    thinking: ThinkingConfig | None = None
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    enable_file_checkpointing: bool = False
    session_store: SessionStore | None = None
    session_store_flush: SessionStoreFlushMode = "batched"
```

| 属性                            | 类型                                                                                    | 默认值                               | 描述                                                                                                                                                                                                                             |
| :------------------------------ | :-------------------------------------------------------------------------------------- | :----------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tools`                         | `list[str] \| ToolsPreset \| None`                                                      | `None`                               | 工具配置。使用 `{"type": "preset", "preset": "claude_code"}` 获取 Claude Code 的默认工具集                                                                                                                                         |
| `allowed_tools`                 | `list[str]`                                                                             | `[]`                                 | 无需提示即可自动批准的工具。这不会将 Claude 限制为仅使用这些工具；未列出的工具会回退到 `permission_mode` 和 `can_use_tool`。使用 `disallowed_tools` 来阻止工具。参见[权限](/en/agent-sdk/permissions#allow-and-deny-rules)               |
| `system_prompt`                 | `str \| SystemPromptPreset \| None`                                                     | `None`                               | 系统提示配置。传入字符串用于自定义提示，或使用 `{"type": "preset", "preset": "claude_code"}` 获取 Claude Code 的系统提示。添加 `"append"` 来扩展预设                                                                                |
| `mcp_servers`                   | `dict[str, McpServerConfig] \| str \| Path`                                             | `{}`                                 | MCP 服务器配置或配置文件路径                                                                                                                                                                                                       |
| `strict_mcp_config`             | `bool`                                                                                  | `False`                              | 为 `True` 时，仅使用 `mcp_servers` 中传入的服务器，忽略项目的 `.mcp.json`、用户设置和插件提供的 MCP 服务器。映射到 CLI 的 `--strict-mcp-config` 标志                                                                                |
| `permission_mode`               | `PermissionMode \| None`                                                                | `None`                               | 工具使用的权限模式                                                                                                                                                                                                                |
| `continue_conversation`         | `bool`                                                                                  | `False`                              | 继续最近的对话                                                                                                                                                                                                                    |
| `resume`                        | `str \| None`                                                                           | `None`                               | 要恢复的会话 ID                                                                                                                                                                                                                   |
| `max_turns`                     | `int \| None`                                                                           | `None`                               | 最大 agentic 轮次（工具使用往返次数）                                                                                                                                                                                              |
| `max_budget_usd`                | `float \| None`                                                                         | `None`                               | 当客户端成本估算达到此 USD 值时停止查询。与 `total_cost_usd` 使用相同的估算方式；准确性的注意事项见[跟踪成本和使用情况](/en/agent-sdk/cost-tracking)                                                                                |
| `disallowed_tools`              | `list[str]`                                                                             | `[]`                                 | 始终拒绝的工具。拒绝规则优先检查，覆盖 `allowed_tools` 和 `permission_mode`（包括 `bypassPermissions`）                                                                                                                            |
| `enable_file_checkpointing`     | `bool`                                                                                  | `False`                              | 启用文件更改跟踪以支持回退。参见[文件检查点](/en/agent-sdk/file-checkpointing)                                                                                                                                                      |
| `model`                         | `str \| None`                                                                           | `None`                               | 要使用的 Claude 模型                                                                                                                                                                                                              |
| `fallback_model`                | `str \| None`                                                                           | `None`                               | 主模型失败时使用的回退模型                                                                                                                                                                                                         |
| `betas`                         | `list[SdkBeta]`                                                                         | `[]`                                 | 要启用的 Beta 功能。参见 [`SdkBeta`](#sdkbeta) 了解可用选项                                                                                                                                                                        |
| `output_format`                 | `dict[str, Any] \| None`                                                                | `None`                               | 结构化响应的输出格式（如 `{"type": "json_schema", "schema": {...}}`）。详见[结构化输出](/en/agent-sdk/structured-outputs)                                                                                                          |
| `permission_prompt_tool_name`   | `str \| None`                                                                           | `None`                               | 用于权限提示的 MCP 工具名称                                                                                                                                                                                                        |
| `cwd`                           | `str \| Path \| None`                                                                   | `None`                               | 当前工作目录                                                                                                                                                                                                                      |
| `cli_path`                      | `str \| Path \| None`                                                                   | `None`                               | Claude Code CLI 可执行文件的自定义路径                                                                                                                                                                                             |
| `settings`                      | `str \| None`                                                                           | `None`                               | 设置文件路径                                                                                                                                                                                                                      |
| `add_dirs`                      | `list[str \| Path]`                                                                     | `[]`                                 | Claude 可以访问的附加目录                                                                                                                                                                                                          |
| `env`                           | `dict[str, str]`                                                                        | `{}`                                 | 在继承的进程环境之上合并的环境变量。底层 CLI 读取的变量见[环境变量](/en/env-vars)，超时相关变量见[处理缓慢或卡住的 API 响应](#处理缓慢或卡住的-api-响应)                                                                               |
| `extra_args`                    | `dict[str, str \| None]`                                                                | `{}`                                 | 直接传递给 CLI 的附加 CLI 参数                                                                                                                                                                                                     |
| `max_buffer_size`               | `int \| None`                                                                           | `None`                               | 缓冲 CLI stdout 时的最大字节数                                                                                                                                                                                                     |
| `debug_stderr`                  | `Any`                                                                                   | `sys.stderr`                         | *已弃用* — 用于调试输出的类文件对象。请改用 `stderr` 回调                                                                                                                                                                           |
| `stderr`                        | `Callable[[str], None] \| None`                                                         | `None`                               | CLI 的 stderr 输出回调函数                                                                                                                                                                                                         |
| `can_use_tool`                  | [`CanUseTool`](#canusetool) ` \| None`                                                  | `None`                               | 工具权限回调函数。详见[权限类型](#canusetool)                                                                                                                                                                                       |
| `hooks`                         | `dict[HookEvent, list[HookMatcher]] \| None`                                            | `None`                               | 用于拦截事件的 Hook 配置                                                                                                                                                                                                           |
| `user`                          | `str \| None`                                                                           | `None`                               | 用户标识符                                                                                                                                                                                                                        |
| `include_partial_messages`      | `bool`                                                                                  | `False`                              | 包含部分消息流事件。启用时会产出 [`StreamEvent`](#streamevent) 消息                                                                                                                                                                 |
| `include_hook_events`           | `bool`                                                                                  | `False`                              | 在消息流中包含 hook 生命周期事件，作为 `HookEventMessage` 对象                                                                                                                                                                      |
| `fork_session`                  | `bool`                                                                                  | `False`                              | 使用 `resume` 恢复时，分叉到新的会话 ID 而非继续原会话                                                                                                                                                                              |
| `agents`                        | `dict[str, AgentDefinition] \| None`                                                    | `None`                               | 以编程方式定义的子代理                                                                                                                                                                                                             |
| `plugins`                       | `list[SdkPluginConfig]`                                                                 | `[]`                                 | 从本地路径加载自定义插件。详见[插件](/en/agent-sdk/plugins)                                                                                                                                                                         |
| `sandbox`                       | [`SandboxSettings`](#sandboxsettings) ` \| None`                                        | `None`                               | 以编程方式配置沙箱行为。详见[沙箱设置](#sandboxsettings)                                                                                                                                                                            |
| `setting_sources`               | `list[SettingSource] \| None`                                                           | `None`（CLI 默认：所有源）            | 控制加载哪些文件系统设置。传入 `[]` 以禁用用户、项目和本地设置。托管策略设置始终加载。参见[使用 Claude Code 功能](/en/agent-sdk/claude-code-features#what-settingsources-does-not-control)                                              |
| `skills`                        | `list[str] \| Literal["all"] \| None`                                                   | `None`                               | 会话可用的技能。传入 `"all"` 启用所有发现的技能，或传入技能名称列表。设置后，SDK 自动启用 Skill 工具，无需在 `allowed_tools` 中列出。参见[技能](/en/agent-sdk/skills)                                                                   |
| `max_thinking_tokens`           | `int \| None`                                                                           | `None`                               | *已弃用* — 思考块的最大 token 数。请改用 `thinking`                                                                                                                                                                                 |
| `thinking`                      | [`ThinkingConfig`](#thinkingconfig) ` \| None`                                          | `None`                               | 控制扩展思考行为。优先于 `max_thinking_tokens`                                                                                                                                                                                      |
| `effort`                        | `Literal["low", "medium", "high", "xhigh", "max"] \| None`                              | `None`                               | 思考深度的努力级别                                                                                                                                                                                                                  |
| `session_store`                 | [`SessionStore`](/en/agent-sdk/session-storage#the-sessionstore-interface) ` \| None`   | `None`                               | 将会话转录镜像到外部后端，以便任何主机都能恢复它们。参见[持久化会话到外部存储](/en/agent-sdk/session-storage)                                                                                                                        |
| `session_store_flush`           | `Literal["batched", "eager"]`                                                           | `"batched"`                          | 何时将镜像的转录条目刷新到 `session_store`。`"batched"` 每轮或缓冲区满时刷新一次；`"eager"` 每帧后触发后台刷新。当 `session_store` 为 `None` 时忽略                                                                                 |

#### 处理缓慢或卡住的 API 响应

CLI 子进程读取几个控制 API 超时和卡顿检测的环境变量。通过 `ClaudeAgentOptions.env` 传入它们：

```python theme={null}
options = ClaudeAgentOptions(
    env={
        "API_TIMEOUT_MS": "120000",
        "CLAUDE_CODE_MAX_RETRIES": "2",
        "CLAUDE_ASYNC_AGENT_STALL_TIMEOUT_MS": "120000",
    },
)
```

* `API_TIMEOUT_MS`：Anthropic 客户端上每个请求的超时时间，以毫秒为单位。默认 `600000`。适用于主循环和所有子代理。
* `CLAUDE_CODE_MAX_RETRIES`：最大 API 重试次数。默认 `10`。每次重试有自己独立的 `API_TIMEOUT_MS` 窗口，因此最坏情况下的实际耗时约为 `API_TIMEOUT_MS × (CLAUDE_CODE_MAX_RETRIES + 1)` 加上退避时间。
* `CLAUDE_ASYNC_AGENT_STALL_TIMEOUT_MS`：通过 `run_in_background` 启动的子代理的卡顿看门狗。默认 `600000`。每次流事件时重置；卡顿时它会中止子代理，将任务标记为失败，并将错误（附部分结果）报告给父代理。不适用于同步子代理。
* `CLAUDE_ENABLE_STREAM_WATCHDOG=1` 配合 `CLAUDE_STREAM_IDLE_TIMEOUT_MS`：当标头已到达但响应体停止流式传输时中止请求。默认关闭。`CLAUDE_STREAM_IDLE_TIMEOUT_MS` 默认为 `300000` 并以此值为下限。中止的请求会走正常的重试路径。

### `OutputFormat`

结构化输出验证的配置。将其作为 `dict` 传递给 `ClaudeAgentOptions` 的 `output_format` 字段：

```python theme={null}
# output_format 的预期字典格式
{
    "type": "json_schema",
    "schema": {...},  # 你的 JSON Schema 定义
}
```

| 字段     | 是否必填 | 描述                                   |
| :------- | :------- | :------------------------------------- |
| `type`   | 是       | 必须为 `"json_schema"` 用于 JSON Schema 验证 |
| `schema` | 是       | 用于输出验证的 JSON Schema 定义            |

### `SystemPromptPreset`

使用 Claude Code 预设系统提示的配置，可附加额外内容。

```python theme={null}
class SystemPromptPreset(TypedDict):
    type: Literal["preset"]
    preset: Literal["claude_code"]
    append: NotRequired[str]
    exclude_dynamic_sections: NotRequired[bool]
```

| 字段                        | 是否必填 | 描述                                                                                                                                                                                                                       |
| :-------------------------- | :------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `type`                      | 是       | 必须为 `"preset"` 以使用预设系统提示                                                                                                                                                                                        |
| `preset`                    | 是       | 必须为 `"claude_code"` 以使用 Claude Code 的系统提示                                                                                                                                                                        |
| `append`                    | 否       | 追加到预设系统提示的额外指令                                                                                                                                                                                                |
| `exclude_dynamic_sections`  | 否       | 将工作目录、git-repo 标志和自动内存路径等每会话上下文从系统提示移到第一条用户消息中。提高跨用户和机器的提示缓存复用率。参见[修改系统提示](/en/agent-sdk/modifying-system-prompts#improve-prompt-caching-across-users-and-machines) |

### `SettingSource`

控制 SDK 从哪些基于文件系统的配置源加载设置。

```python theme={null}
SettingSource = Literal["user", "project", "local"]
```

| 值          | 描述                               | 位置                            |
| :---------- | :--------------------------------- | :------------------------------ |
| `"user"`    | 全局用户设置                        | `~/.claude/settings.json`       |
| `"project"` | 共享项目设置（版本控制）             | `.claude/settings.json`         |
| `"local"`   | 本地项目设置（被 gitignore）         | `.claude/settings.local.json`   |

#### 默认行为

当 `setting_sources` 省略或为 `None` 时，`query()` 加载与 Claude Code CLI 相同的文件系统设置：用户、项目和本地。托管策略设置在任何情况下都会加载。参见 [settingSources 不控制什么](/en/agent-sdk/claude-code-features#what-settingsources-does-not-control) 了解无论此选项如何都会被读取的输入，以及如何禁用它们。

#### 为什么使用 setting_sources

**禁用文件系统设置：**

```python theme={null}
# 不从磁盘加载用户、项目或本地设置
from claude_agent_sdk import query, ClaudeAgentOptions

async for message in query(
    prompt="分析这段代码",
    options=ClaudeAgentOptions(
        setting_sources=[]
    ),
):
    print(message)
```

> **注意：** 在 Python SDK 0.1.59 及更早版本中，空列表被视为与省略该选项相同，因此 `setting_sources=[]` 不会禁用文件系统设置。如果你需要空列表生效，请升级到更新的版本。TypeScript SDK 不受影响。

**显式加载所有文件系统设置：**

```python theme={null}
from claude_agent_sdk import query, ClaudeAgentOptions

async for message in query(
    prompt="分析这段代码",
    options=ClaudeAgentOptions(
        setting_sources=["user", "project", "local"]
    ),
):
    print(message)
```

**仅加载特定的设置源：**

```python theme={null}
# 仅加载项目设置，忽略用户和本地
async for message in query(
    prompt="运行 CI 检查",
    options=ClaudeAgentOptions(
        setting_sources=["project"]  # 仅 .claude/settings.json
    ),
):
    print(message)
```

**测试和 CI 环境：**

```python theme={null}
# 通过排除本地设置在 CI 中确保行为一致
async for message in query(
    prompt="运行测试",
    options=ClaudeAgentOptions(
        setting_sources=["project"],  # 仅团队共享的设置
        permission_mode="bypassPermissions",
    ),
):
    print(message)
```

**纯 SDK 应用：**

```python theme={null}
# 以编程方式定义一切。
# 传入 [] 以退出文件系统设置源。
async for message in query(
    prompt="审查这个 PR",
    options=ClaudeAgentOptions(
        setting_sources=[],
        agents={...},
        mcp_servers={...},
        allowed_tools=["Read", "Grep", "Glob"],
    ),
):
    print(message)
```

**加载 CLAUDE.md 项目指令：**

```python theme={null}
# 加载项目设置以包含 CLAUDE.md 文件
async for message in query(
    prompt="按照项目约定添加新功能",
    options=ClaudeAgentOptions(
        system_prompt={
            "type": "preset",
            "preset": "claude_code",  # 使用 Claude Code 的系统提示
        },
        setting_sources=["project"],  # 从项目加载 CLAUDE.md
        allowed_tools=["Read", "Write", "Edit"],
    ),
):
    print(message)
```

#### 设置优先级

当加载多个源时，设置按以下优先级合并（从高到低）：

1. 本地设置（`.claude/settings.local.json`）
2. 项目设置（`.claude/settings.json`）
3. 用户设置（`~/.claude/settings.json`）

编程式选项（如 `agents` 和 `allowed_tools`）覆盖用户、项目和本地文件系统设置。托管策略设置优先于编程式选项。

### `AgentDefinition`

以编程方式定义的子代理配置。

```python theme={null}
@dataclass
class AgentDefinition:
    description: str
    prompt: str
    tools: list[str] | None = None
    disallowedTools: list[str] | None = None
    model: str | None = None
    skills: list[str] | None = None
    memory: Literal["user", "project", "local"] | None = None
    mcpServers: list[str | dict[str, Any]] | None = None
    initialPrompt: str | None = None
    maxTurns: int | None = None
    background: bool | None = None
    effort: Literal["low", "medium", "high", "xhigh", "max"] | int | None = None
    permissionMode: PermissionMode | None = None
```

| 字段              | 是否必填 | 描述                                                                                          |
| :---------------- | :------- | :-------------------------------------------------------------------------------------------- |
| `description`     | 是       | 何时使用此代理的自然语言描述                                                                    |
| `prompt`          | 是       | 代理的系统提示                                                                                 |
| `tools`           | 否       | 允许的工具名称数组。如果省略，继承所有工具                                                       |
| `disallowedTools` | 否       | 从代理工具集中移除的工具名称数组                                                                 |
| `model`           | 否       | 此代理的模型覆盖。接受别名如 `"sonnet"`、`"opus"`、`"haiku"` 或 `"inherit"`，或完整模型 ID。省略时使用主模型 |
| `skills`          | 否       | 在代理启动时预加载到其上下文中的技能名称列表。未列出的技能仍可通过 Skill 工具调用                  |
| `memory`          | 否       | 此代理的内存来源：`"user"`、`"project"` 或 `"local"`                                             |
| `mcpServers`      | 否       | 此代理可用的 MCP 服务器。每个条目是服务器名称或内联 `{name: config}` 字典                         |
| `initialPrompt`   | 否       | 当此代理作为主线程代理运行时，自动作为第一条用户轮次提交                                          |
| `maxTurns`        | 否       | 代理停止前的最大 agentic 轮次数                                                                  |
| `background`      | 否       | 调用时将此代理作为非阻塞后台任务运行                                                              |
| `effort`          | 否       | 此代理的推理努力级别。接受命名级别或整数                                                          |
| `permissionMode`  | 否       | 此代理内工具执行的权限模式。参见 [`PermissionMode`](#permissionmode)                               |

> **注意：** `AgentDefinition` 字段名使用驼峰命名法，如 `disallowedTools`、`permissionMode` 和 `maxTurns`。这些名称直接映射到与 TypeScript SDK 共享的传输格式。这与 `ClaudeAgentOptions` 不同，后者对等效的顶层字段使用 Python 蛇形命名法，如 `disallowed_tools` 和 `permission_mode`。因为 `AgentDefinition` 是 dataclass，在构造时传入蛇形命名法的关键字会引发 `TypeError`。

### `PermissionMode`

控制工具执行的权限模式。

```python theme={null}
PermissionMode = Literal[
    "default",            # 标准权限行为
    "acceptEdits",        # 自动接受文件编辑
    "plan",               # 规划模式 — 仅只读工具
    "dontAsk",            # 拒绝未预批准的任何内容，而不是提示
    "bypassPermissions",  # 绕过所有权限检查（谨慎使用）
]
```

### `CanUseTool`

工具权限回调函数的类型别名。

```python theme={null}
CanUseTool = Callable[
    [str, dict[str, Any], ToolPermissionContext], Awaitable[PermissionResult]
]
```

回调接收：

* `tool_name`：被调用的工具名称
* `input_data`：工具的输入参数
* `context`：带有附加信息的 `ToolPermissionContext`

返回一个 `PermissionResult`（`PermissionResultAllow` 或 `PermissionResultDeny`）。

### `ToolPermissionContext`

传递给工具权限回调的上下文信息。

```python theme={null}
@dataclass
class ToolPermissionContext:
    signal: Any | None = None  # 未来：中止信号支持
    suggestions: list[PermissionUpdate] = field(default_factory=list)
    blocked_path: str | None = None
    decision_reason: str | None = None
    title: str | None = None
    display_name: str | None = None
    description: str | None = None
```

| 字段               | 类型                       | 描述                                                                                                                                                                          |
| :----------------- | :------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `signal`           | `Any \| None`              | 保留供将来的中止信号支持使用                                                                                                                                                    |
| `suggestions`      | `list[PermissionUpdate]`   | 来自 CLI 的权限更新建议。Bash 提示包含一个带有 `localSettings` 目标的建议，因此在 `updated_permissions` 中返回它会将规则写入 `.claude/settings.local.json` 并跨会话持久化           |
| `blocked_path`     | `str \| None`              | 触发权限请求的文件路径（适用时）。例如，当 Bash 命令尝试访问允许目录之外的路径时                                                                                                  |
| `decision_reason`  | `str \| None`              | 触发此权限请求的原因。当 hook 返回 `"ask"` 时，从 PreToolUse hook 的 `permissionDecisionReason` 转发而来                                                                        |
| `title`            | `str \| None`              | 完整的权限提示句子，如 `Claude 想要读取 foo.txt`。存在时用作主要的提示文本                                                                                                       |
| `display_name`     | `str \| None`              | 工具操作的简短名词短语，如 `读取文件`，适合用作按钮标签                                                                                                                           |
| `description`      | `str \| None`              | 权限 UI 的人类可读副标题                                                                                                                                                         |

### `PermissionResult`

权限回调结果的联合类型。

```python theme={null}
PermissionResult = PermissionResultAllow | PermissionResultDeny
```

### `PermissionResultAllow`

表示应允许工具调用的结果。

```python theme={null}
@dataclass
class PermissionResultAllow:
    behavior: Literal["allow"] = "allow"
    updated_input: dict[str, Any] | None = None
    updated_permissions: list[PermissionUpdate] | None = None
```

| 字段                   | 类型                               | 默认值     | 描述                               |
| :--------------------- | :--------------------------------- | :--------- | :--------------------------------- |
| `behavior`             | `Literal["allow"]`                 | `"allow"`  | 必须为 "allow"                      |
| `updated_input`        | `dict[str, Any] \| None`           | `None`     | 用于替代原始输入的修改后输入          |
| `updated_permissions`  | `list[PermissionUpdate] \| None`   | `None`     | 要应用的权限更新                     |

### `PermissionResultDeny`

表示应拒绝工具调用的结果。

```python theme={null}
@dataclass
class PermissionResultDeny:
    behavior: Literal["deny"] = "deny"
    message: str = ""
    interrupt: bool = False
```

| 字段         | 类型                | 默认值    | 描述                           |
| :----------- | :------------------ | :-------- | :----------------------------- |
| `behavior`   | `Literal["deny"]`   | `"deny"`  | 必须为 "deny"                   |
| `message`    | `str`               | `""`      | 解释工具被拒绝原因的消息          |
| `interrupt`  | `bool`              | `False`   | 是否中断当前执行                  |

### `PermissionUpdate`

以编程方式更新权限的配置。

```python theme={null}
@dataclass
class PermissionUpdate:
    type: Literal[
        "addRules",
        "replaceRules",
        "removeRules",
        "setMode",
        "addDirectories",
        "removeDirectories",
    ]
    rules: list[PermissionRuleValue] | None = None
    behavior: Literal["allow", "deny", "ask"] | None = None
    mode: PermissionMode | None = None
    directories: list[str] | None = None
    destination: (
        Literal["userSettings", "projectSettings", "localSettings", "session"] | None
    ) = None
```

| 字段           | 类型                                        | 描述                           |
| :------------- | :------------------------------------------ | :----------------------------- |
| `type`         | `Literal[...]`                              | 权限更新操作的类型               |
| `rules`        | `list[PermissionRuleValue] \| None`          | 用于添加/替换/移除操作的规则      |
| `behavior`     | `Literal["allow", "deny", "ask"] \| None`    | 基于规则的操作的行为              |
| `mode`         | `PermissionMode \| None`                     | setMode 操作的模式               |
| `directories`  | `list[str] \| None`                          | 添加/移除目录操作的目录           |
| `destination`  | `Literal[...] \| None`                       | 权限更新应用的位置               |

### `PermissionRuleValue`

权限更新中要添加、替换或移除的规则。

```python theme={null}
@dataclass
class PermissionRuleValue:
    tool_name: str
    rule_content: str | None = None
```

### `ToolsPreset`

使用 Claude Code 默认工具集的预设工具配置。

```python theme={null}
class ToolsPreset(TypedDict):
    type: Literal["preset"]
    preset: Literal["claude_code"]
```

### `ThinkingConfig`

控制扩展思考行为。三种配置的联合体：

```python theme={null}
class ThinkingConfigAdaptive(TypedDict):
    type: Literal["adaptive"]


class ThinkingConfigEnabled(TypedDict):
    type: Literal["enabled"]
    budget_tokens: int


class ThinkingConfigDisabled(TypedDict):
    type: Literal["disabled"]


ThinkingConfig = ThinkingConfigAdaptive | ThinkingConfigEnabled | ThinkingConfigDisabled
```

| 变体        | 字段                      | 描述                               |
| :---------- | :------------------------ | :--------------------------------- |
| `adaptive`  | `type`                    | Claude 自适应决定何时思考            |
| `enabled`   | `type`, `budget_tokens`   | 使用特定的 token 预算启用思考        |
| `disabled`  | `type`                    | 禁用思考                           |

由于这些是 `TypedDict` 类，它们在运行时是普通字典。可以用字典字面量构造，也可以像构造函数一样调用类；两种方式都生成 `dict`。使用 `config["budget_tokens"]` 访问字段，而不是 `config.budget_tokens`：

```python theme={null}
from claude_agent_sdk import ClaudeAgentOptions, ThinkingConfigEnabled

# 选项 1：字典字面量（推荐，无需导入）
options = ClaudeAgentOptions(thinking={"type": "enabled", "budget_tokens": 20000})

# 选项 2：构造函数风格（返回普通字典）
config = ThinkingConfigEnabled(type="enabled", budget_tokens=20000)
print(config["budget_tokens"])  # 20000
# config.budget_tokens 会引发 AttributeError
```

### `SdkBeta`

SDK beta 功能的字面量类型。

```python theme={null}
SdkBeta = Literal["context-1m-2025-08-07"]
```

与 `ClaudeAgentOptions` 中的 `betas` 字段一起使用以启用 beta 功能。

> **警告：** `context-1m-2025-08-07` beta 已于 2026 年 4 月 30 日退役。对 Claude Sonnet 4.5 或 Sonnet 4 传递此标头无效，超过标准 200k token 上下文窗口的请求会返回错误。要使用 1M token 上下文窗口，请迁移到 [Claude Sonnet 4.6、Claude Opus 4.6 或 Claude Opus 4.7](https://platform.claude.com/docs/en/about-claude/models/overview)，这些模型以标准定价包含 1M 上下文，无需 beta 标头。

### `McpSdkServerConfig`

使用 `create_sdk_mcp_server()` 创建的 SDK MCP 服务器的配置。

```python theme={null}
class McpSdkServerConfig(TypedDict):
    type: Literal["sdk"]
    name: str
    instance: Any  # MCP Server 实例
```

### `McpServerConfig`

MCP 服务器配置的联合类型。

```python theme={null}
McpServerConfig = (
    McpStdioServerConfig | McpSSEServerConfig | McpHttpServerConfig | McpSdkServerConfig
)
```

#### `McpStdioServerConfig`

```python theme={null}
class McpStdioServerConfig(TypedDict):
    type: NotRequired[Literal["stdio"]]  # 为向后兼容而可选
    command: str
    args: NotRequired[list[str]]
    env: NotRequired[dict[str, str]]
```

#### `McpSSEServerConfig`

```python theme={null}
class McpSSEServerConfig(TypedDict):
    type: Literal["sse"]
    url: str
    headers: NotRequired[dict[str, str]]
```

#### `McpHttpServerConfig`

```python theme={null}
class McpHttpServerConfig(TypedDict):
    type: Literal["http"]
    url: str
    headers: NotRequired[dict[str, str]]
```

### `McpServerStatusConfig`

由 [`get_mcp_status()`](#方法) 报告的 MCP 服务器配置。这是所有 [`McpServerConfig`](#mcpserverconfig) 传输变体的联合体，加上一个用于通过 claude.ai 代理的服务器的仅输出 `claudeai-proxy` 变体。

```python theme={null}
McpServerStatusConfig = (
    McpStdioServerConfig
    | McpSSEServerConfig
    | McpHttpServerConfig
    | McpSdkServerConfigStatus
    | McpClaudeAIProxyServerConfig
)
```

`McpSdkServerConfigStatus` 是 [`McpSdkServerConfig`](#mcpsdkserverconfig) 的可序列化形式，只有 `type`（`"sdk"`）和 `name`（`str`）字段；进程内 `instance` 被省略。`McpClaudeAIProxyServerConfig` 有 `type`（`"claudeai-proxy"`）、`url`（`str`）和 `id`（`str`）字段。

### `McpStatusResponse`

[`ClaudeSDKClient.get_mcp_status()`](#方法) 的响应。在 `mcpServers` 键下包装服务器状态列表。

```python theme={null}
class McpStatusResponse(TypedDict):
    mcpServers: list[McpServerStatus]
```

### `McpServerStatus`

连接的 MCP 服务器的状态，包含在 [`McpStatusResponse`](#mcpstatusresponse) 中。

```python theme={null}
class McpServerStatus(TypedDict):
    name: str
    status: McpServerConnectionStatus  # "connected" | "failed" | "needs-auth" | "pending" | "disabled"
    serverInfo: NotRequired[McpServerInfo]
    error: NotRequired[str]
    config: NotRequired[McpServerStatusConfig]
    scope: NotRequired[str]
    tools: NotRequired[list[McpToolInfo]]
```

| 字段          | 类型                                                           | 描述                                                                                                               |
| :------------ | :------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------- |
| `name`        | `str`                                                          | 服务器名称                                                                                                          |
| `status`      | `str`                                                          | `"connected"`、`"failed"`、`"needs-auth"`、`"pending"` 或 `"disabled"` 之一                                         |
| `serverInfo`  | `dict`（可选）                                                  | 服务器名称和版本（`{"name": str, "version": str}`）                                                                  |
| `error`       | `str`（可选）                                                   | 服务器连接失败时的错误消息                                                                                           |
| `config`      | [`McpServerStatusConfig`](#mcpserverstatusconfig)（可选）       | 服务器配置。与 [`McpServerConfig`](#mcpserverconfig) 形状相同（stdio、SSE、HTTP 或 SDK），外加通过 claude.ai 连接的服务器的 `claudeai-proxy` 变体 |
| `scope`       | `str`（可选）                                                   | 配置作用域                                                                                                          |
| `tools`       | `list`（可选）                                                  | 此服务器提供的工具，每个工具带有 `name`、`description` 和 `annotations` 字段                                           |

### `SdkPluginConfig`

在 SDK 中加载插件的配置。

```python theme={null}
class SdkPluginConfig(TypedDict):
    type: Literal["local"]
    path: str
```

| 字段   | 类型                 | 描述                                              |
| :----- | :------------------- | :------------------------------------------------ |
| `type` | `Literal["local"]`   | 必须为 `"local"`（目前仅支持本地插件）               |
| `path` | `str`                | 插件目录的绝对或相对路径                             |

**示例：**

```python theme={null}
plugins = [
    {"type": "local", "path": "./my-plugin"},
    {"type": "local", "path": "/absolute/path/to/plugin"},
]
```

有关创建和使用插件的完整信息，参见[插件](/en/agent-sdk/plugins)。

## 消息类型

### `Message`

所有可能消息的联合类型。

```python theme={null}
Message = (
    UserMessage
    | AssistantMessage
    | SystemMessage
    | ResultMessage
    | StreamEvent
    | RateLimitEvent
)
```

### `UserMessage`

用户输入消息。

```python theme={null}
@dataclass
class UserMessage:
    content: str | list[ContentBlock]
    uuid: str | None = None
    parent_tool_use_id: str | None = None
    tool_use_result: dict[str, Any] | None = None
```

| 字段                  | 类型                          | 描述                                      |
| :-------------------- | :---------------------------- | :---------------------------------------- |
| `content`             | `str \| list[ContentBlock]`   | 消息内容，可以是文本或内容块                |
| `uuid`                | `str \| None`                 | 唯一消息标识符                              |
| `parent_tool_use_id`  | `str \| None`                 | 如果此消息是工具结果响应，则为工具使用 ID     |
| `tool_use_result`     | `dict[str, Any] \| None`      | 工具结果数据（适用时）                       |

### `AssistantMessage`

带有内容块的助手响应消息。

```python theme={null}
@dataclass
class AssistantMessage:
    content: list[ContentBlock]
    model: str
    parent_tool_use_id: str | None = None
    error: AssistantMessageError | None = None
    usage: dict[str, Any] | None = None
    message_id: str | None = None
```

| 字段                  | 类型                                                           | 描述                                                              |
| :-------------------- | :------------------------------------------------------------- | :---------------------------------------------------------------- |
| `content`             | `list[ContentBlock]`                                           | 响应中的内容块列表                                                  |
| `model`               | `str`                                                          | 生成响应的模型                                                      |
| `parent_tool_use_id`  | `str \| None`                                                  | 如果这是嵌套响应，则为工具使用 ID                                     |
| `error`               | [`AssistantMessageError`](#assistantmessageerror) ` \| None`   | 如果响应遇到错误，则为错误类型                                        |
| `usage`               | `dict[str, Any] \| None`                                       | 每条消息的 token 使用量（与 [`ResultMessage.usage`](#resultmessage) 相同的键） |
| `message_id`          | `str \| None`                                                  | API 消息 ID。同一轮次的多个消息共享相同的 ID                           |

### `AssistantMessageError`

助手消息可能的错误类型。

```python theme={null}
AssistantMessageError = Literal[
    "authentication_failed",
    "billing_error",
    "rate_limit",
    "invalid_request",
    "server_error",
    "max_output_tokens",
    "unknown",
]
```

### `SystemMessage`

带元数据的系统消息。

```python theme={null}
@dataclass
class SystemMessage:
    subtype: str
    data: dict[str, Any]
```

### `ResultMessage`

最终结果消息，包含成本和使用信息。

```python theme={null}
@dataclass
class ResultMessage:
    subtype: str
    duration_ms: int
    duration_api_ms: int
    is_error: bool
    num_turns: int
    session_id: str
    stop_reason: str | None = None
    total_cost_usd: float | None = None
    usage: dict[str, Any] | None = None
    result: str | None = None
    structured_output: Any = None
    model_usage: dict[str, Any] | None = None
    permission_denials: list[Any] | None = None
    deferred_tool_use: DeferredToolUse | None = None
    errors: list[str] | None = None
    api_error_status: int | None = None
    uuid: str | None = None
```

`usage` 字典在存在时包含以下键：

| 键                             | 类型    | 描述                           |
| ------------------------------ | ------- | ------------------------------ |
| `input_tokens`                 | `int`   | 消耗的总输入 token 数            |
| `output_tokens`                | `int`   | 生成的总输出 token 数            |
| `cache_creation_input_tokens`  | `int`   | 用于创建新缓存条目的 token 数     |
| `cache_read_input_tokens`      | `int`   | 从已有缓存条目读取的 token 数     |

`model_usage` 字典将模型名称映射到每个模型的使用情况。内部字典的键使用驼峰命名法，因为该值未经修改地从底层 CLI 进程传递过来，与 TypeScript [`ModelUsage`](/en/agent-sdk/typescript#modelusage) 类型匹配：

| 键                          | 类型     | 描述                                                                              |
| --------------------------- | -------- | --------------------------------------------------------------------------------- |
| `inputTokens`               | `int`    | 此模型的输入 token 数                                                               |
| `outputTokens`              | `int`    | 此模型的输出 token 数                                                               |
| `cacheReadInputTokens`      | `int`    | 此模型的缓存读取 token 数                                                            |
| `cacheCreationInputTokens`  | `int`    | 此模型的缓存创建 token 数                                                            |
| `webSearchRequests`         | `int`    | 此模型发出的网页搜索请求数                                                            |
| `costUSD`                   | `float`  | 此模型的估算成本（USD），客户端计算。计费注意事项见[跟踪成本和使用情况](/en/agent-sdk/cost-tracking) |
| `contextWindow`             | `int`    | 此模型的上下文窗口大小                                                                |
| `maxOutputTokens`           | `int`    | 此模型的最大输出 token 限制                                                           |

### `StreamEvent`

流式传输期间部分消息更新的流事件。仅在 `ClaudeAgentOptions` 中设置 `include_partial_messages=True` 时接收。通过 `from claude_agent_sdk.types import StreamEvent` 导入。

```python theme={null}
@dataclass
class StreamEvent:
    uuid: str
    session_id: str
    event: dict[str, Any]  # 原始 Claude API 流事件
    parent_tool_use_id: str | None = None
```

| 字段                  | 类型              | 描述                                        |
| :-------------------- | :---------------- | :------------------------------------------ |
| `uuid`                | `str`             | 此事件的唯一标识符                             |
| `session_id`          | `str`             | 会话标识符                                    |
| `event`               | `dict[str, Any]`  | 原始 Claude API 流事件数据                     |
| `parent_tool_use_id`  | `str \| None`     | 如果此事件来自子代理，则为父工具使用 ID          |

### `RateLimitEvent`

速率限制状态变化时发出（例如从 `"allowed"` 变为 `"allowed_warning"`）。用于在用户遇到硬限制之前警告他们，或在状态为 `"rejected"` 时退避。

```python theme={null}
@dataclass
class RateLimitEvent:
    rate_limit_info: RateLimitInfo
    uuid: str
    session_id: str
```

| 字段               | 类型                                | 描述                  |
| :----------------- | :---------------------------------- | :-------------------- |
| `rate_limit_info`  | [`RateLimitInfo`](#ratelimitinfo)   | 当前速率限制状态        |
| `uuid`             | `str`                               | 唯一事件标识符          |
| `session_id`       | `str`                               | 会话标识符              |

### `RateLimitInfo`

[`RateLimitEvent`](#ratelimitevent) 携带的速率限制状态。

```python theme={null}
RateLimitStatus = Literal["allowed", "allowed_warning", "rejected"]
RateLimitType = Literal[
    "five_hour", "seven_day", "seven_day_opus", "seven_day_sonnet", "overage"
]


@dataclass
class RateLimitInfo:
    status: RateLimitStatus
    resets_at: int | None = None
    rate_limit_type: RateLimitType | None = None
    utilization: float | None = None
    overage_status: RateLimitStatus | None = None
    overage_resets_at: int | None = None
    overage_disabled_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
```

| 字段                       | 类型                      | 描述                                                              |
| :------------------------- | :------------------------ | :---------------------------------------------------------------- |
| `status`                   | `RateLimitStatus`         | 当前状态。`"allowed_warning"` 表示接近限制；`"rejected"` 表示已达到限制 |
| `resets_at`                | `int \| None`             | 速率限制窗口重置的 Unix 时间戳                                       |
| `rate_limit_type`          | `RateLimitType \| None`   | 适用的速率限制窗口                                                   |
| `utilization`              | `float \| None`           | 已消耗的速率限制比例（0.0 到 1.0）                                    |
| `overage_status`           | `RateLimitStatus \| None` | 按量付费超额使用状态（如适用）                                         |
| `overage_resets_at`        | `int \| None`             | 超额窗口重置的 Unix 时间戳                                            |
| `overage_disabled_reason`  | `str \| None`             | 如果状态为 `"rejected"`，超额不可用的原因                              |
| `raw`                      | `dict[str, Any]`          | 来自 CLI 的完整原始字典，包括上述未建模的字段                            |

### `TaskStartedMessage`

后台任务启动时发出。后台任务包括在主要轮次之外跟踪的任何事物：后台 Bash 命令、[Monitor](#monitor) 监视、通过 Agent 工具生成的子代理或远程代理。`task_type` 字段告诉你具体是哪种。此命名与 `Task` 到 `Agent` 工具的重命名无关。

```python theme={null}
@dataclass
class TaskStartedMessage(SystemMessage):
    task_id: str
    description: str
    uuid: str
    session_id: str
    tool_use_id: str | None = None
    task_type: str | None = None
```

| 字段           | 类型            | 描述                                                                                                   |
| :------------- | :-------------- | :----------------------------------------------------------------------------------------------------- |
| `task_id`      | `str`           | 任务的唯一标识符                                                                                         |
| `description`  | `str`           | 任务的描述                                                                                              |
| `uuid`         | `str`           | 唯一消息标识符                                                                                           |
| `session_id`   | `str`           | 会话标识符                                                                                              |
| `tool_use_id`  | `str \| None`   | 关联的工具使用 ID                                                                                        |
| `task_type`    | `str \| None`   | 后台任务的类型：`"local_bash"` 用于后台 Bash 和 Monitor 监视，`"local_agent"` 或 `"remote_agent"`          |

### `TaskUsage`

后台任务的 token 和计时数据。

```python theme={null}
class TaskUsage(TypedDict):
    total_tokens: int
    tool_uses: int
    duration_ms: int
```

### `TaskProgressMessage`

周期性发出，包含正在运行的后台任务的进度更新。

```python theme={null}
@dataclass
class TaskProgressMessage(SystemMessage):
    task_id: str
    description: str
    usage: TaskUsage
    uuid: str
    session_id: str
    tool_use_id: str | None = None
    last_tool_name: str | None = None
```

| 字段              | 类型            | 描述                       |
| :---------------- | :-------------- | :------------------------- |
| `task_id`         | `str`           | 任务的唯一标识符              |
| `description`     | `str`           | 当前状态描述                 |
| `usage`           | `TaskUsage`     | 此任务目前的 token 使用量      |
| `uuid`            | `str`           | 唯一消息标识符                |
| `session_id`      | `str`           | 会话标识符                   |
| `tool_use_id`     | `str \| None`   | 关联的工具使用 ID             |
| `last_tool_name`  | `str \| None`   | 任务使用的最后一个工具名称      |

### `TaskNotificationMessage`

后台任务完成、失败或停止时发出。后台任务包括 `run_in_background` Bash 命令、Monitor 监视和后台子代理。

```python theme={null}
@dataclass
class TaskNotificationMessage(SystemMessage):
    task_id: str
    status: TaskNotificationStatus  # "completed" | "failed" | "stopped"
    output_file: str
    summary: str
    uuid: str
    session_id: str
    tool_use_id: str | None = None
    usage: TaskUsage | None = None
```

| 字段           | 类型                       | 描述                                          |
| :------------- | :------------------------- | :-------------------------------------------- |
| `task_id`      | `str`                      | 任务的唯一标识符                                 |
| `status`       | `TaskNotificationStatus`   | `"completed"`、`"failed"` 或 `"stopped"` 之一   |
| `output_file`  | `str`                      | 任务输出文件的路径                                |
| `summary`      | `str`                      | 任务结果摘要                                     |
| `uuid`         | `str`                      | 唯一消息标识符                                   |
| `session_id`   | `str`                      | 会话标识符                                       |
| `tool_use_id`  | `str \| None`              | 关联的工具使用 ID                                 |
| `usage`        | `TaskUsage \| None`        | 任务的最终 token 使用量                            |

## 内容块类型

### `ContentBlock`

所有内容块的联合类型。

```python theme={null}
ContentBlock = TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock
```

### `TextBlock`

文本内容块。

```python theme={null}
@dataclass
class TextBlock:
    text: str
```

### `ThinkingBlock`

思考内容块（用于具有思考能力的模型）。

```python theme={null}
@dataclass
class ThinkingBlock:
    thinking: str
    signature: str
```

### `ToolUseBlock`

工具使用请求块。

```python theme={null}
@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
```

### `ToolResultBlock`

工具执行结果块。

```python theme={null}
@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: str | list[dict[str, Any]] | None = None
    is_error: bool | None = None
```

## 错误类型

### `ClaudeSDKError`

所有 SDK 错误的基类异常。

```python theme={null}
class ClaudeSDKError(Exception):
    """Claude SDK 的基类错误。"""
```

### `CLINotFoundError`

Claude Code CLI 未安装或未找到时引发。

```python theme={null}
class CLINotFoundError(CLIConnectionError):
    def __init__(
        self, message: str = "未找到 Claude Code", cli_path: str | None = None
    ):
        """
        Args:
            message: 错误消息（默认："未找到 Claude Code"）
            cli_path: 未找到的 CLI 的可选路径
        """
```

### `CLIConnectionError`

与 Claude Code 连接失败时引发。

```python theme={null}
class CLIConnectionError(ClaudeSDKError):
    """连接 Claude Code 失败。"""
```

### `ProcessError`

Claude Code 进程失败时引发。

```python theme={null}
class ProcessError(ClaudeSDKError):
    def __init__(
        self, message: str, exit_code: int | None = None, stderr: str | None = None
    ):
        self.exit_code = exit_code
        self.stderr = stderr
```

### `CLIJSONDecodeError`

JSON 解析失败时引发。

```python theme={null}
class CLIJSONDecodeError(ClaudeSDKError):
    def __init__(self, line: str, original_error: Exception):
        """
        Args:
            line: 解析失败的那一行
            original_error: 原始的 JSON 解码异常
        """
        self.line = line
        self.original_error = original_error
```

## Hook 类型

关于使用 hooks 的全面指南，包括示例和常见模式，请参见 [Hooks 指南](/en/agent-sdk/hooks)。

### `HookEvent`

支持的 hook 事件类型。

```python theme={null}
HookEvent = Literal[
    "PreToolUse",          # 工具执行前调用
    "PostToolUse",         # 工具执行后调用
    "PostToolUseFailure",  # 工具执行失败时调用
    "UserPromptSubmit",    # 用户提交提示时调用
    "Stop",                # 停止执行时调用
    "SubagentStop",        # 子代理停止时调用
    "PreCompact",          # 消息压缩前调用
    "Notification",        # 通知事件时调用
    "SubagentStart",       # 子代理启动时调用
    "PermissionRequest",   # 需要权限决定时调用
]
```

> **注意：** TypeScript SDK 支持 Python 中尚不可用的额外 hook 事件：`SessionStart`、`SessionEnd`、`Setup`、`TeammateIdle`、`TaskCompleted`、`ConfigChange`、`WorktreeCreate`、`WorktreeRemove` 和 `PostToolBatch`。

### `HookCallback`

Hook 回调函数的类型定义。

```python theme={null}
HookCallback = Callable[[HookInput, str | None, HookContext], Awaitable[HookJSONOutput]]
```

参数：

* `input`：基于 `hook_event_name` 具有可辨识联合类型的强类型 hook 输入（参见 [`HookInput`](#hookinput)）
* `tool_use_id`：可选的工具使用标识符（用于与工具相关的 hooks）
* `context`：带有附加信息的 Hook 上下文

返回一个 [`HookJSONOutput`](#hookjsonoutput)，可能包含：

* `decision`：`"block"` 阻止操作
* `systemMessage`：向用户显示的警告消息
* `hookSpecificOutput`：Hook 特定的输出数据

### `HookContext`

传递给 hook 回调的上下文信息。

```python theme={null}
class HookContext(TypedDict):
    signal: Any | None  # 未来：中止信号支持
```

### `HookMatcher`

配置 hooks 以匹配特定事件或工具的配置。

```python theme={null}
@dataclass
class HookMatcher:
    matcher: str | None = (
        None  # 要匹配的工具名称或模式（如 "Bash"、"Write|Edit"）
    )
    hooks: list[HookCallback] = field(
        default_factory=list
    )  # 要执行的回调列表
    timeout: float | None = (
        None  # 此匹配器中所有 hooks 的超时时间（秒，默认：60）
    )
```

### `HookInput`

所有 hook 输入类型的联合类型。实际类型取决于 `hook_event_name` 字段。

```python theme={null}
HookInput = (
    PreToolUseHookInput
    | PostToolUseHookInput
    | PostToolUseFailureHookInput
    | UserPromptSubmitHookInput
    | StopHookInput
    | SubagentStopHookInput
    | PreCompactHookInput
    | NotificationHookInput
    | SubagentStartHookInput
    | PermissionRequestHookInput
)
```

### `BaseHookInput`

所有 hook 输入类型中都存在的基础字段。

```python theme={null}
class BaseHookInput(TypedDict):
    session_id: str
    transcript_path: str
    cwd: str
    permission_mode: NotRequired[str]
```

| 字段               | 类型              | 描述                       |
| :----------------- | :---------------- | :------------------------- |
| `session_id`       | `str`             | 当前会话标识符               |
| `transcript_path`  | `str`             | 会话转录文件的路径            |
| `cwd`              | `str`             | 当前工作目录                 |
| `permission_mode`  | `str`（可选）      | 当前权限模式                 |

### `PreToolUseHookInput`

`PreToolUse` hook 事件的输入数据。

```python theme={null}
class PreToolUseHookInput(BaseHookInput):
    hook_event_name: Literal["PreToolUse"]
    tool_name: str
    tool_input: dict[str, Any]
    tool_use_id: str
    agent_id: NotRequired[str]
    agent_type: NotRequired[str]
```

| 字段               | 类型                      | 描述                                          |
| :----------------- | :------------------------ | :-------------------------------------------- |
| `hook_event_name`  | `Literal["PreToolUse"]`   | 始终为 "PreToolUse"                             |
| `tool_name`        | `str`                     | 即将执行的工具名称                                |
| `tool_input`       | `dict[str, Any]`          | 工具的输入参数                                   |
| `tool_use_id`      | `str`                     | 此工具使用的唯一标识符                             |
| `agent_id`         | `str`（可选）              | 子代理标识符，在子代理内触发 hook 时存在            |
| `agent_type`       | `str`（可选）              | 子代理类型，在子代理内触发 hook 时存在              |

### `PostToolUseHookInput`

`PostToolUse` hook 事件的输入数据。

```python theme={null}
class PostToolUseHookInput(BaseHookInput):
    hook_event_name: Literal["PostToolUse"]
    tool_name: str
    tool_input: dict[str, Any]
    tool_response: Any
    tool_use_id: str
    agent_id: NotRequired[str]
    agent_type: NotRequired[str]
```

| 字段               | 类型                       | 描述                                          |
| :----------------- | :------------------------- | :-------------------------------------------- |
| `hook_event_name`  | `Literal["PostToolUse"]`   | 始终为 "PostToolUse"                            |
| `tool_name`        | `str`                      | 已执行的工具名称                                  |
| `tool_input`       | `dict[str, Any]`           | 使用的输入参数                                   |
| `tool_response`    | `Any`                      | 工具执行的响应                                   |
| `tool_use_id`      | `str`                      | 此工具使用的唯一标识符                             |
| `agent_id`         | `str`（可选）               | 子代理标识符，在子代理内触发 hook 时存在            |
| `agent_type`       | `str`（可选）               | 子代理类型，在子代理内触发 hook 时存在              |

### `PostToolUseFailureHookInput`

`PostToolUseFailure` hook 事件的输入数据。在工具执行失败时调用。

```python theme={null}
class PostToolUseFailureHookInput(BaseHookInput):
    hook_event_name: Literal["PostToolUseFailure"]
    tool_name: str
    tool_input: dict[str, Any]
    tool_use_id: str
    error: str
    is_interrupt: NotRequired[bool]
    agent_id: NotRequired[str]
    agent_type: NotRequired[str]
```

| 字段               | 类型                              | 描述                                          |
| :----------------- | :-------------------------------- | :-------------------------------------------- |
| `hook_event_name`  | `Literal["PostToolUseFailure"]`   | 始终为 "PostToolUseFailure"                     |
| `tool_name`        | `str`                             | 失败的工具名称                                   |
| `tool_input`       | `dict[str, Any]`                  | 使用的输入参数                                   |
| `tool_use_id`      | `str`                             | 此工具使用的唯一标识符                             |
| `error`            | `str`                             | 失败执行的错误消息                                |
| `is_interrupt`     | `bool`（可选）                     | 失败是否由中断引起                                |
| `agent_id`         | `str`（可选）                      | 子代理标识符，在子代理内触发 hook 时存在            |
| `agent_type`       | `str`（可选）                      | 子代理类型，在子代理内触发 hook 时存在              |

### `UserPromptSubmitHookInput`

`UserPromptSubmit` hook 事件的输入数据。

```python theme={null}
class UserPromptSubmitHookInput(BaseHookInput):
    hook_event_name: Literal["UserPromptSubmit"]
    prompt: str
```

| 字段               | 类型                            | 描述                    |
| :----------------- | :------------------------------ | :---------------------- |
| `hook_event_name`  | `Literal["UserPromptSubmit"]`   | 始终为 "UserPromptSubmit" |
| `prompt`           | `str`                           | 用户提交的提示             |

### `StopHookInput`

`Stop` hook 事件的输入数据。

```python theme={null}
class StopHookInput(BaseHookInput):
    hook_event_name: Literal["Stop"]
    stop_hook_active: bool
```

| 字段                 | 类型                | 描述                       |
| :------------------- | :------------------ | :------------------------- |
| `hook_event_name`    | `Literal["Stop"]`   | 始终为 "Stop"               |
| `stop_hook_active`   | `bool`              | 停止 hook 是否处于活动状态    |

### `SubagentStopHookInput`

`SubagentStop` hook 事件的输入数据。

```python theme={null}
class SubagentStopHookInput(BaseHookInput):
    hook_event_name: Literal["SubagentStop"]
    stop_hook_active: bool
    agent_id: str
    agent_transcript_path: str
    agent_type: str
```

| 字段                      | 类型                        | 描述                        |
| :------------------------ | :-------------------------- | :-------------------------- |
| `hook_event_name`         | `Literal["SubagentStop"]`   | 始终为 "SubagentStop"         |
| `stop_hook_active`        | `bool`                      | 停止 hook 是否处于活动状态     |
| `agent_id`                | `str`                       | 子代理的唯一标识符             |
| `agent_transcript_path`   | `str`                       | 子代理转录文件的路径           |
| `agent_type`              | `str`                       | 子代理的类型                  |

### `PreCompactHookInput`

`PreCompact` hook 事件的输入数据。

```python theme={null}
class PreCompactHookInput(BaseHookInput):
    hook_event_name: Literal["PreCompact"]
    trigger: Literal["manual", "auto"]
    custom_instructions: str | None
```

| 字段                    | 类型                          | 描述                       |
| :---------------------- | :---------------------------- | :------------------------- |
| `hook_event_name`       | `Literal["PreCompact"]`       | 始终为 "PreCompact"          |
| `trigger`               | `Literal["manual", "auto"]`   | 触发压缩的原因               |
| `custom_instructions`   | `str \| None`                 | 压缩的自定义指令              |

### `NotificationHookInput`

`Notification` hook 事件的输入数据。

```python theme={null}
class NotificationHookInput(BaseHookInput):
    hook_event_name: Literal["Notification"]
    message: str
    title: NotRequired[str]
    notification_type: str
```

| 字段                 | 类型                        | 描述                  |
| :------------------- | :-------------------------- | :-------------------- |
| `hook_event_name`    | `Literal["Notification"]`   | 始终为 "Notification"   |
| `message`            | `str`                       | 通知消息内容             |
| `title`              | `str`（可选）                 | 通知标题                |
| `notification_type`  | `str`                       | 通知类型                |

### `SubagentStartHookInput`

`SubagentStart` hook 事件的输入数据。

```python theme={null}
class SubagentStartHookInput(BaseHookInput):
    hook_event_name: Literal["SubagentStart"]
    agent_id: str
    agent_type: str
```

| 字段               | 类型                         | 描述                    |
| :----------------- | :--------------------------- | :---------------------- |
| `hook_event_name`  | `Literal["SubagentStart"]`   | 始终为 "SubagentStart"    |
| `agent_id`         | `str`                        | 子代理的唯一标识符         |
| `agent_type`       | `str`                        | 子代理的类型              |

### `PermissionRequestHookInput`

`PermissionRequest` hook 事件的输入数据。允许 hooks 以编程方式处理权限决策。

```python theme={null}
class PermissionRequestHookInput(BaseHookInput):
    hook_event_name: Literal["PermissionRequest"]
    tool_name: str
    tool_input: dict[str, Any]
    permission_suggestions: NotRequired[list[Any]]
```

| 字段                       | 类型                             | 描述                          |
| :------------------------- | :------------------------------- | :---------------------------- |
| `hook_event_name`          | `Literal["PermissionRequest"]`   | 始终为 "PermissionRequest"     |
| `tool_name`                | `str`                            | 请求权限的工具名称               |
| `tool_input`               | `dict[str, Any]`                 | 工具的输入参数                  |
| `permission_suggestions`   | `list[Any]`（可选）               | 来自 CLI 的建议权限更新          |

### `HookJSONOutput`

Hook 回调返回值的联合类型。

```python theme={null}
HookJSONOutput = AsyncHookJSONOutput | SyncHookJSONOutput
```

#### `SyncHookJSONOutput`

同步 hook 输出，包含控制和决策字段。

```python theme={null}
class SyncHookJSONOutput(TypedDict):
    # 控制字段
    continue_: NotRequired[bool]  # 是否继续（默认：True）
    suppressOutput: NotRequired[bool]  # 从转录中隐藏 stdout
    stopReason: NotRequired[str]  # continue 为 False 时的消息

    # 决策字段
    decision: NotRequired[Literal["block"]]
    systemMessage: NotRequired[str]  # 给用户的警告消息
    reason: NotRequired[str]  # 给 Claude 的反馈

    # Hook 特定输出
    hookSpecificOutput: NotRequired[HookSpecificOutput]
```

> **注意：** 在 Python 代码中使用 `continue_`（带下划线）。发送到 CLI 时会自动转换为 `continue`。

#### `HookSpecificOutput`

一个 `TypedDict`，包含 hook 事件名称和事件特定字段。形状取决于 `hookEventName` 值。关于每个 hook 事件可用字段的完整细节，请参见[用 Hooks 控制执行](/en/agent-sdk/hooks#outputs)。

事件特定输出类型的可辨识联合。`hookEventName` 字段决定哪些字段有效。

```python theme={null}
class PreToolUseHookSpecificOutput(TypedDict):
    hookEventName: Literal["PreToolUse"]
    permissionDecision: NotRequired[Literal["allow", "deny", "ask", "defer"]]
    permissionDecisionReason: NotRequired[str]
    updatedInput: NotRequired[dict[str, Any]]
    additionalContext: NotRequired[str]


class PostToolUseHookSpecificOutput(TypedDict):
    hookEventName: Literal["PostToolUse"]
    additionalContext: NotRequired[str]
    updatedToolOutput: NotRequired[Any]
    updatedMCPToolOutput: NotRequired[Any]


class PostToolUseFailureHookSpecificOutput(TypedDict):
    hookEventName: Literal["PostToolUseFailure"]
    additionalContext: NotRequired[str]


class UserPromptSubmitHookSpecificOutput(TypedDict):
    hookEventName: Literal["UserPromptSubmit"]
    additionalContext: NotRequired[str]


class NotificationHookSpecificOutput(TypedDict):
    hookEventName: Literal["Notification"]
    additionalContext: NotRequired[str]


class SubagentStartHookSpecificOutput(TypedDict):
    hookEventName: Literal["SubagentStart"]
    additionalContext: NotRequired[str]


class PermissionRequestHookSpecificOutput(TypedDict):
    hookEventName: Literal["PermissionRequest"]
    decision: dict[str, Any]


HookSpecificOutput = (
    PreToolUseHookSpecificOutput
    | PostToolUseHookSpecificOutput
    | PostToolUseFailureHookSpecificOutput
    | UserPromptSubmitHookSpecificOutput
    | NotificationHookSpecificOutput
    | SubagentStartHookSpecificOutput
    | PermissionRequestHookSpecificOutput
)
```

#### `AsyncHookJSONOutput`

延迟 hook 执行的异步 hook 输出。

```python theme={null}
class AsyncHookJSONOutput(TypedDict):
    async_: Literal[True]  # 设置为 True 延迟执行
    asyncTimeout: NotRequired[int]  # 超时时间（毫秒）
```

> **注意：** 在 Python 代码中使用 `async_`（带下划线）。发送到 CLI 时会自动转换为 `async`。

### Hook 使用示例

此示例注册两个 hooks：一个阻止危险的 bash 命令如 `rm -rf /`，另一个记录所有工具使用情况用于审计。安全 hook 仅对 Bash 命令运行（通过 `matcher`），而日志 hook 对所有工具运行。

```python theme={null}
from claude_agent_sdk import query, ClaudeAgentOptions, HookMatcher, HookContext
from typing import Any


async def validate_bash_command(
    input_data: dict[str, Any], tool_use_id: str | None, context: HookContext
) -> dict[str, Any]:
    """验证并可能阻止危险的 bash 命令。"""
    if input_data["tool_name"] == "Bash":
        command = input_data["tool_input"].get("command", "")
        if "rm -rf /" in command:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "已阻止危险命令",
                }
            }
    return {}


async def log_tool_use(
    input_data: dict[str, Any], tool_use_id: str | None, context: HookContext
) -> dict[str, Any]:
    """记录所有工具使用情况用于审计。"""
    print(f"使用工具: {input_data.get('tool_name')}")
    return {}


options = ClaudeAgentOptions(
    hooks={
        "PreToolUse": [
            HookMatcher(
                matcher="Bash", hooks=[validate_bash_command], timeout=120
            ),  # 验证用 2 分钟
            HookMatcher(
                hooks=[log_tool_use]
            ),  # 适用于所有工具（默认 60 秒超时）
        ],
        "PostToolUse": [HookMatcher(hooks=[log_tool_use])],
    }
)

async for message in query(prompt="分析这个代码库", options=options):
    print(message)
```

## 工具输入/输出类型

所有内置 Claude Code 工具的输入/输出 schema 文档。虽然 Python SDK 不将这些作为类型导出，但它们代表了消息中工具输入和输出的结构。

### Agent

**工具名称：** `Agent`（以前叫 `Task`，仍作为别名接受）

**输入：**

```python theme={null}
{
    "description": str,  # 任务的简短（3-5 个词）描述
    "prompt": str,  # 代理要执行的任务
    "subagent_type": str,  # 要使用的专用代理类型
}
```

**输出：**

```python theme={null}
{
    "result": str,  # 子代理的最终结果
    "usage": dict | None,  # Token 使用统计
    "total_cost_usd": float | None,  # 估算的总成本（USD）
    "duration_ms": int | None,  # 执行持续时间（毫秒）
}
```

### AskUserQuestion

**工具名称：** `AskUserQuestion`

在执行期间向用户提出澄清性问题。使用详情见[处理批准和用户输入](/en/agent-sdk/user-input#handle-clarifying-questions)。

**输入：**

```python theme={null}
{
    "questions": [  # 向用户提出的问题（1-4 个）
        {
            "question": str,  # 完整的问题
            "header": str,  # 显示为标签的简短标签（最多 12 个字符）
            "options": [  # 可用选项（2-4 个）
                {
                    "label": str,  # 选项的显示文本（1-5 个词）
                    "description": str,  # 选项含义的解释
                }
            ],
            "multiSelect": bool,  # 设置为 true 允许多选
        }
    ],
    "answers": dict[str, str | list[str]] | None,
    # 权限系统填充的用户答案。多选答案可能是标签列表或逗号连接的字符串
}
```

**输出：**

```python theme={null}
{
    "questions": [  # 被问及的问题
        {
            "question": str,
            "header": str,
            "options": [{"label": str, "description": str}],
            "multiSelect": bool,
        }
    ],
    "answers": dict[str, str],  # 映射问题文本到答案字符串
    # 多选答案以逗号分隔
}
```

### Bash

**工具名称：** `Bash`

**输入：**

```python theme={null}
{
    "command": str,  # 要执行的命令
    "timeout": int | None,  # 可选超时（毫秒，最大 600000）
    "description": str | None,  # 清晰简洁的描述（5-10 个词）
    "run_in_background": bool | None,  # 设置为 true 在后台运行
}
```

**输出：**

```python theme={null}
{
    "output": str,  # stdout 和 stderr 的合并输出
    "exitCode": int,  # 命令的退出码
    "killed": bool | None,  # 命令是否因超时被杀死
    "shellId": str | None,  # 后台进程的 Shell ID
}
```

### Monitor

**工具名称：** `Monitor`

运行后台脚本并将每个 stdout 行作为事件传递给 Claude，使其无需轮询即可做出反应。Monitor 遵循与 Bash 相同的权限规则。行为和服务提供商可用性见 [Monitor 工具参考](/en/tools-reference#monitor-tool)。

**输入：**

```python theme={null}
{
    "command": str,  # Shell 脚本；每个 stdout 行是一个事件，退出结束监视
    "description": str,  # 通知中显示的简短描述
    "timeout_ms": int | None,  # 在此截止时间后杀死（默认 300000，最大 3600000）
    "persistent": bool | None,  # 会话生命周期内运行；用 TaskStop 停止
}
```

**输出：**

```python theme={null}
{
    "taskId": str,  # 后台监视任务的 ID
    "timeoutMs": int,  # 超时截止时间（毫秒），persistent 时为 0
    "persistent": bool | None,  # 当运行直到 TaskStop 或会话结束时为 True
}
```

### Edit

**工具名称：** `Edit`

**输入：**

```python theme={null}
{
    "file_path": str,  # 要修改的文件的绝对路径
    "old_string": str,  # 要替换的文本
    "new_string": str,  # 替换文本
    "replace_all": bool | None,  # 替换所有出现（默认 False）
}
```

**输出：**

```python theme={null}
{
    "message": str,  # 确认消息
    "replacements": int,  # 替换次数
    "file_path": str,  # 被编辑的文件路径
}
```

### Read

**工具名称：** `Read`

**输入：**

```python theme={null}
{
    "file_path": str,  # 要读取的文件的绝对路径
    "offset": int | None,  # 从第几行开始读取
    "limit": int | None,  # 要读取的行数
}
```

**输出（文本文件）：**

```python theme={null}
{
    "content": str,  # 带行号的文件内容
    "total_lines": int,  # 文件的总行数
    "lines_returned": int,  # 实际返回的行数
}
```

**输出（图片）：**

```python theme={null}
{
    "image": str,  # Base64 编码的图片数据
    "mime_type": str,  # 图片 MIME 类型
    "file_size": int,  # 文件大小（字节）
}
```

### Write

**工具名称：** `Write`

**输入：**

```python theme={null}
{
    "file_path": str,  # 要写入的文件的绝对路径
    "content": str,  # 要写入文件的内容
}
```

**输出：**

```python theme={null}
{
    "message": str,  # 成功消息
    "bytes_written": int,  # 写入的字节数
    "file_path": str,  # 被写入的文件路径
}
```

### Glob

**工具名称：** `Glob`

**输入：**

```python theme={null}
{
    "pattern": str,  # 用于匹配文件的 glob 模式
    "path": str | None,  # 要搜索的目录（默认为 cwd）
}
```

**输出：**

```python theme={null}
{
    "matches": list[str],  # 匹配文件路径的数组
    "count": int,  # 找到的匹配数
    "search_path": str,  # 使用的搜索目录
}
```

### Grep

**工具名称：** `Grep`

**输入：**

```python theme={null}
{
    "pattern": str,  # 正则表达式模式
    "path": str | None,  # 要搜索的文件或目录
    "glob": str | None,  # 用于过滤文件的 glob 模式
    "type": str | None,  # 要搜索的文件类型
    "output_mode": str | None,  # "content"、"files_with_matches" 或 "count"
    "-i": bool | None,  # 不区分大小写搜索
    "-n": bool | None,  # 显示行号
    "-B": int | None,  # 每个匹配前显示的行数
    "-A": int | None,  # 每个匹配后显示的行数
    "-C": int | None,  # 每个匹配前后显示的行数
    "head_limit": int | None,  # 限制输出到前 N 行/条
    "multiline": bool | None,  # 启用多行模式
}
```

**输出（content 模式）：**

```python theme={null}
{
    "matches": [
        {
            "file": str,
            "line_number": int | None,
            "line": str,
            "before_context": list[str] | None,
            "after_context": list[str] | None,
        }
    ],
    "total_matches": int,
}
```

**输出（files_with_matches 模式）：**

```python theme={null}
{
    "files": list[str],  # 包含匹配的文件
    "count": int,  # 有匹配的文件数
}
```

### NotebookEdit

**工具名称：** `NotebookEdit`

**输入：**

```python theme={null}
{
    "notebook_path": str,  # Jupyter notebook 的绝对路径
    "cell_id": str | None,  # 要编辑的单元格 ID
    "new_source": str,  # 单元格的新源代码
    "cell_type": "code" | "markdown" | None,  # 单元格类型
    "edit_mode": "replace" | "insert" | "delete" | None,  # 编辑操作类型
}
```

**输出：**

```python theme={null}
{
    "message": str,  # 成功消息
    "edit_type": "replaced" | "inserted" | "deleted",  # 执行的编辑类型
    "cell_id": str | None,  # 受影响的单元格 ID
    "total_cells": int,  # 编辑后 notebook 中的总单元格数
}
```

### WebFetch

**工具名称：** `WebFetch`

**输入：**

```python theme={null}
{
    "url": str,  # 要获取内容的 URL
    "prompt": str,  # 对获取的内容运行的提示
}
```

**输出：**

```python theme={null}
{
    "bytes": int,  # 获取的内容大小（字节）
    "code": int,  # HTTP 响应码
    "codeText": str,  # HTTP 响应码文本
    "result": str,  # 对内容应用提示后的处理结果
    "durationMs": int,  # 获取和处理内容的时间（毫秒）
    "url": str,  # 被获取的 URL
}
```

### WebSearch

**工具名称：** `WebSearch`

**输入：**

```python theme={null}
{
    "query": str,  # 要使用的搜索查询
    "allowed_domains": list[str] | None,  # 仅包含来自这些域的结果
    "blocked_domains": list[str] | None,  # 绝不包含来自这些域的结果
}
```

**输出：**

```python theme={null}
{
    "query": str,  # 搜索查询
    "results": list[str | {"tool_use_id": str, "content": list[{"title": str, "url": str}]}],
    "durationSeconds": float,  # 搜索持续时间（秒）
}
```

### TodoWrite

**工具名称：** `TodoWrite`

> **注意：** `TodoWrite` 已弃用，将在未来版本中移除。请改用 `TaskCreate`、`TaskGet`、`TaskUpdate` 和 `TaskList`。设置 `CLAUDE_CODE_ENABLE_TASKS=1` 来启用。监控代码如何更改见[迁移到 Task 工具](/en/agent-sdk/todo-tracking#migrate-to-task-tools)。

**输入：**

```python theme={null}
{
    "todos": [
        {
            "content": str,  # 任务描述
            "status": "pending" | "in_progress" | "completed",  # 任务状态
            "activeForm": str,  # 描述的主动形式
        }
    ]
}
```

**输出：**

```python theme={null}
{
    "message": str,  # 成功消息
    "stats": {"total": int, "pending": int, "in_progress": int, "completed": int},
}
```

### TaskCreate

**工具名称：** `TaskCreate`

**输入：**

```python theme={null}
{
    "subject": str,  # 简短任务标题
    "description": str,  # 详细任务正文
    "activeForm": str | None,  # 进行中时显示的现在时标签
    "metadata": dict | None,  # 任意调用者元数据
}
```

**输出：**

```python theme={null}
{
    "task": {"id": str, "subject": str},  # 创建的任务及分配的 ID
}
```

### TaskUpdate

**工具名称：** `TaskUpdate`

**输入：**

```python theme={null}
{
    "taskId": str,  # 要修补的任务 ID
    "status": Literal["pending", "in_progress", "completed", "deleted"] | None,
    "subject": str | None,
    "description": str | None,
    "activeForm": str | None,
    "addBlocks": list[str] | None,  # 此任务现在阻塞的任务 ID
    "addBlockedBy": list[str] | None,  # 现在阻塞此任务的任务 ID
    "owner": str | None,
    "metadata": dict | None,
}
```

**输出：**

```python theme={null}
{
    "success": bool,
    "taskId": str,
    "updatedFields": list[str],  # 已更改的字段名称
    "error": str | None,
    "statusChange": {"from": str, "to": str} | None,
}
```

### TaskGet

**工具名称：** `TaskGet`

**输入：**

```python theme={null}
{
    "taskId": str,  # 要读取的任务 ID
}
```

**输出：**

```python theme={null}
{
    "task": {
        "id": str,
        "subject": str,
        "description": str,
        "status": Literal["pending", "in_progress", "completed"],
        "blocks": list[str],
        "blockedBy": list[str],
    } | None,  # 未找到 ID 时为 None
}
```

### TaskList

**工具名称：** `TaskList`

**输入：**

```python theme={null}
{}
```

**输出：**

```python theme={null}
{
    "tasks": [
        {
            "id": str,
            "subject": str,
            "status": Literal["pending", "in_progress", "completed"],
            "owner": str | None,
            "blockedBy": list[str],
        }
    ],
}
```

### BashOutput

**工具名称：** `BashOutput`

**输入：**

```python theme={null}
{
    "bash_id": str,  # 后台 shell 的 ID
    "filter": str | None,  # 用于过滤输出行的可选正则表达式
}
```

**输出：**

```python theme={null}
{
    "output": str,  # 自上次检查以来的新输出
    "status": "running" | "completed" | "failed",  # 当前 shell 状态
    "exitCode": int | None,  # 完成时的退出码
}
```

### KillBash

**工具名称：** `KillBash`

**输入：**

```python theme={null}
{
    "shell_id": str  # 要杀死的后台 shell 的 ID
}
```

**输出：**

```python theme={null}
{
    "message": str,  # 成功消息
    "shell_id": str,  # 被杀死的 shell 的 ID
}
```

### ExitPlanMode

**工具名称：** `ExitPlanMode`

**输入：**

```python theme={null}
{
    "plan": str  # 要运行以供用户批准的计划
}
```

**输出：**

```python theme={null}
{
    "message": str,  # 确认消息
    "approved": bool | None,  # 用户是否批准了计划
}
```

### ListMcpResources

**工具名称：** `ListMcpResources`

**输入：**

```python theme={null}
{
    "server": str | None  # 可选的服务器名称，用于过滤资源
}
```

**输出：**

```python theme={null}
{
    "resources": [
        {
            "uri": str,
            "name": str,
            "description": str | None,
            "mimeType": str | None,
            "server": str,
        }
    ],
    "total": int,
}
```

### ReadMcpResource

**工具名称：** `ReadMcpResource`

**输入：**

```python theme={null}
{
    "server": str,  # MCP 服务器名称
    "uri": str,  # 要读取的资源 URI
}
```

**输出：**

```python theme={null}
{
    "contents": [
        {"uri": str, "mimeType": str | None, "text": str | None, "blob": str | None}
    ],
    "server": str,
}
```

## ClaudeSDKClient 高级特性

### 构建持续对话界面

```python theme={null}
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
)
import asyncio


class ConversationSession:
    """维护与 Claude 的单个对话会话。"""

    def __init__(self, options: ClaudeAgentOptions | None = None):
        self.client = ClaudeSDKClient(options)
        self.turn_count = 0

    async def start(self):
        await self.client.connect()
        print("开始对话会话。Claude 会记住上下文。")
        print(
            "命令：'exit' 退出，'interrupt' 停止当前任务，'new' 开始新会话"
        )

        while True:
            user_input = input(f"\n[第 {self.turn_count + 1} 轮] 你: ")

            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "interrupt":
                await self.client.interrupt()
                print("任务已中断！")
                continue
            elif user_input.lower() == "new":
                # 断开并重新连接以获取全新会话
                await self.client.disconnect()
                await self.client.connect()
                self.turn_count = 0
                print("开始新的对话会话（之前的上下文已清除）")
                continue

            # 发送消息 — 会话保留所有先前的消息
            await self.client.query(user_input)
            self.turn_count += 1

            # 处理响应
            print(f"[第 {self.turn_count} 轮] Claude: ", end="")
            async for message in self.client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(block.text, end="")
            print()

        await self.client.disconnect()
        print(f"对话在 {self.turn_count} 轮后结束。")


async def main():
    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Write", "Bash"], permission_mode="acceptEdits"
    )
    session = ConversationSession(options)
    await session.start()


# 示例对话：
# 第 1 轮 - 你: "创建一个叫 hello.py 的文件"
# 第 1 轮 - Claude: "我将为你创建 hello.py 文件..."
# 第 2 轮 - 你: "那个文件里有什么？"
# 第 2 轮 - Claude: "我刚刚创建的 hello.py 文件包含..."（记得！）
# 第 3 轮 - 你: "给它加一个 main 函数"
# 第 3 轮 - Claude: "我将向 hello.py 添加 main 函数..."（知道是哪个文件！）

asyncio.run(main())
```

### 使用 Hooks 修改行为

```python theme={null}
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    HookMatcher,
    HookContext,
)
import asyncio
from typing import Any


async def pre_tool_logger(
    input_data: dict[str, Any], tool_use_id: str | None, context: HookContext
) -> dict[str, Any]:
    """在执行前记录所有工具使用。"""
    tool_name = input_data.get("tool_name", "未知")
    print(f"[PRE-TOOL] 即将使用: {tool_name}")

    # 你可以在此修改或阻止工具执行
    if tool_name == "Bash" and "rm -rf" in str(input_data.get("tool_input", {})):
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "已阻止危险命令",
            }
        }
    return {}


async def post_tool_logger(
    input_data: dict[str, Any], tool_use_id: str | None, context: HookContext
) -> dict[str, Any]:
    """在工具执行后记录结果。"""
    tool_name = input_data.get("tool_name", "未知")
    print(f"[POST-TOOL] 完成: {tool_name}")
    return {}


async def user_prompt_modifier(
    input_data: dict[str, Any], tool_use_id: str | None, context: HookContext
) -> dict[str, Any]:
    """向用户提示添加上下文。"""
    original_prompt = input_data.get("prompt", "")

    # 添加时间戳作为 Claude 可见的附加上下文
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": f"[提交于 {timestamp}] 原始提示: {original_prompt}",
        }
    }


async def main():
    options = ClaudeAgentOptions(
        hooks={
            "PreToolUse": [
                HookMatcher(hooks=[pre_tool_logger]),
                HookMatcher(matcher="Bash", hooks=[pre_tool_logger]),
            ],
            "PostToolUse": [HookMatcher(hooks=[post_tool_logger])],
            "UserPromptSubmit": [HookMatcher(hooks=[user_prompt_modifier])],
        },
        allowed_tools=["Read", "Write", "Bash"],
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query("列出当前目录中的文件")

        async for message in client.receive_response():
            # Hooks 会自动记录工具使用
            pass


asyncio.run(main())
```

### 实时进度监控

```python theme={null}
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    ToolUseBlock,
    ToolResultBlock,
    TextBlock,
)
import asyncio


async def monitor_progress():
    options = ClaudeAgentOptions(
        allowed_tools=["Write", "Bash"], permission_mode="acceptEdits"
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query("创建 5 个包含不同排序算法的 Python 文件")

        # 实时监控进度
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        if block.name == "Write":
                            file_path = block.input.get("file_path", "")
                            print(f"正在创建: {file_path}")
                    elif isinstance(block, ToolResultBlock):
                        print("工具执行完成")
                    elif isinstance(block, TextBlock):
                        print(f"Claude 说: {block.text[:100]}...")

        print("任务完成！")


asyncio.run(monitor_progress())
```

## 示例用法

### 基本文件操作（使用 query）

```python theme={null}
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ToolUseBlock
import asyncio


async def create_project():
    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Write", "Bash"],
        permission_mode="acceptEdits",
        cwd="/home/user/project",
    )

    async for message in query(
        prompt="创建一个包含 setup.py 的 Python 项目结构", options=options
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    print(f"使用工具: {block.name}")


asyncio.run(create_project())
```

### 错误处理

```python theme={null}
from claude_agent_sdk import query, CLINotFoundError, ProcessError, CLIJSONDecodeError

try:
    async for message in query(prompt="你好"):
        print(message)
except CLINotFoundError:
    print(
        "未找到 Claude Code CLI。尝试重新安装：pip install --force-reinstall claude-agent-sdk"
    )
except ProcessError as e:
    print(f"进程失败，退出码: {e.exit_code}")
except CLIJSONDecodeError as e:
    print(f"解析响应失败: {e}")
```

### 使用 Client 的流式模式

```python theme={null}
from claude_agent_sdk import ClaudeSDKClient
import asyncio


async def interactive_session():
    async with ClaudeSDKClient() as client:
        # 发送初始消息
        await client.query("天气怎么样？")

        # 处理响应
        async for msg in client.receive_response():
            print(msg)

        # 发送追问
        await client.query("再给我详细说说")

        # 处理追问响应
        async for msg in client.receive_response():
            print(msg)


asyncio.run(interactive_session())
```

### 使用 ClaudeSDKClient 的自定义工具

```python theme={null}
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    tool,
    create_sdk_mcp_server,
    AssistantMessage,
    TextBlock,
)
import asyncio
from typing import Any


# 使用 @tool 装饰器定义自定义工具
@tool("calculate", "执行数学计算", {"expression": str})
async def calculate(args: dict[str, Any]) -> dict[str, Any]:
    try:
        result = eval(args["expression"], {"__builtins__": {}})
        return {"content": [{"type": "text", "text": f"结果: {result}"}]}
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"错误: {str(e)}"}],
            "is_error": True,
        }


@tool("get_time", "获取当前时间", {})
async def get_time(args: dict[str, Any]) -> dict[str, Any]:
    from datetime import datetime

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {"content": [{"type": "text", "text": f"当前时间: {current_time}"}]}


async def main():
    # 使用自定义工具创建 SDK MCP 服务器
    my_server = create_sdk_mcp_server(
        name="utilities", version="1.0.0", tools=[calculate, get_time]
    )

    # 使用该服务器配置选项
    options = ClaudeAgentOptions(
        mcp_servers={"utils": my_server},
        allowed_tools=["mcp__utils__calculate", "mcp__utils__get_time"],
    )

    # 使用 ClaudeSDKClient 进行交互式工具使用
    async with ClaudeSDKClient(options=options) as client:
        await client.query("123 * 456 等于多少？")

        # 处理计算响应
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"计算结果: {block.text}")

        # 追问时间查询
        await client.query("现在几点了？")

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"时间: {block.text}")


asyncio.run(main())
```

## 沙箱配置

### `SandboxSettings`

沙箱行为的配置。用于以编程方式启用命令沙箱和配置网络限制。

```python theme={null}
class SandboxSettings(TypedDict, total=False):
    enabled: bool
    autoAllowBashIfSandboxed: bool
    excludedCommands: list[str]
    allowUnsandboxedCommands: bool
    network: SandboxNetworkConfig
    ignoreViolations: SandboxIgnoreViolations
    enableWeakerNestedSandbox: bool
```

| 属性                          | 类型                                                    | 默认值   | 描述                                                                                                                                                                        |
| :---------------------------- | :------------------------------------------------------ | :------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `enabled`                     | `bool`                                                  | `False`  | 启用命令执行的沙箱模式                                                                                                                                                        |
| `autoAllowBashIfSandboxed`    | `bool`                                                  | `True`   | 沙箱启用时自动批准 bash 命令                                                                                                                                                  |
| `excludedCommands`            | `list[str]`                                             | `[]`     | 始终绕过沙箱限制的命令（如 `["docker"]`）。这些命令自动无沙箱运行，无需模型参与                                                                                                |
| `allowUnsandboxedCommands`    | `bool`                                                  | `True`   | 允许模型请求在沙箱外运行命令。为 `True` 时，模型可以在工具输入中设置 `dangerouslyDisableSandbox`，这会回退到[权限系统](#无沙箱命令的权限回退)                                     |
| `network`                     | [`SandboxNetworkConfig`](#sandboxnetworkconfig)         | `None`   | 网络特定的沙箱配置                                                                                                                                                            |
| `ignoreViolations`            | [`SandboxIgnoreViolations`](#sandboxignoreviolations)   | `None`   | 配置要忽略的沙箱违规                                                                                                                                                          |
| `enableWeakerNestedSandbox`   | `bool`                                                  | `False`  | 启用较弱的嵌套沙箱以增强兼容性                                                                                                                                                  |

#### 示例用法

```python theme={null}
from claude_agent_sdk import query, ClaudeAgentOptions, SandboxSettings

sandbox_settings: SandboxSettings = {
    "enabled": True,
    "autoAllowBashIfSandboxed": True,
    "network": {"allowLocalBinding": True},
}

async for message in query(
    prompt="构建并测试我的项目",
    options=ClaudeAgentOptions(sandbox=sandbox_settings),
):
    print(message)
```

> **警告：** **Unix 套接字安全性**：`allowUnixSockets` 选项可能授予对强大系统服务的访问权限。例如，允许 `/var/run/docker.sock` 实际上通过 Docker API 授予了完整的主机系统访问权限，绕过了沙箱隔离。只允许严格必要的 Unix 套接字，并理解每个套接字的安全影响。

### `SandboxNetworkConfig`

沙箱模式的网络特定配置。

```python theme={null}
class SandboxNetworkConfig(TypedDict, total=False):
    allowedDomains: list[str]
    deniedDomains: list[str]
    allowManagedDomainsOnly: bool
    allowUnixSockets: list[str]
    allowAllUnixSockets: bool
    allowLocalBinding: bool
    allowMachLookup: list[str]
    httpProxyPort: int
    socksProxyPort: int
```

| 属性                        | 类型          | 默认值   | 描述                                                                                                                       |
| :-------------------------- | :------------ | :------- | :------------------------------------------------------------------------------------------------------------------------- |
| `allowedDomains`            | `list[str]`   | `[]`     | 沙箱进程可以访问的域名                                                                                                      |
| `deniedDomains`             | `list[str]`   | `[]`     | 沙箱进程不能访问的域名。优先于 `allowedDomains`                                                                              |
| `allowManagedDomainsOnly`   | `bool`        | `False`  | 仅托管设置：在托管设置中设置时，忽略非托管设置源中的 `allowedDomains`。通过 SDK 选项设置时无效                                 |
| `allowUnixSockets`          | `list[str]`   | `[]`     | 进程可以访问的 Unix 套接字路径（如 Docker 套接字）                                                                            |
| `allowAllUnixSockets`       | `bool`        | `False`  | 允许访问所有 Unix 套接字                                                                                                     |
| `allowLocalBinding`         | `bool`        | `False`  | 允许进程绑定到本地端口（如用于开发服务器）                                                                                    |
| `allowMachLookup`           | `list[str]`   | `[]`     | 仅 macOS：允许的 XPC/Mach 服务名称。支持尾部通配符                                                                           |
| `httpProxyPort`             | `int`         | `None`   | 网络请求的 HTTP 代理端口                                                                                                     |
| `socksProxyPort`            | `int`         | `None`   | 网络请求的 SOCKS 代理端口                                                                                                    |

> **注意：** 内置沙箱代理基于请求的主机名强制执行网络允许列表，不会终止或检查 TLS 流量，因此诸如[域名前置](https://en.wikipedia.org/wiki/Domain_fronting)等技术可能绕过它。详情见[沙箱安全限制](/en/sandboxing#security-limitations)，配置 TLS 终止代理见[安全部署](/en/agent-sdk/secure-deployment#traffic-forwarding)。

### `SandboxIgnoreViolations`

忽略特定沙箱违规的配置。

```python theme={null}
class SandboxIgnoreViolations(TypedDict, total=False):
    file: list[str]
    network: list[str]
```

| 属性       | 类型          | 默认值   | 描述                           |
| :--------- | :------------ | :------- | :----------------------------- |
| `file`     | `list[str]`   | `[]`     | 忽略违规的文件路径模式            |
| `network`  | `list[str]`   | `[]`     | 忽略违规的网络模式               |

### 无沙箱命令的权限回退

当 `allowUnsandboxedCommands` 启用时，模型可以通过在工具输入中设置 `dangerouslyDisableSandbox: True` 来请求在沙箱外运行命令。这些请求回退到现有的权限系统，这意味着你的 `can_use_tool` 处理程序会被调用，允许你实现自定义授权逻辑。

> **注意：** **`excludedCommands` vs `allowUnsandboxedCommands`：**
>
> * `excludedCommands`：始终自动绕过沙箱的静态命令列表（如 `["docker"]`）。模型对此没有控制权。
> * `allowUnsandboxedCommands`：让模型在运行时决定是否通过设置工具输入中的 `dangerouslyDisableSandbox: True` 来请求无沙箱执行。

```python theme={null}
from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)


async def can_use_tool(
    tool: str, input: dict, context: ToolPermissionContext
) -> PermissionResultAllow | PermissionResultDeny:
    # 检查模型是否请求绕过沙箱
    if tool == "Bash" and input.get("dangerouslyDisableSandbox"):
        # 模型正在请求在沙箱外运行此命令
        print(f"请求无沙箱命令: {input.get('command')}")

        if is_command_authorized(input.get("command")):
            return PermissionResultAllow()
        return PermissionResultDeny(
            message="命令未授权无沙箱执行"
        )
    return PermissionResultAllow()


# 必需：虚拟 hook 保持流为 can_use_tool 打开
async def dummy_hook(input_data, tool_use_id, context):
    return {"continue_": True}


async def prompt_stream():
    yield {
        "type": "user",
        "message": {"role": "user", "content": "部署我的应用"},
    }


async def main():
    async for message in query(
        prompt=prompt_stream(),
        options=ClaudeAgentOptions(
            sandbox={
                "enabled": True,
                "allowUnsandboxedCommands": True,  # 模型可以请求无沙箱执行
            },
            permission_mode="default",
            can_use_tool=can_use_tool,
            hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[dummy_hook])]},
        ),
    ):
        print(message)
```

此模式使你能够：

* **审计模型请求**：记录模型请求无沙箱执行的情况
* **实现允许列表**：仅允许特定命令无沙箱运行
* **添加批准工作流**：要求特权操作的显式授权

> **警告：** 使用 `dangerouslyDisableSandbox: True` 运行的命令拥有完整的系统访问权限。确保你的 `can_use_tool` 处理程序仔细验证这些请求。
>
> 如果 `permission_mode` 设置为 `bypassPermissions` 并且 `allow_unsandboxed_commands` 启用，模型可以在没有任何批准提示的情况下自主在沙箱外执行命令。这种组合实际上允许模型悄无声息地逃脱沙箱隔离。

## 参见

* [SDK 概述](/en/agent-sdk/overview) — 通用 SDK 概念
* [TypeScript SDK 参考](/en/agent-sdk/typescript) — TypeScript SDK 文档
* [CLI 参考](/en/cli-reference) — 命令行界面
* [常见工作流](/en/common-workflows) — 逐步指南
