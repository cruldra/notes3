---
sidebar_position: 3
---

[模型上下文协议 (Model Context Protocol, MCP)](https://modelcontextprotocol.io/) 是一个开放协议，旨在标准化应用程序向语言模型提供上下文的方式。DSPy 支持 MCP，允许你在 DSPy 智能体中使用来自任何 MCP 服务器的工具。

## 安装

安装支持 MCP 的 DSPy：

```bash
pip install -U "dspy[mcp]"
```

## 概览

MCP 使你能够：

- **使用标准化工具** - 连接到任何兼容 MCP 的服务器。
- **跨堆栈共享工具** - 在不同框架之间使用相同的工具。
- **简化集成** - 一行代码即可将 MCP 工具转换为 DSPy 工具。

DSPy 不直接处理 MCP 服务器连接。你可以使用 `mcp` 库的客户端接口建立连接，并将 `mcp.ClientSession` 传递给 `dspy.Tool.from_mcp_tool`，以便将 mcp 工具转换为 DSPy 工具。

## 在 DSPy 中使用 MCP

### 1. HTTP 服务器 (远程)

对于通过 HTTP 的远程 MCP 服务器，请使用可流式 HTTP 传输：

```python
import asyncio
import dspy
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main():
    # 连接到 HTTP MCP 服务器
    async with streamablehttp_client("http://localhost:8000/mcp") as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化会话
            await session.initialize()

            # 列出并转换工具
            response = await session.list_tools()
            dspy_tools = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in response.tools
            ]

            # 创建并使用 ReAct 智能体
            class TaskSignature(dspy.Signature):
                task: str = dspy.InputField()
                result: str = dspy.OutputField()

            react_agent = dspy.ReAct(
                signature=TaskSignature,
                tools=dspy_tools,
                max_iters=5
            )

            result = await react_agent.acall(task="Check the weather in Tokyo")
            print(result.result)

asyncio.run(main())
```

### 2. Stdio 服务器 (本地进程)

使用 MCP 最常见的方式是使用通过 stdio 通信的本地服务器进程：

```python
import asyncio
import dspy
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # 配置 stdio 服务器
    server_params = StdioServerParameters(
        command="python",                    # 要运行的命令
        args=["path/to/your/mcp_server.py"], # 服务器脚本路径
        env=None,                            # 可选的环境变量
    )

    # 连接到服务器
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化会话
            await session.initialize()

            # 列出可用工具
            response = await session.list_tools()

            # 将 MCP 工具转换为 DSPy 工具
            dspy_tools = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in response.tools
            ]

            # 创建一个带有工具的 ReAct 智能体
            class QuestionAnswer(dspy.Signature):
                """Answer questions using available tools."""
                question: str = dspy.InputField()
                answer: str = dspy.OutputField()

            react_agent = dspy.ReAct(
                signature=QuestionAnswer,
                tools=dspy_tools,
                max_iters=5
            )

            # 使用智能体
            result = await react_agent.acall(
                question="What is 25 + 17?"
            )
            print(result.answer)

# 运行异步函数
asyncio.run(main())
```

## 工具转换

DSPy 自动处理从 MCP 工具到 DSPy 工具的转换：

```python
# 来自会话的 MCP 工具
mcp_tool = response.tools[0]

# 转换为 DSPy 工具
dspy_tool = dspy.Tool.from_mcp_tool(session, mcp_tool)

# DSPy 工具保留：
# - 工具名称和描述
# - 参数模式和类型
# - 参数描述
# - 异步执行支持

# 像任何 DSPy 工具一样使用它
result = await dspy_tool.acall(param1="value", param2=123)
```

## 了解更多

- [MCP 官方文档](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [DSPy MCP 教程](https://dspy.ai/tutorials/mcp/)
- DSPy 工具文档

DSPy 中的 MCP 集成使得使用来自任何 MCP 服务器的标准化工具变得容易，从而通过最少的设置实现强大的智能体功能。