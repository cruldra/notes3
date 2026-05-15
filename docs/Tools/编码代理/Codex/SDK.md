---
sidebar_position: 3
title: SDK
---


> 如果你通过 Codex CLI、IDE 扩展或 Codex Web 使用 Codex，你也可以用编程方式控制它。

在以下场景中使用 SDK：

- 将 Codex 作为 CI/CD 流水线的一部分进行控制
- 创建你自己的代理，与 Codex 协作执行复杂的工程任务
- 将 Codex 构建到你自己的内部工具和工作流中
- 将 Codex 集成到你自己的应用中

## TypeScript 库

TypeScript 库提供了一种从你的应用中控制 Codex 的方式，比非交互模式更全面、更灵活。

在服务端使用该库；需要 Node.js 18 或更高版本。

### 安装

使用 `npm` 安装 Codex SDK 以开始使用：

```bash
npm install @openai/codex-sdk
```

### 用法

启动一个与 Codex 的 thread 并用你的提示运行它。

```ts
const codex = new Codex();
const thread = codex.startThread();
const result = await thread.run(
  "制定一个诊断和修复 CI 故障的计划"
);

console.log(result);
```

再次调用 `run()` 在同一 thread 上继续，或通过提供 thread ID 恢复之前的 thread。

```ts
// 运行同一个 thread
const result = await thread.run("执行这个计划");

console.log(result);

// 恢复之前的 thread

const threadId = "<thread-id>";
const thread2 = codex.resumeThread(threadId);
const result2 = await thread2.run("从你中断的地方继续");

console.log(result2);
```

更多详情参见 [TypeScript 仓库](https://github.com/openai/codex/tree/main/sdk/typescript)。

## Python 库

Python SDK 是实验性的，通过 JSON-RPC 控制本地的 Codex app-server。需要 Python 3.10 或更高版本，以及开源 Codex 仓库的本地检出。

### 安装

从 Codex 仓库根目录，以可编辑模式安装 SDK：

```bash
cd sdk/python
python -m pip install -e .
```

对于手动本地 SDK 使用，传入 `AppServerConfig(codex_bin=...)` 指向本地的 `codex` 二进制文件，或使用仓库示例和 notebook 引导。

### 用法

启动 Codex，创建一个 thread，并运行提示：

```python
from codex_app_server import Codex

with Codex() as codex:
    thread = codex.thread_start(model="gpt-5.4")
    result = thread.run("制定一个诊断和修复 CI 故障的计划")
    print(result.final_response)
```

当你的应用已经是异步时使用 `AsyncCodex`：

```python
import asyncio

from codex_app_server import AsyncCodex


async def main() -> None:
    async with AsyncCodex() as codex:
        thread = await codex.thread_start(model="gpt-5.4")
        result = await thread.run("执行这个计划")
        print(result.final_response)


asyncio.run(main())
```

更多详情参见 [Python 仓库](https://github.com/openai/codex/tree/main/sdk/python)。
