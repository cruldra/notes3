---
sidebar_position: 2
title: App Server
---

> `codex app-server` 是 Codex 为富客户端（如 Codex VS Code 扩展）提供支持的接口。适用于需要深度集成到自己产品中的场景：认证、对话历史、审批和流式代理事件。

app-server 实现在 Codex GitHub 仓库中开源（[openai/codex/codex-rs/app-server](https://github.com/openai/codex/tree/main/codex-rs/app-server)）。完整的开源 Codex 组件列表参见[开源](https://developers.openai.com/codex/open-source)页面。

如果你在做自动化任务或在 CI 中运行 Codex，请改用
  <a href="/codex/sdk">Codex SDK</a>。

## 协议

与 [MCP](https://modelcontextprotocol.io/) 类似，`codex app-server` 使用 JSON-RPC 2.0 消息支持双向通信（传输时省略 `"jsonrpc":"2.0"` 头部）。

支持的传输层：

- `stdio`（`--listen stdio://`，默认）：换行分隔的 JSON（JSONL）。
- `websocket`（`--listen ws://IP:PORT`，实验性且不受支持）：每个 WebSocket 文本帧一条 JSON-RPC 消息。
- `off`（`--listen off`）：不暴露本地传输层。

使用 `--listen ws://IP:PORT` 运行时，同一监听器也会提供基本的 HTTP 健康探测：

- `GET /readyz` 在监听器接受新连接后返回 `200 OK`。
- `GET /healthz` 在请求不包含 `Origin` 头部时返回 `200 OK`。
- 带有 `Origin` 头部的请求被拒绝，返回 `403 Forbidden`。

WebSocket 传输是实验性的且不受支持。回环监听器如 `ws://127.0.0.1:PORT` 适用于 localhost 和 SSH 端口转发工作流。非回环 WebSocket 监听器在推广期间默认允许未经认证的连接，因此在远程暴露之前请配置 WebSocket 认证。

支持的 WebSocket 认证标志：

- `--ws-auth capability-token --ws-token-file /absolute/path`
- `--ws-auth capability-token --ws-token-sha256 HEX`
- `--ws-auth signed-bearer-token --ws-shared-secret-file /absolute/path`

对于签名 bearer token，你还可以设置 `--ws-issuer`、`--ws-audience` 和 `--ws-max-clock-skew-seconds`。客户端在 WebSocket 握手期间以 `Authorization: Bearer <token>` 的形式提交凭证，app-server 在 JSON-RPC `initialize` 之前执行认证。

优先使用 `--ws-token-file` 而非在命令行上传递原始 bearer token。只有当客户端将原始高熵 token 保存在单独的本地密钥存储中时，才使用 `--ws-token-sha256`；哈希仅是验证器，客户端仍然需要原始 token。

在 WebSocket 模式下，app-server 使用有界队列。当请求入口已满时，服务器拒绝新请求，返回 JSON-RPC 错误码 `-32001` 和消息 `"Server overloaded; retry later."`。客户端应使用指数递增延迟和抖动进行重试。

## 消息模式

请求包含 `method`、`params` 和 `id`：

```json
{ "method": "thread/start", "id": 10, "params": { "model": "gpt-5.4" } }
```

响应回显 `id`，附带 `result` 或 `error`：

```json
{ "id": 10, "result": { "thread": { "id": "thr_123" } } }
```

```json
{ "id": 10, "error": { "code": 123, "message": "出了点问题" } }
```

通知省略 `id`，仅使用 `method` 和 `params`：

```json
{ "method": "turn/started", "params": { "turn": { "id": "turn_456" } } }
```

你可以从 CLI 生成 TypeScript schema 或 JSON Schema 包。每个输出都特定于你运行的 Codex 版本，因此生成的工件与该版本完全匹配：

```bash
codex app-server generate-ts --out ./schemas
codex app-server generate-json-schema --out ./schemas
```

## 快速入门

1. 使用 `codex app-server`（默认 stdio 传输）或 `codex app-server --listen ws://127.0.0.1:4500`（实验性 WebSocket 传输）启动服务器。
2. 通过选定的传输层连接客户端，然后发送 `initialize`，紧接着发送 `initialized` 通知。
3. 启动一个 thread 和一个 turn，然后持续从活跃的传输流中读取通知。

示例（Node.js / TypeScript）：

```ts
const proc = spawn("codex", ["app-server"], {
  stdio: ["pipe", "pipe", "inherit"],
});
const rl = readline.createInterface({ input: proc.stdout });

const send = (message: unknown) => {
  proc.stdin.write(`${JSON.stringify(message)}\n`);
};

let threadId: string | null = null;

rl.on("line", (line) => {
  const msg = JSON.parse(line) as any;
  console.log("server:", msg);

  if (msg.id === 1 && msg.result?.thread?.id && !threadId) {
    threadId = msg.result.thread.id;
    send({
      method: "turn/start",
      id: 2,
      params: {
        threadId,
        input: [{ type: "text", text: "总结这个仓库。" }],
      },
    });
  }
});

send({
  method: "initialize",
  id: 0,
  params: {
    clientInfo: {
      name: "my_product",
      title: "My Product",
      version: "0.1.0",
    },
  },
});
send({ method: "initialized", params: {} });
send({ method: "thread/start", id: 1, params: { model: "gpt-5.4" } });
```

## 核心原语

- **Thread**：用户与 Codex 代理之间的对话。Thread 包含 turn。
- **Turn**：单个用户请求及随后的代理工作。Turn 包含 item 并流式传输增量更新。
- **Item**：输入或输出单元（用户消息、代理消息、命令运行、文件更改、工具调用等）。

使用 thread API 创建、列出或归档对话。使用 turn API 驱动对话，并通过 turn 通知流式传输进度。

## 生命周期概览

- **每个连接初始化一次**：打开传输连接后，立即发送带有客户端元数据的 `initialize` 请求，然后发出 `initialized`。服务器在此握手之前拒绝该连接上的任何请求。
- **启动（或恢复）一个 thread**：调用 `thread/start` 开始新对话，`thread/resume` 继续已有对话，或 `thread/fork` 将历史分支到新的 thread id。
- **开始一个 turn**：调用 `turn/start`，传入目标 `threadId` 和用户输入。可选字段可覆盖模型、personality、`cwd`、沙箱策略等。
- **操控活跃的 turn**：调用 `turn/steer` 向当前正在进行的 turn 追加用户输入，而不创建新的 turn。
- **流式事件**：`turn/start` 之后，持续读取 stdout 上的通知：`thread/archived`、`thread/unarchived`、`item/started`、`item/completed`、`item/agentMessage/delta`、工具进度及其他更新。
- **完成 turn**：当模型完成或 `turn/interrupt` 取消后，服务器发出带有最终状态的 `turn/completed`。

## 初始化

客户端必须在每个传输连接上发送单个 `initialize` 请求，然后才能在该连接上调用任何其他方法，之后用 `initialized` 通知进行确认。在初始化之前发送的请求会收到 `Not initialized` 错误，在同一连接上重复调用 `initialize` 会返回 `Already initialized`。

服务器返回它将向上游服务呈现的 user agent 字符串，以及描述运行时目标的 `platformFamily` 和 `platformOs` 值。设置 `clientInfo` 来标识你的集成。

`initialize.params.capabilities` 还支持通过 `optOutNotificationMethods` 进行按连接的通知退出，这是一个要对该连接抑制的确切方法名称列表。匹配是精确的（不支持通配符/前缀）。未知的方法名称会被接受并忽略。

**重要**：使用 `clientInfo.name` 为 OpenAI 合规日志平台标识你的客户端。如果你正在开发一个面向企业使用的新 Codex 集成，请联系 OpenAI 将其添加到已知客户端列表中。更多上下文参见 [Codex 日志参考](https://chatgpt.com/admin/api-reference#tag/Logs:-Codex)。

示例（来自 Codex VS Code 扩展）：

```json
{
  "method": "initialize",
  "id": 0,
  "params": {
    "clientInfo": {
      "name": "codex_vscode",
      "title": "Codex VS Code Extension",
      "version": "0.1.0"
    }
  }
}
```

带通知退出的示例：

```json
{
  "method": "initialize",
  "id": 1,
  "params": {
    "clientInfo": {
      "name": "my_client",
      "title": "My Client",
      "version": "0.1.0"
    },
    "capabilities": {
      "experimentalApi": true,
      "optOutNotificationMethods": ["thread/started", "item/agentMessage/delta"]
    }
  }
}
```

## 实验性 API 启用

某些 app-server 方法和字段有意被 `experimentalApi` 能力控制。

- 省略 `capabilities`（或将 `experimentalApi` 设为 `false`）以保持在稳定 API 表面上，服务器会拒绝实验性方法/字段。
- 将 `capabilities.experimentalApi` 设为 `true` 以启用实验性方法和字段。

```json
{
  "method": "initialize",
  "id": 1,
  "params": {
    "clientInfo": {
      "name": "my_client",
      "title": "My Client",
      "version": "0.1.0"
    },
    "capabilities": {
      "experimentalApi": true
    }
  }
}
```

如果客户端在未启用的状态下发送实验性方法或字段，app-server 会拒绝并返回：

`<descriptor> requires experimentalApi capability`

## API 概览

- `thread/start` — 创建新的 thread；发出 `thread/started` 并自动为你订阅该 thread 的 turn/item 事件。
- `thread/resume` — 按 id 重新打开已有的 thread，以便后续的 `turn/start` 调用追加到其中。
- `thread/fork` — 通过复制存储的历史将 thread 分支到新的 thread id；为新 thread 发出 `thread/started`。
- `thread/read` — 按 id 读取存储的 thread 而不恢复它；设置 `includeTurns` 返回完整的 turn 历史。返回的 `thread` 对象包含运行时 `status`。
- `thread/list` — 分页浏览存储的 thread 日志；支持基于游标的分页以及 `modelProviders`、`sourceKinds`、`archived`、`cwd` 和 `searchTerm` 过滤。返回的 `thread` 对象包含运行时 `status`。
- `thread/turns/list` — 分页浏览存储 thread 的 turn 历史而不恢复它。
- `thread/loaded/list` — 列出当前加载在内存中的 thread id。
- `thread/name/set` — 为已加载的 thread 或持久化的 rollout 设置或更新 thread 的用户可见名称；发出 `thread/name/updated`。
- `thread/goal/set` — 为已加载的 thread 设置目标（实验性；需要 `capabilities.experimentalApi`）；发出 `thread/goal/updated`。
- `thread/goal/get` — 读取已加载 thread 的当前目标（实验性；需要 `capabilities.experimentalApi`）。
- `thread/goal/clear` — 清除已加载 thread 的目标（实验性；需要 `capabilities.experimentalApi`）；发出 `thread/goal/cleared`。
- `thread/metadata/update` — 修补 SQLite 支持的存储 thread 元数据；目前支持持久化的 `gitInfo`。
- `thread/archive` — 将 thread 的日志文件移动到归档目录；成功时返回 `{}` 并发出 `thread/archived`。
- `thread/unsubscribe` — 取消此连接对 thread turn/item 事件的订阅。如果这是最后一个订阅者，服务器在无订阅者的不活动宽限期后卸载 thread 并发出 `thread/closed`。
- `thread/unarchive` — 将归档的 thread rollout 恢复到活跃会话目录；返回恢复的 `thread` 并发出 `thread/unarchived`。
- `thread/status/changed` — 当已加载 thread 的运行时 `status` 发生变化时发出的通知。
- `thread/compact/start` — 为 thread 触发对话历史压缩；立即返回 `{}`，同时通过 `turn/*` 和 `item/*` 通知流式传输进度。
- `thread/shellCommand` — 对 thread 运行用户发起的 shell 命令。该命令在沙箱外运行，拥有完全访问权限，不继承 thread 的沙箱策略。
- `thread/backgroundTerminals/clean` — 停止 thread 的所有运行中的后台终端（实验性；需要 `capabilities.experimentalApi`）。
- `thread/rollback` — 从内存上下文中丢弃最后 N 个 turn 并持久化回滚标记；返回更新后的 `thread`。
- `turn/start` — 向 thread 添加用户输入并开始 Codex 生成；响应初始 `turn` 并流式传输事件。对于 `collaborationMode`，`settings.developer_instructions: null` 表示"使用所选模式的内置指令"。
- `thread/inject_items` — 向已加载 thread 的模型可见历史追加原始 Responses API item，而不启动用户 turn。
- `turn/steer` — 向 thread 的活跃进行中 turn 追加用户输入；返回接受的 `turnId`。
- `turn/interrupt` — 请求取消进行中的 turn；成功为 `{}`，turn 以 `status: "interrupted"` 结束。
- `review/start` — 为 thread 启动 Codex reviewer；发出 `enteredReviewMode` 和 `exitedReviewMode` item。
- `command/exec` — 在服务器沙箱下运行单个命令，而不启动 thread/turn。
- `command/exec/write` — 将 `stdin` 字节写入运行中的 `command/exec` 会话或关闭 `stdin`。
- `command/exec/resize` — 调整运行中的 PTY 支持的 `command/exec` 会话的大小。
- `command/exec/terminate` — 停止运行中的 `command/exec` 会话。
- `command/exec/outputDelta`（通知）— 为流式 `command/exec` 会话的 base64 编码 stdout/stderr 块发出。
- `model/list` — 列出可用模型（设置 `includeHidden: true` 包含带有 `hidden: true` 的条目），包括 effort 选项、可选的 `upgrade` 和 `inputModalities`。
- `modelProvider/capabilities/read` — 读取模型/提供商组合的提供商能力边界（实验性；需要 `capabilities.experimentalApi`）。
- `experimentalFeature/list` — 列出带有生命周期阶段元数据和游标分页的功能标志。
- `experimentalFeature/enablement/set` — 为支持的功能键（如 `apps` 和 `plugins`）修补内存中的运行时启用状态。
- `collaborationMode/list` — 列出协作模式预设（实验性，无分页）。
- `skills/list` — 列出一个或多个 `cwd` 值的技能（支持 `forceReload` 和可选的 `perCwdExtraUserRoots`）。
- `skills/changed`（通知）— 当被监视的本地技能文件发生变化时发出。
- `marketplace/add` — 添加远程插件市场并将其持久化到用户的市场配置中。
- `marketplace/upgrade` — 刷新配置的 Git 市场，或省略市场名称时刷新所有配置的 Git 市场。
- `plugin/list` — 列出发现的插件市场和插件状态，包括安装/认证策略元数据、市场加载错误、特色插件 id，以及本地、Git 或远程插件源元数据。
- `plugin/read` — 按市场路径或远程市场名称和插件名称读取单个插件，包括绑定的技能、应用和 MCP 服务器名称（当这些详细信息可用时）。
- `plugin/install` — 从市场路径或远程市场名称安装插件。
- `plugin/uninstall` — 卸载已安装的插件。
- `app/list` — 列出可用应用（连接器），带分页以及可访问性/启用元数据。
- `skills/config/write` — 按路径启用或禁用技能。
- `mcpServer/oauth/login` — 为配置的 MCP 服务器启动 OAuth 登录；返回授权 URL 并在完成时发出 `mcpServer/oauthLogin/completed`。
- `tool/requestUserInput` — 为工具调用向用户提出 1-3 个简短问题（实验性）；问题可设置 `isOther` 提供自由形式的选项。
- `config/mcpServer/reload` — 从磁盘重新加载 MCP 服务器配置并为已加载的 thread 排队刷新。
- `mcpServerStatus/list` — 列出 MCP 服务器、工具、资源和认证状态（游标 + limit 分页）。使用 `detail: "full"` 获取完整数据，或 `detail: "toolsAndAuthOnly"` 省略资源。
- `mcpServer/resource/read` — 通过已初始化的 MCP 服务器读取单个 MCP 资源。
- `mcpServer/tool/call` — 在 thread 配置的 MCP 服务器上调用工具。
- `mcpServer/startupStatus/updated`（通知）— 当配置的 MCP 服务器对已加载 thread 的启动状态发生变化时发出。
- `windowsSandbox/setupStart` — 为 `elevated` 或 `unelevated` 模式启动 Windows 沙箱设置；快速返回，稍后发出 `windowsSandbox/setupCompleted`。
- `feedback/upload` — 提交反馈报告（分类 + 可选原因/日志 + 对话 id，以及可选的 `extraLogFiles` 附件）。
- `config/read` — 获取解析配置层级后磁盘上的有效配置。
- `externalAgentConfig/detect` — 检测可以迁移的外部代理工件，支持 `includeHome` 和可选的 `cwds`；每个检测到的条目包含 `cwd`（home 时为 `null`）。
- `externalAgentConfig/import` — 通过传入显式的 `migrationItems`（带 `cwd`，home 时为 `null`）应用选定的外部代理迁移条目。支持的条目类型包括 config、skills、`AGENTS.md`、plugins、MCP 服务器配置、子代理、hooks、命令和会话；插件导入发出 `externalAgentConfig/import/completed`。
- `config/value/write` — 将单个配置键/值写入用户磁盘上的 `config.toml`。
- `config/batchWrite` — 原子地将配置编辑应用到用户磁盘上的 `config.toml`。
- `configRequirements/read` — 从 `requirements.toml` 和/或 MDM 获取需求，包括允许列表、固定的 `featureRequirements` 以及驻留/网络需求（如果尚未设置，则为 `null`）。
- `fs/readFile`、`fs/writeFile`、`fs/createDirectory`、`fs/getMetadata`、`fs/readDirectory`、`fs/remove`、`fs/copy`、`fs/watch`、`fs/unwatch` 和 `fs/changed`（通知）— 通过 app-server v2 文件系统 API 操作绝对文件系统路径。

插件摘要包含一个 `source` 联合体。本地插件返回
`{ "type": "local", "path": ... }`，Git 支持的市场条目返回
`{ "type": "git", "url": ..., "path": ..., "refName": ..., "sha": ... }`，
远程目录条目返回 `{ "type": "remote" }`。对于仅远程目录的条目，
`PluginMarketplaceEntry.path` 可以为 `null`；读取或安装这些插件时，
传入 `remoteMarketplaceName` 而不是 `marketplacePath`。

## 模型

### 列出模型（`model/list`）

调用 `model/list` 以在渲染模型或 personality 选择器之前发现可用模型及其能力。

```json
{ "method": "model/list", "id": 6, "params": { "limit": 20, "includeHidden": false } }
{ "id": 6, "result": {
  "data": [{
    "id": "gpt-5.4",
    "model": "gpt-5.4",
    "displayName": "GPT-5.4",
    "hidden": false,
    "defaultReasoningEffort": "medium",
    "supportedReasoningEfforts": [{
      "reasoningEffort": "low",
      "description": "更低的延迟"
    }],
    "inputModalities": ["text", "image"],
    "supportsPersonality": true,
    "isDefault": true
  }],
  "nextCursor": null
} }
```

每个模型条目可以包含：

- `supportedReasoningEfforts` — 模型支持的 effort 选项。
- `defaultReasoningEffort` — 给客户端的建议默认 effort。
- `upgrade` — 可选的推荐升级模型 id，用于客户端的迁移提示。
- `upgradeInfo` — 可选的升级元数据，用于客户端的迁移提示。
- `hidden` — 模型是否从默认选择器列表中隐藏。
- `inputModalities` — 模型支持的输入类型（例如 `text`、`image`）。
- `supportsPersonality` — 模型是否支持特定 personality 的指令（如 `/personality`）。
- `isDefault` — 模型是否为推荐的默认值。

默认情况下，`model/list` 仅返回选择器可见的模型。如果你需要完整列表并希望在客户端使用 `hidden` 进行过滤，请设置 `includeHidden: true`。

当 `inputModalities` 缺失时（较旧的模型目录），将其视为 `["text", "image"]` 以保持向后兼容。

### 列出实验性功能（`experimentalFeature/list`）

使用此端点发现带有元数据和生命周期阶段的功能标志：

```json
{ "method": "experimentalFeature/list", "id": 7, "params": { "limit": 20 } }
{ "id": 7, "result": {
  "data": [{
    "name": "unified_exec",
    "stage": "beta",
    "displayName": "统一执行",
    "description": "使用统一的 PTY 支持执行工具。",
    "announcement": "Beta 推出以改进命令执行可靠性。",
    "enabled": false,
    "defaultEnabled": false
  }],
  "nextCursor": null
} }
```

`stage` 可以是 `beta`、`underDevelopment`、`stable`、`deprecated` 或 `removed`。对于非 beta 标志，`displayName`、`description` 和 `announcement` 可能为 `null`。

## Threads

- `thread/read` 读取存储的 thread 而不订阅它；设置 `includeTurns` 以包含 turn。
- `thread/turns/list` 分页浏览存储 thread 的 turn 历史而不恢复它。
- `thread/list` 支持游标分页以及 `modelProviders`、`sourceKinds`、`archived`、`cwd` 和 `searchTerm` 过滤。
- `thread/loaded/list` 返回当前在内存中的 thread ID。
- `thread/archive` 将 thread 的持久化 JSONL 日志移动到归档目录。
- `thread/metadata/update` 修补存储的 thread 元数据，目前包括持久化的 `gitInfo`。
- `thread/unsubscribe` 取消当前连接对已加载 thread 的订阅，在不活动宽限期后可能触发 `thread/closed`。
- `thread/unarchive` 将归档的 thread rollout 恢复到活跃会话目录。
- `thread/compact/start` 触发压缩并立即返回 `{}`。
- `thread/rollback` 从内存上下文中丢弃最后 N 个 turn，并在 thread 的持久化 JSONL 日志中记录回滚标记。
- `thread/inject_items` 向已加载 thread 的模型可见历史追加原始 Responses API item，而不启动用户 turn。

### 启动或恢复 thread

需要新的 Codex 对话时启动一个全新的 thread。

```json
{ "method": "thread/start", "id": 10, "params": {
  "model": "gpt-5.4",
  "cwd": "/Users/me/project",
  "approvalPolicy": "never",
  "sandbox": "workspaceWrite",
  "personality": "friendly",
  "serviceName": "my_app_server_client"
} }
{ "id": 10, "result": {
  "thread": {
    "id": "thr_123",
    "preview": "",
    "ephemeral": false,
    "modelProvider": "openai",
    "createdAt": 1730910000
  }
} }
{ "method": "thread/started", "params": { "thread": { "id": "thr_123" } } }
```

`serviceName` 是可选的。当你希望 app-server 用你的集成的服务名标记 thread 级别的指标时设置它。

要恢复已存储的会话，使用你之前记录的 `thread.id` 调用 `thread/resume`。响应形状与 `thread/start` 匹配。你也可以传递 `thread/start` 支持的相同配置覆盖项，例如 `personality`：

```json
{ "method": "thread/resume", "id": 11, "params": {
  "threadId": "thr_123",
  "personality": "friendly"
} }
{ "id": 11, "result": { "thread": { "id": "thr_123", "name": "Bug 排查笔记", "ephemeral": false } } }
```

恢复 thread 本身不会更新 `thread.updatedAt`（或 rollout 文件的修改时间）。时间戳在你启动 turn 时更新。

如果你在配置中将启用的 MCP 服务器标记为 `required` 且该服务器初始化失败，则 `thread/start` 和 `thread/resume` 会失败而不是在没有它的情况下继续。

`thread/start` 上的 `dynamicTools` 是实验性字段（需要 `capabilities.experimentalApi = true`）。Codex 将这些动态工具持久化在 thread rollout 元数据中，并在你不提供新动态工具时在 `thread/resume` 上恢复它们。

如果你使用与 rollout 中记录的不同的模型恢复，Codex 会发出警告并在下一个 turn 上应用一次性模型切换指令。

要从存储的会话分支，使用 `thread.id` 调用 `thread/fork`。这会创建一个新的 thread id 并为其发出 `thread/started` 通知：

```json
{ "method": "thread/fork", "id": 12, "params": { "threadId": "thr_123" } }
{ "id": 12, "result": { "thread": { "id": "thr_456" } } }
{ "method": "thread/started", "params": { "thread": { "id": "thr_456" } } }
```

当用户可见的 thread 标题已被设置时，app-server 在 `thread/list`、`thread/read`、`thread/resume`、`thread/unarchive` 和 `thread/rollback` 响应中填充 `thread.name`。`thread/start` 和 `thread/fork` 可能省略 `name`（或返回 `null`），直到稍后设置标题。

### 读取存储的 thread（不恢复）

当你需要存储的 thread 数据但不想恢复 thread 或订阅其事件时使用 `thread/read`。

- `includeTurns` — 为 `true` 时，响应包含 thread 的 turn；为 `false` 或省略时，只获取 thread 摘要。
- 返回的 `thread` 对象包含运行时 `status`（`notLoaded`、`idle`、`systemError` 或带有 `activeFlags` 的 `active`）。

```json
{ "method": "thread/read", "id": 19, "params": { "threadId": "thr_123", "includeTurns": true } }
{ "id": 19, "result": { "thread": { "id": "thr_123", "name": "Bug 排查笔记", "ephemeral": false, "status": { "type": "notLoaded" }, "turns": [] } } }
```

与 `thread/resume` 不同，`thread/read` 不会将 thread 加载到内存中或发出 `thread/started`。

### 列出 thread 的 turn

使用 `thread/turns/list` 分页浏览存储 thread 的 turn 历史而不恢复它。结果默认按最新优先排序，因此客户端可以使用 `nextCursor` 获取更早的 turn。响应还包含 `backwardsCursor`；将其作为 `cursor` 传入并配合 `sortDirection: "asc"` 以获取比上一页第一项更新的 turn。

```json
{ "method": "thread/turns/list", "id": 20, "params": {
  "threadId": "thr_123",
  "limit": 50,
  "sortDirection": "desc"
} }
{ "id": 20, "result": {
  "data": [],
  "nextCursor": "older-turns-cursor-or-null",
  "backwardsCursor": "newer-turns-cursor-or-null"
} }
```

### 列出 thread（带分页和过滤）

`thread/list` 允许你渲染历史 UI。结果默认按 `createdAt` 最新优先。过滤在分页之前应用。可传入以下任意组合：

- `cursor` — 来自先前响应的不透明字符串；第一页省略。
- `limit` — 如果未设置，服务器默认为合理的页面大小。
- `sortKey` — `created_at`（默认）或 `updated_at`。
- `modelProviders` — 将结果限制为特定提供商；未设置、null 或空数组包括所有提供商。
- `sourceKinds` — 将结果限制为特定 thread 来源。省略或 `[]` 时，服务器默认仅包括交互式来源：`cli` 和 `vscode`。
- `archived` — 为 `true` 时，仅列出已归档的 thread。为 `false` 或省略时，列出未归档的 thread（默认）。
- `cwd` — 将结果限制为会话当前工作目录与此路径完全匹配的 thread。
- `searchTerm` — 在分页之前搜索存储的 thread 摘要和元数据。

`sourceKinds` 接受以下值：

- `cli`
- `vscode`
- `exec`
- `appServer`
- `subAgent`
- `subAgentReview`
- `subAgentCompact`
- `subAgentThreadSpawn`
- `subAgentOther`
- `unknown`

示例：

```json
{ "method": "thread/list", "id": 20, "params": {
  "cursor": null,
  "limit": 25,
  "sortKey": "created_at"
} }
{ "id": 20, "result": {
  "data": [
    { "id": "thr_a", "preview": "创建一个 TUI", "ephemeral": false, "modelProvider": "openai", "createdAt": 1730831111, "updatedAt": 1730831111, "name": "TUI 原型", "status": { "type": "notLoaded" } },
    { "id": "thr_b", "preview": "修复测试", "ephemeral": true, "modelProvider": "openai", "createdAt": 1730750000, "updatedAt": 1730750000, "status": { "type": "notLoaded" } }
  ],
  "nextCursor": "opaque-token-or-null"
} }
```

当 `nextCursor` 为 `null` 时，你已到达最后一页。

### 更新存储的 thread 元数据

使用 `thread/metadata/update` 修补存储的 thread 元数据而不恢复 thread。目前支持持久化的 `gitInfo`；省略的字段保持不变，显式的 `null` 清除存储的值。

```json
{ "method": "thread/metadata/update", "id": 21, "params": {
  "threadId": "thr_123",
  "gitInfo": { "branch": "feature/sidebar-pr" }
} }
{ "id": 21, "result": {
  "thread": {
    "id": "thr_123",
    "gitInfo": { "sha": null, "branch": "feature/sidebar-pr", "originUrl": null }
  }
} }
```

### 跟踪 thread 状态变化

每当已加载 thread 的运行时状态发生变化时，发出 `thread/status/changed`。负载包括 `threadId` 和新的 `status`。

```json
{
  "method": "thread/status/changed",
  "params": {
    "threadId": "thr_123",
    "status": { "type": "active", "activeFlags": ["waitingOnApproval"] }
  }
}
```

### 列出已加载的 thread

`thread/loaded/list` 返回当前加载在内存中的 thread ID。

```json
{ "method": "thread/loaded/list", "id": 21 }
{ "id": 21, "result": { "data": ["thr_123", "thr_456"] } }
```

### 取消订阅已加载的 thread

`thread/unsubscribe` 移除当前连接对 thread 的订阅。响应状态为以下之一：

- `unsubscribed` — 连接已订阅且现在已移除。
- `notSubscribed` — 连接未订阅该 thread。
- `notLoaded` — thread 未加载。

如果这是最后一个订阅者，服务器会保持 thread 加载，直到它没有订阅者且没有 thread 活动持续 30 分钟。当宽限期到期时，app-server 卸载 thread 并发出 `thread/status/changed` 转换到 `notLoaded` 加上 `thread/closed`。

```json
{ "method": "thread/unsubscribe", "id": 22, "params": { "threadId": "thr_123" } }
{ "id": 22, "result": { "status": "unsubscribed" } }
```

如果 thread 稍后过期：

```json
{ "method": "thread/status/changed", "params": {
    "threadId": "thr_123",
    "status": { "type": "notLoaded" }
} }
{ "method": "thread/closed", "params": { "threadId": "thr_123" } }
```

### 归档 thread

使用 `thread/archive` 将持久化的 thread 日志（以 JSONL 文件形式存储在磁盘上）移动到归档会话目录。

```json
{ "method": "thread/archive", "id": 22, "params": { "threadId": "thr_b" } }
{ "id": 22, "result": {} }
{ "method": "thread/archived", "params": { "threadId": "thr_b" } }
```

已归档的 thread 不会出现在将来的 `thread/list` 调用中，除非你传入 `archived: true`。

### 取消归档 thread

使用 `thread/unarchive` 将归档的 thread rollout 移回活跃会话目录。

```json
{ "method": "thread/unarchive", "id": 24, "params": { "threadId": "thr_b" } }
{ "id": 24, "result": { "thread": { "id": "thr_b", "name": "Bug 排查笔记" } } }
{ "method": "thread/unarchived", "params": { "threadId": "thr_b" } }
```

### 触发 thread 压缩

使用 `thread/compact/start` 为 thread 触发手动历史压缩。请求立即返回 `{}`。

App-server 以标准的 `turn/*` 和 `item/*` 通知在同一 `threadId` 上发出进度，包括 `contextCompaction` item 生命周期（先是 `item/started`，然后是 `item/completed`）。

```json
{ "method": "thread/compact/start", "id": 25, "params": { "threadId": "thr_b" } }
{ "id": 25, "result": {} }
```

### 运行 thread shell 命令

使用 `thread/shellCommand` 执行属于 thread 的用户发起 shell 命令。请求立即返回 `{}`，同时通过标准的 `turn/*` 和 `item/*` 通知流式传输进度。

此 API 在沙箱外运行，拥有完全访问权限，不继承 thread 的沙箱策略。客户端应仅对显式用户发起的命令暴露此 API。

如果 thread 已有活跃的 turn，该命令作为该 turn 上的辅助操作运行，其格式化输出被注入到 turn 的消息流中。如果 thread 空闲，app-server 为该 shell 命令启动一个独立的 turn。

```json
{ "method": "thread/shellCommand", "id": 26, "params": { "threadId": "thr_b", "command": "git status --short" } }
{ "id": 26, "result": {} }
```

### 清理后台终端

使用 `thread/backgroundTerminals/clean` 停止与 thread 关联的所有运行中的后台终端。此方法是实验性的，需要 `capabilities.experimentalApi = true`。

```json
{ "method": "thread/backgroundTerminals/clean", "id": 27, "params": { "threadId": "thr_b" } }
{ "id": 27, "result": {} }
```

### 回滚最近的 turn

使用 `thread/rollback` 从内存上下文中移除最后 `numTurns` 个条目，并在 rollout 日志中持久化回滚标记。返回的 `thread` 包含回滚后填充的 `turns`。

```json
{ "method": "thread/rollback", "id": 28, "params": { "threadId": "thr_b", "numTurns": 1 } }
{ "id": 28, "result": { "thread": { "id": "thr_b", "name": "Bug 排查笔记", "ephemeral": false } } }
```

## Turns

`input` 字段接受一个 item 列表：

- `{ "type": "text", "text": "解释这个 diff" }`
- `{ "type": "image", "url": "https://.../design.png" }`
- `{ "type": "localImage", "path": "/tmp/screenshot.png" }`

你可以按 turn 覆盖配置设置（model、effort、personality、`cwd`、沙箱策略、summary）。指定后，这些设置将成为同一 thread 上后续 turn 的默认值。`outputSchema` 仅适用于当前 turn。对于 `sandboxPolicy.type = "externalSandbox"`，将 `networkAccess` 设置为 `restricted` 或 `enabled`；对于 `workspaceWrite`，`networkAccess` 保持为布尔值。

对于 `turn/start.collaborationMode`，`settings.developer_instructions: null` 表示"使用所选模式的内置指令"，而不是清除模式指令。

### 沙箱读取权限（`ReadOnlyAccess`）

`sandboxPolicy` 支持显式的读取权限控制：

- `readOnly`：可选的 `access`（默认为 `{ "type": "fullAccess" }`，或受限的根目录）。
- `workspaceWrite`：可选的 `readOnlyAccess`（默认为 `{ "type": "fullAccess" }`，或受限的根目录）。

受限读取权限的结构：

```json
{
  "type": "restricted",
  "includePlatformDefaults": true,
  "readableRoots": ["/Users/me/shared-read-only"]
}
```

在 macOS 上，`includePlatformDefaults: true` 会为受限读取会话追加一个精心挑选的平台默认 Seatbelt 策略。这可以改善工具兼容性，而无需广泛地允许整个 `/System`。

示例：

```json
{ "type": "readOnly", "access": { "type": "fullAccess" } }
```

```json
{
  "type": "workspaceWrite",
  "writableRoots": ["/Users/me/project"],
  "readOnlyAccess": {
    "type": "restricted",
    "includePlatformDefaults": true,
    "readableRoots": ["/Users/me/shared-read-only"]
  },
  "networkAccess": false
}
```

### 启动 turn

```json
{ "method": "turn/start", "id": 30, "params": {
  "threadId": "thr_123",
  "input": [ { "type": "text", "text": "运行测试" } ],
  "cwd": "/Users/me/project",
  "approvalPolicy": "unlessTrusted",
  "sandboxPolicy": {
    "type": "workspaceWrite",
    "writableRoots": ["/Users/me/project"],
    "networkAccess": true
  },
  "model": "gpt-5.4",
  "effort": "medium",
  "summary": "concise",
  "personality": "friendly",
  "outputSchema": {
    "type": "object",
    "properties": { "answer": { "type": "string" } },
    "required": ["answer"],
    "additionalProperties": false
  }
} }
{ "id": 30, "result": { "turn": { "id": "turn_456", "status": "inProgress", "items": [], "error": null } } }
```

### 向 thread 注入 item

使用 `thread/inject_items` 将预构建的 Responses API item 追加到已加载 thread 的提示历史中，而不启动用户 turn。这些 item 被持久化到 rollout 中，并包含在后续的模型请求中。

```json
{ "method": "thread/inject_items", "id": 31, "params": {
  "threadId": "thr_123",
  "items": [
    {
      "type": "message",
      "role": "assistant",
      "content": [{ "type": "output_text", "text": "之前计算的上下文。" }]
    }
  ]
} }
{ "id": 31, "result": {} }
```

### 操控活跃的 turn

使用 `turn/steer` 向活跃的进行中 turn 追加更多用户输入。

- 包含 `expectedTurnId`；它必须匹配活跃的 turn id。
- 如果 thread 上没有活跃的 turn，请求会失败。
- `turn/steer` 不发出新的 `turn/started` 通知。
- `turn/steer` 不接受 turn 级别的覆盖项（`model`、`cwd`、`sandboxPolicy` 或 `outputSchema`）。

```json
{ "method": "turn/steer", "id": 32, "params": {
  "threadId": "thr_123",
  "input": [ { "type": "text", "text": "实际上先关注失败的测试。" } ],
  "expectedTurnId": "turn_456"
} }
{ "id": 32, "result": { "turnId": "turn_456" } }
```

### 启动 turn（调用技能）

通过在文本输入中包含 `$<skill-name>` 并附加一个 `skill` 输入 item 来显式调用技能。

```json
{ "method": "turn/start", "id": 33, "params": {
  "threadId": "thr_123",
  "input": [
    { "type": "text", "text": "$skill-creator 添加一个用于梳理不稳定的 CI 的新技能，并包含逐步使用说明。" },
    { "type": "skill", "name": "skill-creator", "path": "/Users/me/.codex/skills/skill-creator/SKILL.md" }
  ]
} }
{ "id": 33, "result": { "turn": { "id": "turn_457", "status": "inProgress", "items": [], "error": null } } }
```

### 中断 turn

```json
{ "method": "turn/interrupt", "id": 31, "params": { "threadId": "thr_123", "turnId": "turn_456" } }
{ "id": 31, "result": {} }
```

成功后，turn 以 `status: "interrupted"` 结束。

## Review

`review/start` 为 thread 运行 Codex reviewer 并流式传输 review item。目标包括：

- `uncommittedChanges`
- `baseBranch`（与分支的 diff）
- `commit`（审查特定提交）
- `custom`（自由形式的指令）

使用 `delivery: "inline"`（默认）在现有 thread 上运行 review，或 `delivery: "detached"` 分支出一个新的 review thread。

请求/响应示例：

```json
{ "method": "review/start", "id": 40, "params": {
  "threadId": "thr_123",
  "delivery": "inline",
  "target": { "type": "commit", "sha": "1234567deadbeef", "title": "优化 tui 颜色" }
} }
{ "id": 40, "result": {
  "turn": {
    "id": "turn_900",
    "status": "inProgress",
    "items": [
      { "type": "userMessage", "id": "turn_900", "content": [ { "type": "text", "text": "审查提交 1234567: 优化 tui 颜色" } ] }
    ],
    "error": null
  },
  "reviewThreadId": "thr_123"
} }
```

对于分离式 review，使用 `"delivery": "detached"`。响应形状相同，但 `reviewThreadId` 将是新 review thread 的 id（不同于原始 `threadId`）。服务器在流式传输 review turn 之前还会为该新 thread 发出 `thread/started` 通知。

Codex 流式传输通常的 `turn/started` 通知，随后是带有 `enteredReviewMode` item 的 `item/started`：

```json
{
  "method": "item/started",
  "params": {
    "item": {
      "type": "enteredReviewMode",
      "id": "turn_900",
      "review": "当前更改"
    }
  }
}
```

当 reviewer 完成时，服务器发出包含 `exitedReviewMode` item 的 `item/started` 和 `item/completed`，其中包含最终的 review 文本：

```json
{
  "method": "item/completed",
  "params": {
    "item": {
      "type": "exitedReviewMode",
      "id": "turn_900",
      "review": "总体上看起来很可靠..."
    }
  }
}
```

使用此通知在你的客户端中渲染 reviewer 输出。

## 命令执行

`command/exec` 在服务器沙箱下运行单个命令（`argv` 数组），而不创建 thread。

```json
{ "method": "command/exec", "id": 50, "params": {
  "command": ["ls", "-la"],
  "cwd": "/Users/me/project",
  "sandboxPolicy": { "type": "workspaceWrite" },
  "timeoutMs": 10000
} }
{ "id": 50, "result": { "exitCode": 0, "stdout": "...", "stderr": "" } }
```

如果你已经对服务器进程进行了沙箱化，并希望 Codex 跳过自己的沙箱执行，请使用 `sandboxPolicy.type = "externalSandbox"`。对于外部沙箱模式，将 `networkAccess` 设置为 `restricted`（默认）或 `enabled`。对于 `readOnly` 和 `workspaceWrite`，使用上面展示的相同可选 `access` / `readOnlyAccess` 结构。

注意事项：

- 服务器拒绝空的 `command` 数组。
- `sandboxPolicy` 接受与 `turn/start` 使用的相同形状（例如 `dangerFullAccess`、`readOnly`、`workspaceWrite`、`externalSandbox`）。
- 省略时，`timeoutMs` 回退到服务器默认值。
- 设置 `tty: true` 用于 PTY 支持的会话，当你计划后续使用 `command/exec/write`、`command/exec/resize` 或 `command/exec/terminate` 时使用 `processId`。
- 设置 `streamStdoutStderr: true` 以在命令运行时接收 `command/exec/outputDelta` 通知。

### 读取管理员需求（`configRequirements/read`）

使用 `configRequirements/read` 检查从 `requirements.toml` 和/或 MDM 加载的有效管理员需求。

```json
{ "method": "configRequirements/read", "id": 52, "params": {} }
{ "id": 52, "result": {
  "requirements": {
    "allowedApprovalPolicies": ["onRequest", "unlessTrusted"],
    "allowedSandboxModes": ["readOnly", "workspaceWrite"],
    "featureRequirements": {
      "personality": true,
      "unified_exec": false
    },
    "network": {
      "enabled": true,
      "allowedDomains": ["api.openai.com"],
      "allowUnixSockets": ["/tmp/example.sock"],
      "dangerouslyAllowAllUnixSockets": false
    }
  }
} }
```

当没有配置需求时，`result.requirements` 为 `null`。关于支持的键和值，参见 [`requirements.toml`](https://developers.openai.com/codex/config-reference#requirementstoml) 文档。

### Windows 沙箱设置（`windowsSandbox/setupStart`）

自定义 Windows 客户端可以异步触发沙箱设置，而不是在启动检查时阻塞。

```json
{ "method": "windowsSandbox/setupStart", "id": 53, "params": { "mode": "elevated" } }
{ "id": 53, "result": { "started": true } }
```

App-server 在后台启动设置，稍后发出完成通知：

```json
{
  "method": "windowsSandbox/setupCompleted",
  "params": { "mode": "elevated", "success": true, "error": null }
}
```

模式：

- `elevated` — 运行提升的 Windows 沙箱设置路径。
- `unelevated` — 运行旧版设置/预检路径。

## 文件系统

v2 文件系统 API 操作绝对路径。当客户端需要在文件或目录更改后使 UI 状态失效时使用 `fs/watch`。

```json
{ "method": "fs/watch", "id": 54, "params": {
  "watchId": "0195ec6b-1d6f-7c2e-8c7a-56f2c4a8b9d1",
  "path": "/Users/me/project/.git/HEAD"
} }
{ "id": 54, "result": { "path": "/Users/me/project/.git/HEAD" } }
{ "method": "fs/changed", "params": {
  "watchId": "0195ec6b-1d6f-7c2e-8c7a-56f2c4a8b9d1",
  "changedPaths": ["/Users/me/project/.git/HEAD"]
} }
{ "method": "fs/unwatch", "id": 55, "params": {
  "watchId": "0195ec6b-1d6f-7c2e-8c7a-56f2c4a8b9d1"
} }
{ "id": 55, "result": {} }
```

监视文件会为该文件路径发出 `fs/changed`，包括由替换或重命名操作发送的更新。

## 事件

事件通知是服务器发起的流，用于 thread 生命周期、turn 生命周期以及其中的 item。在启动或恢复 thread 后，持续从活跃的传输流中读取 `thread/started`、`thread/archived`、`thread/unarchived`、`thread/closed`、`thread/status/changed`、`turn/*`、`item/*` 和 `serverRequest/resolved` 通知。

### 通知退出

客户端可以通过在 `initialize.params.capabilities.optOutNotificationMethods` 中发送确切的方法名称来按连接抑制特定的通知。

- 仅精确匹配：`item/agentMessage/delta` 仅抑制该方法。
- 未知的方法名称被忽略。
- 适用于当前的 `thread/*`、`turn/*`、`item/*` 以及相关的 v2 通知。
- 不适用于请求、响应或错误。

### 模糊文件搜索事件（实验性）

模糊文件搜索会话 API 发出每个查询的通知：

- `fuzzyFileSearch/sessionUpdated` — `{ sessionId, query, files }` 包含活跃查询的当前匹配项。
- `fuzzyFileSearch/sessionCompleted` — `{ sessionId }` 一旦该查询的索引和匹配完成。

### Windows 沙箱设置事件

- `windowsSandbox/setupCompleted` — `{ mode, success, error }` 在 `windowsSandbox/setupStart` 请求完成后发出。

### Turn 事件

- `turn/started` — `{ turn }` 带有 turn id、空的 `items` 和 `status: "inProgress"`。
- `turn/completed` — `{ turn }` 其中 `turn.status` 为 `completed`、`interrupted` 或 `failed`；失败时携带 `{ error: { message, codexErrorInfo?, additionalDetails? } }`。
- `turn/diff/updated` — `{ threadId, turnId, diff }` 包含 turn 中所有文件更改的最新聚合 unified diff。
- `turn/plan/updated` — `{ turnId, explanation?, plan }` 每当代理分享或更改其计划时发出；每个 `plan` 条目为 `{ step, status }`，`status` 为 `pending`、`inProgress` 或 `completed`。
- `thread/tokenUsage/updated` — 活跃 thread 的使用量更新。

`turn/diff/updated` 和 `turn/plan/updated` 目前包含空的 `items` 数组，即使有 item 事件流式传输。使用 `item/*` 通知作为 turn item 的真实来源。

### Item

`ThreadItem` 是 turn 响应和 `item/*` 通知中携带的标记联合体。常见的 item 类型包括：

- `userMessage` — `{id, content}`，其中 `content` 是用户输入列表（`text`、`image` 或 `localImage`）。
- `agentMessage` — `{id, text, phase?}` 包含累积的代理回复。当存在时，`phase` 使用 Responses API 传输值（`commentary`、`final_answer`）。
- `plan` — `{id, text}` 包含计划模式中提议的计划文本。将 `item/completed` 中的最终 `plan` item 视为权威。
- `reasoning` — `{id, summary, content}`，其中 `summary` 保存流式推理摘要，`content` 保存原始推理块。
- `commandExecution` — `{id, command, cwd, status, commandActions, aggregatedOutput?, exitCode?, durationMs?}`。
- `fileChange` — `{id, changes, status}` 描述提议的编辑；`changes` 列表 `{path, kind, diff}`。
- `mcpToolCall` — `{id, server, tool, status, arguments, result?, error?}`。
- `dynamicToolCall` — `{id, tool, arguments, status, contentItems?, success?, durationMs?}` 用于客户端执行的动态工具调用。
- `collabToolCall` — `{id, tool, status, senderThreadId, receiverThreadId?, newThreadId?, prompt?, agentStatus?}`。
- `webSearch` — `{id, query, action?}` 用于代理发出的网页搜索请求。
- `imageView` — `{id, path}` 当代理调用图像查看器工具时发出。
- `enteredReviewMode` — `{id, review}` 当 reviewer 启动时发送。
- `exitedReviewMode` — `{id, review}` 当 reviewer 完成时发出。
- `contextCompaction` — `{id}` 当 Codex 压缩对话历史时发出。

对于 `webSearch.action`，action `type` 可以是 `search`（`query?`、`queries?`）、`openPage`（`url?`）或 `findInPage`（`url?`、`pattern?`）。

App server 弃用了旧版的 `thread/compacted` 通知；请改用 `contextCompaction` item。

所有 item 发出两个共享的生命周期事件：

- `item/started` — 当新的工作单元开始时发出完整的 `item`；`item.id` 与 delta 使用的 `itemId` 匹配。
- `item/completed` — 一旦工作完成发送最终的 `item`；将其视为权威状态。

### Item Delta

- `item/agentMessage/delta` — 追加代理消息的流式文本。
- `item/plan/delta` — 流式传输提议的计划文本。最终的 `plan` item 可能不完全等于连接的 delta。
- `item/reasoning/summaryTextDelta` — 流式传输可读的推理摘要；当新的摘要部分打开时 `summaryIndex` 递增。
- `item/reasoning/summaryPartAdded` — 标记推理摘要部分之间的边界。
- `item/reasoning/textDelta` — 流式传输原始推理文本（当模型支持时）。
- `item/commandExecution/outputDelta` — 流式传输命令的 stdout/stderr；按顺序追加 delta。
- `item/fileChange/outputDelta` — 包含底层 `apply_patch` 工具调用的工具调用响应。

## 错误

如果 turn 失败，服务器发出带有 `{ error: { message, codexErrorInfo?, additionalDetails? } }` 的 `error` 事件，然后以 `status: "failed"` 完成 turn。当上游 HTTP 状态可用时，它会出现在 `codexErrorInfo.httpStatusCode` 中。

常见的 `codexErrorInfo` 值包括：

- `ContextWindowExceeded`
- `UsageLimitExceeded`
- `HttpConnectionFailed`（4xx/5xx 上游错误）
- `ResponseStreamConnectionFailed`
- `ResponseStreamDisconnected`
- `ResponseTooManyFailedAttempts`
- `BadRequest`、`Unauthorized`、`SandboxError`、`InternalServerError`、`Other`

当上游 HTTP 状态可用时，服务器在相关 `codexErrorInfo` 变体的 `httpStatusCode` 中转发它。

## 审批

根据用户的 Codex 设置，命令执行和文件更改可能需要审批。App-server 向客户端发送服务器发起的 JSON-RPC 请求，客户端响应一个决策负载。

- 命令执行决策：`accept`、`acceptForSession`、`decline`、`cancel`，或 `{ "acceptWithExecpolicyAmendment": { "execpolicy_amendment": ["cmd", "..."] } }`。
- 文件更改决策：`accept`、`acceptForSession`、`decline`、`cancel`。

- 请求包含 `threadId` 和 `turnId` — 使用它们将 UI 状态范围限定到活跃对话。
- 服务器恢复或拒绝工作，并以 `item/completed` 结束 item。

### 命令执行审批

消息顺序：

1. `item/started` 显示待处理的 `commandExecution` item，包含 `command`、`cwd` 和其他字段。
2. `item/commandExecution/requestApproval` 包含 `itemId`、`threadId`、`turnId`、可选的 `reason`、可选的 `command`、可选的 `cwd`、可选的 `commandActions`、可选的 `proposedExecpolicyAmendment`、可选的 `networkApprovalContext` 和可选的 `availableDecisions`。当 `initialize.params.capabilities.experimentalApi = true` 时，负载还可以包含实验性的 `additionalPermissions`，描述请求的按命令沙箱访问权限。`additionalPermissions` 内的任何文件系统路径在传输时是绝对的。
3. 客户端使用上述命令执行审批决策之一进行响应。
4. `serverRequest/resolved` 确认待处理的请求已被应答或清除。
5. `item/completed` 返回最终的 `commandExecution` item，状态为 `status: completed | failed | declined`。

当存在 `networkApprovalContext` 时，提示是针对托管网络访问的（不是一般的 shell 命令审批）。当前的 v2 模式暴露目标 `host` 和 `protocol`；客户端应渲染网络特定的提示，不应依赖 `command` 作为用户有意义的 shell 命令预览。

Codex 按目标（`host`、protocol 和 port）对并发的网络审批提示进行分组。因此，app-server 可能发送一个提示来解除对同一目标的多个排队请求的阻塞，而同一主机上的不同端口则分别处理。

### 文件更改审批

消息顺序：

1. `item/started` 发出一个 `fileChange` item，带有提议的 `changes` 和 `status: "inProgress"`。
2. `item/fileChange/requestApproval` 包含 `itemId`、`threadId`、`turnId`、可选的 `reason` 和可选的 `grantRoot`。
3. 客户端使用上述文件更改审批决策之一进行响应。
4. `serverRequest/resolved` 确认待处理的请求已被应答或清除。
5. `item/completed` 返回最终的 `fileChange` item，状态为 `status: completed | failed | declined`。

### `tool/requestUserInput`

当客户端响应 `item/tool/requestUserInput` 时，app-server 发出带有 `{ threadId, requestId }` 的 `serverRequest/resolved`。如果待处理的请求在客户端应答之前因 turn 启动、turn 完成或 turn 中断而被清除，服务器会为该清理发出相同的通知。

### 动态工具调用（实验性）

`thread/start` 上的 `dynamicTools` 以及相应的 `item/tool/call` 请求或响应流程是实验性 API。

当动态工具在 turn 期间被调用时，app-server 发出：

1. `item/started`，其中 `item.type = "dynamicToolCall"`、`status = "inProgress"`，以及 `tool` 和 `arguments`。
2. `item/tool/call` 作为对客户端的服务器请求。
3. 客户端响应负载，包含返回的内容 item。
4. `item/completed`，其中 `item.type = "dynamicToolCall"`，最终 `status`，以及任何返回的 `contentItems` 或 `success` 值。

### MCP 工具调用审批（apps）

App（连接器）工具调用也可能需要审批。当 app 工具调用有副作用时，服务器可能通过 `tool/requestUserInput` 征求审批，选项包括**接受**、**拒绝**和**取消**。破坏性工具注解始终触发审批，即使工具也声明了较低权限的提示。如果用户拒绝或取消，相关的 `mcpToolCall` item 以错误完成，而不是运行工具。

## 技能

通过在用户文本输入中包含 `$<skill-name>` 来调用技能。添加一个 `skill` 输入 item（推荐），以便服务器注入完整的技能指令，而不是依赖模型来解析名称。

```json
{
  "method": "turn/start",
  "id": 101,
  "params": {
    "threadId": "thread-1",
    "input": [
      {
        "type": "text",
        "text": "$skill-creator 添加一个用于梳理不稳定 CI 的新技能。"
      },
      {
        "type": "skill",
        "name": "skill-creator",
        "path": "/Users/me/.codex/skills/skill-creator/SKILL.md"
      }
    ]
  }
}
```

如果你省略 `skill` item，模型仍然会解析 `$<skill-name>` 标记并尝试定位技能，但这可能增加延迟。

示例：

```
$skill-creator 添加一个用于梳理不稳定 CI 的新技能，并包含逐步使用说明。
```

使用 `skills/list` 获取可用技能（可选地按 `cwds` 限定范围，使用 `forceReload`）。你还可以包含 `perCwdExtraUserRoots` 来为特定的 `cwd` 值扫描额外的绝对路径作为 `user` 范围。App-server 忽略其 `cwd` 不在 `cwds` 中的条目。`skills/list` 可能重用每个 `cwd` 的缓存结果；设置 `forceReload: true` 从磁盘刷新。当存在时，服务器从 `SKILL.json` 读取 `interface` 和 `dependencies`。

```json
{ "method": "skills/list", "id": 25, "params": {
  "cwds": ["/Users/me/project", "/Users/me/other-project"],
  "forceReload": true,
  "perCwdExtraUserRoots": [
    {
      "cwd": "/Users/me/project",
      "extraUserRoots": ["/Users/me/shared-skills"]
    }
  ]
} }
{ "id": 25, "result": {
  "data": [{
    "cwd": "/Users/me/project",
    "skills": [
      {
        "name": "skill-creator",
        "description": "创建或更新 Codex 技能",
        "enabled": true,
        "interface": {
          "displayName": "技能创建器",
          "shortDescription": "创建或更新 Codex 技能"
        },
        "dependencies": {
          "tools": [
            {
              "type": "env_var",
              "value": "GITHUB_TOKEN",
              "description": "GitHub API 令牌"
            },
            {
              "type": "mcp",
              "value": "github",
              "transport": "streamable_http",
              "url": "https://example.com/mcp"
            }
          ]
        }
      }
    ],
    "errors": []
  }]
} }
```

当被监视的本地技能文件发生变化时，服务器也会发出 `skills/changed` 通知。将其视为失效信号，在需要时使用你的当前参数重新运行 `skills/list`。

按路径启用或禁用技能：

```json
{
  "method": "skills/config/write",
  "id": 26,
  "params": {
    "path": "/Users/me/.codex/skills/skill-creator/SKILL.md",
    "enabled": false
  }
}
```

## Apps（连接器）

使用 `app/list` 获取可用的 app。在 CLI/TUI 中，`/apps` 是面向用户的选择器；在自定义客户端中，直接调用 `app/list`。每个条目包含 `isAccessible`（用户可用）和 `isEnabled`（在 `config.toml` 中启用），以便客户端区分安装/访问与本地启用状态。App 条目还可以包含可选的 `branding`、`appMetadata` 和 `labels` 字段。

```json
{ "method": "app/list", "id": 50, "params": {
  "cursor": null,
  "limit": 50,
  "threadId": "thread-1",
  "forceRefetch": false
} }
{ "id": 50, "result": {
  "data": [
    {
      "id": "demo-app",
      "name": "Demo App",
      "description": "文档示例连接器。",
      "logoUrl": "https://example.com/demo-app.png",
      "logoUrlDark": null,
      "distributionChannel": null,
      "branding": null,
      "appMetadata": null,
      "labels": null,
      "installUrl": "https://chatgpt.com/apps/demo-app/demo-app",
      "isAccessible": true,
      "isEnabled": true
    }
  ],
  "nextCursor": null
} }
```

如果你提供 `threadId`，app 功能控制（`features.apps`）使用该 thread 的配置快照。省略时，app-server 使用最新的全局配置。

`app/list` 在可访问 app 和目录 app 都加载完成后返回。设置 `forceRefetch: true` 绕过 app 缓存并获取新数据。缓存条目仅在刷新成功时被替换。

当任一来源（可访问 app 或目录 app）完成加载时，服务器也会发出 `app/list/updated` 通知。每个通知包含最新的合并 app 列表。

```json
{
  "method": "app/list/updated",
  "params": {
    "data": [
      {
        "id": "demo-app",
        "name": "Demo App",
        "description": "文档示例连接器。",
        "logoUrl": "https://example.com/demo-app.png",
        "logoUrlDark": null,
        "distributionChannel": null,
        "branding": null,
        "appMetadata": null,
        "labels": null,
        "installUrl": "https://chatgpt.com/apps/demo-app/demo-app",
        "isAccessible": true,
        "isEnabled": true
      }
    ]
  }
}
```

通过在文本输入中插入 `$<app-slug>` 并添加带有 `app://<id>` 路径的 `mention` 输入 item（推荐）来调用 app。

```json
{
  "method": "turn/start",
  "id": 51,
  "params": {
    "threadId": "thread-1",
    "input": [
      {
        "type": "text",
        "text": "$demo-app 拉取团队的最新更新。"
      },
      {
        "type": "mention",
        "name": "Demo App",
        "path": "app://demo-app"
      }
    ]
  }
}
```

### App 设置的 Config RPC 示例

使用 `config/read`、`config/value/write` 和 `config/batchWrite` 检查或更新 `config.toml` 中的 app 控制项。

读取有效的 app 配置形状（包括 `_default` 和每个工具的覆盖项）：

```json
{ "method": "config/read", "id": 60, "params": { "includeLayers": false } }
{ "id": 60, "result": {
  "config": {
    "apps": {
      "_default": {
        "enabled": true,
        "destructive_enabled": true,
        "open_world_enabled": true
      },
      "google_drive": {
        "enabled": true,
        "destructive_enabled": false,
        "default_tools_approval_mode": "prompt",
        "tools": {
          "files/delete": { "enabled": false, "approval_mode": "approve" }
        }
      }
    }
  }
} }
```

更新单个 app 设置：

```json
{
  "method": "config/value/write",
  "id": 61,
  "params": {
    "keyPath": "apps.google_drive.default_tools_approval_mode",
    "value": "prompt",
    "mergeStrategy": "replace"
  }
}
```

原子地应用多个 app 编辑：

```json
{
  "method": "config/batchWrite",
  "id": 62,
  "params": {
    "edits": [
      {
        "keyPath": "apps._default.destructive_enabled",
        "value": false,
        "mergeStrategy": "upsert"
      },
      {
        "keyPath": "apps.google_drive.tools.files/delete.approval_mode",
        "value": "approve",
        "mergeStrategy": "upsert"
      }
    ]
  }
}
```

### 检测和导入外部代理配置

使用 `externalAgentConfig/detect` 发现可以迁移的外部代理工件，然后将选定的条目传递给 `externalAgentConfig/import`。

检测示例：

```json
{ "method": "externalAgentConfig/detect", "id": 63, "params": {
  "includeHome": true,
  "cwds": ["/Users/me/project"]
} }
{ "id": 63, "result": {
  "items": [
    {
      "itemType": "AGENTS_MD",
      "description": "将 /Users/me/project/CLAUDE.md 导入到 /Users/me/project/AGENTS.md。",
      "cwd": "/Users/me/project"
    },
    {
      "itemType": "SKILLS",
      "description": "将技能文件夹从 /Users/me/.claude/skills 复制到 /Users/me/.agents/skills。",
      "cwd": null
    }
  ]
} }
```

导入示例：

```json
{ "method": "externalAgentConfig/import", "id": 64, "params": {
  "migrationItems": [
    {
      "itemType": "AGENTS_MD",
      "description": "将 /Users/me/project/CLAUDE.md 导入到 /Users/me/project/AGENTS.md。",
      "cwd": "/Users/me/project"
    }
  ]
} }
{ "id": 64, "result": {} }
```

当请求包含插件导入时，服务器在导入完成后发出 `externalAgentConfig/import/completed`。此通知可能在响应后立即到达，也可能在后台远程导入完成后到达。

支持的 `itemType` 值为 `AGENTS_MD`、`CONFIG`、`SKILLS`、`PLUGINS`
和 `MCP_SERVER_CONFIG`。对于 `PLUGINS` 条目，`details.plugins` 列出每个
`marketplaceName` 以及 Codex 可以尝试迁移的 `pluginNames`。检测
仅返回仍有工作要做的条目。例如，当 `AGENTS.md` 已存在且非空时，
Codex 跳过 AGENTS 迁移；技能导入不会覆盖已有的技能目录。

当从 `.claude/settings.json` 检测插件时，Codex 从 `extraKnownMarketplaces`
读取配置的市场源。如果 `enabledPlugins` 包含来自
`claude-plugins-official` 的插件但市场源缺失，Codex 推断
`anthropics/claude-plugins-official` 作为源。

## 认证端点

JSON-RPC 认证/账户表面暴露请求/响应方法以及服务器发起的通知（无 `id`）。使用这些来确定认证状态、启动或取消登录、登出、检查 ChatGPT 速率限制，并通知工作区所有者关于信用耗尽或使用限制的情况。

### 认证模式

Codex 支持以下认证模式。`account/updated.authMode` 显示活跃模式，并在可用时包含当前的 ChatGPT `planType`。`account/read` 也报告账户和计划详情。

- **API 密钥（`apikey`）** — 调用者提供一个 OpenAI API 密钥，类型为 `type: "apiKey"`，Codex 将其存储用于 API 请求。
- **ChatGPT 托管（`chatgpt`）** — Codex 拥有 ChatGPT OAuth 流程，持久化令牌，并自动刷新它们。使用 `type: "chatgpt"` 启动浏览器流程，或 `type: "chatgptDeviceCode"` 启动设备码流程。
- **ChatGPT 外部令牌（`chatgptAuthTokens`）** — 实验性的，适用于已经拥有用户 ChatGPT 认证生命周期的宿主应用。宿主应用直接提供 `accessToken`、`chatgptAccountId` 和可选的 `chatgptPlanType`，并且必须在被要求时刷新令牌。

### API 概览

- `account/read` — 获取当前账户信息；可选地刷新令牌。
- `account/login/start` — 开始登录（`apiKey`、`chatgpt`、`chatgptDeviceCode` 或实验性的 `chatgptAuthTokens`）。
- `account/login/completed`（通知）— 当登录尝试完成时发出（成功或错误）。
- `account/login/cancel` — 按 `loginId` 取消待处理的托管 ChatGPT 登录。
- `account/logout` — 登出；触发 `account/updated`。
- `account/updated`（通知）— 每当认证模式变化时发出（`authMode`：`apikey`、`chatgpt`、`chatgptAuthTokens` 或 `null`），并在可用时包含 `planType`。
- `account/chatgptAuthTokens/refresh`（服务器请求）— 在授权错误后请求刷新外部管理的 ChatGPT 令牌。
- `account/rateLimits/read` — 获取 ChatGPT 速率限制。
- `account/rateLimits/updated`（通知）— 每当用户的 ChatGPT 速率限制变化时发出。
- `account/sendAddCreditsNudgeEmail` — 请求 ChatGPT 向工作区所有者发送关于信用耗尽或达到使用限制的邮件。
- `mcpServer/oauthLogin/completed`（通知）— 在 `mcpServer/oauth/login` 流程完成后发出；负载包含 `{ name, success, error? }`。
- `mcpServer/startupStatus/updated`（通知）— 当配置的 MCP 服务器对已加载 thread 的启动状态变化时发出；负载包含 `{ name, status, error }`。

### 1) 检查认证状态

请求：

```json
{ "method": "account/read", "id": 1, "params": { "refreshToken": false } }
```

响应示例：

```json
{ "id": 1, "result": { "account": null, "requiresOpenaiAuth": false } }
```

```json
{ "id": 1, "result": { "account": null, "requiresOpenaiAuth": true } }
```

```json
{
  "id": 1,
  "result": { "account": { "type": "apiKey" }, "requiresOpenaiAuth": true }
}
```

```json
{
  "id": 1,
  "result": {
    "account": {
      "type": "chatgpt",
      "email": "user@example.com",
      "planType": "pro"
    },
    "requiresOpenaiAuth": true
  }
}
```

字段说明：

- `refreshToken`（布尔值）：设为 `true` 以在托管 ChatGPT 模式下强制刷新令牌。在外部令牌模式（`chatgptAuthTokens`）下，app-server 忽略此标志。
- `requiresOpenaiAuth` 反映活跃的提供商；为 `false` 时，Codex 可以在没有 OpenAI 凭证的情况下运行。

### 2) 使用 API 密钥登录

1. 发送：

   ```json
   {
     "method": "account/login/start",
     "id": 2,
     "params": { "type": "apiKey", "apiKey": "sk-..." }
   }
   ```

2. 期望：

   ```json
   { "id": 2, "result": { "type": "apiKey" } }
   ```

3. 通知：

   ```json
   {
     "method": "account/login/completed",
     "params": { "loginId": null, "success": true, "error": null }
   }
   ```

   ```json
   {
     "method": "account/updated",
     "params": { "authMode": "apikey", "planType": null }
   }
   ```

### 3) 使用 ChatGPT 登录（浏览器流程）

1. 启动：

   ```json
   { "method": "account/login/start", "id": 3, "params": { "type": "chatgpt" } }
   ```

   ```json
   {
     "id": 3,
     "result": {
       "type": "chatgpt",
       "loginId": "<uuid>",
       "authUrl": "https://chatgpt.com/...&redirect_uri=http%3A%2F%2Flocalhost%3A<port>%2Fauth%2Fcallback"
     }
   }
   ```

2. 在浏览器中打开 `authUrl`；app-server 托管本地回调。
3. 等待通知：

   ```json
   {
     "method": "account/login/completed",
     "params": { "loginId": "<uuid>", "success": true, "error": null }
   }
   ```

   ```json
   {
     "method": "account/updated",
     "params": { "authMode": "chatgpt", "planType": "plus" }
   }
   ```

### 3b) 使用 ChatGPT 登录（设备码流程）

当你的客户端拥有登录仪式或浏览器回调不可靠时使用此流程。

1. 启动：

   ```json
   {
     "method": "account/login/start",
     "id": 4,
     "params": { "type": "chatgptDeviceCode" }
   }
   ```

   ```json
   {
     "id": 4,
     "result": {
       "type": "chatgptDeviceCode",
       "loginId": "<uuid>",
       "verificationUrl": "https://auth.openai.com/codex/device",
       "userCode": "ABCD-1234"
     }
   }
   ```

2. 向用户展示 `verificationUrl` 和 `userCode`；前端拥有 UX。
3. 等待通知：

   ```json
   {
     "method": "account/login/completed",
     "params": { "loginId": "<uuid>", "success": true, "error": null }
   }
   ```

   ```json
   {
     "method": "account/updated",
     "params": { "authMode": "chatgpt", "planType": "plus" }
   }
   ```

### 3c) 使用外部管理的 ChatGPT 令牌登录（`chatgptAuthTokens`）

仅当宿主应用程序拥有用户的 ChatGPT 认证生命周期并直接提供令牌时，才使用此实验性模式。客户端必须在 `initialize` 期间设置 `capabilities.experimentalApi = true` 才能使用此登录类型。

1. 发送：

   ```json
   {
     "method": "account/login/start",
     "id": 7,
     "params": {
       "type": "chatgptAuthTokens",
       "accessToken": "<jwt>",
       "chatgptAccountId": "org-123",
       "chatgptPlanType": "business"
     }
   }
   ```

2. 期望：

   ```json
   { "id": 7, "result": { "type": "chatgptAuthTokens" } }
   ```

3. 通知：

   ```json
   {
     "method": "account/login/completed",
     "params": { "loginId": null, "success": true, "error": null }
   }
   ```

   ```json
   {
     "method": "account/updated",
     "params": { "authMode": "chatgptAuthTokens", "planType": "business" }
   }
   ```

当服务器收到 `401 Unauthorized` 时，它可能向宿主应用请求刷新后的令牌：

```json
{
  "method": "account/chatgptAuthTokens/refresh",
  "id": 8,
  "params": { "reason": "unauthorized", "previousAccountId": "org-123" }
}
{ "id": 8, "result": { "accessToken": "<jwt>", "chatgptAccountId": "org-123", "chatgptPlanType": "business" } }
```

服务器在成功的刷新响应后重试原始请求。请求大约 10 秒后超时。

### 4) 取消 ChatGPT 登录

```json
{ "method": "account/login/cancel", "id": 4, "params": { "loginId": "<uuid>" } }
{ "method": "account/login/completed", "params": { "loginId": "<uuid>", "success": false, "error": "..." } }
```

### 5) 登出

```json
{ "method": "account/logout", "id": 5 }
{ "id": 5, "result": {} }
{ "method": "account/updated", "params": { "authMode": null, "planType": null } }
```

### 6) 速率限制（ChatGPT）

```json
{ "method": "account/rateLimits/read", "id": 6 }
{ "id": 6, "result": {
  "rateLimits": {
    "limitId": "codex",
    "limitName": null,
    "primary": { "usedPercent": 25, "windowDurationMins": 15, "resetsAt": 1730947200 },
    "secondary": null,
    "rateLimitReachedType": null
  },
  "rateLimitsByLimitId": {
    "codex": {
      "limitId": "codex",
      "limitName": null,
      "primary": { "usedPercent": 25, "windowDurationMins": 15, "resetsAt": 1730947200 },
      "secondary": null,
      "rateLimitReachedType": null
    },
    "codex_other": {
      "limitId": "codex_other",
      "limitName": "codex_other",
      "primary": { "usedPercent": 42, "windowDurationMins": 60, "resetsAt": 1730950800 },
      "secondary": null,
      "rateLimitReachedType": null
    }
  }
} }
{ "method": "account/rateLimits/updated", "params": {
  "rateLimits": {
    "limitId": "codex",
    "primary": { "usedPercent": 31, "windowDurationMins": 15, "resetsAt": 1730948100 }
  }
} }
```

字段说明：

- `rateLimits` 是向后兼容的单桶视图。
- `rateLimitsByLimitId`（当存在时）是按计量的 `limit_id`（例如 `codex`）键控的多桶视图。
- `limitId` 是计量的桶标识符。
- `limitName` 是桶的可选面向用户标签。
- `usedPercent` 是配额窗口内的当前使用率。
- `windowDurationMins` 是配额窗口长度。
- `resetsAt` 是下次重置的 Unix 时间戳（秒）。
- 当后端返回与桶关联的 ChatGPT 计划时，包含 `planType`。
- 当后端返回剩余的工作区信用详情时，包含 `credits`。
- `rateLimitReachedType` 标识达到限制时的后端分类限制状态。

### 7) 通知工作区所有者关于限制

使用 `account/sendAddCreditsNudgeEmail` 请求 ChatGPT 在信用耗尽或达到使用限制时向工作区所有者发送邮件。

```json
{ "method": "account/sendAddCreditsNudgeEmail", "id": 7, "params": { "creditType": "credits" } }
{ "id": 7, "result": { "status": "sent" } }
```

当工作区信用耗尽时使用 `creditType: "credits"`，当工作区使用限制达到时使用 `creditType: "usage_limit"`。如果所有者最近已被通知，响应状态为 `cooldown_active`。
