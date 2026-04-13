---
sidebar_position: 4
---
# 以编程方式运行 Claude Code

> 使用 Agent SDK 从 CLI、Python 或 TypeScript 以编程方式运行 Claude Code。

[Agent SDK](/en/agent-sdk/overview) 为你提供驱动 Claude Code 的相同工具、代理循环和上下文管理。它可作为 CLI 用于脚本和 CI/CD，或作为 [Python](/en/agent-sdk/python) 和 [TypeScript](/en/agent-sdk/typescript) 包用于完整的编程控制。

> **注意**：CLI 以前称为"无头模式"。`-p` 标志和所有 CLI 选项的工作方式相同。

要从 CLI 以编程方式运行 Claude Code，传递 `-p` 和你的提示以及任何 [CLI 选项](/en/cli-reference)：

```bash
claude -p "Find and fix the bug in auth.py" --allowedTools "Read,Edit,Bash"
```

本页涵盖通过 CLI（`claude -p`）使用 Agent SDK。对于带结构化输出、工具批准回调和原生消息对象的 Python 和 TypeScript SDK 包，见[完整 Agent SDK 文档](/en/agent-sdk/overview)。

---

## 基本用法

添加 `-p`（或 `--print`）标志到任何 `claude` 命令以非交互方式运行它。所有 [CLI 选项](/en/cli-reference) 都可以与 `-p` 一起使用，包括：

- `--continue` 用于[继续对话](#继续对话)
- `--allowedTools` 用于[自动批准工具](#自动批准工具)
- `--output-format` 用于[结构化输出](#获取结构化输出)

这个示例询问 Claude 关于你的代码库的问题并打印响应：

```bash
claude -p "What does the auth module do?"
```

### 使用 bare 模式更快启动

添加 `--bare` 通过跳过 hooks、技能、插件、MCP 服务器、自动记忆和 CLAUDE.md 的自动发现来减少启动时间。没有它，`claude -p` 会加载交互会话相同的[上下文](/en/how-claude-code-works#the-context-window)，包括工作目录或 `~/.claude` 中配置的任何内容。

Bare 模式对 CI 和脚本很有用，你在每台机器上需要相同的结果。队友的 `~/.claude` 中的 hook 或项目 `.mcp.json` 中的 MCP 服务器不会运行，因为 bare 模式从不读取它们。只有你显式传递的标志生效。

这个示例在 bare 模式中运行一次性总结任务并预批准 Read 工具，使调用无需权限提示即可完成：

```bash
claude --bare -p "Summarize this file" --allowedTools "Read"
```

在 bare 模式中，Claude 可以访问 Bash、文件读取和文件编辑工具。用标志传递你需要的任何上下文：

| 要加载的内容          | 使用                                                    |
| --------------------- | ------------------------------------------------------- |
| 系统提示补充          | `--append-system-prompt`、`--append-system-prompt-file` |
| 设置                  | `--settings <file-or-json>`                             |
| MCP 服务器            | `--mcp-config <file-or-json>`                           |
| 自定义代理            | `--agents <json>`                                       |
| 插件目录              | `--plugin-dir <path>`                                   |

Bare 模式跳过 OAuth 和钥匙串读取。Anthropic 认证必须来自 `ANTHROPIC_API_KEY` 或传递给 `--settings` 的 JSON 中的 `apiKeyHelper`。Bedrock、Vertex 和 Foundry 使用它们通常的提供商凭据。

> **注意**：`--bare` 是脚本和 SDK 调用的推荐模式，将在未来的版本中成为 `-p` 的默认模式。

---

## 示例

这些示例突出显示了常见的 CLI 模式。对于 CI 和其他脚本调用，添加 [`--bare`](#使用-bare-模式更快启动)，这样它们不会获取本地配置的任何内容。

### 获取结构化输出

使用 `--output-format` 控制响应的返回方式：

- `text`（默认）：纯文本输出
- `json`：带结果、会话 ID 和元数据的结构化 JSON
- `stream-json`：用于实时流式传输的换行符分隔 JSON

这个示例以 JSON 返回项目摘要，带会话元数据，文本结果在 `result` 字段中：

```bash
claude -p "Summarize this project" --output-format json
```

要获取符合特定 schema 的输出，使用 `--output-format json` 与 `--json-schema` 和 [JSON Schema](https://json-schema.org/) 定义。响应包含关于请求的元数据（会话 ID、用量等），结构化输出在 `structured_output` 字段中。

这个示例提取函数名并将它们作为字符串数组返回：

```bash
claude -p "Extract the main function names from auth.py" \
  --output-format json \
  --json-schema '{"type":"object","properties":{"functions":{"type":"array","items":{"type":"string"}}},"required":["functions"]}'
```

> **提示**：使用 [jq](https://jqlang.github.io/jq/) 等工具解析响应并提取特定字段：
>
> ```bash
> # 提取文本结果
> claude -p "Summarize this project" --output-format json | jq -r '.result'
>
> # 提取结构化输出
> claude -p "Extract function names from auth.py" \
>   --output-format json \
>   --json-schema '{"type":"object","properties":{"functions":{"type":"array","items":{"type":"string"}}},"required":["functions"]}' \
>   | jq '.structured_output'
> ```

### 流式响应

使用 `--output-format stream-json` 与 `--verbose` 和 `--include-partial-messages` 在生成时接收 token。每行是一个表示事件的 JSON 对象：

```bash
claude -p "Explain recursion" --output-format stream-json --verbose --include-partial-messages
```

下面的示例使用 [jq](https://jqlang.github.io/jq/) 过滤文本增量并仅显示流式文本。`-r` 标志输出原始字符串（无引号），`-j` 连接时不带换行符，因此 token 连续流式传输：

```bash
claude -p "Write a poem" --output-format stream-json --verbose --include-partial-messages | \
  jq -rj 'select(.type == "stream_event" and .event.delta.type? == "text_delta") | .event.delta.text'
```

当 API 请求因可重试错误失败时，Claude Code 在重试之前发出 `system/api_retry` 事件。你可以使用它来显示重试进度或实现自定义退避逻辑。

| 字段             | 类型                | 描述                                                                                                               |
| ---------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `type`           | `"system"`          | 消息类型                                                                                                           |
| `subtype`        | `"api_retry"`       | 标识这是重试事件                                                                                                   |
| `attempt`        | 整数                | 当前尝试次数，从 1 开始                                                                                            |
| `max_retries`    | 整数                | 允许的总重试次数                                                                                                   |
| `retry_delay_ms` | 整数                | 下次尝试前的毫秒数                                                                                                 |
| `error_status`   | 整数或 null         | HTTP 状态码，或对没有 HTTP 响应的连接错误为 `null`                                                                  |
| `error`          | 字符串              | 错误类别：`authentication_failed`、`billing_error`、`rate_limit`、`invalid_request`、`server_error`、`max_output_tokens` 或 `unknown` |
| `uuid`           | 字符串              | 唯一事件标识符                                                                                                     |
| `session_id`     | 字符串              | 事件所属的会话                                                                                                     |

有关带回调和消息对象的编程流式传输，见 Agent SDK 文档中的[实时流式响应](/en/agent-sdk/streaming-output)。

### 自动批准工具

使用 `--allowedTools` 让 Claude 使用某些工具而无需提示。这个示例运行测试套件并修复失败，允许 Claude 执行 Bash 命令和读取/编辑文件而无需请求许可：

```bash
claude -p "Run the test suite and fix any failures" \
  --allowedTools "Bash,Read,Edit"
```

要为整个会话设置基线而不是列出单个工具，传递[权限模式](/en/permission-modes)。`dontAsk` 拒绝不在你的 `permissions.allow` 规则中的任何内容，这对锁定的 CI 运行很有用。`acceptEdits` 让 Claude 无需提示即可写入文件，并自动批准常见文件系统命令如 `mkdir`、`touch`、`mv` 和 `cp`。其他 shell 命令和网络请求仍然需要 `--allowedTools` 条目或 `permissions.allow` 规则，否则在尝试时会中止运行：

```bash
claude -p "Apply the lint fixes" --permission-mode acceptEdits
```

### 创建提交

这个示例审查暂存的更改并创建带适当消息的提交：

```bash
claude -p "Look at my staged changes and create an appropriate commit" \
  --allowedTools "Bash(git diff *),Bash(git log *),Bash(git status *),Bash(git commit *)"
```

`--allowedTools` 标志使用[权限规则语法](/en/settings#permission-rule-syntax)。尾随的 ` *` 启用前缀匹配，因此 `Bash(git diff *)` 允许以 `git diff` 开头的任何命令。`*` 前面的空格很重要：没有它，`Bash(git diff*)` 也会匹配 `git diff-index`。

> **注意**：用户调用的[技能](/en/skills)如 `/commit` 和[内置命令](/en/commands)仅在交互模式中可用。在 `-p` 模式中，描述你想要完成的任务。

### 自定义系统提示

使用 `--append-system-prompt` 添加指令同时保持 Claude Code 的默认行为。这个示例将 PR diff 管道传输给 Claude 并指示它审查安全漏洞：

```bash
gh pr diff "$1" | claude -p \
  --append-system-prompt "You are a security engineer. Review for vulnerabilities." \
  --output-format json
```

有关更多选项见[系统提示标志](/en/cli-reference#system-prompt-flags)，包括 `--system-prompt` 以完全替换默认提示。

### 继续对话

使用 `--continue` 继续最近的对话，或 `--resume` 带会话 ID 继续特定对话。这个示例运行审查，然后发送后续提示：

```bash
# 第一次请求
claude -p "Review this codebase for performance issues"

# 继续最近的对话
claude -p "Now focus on the database queries" --continue
claude -p "Generate a summary of all issues found" --continue
```

如果你正在运行多个对话，捕获会话 ID 以恢复特定的对话：

```bash
session_id=$(claude -p "Start a review" --output-format json | jq -r '.session_id')
claude -p "Continue that review" --resume "$session_id"
```

---

## 下一步

- [Agent SDK 快速入门](/en/agent-sdk/quickstart)：用 Python 或 TypeScript 构建你的第一个代理
- [CLI 参考](/en/cli-reference)：所有 CLI 标志和选项
- [GitHub Actions](/en/github-actions)：在 GitHub 工作流中使用 Agent SDK
- [GitLab CI/CD](/en/gitlab-ci-cd)：在 GitLab 流水线中使用 Agent SDK

---

## 提交反馈

如果你发现本文档中有不正确、过时或令人困惑的内容，请通过 POST 提交反馈到：
https://code.claude.com/docs/_mintlify/feedback/claude-code/agent-feedback

请求体（JSON）：`{ "path": "/current-page-path", "feedback": "问题描述" }`

仅在你有具体且可操作的报告时提交反馈。
