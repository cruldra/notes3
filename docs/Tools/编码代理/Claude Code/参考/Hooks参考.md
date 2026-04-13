---
sidebar_position: 4
---

# Hooks 参考

> Claude Code hook 事件、配置 schema、JSON 输入/输出格式、退出码、异步 hooks、HTTP hooks、提示 hooks 和 MCP 工具 hooks 的参考。

> **提示**：有关带示例的快速入门指南，见[用 Hooks 自动化工作流](/en/hooks-guide)。

Hooks 是在 Claude Code 生命周期特定节点自动执行的用户定义 shell 命令、HTTP 端点或 LLM 提示。使用此参考查找事件 schema、配置选项、JSON 输入/输出格式以及异步 hooks、HTTP hooks 和 MCP 工具 hooks 等高级功能。如果你首次设置 hooks，请先从[指南](/en/hooks-guide)开始。

---

## Hook 生命周期

Hooks 在 Claude Code 会话期间的特定点触发。当事件触发且匹配器匹配时，Claude Code 将关于事件的 JSON 上下文传递给 hook 处理器。对于命令 hooks，输入通过 stdin 到达。对于 HTTP hooks，它作为 POST 请求体到达。你的处理器可以检查输入、执行操作并可选地返回决策。事件分为三种节奏：每会话一次（`SessionStart`、`SessionEnd`）、每轮一次（`UserPromptSubmit`、`Stop`、`StopFailure`）和代理循环中的每次工具调用（`PreToolUse`、`PostToolUse`）。

下表总结了每个事件的触发时机。[Hook 事件](#hook-事件)部分记录了每个事件的完整输入 schema 和决策控制选项。

| 事件                 | 触发时机                                                                                                                                             |
| :------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SessionStart`       | 会话开始或恢复时                                                                                                                                     |
| `UserPromptSubmit`   | 你提交提示时，在 Claude 处理之前                                                                                                |
| `PreToolUse`         | 工具调用执行之前。可以阻止它                                                                                                                         |
| `PermissionRequest`  | 出现权限对话框时                                                                                                                                     |
| `PermissionDenied`   | 工具调用被自动模式分类器拒绝时。返回 `{retry: true}` 告诉模型可以重试被拒绝的工具调用                                                                |
| `PostToolUse`        | 工具调用成功后                                                                                                                                       |
| `PostToolUseFailure` | 工具调用失败后                                                                                                                                       |
| `Notification`       | Claude Code 发送通知时                                                                                                                               |
| `SubagentStart`      | 生成子代理时                                                                                                                                         |
| `SubagentStop`       | 子代理完成时                                                                                                                                         |
| `TaskCreated`        | 通过 `TaskCreate` 创建任务时                                                                                                                         |
| `TaskCompleted`      | 任务被标记为完成时                                                                                                                                   |
| `Stop`               | Claude 完成响应时                                                                                                                                    |
| `StopFailure`        | 轮次因 API 错误结束时。输出和退出码被忽略                                                                                                             |
| `TeammateIdle`       | [代理团队](/en/agent-teams)队友即将进入空闲时                                                                                                        |
| `InstructionsLoaded` | CLAUDE.md 或 `.claude/rules/*.md` 文件加载到上下文时。在会话开始时和会话期间延迟加载文件时触发                                                       |
| `ConfigChange`       | 会话期间配置文件更改时                                                                                                                               |
| `CwdChanged`         | 工作目录更改时，例如当 Claude 执行 `cd` 命令时。用于与 direnv 等工具的反应性环境管理                                                                  |
| `FileChanged`        | 磁盘上监视的文件更改时。`matcher` 字段指定要监视哪些文件名                                                                                           |
| `WorktreeCreate`     | 通过 `--worktree` 或 `isolation: "worktree"` 创建 worktree 时。替换默认 git 行为                                                                    |
| `WorktreeRemove`     | 移除 worktree 时，在会话退出或子代理完成时                                                                                                           |
| `PreCompact`         | 上下文压缩之前                                                                                                                                       |
| `PostCompact`        | 上下文压缩完成后                                                                                                                                     |
| `Elicitation`        | MCP 服务器在工具调用期间请求用户输入时                                                                                                               |
| `ElicitationResult`  | 用户响应 MCP 诱导后，在响应发送回服务器之前                                                                                                          |
| `SessionEnd`         | 会话终止时                                                                                                                                           |

### Hook 如何解析

要看看这些部分如何组合在一起，考虑这个阻止破坏性 shell 命令的 `PreToolUse` hook。`matcher` 缩小到 Bash 工具调用，`if` 条件进一步缩小到以 `rm` 开头的命令，因此 `block-rm.sh` 仅在两个过滤器都匹配时才生成：

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "if": "Bash(rm *)",
            "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/block-rm.sh"
          }
        ]
      }
    ]
  }
}
```

脚本从 stdin 读取 JSON 输入，提取命令，如果包含 `rm -rf` 则返回 `"deny"` 的 `permissionDecision`：

```bash
#!/bin/bash
# .claude/hooks/block-rm.sh
COMMAND=$(jq -r '.tool_input.command')

if echo "$COMMAND" | grep -q 'rm -rf'; then
  jq -n '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: "Destructive command blocked by hook"
    }
  }'
else
  exit 0  # allow the command
fi
```

现在假设 Claude Code 决定运行 `Bash "rm -rf /tmp/build"`。以下是发生的情况：

**步骤 1：事件触发**

`PreToolUse` 事件触发。Claude Code 将工具输入作为 JSON 通过 stdin 发送到 hook：

```json
{ "tool_name": "Bash", "tool_input": { "command": "rm -rf /tmp/build" }, ... }
```

**步骤 2：匹配器检查**

匹配器 `"Bash"` 匹配工具名称，因此激活此 hook 组。如果你省略匹配器或使用 `"*"`，该组会在事件的每次出现时激活。

**步骤 3：If 条件检查**

`if` 条件 `"Bash(rm *)"` 匹配，因为命令以 `rm` 开头，因此生成此处理器。如果命令是 `npm test`，`if` 检查会失败且 `block-rm.sh` 永远不会运行，避免进程生成开销。`if` 字段是可选的；没有它，匹配组中的每个处理器都会运行。

**步骤 4：Hook 处理器运行**

脚本检查完整命令并找到 `rm -rf`，因此将决策打印到 stdout：

```json
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "deny",
    "permissionDecisionReason": "Destructive command blocked by hook"
  }
}
```

如果命令是更安全的 `rm` 变体如 `rm file.txt`，脚本会命中 `exit 0`，这告诉 Claude Code 允许工具调用而不需要进一步操作。

**步骤 5：Claude Code 对结果执行操作**

Claude Code 读取 JSON 决策，阻止工具调用，并向 Claude 显示原因。

---

## 配置

Hooks 在 JSON 设置文件中定义。配置有三个级别的嵌套：

1. 选择要响应的 [hook 事件](#hook-事件)，如 `PreToolUse` 或 `Stop`
2. 添加 [匹配器组](#匹配器模式) 来过滤何时触发，如"仅适用于 Bash 工具"
3. 定义一个或多个 [hook 处理器](#hook-处理器字段) 在匹配时运行

### Hook 位置

你定义 hook 的位置决定了它的作用域：

| 位置                                                    | 作用域                | 可共享                           |
| :------------------------------------------------------ | :-------------------- | :------------------------------- |
| `~/.claude/settings.json`                               | 你的所有项目          | 否，限于你的机器                 |
| `.claude/settings.json`                                 | 单个项目              | 是，可以提交到仓库               |
| `.claude/settings.local.json`                           | 单个项目              | 否，被 gitignore                 |
| Managed 策略设置                                        | 组织范围              | 是，管理员控制                   |
| [插件](/en/plugins) `hooks/hooks.json`                  | 插件启用时            | 是，与插件捆绑                   |
| [技能](/en/skills) 或 [代理](/en/sub-agents) frontmatter | 组件激活时            | 是，定义在组件文件中             |

### 匹配器模式

`matcher` 字段过滤 hooks 何时触发。匹配器的评估方式取决于它包含的字符：

| 匹配器值                            | 评估为                                                | 示例                                                                                                           |
| :---------------------------------- | :---------------------------------------------------- | :------------------------------------------------------------------------------------------------------------- |
| `"*"`、`""` 或省略                  | 匹配所有                                              | 在事件的每次出现时触发                                                                                         |
| 仅字母、数字、`_` 和 `\|`           | 精确字符串，或 `\|` 分隔的精确字符串列表              | `Bash` 仅匹配 Bash 工具；`Edit\|Write` 精确匹配任一工具                                                        |
| 包含任何其他字符                    | JavaScript 正则表达式                                 | `^Notebook` 匹配以 Notebook 开头的任何工具；`mcp__memory__.*` 匹配来自 `memory` 服务器的每个工具               |

`FileChanged` 事件在构建其监视列表时不遵循这些规则。见 [FileChanged](#filechanged)。

每种事件类型匹配不同的字段：

| 事件                                                                                                               | 匹配器过滤的内容                                           | 示例匹配器值                                                                                                      |
| :----------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------- |
| `PreToolUse`、`PostToolUse`、`PostToolUseFailure`、`PermissionRequest`、`PermissionDenied`                         | 工具名称                                                   | `Bash`、`Edit\|Write`、`mcp__.*`                                                                                  |
| `SessionStart`                                                                                                     | 会话如何启动                                               | `startup`、`resume`、`clear`、`compact`                                                                           |
| `SessionEnd`                                                                                                       | 会话为什么结束                                             | `clear`、`resume`、`logout`、`prompt_input_exit`、`bypass_permissions_disabled`、`other`                          |
| `Notification`                                                                                                     | 通知类型                                                   | `permission_prompt`、`idle_prompt`、`auth_success`、`elicitation_dialog`                                          |
| `SubagentStart`                                                                                                    | 代理类型                                                   | `Bash`、`Explore`、`Plan` 或自定义代理名称                                                                        |
| `PreCompact`、`PostCompact`                                                                                        | 什么触发压缩                                               | `manual`、`auto`                                                                                                  |
| `SubagentStop`                                                                                                     | 代理类型                                                   | 与 `SubagentStart` 相同的值                                                                                       |
| `ConfigChange`                                                                                                     | 配置源                                                     | `user_settings`、`project_settings`、`local_settings`、`policy_settings`、`skills`                                |
| `CwdChanged`                                                                                                       | 不支持匹配器                                               | 每次目录更改时始终触发                                                                                            |
| `FileChanged`                                                                                                      | 要监视的字面文件名（见 [FileChanged](#filechanged)）        | `.envrc\|.env`                                                                                                    |
| `StopFailure`                                                                                                      | 错误类型                                                   | `rate_limit`、`authentication_failed`、`billing_error`、`invalid_request`、`server_error`、`max_output_tokens`、`unknown` |
| `InstructionsLoaded`                                                                                               | 加载原因                                                   | `session_start`、`nested_traversal`、`path_glob_match`、`include`、`compact`                                      |
| `Elicitation`                                                                                                      | MCP 服务器名称                                             | 你配置的 MCP 服务器名称                                                                                           |
| `ElicitationResult`                                                                                                | MCP 服务器名称                                             | 与 `Elicitation` 相同的值                                                                                         |
| `UserPromptSubmit`、`Stop`、`TeammateIdle`、`TaskCreated`、`TaskCompleted`、`WorktreeCreate`、`WorktreeRemove`    | 不支持匹配器                                               | 每次出现时始终触发                                                                                                |

#### 匹配 MCP 工具

[MCP](/en/mcp) 服务器工具在工具事件中显示为常规工具，因此你可以像匹配任何其他工具名称一样匹配它们。

MCP 工具遵循命名模式 `mcp__<server>__<tool>`，例如：

- `mcp__memory__create_entities`：Memory 服务器的创建实体工具
- `mcp__filesystem__read_file`：文件系统服务器的读取文件工具
- `mcp__github__search_repositories`：GitHub 服务器的搜索工具

要匹配来自服务器的每个工具，在服务器前缀后附加 `.*`。`.*` 是必需的：像 `mcp__memory` 这样的匹配器仅包含字母和下划线，因此它作为精确字符串比较且不匹配任何工具。

- `mcp__memory__.*` 匹配来自 `memory` 服务器的所有工具
- `mcp__.*__write.*` 匹配来自任何服务器名称以 `write` 开头的任何工具

### Hook 处理器字段

内部 `hooks` 数组中的每个对象都是一个 hook 处理器：匹配器匹配时运行的 shell 命令、HTTP 端点、LLM 提示或代理。有四种类型：

- **命令 hooks**（`type: "command"`）：运行 shell 命令
- **HTTP hooks**（`type: "http"`）：将事件的 JSON 输入作为 HTTP POST 请求发送到 URL
- **提示 hooks**（`type: "prompt"`）：向 Claude 模型发送提示进行单轮评估
- **代理 hooks**（`type: "agent"`）：生成可以使用工具验证条件的子代理

#### 通用字段

适用于所有 hook 类型：

| 字段            | 必需 | 描述                                                                                                                                                                                                                                       |
| :-------------- | :--- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `type`          | 是   | `"command"`、`"http"`、`"prompt"` 或 `"agent"`                                                                                                                                                                                             |
| `if`            | 否   | 权限规则语法来过滤此 hook 何时运行，如 `"Bash(git *)"` 或 `"Edit(*.ts)"`。仅在工具事件上评估：`PreToolUse`、`PostToolUse`、`PostToolUseFailure`、`PermissionRequest` 和 `PermissionDenied`。在其他事件上，设置了 `if` 的 hook 永远不会运行 |
| `timeout`       | 否   | 取消前的秒数。默认值：command 为 600，prompt 为 30，agent 为 60                                                                                                                                                                            |
| `statusMessage` | 否   | hook 运行时显示的自定义旋转器消息                                                                                                                                                                                                          |
| `once`          | 否   | 如果为 `true`，每会话仅运行一次然后被移除。仅限技能，不适用于代理                                                                                                                                                                          |

#### 命令 hook 字段

| 字段      | 必需 | 描述                                                                                                                                                                                            |
| :-------- | :--- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `command` | 是   | 要执行的 shell 命令                                                                                                                                                                             |
| `async`   | 否   | 如果为 `true`，在后台运行而不阻塞。见[在后台运行 hooks](#在后台运行-hooks)                                                                                                                      |
| `shell`   | 否   | 用于此 hook 的 shell。接受 `"bash"`（默认）或 `"powershell"`。设置 `"powershell"` 时在 Windows 上通过 PowerShell 运行命令。不需要 `CLAUDE_CODE_USE_POWERSHELL_TOOL`，因为 hooks 直接生成 PowerShell |

#### HTTP hook 字段

| 字段             | 必需 | 描述                                                                                                                                                                            |
| :--------------- | :--- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `url`            | 是   | 发送 POST 请求的 URL                                                                                                                                                            |
| `headers`        | 否   | 附加的 HTTP 头部作为键值对。值支持使用 `$VAR_NAME` 或 `${VAR_NAME}` 语法的环境变量插值。仅解析列在 `allowedEnvVars` 中的变量                                                    |
| `allowedEnvVars` | 否   | 可以插值到头部值中的环境变量名列表。对未列出变量的引用会被替换为空字符串。任何环境变量插值都需要此项                                                                              |

#### 提示和代理 hook 字段

| 字段     | 必需 | 描述                                                                                |
| :------- | :--- | :---------------------------------------------------------------------------------- |
| `prompt` | 是   | 发送给模型的提示文本。使用 `$ARGUMENTS` 作为 hook 输入 JSON 的占位符                |
| `model`  | 否   | 用于评估的模型。默认为快速模型                                                      |

### 通过路径引用脚本

使用环境变量引用 hook 脚本，无论 hook 运行时的工作目录是什么：

- `$CLAUDE_PROJECT_DIR`：项目根目录。用引号包装以处理带空格的路径。
- `${CLAUDE_PLUGIN_ROOT}`：插件的安装目录，用于与[插件](/en/plugins)捆绑的脚本。
- `${CLAUDE_PLUGIN_DATA}`：插件的[持久数据目录](/en/plugins-reference#persistent-data-directory)，用于应在插件更新后保留的依赖项和状态。

### 技能和代理中的 hooks

除了设置文件和插件之外，hooks 可以直接在[技能](/en/skills)和[子代理](/en/sub-agents)中使用 frontmatter 定义。这些 hooks 作用域到组件的生命周期，仅在该组件激活时运行。

```yaml
---
name: secure-operations
description: Perform operations with security checks
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/security-check.sh"
---
```

### `/hooks` 菜单

在 Claude Code 中输入 `/hooks` 打开已配置 hooks 的只读浏览器。菜单显示每个 hook 事件及已配置 hooks 的计数，让你深入查看匹配器，并显示每个 hook 处理器的完整详情。

菜单显示所有四种 hook 类型：`command`、`prompt`、`agent` 和 `http`。每个 hook 标有 `[type]` 前缀和来源指示器：

- `User`：来自 `~/.claude/settings.json`
- `Project`：来自 `.claude/settings.json`
- `Local`：来自 `.claude/settings.local.json`
- `Plugin`：来自插件的 `hooks/hooks.json`
- `Session`：为当前会话在内存中注册
- `Built-in`：由 Claude Code 内部注册

### 禁用或移除 hooks

要移除 hook，从设置 JSON 文件中删除其条目。

要临时禁用所有 hooks 而不移除它们，在设置文件中设置 `"disableAllHooks": true`。无法在保留配置的同时禁用单个 hook。

---

## Hook 输入和输出

命令 hooks 通过 stdin 接收 JSON 数据并通过退出码、stdout 和 stderr 通信结果。HTTP hooks 接收相同的 JSON 作为 POST 请求体并通过 HTTP 响应体通信结果。

### 通用输入字段

| 字段              | 描述                                                                                                                                                                                        |
| :---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `session_id`      | 当前会话标识符                                                                                                                                                                              |
| `transcript_path` | 对话 JSON 路径                                                                                                                                                                              |
| `cwd`             | hook 调用时的当前工作目录                                                                                                                                                                   |
| `permission_mode` | 当前[权限模式](/en/permissions#permission-modes)：`"default"`、`"plan"`、`"acceptEdits"`、`"auto"`、`"dontAsk"` 或 `"bypassPermissions"`。不是所有事件都接收此字段                           |
| `hook_event_name` | 触发的事件名称                                                                                                                                                                              |

当使用 `--agent` 运行或在子代理内部时，包含两个额外字段：

| 字段         | 描述                                                                                                                                                                               |
| :----------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `agent_id`   | 子代理的唯一标识符。仅在 hook 在子代理调用内部触发时存在                                                                                                                           |
| `agent_type` | 代理名称（如 `"Explore"` 或 `"security-reviewer"`）。当会话使用 `--agent` 或 hook 在子代理内部触发时存在                                                                           |

### 退出码输出

**退出 0** 表示成功。Claude Code 解析 stdout 查找 [JSON 输出字段](#json-输出)。JSON 输出仅在退出 0 时处理。

**退出 2** 表示阻塞错误。Claude Code 忽略 stdout 和其中的任何 JSON。相反，stderr 文本作为错误消息反馈给 Claude。

**任何其他退出码** 对大多数 hook 事件是非阻塞错误。转录显示 `<hook name> hook error` 通知，后跟 stderr 的第一行。

> **警告**：对于大多数 hook 事件，只有退出码 2 阻止操作。Claude Code 将退出码 1 视为非阻塞错误并继续执行操作，即使 1 是传统的 Unix 失败码。如果你的 hook 用于执行策略，使用 `exit 2`。例外是 `WorktreeCreate`，其中任何非零退出码都会中止 worktree 创建。

#### 每个事件的退出码 2 行为

| Hook 事件            | 可以阻止？ | 退出 2 时会发生什么                                                                                                              |
| :------------------- | :--------- | :------------------------------------------------------------------------------------------------------------------------------- |
| `PreToolUse`         | 是         | 阻止工具调用                                                                                                                     |
| `PermissionRequest`  | 是         | 拒绝权限                                                                                                                         |
| `UserPromptSubmit`   | 是         | 阻止提示处理并擦除提示                                                                                                           |
| `Stop`               | 是         | 阻止 Claude 停止，继续对话                                                                                                       |
| `SubagentStop`       | 是         | 阻止子代理停止                                                                                                                   |
| `TeammateIdle`       | 是         | 阻止队友进入空闲（队友继续工作）                                                                                                 |
| `TaskCreated`        | 是         | 回滚任务创建                                                                                                                     |
| `TaskCompleted`      | 是         | 阻止任务被标记为完成                                                                                                             |
| `ConfigChange`       | 是         | 阻止配置更改生效（`policy_settings` 除外）                                                                                        |
| `StopFailure`        | 否         | 输出和退出码被忽略                                                                                                               |
| `PostToolUse`        | 否         | 向 Claude 显示 stderr（工具已运行）                                                                                                |
| `PostToolUseFailure` | 否         | 向 Claude 显示 stderr（工具已失败）                                                                                                |
| `PermissionDenied`   | 否         | 退出码和 stderr 被忽略（拒绝已发生）。使用 JSON `hookSpecificOutput.retry: true` 告诉模型可以重试                                  |
| `Notification`       | 否         | 仅向用户显示 stderr                                                                                                              |
| `SubagentStart`      | 否         | 仅向用户显示 stderr                                                                                                              |
| `SessionStart`       | 否         | 仅向用户显示 stderr                                                                                                              |
| `SessionEnd`         | 否         | 仅向用户显示 stderr                                                                                                              |
| `CwdChanged`         | 否         | 仅向用户显示 stderr                                                                                                              |
| `FileChanged`        | 否         | 仅向用户显示 stderr                                                                                                              |
| `PreCompact`         | 否         | 仅向用户显示 stderr                                                                                                              |
| `PostCompact`        | 否         | 仅向用户显示 stderr                                                                                                              |
| `Elicitation`        | 是         | 拒绝诱导                                                                                                                         |
| `ElicitationResult`  | 是         | 阻止响应（操作变为拒绝）                                                                                                         |
| `WorktreeCreate`     | 是         | 任何非零退出码导致 worktree 创建失败                                                                                             |
| `WorktreeRemove`     | 否         | 仅在调试模式下记录失败                                                                                                           |
| `InstructionsLoaded` | 否         | 退出码被忽略                                                                                                                     |

### HTTP 响应处理

HTTP hooks 使用 HTTP 状态码和响应体而不是退出码和 stdout：

- **2xx 且空 body**：成功，等同于退出码 0 无输出
- **2xx 且纯文本 body**：成功，文本添加为上下文
- **2xx 且 JSON body**：成功，使用与命令 hooks 相同的 [JSON 输出](#json-输出) schema 解析
- **非 2xx 状态**：非阻塞错误，继续执行
- **连接失败或超时**：非阻塞错误，继续执行

与命令 hooks 不同，HTTP hooks 无法仅通过状态码发出阻塞错误信号。要阻止工具调用或拒绝权限，返回带适当决策字段的 2xx 响应和 JSON body。

### JSON 输出

退出码让你允许或阻止，但 JSON 输出给你更细粒度的控制。不是退出码 2 来阻止，而是退出 0 并将 JSON 对象打印到 stdout。

> **注意**：你必须为每个 hook 选择一种方法，而不是两种：要么单独使用退出码进行信号传递，要么退出 0 并打印 JSON 进行结构化控制。Claude Code 仅在退出 0 时处理 JSON。如果你退出 2，任何 JSON 都会被忽略。

JSON 对象支持三类字段：

- **通用字段** 如 `continue` 适用于所有事件
- **顶级 `decision` 和 `reason`** 被某些事件用于阻止或提供反馈
- **`hookSpecificOutput`** 是需要更丰富控制的事件的嵌套对象

| 字段             | 默认    | 描述                                                                                           |
| :--------------- | :------ | :--------------------------------------------------------------------------------------------- |
| `continue`       | `true`  | 如果为 `false`，Claude 在 hook 运行后完全停止处理。优先于任何事件特定的决策字段                 |
| `stopReason`     | 无      | 当 `continue` 为 `false` 时显示给用户的消息。不显示给 Claude                                    |
| `suppressOutput` | `false` | 如果为 `true`，从调试日志中省略 stdout                                                         |
| `systemMessage`  | 无      | 显示给用户的警告消息                                                                           |

#### 决策控制

并非每个事件都支持通过 JSON 阻止或控制行为。确实支持的每个事件使用不同的字段集来表达该决策：

| 事件                                                                                                                        | 决策模式                        | 关键字段                                                                                                                                                         |
| :-------------------------------------------------------------------------------------------------------------------------- | :------------------------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| UserPromptSubmit、PostToolUse、PostToolUseFailure、Stop、SubagentStop、ConfigChange                                         | 顶级 `decision`                 | `decision: "block"`、`reason`                                                                                                                                    |
| TeammateIdle、TaskCreated、TaskCompleted                                                                                    | 退出码或 `continue: false`      | 退出码 2 阻止操作并带 stderr 反馈。JSON `{"continue": false, "stopReason": "..."}` 也完全停止队友，匹配 `Stop` hook 行为                                          |
| PreToolUse                                                                                                                  | `hookSpecificOutput`            | `permissionDecision`（allow/deny/ask/defer）、`permissionDecisionReason`                                                                                          |
| PermissionRequest                                                                                                           | `hookSpecificOutput`            | `decision.behavior`（allow/deny）                                                                                                                                |
| PermissionDenied                                                                                                            | `hookSpecificOutput`            | `retry: true` 告诉模型可以重试被拒绝的工具调用                                                                                                                    |
| WorktreeCreate                                                                                                              | 路径返回                        | 命令 hook 在 stdout 上打印路径；HTTP hook 返回 `hookSpecificOutput.worktreePath`。Hook 失败或缺失路径导致创建失败                                                |
| Elicitation                                                                                                                 | `hookSpecificOutput`            | `action`（accept/decline/cancel）、`content`（accept 的表单字段值）                                                                                               |
| ElicitationResult                                                                                                           | `hookSpecificOutput`            | `action`（accept/decline/cancel）、`content`（表单字段值覆盖）                                                                                                    |
| WorktreeRemove、Notification、SessionEnd、PreCompact、PostCompact、InstructionsLoaded、StopFailure、CwdChanged、FileChanged | 无                              | 无决策控制。用于日志记录或清理等副作用                                                                                                                            |

---

## Hook 事件

每个事件对应 Claude Code 生命周期中的一个点。下面的部分按生命周期排序：从会话设置通过代理循环到会话结束。

### SessionStart

在 Claude Code 启动新会话或恢复现有会话时运行。适用于加载开发上下文或设置环境变量。

匹配器值对应会话的启动方式：

| 匹配器    | 触发时机                         |
| :-------- | :------------------------------- |
| `startup` | 新会话                           |
| `resume`  | `--resume`、`--continue` 或 `/resume` |
| `clear`   | `/clear`                         |
| `compact` | 自动或手动压缩                   |

#### SessionStart 输入

除通用输入字段外，SessionStart hooks 接收 `source`、`model` 和可选的 `agent_type`。

```json
{
  "session_id": "abc123",
  "transcript_path": "/Users/.../.claude/projects/.../transcript.jsonl",
  "cwd": "/Users/...",
  "hook_event_name": "SessionStart",
  "source": "startup",
  "model": "claude-sonnet-4-6"
}
```

#### SessionStart 决策控制

hook 脚本打印到 stdout 的任何文本都会添加为 Claude 的上下文。

| 字段                | 描述                                                    |
| :------------------ | :------------------------------------------------------ |
| `additionalContext` | 添加到 Claude 上下文的字符串。多个 hooks 的值会连接起来 |

#### 持久化环境变量

SessionStart hooks 可以访问 `CLAUDE_ENV_FILE` 环境变量，它提供了一个文件路径，你可以在其中为后续 Bash 命令持久化环境变量。

```bash
#!/bin/bash

if [ -n "$CLAUDE_ENV_FILE" ]; then
  echo 'export NODE_ENV=production' >> "$CLAUDE_ENV_FILE"
  echo 'export DEBUG_LOG=true' >> "$CLAUDE_ENV_FILE"
fi

exit 0
```

要捕获设置命令的所有环境更改，比较之前和之后导出的变量：

```bash
#!/bin/bash

ENV_BEFORE=$(export -p | sort)

# 运行修改环境的设置命令
source ~/.nvm/nvm.sh
nvm use 20

if [ -n "$CLAUDE_ENV_FILE" ]; then
  ENV_AFTER=$(export -p | sort)
  comm -13 <(echo "$ENV_BEFORE") <(echo "$ENV_AFTER") >> "$CLAUDE_ENV_FILE"
fi

exit 0
```

### InstructionsLoaded

当 `CLAUDE.md` 或 `.claude/rules/*.md` 文件加载到上下文时触发。此事件在会话开始时对预加载的文件触发，稍后在文件延迟加载时再次触发。

匹配器针对 `load_reason` 运行：

| 匹配器               | 触发时机                                       |
| :------------------- | :--------------------------------------------- |
| `session_start`      | 会话开始时加载的文件                           |
| `nested_traversal`   | 访问包含嵌套 `CLAUDE.md` 的子目录时           |
| `path_glob_match`    | 带 `paths:` frontmatter 的条件规则匹配时      |
| `include`            | 被另一个指令文件包含时                         |
| `compact`            | 压缩事件后重新加载指令文件时                   |

InstructionsLoaded hooks 没有决策控制。它们不能阻止或修改指令加载。

### UserPromptSubmit

在用户提交提示时运行，在 Claude 处理之前。这允许你根据提示/对话添加额外上下文、验证提示或阻止某些类型的提示。

#### UserPromptSubmit 输入

除通用输入字段外，UserPromptSubmit hooks 接收包含用户提交文本的 `prompt` 字段。

```json
{
  "session_id": "abc123",
  "transcript_path": "/Users/.../.claude/projects/.../transcript.jsonl",
  "cwd": "/Users/...",
  "permission_mode": "default",
  "hook_event_name": "UserPromptSubmit",
  "prompt": "Write a function to calculate the factorial of a number"
}
```

#### UserPromptSubmit 决策控制

`UserPromptSubmit` hooks 可以控制是否处理用户提示并添加上下文。

| 字段                | 描述                                                                                                  |
| :------------------ | :---------------------------------------------------------------------------------------------------- |
| `decision`          | `"block"` 阻止提示被处理并从上下文中擦除。省略以允许提示继续                                           |
| `reason`            | 当 `decision` 为 `"block"` 时显示给用户。不添加到上下文                                                |
| `additionalContext` | 添加到 Claude 上下文的字符串                                                                          |
| `sessionTitle`      | 设置会话标题，与 `/rename` 效果相同。用于根据提示内容自动命名会话                                    |

### PreToolUse

在 Claude 创建工具参数之后、处理工具调用之前运行。匹配工具名称：`Bash`、`Edit`、`Write`、`Read`、`Glob`、`Grep`、`Agent`、`WebFetch`、`WebSearch`、`AskUserQuestion`、`ExitPlanMode` 和任何 [MCP 工具名称](#匹配-mcp-工具)。

#### PreToolUse 输入

除通用输入字段外，PreToolUse hooks 接收 `tool_name`、`tool_input` 和 `tool_use_id`。

**Bash** 字段：

| 字段                | 类型    | 示例               | 描述                     |
| :------------------ | :------ | :----------------- | :----------------------- |
| `command`           | string  | `"npm test"`       | 要执行的 shell 命令      |
| `description`       | string  | `"Run test suite"` | 命令作用的可选描述       |
| `timeout`           | number  | `120000`           | 可选超时（毫秒）         |
| `run_in_background` | boolean | `false`            | 是否在后台运行命令       |

**Write** 字段：

| 字段        | 类型   | 示例                  | 描述               |
| :---------- | :----- | :-------------------- | :----------------- |
| `file_path` | string | `"/path/to/file.txt"` | 要写入的文件绝对路径 |
| `content`   | string | `"file content"`      | 要写入文件的内容   |

**Edit** 字段：

| 字段          | 类型    | 示例                  | 描述               |
| :------------ | :------ | :-------------------- | :----------------- |
| `file_path`   | string  | `"/path/to/file.txt"` | 要编辑的文件绝对路径 |
| `old_string`  | string  | `"original text"`     | 要查找和替换的文本 |
| `new_string`  | string  | `"replacement text"`  | 替换文本           |
| `replace_all` | boolean | `false`               | 是否替换所有出现   |

**Read** 字段：

| 字段        | 类型   | 示例                  | 描述                           |
| :---------- | :----- | :-------------------- | :----------------------------- |
| `file_path` | string | `"/path/to/file.txt"` | 要读取的文件绝对路径           |
| `offset`    | number | `10`                  | 开始读取的可选行号             |
| `limit`     | number | `50`                  | 要读取的可选行数               |

**Glob** 字段：

| 字段      | 类型   | 示例           | 描述                                                 |
| :-------- | :----- | :------------- | :--------------------------------------------------- |
| `pattern` | string | `"**/*.ts"`    | 匹配文件的 glob 模式                                 |
| `path`    | string | `"/path/to/dir"` | 要搜索的可选目录。默认为当前工作目录                 |

**Grep** 字段：

| 字段          | 类型    | 示例           | 描述                                                                                    |
| :------------ | :------ | :------------- | :-------------------------------------------------------------------------------------- |
| `pattern`     | string  | `"TODO.*fix"`  | 要搜索的正则表达式模式                                                                  |
| `path`        | string  | `"/path/to/dir"` | 要搜索的可选文件或目录                                                                  |
| `glob`        | string  | `"*.ts"`       | 过滤文件的可选 glob 模式                                                                |
| `output_mode` | string  | `"content"`    | `"content"`、`"files_with_matches"` 或 `"count"`。默认为 `"files_with_matches"`          |
| `-i`          | boolean | `true`         | 不区分大小写搜索                                                                        |
| `multiline`   | boolean | `false`        | 启用多行匹配                                                                            |

**WebFetch** 字段：

| 字段     | 类型   | 示例                          | 描述                     |
| :------- | :----- | :---------------------------- | :----------------------- |
| `url`    | string | `"https://example.com/api"`   | 获取内容的 URL           |
| `prompt` | string | `"Extract the API endpoints"` | 在获取内容上运行的提示   |

**WebSearch** 字段：

| 字段              | 类型   | 示例                           | 描述                           |
| :---------------- | :----- | :----------------------------- | :----------------------------- |
| `query`           | string | `"react hooks best practices"` | 搜索查询                       |
| `allowed_domains` | array  | `["docs.example.com"]`         | 可选：仅包含来自这些域的结果   |
| `blocked_domains` | array  | `["spam.example.com"]`         | 可选：排除来自这些域的结果     |

**Agent** 字段：

| 字段            | 类型   | 示例                       | 描述                           |
| :-------------- | :----- | :------------------------- | :----------------------------- |
| `prompt`        | string | `"Find all API endpoints"` | 代理要执行的任务               |
| `description`   | string | `"Find API endpoints"`     | 任务的简短描述                 |
| `subagent_type` | string | `"Explore"`                | 要使用的专用代理类型           |
| `model`         | string | `"sonnet"`                 | 覆盖默认的可选模型别名         |

**AskUserQuestion** 字段：

| 字段        | 类型   | 示例                                                                                                         | 描述                                                                                                                                                   |
| :---------- | :----- | :----------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `questions` | array  | `[{"question": "Which framework?", "header": "Framework", "options": [{"label": "React"}], "multiSelect": false}]` | 要呈现的问题，每个问题带 `question` 字符串、短 `header`、`options` 数组和可选的 `multiSelect` 标志                                                     |
| `answers`   | object | `{"Which framework?": "React"}`                                                                              | 可选。将问题文本映射到所选选项标签。多选答案用逗号连接标签。Claude 不设置此字段；通过 `updatedInput` 提供它以编程方式回答                              |

#### PreToolUse 决策控制

`PreToolUse` hooks 可以控制工具调用是否继续。

| 字段                       | 描述                                                                                                                                                                                                                         |
| :------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `permissionDecision`       | `"allow"` 跳过权限提示。`"deny"` 阻止工具调用。`"ask"` 提示用户确认。`"defer"` 优雅退出以便稍后恢复工具。[拒绝和询问规则](/en/permissions#manage-permissions)在 hook 返回 `"allow"` 时仍然适用                             |
| `permissionDecisionReason` | 对于 `"allow"` 和 `"ask"`，显示给用户但不给 Claude。对于 `"deny"`，显示给 Claude。对于 `"defer"`，忽略                                                                                                                       |
| `updatedInput`             | 在执行前修改工具的输入参数。替换整个输入对象，因此包含未更改的字段以及修改的字段。与 `"allow"` 组合以自动批准，或与 `"ask"` 一起向用户显示修改的输入。对于 `"defer"`，忽略                                                   |
| `additionalContext`        | 在工具执行之前添加到 Claude 上下文的字符串。对于 `"defer"`，忽略                                                                                                                                                             |

当多个 PreToolUse hooks 返回不同决策时，优先级为 `deny` > `defer` > `ask` > `allow`。

#### 延迟工具调用供以后使用

`"defer"` 适用于将 `claude -p` 作为子进程运行的集成，如 Agent SDK 应用或基于 Claude Code 构建的自定义 UI。它让调用进程在工具调用处暂停 Claude，通过自己的界面收集输入，然后从离开的地方恢复。Claude Code 仅在带 `-p` 标志的[非交互模式](/en/headless)中尊重此值。

`AskUserQuestion` 工具是典型情况：Claude 想要询问用户一些问题，但没有终端可以回答。往返工作流程：

1. Claude 调用 `AskUserQuestion`。`PreToolUse` hook 触发。
2. hook 返回 `permissionDecision: "defer"`。工具不执行。进程以 `stop_reason: "tool_deferred"` 退出，待处理的工具调用保留在转录中。
3. 调用进程从 SDK 结果中读取 `deferred_tool_use`，在自己的 UI 中显示问题并等待回答。
4. 调用进程运行 `claude -p --resume <session-id>`。相同的工具调用再次触发 `PreToolUse`。
5. hook 返回 `permissionDecision: "allow"` 并在 `updatedInput` 中携带答案。工具执行，Claude 继续。

`"defer"` 仅在 Claude 在轮次中进行单个工具调用时有效。如果 Claude 一次进行多个工具调用，`"defer"` 会被忽略并带警告。

### PermissionRequest

在向用户显示权限对话框时运行。使用 [PermissionRequest 决策控制](#permissionrequest-决策控制)代表用户允许或拒绝。

匹配工具名称，与 PreToolUse 相同的值。

#### PermissionRequest 输入

PermissionRequest hooks 接收像 PreToolUse hooks 一样的 `tool_name` 和 `tool_input` 字段，但没有 `tool_use_id`。可选的 `permission_suggestions` 数组包含用户通常会在权限对话框中看到的"始终允许"选项。

```json
{
  "session_id": "abc123",
  "transcript_path": "/Users/.../.claude/projects/.../transcript.jsonl",
  "cwd": "/Users/...",
  "permission_mode": "default",
  "hook_event_name": "PermissionRequest",
  "tool_name": "Bash",
  "tool_input": {
    "command": "rm -rf node_modules",
    "description": "Remove node_modules directory"
  },
  "permission_suggestions": [
    {
      "type": "addRules",
      "rules": [{ "toolName": "Bash", "ruleContent": "rm -rf node_modules" }],
      "behavior": "allow",
      "destination": "localSettings"
    }
  ]
}
```

#### PermissionRequest 决策控制

| 字段                 | 描述                                                                                                                                              |
| :------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------ |
| `behavior`           | `"allow"` 授予权限，`"deny"` 拒绝它                                                                                                               |
| `updatedInput`       | 仅适用于 `"allow"`：在执行前修改工具的输入参数。替换整个输入对象                                                                                  |
| `updatedPermissions` | 仅适用于 `"allow"`：要应用的[权限更新条目](#权限更新条目)数组，如添加允许规则或更改会话权限模式                                                   |
| `message`            | 仅适用于 `"deny"`：告诉 Claude 权限为什么被拒绝                                                                                                   |
| `interrupt`          | 仅适用于 `"deny"`：如果为 `true`，停止 Claude                                                                                                     |

#### 权限更新条目

`updatedPermissions` 输出字段和 [`permission_suggestions` 输入字段](#permissionrequest-输入)都使用相同的条目对象数组。

| `type`              | 字段                               | 效果                                                                                                                                    |
| :------------------ | :--------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------- |
| `addRules`          | `rules`、`behavior`、`destination` | 添加权限规则。`rules` 是 `{toolName, ruleContent?}` 对象数组。省略 `ruleContent` 以匹配整个工具                                         |
| `replaceRules`      | `rules`、`behavior`、`destination` | 用提供的 `rules` 替换 `destination` 处给定 `behavior` 的所有规则                                                                        |
| `removeRules`       | `rules`、`behavior`、`destination` | 移除给定 `behavior` 的匹配规则                                                                                                          |
| `setMode`           | `mode`、`destination`              | 更改权限模式。有效模式是 `default`、`acceptEdits`、`dontAsk`、`bypassPermissions` 和 `plan`                                             |
| `addDirectories`    | `directories`、`destination`       | 添加工作目录。`directories` 是路径字符串数组                                                                                            |
| `removeDirectories` | `directories`、`destination`       | 移除工作目录                                                                                                                            |

`destination` 字段决定更改是保留在内存中还是持久化到设置文件：

| `destination`     | 写入位置                          |
| :---------------- | :-------------------------------- |
| `session`         | 仅内存，会话结束时丢弃            |
| `localSettings`   | `.claude/settings.local.json`     |
| `projectSettings` | `.claude/settings.json`           |
| `userSettings`    | `~/.claude/settings.json`         |

### PostToolUse

在工具成功完成后立即运行。

匹配工具名称，与 PreToolUse 相同的值。

#### PostToolUse 输入

`PostToolUse` hooks 在工具已成功执行后触发。输入包括 `tool_input`（发送给工具的参数）和 `tool_response`（它返回的结果）。

```json
{
  "session_id": "abc123",
  "transcript_path": "/Users/.../.claude/projects/.../transcript.jsonl",
  "cwd": "/Users/...",
  "permission_mode": "default",
  "hook_event_name": "PostToolUse",
  "tool_name": "Write",
  "tool_input": {
    "file_path": "/path/to/file.txt",
    "content": "file content"
  },
  "tool_response": {
    "filePath": "/path/to/file.txt",
    "success": true
  },
  "tool_use_id": "toolu_01ABC123..."
}
```

#### PostToolUse 决策控制

| 字段                   | 描述                                                                               |
| :--------------------- | :--------------------------------------------------------------------------------- |
| `decision`             | `"block"` 提示 Claude 带 `reason`。省略以允许操作继续                              |
| `reason`               | 当 `decision` 为 `"block"` 时显示给 Claude 的解释                                  |
| `additionalContext`    | Claude 要考虑的额外上下文                                                          |
| `updatedMCPToolOutput` | 仅适用于 [MCP 工具](#匹配-mcp-工具)：用提供的值替换工具的输出                      |

### PostToolUseFailure

在工具执行失败时运行。此事件针对抛出错误或返回失败结果的工具调用触发。

匹配工具名称，与 PreToolUse 相同的值。

#### PostToolUseFailure 输入

| 字段           | 描述                                                     |
| :------------- | :------------------------------------------------------- |
| `error`        | 描述出了什么问题的字符串                                 |
| `is_interrupt` | 可选布尔值，指示失败是否由用户中断引起                   |

#### PostToolUseFailure 决策控制

| 字段                | 描述                                                     |
| :------------------ | :------------------------------------------------------- |
| `additionalContext` | Claude 要与错误一起考虑的额外上下文                      |

### PermissionDenied

当[自动模式](/en/permission-modes#eliminate-prompts-with-auto-mode)分类器拒绝工具调用时运行。此 hook 仅在自动模式中触发。

匹配工具名称，与 PreToolUse 相同的值。

#### PermissionDenied 输入

| 字段     | 描述                                                 |
| :------- | :--------------------------------------------------- |
| `reason` | 分类器解释为什么工具调用被拒绝的原因                 |

#### PermissionDenied 决策控制

PermissionDenied hooks 可以告诉模型可以重试被拒绝的工具调用。返回带 `hookSpecificOutput.retry` 设置为 `true` 的 JSON 对象：

```json
{
  "hookSpecificOutput": {
    "hookEventName": "PermissionDenied",
    "retry": true
  }
}
```

### Notification

在 Claude Code 发送通知时运行。匹配通知类型：`permission_prompt`、`idle_prompt`、`auth_success`、`elicitation_dialog`。

#### Notification 输入

| 字段                | 描述                           |
| :------------------ | :----------------------------- |
| `message`           | 带通知文本的消息               |
| `title`             | 可选标题                       |
| `notification_type` | 指示触发了哪种类型             |

Notification hooks 无法阻止或修改通知。你可以返回 `additionalContext` 向对话添加上下文。

### SubagentStart

在通过 Agent 工具生成 Claude Code 子代理时运行。支持匹配器按代理类型名称过滤。

#### SubagentStart 输入

| 字段         | 描述                                                                                           |
| :----------- | :--------------------------------------------------------------------------------------------- |
| `agent_id`   | 子代理的唯一标识符                                                                             |
| `agent_type` | 代理名称（内置代理如 `"Bash"`、`"Explore"`、`"Plan"` 或来自 `.claude/agents/` 的自定义代理名称） |

SubagentStart hooks 无法阻止子代理创建，但它们可以向子代理注入上下文。

### SubagentStop

在 Claude Code 子代理完成响应时运行。匹配代理类型，与 SubagentStart 相同的值。

#### SubagentStop 输入

| 字段                    | 描述                                                                                                                                         |
| :---------------------- | :------------------------------------------------------------------------------------------------------------------------------------------- |
| `stop_hook_active`      | 当 Claude Code 已经由于 stop hook 而继续时为 `true`                                                                                          |
| `agent_id`              | 子代理的唯一标识符                                                                                                                           |
| `agent_type`            | 用于匹配器过滤的值                                                                                                                           |
| `agent_transcript_path` | 子代理自己的转录，存储在嵌套的 `subagents/` 文件夹中                                                                                         |
| `last_assistant_message` | 子代理最终响应的文本内容，因此 hooks 可以访问它而无需解析转录文件                                    |

SubagentStop hooks 使用与 [Stop hooks](#stop-决策控制) 相同的决策控制格式。

### TaskCreated

在通过 `TaskCreate` 工具创建任务时运行。用于强制执行命名约定、要求任务描述或阻止创建某些任务。

当 `TaskCreated` hook 以退出码 2 退出时，不会创建任务，stderr 消息会作为反馈反馈给模型。

#### TaskCreated 输入

| 字段               | 描述                           |
| :----------------- | :----------------------------- |
| `task_id`          | 正在创建的任务的标识符         |
| `task_subject`     | 任务的标题                     |
| `task_description` | 任务的详细描述。可能不存在     |
| `teammate_name`    | 创建任务的队友的名称。可能不存在 |
| `team_name`        | 团队的名称。可能不存在         |

#### TaskCreated 决策控制

- **退出码 2**：不创建任务，stderr 消息作为反馈反馈给模型。
- **JSON `{"continue": false, "stopReason": "..."}`**：完全停止队友，匹配 `Stop` hook 行为。

### TaskCompleted

在任务被标记为完成时运行。在两种情况下触发：当任何代理通过 TaskUpdate 工具明确将任务标记为完成时，或当[代理团队](/en/agent-teams)队友在其轮次结束时带有进行中的任务时。

#### TaskCompleted 输入

| 字段               | 描述                           |
| :----------------- | :----------------------------- |
| `task_id`          | 正在完成的任务的标识符         |
| `task_subject`     | 任务的标题                     |
| `task_description` | 任务的详细描述。可能不存在     |
| `teammate_name`    | 完成任务的队友的名称。可能不存在 |
| `team_name`        | 团队的名称。可能不存在         |

#### TaskCompleted 决策控制

- **退出码 2**：任务未被标记为完成，stderr 消息作为反馈反馈给模型。
- **JSON `{"continue": false, "stopReason": "..."}`**：完全停止队友。

### Stop

在主 Claude Code 代理完成响应时运行。如果由于用户中断而发生停止则不运行。

#### Stop 输入

| 字段                   | 描述                                                                                           |
| :--------------------- | :--------------------------------------------------------------------------------------------- |
| `stop_hook_active`     | 当 Claude Code 已经由于 stop hook 而继续时为 `true`。检查此值或处理转录以防止 Claude Code 无限运行 |
| `last_assistant_message` | Claude 最终响应的文本内容，因此 hooks 可以访问它而无需解析转录文件                           |

#### Stop 决策控制

| 字段       | 描述                                                                   |
| :--------- | :--------------------------------------------------------------------- |
| `decision` | `"block"` 阻止 Claude 停止。省略以允许 Claude 停止                     |
| `reason`   | 当 `decision` 为 `"block"` 时必需。告诉 Claude 为什么它应该继续        |

### StopFailure

在轮次因 API 错误结束时替代 [Stop](#stop) 运行。输出和退出码被忽略。

#### StopFailure 输入

| 字段                     | 描述                                                                                                                                                                     |
| :----------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `error`                  | 错误类型：`rate_limit`、`authentication_failed`、`billing_error`、`invalid_request`、`server_error`、`max_output_tokens` 或 `unknown`                                     |
| `error_details`          | 关于错误的额外详情（如果可用）                                                                                                                                           |
| `last_assistant_message` | 对话中显示的错误文本。与 `Stop` 和 `SubagentStop` 不同（此字段包含 Claude 的对话输出），对于 `StopFailure` 它包含 API 错误字符串本身，如 `"API Error: Rate limit reached"` |

StopFailure hooks 没有决策控制。它们仅用于通知和日志记录。

### TeammateIdle

当[代理团队](/en/agent-teams)队友在完成其轮次后即将进入空闲时运行。

#### TeammateIdle 输入

| 字段            | 描述                               |
| :-------------- | :--------------------------------- |
| `teammate_name` | 即将进入空闲的队友的名称           |
| `team_name`     | 团队的名称                         |

#### TeammateIdle 决策控制

- **退出码 2**：队友收到 stderr 消息作为反馈并继续工作而不是进入空闲。
- **JSON `{"continue": false, "stopReason": "..."}`**：完全停止队友。

### ConfigChange

在会话期间配置文件更改时运行。用于审计设置更改、执行安全策略或阻止对配置文件的未经授权修改。

匹配器过滤配置源：

| 匹配器             | 触发时机                              |
| :----------------- | :------------------------------------ |
| `user_settings`    | `~/.claude/settings.json` 更改        |
| `project_settings` | `.claude/settings.json` 更改          |
| `local_settings`   | `.claude/settings.local.json` 更改    |
| `policy_settings`  | Managed 策略设置更改                  |
| `skills`           | `.claude/skills/` 中的技能文件更改    |

#### ConfigChange 输入

| 字段         | 描述                                         |
| :----------- | :------------------------------------------- |
| `source`     | 指示哪种配置类型更改                         |
| `file_path`  | 提供被修改的特定文件的路径（可选）           |

#### ConfigChange 决策控制

| 字段       | 描述                                                                              |
| :--------- | :-------------------------------------------------------------------------------- |
| `decision` | `"block"` 阻止配置更改生效。省略以允许更改                                        |
| `reason`   | 当 `decision` 为 `"block"` 时显示给用户的解释                                     |

`policy_settings` 更改无法被阻止。

### CwdChanged

在会话期间工作目录更改时运行，例如当 Claude 执行 `cd` 命令时。

CwdChanged hooks 可以访问 `CLAUDE_ENV_FILE`。写入该文件的变量会持久化到会话的后续 Bash 命令中。

#### CwdChanged 输入

| 字段      | 描述           |
| :-------- | :------------- |
| `old_cwd` | 旧的工作目录   |
| `new_cwd` | 新的工作目录   |

#### CwdChanged 输出

| 字段         | 描述                                                                                                                                                                |
| :----------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `watchPaths` | 绝对路径数组。替换当前动态监视列表（来自你的 `matcher` 配置的路径始终被监视）。返回空数组清除动态列表，这在进入新目录时是典型的                                    |

CwdChanged hooks 没有决策控制。

### FileChanged

在磁盘上监视的文件更改时运行。

`matcher` 对此事件有两个作用：

- **构建监视列表**：值在 `|` 上分割，每个段作为工作目录中的字面文件名注册。
- **过滤哪些 hooks 运行**：当监视的文件更改时，相同的值使用标准[匹配器规则](#匹配器模式)针对更改文件的 basename 过滤哪些 hook 组运行。

#### FileChanged 输入

| 字段        | 描述                                                          |
| :---------- | :------------------------------------------------------------ |
| `file_path` | 更改的文件的绝对路径                                          |
| `event`     | 发生了什么：`"change"`（文件修改）、`"add"`（文件创建）或 `"unlink"`（文件删除） |

#### FileChanged 输出

| 字段         | 描述                                                                                                                                                            |
| :----------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `watchPaths` | 绝对路径数组。替换当前动态监视列表。当你的 hook 脚本基于更改的文件发现要监视的额外文件时使用此字段                                                              |

FileChanged hooks 没有决策控制。

### WorktreeCreate

当你运行 `claude --worktree` 或[子代理使用 `isolation: "worktree"`](/en/sub-agents#choose-the-subagent-scope) 时，Claude Code 使用 `git worktree` 创建隔离的工作副本。如果你配置了 WorktreeCreate hook，它会替换默认的 git 行为。

Hook 必须返回创建的 worktree 目录的绝对路径。

#### WorktreeCreate 输入

| 字段   | 描述                                                                                         |
| :----- | :------------------------------------------------------------------------------------------- |
| `name` | 新 worktree 的 slug 标识符，由用户指定或自动生成（例如 `bold-oak-a3f2`）                     |

#### WorktreeCreate 输出

- **命令 hooks**：在 stdout 上打印路径。
- **HTTP hooks**：在响应体中返回 `{ "hookSpecificOutput": { "hookEventName": "WorktreeCreate", "worktreePath": "/absolute/path" } }`。

### WorktreeRemove

[WorktreeCreate](#worktreecreate) 的对应清理部分。当 worktree 被移除时触发。

#### WorktreeRemove 输入

| 字段              | 描述                             |
| :---------------- | :------------------------------- |
| `worktree_path`   | 正在被移除的 worktree 的绝对路径 |

WorktreeRemove hooks 没有决策控制。

### PreCompact

在 Claude Code 即将运行压缩操作之前运行。

| 匹配器   | 触发时机                       |
| :------- | :----------------------------- |
| `manual` | `/compact`                     |
| `auto`   | 上下文窗口满时的自动压缩       |

#### PreCompact 输入

| 字段                  | 描述                                                                                         |
| :-------------------- | :------------------------------------------------------------------------------------------- |
| `trigger`             | `"manual"` 或 `"auto"`                                                                       |
| `custom_instructions` | 对于 `manual`，包含用户传递给 `/compact` 的内容。对于 `auto`，`custom_instructions` 为空。   |

### PostCompact

在 Claude Code 完成压缩操作之后运行。

#### PostCompact 输入

| 字段               | 描述                                               |
| :----------------- | :------------------------------------------------- |
| `trigger`          | `"manual"` 或 `"auto"`                             |
| `compact_summary`  | 包含压缩操作生成的对话摘要                         |

PostCompact hooks 没有决策控制。

### SessionEnd

在 Claude Code 会话结束时运行。

| Reason                        | 描述                                    |
| :---------------------------- | :-------------------------------------- |
| `clear`                       | 使用 `/clear` 命令清除会话              |
| `resume`                      | 通过交互式 `/resume` 切换会话           |
| `logout`                      | 用户登出                                |
| `prompt_input_exit`           | 用户在提示输入可见时退出                |
| `bypass_permissions_disabled` | 绕过权限模式被禁用                      |
| `other`                       | 其他退出原因                            |

#### SessionEnd 输入

| 字段     | 描述                                 |
| :------- | :----------------------------------- |
| `reason` | 指示会话为什么结束                   |

SessionEnd hooks 没有决策控制。默认超时为 1.5 秒。

### Elicitation

在 MCP 服务器在任务中途请求用户输入时运行。

匹配器字段匹配 MCP 服务器名称。

#### Elicitation 输入

| 字段                | 描述                                                       |
| :------------------ | :--------------------------------------------------------- |
| `mcp_server_name`   | MCP 服务器名称                                             |
| `message`           | 请求消息                                                   |
| `mode`              | `"form"`（最常见的情况）或 `"url"`（基于浏览器的认证）     |
| `requested_schema`  | 表单模式 elicitation 的请求 schema                         |
| `url`               | URL 模式 elicitation 的认证 URL                            |
| `elicitation_id`    | 可选的唯一 elicitation 标识符                              |

#### Elicitation 输出

要以编程方式响应而不显示对话框，返回带 `hookSpecificOutput` 的 JSON 对象：

| 字段      | 值                              | 描述                                         |
| :-------- | :------------------------------ | :------------------------------------------- |
| `action`  | `accept`、`decline`、`cancel`   | 是否接受、拒绝或取消请求                     |
| `content` | object                          | 要提交的表单字段值。仅在 `action` 为 `accept` 时使用 |

退出码 2 拒绝 elicitation 并向用户显示 stderr。

### ElicitationResult

在用户响应 MCP elicitation 之后运行。

#### ElicitationResult 输入

| 字段                | 描述                                         |
| :------------------ | :------------------------------------------- |
| `mcp_server_name`   | MCP 服务器名称                               |
| `action`            | 用户的操作                                   |
| `content`           | 用户提交的表单字段值                         |
| `mode`              | `"form"` 或 `"url"`                          |
| `elicitation_id`    | 可选的唯一 elicitation 标识符                |

#### ElicitationResult 输出

要覆盖用户的响应，返回带 `hookSpecificOutput` 的 JSON 对象。

退出码 2 阻止响应，将有效操作更改为 `decline`。

---

## 基于提示的 hooks

除了命令和 HTTP hooks，Claude Code 还支持使用 LLM 评估是否允许或阻止操作的基于提示的 hooks（`type: "prompt"`），以及生成带工具访问的代理验证器的代理 hooks（`type: "agent"`）。

支持所有四种 hook 类型的事件：

- `PermissionRequest`、`PostToolUse`、`PostToolUseFailure`、`PreToolUse`、`Stop`、`SubagentStop`、`TaskCompleted`、`TaskCreated`、`UserPromptSubmit`

支持 `command` 和 `http` hooks 但不支持 `prompt` 或 `agent` 的事件：

- `ConfigChange`、`CwdChanged`、`Elicitation`、`ElicitationResult`、`FileChanged`、`InstructionsLoaded`、`Notification`、`PermissionDenied`、`PostCompact`、`PreCompact`、`SessionEnd`、`StopFailure`、`SubagentStart`、`TeammateIdle`、`WorktreeCreate`、`WorktreeRemove`

`SessionStart` 仅支持 `command` hooks。

### 基于提示的 hooks 如何工作

1. 将 hook 输入和你的提示发送到 Claude 模型，默认 Haiku
2. LLM 响应带包含决策的结构化 JSON
3. Claude Code 自动处理决策

### 提示 hook 配置

| 字段      | 必需 | 描述                                                                                 |
| :-------- | :--- | :----------------------------------------------------------------------------------- |
| `type`    | 是   | 必须是 `"prompt"`                                                                    |
| `prompt`  | 是   | 要发送给 LLM 的提示文本。使用 `$ARGUMENTS` 作为 hook 输入 JSON 的占位符              |
| `model`   | 否   | 用于评估的模型。默认为快速模型                                                       |
| `timeout` | 否   | 超时秒数。默认：30                                                                   |

### 响应 schema

LLM 必须响应包含以下内容的 JSON：

```json
{
  "ok": true | false,
  "reason": "Explanation for the decision"
}
```

| 字段     | 描述                                             |
| :------- | :----------------------------------------------- |
| `ok`     | `true` 允许操作，`false` 阻止它                  |
| `reason` | 当 `ok` 为 `false` 时必需。向 Claude 显示的解释  |

---

## 基于代理的 hooks

基于代理的 hooks（`type: "agent"`）类似于基于提示的 hooks，但具有多轮工具访问。

### 基于代理的 hooks 如何工作

1. Claude Code 生成带你的提示和 hook 的 JSON 输入的子代理
2. 子代理可以使用 Read、Grep 和 Glob 等工具进行调查
3. 最多 50 轮后，子代理返回结构化的 `{ "ok": true/false }` 决策
4. Claude Code 以与提示 hook 相同的方式处理决策

### 代理 hook 配置

| 字段      | 必需 | 描述                                                                                 |
| :-------- | :--- | :----------------------------------------------------------------------------------- |
| `type`    | 是   | 必须是 `"agent"`                                                                     |
| `prompt`  | 是   | 描述要验证的内容的提示。使用 `$ARGUMENTS` 作为 hook 输入 JSON 的占位符               |
| `model`   | 否   | 使用的模型。默认为快速模型                                                           |
| `timeout` | 否   | 超时秒数。默认：60                                                                   |

---

## 在后台运行 hooks

默认情况下，hooks 阻塞 Claude 的执行直到它们完成。对于长时间运行的任务，设置 `"async": true` 在后台运行 hook 而 Claude 继续工作。

### 配置异步 hook

在命令 hook 的配置中添加 `"async": true` 以在后台运行它而不阻塞 Claude。此字段仅适用于 `type: "command"` hooks。

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write",
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/run-tests.sh",
            "async": true,
            "timeout": 120
          }
        ]
      }
    ]
  }
}
```

### 异步 hooks 如何执行

当异步 hook 触发时，Claude Code 启动 hook 进程并立即继续而不等待它完成。

### 限制

- 仅 `type: "command"` hooks 支持 `async`。基于提示的 hooks 无法异步运行。
- 异步 hooks 无法阻止工具调用或返回决策。
- Hook 输出在下一个对话轮次传递。
- 每次执行创建单独的后台进程。相同异步 hook 的多次触发之间没有去重。

---

## 安全注意事项

### 免责声明

命令 hook 以你的系统用户的完整权限运行。

> **警告**：命令 hook 以你的完整用户权限执行 shell 命令。它们可以修改、删除或访问你的用户账户可以访问的任何文件。在将 hook 命令添加到配置之前，审查并测试所有 hook 命令。

### 安全最佳实践

编写 hooks 时牢记这些实践：

- **验证和清理输入**：永远不要盲目信任输入数据
- **始终引用 shell 变量**：使用 `"$VAR"` 而不是 `$VAR`
- **阻止路径遍历**：检查文件路径中的 `..`
- **使用绝对路径**：为脚本指定完整路径，使用 `"$CLAUDE_PROJECT_DIR"` 表示项目根目录
- **跳过敏感文件**：避免 `.env`、`.git/`、密钥等

---

## Windows PowerShell 工具

在 Windows 上，你可以在命令 hook 上设置 `"shell": "powershell"` 在 PowerShell 中运行单个 hooks。

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write",
        "hooks": [
          {
            "type": "command",
            "shell": "powershell",
            "command": "Write-Host 'File written'"
          }
        ]
      }
    ]
  }
}
```

---

## 调试 hooks

Hook 执行详情，包括哪些 hooks 匹配、它们的退出码和完整的 stdout 和 stderr，会写入调试日志文件。用 `claude --debug-file <path>` 启动 Claude Code 将日志写入已知位置，或运行 `claude --debug` 并在 `~/.claude/debug/<session-id>.txt` 读取日志。

对于更细粒度的 hook 匹配详情，设置 `CLAUDE_CODE_DEBUG_LOG_LEVEL=verbose` 查看额外的日志行，如 hook 匹配器计数和查询匹配。

有关常见问题如 hooks 未触发、无限 Stop hook 循环或配置错误的故障排除，见[指南中的限制和故障排除](/en/hooksguide#limitations-and-troubleshooting)。

---

## 另见

- [用 Hooks 自动化工作流](/en/hooksguide)：带示例的快速入门指南
- [设置](/en/settings)：配置选项
- [权限](/en/permissions)：权限系统和规则语法
- [插件](/en/plugins)：打包和分发 hooks

---

## 提交反馈

如果你发现本文档中有不正确、过时或令人困惑的内容，请通过 POST 提交反馈到：
https://code.claude.com/docs/_mintlify/feedback/claude-code/agent-feedback

请求体（JSON）：`{ "path": "/current-page-path", "feedback": "问题描述" }`

仅在你有具体且可操作的报告时提交反馈。
