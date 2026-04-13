---
sidebar_position: 1
---

# 用 Hooks 自动化工作流

> 当 Claude Code 编辑文件、完成任务或需要输入时自动运行 shell 命令。格式化代码、发送通知、验证命令和执行项目规则。

Hooks 是在 Claude Code 生命周期特定时刻执行的用户定义 shell 命令。它们提供对 Claude Code 行为的确定性控制，确保某些操作始终发生，而不是依赖 LLM 选择运行它们。使用 hooks 执行项目规则、自动化重复任务并将 Claude Code 与你的现有工具集成。

对于需要判断而非确定性规则的决策，你也可以使用[基于提示的 hooks](#基于提示的-hooks) 或[基于代理的 hooks](#基于代理的-hooks)，它们使用 Claude 模型来评估条件。

有关扩展 Claude Code 的其他方式，见[技能](/en/skills)给 Claude 添加额外指令和可执行命令、[子代理](/en/sub-agents)在隔离上下文中运行任务，以及[插件](/en/plugins)打包扩展以跨项目分享。

> **提示**：本指南涵盖常见用例和如何开始。有关完整的事件 schema、JSON 输入/输出格式和高级功能如异步 hooks 和 MCP 工具 hooks，见 [Hooks 参考](/en/hooks)。

---

## 设置第一个 hook

要创建 hook，在[设置文件](#配置-hook-位置)中添加 `hooks` 块。这个演练创建一个桌面通知 hook，这样当 Claude 等待你的输入时你会收到提醒，而不是盯着终端。

**步骤 1：将 hook 添加到你的设置**

打开 `~/.claude/settings.json` 并添加一个 `Notification` hook。下面的示例使用 macOS 的 `osascript`；Linux 和 Windows 命令见[当 Claude 需要输入时获取通知](#当-claude-需要输入时获取通知)。

```json
{
  "hooks": {
    "Notification": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "osascript -e 'display notification \"Claude Code needs your attention\" with title \"Claude Code\"'"
          }
        ]
      }
    ]
  }
}
```

如果你的设置文件已经有 `hooks` 键，将 `Notification` 添加为现有事件键的兄弟而不是替换整个对象。每个事件名称是单个 `hooks` 对象内的键：

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [{ "type": "command", "command": "jq -r '.tool_input.file_path' | xargs npx prettier --write" }]
      }
    ],
    "Notification": [
      {
        "matcher": "",
        "hooks": [{ "type": "command", "command": "osascript -e 'display notification \"Claude Code needs your attention\" with title \"Claude Code\"'" }]
      }
    ]
  }
}
```

你也可以通过在 CLI 中描述你想要什么来让 Claude 为你编写 hook。

**步骤 2：验证配置**

输入 `/hooks` 打开 hooks 浏览器。你会看到所有可用 hook 事件的列表，每个已配置 hook 的事件旁边有计数。选择 `Notification` 确认你的新 hook 出现在列表中。选择 hook 显示其详情：事件、匹配器、类型、源文件和命令。

**步骤 3：测试 hook**

按 `Esc` 返回 CLI。让 Claude 做需要权限的事情，然后从终端切换开。你应该收到桌面通知。

> **提示**：`/hooks` 菜单是只读的。要添加、修改或移除 hooks，直接编辑你的设置 JSON 或让 Claude 进行修改。

---

## 你可以自动化的内容

Hooks 让你在 Claude Code 生命周期的关键时刻运行代码：编辑后格式化文件、在执行前阻止命令、当 Claude 需要输入时发送通知、在会话开始时注入上下文等。有关 hook 事件的完整列表，见 [Hooks 参考](/en/hooks#hook-lifecycle)。

每个示例包含一个可直接使用的配置块，你将其添加到[设置文件](#配置-hook-位置)。最常见的模式：

- [当 Claude 需要输入时获取通知](#当-claude-需要输入时获取通知)
- [编辑后自动格式化代码](#编辑后自动格式化代码)
- [阻止对受保护文件的编辑](#阻止对受保护文件的编辑)
- [压缩后重新注入上下文](#压缩后重新注入上下文)
- [审计配置更改](#审计配置更改)
- [当目录或文件更改时重新加载环境](#当目录或文件更改时重新加载环境)
- [自动批准特定权限提示](#自动批准特定权限提示)

### 当 Claude 需要输入时获取通知

当 Claude 完成工作并需要你的输入时获取桌面通知，这样你可以切换到其他任务而无需检查终端。

这个 hook 使用 `Notification` 事件，当 Claude 等待输入或权限时触发。下面的每个选项卡使用平台的原生通知命令。将其添加到 `~/.claude/settings.json`：

<details>
<summary><strong>macOS</strong></summary>

```json
{
  "hooks": {
    "Notification": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "osascript -e 'display notification \"Claude Code needs your attention\" with title \"Claude Code\"'"
          }
        ]
      }
    ]
  }
}
```

> 如果没有通知出现：`osascript` 通过内置的脚本编辑器应用路由通知。如果脚本编辑器没有通知权限，命令会静默失败，macOS 不会提示你授予权限。在终端中运行一次让脚本编辑器出现在你的通知设置中：
>
> ```bash
> osascript -e 'display notification "test"'
> ```
>
> 暂时不会出现任何内容。打开**系统设置 > 通知**，在列表中找到**脚本编辑器**，然后打开**允许通知**。再次运行命令确认测试通知出现。
</details>

<details>
<summary><strong>Linux</strong></summary>

```json
{
  "hooks": {
    "Notification": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "notify-send 'Claude Code' 'Claude Code needs your attention'"
          }
        ]
      }
    ]
  }
}
```
</details>

<details>
<summary><strong>Windows (PowerShell)</strong></summary>

```json
{
  "hooks": {
    "Notification": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "powershell.exe -Command \"[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms'); [System.Windows.Forms.MessageBox]::Show('Claude Code needs your attention', 'Claude Code')\""
          }
        ]
      }
    ]
  }
}
```
</details>

### 编辑后自动格式化代码

自动在 Claude 编辑的每个文件上运行 [Prettier](https://prettier.io/)，这样格式保持一致而无需手动干预。

这个 hook 使用带 `Edit|Write` 匹配器的 `PostToolUse` 事件，因此它仅在文件编辑工具之后运行。命令用 [`jq`](https://jqlang.github.io/jq/) 提取编辑的文件路径并将其传递给 Prettier。将其添加到项目根目录中的 `.claude/settings.json`：

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "jq -r '.tool_input.file_path' | xargs npx prettier --write"
          }
        ]
      }
    ]
  }
}
```

> **注意**：本页的 Bash 示例使用 `jq` 进行 JSON 解析。用 `brew install jq`（macOS）、`apt-get install jq`（Debian/Ubuntu）安装，或见[`jq` 下载](https://jqlang.github.io/jq/download/)。

### 阻止对受保护文件的编辑

防止 Claude 修改 `.env`、`package-lock.json` 或 `.git/` 中的任何内容等敏感文件。Claude 会收到解释为什么编辑被阻止的反馈，因此它可以调整方法。

这个示例使用 hook 调用的单独脚本文件。脚本检查目标文件路径是否与受保护模式列表匹配并以退出码 2 退出以阻止编辑。

**步骤 1：创建 hook 脚本**

保存到 `.claude/hooks/protect-files.sh`：

```bash
#!/bin/bash
# protect-files.sh

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

PROTECTED_PATTERNS=(".env" "package-lock.json" ".git/")

for pattern in "${PROTECTED_PATTERNS[@]}"; do
  if [[ "$FILE_PATH" == *"$pattern"* ]]; then
    echo "Blocked: $FILE_PATH matches protected pattern '$pattern'" >&2
    exit 2
  fi
done

exit 0
```

**步骤 2：使脚本可执行（macOS/Linux）**

Hook 脚本必须是可执行的，Claude Code 才能运行它们：

```bash
chmod +x .claude/hooks/protect-files.sh
```

**步骤 3：注册 hook**

在 `.claude/settings.json` 中添加 `PreToolUse` hook，在任何 `Edit` 或 `Write` 工具调用之前运行脚本：

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/protect-files.sh"
          }
        ]
      }
    ]
  }
}
```

### 压缩后重新注入上下文

当 Claude 的上下文窗口填满时，压缩会总结对话以释放空间。这可能会丢失重要细节。使用带 `compact` 匹配器的 `SessionStart` hook 在每次压缩后重新注入关键上下文。

你的命令写入 stdout 的任何文本都会添加到 Claude 的上下文中。这个示例提醒 Claude 项目约定和最近的工作。将其添加到项目根目录中的 `.claude/settings.json`：

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "compact",
        "hooks": [
          {
            "type": "command",
            "command": "echo 'Reminder: use Bun, not npm. Run bun test before committing. Current sprint: auth refactor.'"
          }
        ]
      }
    ]
  }
}
```

你可以用任何生成动态输出的命令替换 `echo`，如 `git log --oneline -5` 显示最近的提交。对于每次会话启动时注入上下文，考虑改用 [CLAUDE.md](/en/memory)。对于环境变量，见参考中的 [`CLAUDE_ENV_FILE`](/en/hooks#persist-environment-variables)。

### 审计配置更改

跟踪会话期间设置或技能文件的更改。当外部进程或编辑器修改配置文件时，`ConfigChange` 事件会触发，因此你可以记录更改以符合合规性或阻止未经授权的修改。

这个示例将每次更改追加到审计日志。将其添加到 `~/.claude/settings.json`：

```json
{
  "hooks": {
    "ConfigChange": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "jq -c '{timestamp: now | todate, source: .source, file: .file_path}' >> ~/claude-config-audit.log"
          }
        ]
      }
    ]
  }
}
```

匹配器按配置类型过滤：`user_settings`、`project_settings`、`local_settings`、`policy_settings` 或 `skills`。要阻止更改生效，以退出码 2 退出或返回 `{"decision": "block"}`。有关完整输入 schema，见 [ConfigChange 参考](/en/hooks#configchange)。

### 当目录或文件更改时重新加载环境

某些项目根据你所在的目录设置不同的环境变量。像 [direnv](https://direnv.net/) 这样的工具在你的 shell 中自动执行此操作，但 Claude 的 Bash 工具不会自行获取这些更改。

`CwdChanged` hook 可以解决这个问题：它每次 Claude 更改目录时运行，因此你可以为新位置重新加载正确的变量。hook 将更新的值写入 `CLAUDE_ENV_FILE`，Claude Code 在每个 Bash 命令之前应用它。将其添加到 `~/.claude/settings.json`：

```json
{
  "hooks": {
    "CwdChanged": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "direnv export bash >> \"$CLAUDE_ENV_FILE\""
          }
        ]
      }
    ]
  }
}
```

要响应特定文件而不是每次目录更改，使用带 `matcher` 的 `FileChanged`，列出要监视的文件名，用 `|` 分隔。要构建监视列表，此值被拆分为字面文件名而不是作为正则表达式评估。这个示例监视工作目录中的 `.envrc` 和 `.env`：

```json
{
  "hooks": {
    "FileChanged": [
      {
        "matcher": ".envrc|.env",
        "hooks": [
          {
            "type": "command",
            "command": "direnv export bash >> \"$CLAUDE_ENV_FILE\""
          }
        ]
      }
    ]
  }
}
```

见 [CwdChanged](/en/hooks#cwdchanged) 和 [FileChanged](/en/hooks#filechanged) 参考条目获取输入 schema、`watchPaths` 输出和 `CLAUDE_ENV_FILE` 详情。

### 自动批准特定权限提示

跳过你始终允许的工具调用的批准对话框。这个示例自动批准 `ExitPlanMode`，这是 Claude 在完成呈现计划并询问是否继续时调用的工具，这样你每次计划准备好时不会被提示。

与上面的退出码示例不同，自动批准需要你的 hook 将 JSON 决策写入 stdout。当 Claude Code 即将显示权限对话框时触发 `PermissionRequest` hook，返回 `"behavior": "allow"` 代表你回答。

匹配器将 hook 限定为仅 `ExitPlanMode`，因此不会影响其他提示。将其添加到 `~/.claude/settings.json`：

```json
{
  "hooks": {
    "PermissionRequest": [
      {
        "matcher": "ExitPlanMode",
        "hooks": [
          {
            "type": "command",
            "command": "echo '{\"hookSpecificOutput\": {\"hookEventName\": \"PermissionRequest\", \"decision\": {\"behavior\": \"allow\"}}}'"
          }
        ]
      }
    ]
  }
}
```

当 hook 批准时，Claude Code 退出计划模式并恢复进入计划模式之前的任何权限模式。转录显示"Allowed by PermissionRequest hook"在对话框会出现的位置。hook 路径始终保持当前对话：它不能像对话框那样清除上下文并启动新的实现会话。

要设置特定权限模式，你的 hook 输出可以包含带 `setMode` 条目的 `updatedPermissions` 数组。`mode` 值是任何权限模式如 `default`、`acceptEdits` 或 `bypassPermissions`，`destination: "session"` 仅将其应用于当前会话。

要将会话切换到 `acceptEdits`，你的 hook 将此 JSON 写入 stdout：

```json
{
  "hookSpecificOutput": {
    "hookEventName": "PermissionRequest",
    "decision": {
      "behavior": "allow",
      "updatedPermissions": [
        { "type": "setMode", "mode": "acceptEdits", "destination": "session" }
      ]
    }
  }
}
```

保持匹配器尽可能狭窄。在 `.*` 上匹配或留空匹配器会自动批准每个权限提示，包括文件写入和 shell 命令。有关决策字段的完整集合，见 [PermissionRequest 参考](/en/hooks#permissionrequest-decision-control)。

---

## Hooks 如何工作

Hook 事件在 Claude Code 的特定生命周期点触发。当事件触发时，所有匹配的 hooks 并行运行，相同的 hook 命令会自动去重。下表显示每个事件及其触发时机：

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

当多个 hooks 匹配时，每个返回自己的结果。对于决策，Claude Code 选择最限制的答案。返回 `deny` 的 `PreToolUse` hook 会取消工具调用，无论其他返回什么。一个返回 `ask` 的 hook 会强制权限提示，即使其余都返回 `allow`。来自每个 hook 的 `additionalContext` 文本都会被保留并一起传递给 Claude。

每个 hook 有一个 `type` 决定它如何运行。大多数 hooks 使用 `"type": "command"`，运行 shell 命令。还有三种其他类型可用：

- `"type": "http"`：POST 事件数据到 URL。见[HTTP hooks](#http-hooks)。
- `"type": "prompt"`：单轮 LLM 评估。见[基于提示的 hooks](#基于提示的-hooks)。
- `"type": "agent"`：带工具访问的多轮验证。见[基于代理的 hooks](#基于代理的-hooks)。

### 读取输入和返回输出

Hooks 通过 stdin、stdout、stderr 和退出码与 Claude Code 通信。当事件触发时，Claude Code 将事件特定数据作为 JSON 传递给你脚本的 stdin。你的脚本读取该数据，执行工作，并通过退出码告诉 Claude Code 接下来做什么。

#### Hook 输入

每个事件包含 `session_id` 和 `cwd` 等通用字段，但每种事件类型添加不同的数据。例如，当 Claude 运行 Bash 命令时，`PreToolUse` hook 在 stdin 上接收类似这样的内容：

```json
{
  "session_id": "abc123",
  "cwd": "/Users/sarah/myproject",
  "hook_event_name": "PreToolUse",
  "tool_name": "Bash",
  "tool_input": {
    "command": "npm test"
  }
}
```

你的脚本可以解析该 JSON 并对任何这些字段执行操作。`UserPromptSubmit` hooks 获取 `prompt` 文本，`SessionStart` hooks 获取 `source`（startup、resume、clear、compact）等。见参考中的[通用输入字段](/en/hooks#common-input-fields)获取共享字段，每个事件的部分获取事件特定 schema。

#### Hook 输出

你的脚本通过写入 stdout 或 stderr 并以特定退出码退出告诉 Claude Code 接下来做什么。例如，想要阻止命令的 `PreToolUse` hook：

```bash
#!/bin/bash
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command')

if echo "$COMMAND" | grep -q "drop table"; then
  echo "Blocked: dropping tables is not allowed" >&2
  exit 2
fi

exit 0
```

退出码决定接下来发生什么：

- **退出 0**：操作继续。对于 `UserPromptSubmit` 和 `SessionStart` hooks，你写入 stdout 的任何内容都会添加到 Claude 的上下文中。
- **退出 2**：操作被阻止。将原因写入 stderr，Claude 接收它作为反馈以便调整。
- **任何其他退出码**：操作继续。转录显示 `<hook name> hook error` 通知，后跟 stderr 的第一行；完整的 stderr 进入[调试日志](/en/hooks#debug-hooks)。

#### 结构化 JSON 输出

退出码给你两个选项：允许或阻止。要获得更多控制，退出 0 并改为将 JSON 对象打印到 stdout。

> **注意**：使用退出 2 以 stderr 消息阻止，或退出 0 以 JSON 进行结构化控制。不要混用它们：当你退出 2 时 Claude Code 会忽略 JSON。

例如，`PreToolUse` hook 可以拒绝工具调用并告诉 Claude 为什么，或升级给用户批准：

```json
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "deny",
    "permissionDecisionReason": "Use rg instead of grep for better performance"
  }
}
```

使用 `"deny"` 时，Claude Code 取消工具调用并将 `permissionDecisionReason` 反馈给 Claude。这些 `permissionDecision` 值特定于 `PreToolUse`：

- `"allow"`：跳过交互式权限提示。拒绝和询问规则（包括企业 managed 拒绝列表）仍然适用
- `"deny"`：取消工具调用并将原因发送给 Claude
- `"ask"`：正常向用户显示权限提示

第四个值 `"defer"` 在带 `-p` 标志的[非交互模式](/en/headless)中可用。它退出进程并保留工具调用，以便 Agent SDK 包装器可以收集输入并恢复。见参考中的[延迟工具调用供以后使用](/en/hooks#defer-a-tool-call-for-later)。

返回 `"allow"` 会跳过交互式提示但不会覆盖[权限规则](/en/permissions#manage-permissions)。如果拒绝规则匹配工具调用，即使你的 hook 返回 `"allow"`，调用也会被阻止。如果询问规则匹配，用户仍然会被提示。这意味着来自任何设置作用域的拒绝规则，包括 [managed 设置](/en/settings#settings-files)，始终优先于 hook 批准。

其他事件使用不同的决策模式。例如，`PostToolUse` 和 `Stop` hooks 使用顶级 `decision: "block"` 字段，而 `PermissionRequest` 使用 `hookSpecificOutput.decision.behavior`。见参考中的[摘要表](/en/hooks#decision-control)获取按事件的完整分解。

对于 `UserPromptSubmit` hooks，改用 `additionalContext` 将文本注入 Claude 的上下文。基于提示的 hooks（`type: "prompt"`）处理输出的方式不同：见[基于提示的 hooks](#基于提示的-hooks)。

### 用匹配器过滤 hooks

没有匹配器时，hook 在其事件的每次出现时触发。匹配器让你缩小范围。例如，如果你只想在文件编辑后运行格式化程序（而不是每次工具调用后），在你的 `PostToolUse` hook 中添加匹配器：

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          { "type": "command", "command": "prettier --write ..." }
        ]
      }
    ]
  }
}
```

`"Edit|Write"` 匹配器仅在 Claude 使用 `Edit` 或 `Write` 工具时触发，而不是在使用 `Bash`、`Read` 或任何其他工具时。有关纯名称和正则表达式如何评估，见[匹配器模式](/en/hooks#matcher-patterns)。

每种事件类型匹配特定字段：

| 事件                                                                                                                       | 匹配器过滤的内容                                                      | 示例匹配器值                                                                                                            |
| :------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------- |
| `PreToolUse`、`PostToolUse`、`PostToolUseFailure`、`PermissionRequest`、`PermissionDenied`                                 | 工具名称                                                              | `Bash`、`Edit\|Write`、`mcp__.*`                                                                                        |
| `SessionStart`                                                                                                             | 会话如何启动                                                          | `startup`、`resume`、`clear`、`compact`                                                                                 |
| `SessionEnd`                                                                                                               | 会话为什么结束                                                        | `clear`、`resume`、`logout`、`prompt_input_exit`、`bypass_permissions_disabled`、`other`                                |
| `Notification`                                                                                                             | 通知类型                                                              | `permission_prompt`、`idle_prompt`、`auth_success`、`elicitation_dialog`                                                |
| `SubagentStart`                                                                                                            | 代理类型                                                              | `Bash`、`Explore`、`Plan` 或自定义代理名称                                                                              |
| `PreCompact`、`PostCompact`                                                                                                | 什么触发压缩                                                          | `manual`、`auto`                                                                                                        |
| `SubagentStop`                                                                                                             | 代理类型                                                              | 与 `SubagentStart` 相同的值                                                                                             |
| `ConfigChange`                                                                                                             | 配置源                                                                | `user_settings`、`project_settings`、`local_settings`、`policy_settings`、`skills`                                      |
| `StopFailure`                                                                                                              | 错误类型                                                              | `rate_limit`、`authentication_failed`、`billing_error`、`invalid_request`、`server_error`、`max_output_tokens`、`unknown` |
| `InstructionsLoaded`                                                                                                       | 加载原因                                                              | `session_start`、`nested_traversal`、`path_glob_match`、`include`、`compact`                                            |
| `Elicitation`                                                                                                              | MCP 服务器名称                                                        | 你配置的 MCP 服务器名称                                                                                                 |
| `ElicitationResult`                                                                                                        | MCP 服务器名称                                                        | 与 `Elicitation` 相同的值                                                                                               |
| `FileChanged`                                                                                                              | 要监视的字面文件名（见 [FileChanged](/en/hooks#filechanged)）          | `.envrc\|.env`                                                                                                          |
| `UserPromptSubmit`、`Stop`、`TeammateIdle`、`TaskCreated`、`TaskCompleted`、`WorktreeCreate`、`WorktreeRemove`、`CwdChanged` | 不支持匹配器                                                          | 每次出现时始终触发                                                                                                      |

在不同事件类型上使用匹配器的几个更多示例：

<details>
<summary><strong>记录每个 Bash 命令</strong></summary>

仅匹配 `Bash` 工具调用并将每个命令记录到文件。`PostToolUse` 事件在命令完成后触发，因此 `tool_input.command` 包含运行的内容。hook 通过 stdin 接收事件数据作为 JSON，`jq -r '.tool_input.command'` 仅提取命令字符串，`>>` 将其追加到日志文件：

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "jq -r '.tool_input.command' >> ~/.claude/command-log.txt"
          }
        ]
      }
    ]
  }
}
```
</details>

<details>
<summary><strong>匹配 MCP 工具</strong></summary>

MCP 工具使用与内置工具不同的命名约定：`mcp__<server>__<tool>`，其中 `<server>` 是 MCP 服务器名称，`<tool>` 是它提供的工具。例如 `mcp__github__search_repositories` 或 `mcp__filesystem__read_file`。使用正则表达式匹配器定位来自特定服务器的所有工具，或用 `mcp__.*__write.*` 等模式跨服务器匹配。见参考中的[匹配 MCP 工具](/en/hooks#match-mcp-tools)获取完整示例列表。

下面的命令从 hook 的 JSON 输入中用 `jq` 提取工具名称并将其写入 stderr。写入 stderr 保持 stdout 干净用于 JSON 输出并将消息发送到[调试日志](/en/hooks#debug-hooks)：

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "mcp__github__.*",
        "hooks": [
          {
            "type": "command",
            "command": "echo \"GitHub tool called: $(jq -r '.tool_name')\" >&2"
          }
        ]
      }
    ]
  }
}
```
</details>

<details>
<summary><strong>会话结束时清理</strong></summary>

`SessionEnd` 事件支持匹配器在会话结束的原因上。这个 hook 仅在 `clear` 时触发（当你运行 `/clear` 时），而不是正常退出时：

```json
{
  "hooks": {
    "SessionEnd": [
      {
        "matcher": "clear",
        "hooks": [
          {
            "type": "command",
            "command": "rm -f /tmp/claude-scratch-*.txt"
          }
        ]
      }
    ]
  }
}
```
</details>

有关完整匹配器语法，见 [Hooks 参考](/en/hooks#configuration)。

#### 用 `if` 字段按工具名称和参数过滤

> **注意**：`if` 字段需要 Claude Code v2.1.85 或更高版本。早期版本忽略它并在每次匹配调用时运行 hook。

`if` 字段使用[权限规则语法](/en/permissions)按工具名称和参数一起过滤 hooks，因此 hook 进程仅在工具调用匹配时生成。这超越了 `matcher`，后者仅在组级别按工具名称过滤。

例如，仅在 Claude 使用 `git` 命令而不是所有 Bash 命令时运行 hook：

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "if": "Bash(git *)",
            "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/check-git-policy.sh"
          }
        ]
      }
    ]
  }
}
```

hook 进程仅在 Bash 命令以 `git` 开头时生成。其他 Bash 命令完全跳过此处理器。`if` 字段接受与权限规则相同的模式：`"Bash(git *)"`、`"Edit(*.ts)"` 等。要匹配多个工具名称，使用各自带自己 `if` 值的单独处理器，或在支持管道交替的 `matcher` 级别匹配。

`if` 仅适用于工具事件：`PreToolUse`、`PostToolUse`、`PostToolUseFailure`、`PermissionRequest` 和 `PermissionDenied`。将其添加到任何其他事件会阻止 hook 运行。

### 配置 hook 位置

你在哪里添加 hook 决定了它的作用域：

| 位置                                                    | 作用域                | 可共享                           |
| :------------------------------------------------------ | :-------------------- | :------------------------------- |
| `~/.claude/settings.json`                               | 你的所有项目          | 否，限于你的机器                 |
| `.claude/settings.json`                                 | 单个项目              | 是，可以提交到仓库               |
| `.claude/settings.local.json`                           | 单个项目              | 否，被 gitignore                 |
| Managed 策略设置                                        | 组织范围              | 是，管理员控制                   |
| [插件](/en/plugins) `hooks/hooks.json`                  | 插件启用时            | 是，与插件捆绑                   |
| [技能](/en/skills) 或 [代理](/en/sub-agents) frontmatter | 技能或代理激活时      | 是，定义在组件文件中             |

在 Claude Code 中运行 [`/hooks`](/en/hooks#the-hooks-menu) 浏览按事件分组的所有已配置 hooks。要一次性禁用所有 hooks，在设置文件中设置 `"disableAllHooks": true`。

如果你在 Claude Code 运行时直接编辑设置文件，文件监视器通常会自动获取 hook 更改。

---

## 基于提示的 hooks

对于需要判断而非确定性规则的决策，使用 `type: "prompt"` hooks。Claude Code 不会运行 shell 命令，而是将你的提示和 hook 的输入数据发送给 Claude 模型（默认 Haiku）来做决策。如果你需要更多功能，可以用 `model` 字段指定不同的模型。

模型的唯一工作是返回 yes/no 决策作为 JSON：

- `"ok": true`：操作继续
- `"ok": false`：操作被阻止。模型的 `"reason"` 被反馈给 Claude 以便调整。

这个示例使用 `Stop` hook 询问模型是否所有请求的任务都完成。如果模型返回 `"ok": false`，Claude 继续工作并使用 `reason` 作为其下一个指令：

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "prompt",
            "prompt": "Check if all tasks are complete. If not, respond with {\"ok\": false, \"reason\": \"what remains to be done\"}."
          }
        ]
      }
    ]
  }
}
```

有关完整配置选项，见参考中的[基于提示的 hooks](/en/hooks#prompt-based-hooks)。

---

## 基于代理的 hooks

当验证需要检查文件或运行命令时，使用 `type: "agent"` hooks。与进行单次 LLM 调用的提示 hooks 不同，代理 hooks 生成一个可以读取文件、搜索代码并使用其他工具在返回决策之前验证条件的子代理。

代理 hooks 使用与提示 hooks 相同的 `"ok"` / `"reason"` 响应格式，但默认超时更长为 60 秒，最多 50 次工具使用轮次。

这个示例在允许 Claude 停止之前验证测试是否通过：

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "agent",
            "prompt": "Verify that all unit tests pass. Run the test suite and check the results. $ARGUMENTS",
            "timeout": 120
          }
        ]
      }
    ]
  }
}
```

当 hook 输入数据足以做决策时使用提示 hooks。当你需要针对代码库的实际状态验证某些内容时使用代理 hooks。

有关完整配置选项，见参考中的[基于代理的 hooks](/en/hooks#agent-based-hooks)。

---

## HTTP hooks

使用 `type: "http"` hooks 将事件数据 POST 到 HTTP 端点而不是运行 shell 命令。端点接收与命令 hook 在 stdin 上接收的相同 JSON，并通过 HTTP 响应体使用相同的 JSON 格式返回结果。

HTTP hooks 在你想要 web 服务器、云函数或外部服务处理 hook 逻辑时有用：例如，共享审计服务记录团队中所有工具使用事件。

这个示例将每次工具使用 POST 到本地日志服务：

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "hooks": [
          {
            "type": "http",
            "url": "http://localhost:8080/hooks/tool-use",
            "headers": {
              "Authorization": "Bearer $MY_TOKEN"
            },
            "allowedEnvVars": ["MY_TOKEN"]
          }
        ]
      }
    ]
  }
}
```

端点应使用与命令 hooks 相同的[输出格式](/en/hooks#json-output)返回 JSON 响应体。要阻止工具调用，返回带适当 `hookSpecificOutput` 字段的 2xx 响应。仅 HTTP 状态码无法阻止操作。

头部值支持使用 `$VAR_NAME` 或 `${VAR_NAME}` 语法的环境变量插值。仅 `allowedEnvVars` 数组中列出的变量会被解析；所有其他 `$VAR` 引用保持为空。

有关完整配置选项和响应处理，见参考中的[HTTP hooks](/en/hooks#http-hook-fields)。

---

## 限制和故障排除

### 限制

- 命令 hooks 仅通过 stdout、stderr 和退出码通信。它们无法触发 `/` 命令或工具调用。通过 `additionalContext` 返回的文本作为 Claude 作为纯文本读取的系统提醒注入。HTTP hooks 通过响应体通信。
- Hook 超时默认为 10 分钟，可用 hook 的 `timeout` 字段（秒）配置。
- `PostToolUse` hooks 无法撤销操作，因为工具已经执行。
- `PermissionRequest` hooks 在[非交互模式](/en/headless)（`-p`）中不触发。改用 `PreToolUse` hooks 进行自动化权限决策。
- `Stop` hooks 在 Claude 每次完成响应时触发，不仅在任务完成时。它们不在用户中断时触发。API 错误改为触发 [StopFailure](/en/hooks#stopfailure)。
- 当多个 PreToolUse hooks 返回 [`updatedInput`](/en/hooks#pretooluse) 重写工具参数时，最后完成的获胜。由于 hooks 并行运行，顺序是不确定的。避免让多个 hook 修改同一工具的输入。

### Hooks 和权限模式

PreToolUse hooks 在任何权限模式检查之前触发。返回 `permissionDecision: "deny"` 的 hook 即使在 `bypassPermissions` 模式或带 `--dangerously-skip-permissions` 时也会阻止工具。这让你可以执行用户无法通过更改权限模式绕过的策略。

反之不成立：返回 `"allow"` 的 hook 不会绕过设置中的拒绝规则。hooks 可以收紧限制但不能放松超过权限规则允许的限制。

### Hook 未触发

hook 已配置但从不执行。

- 运行 `/hooks` 并确认 hook 出现在正确事件下
- 检查匹配器模式是否完全匹配工具名称（匹配器区分大小写）
- 验证你触发的是正确的事件类型（例如 `PreToolUse` 在工具执行之前触发，`PostToolUse` 在之后触发）
- 如果在非交互模式（`-p`）中使用 `PermissionRequest` hooks，改用 `PreToolUse`

### 输出中的 hook 错误

你在转录中看到类似"PreToolUse hook error: ..."的消息。

- 你的脚本意外地以非零码退出。通过管道传输示例 JSON 手动测试它：
  ```bash
  echo '{"tool_name":"Bash","tool_input":{"command":"ls"}}' | ./my-hook.sh
  echo $?  # 检查退出码
  ```
- 如果你看到"command not found"，使用绝对路径或 `$CLAUDE_PROJECT_DIR` 引用脚本
- 如果你看到"jq: command not found"，安装 `jq` 或使用 Python/Node.js 进行 JSON 解析
- 如果脚本根本没有运行，使其可执行：`chmod +x ./my-hook.sh`

### `/hooks` 显示没有配置 hooks

你编辑了设置文件但 hooks 没有出现在菜单中。

- 文件编辑通常会自动获取。如果几秒后还没出现，文件监视器可能错过了更改：重启会话强制重新加载。
- 验证你的 JSON 是有效的（不允许尾随逗号和注释）
- 确认设置文件在正确的位置：`.claude/settings.json` 用于项目 hooks，`~/.claude/settings.json` 用于全局 hooks

### Stop hook 永远运行

Claude 继续工作陷入无限循环而不是停止。

你的 Stop hook 脚本需要检查它是否已经触发了继续。从 JSON 输入中解析 `stop_hook_active` 字段并在它为 `true` 时提前退出：

```bash
#!/bin/bash
INPUT=$(cat)
if [ "$(echo "$INPUT" | jq -r '.stop_hook_active')" = "true" ]; then
  exit 0  # Allow Claude to stop
fi
# ... rest of your hook logic
```

### JSON 验证失败

即使你的 hook 脚本输出有效的 JSON，Claude Code 仍显示 JSON 解析错误。

当 Claude Code 运行 hook 时，它会生成一个 shell 来获取你的 profile（`~/.zshrc` 或 `~/.bashrc`）。如果你的 profile 包含无条件的 `echo` 语句，该输出会被添加到你的 hook 的 JSON 前面：

```text
Shell ready on arm64
{"decision": "block", "reason": "Not allowed"}
```

Claude Code 尝试将此解析为 JSON 并失败。要修复此问题，在 shell profile 中包装 echo 语句，使它们仅在交互式 shell 中运行：

```bash
# In ~/.zshrc or ~/.bashrc
if [[ $- == *i* ]]; then
  echo "Shell ready"
fi
```

`$-` 变量包含 shell 标志，`i` 表示交互式。hooks 在非交互式 shell 中运行，因此 echo 被跳过。

### 调试技术

用 `Ctrl+O` 切换的转录视图显示每个触发的 hook 的单行摘要：成功是静默的，阻止错误显示 stderr，非阻止错误显示 `<hook name> hook error` 通知后跟 stderr 的第一行。

有关完整执行详情（包括哪些 hooks 匹配、它们的退出码、stdout 和 stderr），读取调试日志。使用 `claude --debug-file /tmp/claude.log` 启动 Claude Code 写入已知路径，然后在另一个终端中 `tail -f /tmp/claude.log`。如果你在没有该标志的情况下启动，在会话中运行 `/debug` 启用日志并找到日志路径。

---

## 了解更多

- [Hooks 参考](/en/hooks)：完整事件 schema、JSON 输出格式、异步 hooks 和 MCP 工具 hooks
- [安全注意事项](/en/hooks#security-considerations)：在共享或生产环境中部署 hooks 前查看
- [Bash 命令验证器示例](https://github.com/anthropics/claude-code/blob/main/examples/hooks/bash_command_validator_example.py)：完整参考实现

---

## 提交反馈

如果你发现本文档中有不正确、过时或令人困惑的内容，请通过 POST 提交反馈到：
https://code.claude.com/docs/_mintlify/feedback/claude-code/agent-feedback

请求体（JSON）：`{ "path": "/current-page-path", "feedback": "问题描述" }`

仅在你有具体且可操作的报告时提交反馈。
