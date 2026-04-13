# Claude Code 配置指南

> 通过全局和项目级设置以及环境变量来配置 Claude Code。
>
> 完整文档索引请访问：https://code.claude.com/docs/llms.txt

Claude Code 提供了丰富的设置项，你可以根据需求通过交互式 REPL 中的 `/config` 命令打开标签页式设置界面来查看和修改配置。

---

## 配置作用域（Scopes）

Claude Code 使用**作用域系统**来决定配置在哪里生效以及谁可以共享它们。

### 可用作用域

| 作用域     | 存储位置                                                            | 影响范围                | 是否与团队共享？    |
| :--------- | :------------------------------------------------------------------ | :---------------------- | :------------------ |
| **Managed** | 服务器管理设置、plist/注册表或系统级 `managed-settings.json`         | 机器上的所有用户         | 是（由 IT 部署）     |
| **User**    | `~/.claude/` 目录                                                   | 你在所有项目中的设置     | 否                  |
| **Project** | 仓库中的 `.claude/` 目录                                             | 此仓库的所有协作者       | 是（提交到 git）     |
| **Local**   | `.claude/settings.local.json`                                       | 仅你在当前仓库中的设置   | 否（被 gitignore）   |

### 何时使用各作用域

**Managed 作用域**适用于：
- 必须在组织范围内强制执行的安全策略
- 不能被覆盖的合规要求
- IT/DevOps 部署的标准化配置

**User 作用域**适用于：
- 你在所有地方都想要的个人偏好（主题、编辑器设置）
- 你在所有项目中使用的工具和插件
- API 密钥和身份认证（安全存储）

**Project 作用域**适用于：
- 团队共享的设置（权限、hooks、MCP 服务器）
- 整个团队应该拥有的插件
- 跨协作者的标准化工具链

**Local 作用域**适用于：
- 特定项目的个人覆盖设置
- 在共享给团队之前测试配置
- 对他人不适用的机器特定设置

### 作用域优先级

当同一设置在多个作用域中配置时，更具体的作用域优先级更高：

1. **Managed**（最高）— 不能被任何东西覆盖
2. **命令行参数** — 临时会话覆盖
3. **Local** — 覆盖 project 和 user 设置
4. **Project** — 覆盖 user 设置
5. **User**（最低）— 当没有其他作用域指定时生效

例如，如果 user 设置允许某个权限，但 project 设置拒绝它，则 project 设置生效，该权限被阻止。

### 哪些功能使用作用域

| 功能            | User 位置                   | Project 位置                         | Local 位置                       |
| :-------------- | :-------------------------- | :----------------------------------- | :------------------------------- |
| **设置**        | `~/.claude/settings.json`   | `.claude/settings.json`              | `.claude/settings.local.json`    |
| **子代理**      | `~/.claude/agents/`         | `.claude/agents/`                    | 无                               |
| **MCP 服务器**  | `~/.claude.json`            | `.mcp.json`                          | `~/.claude.json`（每项目）       |
| **插件**        | `~/.claude/settings.json`   | `.claude/settings.json`              | `.claude/settings.local.json`    |
| **CLAUDE.md**   | `~/.claude/CLAUDE.md`       | `CLAUDE.md` 或 `.claude/CLAUDE.md`   | `CLAUDE.local.md`                |

---

## 设置文件

`settings.json` 是通过层次化设置配置 Claude Code 的官方机制：

- **User 设置** 定义在 `~/.claude/settings.json`，适用于所有项目
- **Project 设置** 保存在项目目录中：
  - `.claude/settings.json` 用于检入源代码控制并与团队共享的设置
  - `.claude/settings.local.json` 用于不检入的设置，适合个人偏好和实验。Claude Code 会在创建时自动配置 git 忽略它
- **Managed 设置**：对于需要集中控制的组织，Claude Code 支持多种 managed 设置的交付机制。它们都使用相同的 JSON 格式，且不能被 user 或 project 设置覆盖：
  - **服务器管理设置**：通过 Claude.ai 管理控制台从 Anthropic 服务器交付
  - **MDM/OS 级策略**：通过 macOS 和 Windows 上的原生设备管理交付
  - **基于文件**：部署到系统目录的 `managed-settings.json` 和 `managed-mcp.json`

    macOS: `/Library/Application Support/ClaudeCode/`
    Linux 和 WSL: `/etc/claude-code/`
    Windows: `C:\Program Files\ClaudeCode\`

    基于文件的 managed 设置还支持在同级目录下的 `managed-settings.d/` 放置目录。这允许不同团队独立部署策略片段，无需协调编辑单个文件。

    按照 systemd 约定，`managed-settings.json` 首先作为基础合并，然后放置目录中的所有 `*.json` 文件按字母顺序排序并合并。后面的文件覆盖前面的文件的标量值；数组会合并去重；对象会深度合并。以 `.` 开头的隐藏文件会被忽略。

    使用数字前缀控制合并顺序，例如 `10-telemetry.json` 和 `20-security.json`。

- **其他配置** 存储在 `~/.claude.json`。此文件包含你的偏好（主题、通知设置、编辑器模式）、OAuth 会话、MCP 服务器配置（user 和 local 作用域）、每项目状态（允许的工具、信任设置）和各种缓存。Project 作用域的 MCP 服务器单独存储在 `.mcp.json` 中。

> **注意**：Claude Code 会自动创建设置文件的时间戳备份并保留最近五个备份，以防止数据丢失。

```json 示例 settings.json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "permissions": {
    "allow": [
      "Bash(npm run lint)",
      "Bash(npm run test *)",
      "Read(~/.zshrc)"
    ],
    "deny": [
      "Bash(curl *)",
      "Read(./.env)",
      "Read(./.env.*)",
      "Read(./secrets/**)"
    ]
  },
  "env": {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
    "OTEL_METRICS_EXPORTER": "otlp"
  },
  "companyAnnouncements": [
    "欢迎来到 Acme Corp！查看我们的代码规范 docs.acme.com",
    "提醒：所有 PR 都需要代码审查",
    "新的安全策略已生效"
  ]
}
```

`$schema` 行指向 [官方 JSON Schema](https://json.schemastore.org/claude-code-settings.json)。添加到你的 `settings.json` 后可以在 VS Code、Cursor 和任何支持 JSON Schema 验证的编辑器中启用自动补全和内联验证。

### 可用设置

`settings.json` 支持以下选项：

| 键名                              | 描述                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | 示例                                                                                                                         |
| :-------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------------------------------------------- |
| `agent`                           | 将主线程作为命名的子代理运行。应用该子代理的系统提示、工具限制和模型                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | `"code-reviewer"`                                                                                                            |
| `allowedChannelPlugins`           | （仅 managed 设置）允许推送消息的频道插件白名单。设置时会替代 Anthropic 默认白名单。未定义 = 回退到默认值，空数组 = 阻止所有频道插件。需要 `channelsEnabled: true`                                                                                                                                                                                                                                                                                                                                                                                   | `[{ "marketplace": "claude-plugins-official", "plugin": "telegram" }]`                                                       |
| `allowedHttpHookUrls`             | HTTP hooks 可以定位的 URL 模式白名单。支持 `*` 通配符。设置时，不匹配的 URL 的 hooks 会被阻止。未定义 = 无限制，空数组 = 阻止所有 HTTP hooks。数组跨设置源合并                                                                                                                                                                                                                                                                                                                                                                                     | `["https://hooks.example.com/*"]`                                                                                            |
| `allowedMcpServers`               | 在 managed-settings.json 中设置时，用户可以配置的 MCP 服务器白名单。未定义 = 无限制，空数组 = 锁定。适用于所有作用域。黑名单优先级更高                                                                                                                                                                                                                                                                                                                                                                                                                  | `[{ "serverName": "github" }]`                                                                                               |
| `allowManagedHooksOnly`           | （仅 managed 设置）仅加载 managed hooks、SDK hooks 和 managed 设置 `enabledPlugins` 中强制启用的插件的 hooks。阻止 user、project 和所有其他插件 hooks                                                                                                                                                                                                                                                                                                                                                                                              | `true`                                                                                                                       |
| `allowManagedMcpServersOnly`      | （仅 managed 设置）仅尊重 managed 设置中的 `allowedMcpServers`。`deniedMcpServers` 仍然从所有作用域合并。用户仍然可以添加 MCP 服务器，但仅应用管理员定义的白名单                                                                                                                                                                                                                                                                                                                                                                                     | `true`                                                                                                                       |
| `allowManagedPermissionRulesOnly` | （仅 managed 设置）阻止 user 和 project 设置定义 `allow`、`ask` 或 `deny` 权限规则。仅应用 managed 设置中的规则                                                                                                                                                                                                                                                                                                                                                                                                                                        | `true`                                                                                                                       |
| `alwaysThinkingEnabled`           | 默认为所有会话启用[扩展思考](/en/common-workflows#use-extended-thinking-thinking-mode)                                                                                                                                                                                                                                                                                                                                                                                                                                                              | `true`                                                                                                                       |
| `apiKeyHelper`                    | 在 `/bin/sh` 中执行的自定义脚本，用于生成认证值。此值将作为 `X-Api-Key` 和 `Authorization: Bearer` 头发送用于模型请求                                                                                                                                                                                                                                                                                                                                                                                                                                 | `/bin/generate_temp_api_key.sh`                                                                                              |
| `attribution`                     | 自定义 git 提交和拉取请求的归属信息。见[归属设置](#归属设置)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | `{"commit": "🤖 由 Claude Code 生成", "pr": ""}`                                                                             |
| `autoMemoryDirectory`             | [自动记忆](/en/memory#storage-location)存储的自定义目录。接受 `~/` 展开路径。不接受 project 设置（`.claude/settings.json`）以防止共享仓库将记忆写入重定向到敏感位置。接受 policy、local 和 user 设置                                                                                                                                                                                                                                                                                                                                                      | `"~/my-memory-dir"`                                                                                                          |
| `autoMode`                        | 自定义[自动模式](/en/permission-modes#eliminate-prompts-with-auto-mode)分类器阻止和允许的内容。包含 `environment`、`allow` 和 `soft_deny` 散文规则数组。不从共享 project 设置读取                                                                                                                                                                                                                                                                                                                                                                     | `{"environment": ["可信仓库: github.example.com/acme"]}`                                                                     |
| `autoUpdatesChannel`              | 更新关注的发布通道。使用 `"stable"` 获取通常约一周前的版本并跳过有重大回归的版本，或使用 `"latest"`（默认）获取最新发布                                                                                                                                                                                                                                                                                                                                                                                                                                  | `"stable"`                                                                                                                   |
| `availableModels`                 | 限制用户可以通过 `/model`、`--model`、配置工具或 `ANTHROPIC_MODEL` 选择哪些模型。不影响 Default 选项                                                                                                                                                                                                                                                                                                                                                                                                                                                   | `["sonnet", "haiku"]`                                                                                                        |
| `awsAuthRefresh`                  | 修改 `.aws` 目录的自定义脚本                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | `aws sso login --profile myprofile`                                                                                          |
| `awsCredentialExport`             | 输出包含 AWS 凭证的 JSON 的自定义脚本                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | `/bin/generate_aws_grant.sh`                                                                                                 |
| `blockedMarketplaces`             | （仅 managed 设置）被阻止的市场源黑名单。被阻止的源在下载前就被检查，因此永远不会接触文件系统                                                                                                                                                                                                                                                                                                                                                                                                                                                             | `[{ "source": "github", "repo": "untrusted/plugins" }]`                                                                      |
| `channelsEnabled`                 | （仅 managed 设置）允许 Team 和 Enterprise 用户使用[频道](/en/channels)。未设置或 `false` 时无论用户传递给 `--channels` 什么都阻止频道消息传递                                                                                                                                                                                                                                                                                                                                                                                                          | `true`                                                                                                                       |
| `cleanupPeriodDays`               | 超过此期限的会话文件会在启动时被删除（默认：30 天，最小 1 天）。设置为 `0` 会被验证错误拒绝。还控制启动时自动删除[孤离子代理工作树](/en/common-workflows#worktree-cleanup)的年龄截止。要在非交互模式（`-p`）下完全禁用转录写入，使用 `--no-session-persistence` 标志或 `persistSession: false` SDK 选项                                                                 | `20`                                                                                                                         |
| `companyAnnouncements`            | 启动时显示的公告。如果提供了多个公告，将随机循环显示                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | `["欢迎来到 Acme Corp！查看我们的代码规范 docs.acme.com"]`                                                                    |
| `defaultShell`                    | 输入框 `!` 命令的默认 shell。接受 `"bash"`（默认）或 `"powershell"`。设置 `"powershell"` 时在 Windows 上将交互式 `!` 命令路由到 PowerShell。需要 `CLAUDE_CODE_USE_POWERSHELL_TOOL=1`                                                                                                                                                                                                                                                                                                                                                                   | `"powershell"`                                                                                                               |
| `deniedMcpServers`                | 在 managed-settings.json 中设置时，被明确阻止的 MCP 服务器黑名单。适用于所有作用域包括 managed 服务器。黑名单优先级高于白名单                                                                                                                                                                                                                                                                                                                                                                                                                            | `[{ "serverName": "filesystem" }]`                                                                                           |
| `disableAllHooks`                 | 禁用所有 [hooks](/en/hooks) 和任何自定义[状态行](/en/statusline)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | `true`                                                                                                                       |
| `disableAutoMode`                 | 设置为 `"disable"` 以阻止[自动模式](/en/permission-modes#eliminate-prompts-with-auto-mode)被激活。从 `Shift+Tab` 循环中移除 `auto` 并在启动时拒绝 `--permission-mode auto`。在[managed 设置](/en/permissions#managed-settings)中最有用，用户无法覆盖它                                                                                                                                                                                                                                                                                                  | `"disable"`                                                                                                                  |
| `disableDeepLinkRegistration`     | 设置为 `"disable"` 以阻止 Claude Code 在启动时向操作系统注册 `claude-cli://` 协议处理器。深层链接让外部工具通过 `claude-cli://open?q=...` 打开一个预填提示的 Claude Code 会话。`q` 参数支持使用 URL 编码换行符（`%0A`）的多行提示                                                                                                                                                                                                                                                                                                                       | `"disable"`                                                                                                                  |
| `disabledMcpjsonServers`          | 要拒绝的来自 `.mcp.json` 文件的特定 MCP 服务器列表                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | `["filesystem"]`                                                                                                             |
| `disableSkillShellExecution`      | 禁用来自 user、project、插件或附加目录源的[技能](/en/skills)和自定义命令中 `` !`...` `` 和 ` ```! ` 块的内联 shell 执行。命令会被替换为 `[shell command execution disabled by policy]` 而不是运行。捆绑和 managed 技能不受影响。在[managed 设置](/en/permissions#managed-settings)中最有用，用户无法覆盖它                                                                                                                                                    | `true`                                                                                                                       |
| `effortLevel`                     | 持久化[努力级别](/en/model-config#adjust-effort-level)跨会话。接受 `"low"`、`"medium"` 或 `"high"`。当你运行 `/effort low`、`/effort medium` 或 `/effort high` 时自动写入。Opus 4.6 和 Sonnet 4.6 支持                                                                                                                                                                                                                                                                                 | `"medium"`                                                                                                                   |
| `enableAllProjectMcpServers`      | 自动批准项目 `.mcp.json` 文件中定义的所有 MCP 服务器                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | `true`                                                                                                                       |
| `enabledMcpjsonServers`           | 要批准的来自 `.mcp.json` 文件的特定 MCP 服务器列表                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | `["memory", "github"]`                                                                                                       |
| `env`                             | 将应用于每个会话的环境变量                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | `{"FOO": "bar"}`                                                                                                             |
| `fastModePerSessionOptIn`         | 当为 `true` 时，快速模式不会跨会话持久化。每个会话以快速模式关闭开始，需要用 `/fast` 启用。用户的快速模式偏好仍然被保存                                                                                                                                                                                                                                                                                                                                                                                                                                    | `true`                                                                                                                       |
| `feedbackSurveyRate`              | 出现[会话质量调查](/en/data-usage#session-quality-surveys)的概率（0-1）。设置为 `0` 完全抑制。在使用 Bedrock、Vertex 或 Foundry 时有用，默认采样率不适用                                                                                                                                                                                                                                                                                                                                   | `0.05`                                                                                                                       |
| `fileSuggestion`                  | 配置 `@` 文件自动补全的自定义脚本。见[文件建议设置](#文件建议设置)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | `{"type": "command", "command": "~/.claude/file-suggestion.sh"}`                                                             |
| `forceLoginMethod`                | 使用 `claudeai` 限制只能登录 Claude.ai 账户，使用 `console` 限制只能登录 Claude Console（API 使用计费）账户                                                                                                                                                                                                                                                                                                                                                                                                                                               | `claudeai`                                                                                                                   |
| `forceLoginOrgUUID`               | 要求登录属于特定组织。接受单个 UUID 字符串，这会在登录时预选该组织，或 UUID 数组，其中任何列出的组织都被接受但不预选。在 managed 设置中设置时，如果认证的账户不属于列出的组织，登录将失败；空数组会关闭并阻止登录并显示配置错误消息                                                                                                                              | `"xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"` 或 `["uuid1", "uuid2"]` |
| `forceRemoteSettingsRefresh`      | （仅 managed 设置）阻止 CLI 启动直到远程 managed 设置被新鲜地从服务器获取。如果获取失败，CLI 退出而不是继续使用缓存的或没有的设置。未设置时，启动不等待远程设置继续。见[故障关闭执行](/en/server-managed-settings#enforce-fail-closed-startup)                                                                                                                                                                                                                                                              | `true`                                                                                                                       |
| `hooks`                           | 配置在生命周期事件中执行的自定义命令。见[hooks 文档](/en/hooks)了解格式                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | 见 [hooks](/en/hooks)                                                                                                        |
| `httpHookAllowedEnvVars`          | HTTP hooks 可以插值到头部的环境变量名白名单。设置时，每个 hook 的有效 `allowedEnvVars` 是其自身列表与此设置的交集。未定义 = 无限制。数组跨设置源合并                                                                                                                                                                                                                                                                                                                                                                                                          | `["MY_TOKEN", "HOOK_SECRET"]`                                                                                                |
| `includeCoAuthoredBy`             | **已弃用**：使用 `attribution` 代替。是否在 git 提交和拉取请求中包含 `co-authored-by Claude` 署名（默认：`true`）                                                                                                                                                                                                                                                                                                                                                                                                                                            | `false`                                                                                                                      |
| `includeGitInstructions`          | 在 Claude 的系统提示中包含内置的 git 状态快照和 commit/PR 工作流指令（默认：`true`）。设置为 `false` 以移除两者，例如当你使用自己的 git 工作流技能时。`CLAUDE_CODE_DISABLE_GIT_INSTRUCTIONS` 环境变量在被设置时优先于此设置                                                                                                                                                                                                                                                                                                                                 | `false`                                                                                                                      |
| `language`                        | 配置 Claude 的首选响应语言（例如 `"japanese"`、`"spanish"`、`"french"`、`"chinese"`）。Claude 将默认使用此语言响应。同时设置[语音听写](/en/voice-dictation#change-the-dictation-language)语言                                                                                                                                                                                                                                                                                                                      | `"japanese"`                                                                                                                 |
| `model`                           | 覆盖 Claude Code 的默认模型                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | `"claude-sonnet-4-6"`                                                                                                        |
| `modelOverrides`                  | 将 Anthropic 模型 ID 映射到提供商标识符如 Bedrock 推理配置文件 ARN。每个模型选择器条目在调用提供商 API 时使用其映射值。见[按版本覆盖模型 ID](/en/model-config#override-model-ids-per-version)                                                                                                                                                                                                                                                                                                                                                               | `{"claude-opus-4-6": "arn:aws:bedrock:..."}`                                                                                 |
| `otelHeadersHelper`               | 生成动态 OpenTelemetry 头部的脚本。在启动时和定期运行（见[动态头部](/en/monitoring-usage#dynamic-headers)）                                                                                                                                                                                                                                                                                                                                                                                                               | `/bin/generate_otel_headers.sh`                                                                                              |
| `outputStyle`                     | 配置输出样式以调整系统提示。见[输出样式文档](/en/output-styles)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | `"Explanatory"`                                                                                                              |
| `permissions`                     | 见下方权限结构表                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |                                                                                                                              |
| `plansDirectory`                  | 自定义计划文件的存储位置。路径相对于项目根目录。默认：`~/.claude/plans`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | `"./plans"`                                                                                                                  |
| `pluginTrustMessage`              | （仅 managed 设置）自定义消息附加到安装前显示的插件信任警告。使用此选项添加组织特定的上下文，例如确认来自内部市场的插件已通过审查                                                                                                                                                                                                                                                                                                                                                                                                                              | `"我们市场中的所有插件都经过 IT 审查"`                                                                                             |
| `prefersReducedMotion`            | 减少或禁用 UI 动画（旋转器、闪烁、闪光效果）以支持无障碍                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | `true`                                                                                                                       |
| `respectGitignore`                | 控制 `@` 文件选择器是否尊重 `.gitignore` 模式。当为 `true`（默认）时，匹配 `.gitignore` 模式的文件从建议中排除                                                                                                                                                                                                                                                                                                                                                                                                                                                   | `false`                                                                                                                      |
| `showClearContextOnPlanAccept`    | 在计划接受屏幕上显示"清除上下文"选项。默认：`false`。设置为 `true` 以恢复该选项                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | `true`                                                                                                                       |
| `showThinkingSummaries`           | 在交互会话中显示[扩展思考](/en/common-workflows#use-extended-thinking-thinking-mode)摘要。当未设置或 `false`（交互模式默认）时，思考块被 API 编辑并以折叠存根显示。编辑仅改变你看到的内容，不改变模型生成的内容：要减少思考花费，[降低预算或禁用思考](/en/common-workflows#use-extended-thinking-thinking-mode)。非交互模式（`-p`）和 SDK 调用者始终接收摘要，不管此设置如何 | `true`                                                                                                                       |
| `spinnerTipsEnabled`              | 在 Claude 工作时在旋转器中显示提示。设置为 `false` 以禁用提示（默认：`true`）                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | `false`                                                                                                                      |
| `spinnerTipsOverride`             | 用自定义字符串覆盖旋转器提示。`tips`：提示字符串数组。`excludeDefault`：如果为 `true`，仅显示自定义提示；如果为 `false` 或缺失，自定义提示与内置提示合并                                                                                                                                                                                                                                                                                                                                                                                                             | `{ "excludeDefault": true, "tips": ["使用我们的内部工具 X"] }`                                                               |
| `spinnerVerbs`                    | 自定义旋转器和轮次持续时间消息中显示的动作动词。将 `mode` 设置为 `"replace"` 以仅使用你的动词，或 `"append"` 将它们添加到默认值中                                                                                                                                                                                                                                                                                                                                                                                                                                  | `{"mode": "append", "verbs": ["思考中", "构建中"]}`                                                                           |
| `statusLine`                      | 配置自定义状态行以显示上下文。见[`statusLine` 文档](/en/statusline)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | `{"type": "command", "command": "~/.claude/statusline.sh"}`                                                                  |
| `strictKnownMarketplaces`         | （仅 managed 设置）允许用户添加的插件市场白名单。未定义 = 无限制，空数组 = 锁定。仅适用于市场添加。见[市场限制](/en/plugin-marketplaces#managed-marketplace-restrictions)                                                                                                                                                                                                                                                                                                                                                                                                  | `[{ "source": "github", "repo": "acme-corp/plugins" }]`                                                                      |
| `useAutoModeDuringPlan`           | 当自动模式可用时，计划模式是否使用自动模式语义。默认：`true`。不从共享 project 设置读取。在 `/config` 中显示为"在计划期间使用自动模式"                                                                                                                                                                                                                                                                                                                                                                                                                                   | `false`                                                                                                                      |
| `voiceEnabled`                    | 启用按键说话[语音听写](/en/voice-dictation)。运行 `/voice` 时自动写入。需要 Claude.ai 账户                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | `true`                                                                                                                       |

### 全局配置设置

这些设置存储在 `~/.claude.json` 而不是 `settings.json` 中。将它们添加到 `settings.json` 将触发 schema 验证错误。

| 键名                          | 描述                                                                                                                                                                                                                                                                                                              | 示例           |
| :---------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------- |
| `autoConnectIde`              | Claude Code 从外部终端启动时自动连接到运行的 IDE。默认：`false`。在 VS Code 或 JetBrains 终端外部运行时在 `/config` 中显示为 **自动连接到 IDE（外部终端）**                                                                                 | `true`         |
| `autoInstallIdeExtension`     | 从 VS Code 终端运行时自动安装 Claude Code IDE 扩展。默认：`true`。在 VS Code 或 JetBrains 终端内运行时在 `/config` 中显示为 **自动安装 IDE 扩展**。你也可以设置 [`CLAUDE_CODE_IDE_SKIP_AUTO_INSTALL`](/en/env-vars) 环境变量 | `false`        |
| `editorMode`                  | 输入提示的键绑定模式：`"normal"` 或 `"vim"`。默认：`"normal"`。在 `/config` 中显示为 **编辑器模式**                                                                                                                                                                                           | `"vim"`        |
| `showTurnDuration`            | 在响应后显示轮次持续时间消息，例如 "Cooked for 1m 6s"。默认：`true`。在 `/config` 中显示为 **显示轮次持续时间**                                                                                                                                                                                | `false`        |
| `terminalProgressBarEnabled`  | 在支持的终端中显示终端进度条：ConEmu、Ghostty 1.2.0+ 和 iTerm2 3.6.6+。默认：`true`。在 `/config` 中显示为 **终端进度条**                                                                                                                                                 | `false`        |
| `teammateMode`               | [代理团队](/en/agent-teams)队友的显示方式：`auto`（在 tmux 或 iTerm2 中选择分屏，否则在进程内），`in-process`，或 `tmux`。见[选择显示模式](/en/agent-teams#choose-a-display-mode)                                                                                                 | `"in-process"` |

### 工作树（Worktree）设置

配置 `--worktree` 如何创建和管理 git 工作树。使用这些设置减少大型 monorepo 中的磁盘使用和启动时间。

| 键名                           | 描述                                                                                                                                                  | 示例                                |
| :---------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------- |
| `worktree.symlinkDirectories` | 要从主仓库符号链接到每个工作树的目录，以避免在磁盘上复制大型目录。默认不符号链接任何目录   | `["node_modules", ".cache"]`        |
| `worktree.sparsePaths`        | 通过 git sparse-checkout（cone 模式）在每个工作树中签出的目录。仅列出的路径被写入磁盘，在大型 monorepo 中更快 | `["packages/my-app", "shared/utils"]` |

要将 `.env` 等 gitignore 文件复制到新工作树中，在项目根目录使用 [`.worktreeinclude` 文件](/en/common-workflows#copy-gitignored-files-to-worktrees) 而不是设置。

### 权限设置

| 键名                                | 描述                                                                                                                                                                                                                                                                            | 示例                                                                   |
| :---------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :--------------------------------------------------------------------- |
| `allow`                             | 允许工具使用的权限规则数组。见下方[权限规则语法](#权限规则语法)了解模式匹配细节                                                                                                                                                         | `[ "Bash(git diff *)" ]`                                               |
| `ask`                               | 工具使用时要求确认的权限规则数组。见下方[权限规则语法](#权限规则语法)                                                                                                                                                                     | `[ "Bash(git push *)" ]`                                               |
| `deny`                              | 拒绝工具使用的权限规则数组。使用此选项从 Claude Code 访问中排除敏感文件。见下方[权限规则语法](#权限规则语法)和[Bash 权限限制](/en/permissions#tool-specific-permission-rules)                                     | `[ "WebFetch", "Bash(curl *)", "Read(./.env)", "Read(./secrets/**)" ]` |
| `additionalDirectories`             | 文件访问的附加[工作目录](/en/permissions#working-directories)。大多数 `.claude/` 配置[不会从这些目录中发现](/en/permissions#additional-directories-grant-file-access-not-configuration)                                                  | `[ "../docs/" ]`                                                       |
| `defaultMode`                       | 打开 Claude Code 时的默认[权限模式](/en/permission-modes)。有效值：`default`、`acceptEdits`、`plan`、`auto`、`dontAsk`、`bypassPermissions`。`--permission-mode` CLI 标志覆盖此设置用于单次会话                                         | `"acceptEdits"`                                                        |
| `disableBypassPermissionsMode`      | 设置为 `"disable"` 以阻止 `bypassPermissions` 模式被激活。这会禁用 `--dangerously-skip-permissions` 命令行标志。通常放在[managed 设置](/en/permissions#managed-settings)中以执行组织策略，但也适用于任何作用域 | `"disable"`                                                            |
| `skipDangerousModePermissionPrompt` | 跳过通过 `--dangerously-skip-permissions` 或 `defaultMode: "bypassPermissions"` 进入绕过权限模式之前显示的确认提示。在 project 设置（`.claude/settings.json`）中设置时被忽略，以防止不受信任的仓库自动绕过提示 | `true`                                                                 |

### 权限规则语法

权限规则遵循 `Tool` 或 `Tool(specifier)` 格式。规则按顺序评估：先 deny 规则，然后 ask，然后 allow。第一个匹配的规则获胜。

快速示例：

| 规则                           | 效果                                   |
| :----------------------------- | :------------------------------------- |
| `Bash`                         | 匹配所有 Bash 命令                     |
| `Bash(npm run *)`              | 匹配以 `npm run` 开头的命令            |
| `Read(./.env)`                 | 匹配读取 `.env` 文件                   |
| `WebFetch(domain:example.com)` | 匹配对 example.com 的 fetch 请求       |

有关完整的规则语法参考，包括通配符行为、Read、Edit、WebFetch、MCP 和 Agent 规则的特定工具模式，以及 Bash 模式的安全限制，见[权限规则语法](/en/permissions#permission-rule-syntax)。

### 沙箱设置

配置高级沙箱行为。沙箱将 bash 命令与你的文件系统和网络隔离。见[沙箱](/en/sandboxing)了解详情。

| 键名                                   | 描述                                                                                                                                                                                                                                                                                                                                     | 示例                            |
| :------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------ |
| `enabled`                              | 启用 bash 沙箱（macOS、Linux 和 WSL2）。默认：false                                                                                                                                                                                                                                                                                      | `true`                          |
| `failIfUnavailable`                    | 如果 `sandbox.enabled` 为 true 但沙箱无法启动（缺少依赖、不支持的平台或平台限制），在启动时退出并报错。当为 false（默认）时，显示警告并命令在无沙箱的情况下运行。用于需要沙箱作为硬门的 managed 设置部署                         | `true`                          |
| `autoAllowBashIfSandboxed`             | 沙箱化时自动批准 bash 命令。默认：true                                                                                                                                                                                                                                                                                                   | `true`                          |
| `excludedCommands`                     | 应在沙箱外运行的命令                                                                                                                                                                                                                                                                                                                     | `["docker *"]`                  |
| `allowUnsandboxedCommands`             | 允许通过 `dangerouslyDisableSandbox` 参数在沙箱外运行命令。当设置为 `false` 时，`dangerouslyDisableSandbox` 逃生舱完全禁用，所有命令必须在沙箱内运行（或在 `excludedCommands` 中）。用于需要严格沙箱的企业策略。默认：true               | `false`                         |
| `filesystem.allowWrite`                | 沙箱化命令可以写入的附加路径。数组跨所有设置作用域合并：user、project 和 managed 路径被合并，而不是替代。也与 `Edit(...)` 允许权限规则的路径合并。见下方[沙箱路径前缀](#沙箱路径前缀)。                                                              | `["/tmp/build", "~/.kube"]`     |
| `filesystem.denyWrite`                 | 沙箱化命令不能写入的路径。数组跨所有设置作用域合并。也与 `Edit(...)` 拒绝权限规则的路径合并。                                                                                                                                                                              | `["/etc", "/usr/local/bin"]`    |
| `filesystem.denyRead`                  | 沙箱化命令不能读取的路径。数组跨所有设置作用域合并。也与 `Read(...)` 拒绝权限规则的路径合并。                                                                                                                                                                              | `["~/.aws/credentials"]`        |
| `filesystem.allowRead`                 | 在 `denyRead` 区域内重新允许读取的路径。优先于 `denyRead`。数组跨所有设置作用域合并。使用此选项创建仅工作区的读取访问模式。                                                                                                                                                | `["."]`                         |
| `filesystem.allowManagedReadPathsOnly` | （仅 managed 设置）仅尊重 managed 设置中的 `filesystem.allowRead` 路径。`denyRead` 仍然从所有作用域合并。默认：false                                                                                                                                                       | `true`                          |
| `network.allowUnixSockets`             | 沙箱中可访问的 Unix socket 路径（用于 SSH 代理等）                                                                                                                                                                                                                         | `["~/.ssh/agent-socket"]`       |
| `network.allowAllUnixSockets`          | 允许沙箱中的所有 Unix socket 连接。默认：false                                                                                                                                                                                                                             | `true`                          |
| `network.allowLocalBinding`            | 允许绑定到 localhost 端口（仅 macOS）。默认：false                                                                                                                                                                                                                         | `true`                          |
| `network.allowMachLookup`              | 沙箱可以查找的附加 XPC/Mach 服务名（仅 macOS）。支持单个尾随 `*` 用于前缀匹配。用于通过 XPC 通信的工具如 iOS 模拟器或 Playwright。                                                                                                                                         | `["com.apple.coresimulator.*"]` |
| `network.allowedDomains`               | 允许出站网络流量的域名数组。支持通配符（例如 `*.example.com`）。                                                                                                                                                                                                           | `["github.com", "*.npmjs.org"]` |
| `network.allowManagedDomainsOnly`      | （仅 managed 设置）仅尊重 managed 设置中的 `allowedDomains` 和 `WebFetch(domain:...)` 允许规则。忽略 user、project 和 local 设置中的域名。未允许的域名自动被阻止而不提示用户。拒绝的域名仍然尊重所有来源。默认：false              | `true`                          |
| `network.httpProxyPort`                | 如果你想使用自己的代理的 HTTP 代理端口。如果未指定，Claude 将运行自己的代理。                                                                                                                                                                                              | `8080`                          |
| `network.socksProxyPort`               | 如果你想使用自己的代理的 SOCKS5 代理端口。如果未指定，Claude 将运行自己的代理。                                                                                                                                                                                            | `8081`                          |
| `enableWeakerNestedSandbox`            | 在无特权的 Docker 环境中启用较弱的沙箱（仅 Linux 和 WSL2）。**降低安全性。**默认：false                                                                                                                                                                                    | `true`                          |
| `enableWeakerNetworkIsolation`         | （仅 macOS）允许在沙箱中访问系统 TLS 信任服务（`com.apple.trustd.agent`）。使用 `httpProxyPort` 与 MITM 代理和自定义 CA 时，Go 类工具如 `gh`、`gcloud` 和 `terraform` 验证 TLS 证书需要。**降低安全性**，可能打开数据泄露路径。默认：false | `true`                          |

#### 沙箱路径前缀

`filesystem.allowWrite`、`filesystem.denyWrite`、`filesystem.denyRead` 和 `filesystem.allowRead` 中的路径支持这些前缀：

| 前缀            | 含义                                                                                | 示例                                                                    |
| :-------------- | :---------------------------------------------------------------------------------- | :---------------------------------------------------------------------- |
| `/`             | 从文件系统根的绝对路径                                                                | `/tmp/build` 保持 `/tmp/build`                                          |
| `~/`            | 相对于家目录的路径                                                                  | `~/.kube` 变为 `$HOME/.kube`                                            |
| `./` 或无前缀   | 对于 project 设置相对于项目根，对于 user 设置相对于 `~/.claude`                     | `./output` 在 `.claude/settings.json` 中解析为 `<project-root>/output`  |

较旧的 `//path` 前缀用于绝对路径仍然有效。如果你以前使用单斜杠 `/path` 期望项目相对解析，切换到 `./path`。此语法不同于[Read 和 Edit 权限规则](/en/permissions#read-and-edit)，后者使用 `//path` 表示绝对和 `/path` 表示项目相对。沙箱文件系统路径使用标准约定：`/tmp/build` 是绝对路径。

**配置示例：**

```json
{
  "sandbox": {
    "enabled": true,
    "autoAllowBashIfSandboxed": true,
    "excludedCommands": ["docker *"],
    "filesystem": {
      "allowWrite": ["/tmp/build", "~/.kube"],
      "denyRead": ["~/.aws/credentials"]
    },
    "network": {
      "allowedDomains": ["github.com", "*.npmjs.org", "registry.yarnpkg.com"],
      "allowUnixSockets": [
        "/var/run/docker.sock"
      ],
      "allowLocalBinding": true
    }
  }
}
```

**文件系统和网络限制**可以通过两种方式配置并合并：

- **`sandbox.filesystem` 设置**（如上所示）：在 OS 级沙箱边界控制路径。这些限制适用于所有子进程命令（例如 `kubectl`、`terraform`、`npm`），而不仅是 Claude 的文件工具
- **权限规则**：使用 `Edit` 允许/拒绝规则控制 Claude 的文件工具访问，`Read` 拒绝规则阻止读取，`WebFetch` 允许/拒绝规则控制网络域名。这些规则的路径也被合并到沙箱配置中

### 归属设置

Claude Code 为 git 提交和拉取请求添加归属。这些分别配置：

- 提交默认使用 [git trailers](https://git-scm.com/docs/git-interpret-trailers)（如 `Co-Authored-By`），可以自定义或禁用
- 拉取请求描述是纯文本

| 键名     | 描述                                                                                |
| :------- | :---------------------------------------------------------------------------------- |
| `commit` | git 提交的归属信息，包括任何 trailers。空字符串隐藏提交归属 |
| `pr`     | 拉取请求描述的归属信息。空字符串隐藏拉取请求归属     |

**默认提交归属：**

```text
🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```

**默认拉取请求归属：**

```text
🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

**示例：**

```json
{
  "attribution": {
    "commit": "Generated with AI\n\nCo-Authored-By: AI <ai@example.com>",
    "pr": ""
  }
}
```

> **注意**：`attribution` 设置优先于已弃用的 `includeCoAuthoredBy` 设置。要隐藏所有归属，将 `commit` 和 `pr` 设置为空字符串。

### 文件建议设置

为 `@` 文件路径自动补全配置自定义命令。内置文件建议使用快速文件系统遍历，但大型 monorepo 可能受益于项目特定的索引如预构建的文件索引或自定义工具。

```json
{
  "fileSuggestion": {
    "type": "command",
    "command": "~/.claude/file-suggestion.sh"
  }
}
```

命令运行时有与 [hooks](/en/hooks) 相同的环境变量，包括 `CLAUDE_PROJECT_DIR`。它通过 stdin 接收包含 `query` 字段的 JSON：

```json
{"query": "src/comp"}
```

输出换行分隔的文件路径到 stdout（当前限制为 15 个）：

```text
src/components/Button.tsx
src/components/Modal.tsx
src/components/Form.tsx
```

**示例：**

```bash
#!/bin/bash
query=$(cat | jq -r '.query')
your-repo-file-index --query "$query" | head -20
```

### Hook 配置

这些设置控制哪些 hooks 被允许运行以及 HTTP hooks 可以访问什么。`allowManagedHooksOnly` 设置只能在 [managed 设置](#设置文件)中配置。URL 和环境变量白名单可以在任何设置级别设置并跨源合并。

**当 `allowManagedHooksOnly` 为 `true` 时的行为：**

- 加载 Managed hooks 和 SDK hooks
- 加载 managed 设置 `enabledPlugins` 中强制启用的插件的 hooks。这允许管理员通过组织市场分发经过审查的 hooks，同时阻止其他所有。信任通过完整的 `plugin@marketplace` ID 授予，因此来自不同市场的同名插件仍然被阻止
- 阻止 user hooks、project hooks 和所有其他插件 hooks

**限制 HTTP hook URL：**

限制 HTTP hooks 可以定位的 URL。支持 `*` 通配符用于匹配。当数组被定义时，定位不匹配 URL 的 HTTP hooks 被静默阻止。

```json
{
  "allowedHttpHookUrls": ["https://hooks.example.com/*", "http://localhost:*"]
}
```

**限制 HTTP hook 环境变量：**

限制 HTTP hooks 可以插值到头部值中的环境变量名。每个 hook 的有效 `allowedEnvVars` 是其自身列表与此设置的交集。

```json
{
  "httpHookAllowedEnvVars": ["MY_TOKEN", "HOOK_SECRET"]
}
```

### 设置优先级

设置按优先级顺序应用。从高到低：

1. **Managed 设置**（[服务器管理](/en/server-managed-settings)、[MDM/OS 级策略](#配置作用域)或[managed 设置](#设置文件)）
   - 通过服务器交付、MDM 配置策略、注册表策略或 managed 设置文件部署的策略
   - 不能被任何其他级别覆盖，包括命令行参数
   - 在 managed 层级内，优先级为：服务器管理 > MDM/OS 级策略 > 基于文件（`managed-settings.d/*.json` + `managed-settings.json`）> HKCU 注册表（仅 Windows）。仅使用一个 managed 源；源不跨层级合并。在基于文件的层级内，放置文件和基础文件被合并

2. **命令行参数**
   - 特定会话的临时覆盖

3. **Local 项目设置**（`.claude/settings.local.json`）
   - 个人项目特定设置

4. **共享项目设置**（`.claude/settings.json`）
   - 源代码控制中的团队共享设置

5. **User 设置**（`~/.claude/settings.json`）
   - 个人全局设置

这个层级结构确保组织策略始终被强制执行，同时允许团队和个人自定义他们的体验。无论你从 CLI、[VS Code 扩展](/en/vs-code)还是[JetBrains IDE](/en/jetbrains)运行 Claude Code，相同的优先级都适用。

例如，如果你的 user 设置允许 `Bash(npm run *)` 但项目的共享设置拒绝它，则 project 设置生效，命令被阻止。

> **注意**：**数组设置跨作用域合并。** 当同一数组值设置（如 `sandbox.filesystem.allowWrite` 或 `permissions.allow`）出现在多个作用域时，数组**被连接和去重**，而不是替代。这意味着低优先级作用域可以添加条目而不覆盖高优先级作用域设置的条目，反之亦然。例如，如果 managed 设置将 `allowWrite` 设置为 `["/opt/company-tools"]` 而用户添加了 `["~/.kube"]`，两个路径都包含在最终配置中。

### 验证活动设置

在 Claude Code 中运行 `/status` 以查看哪些设置源是活动的以及它们来自哪里。输出显示每个配置层（managed、user、project）及其来源，如 `Enterprise managed settings (remote)`、`Enterprise managed settings (plist)`、`Enterprise managed settings (HKLM)` 或 `Enterprise managed settings (file)`。如果设置文件包含错误，`/status` 会报告问题以便你修复它。

### 配置系统要点

- **记忆文件（`CLAUDE.md`）**：包含 Claude 在启动时加载的指令和上下文
- **设置文件（JSON）**：配置权限、环境变量和工具行为
- **技能**：可以用 `/skill-name` 调用或由 Claude 自动加载的自定义提示
- **MCP 服务器**：用额外的工具和集成扩展 Claude Code
- **优先级**：更高级别的配置（Managed）覆盖更低级别的配置（User/Project）
- **继承**：设置被合并，更具体的设置添加到或覆盖更广泛的设置

### 排除敏感文件

要阻止 Claude Code 访问包含敏感信息如 API 密钥、密钥和环境变量的文件，在 `.claude/settings.json` 文件中使用 `permissions.deny` 设置：

```json
{
  "permissions": {
    "deny": [
      "Read(./.env)",
      "Read(./.env.*)",
      "Read(./secrets/**)",
      "Read(./config/credentials.json)",
      "Read(./build)"
    ]
  }
}
```

这替代了已弃用的 `ignorePatterns` 配置。匹配这些模式的文件被从文件发现和搜索结果中排除，并且对这些文件的读取操作被拒绝。

---

## 子代理配置

Claude Code 支持自定义 AI 子代理，可以在 user 和 project 级别配置。这些子代理存储为带 YAML frontmatter 的 Markdown 文件：

- **User 子代理**：`~/.claude/agents/` — 在你的所有项目中可用
- **Project 子代理**：`.claude/agents/` — 特定于你的项目并可以与团队共享

子代理文件定义了具有自定义提示和工具权限的专业 AI 助手。在[子代理文档](/en/sub-agents)中了解更多关于创建和使用子代理的信息。

---

## 插件配置

Claude Code 支持插件系统，可以用技能、代理、hooks 和 MCP 服务器扩展功能。插件通过市场分发，可以在 user 和仓库级别配置。

### 插件设置

`settings.json` 中插件相关的设置：

```json
{
  "enabledPlugins": {
    "formatter@acme-tools": true,
    "deployer@acme-tools": true,
    "analyzer@security-plugins": false
  },
  "extraKnownMarketplaces": {
    "acme-tools": {
      "source": "github",
      "repo": "acme-corp/claude-plugins"
    }
  }
}
```

#### `enabledPlugins`

控制哪些插件被启用。格式：`"plugin-name@marketplace-name": true/false`

**作用域**：

- **User 设置**（`~/.claude/settings.json`）：个人插件偏好
- **Project 设置**（`.claude/settings.json`）：项目特定插件与团队共享
- **Local 设置**（`.claude/settings.local.json`）：每台机器的覆盖（不提交）
- **Managed 设置**（`managed-settings.json`）：组织范围的策略覆盖，在所有作用域阻止安装并从市场隐藏插件

**示例**：

```json
{
  "enabledPlugins": {
    "code-formatter@team-tools": true,
    "deployment-tools@team-tools": true,
    "experimental-features@personal": false
  }
}
```

#### `extraKnownMarketplaces`

定义应该为仓库提供的附加市场。通常在仓库级设置中使用，以确保团队成员可以访问所需的插件源。

**当仓库包含 `extraKnownMarketplaces` 时**：

1. 团队成员在信任文件夹时被提示安装市场
2. 然后团队成员被提示从该市场安装插件
3. 用户可以跳过不需要的市场或插件（存储在 user 设置中）
4. 安装尊重信任边界并需要明确同意

**示例**：

```json
{
  "extraKnownMarketplaces": {
    "acme-tools": {
      "source": {
        "source": "github",
        "repo": "acme-corp/claude-plugins"
      }
    },
    "security-plugins": {
      "source": {
        "source": "git",
        "url": "https://git.example.com/security/plugins.git"
      }
    }
  }
}
```

**市场源类型**：

- `github`：GitHub 仓库（使用 `repo`）
- `git`：任何 git URL（使用 `url`）
- `directory`：本地文件系统路径（仅用于开发）
- `hostPattern`：匹配市场主机的正则表达式模式（使用 `hostPattern`）
- `settings`：直接在 settings.json 中声明的内联市场，无需单独的托管仓库（使用 `name` 和 `plugins`）

使用 `source: 'settings'` 内联声明少量插件而无需设置托管市场仓库。这里的插件必须引用外部源如 GitHub 或 npm。你仍然需要在 `enabledPlugins` 中单独启用每个插件。

```json
{
  "extraKnownMarketplaces": {
    "team-tools": {
      "source": {
        "source": "settings",
        "name": "team-tools",
        "plugins": [
          {
            "name": "code-formatter",
            "source": {
              "source": "github",
              "repo": "acme-corp/code-formatter"
            }
          }
        ]
      }
    }
  }
}
```

#### `strictKnownMarketplaces`

**仅 managed 设置**：控制允许用户添加哪些插件市场。此设置只能在 [managed 设置](#设置文件)中配置，为管理员提供对市场源头的严格控制。

**Managed 设置文件位置**：

- **macOS**：`/Library/Application Support/ClaudeCode/managed-settings.json`
- **Linux 和 WSL**：`/etc/claude-code/managed-settings.json`
- **Windows**：`C:\Program Files\ClaudeCode\managed-settings.json`

**关键特性**：

- 仅在 managed 设置中可用（`managed-settings.json`）
- 不能被 user 或 project 设置覆盖（最高优先级）
- 在网络/文件系统操作之前强制执行（被阻止的源永远不会执行）
- 使用精确匹配来指定源（包括 `ref`、`git` 源的 `path`），除了 `hostPattern` 使用正则表达式匹配

**白名单行为**：

- `undefined`（默认）：无限制 — 用户可以添加任何市场
- 空数组 `[]`：完全锁定 — 用户无法添加任何新市场
- 源列表：用户只能添加精确匹配的市场

**所有支持的源类型**：

白名单支持多种市场源类型。大多数源使用精确匹配，而 `hostPattern` 对市场主机使用正则表达式匹配。

1. **GitHub 仓库**：

```json
{ "source": "github", "repo": "acme-corp/approved-plugins" }
{ "source": "github", "repo": "acme-corp/security-tools", "ref": "v2.0" }
{ "source": "github", "repo": "acme-corp/plugins", "ref": "main", "path": "marketplace" }
```

字段：`repo`（必需），`ref`（可选：分支/标签/SHA），`path`（可选：子目录）

2. **Git 仓库**：

```json
{ "source": "git", "url": "https://gitlab.example.com/tools/plugins.git" }
{ "source": "git", "url": "https://bitbucket.org/acme-corp/plugins.git", "ref": "production" }
{ "source": "git", "url": "ssh://git@git.example.com/plugins.git", "ref": "v3.1", "path": "approved" }
```

字段：`url`（必需），`ref`（可选：分支/标签/SHA），`path`（可选：子目录）

3. **基于 URL 的市场**：

```json
{ "source": "url", "url": "https://plugins.example.com/marketplace.json" }
{ "source": "url", "url": "https://cdn.example.com/marketplace.json", "headers": { "Authorization": "Bearer ${TOKEN}" } }
```

字段：`url`（必需），`headers`（可选：用于认证访问的 HTTP 头部）

> 基于 URL 的市场仅下载 `marketplace.json` 文件。它们不从服务器下载插件文件。基于 URL 的市场中的插件必须使用外部源（GitHub、npm 或 git URL），而不是相对路径。对于使用相对路径的插件，使用基于 Git 的市场。

4. **NPM 包**：

```json
{ "source": "npm", "package": "@acme-corp/claude-plugins" }
{ "source": "npm", "package": "@acme-corp/approved-marketplace" }
```

字段：`package`（必需，支持作用域包）

5. **文件路径**：

```json
{ "source": "file", "path": "/usr/local/share/claude/acme-marketplace.json" }
{ "source": "file", "path": "/opt/acme-corp/plugins/marketplace.json" }
```

字段：`path`（必需：marketplace.json 文件的绝对路径）

6. **目录路径**：

```json
{ "source": "directory", "path": "/usr/local/share/claude/acme-plugins" }
{ "source": "directory", "path": "/opt/acme-corp/approved-marketplaces" }
```

字段：`path`（必需：包含 `.claude-plugin/marketplace.json` 的目录的绝对路径）

7. **主机模式匹配**：

```json
{ "source": "hostPattern", "hostPattern": "^github\\.example\\.com$" }
{ "source": "hostPattern", "hostPattern": "^gitlab\\.internal\\.example\\.com$" }
```

字段：`hostPattern`（必需：用于匹配市场主机的正则表达式模式）

当你想允许来自特定主机的所有市场而无需列举每个仓库时使用主机模式匹配。这对于拥有内部 GitHub Enterprise 或 GitLab 服务器的组织很有用，开发人员可以在其中创建自己的市场。

按源类型提取主机：

- `github`：始终匹配 `github.com`
- `git`：从 URL 提取主机名（支持 HTTPS 和 SSH 格式）
- `url`：从 URL 提取主机名
- `npm`、`file`、`directory`：不支持主机模式匹配

**配置示例**：

示例：仅允许特定市场：

```json
{
  "strictKnownMarketplaces": [
    {
      "source": "github",
      "repo": "acme-corp/approved-plugins"
    },
    {
      "source": "github",
      "repo": "acme-corp/security-tools",
      "ref": "v2.0"
    },
    {
      "source": "url",
      "url": "https://plugins.example.com/marketplace.json"
    },
    {
      "source": "npm",
      "package": "@acme-corp/compliance-plugins"
    }
  ]
}
```

示例 — 禁用所有市场添加：

```json
{
  "strictKnownMarketplaces": []
}
```

示例：允许来自内部 git 服务器的所有市场：

```json
{
  "strictKnownMarketplaces": [
    {
      "source": "hostPattern",
      "hostPattern": "^github\\.example\\.com$"
    }
  ]
}
```

**精确匹配要求**：

市场源必须**精确**匹配才能允许用户添加。对于基于 git 的源（`github` 和 `git`），这包括所有可选字段：

- `repo` 或 `url` 必须精确匹配
- `ref` 字段必须精确匹配（或两者都未定义）
- `path` 字段必须精确匹配（或两者都未定义）

**不匹配**的源示例：

```json
// 这些是不同源：
{ "source": "github", "repo": "acme-corp/plugins" }
{ "source": "github", "repo": "acme-corp/plugins", "ref": "main" }

// 这些也不同：
{ "source": "github", "repo": "acme-corp/plugins", "path": "marketplace" }
{ "source": "github", "repo": "acme-corp/plugins" }
```

**与 `extraKnownMarketplaces` 的比较**：

| 方面                | `strictKnownMarketplaces`          | `extraKnownMarketplaces`           |
| ------------------- | ----------------------------------- | ----------------------------------- |
| **目的**           | 组织策略执行                        | 团队便利性                          |
| **设置文件**       | `managed-settings.json` 仅          | 任何设置文件                        |
| **行为**           | 阻止未列入白名单的添加              | 自动安装缺失的市场                  |
| **何时执行**       | 在网络/文件系统操作之前             | 在用户信任提示之后                  |
| **可被覆盖**       | 否（最高优先级）                    | 是（由更高优先级设置）              |
| **源格式**         | 直接源对象                          | 带嵌套源的命名市场                  |
| **用例**           | 合规、安全限制                      | 入职、标准化                        |

**格式差异**：

`strictKnownMarketplaces` 使用直接源对象：

```json
{
  "strictKnownMarketplaces": [
    { "source": "github", "repo": "acme-corp/plugins" }
  ]
}
```

`extraKnownMarketplaces` 需要命名市场：

```json
{
  "extraKnownMarketplaces": {
    "acme-tools": {
      "source": { "source": "github", "repo": "acme-corp/plugins" }
    }
  }
}
```

**一起使用**：

`strictKnownMarketplaces` 是策略门控：它控制用户可以添加什么但不注册任何市场。要同时限制和为所有用户预注册市场，在 `managed-settings.json` 中同时设置：

```json
{
  "strictKnownMarketplaces": [
    { "source": "github", "repo": "acme-corp/plugins" }
  ],
  "extraKnownMarketplaces": {
    "acme-tools": {
      "source": { "source": "github", "repo": "acme-corp/plugins" }
    }
  }
}
```

仅设置 `strictKnownMarketplaces` 时，用户仍然可以通过 `/plugin marketplace add` 手动添加允许的市场，但不会自动可用。

**重要事项**：

- 限制在任何网络请求或文件系统操作之前被检查
- 当被阻止时，用户看到清晰的错误消息，指出该源被 managed 策略阻止
- 限制仅适用于添加新市场；之前安装的市场仍然可以访问
- Managed 设置具有最高优先级且不能被覆盖

见[Managed 市场限制](/en/plugin-marketplaces#managed-marketplace-restrictions)了解面向用户的文档。

### 管理插件

使用 `/plugin` 命令以交互方式管理插件：

- 浏览市场中的可用插件
- 安装/卸载插件
- 启用/禁用插件
- 查看插件详情（技能、代理、hooks）
- 添加/移除市场

在[插件文档](/en/plugins)中了解更多关于插件系统的信息。

---

## 环境变量

环境变量让你无需编辑设置文件即可控制 Claude Code 行为。任何变量也可以在 [`settings.json`](#可用设置) 的 `env` 键下配置以应用于每个会话或推广给你的团队。

见[环境变量参考](/en/env-vars)获取完整列表。

---

## Claude 可用的工具

Claude Code 可以访问一组用于读取、编辑、搜索、运行命令和编排子代理的工具。工具名称是你在权限规则和 hook 匹配器中使用的精确字符串。

见[工具参考](/en/tools-reference)获取完整列表和 Bash 工具行为详情。

---

## 提交反馈

如果你发现本文档中有不正确、过时或令人困惑的内容，请通过 POST 提交反馈到：
https://code.claude.com/docs/_mintlify/feedback/claude-code/agent-feedback

请求体（JSON）：`{ "path": "/current-page-path", "feedback": "问题描述" }`

仅在你有具体且可操作的报告时提交反馈 — 不要为你访问的每个页面提交反馈。
