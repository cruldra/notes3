# 通过 MCP 连接 Claude Code 到工具

> 了解如何通过模型上下文协议（Model Context Protocol）将 Claude Code 连接到你的工具。

Claude Code 可以通过 [模型上下文协议（MCP）](https://modelcontextprotocol.io/introduction) 连接到数百种外部工具和数据源，这是一个用于 AI 工具集成的开源标准。MCP 服务器让 Claude Code 能够访问你的工具、数据库和 API。

当你发现自己从其他工具（如问题跟踪器或监控仪表板）复制数据到聊天中时，连接一个服务器。连接后，Claude 可以直接读取和操作该系统，而不是依赖你粘贴的内容。

---

## 使用 MCP 可以做什么

连接 MCP 服务器后，你可以要求 Claude Code：

- **从问题跟踪器实现功能**："添加 JIRA 问题 ENG-4521 中描述的功能并在 GitHub 上创建 PR。"
- **分析监控数据**："检查 Sentry 和 Statsig 以查看 ENG-4521 中描述的功能的使用情况。"
- **查询数据库**："根据我们的 PostgreSQL 数据库，找到使用功能 ENG-4521 的 10 个随机用户的邮箱。"
- **集成设计**："根据 Slack 中发布的新 Figma 设计更新我们的标准邮件模板。"
- **自动化工作流**："创建 Gmail 草稿邀请这 10 个用户参加关于新功能反馈会话。"
- **响应外部事件**：MCP 服务器还可以作为[频道](/en/channels)将消息推送到你的会话中，这样 Claude 可以在你离开时响应 Telegram 消息、Discord 聊天或 webhook 事件。

---

## 流行的 MCP 服务器

以下是一些可以连接到 Claude Code 的常用 MCP 服务器：

> **警告**：使用第三方 MCP 服务器的风险由你自己承担 — Anthropic 未验证所有这些服务器的正确性或安全性。确保你信任安装的 MCP 服务器。特别小心使用可能获取不受信任内容的 MCP 服务器，因为它们可能使你面临提示注入风险。

> **注意**：**需要特定的集成？** [在 GitHub 上查找数百个更多 MCP 服务器](https://github.com/modelcontextprotocol/servers)，或使用 [MCP SDK](https://modelcontextprotocol.io/quickstart/server) 构建你自己的。

---

## 安装 MCP 服务器

MCP 服务器可以根据你的需求通过三种不同方式配置：

### 选项 1：添加远程 HTTP 服务器

HTTP 服务器是连接远程 MCP 服务器的推荐方式。这是基于云的服务最广泛支持的传输方式。

```bash
# 基本语法
claude mcp add --transport http <名称> <url>

# 实际示例：连接到 Notion
claude mcp add --transport http notion https://mcp.notion.com/mcp

# 带 Bearer token 示例
claude mcp add --transport http secure-api https://api.example.com/mcp \
  --header "Authorization: Bearer your-token"
```

### 选项 2：添加远程 SSE 服务器

> **警告**：SSE（Server-Sent Events）传输已弃用。在可用的情况下改用 HTTP 服务器。

```bash
# 基本语法
claude mcp add --transport sse <名称> <url>

# 实际示例：连接到 Asana
claude mcp add --transport sse asana https://mcp.asana.com/sse

# 带认证头示例
claude mcp add --transport sse private-api https://api.company.com/sse \
  --header "X-API-Key: your-key-here"
```

### 选项 3：添加本地 stdio 服务器

Stdio 服务器作为本地进程在你的机器上运行。它们非常适合需要直接系统访问或自定义脚本的工具。

```bash
# 基本语法
claude mcp add [选项] <名称> -- <命令> [参数...]

# 实际示例：添加 Airtable 服务器
claude mcp add --transport stdio --env AIRTABLE_API_KEY=YOUR_KEY airtable \
  -- npx -y airtable-mcp-server
```

> **注意**：**重要：选项顺序**
>
> 所有选项（`--transport`、`--env`、`--scope`、`--header`）必须放在服务器名称**之前**。`--`（双破折号）然后将服务器名称与传递给 MCP 服务器的命令和参数分隔开。
>
> 例如：
>
> - `claude mcp add --transport stdio myserver -- npx server` → 运行 `npx server`
> - `claude mcp add --transport stdio --env KEY=value myserver -- python server.py --port 8080` → 在 `KEY=value` 环境变量下运行 `python server.py --port 8080`
>
> 这防止了 Claude 的标志和服务器的标志之间的冲突。

### 管理你的服务器

配置完成后，你可以使用以下命令管理你的 MCP 服务器：

```bash
# 列出所有已配置的服务器
claude mcp list

# 获取特定服务器的详细信息
claude mcp get github

# 移除服务器
claude mcp remove github

# （在 Claude Code 内）检查服务器状态
/mcp
```

### 动态工具更新

Claude Code 支持 MCP `list_changed` 通知，允许 MCP 服务器动态更新其可用工具、提示和资源，无需你断开和重新连接。当 MCP 服务器发送 `list_changed` 通知时，Claude Code 会自动刷新该服务器的可用功能。

### 通过频道推送消息

MCP 服务器还可以直接推送消息到你的会话，这样 Claude 可以响应外部事件，如 CI 结果、监控警报或聊天消息。要启用此功能，你的服务器声明 `claude/channel` 能力，并在启动时用 `--channels` 标志选择加入。见[频道](/en/channels)使用官方支持的频道，或[频道参考](/en/channels-reference)构建你自己的。

> **提示**：
>
> - 使用 `--scope` 标志指定配置的存储位置：
>   - `local`（默认）：仅在当前项目中对你可用（在旧版本中称为 `project`）
>   - `project`：通过 `.mcp.json` 文件与项目中的所有人共享
>   - `user`：在所有项目中对你可用（在旧版本中称为 `global`）
> - 使用 `--env` 标志设置环境变量（例如 `--env KEY=value`）
> - 使用 MCP_TIMEOUT 环境变量配置 MCP 服务器启动超时（例如 `MCP_TIMEOUT=10000 claude` 设置 10 秒超时）
> - 当 MCP 工具输出超过 10,000 token 时，Claude Code 会显示警告。要增加此限制，设置 MAX_MCP_OUTPUT_TOKENS 环境变量（例如 `MAX_MCP_OUTPUT_TOKENS=50000`）
> - 使用 `/mcp` 对需要 OAuth 2.0 认证的远程服务器进行认证

> **警告**：**Windows 用户**：在原生 Windows 上（非 WSL），使用 `npx` 的本地 MCP 服务器需要 `cmd /c` 包装器以确保正确执行。
>
> ```bash
> # 这会创建 Windows 可以执行的 command="cmd"
> claude mcp add --transport stdio my-server -- cmd /c npx -y @some/package
> ```
>
> 没有 `cmd /c` 包装器，你会遇到"Connection closed"错误，因为 Windows 无法直接执行 `npx`。

### 插件提供的 MCP 服务器

[插件](/en/plugins) 可以捆绑 MCP 服务器，在插件启用时自动提供工具和集成。插件 MCP 服务器与用户配置的服务器工作方式相同。

**插件 MCP 服务器如何工作**：

- 插件在插件根目录的 `.mcp.json` 或 `plugin.json` 中内联定义 MCP 服务器
- 当插件启用时，其 MCP 服务器自动启动
- 插件 MCP 工具与手动配置的 MCP 工具一起出现
- 插件服务器通过插件安装管理（而不是 `/mcp` 命令）

**插件 MCP 配置示例**：

在插件根目录的 `.mcp.json` 中：

```json
{
  "mcpServers": {
    "database-tools": {
      "command": "${CLAUDE_PLUGIN_ROOT}/servers/db-server",
      "args": ["--config", "${CLAUDE_PLUGIN_ROOT}/config.json"],
      "env": {
        "DB_URL": "${DB_URL}"
      }
    }
  }
}
```

或在 `plugin.json` 中内联：

```json
{
  "name": "my-plugin",
  "mcpServers": {
    "plugin-api": {
      "command": "${CLAUDE_PLUGIN_ROOT}/servers/api-server",
      "args": ["--port", "8080"]
    }
  }
}
```

**插件 MCP 功能**：

- **自动生命周期**：在会话启动时，已启用插件的服务器自动连接。如果在会话期间启用或禁用插件，运行 `/reload-plugins` 来连接或断开其 MCP 服务器
- **环境变量**：使用 `${CLAUDE_PLUGIN_ROOT}` 引用捆绑的插件文件，使用 `${CLAUDE_PLUGIN_DATA}` 引用[持久状态目录](/en/plugins-reference#persistent-data-directory)（在插件更新后保留）
- **用户环境访问**：与手动配置的服务器相同的环境变量访问
- **多种传输类型**：支持 stdio、SSE 和 HTTP 传输（传输支持可能因服务器而异）

**查看插件 MCP 服务器**：

```bash
# 在 Claude Code 内，查看所有 MCP 服务器包括插件的
/mcp
```

插件服务器在列表中显示，带有来自插件的指示器。

**插件 MCP 服务器的好处**：

- **捆绑分发**：工具和服务器打包在一起
- **自动设置**：无需手动 MCP 配置
- **团队一致性**：安装插件后每个人都获得相同的工具

有关使用插件捆绑 MCP 服务器的详细信息，见[插件组件参考](/en/plugins-reference#mcp-servers)。

---

## MCP 安装作用域

MCP 服务器可以在三个作用域配置。你选择的作用域控制服务器在哪些项目中加载以及配置是否与团队共享。

| 作用域                  | 加载位置         | 与团队共享               | 存储位置                    |
| ----------------------- | ---------------- | ------------------------ | --------------------------- |
| [Local](#local-作用域)  | 仅当前项目       | 否                       | `~/.claude.json`            |
| [Project](#project-作用域) | 仅当前项目    | 是，通过版本控制         | 项目根目录中的 `.mcp.json`  |
| [User](#user-作用域)    | 你的所有项目     | 否                       | `~/.claude.json`            |

### Local 作用域

Local 作用域是默认值。Local 作用域的服务器仅在你添加它的项目中加载，并且对你保持私有。Claude Code 将其存储在 `~/.claude.json` 中该项目路径下，因此相同的服务器不会出现在你的其他项目中。将 local 作用域用于个人开发服务器、实验性配置或包含你不想放入版本控制的凭据的服务器。

> **注意**：MCP 服务器的"local 作用域"术语与常规 local 设置不同。MCP local 作用域的服务器存储在 `~/.claude.json`（你的主目录），而常规 local 设置使用 `.claude/settings.local.json`（在项目目录中）。详情见[设置](/en/settings#settings-files)。

```bash
# 添加 local 作用域的服务器（默认）
claude mcp add --transport http stripe https://mcp.stripe.com

# 显式指定 local 作用域
claude mcp add --transport http stripe --scope local https://mcp.stripe.com
```

该命令将服务器写入 `~/.claude.json` 中你当前项目的条目。下面的示例展示了从 `/path/to/your/project` 运行时的结果：

```json
{
  "projects": {
    "/path/to/your/project": {
      "mcpServers": {
        "stripe": {
          "type": "http",
          "url": "https://mcp.stripe.com"
        }
      }
    }
  }
}
```

### Project 作用域

Project 作用域的服务器通过将配置存储在项目根目录的 `.mcp.json` 文件中来实现团队协作。此文件设计为检入版本控制，确保所有团队成员都可以访问相同的 MCP 工具和服务。当你添加 project 作用域的服务器时，Claude Code 会自动创建或更新此文件，使用适当的配置结构。

```bash
# 添加 project 作用域的服务器
claude mcp add --transport http paypal --scope project https://mcp.paypal.com/mcp
```

生成的 `.mcp.json` 文件遵循标准化格式：

```json
{
  "mcpServers": {
    "shared-server": {
      "command": "/path/to/server",
      "args": [],
      "env": {}
    }
  }
}
```

出于安全原因，Claude Code 在使用 `.mcp.json` 中的 project 作用域服务器之前会提示批准。如果你需要重置这些批准选择，使用 `claude mcp reset-project-choices` 命令。

### User 作用域

User 作用域的服务器存储在 `~/.claude.json` 中，提供跨项目的可访问性，使它们在你机器上的所有项目中可用，同时对你用户账户保持私有。此作用域适合个人实用服务器、开发工具或你在不同项目中经常使用的服务。

```bash
# 添加 user 服务器
claude mcp add --transport http hubspot --scope user https://mcp.hubspot.com/anthropic
```

### 作用域层级和优先级

当同一服务器在多个位置定义时，Claude Code 只连接一次，使用来自最高优先级源的定义：

1. Local 作用域
2. Project 作用域
3. User 作用域
4. [插件提供的服务器](/en/plugins)
5. [claude.ai 连接器](#使用-claude-ai-的-mcp-服务器)

三个作用域按名称匹配重复项。插件和连接器按端点匹配，因此指向相同 URL 或命令的服务器被视为重复项。

### `.mcp.json` 中的环境变量展开

Claude Code 支持 `.mcp.json` 文件中的环境变量展开，允许团队共享配置，同时保持机器特定路径和 API 密钥等敏感值的灵活性。

**支持的语法：**

- `${VAR}` — 展开为环境变量 `VAR` 的值
- `${VAR:-default}` — 如果 `VAR` 已设置则展开为该值，否则使用 `default`

**展开位置：**
环境变量可以在以下位置展开：

- `command` — 服务器可执行文件路径
- `args` — 命令行参数
- `env` — 传递给服务器的环境变量
- `url` — 对于 HTTP 服务器类型
- `headers` — 对于 HTTP 服务器认证

**带变量展开的示例：**

```json
{
  "mcpServers": {
    "api-server": {
      "type": "http",
      "url": "${API_BASE_URL:-https://api.example.com}/mcp",
      "headers": {
        "Authorization": "Bearer ${API_KEY}"
      }
    }
  }
}
```

如果必需的环境变量未设置且没有默认值，Claude Code 将无法解析配置。

---

## 实际示例

### 示例：使用 Sentry 监控错误

```bash
claude mcp add --transport http sentry https://mcp.sentry.dev/mcp
```

使用你的 Sentry 账户认证：

```text
/mcp
```

然后调试生产问题：

```text
What are the most common errors in the last 24 hours?
```

```text
Show me the stack trace for error ID abc123
```

```text
Which deployment introduced these new errors?
```

### 示例：连接到 GitHub 进行代码审查

```bash
claude mcp add --transport http github https://api.githubcopilot.com/mcp/
```

如果需要，为 GitHub 选择"Authenticate"进行认证：

```text
/mcp
```

然后与 GitHub 交互：

```text
Review PR #456 and suggest improvements
```

```text
Create a new issue for the bug we just found
```

```text
Show me all open PRs assigned to me
```

### 示例：查询你的 PostgreSQL 数据库

```bash
claude mcp add --transport stdio db -- npx -y @bytebase/dbhub \
  --dsn "postgresql://readonly:pass@prod.db.com:5432/analytics"
```

然后自然地查询你的数据库：

```text
What's our total revenue this month?
```

```text
Show me the schema for the orders table
```

```text
Find customers who haven't made a purchase in 90 days
```

---

## 认证远程 MCP 服务器

许多基于云的 MCP 服务器需要认证。Claude Code 支持 OAuth 2.0 以实现安全连接。

**步骤 1：添加需要认证的服务器**

例如：

```bash
claude mcp add --transport http sentry https://mcp.sentry.dev/mcp
```

**步骤 2：在 Claude Code 内使用 /mcp 命令**

在 Claude code 中，使用命令：

```text
/mcp
```

然后按照浏览器中的步骤登录。

> **提示**：
>
> - 认证令牌安全存储并自动刷新
> - 使用 `/mcp` 菜单中的"Clear authentication"撤销访问
> - 如果你的浏览器未自动打开，复制提供的 URL 并手动打开
> - 如果认证后浏览器重定向失败并出现连接错误，将浏览器地址栏中的完整回调 URL 粘贴到 Claude Code 中出现的 URL 提示中
> - OAuth 认证适用于 HTTP 服务器

### 使用固定的 OAuth 回调端口

某些 MCP 服务器需要提前注册特定的重定向 URI。默认情况下，Claude Code 为 OAuth 回调选择一个随机的可用端口。使用 `--callback-port` 固定端口，使其匹配预注册的重定向 URI，格式为 `http://localhost:PORT/callback`。

你可以单独使用 `--callback-port`（带动态客户端注册）或与 `--client-id` 一起使用（带预配置的凭据）。

```bash
# 带动态客户端注册的固定回调端口
claude mcp add --transport http \
  --callback-port 8080 \
  my-server https://mcp.example.com/mcp
```

### 使用预配置的 OAuth 凭据

某些 MCP 服务器不支持通过动态客户端注册自动 OAuth 设置。如果你看到"Incompatible auth server: does not support dynamic client registration"之类的错误，该服务器需要预配置的凭据。Claude Code 还支持使用客户端 ID 元数据文档（CIMD）而不是动态客户端注册的服务器，并自动发现它们。如果自动发现失败，首先通过服务器的开发者门户注册 OAuth 应用，然后在添加服务器时提供凭据。

**步骤 1：在服务器上注册 OAuth 应用**

通过服务器的开发者门户创建应用并记录你的客户端 ID 和客户端密钥。

许多服务器还需要重定向 URI。如果是这样，选择一个端口并以 `http://localhost:PORT/callback` 格式注册重定向 URI。在下一步中使用 `--callback-port` 时使用相同的端口。

**步骤 2：使用你的凭据添加服务器**

选择以下方法之一。`--callback-port` 使用的端口可以是任何可用端口。它只需要与你在上一步中注册的重定向 URI 匹配。

```bash
# 使用 claude mcp add
claude mcp add --transport http \
  --client-id your-client-id --client-secret --callback-port 8080 \
  my-server https://mcp.example.com/mcp

# 使用 claude mcp add-json
claude mcp add-json my-server \
  '{"type":"http","url":"https://mcp.example.com/mcp","oauth":{"clientId":"your-client-id","callbackPort":8080}}' \
  --client-secret

# 仅回调端口（动态客户端注册）
claude mcp add-json my-server \
  '{"type":"http","url":"https://mcp.example.com/mcp","oauth":{"callbackPort":8080}}'

# CI / 环境变量
MCP_CLIENT_SECRET=your-secret claude mcp add --transport http \
  --client-id your-client-id --client-secret --callback-port 8080 \
  my-server https://mcp.example.com/mcp
```

**步骤 3：在 Claude Code 中认证**

在 Claude Code 中运行 `/mcp` 并按照浏览器登录流程操作。

> **提示**：
>
> - 客户端密钥安全存储在你的系统钥匙串（macOS）或凭据文件中，而不是配置中
> - 如果服务器使用没有密钥的公共 OAuth 客户端，仅使用 `--client-id` 而不使用 `--client-secret`
> - `--callback-port` 可以与或没有 `--client-id` 一起使用
> - 这些标志仅适用于 HTTP 和 SSE 传输。它们对 stdio 服务器无效
> - 使用 `claude mcp get <name>` 验证服务器的 OAuth 凭据是否已配置

### 覆盖 OAuth 元数据发现

如果你的 MCP 服务器的标准 OAuth元数据端点返回错误，但服务器暴露了可用的 OIDC 端点，你可以将 Claude Code 指向特定的元数据 URL 以绕过默认发现链。默认情况下，Claude Code 首先检查 RFC 9728 受保护资源元数据（位于 `/.well-known/oauth-protected-resource`），然后回退到 RFC 8414 授权服务器元数据（位于 `/.well-known/oauth-authorization-server`）。

在 `.mcp.json` 中服务器配置的 `oauth` 对象中设置 `authServerMetadataUrl`：

```json
{
  "mcpServers": {
    "my-server": {
      "type": "http",
      "url": "https://mcp.example.com/mcp",
      "oauth": {
        "authServerMetadataUrl": "https://auth.example.com/.well-known/openid-configuration"
      }
    }
  }
}
```

URL 必须使用 `https://`。此选项需要 Claude Code v2.1.64 或更高版本。

### 使用动态头部进行自定义认证

如果你的 MCP 服务器使用 OAuth 以外的认证方案（如 Kerberos、短期令牌或内部 SSO），使用 `headersHelper` 在连接时生成请求头部。Claude Code 运行命令并将其输出合并到连接头部。

```json
{
  "mcpServers": {
    "internal-api": {
      "type": "http",
      "url": "https://mcp.internal.example.com",
      "headersHelper": "/opt/bin/get-mcp-auth-headers.sh"
    }
  }
}
```

命令也可以内联：

```json
{
  "mcpServers": {
    "internal-api": {
      "type": "http",
      "url": "https://mcp.internal.example.com",
      "headersHelper": "echo '{\"Authorization\": \"Bearer '\"$(get-token)\"'\"}'"
    }
  }
}
```

**要求：**

- 命令必须将字符串键值对的 JSON 对象写入 stdout
- 命令在 shell 中运行，超时时间为 10 秒
- 动态头部会覆盖同名的任何静态 `headers`

辅助脚本在每次连接时全新运行（会话启动和重新连接时）。没有缓存，因此你的脚本负责任何令牌重用。

Claude Code 在执行辅助脚本时设置这些环境变量：

| 变量                          | 值              |
| :---------------------------- | :-------------- |
| `CLAUDE_CODE_MCP_SERVER_NAME` | MCP 服务器的名称 |
| `CLAUDE_CODE_MCP_SERVER_URL`  | MCP 服务器的 URL |

使用这些可以编写服务于多个 MCP 服务器的单个辅助脚本。

> **注意**：`headersHelper` 执行任意的 shell 命令。当在 project 或 local 作用域定义时，它仅在你接受工作区信任对话框后运行。

---

## 从 JSON 配置添加 MCP 服务器

如果你有 MCP 服务器的 JSON 配置，可以直接添加：

**步骤 1：从 JSON 添加 MCP 服务器**

```bash
# 基本语法
claude mcp add-json <名称> '<json>'

# 示例：添加带 JSON 配置的 HTTP 服务器
claude mcp add-json weather-api '{"type":"http","url":"https://api.weather.com/mcp","headers":{"Authorization":"Bearer token"}}'

# 示例：添加带 JSON 配置的 stdio 服务器
claude mcp add-json local-weather '{"type":"stdio","command":"/path/to/weather-cli","args":["--api-key","abc123"],"env":{"CACHE_DIR":"/tmp"}}'

# 示例：添加带预配置 OAuth 凭据的 HTTP 服务器
claude mcp add-json my-server '{"type":"http","url":"https://mcp.example.com/mcp","oauth":{"clientId":"your-client-id","callbackPort":8080}}' --client-secret
```

**步骤 2：验证服务器已添加**

```bash
claude mcp get weather-api
```

> **提示**：
>
> - 确保 JSON 在你的 shell 中正确转义
> - JSON 必须符合 MCP 服务器配置 schema
> - 你可以使用 `--scope user` 将服务器添加到用户配置而不是特定于项目的配置

---

## 从 Claude Desktop 导入 MCP 服务器

如果你已经在 Claude Desktop 中配置了 MCP 服务器，可以导入它们：

**步骤 1：从 Claude Desktop 导入服务器**

```bash
claude mcp add-from-claude-desktop
```

**步骤 2：选择要导入的服务器**

运行命令后，你会看到一个交互式对话框，允许你选择要导入的服务器。

**步骤 3：验证服务器已导入**

```bash
claude mcp list
```

> **提示**：
>
> - 此功能仅适用于 macOS 和 Windows Subsystem for Linux (WSL)
> - 它从这些平台上的标准位置读取 Claude Desktop 配置文件
> - 使用 `--scope user` 标志将服务器添加到用户配置
> - 导入的服务器将与 Claude Desktop 中的名称相同
> - 如果已存在同名服务器，它们将获得数字后缀（例如 `server_1`）

---

## 使用 Claude.ai 的 MCP 服务器

如果你已使用 [Claude.ai](https://claude.ai) 账户登录 Claude Code，你在 Claude.ai 中添加的 MCP 服务器在 Claude Code 中自动可用：

**步骤 1：在 Claude.ai 中配置 MCP 服务器**

在 [claude.ai/settings/connectors](https://claude.ai/settings/connectors) 添加服务器。在 Team 和 Enterprise 计划中，只有管理员可以添加服务器。

**步骤 2：认证 MCP 服务器**

在 Claude.ai 中完成任何必需的认证步骤。

**步骤 3：在 Claude Code 中查看和管理服务器**

在 Claude Code 中，使用命令：

```text
/mcp
```

Claude.ai 服务器在列表中显示，带有来自 Claude.ai 的指示器。

要在 Claude Code 中禁用 claude.ai MCP 服务器，设置 `ENABLE_CLAUDEAI_MCP_SERVERS` 环境变量为 `false`：

```bash
ENABLE_CLAUDEAI_MCP_SERVERS=false claude
```

---

## 将 Claude Code 用作 MCP 服务器

你可以将 Claude Code 本身用作其他应用程序可以连接的 MCP 服务器：

```bash
# 将 Claude 作为 stdio MCP 服务器启动
claude mcp serve
```

你可以在 Claude Desktop 中使用此功能，将此配置添加到 claude_desktop_config.json：

```json
{
  "mcpServers": {
    "claude-code": {
      "type": "stdio",
      "command": "claude",
      "args": ["mcp", "serve"],
      "env": {}
    }
  }
}
```

> **警告**：**配置可执行文件路径**：`command` 字段必须引用 Claude Code 可执行文件。如果 `claude` 命令不在系统 PATH 中，你需要指定可执行文件的完整路径。
>
> 要查找完整路径：
>
> ```bash
> which claude
> ```
>
> 然后在配置中使用完整路径：
>
> ```json
> {
>   "mcpServers": {
>     "claude-code": {
>       "type": "stdio",
>       "command": "/full/path/to/claude",
>       "args": ["mcp", "serve"],
>       "env": {}
>     }
>   }
> }
> ```
>
> 没有正确的可执行文件路径，你会遇到 `spawn claude ENOENT` 之类的错误。

> **提示**：
>
> - 服务器提供对 Claude 的工具的访问，如 View、Edit、LS 等
> - 在 Claude Desktop 中，尝试要求 Claude 读取目录中的文件、进行编辑等
> - 注意此 MCP 服务器仅将 Claude Code 的工具暴露给你的 MCP 客户端，因此你自己的客户端负责为单个工具调用实现用户确认

---

## MCP 输出限制和警告

当 MCP 工具产生大量输出时，Claude Code 帮助管理 token 用量以防止压垮你的对话上下文：

- **输出警告阈值**：当任何 MCP 工具输出超过 10,000 token 时，Claude Code 显示警告
- **可配置限制**：你可以使用 `MAX_MCP_OUTPUT_TOKENS` 环境变量调整允许的最大 MCP 输出 token
- **默认限制**：默认最大值为 25,000 token
- **作用域**：环境变量适用于未声明自己限制的工具。设置 [`anthropic/maxResultSizeChars`](#提高特定工具的限制) 的工具使用该值代替文本内容，无论 `MAX_MCP_OUTPUT_TOKENS` 设置为什么。返回图像数据的工具仍然受 `MAX_MCP_OUTPUT_TOKENS` 限制

要为产生大量输出的工具增加限制：

```bash
export MAX_MCP_OUTPUT_TOKENS=50000
claude
```

这在使用以下 MCP 服务器时特别有用：

- 查询大型数据集或数据库
- 生成详细的报告或文档
- 处理广泛的日志文件或调试信息

### 提高特定工具的限制

如果你正在构建 MCP 服务器，你可以通过在工具的 `tools/list` 响应条目中设置 `_meta["anthropic/maxResultSizeChars"]` 来允许单个工具返回大于默认持久化到磁盘阈值的结果。Claude Code 将该工具的阈值提高到注释值，硬上限为 500,000 字符。

这对于返回固有地大但必要的输出的工具很有用，如数据库 schema 或完整文件树。没有注释时，超过默认阈值的结果会持久化到磁盘并在对话中替换为文件引用。

```json
{
  "name": "get_schema",
  "description": "Returns the full database schema",
  "_meta": {
    "anthropic/maxResultSizeChars": 200000
  }
}
```

该注释独立于 `MAX_MCP_OUTPUT_TOKENS` 适用于文本内容，因此用户无需为声明它的工具提高环境变量。返回图像数据的工具仍然受 token 限制。

> **警告**：如果你经常遇到不受你控制的特定 MCP 服务器的输出警告，考虑增加 `MAX_MCP_OUTPUT_TOKENS` 限制。你也可以要求服务器作者添加 `anthropic/maxResultSizeChars` 注释或对他们的响应进行分页。该注释对返回图像内容的工具无效；对于这些工具，提高 `MAX_MCP_OUTPUT_TOKENS` 是唯一的选择。

---

## 响应 MCP 诱导请求

MCP 服务器可以在任务中途使用诱导请求你的结构化输入。当服务器需要它无法自行获取的信息时，Claude Code 显示交互式对话框并将你的响应传递回服务器。你这边无需配置：当服务器请求时，诱导对话框自动出现。

服务器可以通过两种方式请求输入：

- **表单模式**：Claude Code 显示一个带服务器定义的表单字段的对话框（例如用户名和密码提示）。填写字段并提交。
- **URL 模式**：Claude Code 打开浏览器 URL 进行认证或批准。在浏览器中完成流程，然后在 CLI 中确认。

要在不显示对话框的情况下自动响应诱导请求，使用 [`Elicitation` hook](/en/hooks#elicitation)。

如果你正在构建使用诱导的 MCP 服务器，见 [MCP 诱导规范](https://modelcontextprotocol.io/docs/learn/client-concepts#elicitation)了解协议详情和 schema 示例。

---

## 使用 MCP 资源

MCP 服务器可以暴露资源，你可以使用 @ 提及引用这些资源，类似于引用文件的方式。

### 引用 MCP 资源

**步骤 1：列出可用资源**

在提示中输入 `@` 查看所有已连接 MCP 服务器的可用资源。资源与文件一起出现在自动补全菜单中。

**步骤 2：引用特定资源**

使用格式 `@server:protocol://resource/path` 引用资源：

```text
Can you analyze @github:issue://123 and suggest a fix?
```

```text
Please review the API documentation at @docs:file://api/authentication
```

**步骤 3：多个资源引用**

你可以在单个提示中引用多个资源：

```text
Compare @postgres:schema://users with @docs:file://database/user-model
```

> **提示**：
>
> - 引用时资源自动获取并作为附件包含
> - 资源路径在 @ 提及自动补全中支持模糊搜索
> - 当服务器支持时，Claude Code 自动提供列出和读取 MCP 资源的工具
> - 资源可以包含 MCP 服务器提供的任何类型的内容（文本、JSON、结构化数据等）

---

## 使用 MCP 工具搜索扩展规模

工具搜索通过延迟工具定义直到 Claude 需要它们来保持 MCP 上下文用量较低。只有工具名称在会话启动时加载，因此添加更多 MCP 服务器对你的上下文窗口影响最小。

### 工作原理

工具搜索默认启用。MCP 工具被延迟而不是预先加载到上下文中，Claude 使用搜索工具在任务需要时发现相关工具。只有 Claude 实际使用的工具进入上下文。从你的角度来看，MCP 工具的工作方式与以前完全相同。

如果你偏好基于阈值的加载，设置 `ENABLE_TOOL_SEARCH=auto` 在适应 10% 上下文窗口时预先加载 schema，仅延迟溢出部分。见[配置工具搜索](#配置工具搜索)了解所有选项。

### 对于 MCP 服务器作者

如果你正在构建 MCP 服务器，服务器指令字段在启用工具搜索时变得更有用。服务器指令帮助 Claude 理解何时搜索你的工具，类似于[技能](/en/skills)的工作方式。

添加清晰、描述性的服务器指令来解释：

- 你的工具处理的任务类别
- Claude 何时应该搜索你的工具
- 你的服务器提供的关键功能

Claude Code 将工具描述和服务器指令分别截断为 2KB。保持简洁以避免截断，并将关键细节放在开头。

### 配置工具搜索

工具搜索默认启用：MCP 工具被延迟并按需发现。当 `ANTHROPIC_BASE_URL` 指向非第一方主机时，工具搜索默认禁用，因为大多数代理不转发 `tool_reference` 块。如果你的代理不转发，显式设置 `ENABLE_TOOL_SEARCH`。此功能需要支持 `tool_reference` 块的模型：Sonnet 4 及更高版本，或 Opus 4 及更高版本。Haiku 模型不支持工具搜索。

使用 `ENABLE_TOOL_SEARCH` 环境变量控制工具搜索行为：

| 值         | 行为                                                                                                                    |
| :--------- | :---------------------------------------------------------------------------------------------------------------------- |
| （未设置） | 所有 MCP 工具延迟并按需加载。当 `ANTHROPIC_BASE_URL` 是非第一方主机时回退到预加载                                     |
| `true`     | 所有 MCP 工具延迟，包括非第一方 `ANTHROPIC_BASE_URL`                                                                   |
| `auto`     | 阈值模式：如果工具适应 10% 上下文窗口则预加载，否则延迟                                                                 |
| `auto:<N>` | 带自定义百分比的阈值模式，其中 `<N>` 为 0-100（例如 `auto:5` 表示 5%）                                                   |
| `false`    | 所有 MCP 工具预加载，不延迟                                                                                             |

```bash
# 使用自定义 5% 阈值
ENABLE_TOOL_SEARCH=auto:5 claude

# 完全禁用工具搜索
ENABLE_TOOL_SEARCH=false claude
```

或在你的 [settings.json `env` 字段](/en/settings#available-settings)中设置值。

你也可以禁用 `ToolSearch` 工具：

```json
{
  "permissions": {
    "deny": ["ToolSearch"]
  }
}
```

---

## 将 MCP 提示用作命令

MCP 服务器可以暴露提示，这些提示在 Claude Code 中作为命令可用。

### 执行 MCP 提示

**步骤 1：发现可用提示**

输入 `/` 查看所有可用命令，包括来自 MCP 服务器的命令。MCP 提示以 `/mcp__servername__promptname` 格式出现。

**步骤 2：执行不带参数的提示**

```text
/mcp__github__list_prs
```

**步骤 3：执行带参数的提示**

许多提示接受参数。在命令后以空格分隔传递：

```text
/mcp__github__pr_review 456
```

```text
/mcp__jira__create_issue "Bug in login flow" high
```

> **提示**：
>
> - MCP 提示从已连接服务器动态发现
> - 参数根据提示定义的参数解析
> - 提示结果直接注入对话
> - 服务器和提示名称被规范化（空格变为下划线）

---

## Managed MCP 配置

对于需要集中控制 MCP 服务器的组织，Claude Code 支持两种配置选项：

1. **使用 `managed-mcp.json` 的独占控制**：部署一组固定的 MCP 服务器，用户无法修改或扩展
2. **使用白名单/黑名单的基于策略的控制**：允许用户添加自己的服务器，但限制哪些服务器被允许

这些选项允许 IT 管理员：

- **控制员工可以访问哪些 MCP 服务器**：在整个组织中部署一组标准化的已批准 MCP 服务器
- **防止未经授权的 MCP 服务器**：限制用户添加未经批准的 MCP 服务器
- **完全禁用 MCP**：如果需要，移除 MCP 功能

### 选项 1：使用 managed-mcp.json 的独占控制

当你部署 `managed-mcp.json` 文件时，它对所有 MCP 服务器取得**独占控制**。用户无法添加、修改或使用此文件之外的任何 MCP 服务器。这是想要完全控制的组织的最简单方法。

系统管理员将配置文件部署到系统范围的目录：

- macOS: `/Library/Application Support/ClaudeCode/managed-mcp.json`
- Linux 和 WSL: `/etc/claude-code/managed-mcp.json`
- Windows: `C:\Program Files\ClaudeCode\managed-mcp.json`

> **注意**：这些是系统范围路径（不是用户主目录如 `~/Library/...`），需要管理员权限。它们设计为由 IT 管理员部署。

`managed-mcp.json` 文件使用与标准 `.mcp.json` 文件相同的格式：

```json
{
  "mcpServers": {
    "github": {
      "type": "http",
      "url": "https://api.githubcopilot.com/mcp/"
    },
    "sentry": {
      "type": "http",
      "url": "https://mcp.sentry.dev/mcp"
    },
    "company-internal": {
      "type": "stdio",
      "command": "/usr/local/bin/company-mcp-server",
      "args": ["--config", "/etc/company/mcp-config.json"],
      "env": {
        "COMPANY_API_URL": "https://internal.company.com"
      }
    }
  }
}
```

### 选项 2：使用白名单和黑名单的基于策略的控制

管理员可以允许用户配置自己的 MCP 服务器，同时强制执行哪些服务器被允许的限制，而不是独占控制。这在 [managed 设置文件](/en/settings#settings-files)中使用 `allowedMcpServers` 和 `deniedMcpServers`。

> **注意**：**在选项之间选择**：当你想要部署一组固定的服务器且不允许用户自定义时使用选项 1（`managed-mcp.json`）。当你想要允许用户在策略约束内添加自己的服务器时使用选项 2（白名单/黑名单）。

#### 限制选项

白名单或黑名单中的每个条目可以通过三种方式限制服务器：

1. **按服务器名称**（`serverName`）：匹配服务器的配置名称
2. **按命令**（`serverCommand`）：匹配用于启动 stdio 服务器的确切命令和参数
3. **按 URL 模式**（`serverUrl`）：匹配远程服务器 URL，支持通配符

**重要**：每个条目必须恰好有 `serverName`、`serverCommand` 或 `serverUrl` 中的一个。

#### 示例配置

```json
{
  "allowedMcpServers": [
    // 按服务器名称允许
    { "serverName": "github" },
    { "serverName": "sentry" },

    // 按确切命令允许（对于 stdio 服务器）
    { "serverCommand": ["npx", "-y", "@modelcontextprotocol/server-filesystem"] },
    { "serverCommand": ["python", "/usr/local/bin/approved-server.py"] },

    // 按 URL 模式允许（对于远程服务器）
    { "serverUrl": "https://mcp.company.com/*" },
    { "serverUrl": "https://*.internal.corp/*" }
  ],
  "deniedMcpServers": [
    // 按服务器名称阻止
    { "serverName": "dangerous-server" },

    // 按确切命令阻止（对于 stdio 服务器）
    { "serverCommand": ["npx", "-y", "unapproved-package"] },

    // 按 URL 模式阻止（对于远程服务器）
    { "serverUrl": "https://*.untrusted.com/*" }
  ]
}
```

#### 基于命令的限制如何工作

**精确匹配**：

- 命令数组必须**完全**匹配 — 命令和所有参数按正确顺序
- 示例：`["npx", "-y", "server"]` 不会匹配 `["npx", "server"]` 或 `["npx", "-y", "server", "--flag"]`

**Stdio 服务器行为**：

- 当白名单包含**任何** `serverCommand` 条目时，stdio 服务器**必须**匹配其中一个命令
- 当存在命令限制时，stdio 服务器不能仅按名称通过
- 这确保管理员可以强制执行允许运行的命令

**非 stdio 服务器行为**：

- 远程服务器（HTTP、SSE、WebSocket）在白名单中存在 `serverUrl` 条目时使用基于 URL 的匹配
- 如果不存在 URL 条目，远程服务器回退到基于名称的匹配
- 命令限制不适用于远程服务器

#### 基于 URL 的限制如何工作

URL 模式支持使用 `*` 的通配符来匹配任何字符序列。这对于允许整个域或子域很有用。

**通配符示例**：

- `https://mcp.company.com/*` — 允许特定域上的所有路径
- `https://*.example.com/*` — 允许 example.com 的任何子域
- `http://localhost:*/*` — 允许 localhost 上的任何端口

**远程服务器行为**：

- 当白名单包含**任何** `serverUrl` 条目时，远程服务器**必须**匹配其中一个 URL 模式
- 当存在 URL 限制时，远程服务器不能仅按名称通过
- 这确保管理员可以强制执行允许的远程端点

<details>
<summary>示例：仅 URL 白名单</summary>

```json
{
  "allowedMcpServers": [
    { "serverUrl": "https://mcp.company.com/*" },
    { "serverUrl": "https://*.internal.corp/*" }
  ]
}
```

**结果**：

- `https://mcp.company.com/api` 的 HTTP 服务器：✅ 允许（匹配 URL 模式）
- `https://api.internal.corp/mcp` 的 HTTP 服务器：✅ 允许（匹配通配符子域）
- `https://external.com/mcp` 的 HTTP 服务器：❌ 阻止（不匹配任何 URL 模式）
- 任何命令的 Stdio 服务器：❌ 阻止（没有名称或命令条目匹配）
</details>

<details>
<summary>示例：仅命令白名单</summary>

```json
{
  "allowedMcpServers": [
    { "serverCommand": ["npx", "-y", "approved-package"] }
  ]
}
```

**结果**：

- 使用 `["npx", "-y", "approved-package"]` 的 Stdio 服务器：✅ 允许（匹配命令）
- 使用 `["node", "server.js"]` 的 Stdio 服务器：❌ 阻止（不匹配命令）
- 名为 "my-api" 的 HTTP 服务器：❌ 阻止（没有名称条目匹配）
</details>

<details>
<summary>示例：混合名称和命令白名单</summary>

```json
{
  "allowedMcpServers": [
    { "serverName": "github" },
    { "serverCommand": ["npx", "-y", "approved-package"] }
  ]
}
```

**结果**：

- 使用 `["npx", "-y", "approved-package"]` 名为 "local-tool" 的 Stdio 服务器：✅ 允许（匹配命令）
- 使用 `["node", "server.js"]` 名为 "local-tool" 的 Stdio 服务器：❌ 阻止（存在命令条目但不匹配）
- 使用 `["node", "server.js"]` 名为 "github" 的 Stdio 服务器：❌ 阻止（当存在命令条目时 stdio 服务器必须匹配命令）
- 名为 "github" 的 HTTP 服务器：✅ 允许（匹配名称）
- 名为 "other-api" 的 HTTP 服务器：❌ 阻止（名称不匹配）
</details>

<details>
<summary>示例：仅名称白名单</summary>

```json
{
  "allowedMcpServers": [
    { "serverName": "github" },
    { "serverName": "internal-tool" }
  ]
}
```

**结果**：

- 任何命令名为 "github" 的 Stdio 服务器：✅ 允许（无命令限制）
- 任何命令名为 "internal-tool" 的 Stdio 服务器：✅ 允许（无命令限制）
- 名为 "github" 的 HTTP 服务器：✅ 允许（匹配名称）
- 任何名为 "other" 的服务器：❌ 阻止（名称不匹配）
</details>

#### 白名单行为（`allowedMcpServers`）

- `undefined`（默认）：无限制 — 用户可以配置任何 MCP 服务器
- 空数组 `[]`：完全锁定 — 用户无法配置任何 MCP 服务器
- 条目列表：用户只能配置按名称、命令或 URL 模式匹配的服务器

#### 黑名单行为（`deniedMcpServers`）

- `undefined`（默认）：没有服务器被阻止
- 空数组 `[]`：没有服务器被阻止
- 条目列表：指定的服务器在所有作用域中被显式阻止

#### 重要事项

- **选项 1 和选项 2 可以组合**：如果 `managed-mcp.json` 存在，它具有独占控制，用户无法添加服务器。白名单/黑名单仍然适用于 managed 服务器本身的过滤。
- **黑名单具有绝对优先权**：如果服务器匹配黑名单条目（按名称、命令或 URL），即使在白名单上也会被阻止
- 基于名称、命令和 URL 的限制共同工作：服务器如果匹配名称条目、命令条目或 URL 模式中的**任何一个**则通过（除非被黑名单阻止）

> **注意**：**使用 `managed-mcp.json` 时**：用户无法通过 `claude mcp add` 或配置文件添加 MCP 服务器。`allowedMcpServers` 和 `deniedMcpServers` 设置仍然适用以过滤哪些 managed 服务器实际被加载。

---

## 提交反馈

如果你发现本文档中有不正确、过时或令人困惑的内容，请通过 POST 提交反馈到：
https://code.claude.com/docs/_mintlify/feedback/claude-code/agent-feedback

请求体（JSON）：`{ "path": "/current-page-path", "feedback": "问题描述" }`

仅在你有具体且可操作的报告时提交反馈。
