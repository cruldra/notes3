添加本地和远程 MCP 工具

您可以使用 *Model Context Protocol* (MCP) 向 OpenCode 添加外部工具。OpenCode 支持本地和远程服务器。

添加后，MCP 工具将自动与内置工具一起可供 LLM 使用。

---

#### [注意事项](#caveats)

当您使用 MCP 服务器时，它会增加上下文。如果您有很多工具，这很快就会累积起来。因此，我们建议谨慎选择使用的 MCP 服务器。

提示

MCP 服务器会增加您的上下文，因此您需要谨慎选择启用的服务器。

某些 MCP 服务器，如 GitHub MCP 服务器，往往会添加大量 token，很容易超出上下文限制。

---

## [启用](#enable)

您可以在 [OpenCode 配置](https://opencode.ai/docs/config/) 的 `mcp` 下定义 MCP 服务器。为每个 MCP 添加一个唯一的名称。您可以在提示 LLM 时通过名称引用该 MCP。

opencode.jsonc

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "name-of-mcp-server": {
      // ...
      "enabled": true,
    },
    "name-of-other-mcp-server": {
      // ...
    },
  },
}
```

您还可以通过将 `enabled` 设置为 `false` 来禁用服务器。如果您想在不从配置中删除服务器的情况下暂时禁用它，这很有用。

---

### [覆盖远程默认值](#overriding-remote-defaults)

组织可以通过其 `.well-known/opencode` 端点提供默认的 MCP 服务器。这些服务器默认可能被禁用，允许用户选择他们需要的服务器。

要从您的组织的远程配置中启用特定服务器，请将其添加到您的本地配置中并设置 `enabled: true`：

opencode.json

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "jira": {
      "type": "remote",
      "url": "https://jira.example.com/mcp",
      "enabled": true
    }
  }
}
```

您的本地配置值会覆盖远程默认值。

---

## [本地](#local)

在 MCP 对象中将 `type` 设置为 `"local"` 以添加本地 MCP 服务器。

opencode.jsonc

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "my-local-mcp-server": {
      "type": "local",
      // 或者 ["bun", "x", "my-mcp-command"]
      "command": ["npx", "-y", "my-mcp-command"],
      "enabled": true,
      "environment": {
        "MY_ENV_VAR": "my_env_var_value",
      },
    },
  },
}
```

Command 是启动本地 MCP 服务器的方式。您还可以传入环境变量列表。

例如，以下是如何添加测试 [`@modelcontextprotocol/server-everything`](https://www.npmjs.com/package/%40modelcontextprotocol/server-everything) MCP 服务器。

opencode.jsonc

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "mcp_everything": {
      "type": "local",
      "command": ["npx", "-y", "@modelcontextprotocol/server-everything"],
    },
  },
}
```

要使用它，我可以在提示词中添加 `use the mcp_everything tool`。

```text
use the mcp_everything tool to add the number 3 and 4
```

---

#### [选项](#options)

以下是配置本地 MCP 服务器的所有选项。

| 选项 | 类型 | 必填 | 描述 |
| --- | --- | --- | --- |
| `type` | String | 是 | MCP 服务器连接类型，必须为 `"local"`。 |
| `command` | Array | 是 | 运行 MCP 服务器的命令和参数。 |
| `environment` | Object |  | 运行服务器时要设置的环境变量。 |
| `enabled` | Boolean |  | 启动时启用或禁用 MCP 服务器。 |
| `timeout` | Number |  | 从 MCP 服务器获取工具的超时时间（毫秒）。默认为 5000（5秒）。 |

---

## [远程](#remote)

通过将 `type` 设置为 `"remote"` 来添加远程 MCP 服务器。

opencode.json

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "my-remote-mcp": {
      "type": "remote",
      "url": "https://my-mcp-server.com",
      "enabled": true,
      "headers": {
        "Authorization": "Bearer MY_API_KEY"
      }
    }
  }
}
```

`url` 是远程 MCP 服务器的 URL，使用 `headers` 选项可以传入标头列表。

---

#### [选项](#options-1)

| 选项 | 类型 | 必填 | 描述 |
| --- | --- | --- | --- |
| `type` | String | 是 | MCP 服务器连接类型，必须为 `"remote"`。 |
| `url` | String | 是 | 远程 MCP 服务器的 URL。 |
| `enabled` | Boolean |  | 启动时启用或禁用 MCP 服务器。 |
| `headers` | Object |  | 随请求发送的标头。 |
| `oauth` | Object |  | OAuth 认证配置。见下文 [OAuth](#oauth) 部分。 |
| `timeout` | Number |  | 从 MCP 服务器获取工具的超时时间（毫秒）。默认为 5000（5秒）。 |

---

## [OAuth (认证)](#oauth)

OpenCode 自动处理远程 MCP 服务器的 OAuth 认证。当服务器需要认证时，OpenCode 将：

1.  检测 401 响应并启动 OAuth 流程
2.  如果服务器支持，使用 **动态客户端注册 (RFC 7591)**
3.  安全存储令牌以供将来请求使用

---

### [自动](#automatic)

对于大多数启用了 OAuth 的 MCP 服务器，无需特殊配置。只需配置远程服务器：

opencode.json

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "my-oauth-server": {
      "type": "remote",
      "url": "https://mcp.example.com/mcp"
    }
  }
}
```

如果服务器需要认证，OpenCode 将在您首次尝试使用它时提示您进行认证。如果没有，您可以使用 `opencode mcp auth <server-name>` [手动触发流程](#authenticating)。

---

### [预注册](#pre-registered)

如果您有来自 MCP 服务器提供商的客户端凭据，您可以配置它们：

opencode.json

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "my-oauth-server": {
      "type": "remote",
      "url": "https://mcp.example.com/mcp",
      "oauth": {
        "clientId": "{env:MY_MCP_CLIENT_ID}",
        "clientSecret": "{env:MY_MCP_CLIENT_SECRET}",
        "scope": "tools:read tools:execute"
      }
    }
  }
}
```

---

### [认证](#authenticating)

您可以手动触发认证或管理凭据。

使用特定 MCP 服务器进行认证：

Terminal window

```bash
opencode mcp auth my-oauth-server
```

列出所有 MCP 服务器及其认证状态：

Terminal window

```bash
opencode mcp list
```

删除存储的凭据：

Terminal window

```bash
opencode mcp logout my-oauth-server
```

`mcp auth` 命令将打开浏览器进行授权。授权后，OpenCode 将令牌安全地存储在 `~/.local/share/opencode/mcp-auth.json` 中。

---

#### [禁用 OAuth](#disabling-oauth)

如果您想为服务器禁用自动 OAuth（例如，对于使用 API 密钥的服务器），请将 `oauth` 设置为 `false`：

opencode.json

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "my-api-key-server": {
      "type": "remote",
      "url": "https://mcp.example.com/mcp",
      "oauth": false,
      "headers": {
        "Authorization": "Bearer {env:MY_API_KEY}"
      }
    }
  }
}
```

---

#### [OAuth 选项](#oauth-options)

| 选项 | 类型 | 描述 |
| --- | --- | --- |
| `oauth` | Object | false | OAuth 配置对象，或 `false` 以禁用 OAuth 自动检测。 |
| `clientId` | String | OAuth 客户端 ID。如果未提供，将尝试动态客户端注册。 |
| `clientSecret` | String | OAuth 客户端密钥，如果授权服务器需要。 |
| `scope` | String | 授权期间请求的 OAuth 范围。 |

#### [调试](#debugging)

如果远程 MCP 服务器认证失败，您可以使用以下命令诊断问题：

Terminal window

```bash
# 查看所有支持 OAuth 的服务器的认证状态
opencode mcp auth list

# 调试特定服务器的连接和 OAuth 流程
opencode mcp debug my-oauth-server
```

`mcp debug` 命令显示当前认证状态，测试 HTTP 连接，并尝试 OAuth 发现流程。

---

## [管理](#manage)

您的 MCP 作为 OpenCode 中的工具可用，与内置工具并列。因此，您可以像任何其他工具一样通过 OpenCode 配置来管理它们。

---

### [全局](#global)

这意味着您可以全局启用或禁用它们。

opencode.json

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "my-mcp-foo": {
      "type": "local",
      "command": ["bun", "x", "my-mcp-command-foo"]
    },
    "my-mcp-bar": {
      "type": "local",
      "command": ["bun", "x", "my-mcp-command-bar"]
    }
  },
  "tools": {
    "my-mcp-foo": false
  }
}
```

我们也可以使用 glob 模式来禁用所有匹配的 MCP。

opencode.json

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "my-mcp-foo": {
      "type": "local",
      "command": ["bun", "x", "my-mcp-command-foo"]
    },
    "my-mcp-bar": {
      "type": "local",
      "command": ["bun", "x", "my-mcp-command-bar"]
    }
  },
  "tools": {
    "my-mcp*": false
  }
}
```

这里我们使用 glob 模式 `my-mcp*` 来禁用所有 MCP。

---

### [每个智能体](#per-agent)

如果您有大量的 MCP 服务器，您可能希望仅针对每个智能体启用它们，而在全局范围内禁用它们。为此：

1.  作为工具全局禁用它。
2.  在您的智能体配置中，启用 MCP 服务器作为工具。

opencode.json

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "my-mcp": {
      "type": "local",
      "command": ["bun", "x", "my-mcp-command"],
      "enabled": true
    }
  },
  "tools": {
    "my-mcp*": false
  },
  "agent": {
    "my-agent": {
      "tools": {
        "my-mcp*": true
      }
    }
  }
}
```

---

#### [Glob 模式](#glob-patterns)

Glob 模式使用简单的正则表达式 glob 模式：

*   `*` 匹配零个或多个任意字符（例如，`"my-mcp*"` 匹配 `my-mcp_search`、`my-mcp_list` 等）
*   `?` 匹配正好一个字符
*   所有其他字符按字面匹配

注意

MCP 服务器工具使用服务器名称作为前缀注册，因此要禁用服务器的所有工具，只需使用：

```json
"mymcpservername_*": false
```

---

## [示例](#examples)

以下是一些常见 MCP 服务器的示例。如果您想记录其他服务器，可以提交 PR。

---

### [Sentry](#sentry)

添加 [Sentry MCP 服务器](https://mcp.sentry.dev) 以与您的 Sentry 项目和问题进行交互。

opencode.json

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "sentry": {
      "type": "remote",
      "url": "https://mcp.sentry.dev/mcp",
      "oauth": {}
    }
  }
}
```

添加配置后，与 Sentry 进行认证：

Terminal window

```bash
opencode mcp auth sentry
```

这将打开一个浏览器窗口以完成 OAuth 流程并将 OpenCode 连接到您的 Sentry 帐户。

认证后，您可以在提示词中使用 Sentry 工具来查询问题、项目和错误数据。

```text
Show me the latest unresolved issues in my project. use sentry
```

---

### [Context7](#context7)

添加 [Context7 MCP 服务器](https://github.com/upstash/context7) 以搜索文档。

opencode.json

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "context7": {
      "type": "remote",
      "url": "https://mcp.context7.com/mcp"
    }
  }
}
```

如果您已注册免费帐户，可以使用您的 API 密钥并获得更高的速率限制。

opencode.json

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "context7": {
      "type": "remote",
      "url": "https://mcp.context7.com/mcp",
      "headers": {
        "CONTEXT7_API_KEY": "{env:CONTEXT7_API_KEY}"
      }
    }
  }
}
```

这里我们假设您已设置 `CONTEXT7_API_KEY` 环境变量。

在您的提示词中添加 `use context7` 以使用 Context7 MCP 服务器。

```text
Configure a Cloudflare Worker script to cache JSON API responses for five minutes. use context7
```

或者，您可以添加如下内容到您的 AGENTS.md。

AGENTS.md

```markdown
When you need to search docs, use `context7` tools.
```

---

### [Grep by Vercel](#grep-by-vercel)

添加 [Grep by Vercel](https://grep.app) MCP 服务器以搜索 GitHub 上的代码片段。

opencode.json

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "gh_grep": {
      "type": "remote",
      "url": "https://mcp.grep.app"
    }
  }
}
```

由于我们将 MCP 服务器命名为 `gh_grep`，您可以在提示词中添加 `use the gh_grep tool` 以让智能体使用它。

```text
What's the right way to set a custom domain in an SST Astro component? use the gh_grep tool
```

或者，您可以添加如下内容到您的 AGENTS.md。

AGENTS.md

```markdown
If you are unsure how to do something, use `gh_grep` to search code examples from GitHub.
```
