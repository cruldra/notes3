# Serena - 语义代码检索与编辑工具

![Serena logo](https://avatars.githubusercontent.com/u/181485370?v=4)

**作者**: [oraios](https://github.com/oraios) · GitHub Stars: 14,883

## 简介

Serena 是一个强大的**编码代理工具包**，能够将大语言模型(LLM)转变为功能完整的代理，**直接在你的代码库上工作**。

### 核心特性

- 🚀 **独立性强** - 不绑定特定的LLM、框架或界面，可以灵活使用
- 🔧 **语义代码工具** - 提供类似IDE的语义代码检索和编辑工具，在符号级别提取代码实体并利用关系结构
- 🆓 **免费开源** - 完全免费，增强你已有的LLM能力

Serena 为你的LLM/编码代理提供类似IDE的工具。有了它，代理不再需要读取整个文件、执行类似grep的搜索或字符串替换来查找和编辑正确的代码。相反，它可以使用以代码为中心的工具，如 `find_symbol`、`find_referencing_symbols` 和 `insert_after_symbol`。

## LLM集成方式

Serena 提供必要的工具用于编码工作流，但需要LLM来执行实际工作，协调工具使用。

### 集成方式

1. **通过模型上下文协议(MCP)** - Serena 提供MCP服务器，可集成到：
   - Claude Code 和 Claude Desktop
   - 基于终端的客户端：Codex、Gemini-CLI、Qwen3-Coder、rovodev、OpenHands CLI等
   - IDE：VSCode、Cursor、IntelliJ
   - 扩展：Cline、Roo Code
   - 本地客户端：OpenWebUI、Jan、Agno等

2. **通过mcpo连接到ChatGPT** - 适用于不支持MCP但支持OpenAPI工具调用的客户端

3. **集成到自定义代理框架** - Serena的工具实现与框架解耦，易于适配任何代理框架

## 编程语言支持

Serena 的语义代码分析能力基于**语言服务器**，使用广泛实施的语言服务器协议(LSP)。

### 开箱即用支持的语言

- **Python**
- **TypeScript/JavaScript**
- **PHP** (使用Intelephense LSP)
- **Go** (需要安装gopls)
- **R** (需要安装languageserver包)
- **Rust** (需要rustup)
- **C/C++**
- **Zig** (需要安装ZLS)
- **C#**
- **Ruby** (默认使用ruby-lsp)
- **Swift**
- **Kotlin**
- **Java**
- **Clojure**
- **Dart**
- **Bash**
- **Lua** (自动下载lua-language-server)
- **Nix** (需要安装nixd)
- **Elixir** (需要安装NextLS，Windows不支持)
- **Elm** (自动下载elm-language-server)
- **Scala** (需要手动设置，使用Metals LSP)
- **Erlang** (实验性)
- **Perl**
- **Haskell**
- **Julia**
- **AL**
- **Markdown**

## 快速开始

### 前置要求

Serena 由 `uv` 管理，需要先[安装uv](https://docs.astral.sh/uv/getting-started/installation/)。

### 运行MCP服务器

#### 使用uvx (推荐)

```bash
uvx --from git+https://github.com/oraios/serena serena start-mcp-server
```

#### 本地安装

```bash
# 1. 克隆仓库
git clone https://github.com/oraios/serena
cd serena

# 2. (可选) 编辑配置
uv run serena config edit

# 3. 运行服务器
uv run serena start-mcp-server
```

#### 使用Docker (实验性)

```bash
docker run --rm -i --network host -v /path/to/your/projects:/workspaces/projects ghcr.io/oraios/serena:latest serena start-mcp-server --transport stdio
```

#### 使用Nix

```bash
nix run github:oraios/serena -- start-mcp-server --transport stdio
```

### 配置

Serena 在四个地方进行配置：

1. **全局配置** - `~/.serena/serena_config.yml`
2. **启动参数** - 在客户端配置中传递给 `start-mcp-server` 的参数
3. **项目配置** - 项目目录中的 `.serena/project.yml`
4. **上下文和模式** - 通过上下文和模式进一步自定义

### 项目激活与索引

推荐方式是让LLM激活项目：

```
"激活项目 /path/to/my_project"
"激活项目 my_project"
```

对于大型项目，建议预先索引以加速工具：

```bash
uvx --from git+https://github.com/oraios/serena serena project index
```

## 客户端集成

### Claude Code

从项目目录运行：

```bash
claude mcp add serena -- uvx --from git+https://github.com/oraios/serena serena start-mcp-server --context ide-assistant --project "$(pwd)"
```

### Codex

在 `~/.codex/config.toml` 中添加：

```toml
[mcp_servers.serena]
command = "uvx"
args = ["--from", "git+https://github.com/oraios/serena", "serena", "start-mcp-server", "--context", "codex"]
```

启动后激活项目：
```
"使用serena激活当前目录作为项目"
```

### Claude Desktop

编辑 `claude_desktop_config.json`：

```json
{
    "mcpServers": {
        "serena": {
            "command": "/abs/path/to/uvx",
            "args": ["--from", "git+https://github.com/oraios/serena", "serena", "start-mcp-server"]
        }
    }
}
```

### IDE集成 (Cline, Roo-Code, Cursor, Windsurf等)

建议使用 `ide-assistant` 上下文，在MCP客户端配置中添加 `"--context", "ide-assistant"` 到参数列表。

### 本地GUI和框架

支持的本地GUI技术：
- Jan
- OpenHands
- OpenWebUI
- Agno

## 模式和上下文

### 上下文

上下文定义Serena运行的一般环境：

- `desktop-app` - 用于桌面应用(如Claude Desktop)，默认选项
- `agent` - 用于更自主的代理场景
- `ide-assistant` - 优化用于IDE集成

启动时指定：`--context <context-name>`

### 模式

模式进一步细化Serena的行为，可同时激活多个：

- `planning` - 专注于规划和分析任务
- `editing` - 优化代码修改任务
- `interactive` - 适合对话式交互
- `one-shot` - 单次响应完成任务
- `no-onboarding` - 跳过初始入职流程
- `onboarding` - 专注于项目入职流程

启动时指定：`--mode <mode-name>`

可在会话期间动态切换模式：使用 `switch_modes` 工具。

## 入职流程与记忆

### 入职流程

首次启动项目时，Serena会执行**入职流程**，熟悉项目并存储记忆，供未来交互使用。

### 记忆系统

记忆文件存储在项目的 `.serena/memories/` 目录中。代理可以在后续交互中选择读取这些记忆。你可以：
- 查看和调整现有记忆
- 手动添加新记忆
- 每个文件都是一个记忆文件

## 项目准备建议

### 代码结构

- 使用良好结构的代码
- 对于非静态类型语言，添加类型注解

### 从干净状态开始

从干净的git状态开始任务。在Windows上设置：

```bash
git config --global core.autocrlf true
```

### 日志、检查和自动化测试

- 设计有意义的可解释输出(如日志消息)
- 保持良好的测试覆盖率
- 从所有检查和测试通过的状态开始编辑任务

## 日志和仪表板

Serena提供两种访问日志的方式：

1. **Web仪表板** (默认启用)
   - 访问地址：`http://localhost:24282/dashboard/index.html`
   - 支持所有平台

2. **GUI工具** (默认禁用)
   - 主要支持Windows，可能在Linux上工作
   - macOS不支持

两者都可以在 `serena_config.yml` 中配置。

### serena_config.yml 配置详解

全局配置文件位于 `~/.serena/serena_config.yml`，包含以下配置选项：

#### 日志和调试选项

```yaml
# GUI 日志窗口（主要支持 Windows 和部分 Linux，macOS 不可用）
gui_log_window: False

# Web 仪表板（所有平台支持）
web_dashboard: True

# 启动时自动打开浏览器
web_dashboard_open_on_launch: True

# 日志级别（10=debug, 20=info, 30=warning, 40=error）
log_level: 20

# 跟踪 LSP 通信（用于调试语言服务器问题）
trace_lsp_communication: False
```

**说明**：
- **gui_log_window**: 图形日志窗口，主要用于 Windows
- **web_dashboard**: Web 仪表板，默认地址 `http://localhost:24282/dashboard/`
- **web_dashboard_open_on_launch**: 是否在启动时自动打开浏览器
- **log_level**: 最小日志级别
- **trace_lsp_communication**: 跟踪与语言服务器的通信

#### 工具配置

```yaml
# 工具执行超时时间（秒）
tool_timeout: 240

# 全局排除的工具列表
excluded_tools: []

# 包含的可选工具列表（默认禁用的工具）
included_optional_tools: []

# 工具返回结果的最大字符数
default_max_tool_answer_chars: 150000
```

**说明**：
- **tool_timeout**: 工具执行超时时间
- **excluded_tools**: 要禁用的工具名称列表
- **included_optional_tools**: 要启用的可选工具列表
- **default_max_tool_answer_chars**: 工具返回结果的默认最大长度

#### 语言服务器配置

```yaml
# 语言服务器特定设置
ls_specific_settings: {}
# 示例：
# ls_specific_settings:
#   python:
#     some_option: value
#   typescript:
#     another_option: value
```

**说明**：
- 高级配置选项，用于配置特定语言服务器的实现选项
- 键为语言名称（与 project.yml 中相同）
- 值为该语言服务器的配置选项

#### JetBrains 集成

```yaml
# 启用 JetBrains 模式（使用 JetBrains IDE 插件）
jetbrains: False
```

**说明**：
- 启用后使用基于 JetBrains IDE 插件的工具
- 注意：插件尚未发布，仅供 Serena 开发者使用

#### 统计和分析

```yaml
# 记录工具使用统计
record_tool_usage_stats: False

# Token 计数估算器（仅在 record_tool_usage_stats 为 True 时相关）
token_count_estimator: TIKTOKEN_GPT4O
```

**说明**：
- **record_tool_usage_stats**: 是否记录工具使用统计
- **token_count_estimator**: Token 计数方法，可选值见 `RegisteredTokenCountEstimator` 枚举

#### 项目管理

```yaml
# 已注册的项目列表（由 Serena 自动管理）
projects: []
```

**说明**：
- 此部分由 Serena 自动管理
- 当你激活项目时，会自动添加到此列表
- 不建议手动编辑

#### 完整配置示例

```yaml
gui_log_window: False
web_dashboard: True
web_dashboard_open_on_launch: True
log_level: 20
trace_lsp_communication: False

ls_specific_settings: {}

tool_timeout: 240
excluded_tools: []
included_optional_tools: []

jetbrains: False

default_max_tool_answer_chars: 150000

record_tool_usage_stats: False
token_count_estimator: TIKTOKEN_GPT4O

projects: []
```

#### 常用配置场景

**场景 1: 禁用自动打开浏览器**
```yaml
web_dashboard_open_on_launch: False
```

**场景 2: 启用调试模式**
```yaml
log_level: 10  # debug 级别
trace_lsp_communication: True
```

**场景 3: 启用工具使用统计**
```yaml
record_tool_usage_stats: True
token_count_estimator: TIKTOKEN_GPT4O
```

**场景 4: 禁用某些工具**
```yaml
excluded_tools:
  - execute_shell_command  # 禁用 shell 执行
  - delete_lines           # 禁用删除行
```

**场景 5: 启用可选工具**
```yaml
included_optional_tools:
  - initial_instructions   # 启用初始指令工具
  - jet_brains_find_symbol # 启用 JetBrains 符号查找
```

## 工具列表

### 默认工具

- `activate_project` - 基于项目名称或路径激活项目
- `check_onboarding_performed` - 检查是否已执行项目入职
- `create_text_file` - 在项目目录中创建/覆盖文件
- `delete_memory` - 从记忆存储中删除记忆
- `execute_shell_command` - 执行shell命令
- `find_file` - 在给定相对路径中查找文件
- `find_referencing_symbols` - 查找引用给定位置符号的符号
- `find_symbol` - 全局(或局部)搜索包含给定名称/子字符串的符号
- `get_current_config` - 打印代理的当前配置
- `get_symbols_overview` - 获取给定文件中定义的顶级符号概览
- `insert_after_symbol` - 在给定符号定义结束后插入内容
- `insert_before_symbol` - 在给定符号定义开始前插入内容
- `list_dir` - 列出给定目录中的文件和目录
- `list_memories` - 列出记忆存储中的记忆
- `onboarding` - 执行入职流程
- `prepare_for_new_conversation` - 为新对话准备指令
- `read_file` - 读取项目目录中的文件
- `read_memory` - 从记忆存储中读取给定名称的记忆
- `rename_symbol` - 使用语言服务器重构功能在整个代码库中重命名符号
- `replace_regex` - 使用正则表达式替换文件中的内容
- `replace_symbol_body` - 替换符号的完整定义
- `search_for_pattern` - 在项目中搜索模式
- `think_about_collected_information` - 思考工具，用于考虑收集信息的完整性
- `think_about_task_adherence` - 思考工具，用于确定代理是否仍在正确轨道上
- `think_about_whether_you_are_done` - 思考工具，用于确定任务是否真正完成
- `write_memory` - 将命名记忆写入记忆存储

### 可选工具

默认禁用，需要显式启用：

- `delete_lines` - 删除文件中的行范围
- `initial_instructions` - 获取当前项目的初始指令
- `insert_at_line` - 在文件的给定行插入内容
- `jet_brains_find_referencing_symbols` - 查找引用给定符号的符号
- `jet_brains_find_symbol` - 执行符号搜索
- `jet_brains_get_symbols_overview` - 检索指定文件中的顶级符号概览
- `remove_project` - 从Serena配置中删除项目
- `replace_lines` - 用新内容替换文件中的行范围
- `restart_language_server` - 重启语言服务器
- `summarize_changes` - 提供总结代码库更改的指令
- `switch_modes` - 通过提供模式名称列表来激活模式

## 与其他编码代理的比较

### 优势

- **免费开源** - 不需要订阅或API密钥
- **不绑定特定工具** - 可与任何MCP客户端、LLM使用
- **语义理解** - 使用语言服务器进行符号级代码理解
- **易于扩展** - 小型代码库，易于修改和扩展

### 与订阅型代理的区别

类似于Cursor的Agent、Windsurf的Cascade或VSCode的代理模式，但：
- 不需要订阅
- 不直接集成到IDE中
- 更灵活的集成方式

### 与API型代理的区别

可以作为API型代理使用，但独特之处在于也可以作为MCP服务器使用，无需API密钥。

### 与其他MCP编码代理的区别

Serena是唯一提供语义代码检索和编辑工具的MCP服务器，而不仅仅依赖基于文本的分析。

## 致谢

- **赞助商** - Visual Studio Code团队、Microsoft开源项目办公室和GitHub开源提供了一次性赞助
- **社区贡献** - 许多语言支持由开源社区贡献
- **技术基础**:
  - multilspy - 语言服务器包装库
  - Python MCP SDK
  - Agno和agent-ui
  - 各种语言服务器

## 扩展Serena

可以通过以下方式扩展Serena：

1. **添加新工具** - 继承 `serena.agent.Tool` 并实现 `apply` 方法
2. **添加新语言支持** - 为新语言服务器实现提供浅层适配器

详见[贡献指南](https://raw.githubusercontent.com/oraios/serena/main/CONTRIBUTING.md)。

## 资源链接

- **GitHub**: https://github.com/oraios/serena
- **更新日志**: [CHANGELOG.md](https://github.com/oraios/serena/blob/main/CHANGELOG.md)
- **路线图**: [roadmap.md](https://github.com/oraios/serena/blob/main/roadmap.md)
- **经验教训**: [lessons_learned.md](https://github.com/oraios/serena/blob/main/lessons_learned.md)
- **Docker文档**: [DOCKER.md](https://github.com/oraios/serena/blob/main/DOCKER.md)
- **ChatGPT集成**: [serena_on_chatgpt.md](https://github.com/oraios/serena/blob/main/docs/serena_on_chatgpt.md)
- **自定义代理**: [custom_agent.md](https://github.com/oraios/serena/blob/main/docs/custom_agent.md)

---

**注**: Serena正在积极开发中，请查看最新更新、即将推出的功能和经验教训以保持最新。

