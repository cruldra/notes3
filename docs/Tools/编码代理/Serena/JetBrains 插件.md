---
sidebar_position: 5
---

[JetBrains 插件](https://plugins.jetbrains.com/plugin/28946-serena/) 让 Serena MCP 服务器能够利用 JetBrains IDE 强大的代码分析和编辑能力。本页介绍如何安装插件以及如何正确配置 Serena。你仍然需要自行设置 Serena MCP 服务器，因此除了遵循以下说明外，请确保按照[安装说明](020_running.md)操作，并按照[客户端设置](030_clients.md)中的描述将 MCP 服务器连接到基于 LLM 的客户端。

[Serena JetBrains 插件 — JetBrains Marketplace](https://plugins.jetbrains.com/plugin/28946-serena/)

我们推荐将 JetBrains 插件作为使用 Serena 的首选方式，尤其是对于 JetBrains IDE 的用户。

**工作原理：**
1. 在 JetBrains IDE 中安装插件
2. 配置 Serena 使用 JetBrains 语言后端（见[下文](configure-jetbrains)）
3. 在 JetBrains IDE 中打开你要处理的项目，并在 Serena 中激活它（见[下文](jetbrains-workflow)）
4. 像往常一样通过 MCP 客户端开始编码

> **注意**：该插件是 Serena MCP 服务器的语言智能后端。它*不是*用于直接代理交互的 UI 扩展（如 Copilot）或类似的东西。你仍然通过你的常规客户端进行交互——无论是在 IDE 外部（如 Claude Code CLI）还是内部（如 Copilot 或 JetBrains AI Assistant）——并将其连接到 Serena MCP 服务器。该插件只是让 Serena MCP 服务器能够直接利用 JetBrains IDE 的能力！

**购买 JetBrains 插件是对 Serena 项目的支持。** 插件销售收入使我们能够投入更多资源来进一步开发和改进 Serena。

## JetBrains 插件的优势

以下多项功能仅在使用 JetBrains 插件时可用：

- **外部库索引**：依赖项和库被完全索引并可被 Serena 访问
- **增强的检索和重构能力**：插件新增了额外的[工具](../01-about/035_tools)（如类型层次结构检索、移动、查找声明、内联符号等），并将共享工具的底层机制改造为基于 IDE 的能力构建
- **交互式调试**：代理可以设置断点、检查变量、求值表达式并控制执行流程，通过直接与 IDE 调试器交互，使用 REPL 风格的界面以获得最大灵活性
- **改进的多代理支持**：单个 IDE 实例天然可以服务任意数量的代理会话，无需额外资源
- **增强的性能**：借助优化的 IDE 集成，工具执行更快
- **多语言卓越**和**框架支持**：对多语言多语言项目及框架提供一流支持（任何被 IDE 识别为符号的内容对 Serena 同样可用）
- **无需额外设置**：无需下载或配置单独的语言服务器

我们还在开发更多功能，如调试和高级内省能力，这些将仅通过 JetBrains 插件提供。

> **注意**：通过 Serena 的 JetBrains 工具，我们努力提供最新的功能。因此，其中一些被视为测试版功能（见[工具列表](../01-about/035_tools)），可能存在一些小问题。如果这些工具未按预期工作，请反馈你的使用体验。

(configure-jetbrains)=
## 配置 Serena 使用 JetBrains 插件

安装插件后，你需要配置 Serena 以使用它。

**全局配置**。

你可以运行：

```shell
serena init -b JetBrains
```

在全局 Serena 配置文件中将默认代码智能后端设置为 JetBrains。

或者，手动编辑配置文件 `~/.serena/serena_config.yml`（Windows 上为 `%USERPROFILE%\.serena\serena_config.yml`），设置：

```yaml
language_backend: JetBrains
```

请注意，如果你之前从未执行过 Serena，该文件可能尚不存在。

**按服务器实例配置**。全局配置文件中的配置设置可以在每个实例的基础上覆盖，只需在启动 Serena MCP 服务器时提供参数 `--language-backend JetBrains` 即可。

(per-project-language-backend)=
**按项目配置**。你也可以在项目的 `.serena/project.yml` 文件中按项目设置语言后端：

```yaml
language_backend: JetBrains
```

如果设置了此项，当项目在启动时被激活（通过 `--project` 标志），它会覆盖会话的全局 `language_backend` 设置。

> **重要**：语言后端在启动时确定一次，在运行会话期间无法更改。如果在启动后激活了使用不同后端的项目，Serena 将返回错误。
>
> 如果你需要处理使用不同后端的项目，可以：
> 1. 使用 `--project` 标志在启动时激活项目，这将使用其配置的后端。
> 2. 在客户端中配置单独的 MCP 服务器实例（每个后端一个）。

**验证设置**。你可以通过以下方式验证 Serena 正在使用 JetBrains 插件：检查仪表板，你将在配置概览中看到 `Languages: Using JetBrains backend`。你还会注意到你的客户端将使用 JetBrains 特有的工具，如 `jet_brains_find_symbol` 等。

(jetbrains-workflow)=
## 工作流

在 IDE 中安装插件并配置 Serena 使用 JetBrains 后端后，一般工作流很简单：

1. 在 JetBrains IDE 中打开你要处理的项目。请注意，项目必须在 IDE 中正确设置，即所有相关编程语言和框架的符号查找应在 IDE 中正常工作。
2. 在 Serena 中将项目的根文件夹激活为项目（见[项目创建](project-creation-indexing)和[项目激活](project-activation)）。
3. 像往常一样开始使用 Serena 的工具。

请注意，IDE 中打开的项目文件夹和 Serena 的项目根文件夹必须一致。

> **提示**：如果你需要在同一个代理会话中处理多个项目，请创建一个包含所有项目的单体仓库文件夹，并在 Serena 和 IDE 中都打开该文件夹。

## 高级用法和配置

### 在 Multi-Module 项目中使用 Serena

JetBrains IDE 支持*多模块项目*，其中一个项目可以将其他项目作为模块引用。然而，Serena 要求项目在单个根文件夹内自包含。项目根文件夹和 IDE 中打开的文件夹必须是一对一的关系。

因此，要让多模块设置与 Serena 协同工作，推荐的方法是创建一个**单体仓库文件夹**，即一个包含所有项目作为子文件夹的文件夹，并在 Serena 和 IDE 中都打开该单体仓库文件夹。

你不一定需要物理上将项目移动到公共父文件夹中；你也可以使用符号链接达到同样的效果（即在 Windows 上使用 `mklink`，在 Linux/macOS 上使用 `ln`，将项目文件夹链接到公共父文件夹中）。

### 在 Windows Subsystem for Linux (WSL) 中使用 Serena

JetBrains IDE 内置了对 WSL 的支持，允许你在 Windows 上运行 IDE，同时在 WSL 环境中处理代码。Serena JetBrains 插件在这种设置下同样可以无缝工作。

#### 使用 JetBrains Remote Development

推荐配置：
- 你的项目位于 WSL 文件系统中
- Serena 在 WSL 中运行（而非 Windows）
- IDE 有一个主机组件（在 WSL 中）和一个客户端组件（在 Windows 上）。Serena JetBrains 插件通常应**安装在主机**（而非客户端）中，以便代码智能功能可访问。

> **注意（插件安装位置）**：如果插件已安装，请检查禁用插件按钮上的选项。选择相应的选项以确保正确的安装位置（即主机，必要时从客户端中移除）。

> **警告**：不建议在 WSL 中使用映射的 Windows 路径！将项目保留在 Windows 文件系统中并通过 WSL 中的 `/mnt/` 访问非常缓慢，不推荐这样做。

**特殊网络设置**。如果你使用 Serena 和 IDE 运行在不同机器上的特殊设置，请确保 Serena 可以与 JetBrains 插件通信。你可以在 [serena_config.yml](050_configuration) 中配置 `jetbrains_plugin_server_address`，并通过 IDE 中的 Settings / Tools / Serena 配置 JetBrains 插件的监听地址（例如设置为 0.0.0.0 以在所有接口上监听，但要注意这样做的安全隐患）。

#### 其他 WSL 集成（例如 WSL 解释器）

- 你的项目位于 Windows 文件系统中
- WSL 仅用于运行工具（例如在 IDE 中使用 WSL Python 解释器）
- Serena、IDE 和插件均在 Windows 上运行

在这种配置下，无需特殊设置。

## Serena 插件配置选项

你可以在 IDE 的 Settings / Tools / Serena 下配置插件选项。

- **监听地址**（默认：`127.0.0.1`）——插件服务器监听的地址。只要 Serena 在同一台机器（或使用镜像网络的虚拟机上）运行，默认值即可正常工作。但如果 Serena MCP 服务器运行在另一台机器上，请配置监听地址以确保可以建立连接。你可以使用 `0.0.0.0` 在所有接口上监听（但要注意这样做的安全隐患）。

- **每次操作前同步文件系统**（默认：启用）——在处理 Serena 的请求之前是否同步文件系统状态。这对于确保插件不会读取过时数据很重要，但可能会对性能产生影响，尤其是在使用慢速文件系统时（例如 IDE 在 Windows 上运行而项目位于 WSL 文件系统中）。但请注意，如果没有 Serena 插件强制同步，你需要自行确保同步。不是由 IDE 自身或 Serena 在项目文件中应用的更改操作，可能不会被 IDE 感知到。通常，IDE 在获得焦点时会自动同步，使用文件监视器来实现（尽管这对于 WSL 文件系统可能可靠也可能不可靠）。此外，如果你主要在另一个应用程序（如 AI 聊天）中工作，IDE 可能不会频繁获得焦点。因此，当项目发生外部更改时，你需要要么让 IDE 获得焦点（如果这有效的话），要么手动触发同步（右键点击根文件夹 / Reload from Disk）。此外，请注意，即使使用例如 Claude Code 的内置编辑工具进行的编辑也算作外部修改。只有 Serena 的编辑工具是"JetBrains 感知"的，会告诉 IDE 更新已编辑文件的状态。因此，如果你使用 Serena 工具以外的工具进行基于 AI 的编辑，在决定禁用此选项时，请确保同步缺失不会成为问题。

## 与其他编辑器配合使用

我们了解并非所有人都使用 JetBrains IDE 作为主要代码编辑器。你仍然可以通过在你偏好的编辑器旁边运行一个 JetBrains IDE 实例来利用 JetBrains 插件。大多数 JetBrains IDE 都有免费的社区版可供你用于此目的。你只需要确保你正在处理的项目在 JetBrains IDE 中已打开并完成索引，这样 Serena 就可以连接到它。
