# 用技能扩展 Claude

> 创建、管理和共享技能，在 Claude Code 中扩展 Claude 的功能。包括自定义命令和捆绑技能。

技能扩展了 Claude 能做什么。创建一个包含指令的 `SKILL.md` 文件，Claude 会将其添加到工具集中。Claude 在相关时使用技能，或者你可以用 `/skill-name` 直接调用。

当你反复粘贴相同的工作手册、检查表或多步骤过程到聊天中时，或者当 CLAUDE.md 的某个部分已经成长为过程而非事实时，创建一个技能。与 CLAUDE.md 内容不同，技能的内容仅在使用时加载，所以长参考材料在需要之前几乎不消耗任何成本。

> **注意**：有关 `/help` 和 `/compact` 等内置命令以及 `/debug` 和 `/simplify` 等捆绑技能，见[命令参考](/en/commands)。
>
> **自定义命令已合并到技能中。** 位于 `.claude/commands/deploy.md` 的文件和位于 `.claude/skills/deploy/SKILL.md` 的技能都会创建 `/deploy` 且工作方式相同。你现有的 `.claude/commands/` 文件继续工作。技能添加了可选功能：支持文件的目录、[控制你或 Claude 调用它们](#控制谁调用技能)的 frontmatter，以及 Claude 在相关时自动加载它们的能力。

Claude Code 技能遵循 [Agent Skills](https://agentskills.io) 开放标准，该标准适用于多个 AI 工具。Claude Code 扩展了该标准，增加了[调用控制](#控制谁调用技能)、[子代理执行](#在子代理中运行技能)和[动态上下文注入](#注入动态上下文)等额外功能。

---

## 捆绑技能

Claude Code 包含一组在每个会话中可用的捆绑技能，包括 `/simplify`、`/batch`、`/debug`、`/loop` 和 `/claude-api`。与直接执行固定逻辑的内置命令不同，捆绑技能是基于提示的：它们给 Claude 一个详细的计划，让它使用自己的工具来编排工作。你调用它们的方式与任何其他技能相同，输入 `/` 后跟技能名称。

捆绑技能在[命令参考](/en/commands)中与内置命令一起列出，在用途列中标记为 **Skill**。

---

## 入门

### 创建第一个技能

这个示例创建一个技能，教 Claude 用可视化图表和类比来解释代码。由于它使用默认 frontmatter，Claude 可以在你询问代码如何工作时自动加载它，或者你可以直接用 `/explain-code` 调用它。

**步骤 1：创建技能目录**

在你的个人技能文件夹中为技能创建目录。个人技能在你的所有项目中可用。

```bash
mkdir -p ~/.claude/skills/explain-code
```

**步骤 2：编写 SKILL.md**

每个技能都需要一个 `SKILL.md` 文件，包含两部分：YAML frontmatter（在 `---` 标记之间）告诉 Claude 何时使用技能，以及 Markdown 内容包含技能被调用时 Claude 遵循的指令。`name` 字段成为 `/斜杠命令`，`description` 帮助 Claude 决定何时自动加载它。

创建 `~/.claude/skills/explain-code/SKILL.md`：

```yaml
---
name: explain-code
description: Explains code with visual diagrams and analogies. Use when explaining how code works, teaching about a codebase, or when the user asks "how does this work?"
---

When explaining code, always include:

1. **Start with an analogy**: Compare the code to something from everyday life
2. **Draw a diagram**: Use ASCII art to show the flow, structure, or relationships
3. **Walk through the code**: Explain step-by-step what happens
4. **Highlight a gotcha**: What's a common mistake or misconception?

Keep explanations conversational. For complex concepts, use multiple analogies.
```

**步骤 3：测试技能**

你可以通过两种方式测试：

**让 Claude 自动调用它**，询问与描述匹配的内容：

```text
How does this code work?
```

**或直接调用**，使用技能名称：

```text
/explain-code src/auth/login.ts
```

无论哪种方式，Claude 都应在其解释中包含类比和 ASCII 图。

### 技能存储位置

你存储技能的位置决定了谁可以使用它：

| 位置       | 路径                                              | 适用于                     |
| :--------- | :------------------------------------------------ | :------------------------- |
| Enterprise | 见 [managed 设置](/en/settings#settings-files)     | 组织中的所有用户           |
| 个人       | `~/.claude/skills/<skill-name>/SKILL.md`          | 你的所有项目               |
| 项目       | `.claude/skills/<skill-name>/SKILL.md`            | 仅此项目                   |
| 插件       | `<plugin>/skills/<skill-name>/SKILL.md`           | 插件启用的位置             |

当技能在不同层级共享相同名称时，更高优先级的位置获胜：enterprise > 个人 > 项目。插件技能使用 `plugin-name:skill-name` 命名空间，因此它们不会与其他层级冲突。如果你在 `.claude/commands/` 中有文件，它们的工作方式相同，但如果技能和命令共享相同名称，技能优先。

#### 从嵌套目录自动发现

当你在子目录中的文件工作时，Claude Code 会自动从嵌套的 `.claude/skills/` 目录发现技能。例如，如果你在 `packages/frontend/` 中编辑文件，Claude Code 还会在 `packages/frontend/.claude/skills/` 中查找技能。这支持包拥有自己技能的 monorepo 设置。

每个技能是一个以 `SKILL.md` 为入口点的目录：

```text
my-skill/
├── SKILL.md           # 主指令（必需）
├── template.md        # Claude 填充的模板
├── examples/
│   └── sample.md      # 显示预期格式的示例输出
└── scripts/
    └── validate.sh    # Claude 可以执行的脚本
```

`SKILL.md` 包含主指令且是必需的。其他文件是可选的，让你可以构建更强大的技能：Claude 填充的模板、显示预期格式的示例输出、Claude 可以执行的脚本或详细的参考文档。从你的 `SKILL.md` 中引用这些文件，这样 Claude 就知道它们包含什么以及何时加载它们。见[添加支持文件](#添加支持文件)了解更多详情。

> **注意**：`.claude/commands/` 中的文件仍然工作且支持相同的 [frontmatter](#frontmatter-参考)。推荐使用技能，因为它们支持支持文件等额外功能。

#### 来自附加目录的技能

`--add-dir` 标志[授予文件访问权限](/en/permissions#additional-directories-grant-file-access-not-configuration)而不是配置发现，但技能是一个例外：添加目录中的 `.claude/skills/` 会自动加载并被实时更改检测捕获，因此你可以在会话期间编辑这些技能而无需重启。

其他 `.claude/` 配置（如子代理、命令和输出样式）不会从附加目录加载。见[异常表](/en/permissions#additional-directories-grant-file-access-not-configuration)获取加载和未加载内容的完整列表，以及跨项目共享配置的推荐方式。

> **注意**：`--add-dir` 目录中的 CLAUDE.md 文件默认不加载。要加载它们，设置 `CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD=1`。见[从附加目录加载](/en/memory#load-from-additional-directories)。

---

## 配置技能

技能通过 `SKILL.md` 顶部的 YAML frontmatter 和随后的 Markdown 内容进行配置。

### 技能内容的类型

技能文件可以包含任何指令，但考虑你如何想要调用它们有助于指导包含什么：

**参考内容**添加 Claude 应用于你当前工作的知识。约定、模式、风格指南、领域知识。这些内容内联运行，因此 Claude 可以在你的对话上下文旁边使用它。

```yaml
---
name: api-conventions
description: API design patterns for this codebase
---

When writing API endpoints:
- Use RESTful naming conventions
- Return consistent error formats
- Include request validation
```

**任务内容**给 Claude 特定操作的逐步指令，如部署、提交或代码生成。这些通常是你想要直接用 `/skill-name` 调用而不是让 Claude 决定何时运行的操作。添加 `disable-model-invocation: true` 以防止 Claude 自动触发它。

```yaml
---
name: deploy
description: Deploy the application to production
context: fork
disable-model-invocation: true
---

Deploy the application:
1. Run the test suite
2. Build the application
3. Push to the deployment target
```

你的 `SKILL.md` 可以包含任何内容，但考虑你如何想要调用技能（由你、由 Claude 或两者）以及你希望它在哪里运行（内联或在子代理中）有助于指导包含什么。对于复杂的技能，你还可以[添加支持文件](#添加支持文件)以保持主技能专注。

### Frontmatter 参考

除了 Markdown 内容，你可以使用 `SKILL.md` 文件顶部 `---` 标记之间的 YAML frontmatter 字段来配置技能行为：

```yaml
---
name: my-skill
description: What this skill does
disable-model-invocation: true
allowed-tools: Read Grep
---

Your skill instructions here...
```

所有字段都是可选的。只有 `description` 是推荐的，这样 Claude 就知道何时使用技能。

| 字段                       | 必需       | 描述                                                                                                                                                                                                                       |
| :------------------------- | :--------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`                     | 否         | 技能的显示名称。如果省略，使用目录名称。仅小写字母、数字和连字符（最多 64 个字符）。                                                                                                                                       |
| `description`              | 推荐       | 技能的作用和何时使用它。Claude 使用此来决定何时应用技能。如果省略，使用 Markdown 内容的第一段。前置关键用例：描述超过 250 个字符时会在技能列表中截断以减少上下文用量。                                                     |
| `argument-hint`            | 否         | 自动补全期间显示的提示，指示预期参数。示例：`[issue-number]` 或 `[filename] [format]`。                                                                                                                                    |
| `disable-model-invocation` | 否         | 设置为 `true` 以防止 Claude 自动加载此技能。用于你想要用 `/name` 手动触发的工作流。默认：`false`。                                                                                                                         |
| `user-invocable`           | 否         | 设置为 `false` 以从 `/` 菜单隐藏。用于用户不应直接调用的背景知识。默认：`true`。                                                                                                                                           |
| `allowed-tools`            | 否         | 此技能激活时 Claude 可以在不请求许可的情况下使用的工具。接受空格分隔的字符串或 YAML 列表。                                                                                                                                 |
| `model`                    | 否         | 此技能激活时使用的模型。                                                                                                                                                                                                   |
| `effort`                   | 否         | 此技能激活时的[努力级别](/en/model-config#adjust-effort-level)。覆盖会话努力级别。默认：继承自会话。选项：`low`、`medium`、`high`、`max`（仅 Opus 4.6）。                                                                    |
| `context`                  | 否         | 设置为 `fork` 以在分叉的子代理上下文中运行。                                                                                                                                                                               |
| `agent`                    | 否         | 设置 `context: fork` 时使用哪种子代理类型。                                                                                                                                                                                |
| `hooks`                    | 否         | 作用域到此技能生命周期的 hooks。见[技能和代理中的 hooks](/en/hooks#hooks-in-skills-and-agents)获取配置格式。                                                                                                               |
| `paths`                    | 否         | 限制此技能何时激活的 glob 模式。接受逗号分隔的字符串或 YAML 列表。设置时，Claude 仅在使用与模式匹配的文件工作时自动加载技能。使用与[路径特定规则](/en/memory#path-specific-rules)相同的格式。                              |
| `shell`                    | 否         | 用于此技能中 `` !`command` `` 和 ` ```! ` ` 块的 shell。接受 `bash`（默认）或 `powershell`。设置 `powershell` 时在 Windows 上通过 PowerShell 运行内联 shell 命令。需要 `CLAUDE_CODE_USE_POWERSHELL_TOOL=1`。                |

#### 可用的字符串替换

技能支持在技能内容中替换动态值：

| 变量                   | 描述                                                                                                                                                                                                                           |
| :--------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `$ARGUMENTS`           | 调用技能时传递的所有参数。如果内容中不存在 `$ARGUMENTS`，参数会附加为 `ARGUMENTS: <value>`。                                                                                                                                   |
| `$ARGUMENTS[N]`        | 通过从 0 开始的索引访问特定参数，如 `$ARGUMENTS[0]` 访问第一个参数。                                                                                                                                                           |
| `$N`                   | `$ARGUMENTS[N]` 的简写，如 `$0` 表示第一个参数或 `$1` 表示第二个参数。                                                                                                                                                         |
| `${CLAUDE_SESSION_ID}` | 当前会话 ID。用于日志记录、创建会话特定文件或将会话输出与技能关联。                                                                                                                                                            |
| `${CLAUDE_SKILL_DIR}`  | 包含技能 `SKILL.md` 文件的目录。对于插件技能，这是插件内技能的子目录，而不是插件根目录。在内联注入命令中使用它来引用与技能捆绑的脚本或文件，无论当前工作目录是什么。                                                             |

索引参数使用 shell 风格的引号，因此用引号包装多字值以将它们作为单个参数传递。例如，`/my-skill "hello world" second` 使 `$0` 展开为 `hello world`，`$1` 展开为 `second`。`$ARGUMENTS` 占位符始终展开为输入的完整参数字符串。

**使用替换的示例：**

```yaml
---
name: session-logger
description: Log activity for this session
---

Log the following to logs/${CLAUDE_SESSION_ID}.log:

$ARGUMENTS
```

### 添加支持文件

技能可以在其目录中包含多个文件。这让 `SKILL.md` 专注于要点，同时让 Claude 仅在需要时访问详细的参考材料。大型参考文档、API 规范或示例集合不需要在每次技能运行时加载到上下文中。

```text
my-skill/
├── SKILL.md（必需 - 概览和导航）
├── reference.md（详细 API 文档 - 需要时加载）
├── examples.md（使用示例 - 需要时加载）
└── scripts/
    └── helper.py（实用脚本 - 执行而不是加载）
```

从 `SKILL.md` 中引用支持文件，这样 Claude 就知道每个文件包含什么以及何时加载它：

```markdown
## Additional resources

- For complete API details, see [reference.md](reference.md)
- For usage examples, see [examples.md](examples.md)
```

> **提示**：保持 `SKILL.md` 在 500 行以内。将详细的参考材料移到单独的文件中。

### 控制谁调用技能

默认情况下，你和 Claude 都可以调用任何技能。你可以输入 `/skill-name` 直接调用它，Claude 可以在与你的对话相关时自动加载它。两个 frontmatter 字段让你限制这个：

- **`disable-model-invocation: true`**：只有你可以调用技能。用于有副作用或你想要控制时序的工作流，如 `/commit`、`/deploy` 或 `/send-slack-message`。你不希望 Claude 因为你的代码看起来准备好就决定部署。

- **`user-invocable: false`**：只有 Claude 可以调用技能。用于不可作为命令操作的背景知识。`legacy-system-context` 技能解释旧系统如何工作。Claude 在相关时应该知道这个，但 `/legacy-system-context` 对用户来说不是有意义的操作。

这个示例创建一个只有你可以触发的部署技能。`disable-model-invocation: true` 字段阻止 Claude 自动运行它：

```yaml
---
name: deploy
description: Deploy the application to production
disable-model-invocation: true
---

Deploy $ARGUMENTS to production:

1. Run the test suite
2. Build the application
3. Push to the deployment target
4. Verify the deployment succeeded
```

以下是两个字段如何影响调用和上下文加载：

| Frontmatter                      | 你可以调用 | Claude 可以调用 | 何时加载到上下文                                     |
| :------------------------------- | :--------- | :-------------- | :--------------------------------------------------- |
| （默认）                         | 是         | 是              | 描述始终在上下文中，完整技能在被调用时加载           |
| `disable-model-invocation: true` | 是         | 否              | 描述不在上下文中，完整技能在你调用时加载             |
| `user-invocable: false`          | 否         | 是              | 描述始终在上下文中，完整技能在被调用时加载           |

> **注意**：在常规会话中，技能描述被加载到上下文中，这样 Claude 就知道有什么可用的，但完整技能内容仅在被调用时加载。[预加载技能的子代理](/en/sub-agents#preload-skills-into-subagents)工作方式不同：完整技能内容在启动时注入。

### 技能内容生命周期

当你或 Claude 调用技能时，渲染的 `SKILL.md` 内容作为单条消息进入对话并在会话的剩余部分保留在那里。Claude Code 不会在后续轮次中重新读取技能文件，因此将应该在整个任务期间适用的指导写为常驻指令而不是一次性步骤。

[自动压缩](/en/how-claude-code-works#when-context-fills-up)在 token 预算内携带已调用的技能向前。当对话被总结以释放上下文时，Claude Code 在总结后重新附加每个技能的最新调用，保留每个的前 5,000 个 token。重新附加的技能共享 25,000 个 token 的合并预算。Claude Code 从最近调用的技能开始填充此预算，因此如果你在一个会话中调用了许多技能，旧的技能在压缩后可能会被完全丢弃。

如果技能似乎在第一次响应后停止影响行为，内容通常仍然存在，模型正在选择其他工具或方法。加强技能的 `description` 和指令，这样模型会继续保持偏好，或使用 [hooks](/en/hooks) 来确定性地强制执行行为。如果技能很大或你在它之后调用了其他几个技能，在压缩后重新调用它以恢复完整内容。

### 为技能预批准工具

`allowed-tools` 字段在技能激活时授予列出工具的许可，因此 Claude 可以在不提示你批准的情况下使用它们。它不限制哪些工具可用：每个工具仍然可调用，你的[权限设置](/en/permissions)仍然管理未列出的工具。

这个技能让你在你调用它时运行 git 命令而无需每次批准：

```yaml
---
name: commit
description: Stage and commit the current changes
disable-model-invocation: true
allowed-tools: Bash(git add *) Bash(git commit *) Bash(git status *)
---
```

要阻止技能使用某些工具，改为在你的[权限设置](/en/permissions)中添加拒绝规则。

### 向技能传递参数

你和 Claude 都可以在调用技能时传递参数。参数通过 `$ARGUMENTS` 占位符可用。

这个技能按编号修复 GitHub 问题。`$ARGUMENTS` 占位符会被技能名称后面的任何内容替换：

```yaml
---
name: fix-issue
description: Fix a GitHub issue
disable-model-invocation: true
---

Fix GitHub issue $ARGUMENTS following our coding standards.

1. Read the issue description
2. Understand the requirements
3. Implement the fix
4. Write tests
5. Create a commit
```

当你运行 `/fix-issue 123` 时，Claude 收到"Fix GitHub issue 123 following our coding standards..."

如果你用参数调用技能但技能不包含 `$ARGUMENTS`，Claude Code 会将 `ARGUMENTS: <your input>` 附加到技能内容的末尾，这样 Claude 仍然看到你输入的内容。

要按位置访问各个参数，使用 `$ARGUMENTS[N]` 或更短的 `$N`：

```yaml
---
name: migrate-component
description: Migrate a component from one framework to another
---

Migrate the $ARGUMENTS[0] component from $ARGUMENTS[1] to $ARGUMENTS[2].
Preserve all existing behavior and tests.
```

运行 `/migrate-component SearchBar React Vue` 将 `$ARGUMENTS[0]` 替换为 `SearchBar`，`$ARGUMENTS[1]` 替换为 `React`，`$ARGUMENTS[2]` 替换为 `Vue`。使用 `$N` 简写的相同技能：

```yaml
---
name: migrate-component
description: Migrate a component from one framework to another
---

Migrate the $0 component from $1 to $2.
Preserve all existing behavior and tests.
```

---

## 高级模式

### 注入动态上下文

`` !`<command>` `` 语法在技能内容发送给 Claude 之前运行 shell 命令。命令输出替换占位符，因此 Claude 接收实际数据，而不是命令本身。

这个技能通过 GitHub CLI 获取实时 PR 数据来总结拉取请求。`` !`gh pr diff` `` 和其他命令首先运行，它们的输出被插入到提示中：

```yaml
---
name: pr-summary
description: Summarize changes in a pull request
context: fork
agent: Explore
allowed-tools: Bash(gh *)
---

## Pull request context
- PR diff: !`gh pr diff`
- PR comments: !`gh pr view --comments`
- Changed files: !`gh pr diff --name-only`

## Your task
Summarize this pull request...
```

当此技能运行时：

1. 每个 `` !`<command>` `` 立即执行（在 Claude 看到任何内容之前）
2. 输出替换技能内容中的占位符
3. Claude 接收带有实际 PR 数据的完全渲染的提示

这是预处理，不是 Claude 执行的东西。Claude 只看到最终结果。

对于多行命令，使用以 ` ```! ` 打开的围栏代码块而不是内联形式：

````markdown
## Environment
```!
node --version
npm --version
git status --short
```
````

要为用户、项目、插件或[附加目录](#来自附加目录的技能)来源的技能和自定义命令禁用此行为，在[设置](/en/settings)中设置 `"disableSkillShellExecution": true`。每个命令会被替换为 `[shell command execution disabled by policy]` 而不是运行。捆绑和 managed 技能不受影响。此设置在 [managed 设置](/en/permissions#managed-settings)中最有用，用户无法覆盖它。

> **提示**：要在技能中启用[扩展思考](/en/common-workflows#use-extended-thinking-thinking-mode)，在技能内容中的任何位置包含单词"ultrathink"。

### 在子代理中运行技能

当你想要技能在隔离中运行时，在 frontmatter 中添加 `context: fork`。技能内容成为驱动子代理的提示。它无法访问你的对话历史。

> **警告**：`context: fork` 仅对带有显式指令的技能有意义。如果你的技能包含"使用这些 API 约定"等指南而没有任务，子代理会收到指南但没有可操作的提示，并返回而没有有意义的输出。

技能和[子代理](/en/sub-agents)以两个方向协同工作：

| 方式                         | 系统提示                          | 任务              | 还加载                     |
| :--------------------------- | :-------------------------------- | :---------------- | :------------------------- |
| 带 `context: fork` 的技能    | 来自代理类型（Explore、Plan 等）  | SKILL.md 内容     | CLAUDE.md                  |
| 带 `skills` 字段的子代理     | 子代理的 markdown body            | Claude 的委托消息 | 预加载技能 + CLAUDE.md     |

使用 `context: fork` 时，你在技能中编写任务并选择代理类型来执行它。对于反向（定义使用技能作为参考材料的自定义子代理），见[子代理](/en/sub-agents#preload-skills-into-subagents)。

#### 示例：使用 Explore 代理的研究技能

这个技能在分叉的 Explore 代理中运行研究。技能内容成为任务，代理提供优化用于代码库探索的只读工具：

```yaml
---
name: deep-research
description: Research a topic thoroughly
context: fork
agent: Explore
---

Research $ARGUMENTS thoroughly:

1. Find relevant files using Glob and Grep
2. Read and analyze the code
3. Summarize findings with specific file references
```

当此技能运行时：

1. 创建一个新的隔离上下文
2. 子代理接收技能内容作为其提示（"Research $ARGUMENTS thoroughly..."）
3. `agent` 字段确定执行环境（模型、工具和权限）
4. 结果被总结并返回到你的主对话

`agent` 字段指定使用哪个子代理配置。选项包括内置代理（`Explore`、`Plan`、`general-purpose`）或来自 `.claude/agents/` 的任何自定义子代理。如果省略，使用 `general-purpose`。

### 限制 Claude 的技能访问

默认情况下，Claude 可以调用任何没有设置 `disable-model-invocation: true` 的技能。定义 `allowed-tools` 的技能在技能激活时授予 Claude 访问这些工具的权限而无需每次批准。你的[权限设置](/en/permissions)仍然管理所有其他工具的基线批准行为。`/compact` 和 `/init` 等内置命令无法通过 Skill 工具使用。

三种方式控制 Claude 可以调用哪些技能：

**禁用所有技能**，通过在 `/permissions` 中拒绝 Skill 工具：

```text
# 添加到拒绝规则：
Skill
```

**允许或拒绝特定技能**，使用[权限规则](/en/permissions)：

```text
# 仅允许特定技能
Skill(commit)
Skill(review-pr *)

# 拒绝特定技能
Skill(deploy *)
```

权限语法：`Skill(name)` 精确匹配，`Skill(name *)` 带任何参数的前缀匹配。

**隐藏个别技能**，通过在 frontmatter 中添加 `disable-model-invocation: true`。这从 Claude 的上下文中完全移除技能。

> **注意**：`user-invocable` 字段仅控制菜单可见性，不控制 Skill 工具访问。使用 `disable-model-invocation: true` 阻止编程调用。

---

## 共享技能

技能可以根据你的受众在不同的作用域分发：

- **项目技能**：将 `.claude/skills/` 提交到版本控制
- **插件**：在你的[插件](/en/plugins)中创建 `skills/` 目录
- **Managed**：通过 [managed 设置](/en/settings#settings-files)部署到组织范围

### 生成可视化输出

技能可以捆绑并运行任何语言的脚本，给 Claude 超出单个提示的能力。一个强大的模式是生成可视化输出：在你的浏览器中打开以探索数据、调试或创建报告的交互式 HTML 文件。

这个示例创建一个代码库浏览器：一个交互式树视图，你可以展开和折叠目录、一目了然地查看文件大小并通过颜色识别文件类型。

创建技能目录：

```bash
mkdir -p ~/.claude/skills/codebase-visualizer/scripts
```

创建 `~/.claude/skills/codebase-visualizer/SKILL.md`。描述告诉 Claude 何时激活此技能，指令告诉 Claude 运行捆绑的脚本：

````yaml
---
name: codebase-visualizer
description: Generate an interactive collapsible tree visualization of your codebase. Use when exploring a new repo, understanding project structure, or identifying large files.
allowed-tools: Bash(python *)
---

# Codebase Visualizer

Generate an interactive HTML tree view that shows your project's file structure with collapsible directories.

## Usage

Run the visualization script from your project root:

```bash
python ~/.claude/skills/codebase-visualizer/scripts/visualize.py .
```

This creates `codebase-map.html` in the current directory and opens it in your default browser.

## What the visualization shows

- **Collapsible directories**: Click folders to expand/collapse
- **File sizes**: Displayed next to each file
- **Colors**: Different colors for different file types
- **Directory totals**: Shows aggregate size of each folder
````

创建 `~/.claude/skills/codebase-visualizer/scripts/visualize.py`。此脚本扫描目录树并生成一个自包含的 HTML 文件，包含：

- **摘要侧边栏**显示文件计数、目录计数、总大小和文件类型数量
- **条形图**按文件类型分解代码库（按大小前 8 个）
- **可折叠树**你可以展开和折叠目录，带颜色编码的文件类型指示器

脚本需要 Python 但仅使用内置库，因此无需安装包：

```python
#!/usr/bin/env python3
"""Generate an interactive collapsible tree visualization of a codebase."""

import json
import sys
import webbrowser
from pathlib import Path
from collections import Counter

IGNORE = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', 'dist', 'build'}

def scan(Path: Path, stats: dict) -> dict:
    result = {"name": path.name, "children": [], "size": 0}
    try:
        for item in sorted(path.iterdir()):
            if item.name in IGNORE or item.name.startswith('.'):
                continue
            if item.is_file():
                size = item.stat().st_size
                ext = item.suffix.lower() or '(no ext)'
                result["children"].append({"name": item.name, "size": size, "ext": ext})
                result["size"] += size
                stats["files"] += 1
                stats["extensions"][ext] += 1
                stats["ext_sizes"][ext] += size
            elif item.is_dir():
                stats["dirs"] += 1
                child = scan(item, stats)
                if child["children"]:
                    result["children"].append(child)
                    result["size"] += child["size"]
    except PermissionError:
        pass
    return result

def generate_html(data: dict, stats: dict, output: Path) -> None:
    ext_sizes = stats["ext_sizes"]
    total_size = sum(ext_sizes.values()) or 1
    sorted_exts = sorted(ext_sizes.items(), key=lambda x: -x[1])[:8]
    colors = {
        '.js': '#f7df1e', '.ts': '#3178c6', '.py': '#3776ab', '.go': '#00add8',
        '.rs': '#dea584', '.rb': '#cc342d', '.css': '#264de4', '.html': '#e34c26',
        '.json': '#6b7280', '.md': '#083fa1', '.yaml': '#cb171e', '.yml': '#cb171e',
        '.mdx': '#083fa1', '.tsx': '#3178c6', '.jsx': '#61dafb', '.sh': '#4eaa25',
    }
    lang_bars = "".join(
        f'<div class="bar-row"><span class="bar-label">{ext}</span>'
        f'<div class="bar" style="width:{(size/total_size)*100}%;background:{colors.get(ext,"#6b7280")}"></div>'
        f'<span class="bar-pct">{(size/total_size)*100:.1f}%</span></div>'
        for ext, size in sorted_exts
    )
    def fmt(b):
        if b < 1024: return f"{b} B"
        if b < 1048576: return f"{b/1024:.1f} KB"
        return f"{b/1048576:.1f} MB"

    html = f'''<!DOCTYPE html>
<html><head>
  <meta charset="utf-8"><title>Codebase Explorer</title>
  ...
</head><body>
  ...
</body></html>'''
    output.write_text(html)

if __name__ == '__main__':
    target = Path(sys.argv[1] if len(sys.argv) > 1 else '.').resolve()
    stats = {"files": 0, "dirs": 0, "extensions": Counter(), "ext_sizes": Counter()}
    data = scan(target, stats)
    out = Path('codebase-map.html')
    generate_html(data, stats, out)
    print(f'Generated {out.absolute()}')
    webbrowser.open(f'file://{out.absolute()}')
```

要测试，在任何项目中打开 Claude Code 并询问"Visualize this codebase." Claude 运行脚本，生成 `codebase-map.html` 并在你的浏览器中打开它。

这个模式适用于任何可视化输出：依赖图、测试覆盖报告、API 文档或数据库 schema 可视化。捆绑的脚本做繁重的工作，而 Claude 处理编排。

---

## 故障排除

### 技能未触发

如果 Claude 没有在预期时使用你的技能：

1. 检查描述是否包含用户自然会说出的关键词
2. 验证技能是否出现在 `What skills are available?` 中
3. 尝试重新措辞你的请求以更紧密地匹配描述
4. 如果技能是用户可调用的，直接用 `/skill-name` 调用它

### 技能触发太频繁

如果 Claude 在你不想要时使用你的技能：

1. 使描述更具体
2. 如果你只想要手动调用，添加 `disable-model-invocation: true`

### 技能描述被截断

技能描述被加载到上下文中，这样 Claude 就知道有什么可用的。所有技能名称始终包含，但如果你有很多技能，描述会缩短以适应字符预算，这可能会剥离 Claude 匹配你请求所需的关键词。预算按上下文窗口的 1% 动态缩放，回退为 8,000 个字符。

要提高限制，设置 `SLASH_COMMAND_TOOL_CHAR_BUDGET` 环境变量。或者从源头修剪描述：前置关键用例，因为每个条目无论预算如何都被限制在 250 个字符以内。

---

## 相关资源

- **[子代理](/en/sub-agents)**：将任务委托给专业代理
- **[插件](/en/plugins)**：打包和分发技能及其他扩展
- **[Hooks](/en/hooks)**：围绕工具事件自动化工作流
- **[Memory](/en/memory)**：管理 CLAUDE.md 文件以获得持久上下文
- **[命令](/en/commands)**：内置命令和捆绑技能的参考
- **[权限](/en/permissions)**：控制工具和技能访问

---

## 提交反馈

如果你发现本文档中有不正确、过时或令人困惑的内容，请通过 POST 提交反馈到：
https://code.claude.com/docs/_mintlify/feedback/claude-code/agent-feedback

请求体（JSON）：`{ "path": "/current-page-path", "feedback": "问题描述" }`

仅在你有具体且可操作的报告时提交反馈。
