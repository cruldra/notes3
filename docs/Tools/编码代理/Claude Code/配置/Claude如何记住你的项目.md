---
sidebar_position: 7
---

**每个 Claude Code 会话都从一个全新的上下文窗口开始**，要让知识跨越会话延续下去，需要依赖两种机制：

* **CLAUDE.md 文件**：你写下的、用来给 Claude 提供持久上下文的指令
* **自动记忆（auto memory）**：Claude 根据你的纠正和偏好自行写下的笔记

本页涵盖以下内容：

* [编写并组织 CLAUDE.md 文件](#claude.md-文件)
* [用 `.claude/rules/` 将规则限定到特定文件类型](#用-claude/rules/-组织规则)
* [配置自动记忆](#自动记忆)，让 Claude 自动做笔记
* [当指令未被遵循时的排查](#排查记忆相关问题)

## CLAUDE.md 与自动记忆的对比

Claude Code 有两套互补的记忆系统，二者都会在每次对话开始时加载。Claude 把它们当作上下文而非强制配置——你的指令越具体、越简洁，Claude 就越能稳定地遵循。

|              | CLAUDE.md 文件                       | 自动记忆                                                          |
| :----------- | :----------------------------------- | :--------------------------------------------------------------- |
| **谁来写**   | 你                                   | Claude                                                           |
| **写什么**   | 指令和规则                           | 学到的经验和模式                                                  |
| **作用域**   | 项目、用户或组织                     | 单个工作树（per working tree）                                    |
| **加载到**   | 每个会话                             | 每个会话（前 200 行或 25KB）                                       |
| **适用于**   | 编码规范、工作流程、项目架构          | 构建命令、调试洞察、Claude 自行发现的偏好                          |

当你想引导 Claude 的行为时，使用 CLAUDE.md 文件。自动记忆则让 Claude 无需人工干预即可从你的纠正中学习。

子代理（subagent）也能维护自己独立的自动记忆。详见[子代理配置](/en/sub-agents#enable-persistent-memory)。

## CLAUDE.md 文件

CLAUDE.md 是 Markdown 文件，为某个项目、你的个人工作流或你的整个组织提供给 Claude 的持久指令。你用纯文本编写它们，Claude 在每个会话开始时读取。

### 何时应当往 CLAUDE.md 中添加内容

把 CLAUDE.md 当作你不愿重复解释的事情的安放之处。下列情况就该添加：

* Claude 第二次犯了同样的错误
* 代码评审指出了 Claude 本该知道的关于该代码库的事情
* 你在聊天里输入了与上一次会话相同的纠正或澄清
* 一个新同事需要同样的上下文才能上手

只保留那些 Claude 在每次会话都应该掌握的事实：构建命令、约定、项目结构、"始终执行 X"之类的规则。如果某条内容是多步骤流程，或只对代码库的某个部分有意义，就把它移到[技能](/en/skills)或[路径限定的规则](#用-claude/rules/-组织规则)中。[扩展机制概览](/en/features-overview#build-your-setup-over-time)介绍了各机制的适用场景。

### 选择 CLAUDE.md 文件的存放位置

CLAUDE.md 可以放在多个位置，每个位置作用域不同。位置越具体优先级越高，会覆盖范围更广的设置。

| 作用域            | 位置                                                                                                                                                                            | 用途                                                       | 典型用例                                                  | 共享给                       |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | --------------------------------------------------------- | ---------------------------- |
| **托管策略**       | • macOS：`/Library/Application Support/ClaudeCode/CLAUDE.md`<br />• Linux 和 WSL：`/etc/claude-code/CLAUDE.md`<br />• Windows：`C:\Program Files\ClaudeCode\CLAUDE.md` | 由 IT/DevOps 管理的组织级指令                              | 公司编码标准、安全策略、合规要求                          | 组织中的所有用户            |
| **项目级指令**     | `./CLAUDE.md` 或 `./.claude/CLAUDE.md`                                                                                                                                          | 团队共享的项目指令                                         | 项目架构、编码规范、常用工作流                            | 通过版本控制共享给团队成员  |
| **用户级指令**     | `~/.claude/CLAUDE.md`                                                                                                                                                           | 对所有项目都生效的个人偏好                                  | 代码风格偏好、个人工具快捷方式                            | 仅你自己（所有项目）         |
| **本地级指令**     | `./CLAUDE.local.md`                                                                                                                                                             | 该项目的个人偏好，应加入 `.gitignore`                       | 你的沙箱 URL、惯用测试数据                                | 仅你自己（当前项目）         |

工作目录及其上层目录中的 CLAUDE.md 和 CLAUDE.local.md 文件会在启动时完整加载。子目录中的文件只在 Claude 读取该子目录下文件时按需加载。完整的解析顺序见 [CLAUDE.md 文件如何加载](#claude.md-文件如何加载)。

对于大型项目，你可以使用[项目规则](#用-claude/rules/-组织规则)按主题拆分指令。规则可以按文件类型或子目录限定作用域。

### 设置项目级 CLAUDE.md

项目级 CLAUDE.md 可以放在 `./CLAUDE.md` 或 `./.claude/CLAUDE.md`。在文件里加入所有项目协作者都适用的指令：构建与测试命令、编码规范、架构决策、命名约定和常用工作流。这些指令通过版本控制共享给团队，所以应聚焦项目级标准，而非个人偏好。

> **提示**：运行 `/init` 可自动生成初始的 CLAUDE.md。Claude 会分析你的代码库，写入它发现的构建命令、测试说明和项目约定。如果 CLAUDE.md 已存在，`/init` 会建议改进而非覆盖。之后再补充 Claude 不能自己发现的指令即可。
>
> 设置 `CLAUDE_CODE_NEW_INIT=1` 可启用交互式多阶段流程。`/init` 会询问需要设置哪些工件：CLAUDE.md 文件、技能和钩子；然后由子代理探索你的代码库、通过后续问答补全空白，最后呈现一份可评审的提案，再写入任何文件。

### 编写有效的指令

CLAUDE.md 文件在每个会话开始时加载到上下文窗口，与对话一起消耗 token。[上下文窗口可视化](/en/context-window)展示了 CLAUDE.md 在启动上下文中的相对位置。由于它是上下文而非强制配置，**指令的写法直接决定 Claude 遵循的稳定性**——具体、简洁、结构清晰的指令最有效。

**大小**：每份 CLAUDE.md 控制在 200 行以内。文件越长消耗的上下文越多，遵循度也越低。如果指令逐渐膨胀，使用[路径限定规则](#路径限定规则)让指令只在 Claude 处理匹配文件时加载。你也可以将内容拆成多份[导入文件](#导入额外文件)以便组织，但被导入的文件仍会在启动时加载并进入上下文窗口。

**结构**：用 Markdown 标题和列表对相关指令分组。Claude 扫描结构的方式跟人类读者类似——有组织的章节比稠密的段落更容易跟随。

**具体性**：写出可验证的具体指令，例如：

* 用「使用 2 空格缩进」，而不是「正确地格式化代码」
* 用「提交前运行 `npm test`」，而不是「测试你的改动」
* 用「API 处理器位于 `src/api/handlers/`」，而不是「保持文件有组织」

**一致性**：如果两条规则相互矛盾，Claude 可能会任意挑一条。定期检查你的 CLAUDE.md、子目录里的嵌套 CLAUDE.md，以及 [`.claude/rules/`](#用-claude/rules/-组织规则)，删除过时或冲突的指令。在 monorepo 中，使用 [`claudeMdExcludes`](#排除特定的-claude.md-文件) 跳过其他团队那些与你无关的 CLAUDE.md。

### 导入额外文件

CLAUDE.md 文件可通过 `@path/to/import` 语法导入其他文件。被导入的文件会在启动时与引用它的 CLAUDE.md 一起展开加载到上下文。

支持相对路径和绝对路径。**相对路径基于包含 `@` 引用的文件解析，而非工作目录**。被导入的文件可以递归导入其他文件，最大深度为 5 层。

例如要拉入一个 README、package.json 和工作流指南，在 CLAUDE.md 中任意位置用 `@` 语法引用即可：

```text
See @README for project overview and @package.json for available npm commands for this project.

# Additional Instructions
- git workflow @docs/git-instructions.md
```

对于不应入库的私人项目偏好，在项目根目录创建 `CLAUDE.local.md`。它会与 `CLAUDE.md` 一同加载，处理方式相同。把 `CLAUDE.local.md` 加入 `.gitignore` 以避免被提交；运行 `/init` 并选择"个人"选项会自动做这件事。

如果你在同一个仓库的多个 git 工作树（worktree）中工作，被 gitignore 的 `CLAUDE.local.md` 只存在于创建它的那个工作树中。要在多个工作树之间共享个人指令，可从主目录导入一个文件：

```text
# Individual Preferences
- @~/.claude/my-project-instructions.md
```

> **警告**：Claude Code 在项目中首次遇到外部导入时会显示一个列出这些文件的批准对话框。如果你拒绝，该导入保持禁用，对话框也不会再出现。

要更结构化地组织指令，参见 [`.claude/rules/`](#用-claude/rules/-组织规则)。

### AGENTS.md

Claude Code 只读 `CLAUDE.md`，不读 `AGENTS.md`。如果你的仓库已经为其他编码代理使用了 `AGENTS.md`，可以创建一个 `CLAUDE.md` 来导入它，让两套工具读取同一份指令，无需重复维护。你也可以在导入之下追加 Claude 专用的指令。Claude 在会话开始时加载被导入的文件，然后追加其余部分：

```markdown title="CLAUDE.md"
@AGENTS.md

## Claude Code

Use plan mode for changes under `src/billing/`.
```

如果不需要添加 Claude 专用内容，也可以使用符号链接：

```bash
ln -s AGENTS.md CLAUDE.md
```

在 Windows 上，创建符号链接需要管理员权限或开发者模式，所以请改用 `@AGENTS.md` 导入。

在已经包含 `AGENTS.md` 的仓库中运行 [`/init`](/en/commands) 时，它会读取该文件并将相关部分整合进生成的 `CLAUDE.md`。`/init` 也会读取其他工具的配置，例如 `.cursorrules` 和 `.windsurfrules`。

### CLAUDE.md 文件如何加载

Claude Code 通过沿着目录树向上回溯来读取 CLAUDE.md：从当前工作目录开始，在每个父目录里查找 `CLAUDE.md` 和 `CLAUDE.local.md`。这意味着如果你在 `foo/bar/` 里运行 Claude Code，它会加载 `foo/bar/CLAUDE.md`、`foo/CLAUDE.md` 以及它们旁边的任何 `CLAUDE.local.md`。

所有发现的文件会被拼接进上下文，而不是相互覆盖。在整个目录树上，内容按从文件系统根到工作目录的顺序排列。以 `foo/bar/` 为例，`foo/CLAUDE.md` 会比 `foo/bar/CLAUDE.md` 更早出现在上下文中——也就是说，越靠近你启动 Claude 的位置，越靠后被读到。同一目录内，`CLAUDE.local.md` 追加在 `CLAUDE.md` 之后，所以你的个人笔记是 Claude 在该层级读到的最后一段。

Claude 也会发现工作目录之下子目录里的 `CLAUDE.md` 和 `CLAUDE.local.md`。这些不会在启动时加载，而是在 Claude 读取那些子目录下的文件时被纳入。

如果你工作在一个大型 monorepo 里、其他团队的 CLAUDE.md 也被拾起，使用 [`claudeMdExcludes`](#排除特定的-claude.md-文件) 跳过它们。

CLAUDE.md 文件里的**块级 HTML 注释**（`<!-- 维护者笔记 -->`）会在内容被注入到 Claude 的上下文之前剥除。你可以用它给人类维护者留笔记，而不在上下文上花 token。代码块内部的注释会保留。如果你直接用 Read 工具打开 CLAUDE.md，注释仍然可见。

#### 从额外目录加载

`--add-dir` 标志让 Claude 能访问主工作目录之外的额外目录。默认情况下，这些目录里的 CLAUDE.md 文件不会被加载。

要同时加载额外目录中的记忆文件，设置 `CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD` 环境变量：

```bash
CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD=1 claude --add-dir ../shared-config
```

这会从该额外目录加载 `CLAUDE.md`、`.claude/CLAUDE.md`、`.claude/rules/*.md` 和 `CLAUDE.local.md`。如果你通过 [`--setting-sources`](/en/cli-reference) 排除了 `local`，则会跳过 `CLAUDE.local.md`。

### 用 `.claude/rules/` 组织规则

对于较大项目，你可以使用 `.claude/rules/` 目录把指令拆成多个文件，这样指令保持模块化、团队也更易维护。规则还能[限定在特定文件路径上](#路径限定规则)，只在 Claude 处理匹配文件时才进入上下文，减少噪声、节省上下文空间。

> **注意**：规则在每个会话开始时加载，或在匹配文件被打开时加载。对于不需要长期驻留上下文的任务专属指令，请使用[技能](/en/skills)——技能只在你显式调用、或 Claude 判断它与你的 prompt 相关时才加载。

#### 设置规则

把 Markdown 文件放进项目的 `.claude/rules/` 目录，每个文件覆盖一个主题，文件名要有描述性，如 `testing.md` 或 `api-design.md`。所有 `.md` 文件会被递归发现，所以你可以把规则组织成像 `frontend/` 或 `backend/` 这样的子目录：

```text
your-project/
├── .claude/
│   ├── CLAUDE.md           # 主项目指令
│   └── rules/
│       ├── code-style.md   # 代码风格指南
│       ├── testing.md      # 测试约定
│       └── security.md     # 安全要求
```

没有 [`paths` frontmatter](#路径限定规则) 的规则会在启动时加载，优先级与 `.claude/CLAUDE.md` 相同。

#### 路径限定规则

规则可以通过 YAML frontmatter 的 `paths` 字段限定到特定文件。这类有条件的规则只在 Claude 处理匹配指定模式的文件时才会生效。

```markdown
---
paths:
  - "src/api/**/*.ts"
---

# API 开发规则

- 所有 API 端点都必须包含输入验证
- 使用标准的错误响应格式
- 添加 OpenAPI 文档注释
```

没有 `paths` 字段的规则无条件加载，对所有文件生效。路径限定规则在 Claude 读取匹配模式的文件时触发，而不是在每次工具调用时触发。

在 `paths` 字段里使用 glob 模式按扩展名、目录或任意组合匹配文件：

| 模式                    | 匹配                                    |
| ----------------------- | --------------------------------------- |
| `**/*.ts`               | 任意目录下的所有 TypeScript 文件        |
| `src/**/*`              | `src/` 下的所有文件                     |
| `*.md`                  | 项目根目录下的 Markdown 文件            |
| `src/components/*.tsx`  | 特定目录下的 React 组件                 |

你可以指定多个模式，并用花括号扩展在一个模式里匹配多个扩展名：

```markdown
---
paths:
  - "src/**/*.{ts,tsx}"
  - "lib/**/*.ts"
  - "tests/**/*.test.ts"
---
```

#### 通过符号链接跨项目共享规则

`.claude/rules/` 目录支持符号链接，你可以维护一份共享规则集，并把它链接到多个项目里。符号链接被解析后正常加载，循环符号链接会被检测并妥善处理。

下面这个例子既链接一个共享目录，也链接一个独立文件：

```bash
ln -s ~/shared-claude-rules .claude/rules/shared
ln -s ~/company-standards/security.md .claude/rules/security.md
```

#### 用户级规则

放在 `~/.claude/rules/` 的个人规则会应用到本机的每个项目。适合那些与项目无关的偏好：

```text
~/.claude/rules/
├── preferences.md    # 你的个人编码偏好
└── workflows.md      # 你的偏好工作流
```

用户级规则先于项目规则加载，因此项目规则优先级更高。

### 为大型团队管理 CLAUDE.md

对于在团队中部署 Claude Code 的组织，你可以集中管理指令并控制哪些 CLAUDE.md 文件会被加载。

#### 部署组织级 CLAUDE.md

组织可以部署一份集中管理的 CLAUDE.md，对一台机器上的所有用户生效。该文件不能被个人设置排除。

1. **在托管策略位置创建文件**：
   * macOS：`/Library/Application Support/ClaudeCode/CLAUDE.md`
   * Linux 和 WSL：`/etc/claude-code/CLAUDE.md`
   * Windows：`C:\Program Files\ClaudeCode\CLAUDE.md`
2. **用配置管理系统分发**：使用 MDM、组策略、Ansible 或类似工具把该文件分发到开发者机器上。其他组织级配置选项见[托管设置](/en/permissions#managed-settings)。

托管 CLAUDE.md 与[托管设置](/en/settings#settings-files)用途不同。用设置进行技术性强制约束，用 CLAUDE.md 提供行为引导：

| 关注点                                            | 配置位置                                                  |
| :------------------------------------------------ | :-------------------------------------------------------- |
| 阻止特定工具、命令或文件路径                       | 托管设置：`permissions.deny`                              |
| 强制沙箱隔离                                       | 托管设置：`sandbox.enabled`                               |
| 环境变量与 API 提供方路由                          | 托管设置：`env`                                           |
| 认证方式与组织绑定                                 | 托管设置：`forceLoginMethod`、`forceLoginOrgUUID`         |
| 代码风格与质量指南                                 | 托管 CLAUDE.md                                            |
| 数据处理与合规提醒                                 | 托管 CLAUDE.md                                            |
| Claude 的行为指令                                  | 托管 CLAUDE.md                                            |

设置规则由客户端强制执行，与 Claude 的判断无关；CLAUDE.md 指令塑造 Claude 的行为，但不是强制层。

#### 排除特定的 CLAUDE.md 文件

在大型 monorepo 中，祖先目录的 CLAUDE.md 可能包含与你工作无关的指令。`claudeMdExcludes` 设置允许你按路径或 glob 模式跳过特定文件。

下例排除了一个顶层 CLAUDE.md 和上级目录里的某个规则目录。把它加到 `.claude/settings.local.json` 里，使排除只在你本机生效：

```json
{
  "claudeMdExcludes": [
    "**/monorepo/CLAUDE.md",
    "/home/user/monorepo/other-team/.claude/rules/**"
  ]
}
```

模式按 glob 语法匹配绝对文件路径。你可以在任意[设置层级](/en/settings#settings-files)（用户、项目、本地、托管策略）配置 `claudeMdExcludes`，数组会跨层级合并。

托管策略 CLAUDE.md 文件**不能**被排除。这确保组织级指令无论个人设置如何始终生效。

## 自动记忆

自动记忆让 Claude 跨会话累积知识，而你什么都不用写。Claude 在工作过程中给自己留笔记：构建命令、调试洞察、架构说明、代码风格偏好、工作流习惯。Claude 并不在每个会话都做记录，而是根据信息在未来对话里是否有用来决定值不值得记。

> **注意**：自动记忆需要 Claude Code v2.1.59 或更高版本。可用 `claude --version` 查看你的版本。

### 启用或禁用自动记忆

自动记忆默认开启。要切换状态，在会话中打开 `/memory` 使用自动记忆切换开关，或在项目设置中设置 `autoMemoryEnabled`：

```json
{
  "autoMemoryEnabled": false
}
```

通过环境变量禁用自动记忆：设置 `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`。

### 存储位置

每个项目在 `~/.claude/projects/<project>/memory/` 下都有自己的记忆目录。`<project>` 路径由 git 仓库推导而来，所以同一仓库内的所有工作树和子目录共享同一个自动记忆目录。在 git 仓库之外时，使用项目根目录代替。

要把自动记忆存到其他位置，在你的用户设置 `~/.claude/settings.json` 中设置 `autoMemoryDirectory`：

```json
{
  "autoMemoryDirectory": "~/my-custom-memory-dir"
}
```

该值必须是绝对路径或以 `~/` 开头。此设置可以从策略和用户设置中接受，也可以通过 `--settings` 标志接受，**但不能从项目设置或本地设置中接受**——因为这两个文件位于项目目录里，被克隆的仓库可能借此把自动记忆写到敏感位置。

该目录包含一个 `MEMORY.md` 入口文件以及可选的主题文件：

```text
~/.claude/projects/<project>/memory/
├── MEMORY.md          # 简明索引，每个会话都加载
├── debugging.md       # 调试模式的详细说明
├── api-conventions.md # API 设计决策
└── ...                # Claude 创建的其他主题文件
```

`MEMORY.md` 是记忆目录的索引。Claude 在整个会话中读写该目录里的文件，并用 `MEMORY.md` 跟踪什么内容存在了哪里。

自动记忆是**机器本地**的。同一 git 仓库内的所有工作树和子目录共享一份自动记忆目录，但这些文件不会跨机器或跨云端环境共享。

### 工作机制

每个对话开始时加载 `MEMORY.md` 的前 200 行或前 25KB（以先到者为准）。超过该阈值的内容不会在会话开始时加载。Claude 把详细笔记移到独立的主题文件里，从而让 `MEMORY.md` 保持简洁。

该限制只适用于 `MEMORY.md`。CLAUDE.md 文件无论多长都会被完整加载，但更短的文件能带来更好的遵循度。

像 `debugging.md` 或 `patterns.md` 这样的主题文件**不在启动时加载**，Claude 会在需要时用标准文件工具按需读取它们。

Claude 在会话期间读写记忆文件。当你在 Claude Code 界面看到「正在写入记忆」或「已回想起记忆」时，Claude 正在主动更新或读取 `~/.claude/projects/<project>/memory/`。

### 审计与编辑你的记忆

自动记忆文件是普通的 Markdown，你可以随时编辑或删除。在会话中运行 [`/memory`](#用-memory-查看与编辑) 即可浏览并打开记忆文件。

## 用 `/memory` 查看与编辑

`/memory` 命令列出当前会话中已加载的所有 CLAUDE.md、CLAUDE.local.md 和规则文件，让你切换自动记忆的开关，并提供打开自动记忆目录的链接。选择任一文件即可在编辑器中打开。

当你要求 Claude 记住某件事（比如「以后都用 pnpm，不要用 npm」或「记住 API 测试需要一个本地的 Redis 实例」），Claude 会把它保存到自动记忆。如果你想改写到 CLAUDE.md 里，可以直接告诉 Claude「把这一条加到 CLAUDE.md」，或者通过 `/memory` 自己编辑文件。

## 排查记忆相关问题

下面是 CLAUDE.md 和自动记忆最常见的几类问题，以及调试步骤。

### Claude 没有遵循我的 CLAUDE.md

CLAUDE.md 内容是作为**用户消息**在系统提示之后送达的，并不是系统提示本身。Claude 会读它并尝试遵循，但并不保证严格遵守——尤其当指令含糊或互相冲突时。

排查方法：

* 运行 `/memory` 确认你的 CLAUDE.md 和 CLAUDE.local.md 文件被加载了。如果某个文件没出现在列表里，Claude 就根本看不到它。
* 检查相关的 CLAUDE.md 是否位于会话会加载的位置上（参见[选择 CLAUDE.md 文件的存放位置](#选择-claude.md-文件的存放位置)）。
* 让指令更具体。「使用 2 空格缩进」比「漂亮地格式化代码」更有效。
* 查找 CLAUDE.md 文件之间的冲突指令。如果两份文件对同一行为给出不同指导，Claude 可能任意挑一条。

如果某条指令必须在特定时机执行——比如每次提交前、或每次文件编辑后——请把它写成[钩子（hook）](/en/hooks-guide)而不是写进 CLAUDE.md。钩子作为 shell 命令在固定生命周期事件上执行，与 Claude 的决策无关。

对于希望放到系统提示层级的指令，使用 [`--append-system-prompt`](/en/cli-reference#system-prompt-flags)。它必须每次调用时传入，所以更适合脚本和自动化，而非交互式使用。

> **提示**：用 [`InstructionsLoaded` 钩子](/en/hooks#instructionsloaded) 精确记录加载了哪些指令文件、何时加载、为何加载。这对调试路径限定规则或子目录中按需加载的文件很有用。

### 我不知道自动记忆里保存了什么

运行 `/memory` 并选择自动记忆文件夹即可浏览 Claude 保存的所有内容。一切都是纯文本 Markdown，可读、可编辑、可删除。

### 我的 CLAUDE.md 太大

超过 200 行的文件会消耗更多上下文，并可能降低遵循度。使用[路径限定规则](#路径限定规则)让指令只在 Claude 处理匹配文件时加载，或者裁掉每个会话都不需要的内容。拆分为 [`@path` 导入](#导入额外文件) 有助于组织，但**并不会减少上下文消耗**，因为被导入的文件会在启动时一并加载。

### `/compact` 之后指令似乎丢失了

**项目根目录的 CLAUDE.md 在压缩之后仍然存在**：`/compact` 后 Claude 会从磁盘重新读取并重新注入到会话中。子目录里的嵌套 CLAUDE.md **不会自动重新注入**，它们会在下一次 Claude 读取该子目录下文件时重新加载。

如果某条指令在压缩之后消失，要么是它只在对话中给出过，要么是它在某个尚未重新加载的嵌套 CLAUDE.md 里。把仅在对话中给出的指令写入 CLAUDE.md，让它持久生效。完整说明见[压缩后会保留什么](/en/context-window#what-survives-compaction)。

关于大小、结构和具体性的指南，参见[编写有效的指令](#编写有效的指令)。

## 相关资源

* [调试你的配置](/en/debug-your-config)：诊断 CLAUDE.md 或设置为何未生效
* [技能（Skills）](/en/skills)：将可重复的工作流封装成按需加载的技能
* [设置（Settings）](/en/settings)：通过配置文件控制 Claude Code 的行为
* [子代理记忆](/en/sub-agents#enable-persistent-memory)：让子代理维护自己的自动记忆
