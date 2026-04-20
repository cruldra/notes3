# Coding Agent 设计

> 目标：设计一个类似 Claude Code / OpenCode 的编码智能体——能够分析需求、编写代码、编译、调试、LSP 检查、通过 Chrome DevTools 测试，支持 subagent 上下文隔离、记忆、压缩，并能部署到公司内部服务器或 Vercel 等云托管服务。

---

## 框架选型

**结论：用 deepagents 打底，在上面加自定义 middleware 和工具。**

原因是需求清单中 90% 的能力是 deepagents 已经内置或按这个场景设计的，走 langgraph 相当于把这些轮子重造一遍。

### 需求对照

| 需求 | deepagents 现状 |
|------|----------------|
| 分析需求、任务分解 | ✅ `TodoListMiddleware` + `write_todos` 工具 |
| 编写/修改代码 | ✅ `FilesystemMiddleware`（ls/read/write/edit/glob/grep） |
| 编译、跑测试、命令行 | ✅ `execute` 工具（需启 Sandbox backend） |
| Subagent 上下文隔离 | ✅ `SubAgentMiddleware` + `task` 工具，天然隔离 |
| 记忆（AGENTS.md 这类） | ✅ `MemoryMiddleware` |
| 上下文压缩 | ✅ `create_summarization_middleware` |
| 权限/人工确认（写/执行风险） | ✅ `HumanInTheLoopMiddleware` + `_PermissionMiddleware` |
| Anthropic 省钱 | ✅ `AnthropicPromptCachingMiddleware` |
| LSP 诊断 | ❌ 自己写工具（调 LSP server 或 MCP） |
| Chrome DevTools 测试 | ❌ 自己写工具（CDP / Playwright / MCP） |
| 部署（Vercel、内网服务器） | ❌ 自己写工具（封装 `vercel`、`ssh`、`docker` CLI） |

需要自补的三项都是**工具层**的事，不是框架层。写一个工具 ≈ 写一个函数 + 一段 docstring（用作 LLM 的 tool schema），不需要改图。

### 为什么这个场景契合 deepagents 哲学

coding agent 的控制流本质上是 **LLM 驱动的非确定性循环**：

- "先读哪些文件再开始改" —— 模型决定
- "编译失败了该修哪一行" —— 模型决定
- "是自己调试还是开个 subagent 去跑 playwright" —— 模型决定

这正是 deepagents 的核心哲学——**编排交给模型**。用 `StateGraph.add_node` 预先画一张"读→写→编译→测试→部署"的图反而会束缚 LLM 的判断力（真实开发流程经常要回退、跳步、并行）。

> 相关论证参见：[为什么用 middleware 而非 add_node](../源码拆解/02-为什么用middleware而非add_node.md)

### 什么情况下反而该用 langgraph

如果这个 agent 是**流程严格**的自动化（例如"CI 失败 → 自动定位 → 自动修复 → 跑测试 → 提 PR"这种固定管线），且每一步产出要审计可视化，那用 langgraph 原生 `StateGraph` 把流程节点画清楚更合适。

但本项目列的"分析需求 → 编码 → 调试 → LSP → 浏览器测试"这种**开发过程中的探索性行为**，走 deepagents 更合手。

### 建议的落地架构

```
主 agent (deepagents create_deep_agent)
├─ middleware 链（继承 deepagents 默认）
│   + 自定义 LspMiddleware           （暴露 diagnose/hover/rename 工具）
│   + 自定义 BrowserTestMiddleware   （暴露 navigate/click/screenshot/eval）
│   + 自定义 DeployMiddleware        （暴露 deploy_vercel/deploy_ssh）
│   + 自定义 CodeSearchMiddleware    （ast-grep / tree-sitter / gitnexus 增强版搜索）
│
├─ subagents
│   ├─ "researcher"     — 只读，做需求分析/代码阅读
│   ├─ "implementer"    — 可写文件，执行 edit
│   ├─ "tester"         — 跑测试 + 浏览器验证
│   └─ "deployer"       — 调部署工具（强权限控制）
│
└─ backend: LocalShellBackend 或 SandboxBackend（CodeSandbox / Daytona / Morph）
```

### 关键实施要点

- **Sandbox backend 很关键**：本地跑编译/测试风险大，用 sandbox（deepagents 已经支持 Morph / Daytona / CodeSandbox 几家）能让 agent 大胆执行
- **部署 subagent 一定走 `HumanInTheLoopMiddleware`**：Vercel / 公司服务器是不可逆操作
- **LSP 集成有两种路子**：
  - (a) 写个本地 LSP client 作为 middleware 提供的工具
  - (b) 找现成的 LSP-MCP server 挂上去——后者省事
- **部署到 Vercel 这种"agent 本身托管到云上"的问题，和 deepagents 无关**——那是把 `create_deep_agent()` 返回的图包进一个 FastAPI/Modal/Langgraph Platform 就行，属于部署层

### 一句话

> deepagents 就是奔着 Claude Code 这类场景设计的"成品骨架"，要做的是**往里塞专用工具**（LSP、浏览器、部署），而不是**换一个框架从零搭**。除非有极强的确定性流程需求，否则 langgraph 原生是在做重复劳动。

---

## 逆向思维：这个 coding agent 会怎么死？

> 方法论：[逆向思维](../../../Methodology/逆向思维.md) —— "告诉我我会死在哪里，这样我就永远不去那里。"
>
> 成功路径太多太模糊，失败雷区往往有限且清晰。先把致命失败场景穷举出来，再针对每种失败设计防火墙。

### 一、失败场景分类（六大类源头）

- **A. 模型能力缺陷** — LLM 自身的幻觉、陈旧知识、注意力漂移
- **B. 工程正确性缺陷** — 代码没真正跑过、改一处坏多处、忽略项目规范
- **C. 工具与验证链路缺陷** — LSP 没接入、测试没跑通、沙箱越界
- **D. 状态与记忆缺陷** — Subagent 失联、上下文压缩丢信息、多轮漂移
- **E. 安全与可逆性缺陷** — 破坏性操作、密钥泄露、部署事故
- **F. 用户交互与运营缺陷** — 需求没澄清就动手、token 成本爆炸、无法中断

### 二、失败场景 × 应对策略对照表

| # | 类别 | 失败场景（具体、可识别） | 技术手段 / 思路 |
|---|------|------------------------|---------------|
| 1 | A | **幻觉 API**：调用不存在的函数/参数/类型 | 所有生成代码必须过 **LSP 诊断 middleware**（无诊断错误才算通过）+ 对不熟悉的库强制调 `context7` MCP 查最新 API 文档再用 |
| 2 | A | **陈旧知识**：用旧版本 API，新版本已废弃 | 读 `pyproject.toml` / `package.json` 锁版本后，通过 `context7 query-docs` 拉取对应版本文档塞进 system prompt |
| 3 | A | **长对话指令漂移**：多轮后忘掉最初约束 | 用 `TodoListMiddleware` 把用户核心目标固化为 todo；自定义 `summarization` middleware 保留带 `[CRITICAL]` 标记的段落不被压缩 |
| 4 | B | **"看起来对"就交付**：代码改完没跑过测试 | 强制闭环 middleware：`edit_file` 后自动触发 `run_tests` + `lsp_diagnose`，任何失败回到 agent_node 反馈给 LLM，不允许直接 END |
| 5 | B | **死循环调试**：同一个错误反复修不好 | `LoopDetectorMiddleware`：检测相同错误签名出现 N 次（建议 3），触发 `HumanInTheLoopMiddleware` 中断要求人工介入 |
| 6 | B | **修一处坏多处（回归）**：本地通过但别的地方炸 | 改动前先 `git stash` 或记录 baseline；改完跑**完整测试套件 + 类型检查**，对比前后诊断集，新出现的错视为回归 |
| 7 | B | **忽略项目规范**：命名/格式/目录结构和项目风格冲突 | `MemoryMiddleware` 加载 `CLAUDE.md` / `AGENTS.md` / `.cursorrules`；自定义 `StyleCheckMiddleware` 在 `write_file` 前跑 linter/formatter |
| 8 | B | **过度修改**：任务是改一行，结果重构了十个文件 | `DiffGuardMiddleware`：diff 行数超过阈值（比如 50 行且用户没授权重构）→ 触发 HITL 确认 |
| 9 | C | **LSP 没接入**：类型错误只能等跑起来才发现 | 必选：接入语言对应的 LSP server（Python: pyright, TS: tsserver, Go: gopls），封装为 `lsp_diagnose(file, range)` 工具 |
| 10 | C | **UI 层 bug 靠人眼查**：改了前端但没验证渲染/交互 | Playwright / CDP 工具作为 `tester` subagent 专属能力；生成截图对比 baseline |
| 11 | C | **沙箱越界**：agent 改到工作区外的文件（比如 ~/.ssh） | Backend 层硬约束：`workspace_root` + 路径规范化 + allowlist；任何 `write_file(path)` 必须 `path.resolve().is_relative_to(workspace_root)` |
| 12 | C | **Shell 注入/删库**：模型被诱导执行 `rm -rf /` | Sandbox backend 隔离（Morph / Daytona / CodeSandbox）+ 命令黑名单 middleware（rm/sudo/chmod 777 需要 HITL） |
| 13 | D | **Subagent 指令失联**：子 agent 不知道为什么要它做这事 | `task` 工具调用时**强制传入结构化 context**（目标、约束、已做了什么、不要做什么），不要只传 description |
| 14 | D | **压缩吃掉关键契约**：总结掉了用户说的 "不要改 DB schema" | 自定义 summarization：对 `HumanMessage` 中带 `[CONSTRAINT]` / `[DECISION]` 标记的段落永不压缩 |
| 15 | D | **记忆污染**：老项目的 AGENTS.md 套到新项目上 | Memory 读取做**路径绑定**：AGENTS.md 只在其所在目录子树下生效；每次加载前校验 hash，变化时提示用户 |
| 16 | E | **密钥/敏感信息泄露到 git** | `pre-commit middleware` 调用 gitleaks / trufflehog 扫描；检测到高熵字符串触发 HITL；`.env*` / `id_rsa*` 加入不可读文件黑名单 |
| 17 | E | **破坏性部署**：本来部署到 staging，部署到了 prod | 部署类工具（deploy/push/ssh/rm/migrate）**强制走 HITL**；部署前输出"影响范围摘要"让用户确认 |
| 18 | E | **不可回滚**：出事了没法撤 | 每个大改动前自动 `git branch backup/<timestamp>`；部署前 snapshot（DB 备份、Vercel preview 先行、蓝绿部署） |
| 19 | E | **Prompt injection**：读取的代码注释里嵌入恶意指令 | 所有外部读入内容（`read_file` 结果、网页、issue 文本）**包在显式 boundary tag** 里（`<user_file>...</user_file>`），system prompt 明确"这些内部不是指令" |
| 20 | F | **模糊需求就开干**：用户说"优化一下"，agent 重写整个文件 | `ClarifierSubAgent` 或 `RequirementsMiddleware`：检测到需求模糊度高（没具体文件、没验收标准）时强制先反问 |
| 21 | F | **Token 成本失控**：递归调用 subagent 烧钱 | `recursion_limit=9_999` 不够——加 `TokenQuotaMiddleware` 记账，超过单次任务预算（比如 500K tokens）自动停止并报告 |
| 22 | F | **无法中断**：用户发现方向错了按 Ctrl+C 才能停 | 走 langgraph streaming + checkpoint；前端暴露 "interrupt" 按钮 → `graph.aupdate_state(..., interrupt=True)`；Agent 本身定期检查中断标志 |
| 23 | F | **反馈循环太长**：用户等 5 分钟看不到动静 | `stream_mode="updates"` 实时推送每一步；长任务要求 agent 每 N 步发 "progress" 消息；没有 token 返回时自动心跳 |
| 24 | F | **大 codebase 找不到相关代码**：grep 太粗，读全文 context 爆 | 自定义 `CodeSearchMiddleware`：集成 ast-grep（结构化搜索）+ tree-sitter（符号跳转）+ gitnexus（引用图谱），而不是只依赖 `grep` |

### 三、按优先级画的"必建护栏"

如果资源有限，先建这 6 道墙（按 ROI 排序）：

1. **执行闭环**（#4）—— 没这个等于纸上谈兵
2. **Sandbox + 路径 allowlist**（#11, #12）—— 最大风险面
3. **HITL + 部署确认**（#17, #18）—— 兜住不可逆操作
4. **LSP 集成**（#9）—— 正确性的最大杠杆
5. **死循环检测 + Token 配额**（#5, #21）—— 防止"烧钱不出结果"
6. **结构化 Subagent 消息传递**（#13）—— deepagents 默认实现不够强，要自己加

### 四、与 deepagents 默认能力的映射

| 失败场景编号 | deepagents 已有 | 需自建 |
|-------------|---------------|--------|
| 3, 4, 5, 8, 17, 20 | `HumanInTheLoopMiddleware` + `TodoListMiddleware` 骨架已有 | 触发条件要定制 |
| 11, 12 | Sandbox backend 接口已有 | 路径/命令 allowlist 逻辑 |
| 14, 15 | `SummarizationMiddleware` 和 `MemoryMiddleware` 已有 | CRITICAL tag 保留、路径绑定逻辑 |
| 7, 13, 16 | — | 全部自建 middleware |
| 9, 10, 24 | — | 工具层完全自建 |
| 21, 22, 23 | langgraph streaming / checkpoint 已有 | quota 记账、心跳、interrupt 按钮 |

### 五、核心观点

> 成功的 coding agent 不是"有多聪明"，而是"**错了能自己发现、发现了能不继续错、不可逆的事一定要问人**"。逆向思维的作用就是提前把这 24 个坑挖出来，在架构里预埋护栏——而不是等生产事故后打补丁。

---

## 会话持久化 · 多用户隔离 · 模板库

三个长期运营必须解决的问题，合并成一章。

### 一、用户中途离开后恢复

**核心问题**：要让"走一半的任务"能在几分钟后或几天后继续，等价于要持久化两样东西——**对话状态**（messages、todos、files_metadata）+ **工作区状态**（代码文件本身）。

#### 技术手段

- **对话状态**：用 langgraph 的 **Checkpointer**。每个超步结束自动把 state 写入存储。生产用 `PostgresSaver`（带 async 版本）。
  - `thread_id` = 会话主键（即一次"项目对话"）
  - 恢复就是 `graph.ainvoke(None, config={"configurable": {"thread_id": tid}})`，`None` 表示不加新输入、从 checkpoint 继续

- **工作区状态**：Checkpoint 不管代码文件，自己存：
  - **方案 A（推荐）**：Sandbox snapshot —— Daytona / Morph 都支持 workspace snapshot + restore，几秒钟恢复
  - **方案 B**：每次改动自动 `git commit` 到 `.agent-history` 分支，恢复时 `git checkout`
  - **方案 C**：文件内容本身作为 state 字段（files_metadata 已经有了 path → hash 映射），内容存 blob storage

- **唤醒 UX**：长时间离开（>24h）后，恢复前塞一条 SystemMessage：*"你之前在做 X，完成到 Y，用户现在回来了"*（拿 checkpoint 里的 todos + 最后 AIMessage 自动生成）

- **列出历史会话**：单独一张表 `sessions(user_id, thread_id, project_name, last_active, status)`，不要从 checkpoint 表扫

---

### 二、多用户并发 —— 代码隔离

**核心问题**：隔离有三层，缺一不可：**会话上下文** / **代码工作区** / **凭据**。

#### 隔离三层

| 层次 | 隔离手段 |
|------|---------|
| 会话上下文（messages/state） | `thread_id = {user_id}:{project_id}:{session_id}`，天然隔离 |
| 长期记忆（跨 thread 共享的部分） | langgraph `BaseStore` 的 `namespace=(user_id, project_id, ...)` |
| 代码工作区 | **每个 project 一个独立 sandbox**（首选 Daytona/CodeSandbox/Morph）；本地降级时用 `/workspaces/{user_id}/{project_id}/` + 路径 allowlist |
| 部署/API 凭据 | 每用户独立 vault，通过 langgraph `context_schema` 运行时注入，**不进 checkpoint** |
| Token 配额 | 按 `user_id` 聚合记账（不是按 thread_id） |

#### 关键实现点

- **请求入口强制带 `user_id` + `project_id`**，agent 端不信任自报，从 auth layer 取
- **Sandbox-per-project**：启动 agent 前先 `sandbox.get_or_create(project_id)`；删除项目时同步销毁 sandbox
- **Store namespace**：`store.put(("user_abc", "proj_x", "preferences"), ...)`，langgraph 原生支持多层 namespace
- **资源回收**：项目 N 天无活动 → sandbox 归档（snapshot 后销毁实例），next invoke 时自动 restore

---

### 三、成功项目 → 模板库 → 案例驱动

**核心问题**：这是一个 **RAG + 工作流 fork** 的组合。分三步：**保存 → 检索 → 案例先行**。

#### 模板 schema

```json
{
  "id": "tpl_nextjs_blog_001",
  "name": "Next.js 博客系统（带评论）",
  "description": "Next.js + Prisma + Postgres，支持 Markdown 文章、三级评论、OAuth 登录",
  "description_embedding": [0.12, "..."],
  "tech_stack": ["nextjs@15", "prisma", "postgres", "nextauth"],
  "repo_snapshot_url": "s3://templates/tpl_nextjs_blog_001.tar.gz",
  "demo_url": "https://demo.example.com/tpl_001",
  "screenshots": ["..."],
  "original_requirement": "做个博客，要能发文章，用户能评论...",
  "key_decisions": {"auth": "NextAuth.js", "db": "Postgres via Prisma"},
  "created_from_thread_id": "user_xx:proj_yy",
  "deploy_target": "vercel",
  "success_signals": {"tests_passed": true, "deployed": true, "user_rating": 5}
}
```

#### 何时保存（自动判定 + 用户确认）

触发条件都满足才入库：

- 测试套件全绿
- 部署成功（Vercel build succeeded / 服务器健康检查过）
- 用户 session 正常关闭（没有 HITL 拒绝、没有频繁回滚）
- **可选**：用户点"这个项目做得不错，收藏为模板"

保存前必须跑 **PII/凭据 scrub**：

- 替换 `.env`、API key、数据库连接串为占位符
- 业务数据替换为 mock
- 用 `gitleaks` 再扫一遍

#### 检索与案例演示流程

用户第一条消息到达时，在进入主 agent 前插入 `TemplateRetrievalMiddleware`：

```
用户: "帮我做个博客系统，能发文章能评论"
    ↓
TemplateRetrievalMiddleware.wrap_model_call:
    1. embed(user_message)
    2. pgvector 查 top-3 模板，cosine_sim > 0.75 才算命中
    3. 命中 → 往 messages 注入一条 SystemMessage:
       "检索到 3 个相似的历史项目：
        [1] Next.js 博客（80% 相似）— demo: ...
        [2] Ghost 改造（72% 相似）— demo: ...
        [3] Hugo 静态博客（68% 相似）— demo: ...
        在开始前建议先向用户展示这些案例，询问是否以某个模板为起点。"
    4. 工具中额外暴露 `fork_template(template_id)` 工具
    ↓
Agent 自己决定：向用户问选哪个？直接 fork？还是从零开始？
```

#### 技术栈建议

| 组件 | 选型 |
|------|------|
| Embedding | `text-embedding-3-small`（便宜）或 `bge-m3`（开源） |
| 向量库 | Postgres + pgvector（与 checkpointer 同库最省事） |
| 模板仓库 | S3 + git bundle，或直接 GitHub Private Org |
| Fork 机制 | sandbox 里 `git clone <template_repo>` 后继续让 agent 改 |
| 冷启动 | 预置 10-20 个官方模板（手工策展），再逐步积累用户项目 |

#### 反模式警告

- **不要无脑全量入库**：质量差的模板会污染检索结果。要么人工审核，要么设置高阈值（测试通过 + 部署成功 + 用户好评同时满足）
- **不要把"相似项目"藏起来不告诉用户**：黑箱 fork 会让用户困惑"怎么和我说的不太一样"。默认应该**公开展示候选模板，让用户选**
- **不要让 retrieval 结果污染 system prompt 固定段**：用临时 SystemMessage 注入，任务结束后清理，避免后续对话中模型一直念叨旧模板

---

*记录时间：2026-04-20*
