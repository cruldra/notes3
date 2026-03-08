# GitNexus 使用教程

> **GitNexus**: 零服务器代码智能引擎 —— 客户端知识图谱创建器，完全在浏览器中运行。

GitNexus 是一个强大的代码库分析工具，可以将任何 GitHub 仓库或 ZIP 文件转换为交互式知识图谱，并内置 Graph RAG Agent。它帮助 AI 编码代理更好地理解代码库结构，避免依赖遗漏和破坏性修改。

---

## 📚 目录

1. [什么是 GitNexus](#什么是-gitnexus)
2. [为什么需要 GitNexus](#为什么需要-gitnexus)
3. [两种使用方式](#两种使用方式)
4. [CLI + MCP 完整指南](#cli--mcp-完整指南)
5. [Web UI 使用指南](#web-ui-使用指南)
6. [核心功能详解](#核心功能详解)
7. [支持的编程语言](#支持的编程语言)
8. [实际使用场景](#实际使用场景)
9. [故障排除](#故障排除)

---

## 什么是 GitNexus

GitNexus 是一个**客户端代码知识图谱引擎**，它：

- 📊 将代码库索引为知识图谱（依赖、调用链、集群、执行流）
- 🔍 通过智能工具向 AI 代理暴露图谱数据
- 🤖 让 AI 编码代理不再遗漏依赖、破坏调用链、盲目编辑

**核心理念**：预先在索引阶段计算代码结构（聚类、追踪、评分），让工具在一次调用中返回完整的上下文。

### 传统方式 vs GitNexus

| 方面 | 传统方式 | GitNexus |
|------|---------|----------|
| 代码理解 | AI 通过多次查询自行探索 | 预计算的结构化响应 |
| 可靠性 | 可能遗漏上下文 | 上下文已在工具响应中 |
| Token 效率 | 需要 10+ 次查询链 | 单次查询完整答案 |
| 模型要求 | 需要大型模型 | 小型模型也能高效工作 |

---

## 为什么需要 GitNexus

### 编码代理面临的挑战

现代 AI 编码工具（Cursor、Claude Code、Windsurf 等）功能强大，但它们**并不真正了解你的代码库结构**：

```
❌ 问题场景：
1. AI 修改 UserService.validate()
2. 不知道有 47 个函数依赖其返回类型
3. 破坏性变更被提交
```

### GitNexus 如何解决

```
✅ GitNexus 解决方案：
1. 预计算代码库的知识图谱
2. 提供影响范围分析工具
3. AI 在修改前了解所有依赖关系
```

**主要优势**：
- **可靠性**：LLM 不会遗漏上下文，因为数据已在工具响应中
- **Token 效率**：无需 10 次查询链来理解一个函数
- **模型民主化**：工具承担了繁重工作，小型 LLM 也能胜任

---

## 两种使用方式

GitNexus 提供两种互补的使用方式：

### 方式对比

| 特性 | CLI + MCP | Web UI |
|------|-----------|--------|
| **用途** | 本地索引仓库，通过 MCP 连接 AI 代理 | 浏览器中的可视化图谱 + AI 对话 |
| **适用场景** | 日常开发（Cursor、Claude Code、Windsurf、OpenCode） | 快速探索、演示、一次性分析 |
| **规模** | 完整仓库，任意大小 | 受浏览器内存限制（约 5000 文件） |
| **安装** | `npm install -g gitnexus` | 无需安装 —— gitnexus.vercel.app |
| **存储** | KuzuDB 原生（快速、持久） | KuzuDB WASM（内存中，每会话） |
| **解析** | Tree-sitter 原生绑定 | Tree-sitter WASM |
| **隐私** | 完全本地，无网络 | 完全浏览器内，无服务器 |

### 桥接模式

运行 `gitnexus serve` 可以连接两种方式 —— Web UI 自动检测本地服务器，可浏览所有 CLI 索引的仓库，无需重新上传或重新索引。

---

## CLI + MCP 完整指南

### 安装

```bash
# 全局安装
npm install -g gitnexus

# 或使用 npx（无需安装）
npx gitnexus
```

### 快速开始

```bash
# 在仓库根目录执行索引
npx gitnexus analyze
```

这就完成了！该命令会：
- 索引代码库
- 安装代理技能
- 注册 Claude Code 钩子
- 创建 `AGENTS.md` / `CLAUDE.md` 上下文文件

### MCP 配置

#### 自动配置（推荐）

```bash
# 自动检测编辑器并写入正确的全局 MCP 配置
npx gitnexus setup
```

只需运行一次。

#### 手动配置

**Claude Code**（完整支持 —— MCP + 技能 + 钩子）：
```bash
claude mcp add gitnexus -- npx -y gitnexus@latest mcp
```

**Cursor**（`~/.cursor/mcp.json` —— 全局，适用于所有项目）：
```json
{
  "mcpServers": {
    "gitnexus": {
      "command": "npx",
      "args": ["-y", "gitnexus@latest", "mcp"]
    }
  }
}
```

**OpenCode**（`~/.config/opencode/config.json`）：
```json
{
  "mcp": {
    "gitnexus": {
      "command": "npx",
      "args": ["-y", "gitnexus@latest", "mcp"]
    }
  }
}
```

### CLI 命令参考

| 命令 | 说明 |
|------|------|
| `gitnexus setup` | 为编辑器配置 MCP（一次性） |
| `gitnexus analyze [path]` | 索引仓库（或更新过期索引） |
| `gitnexus analyze --force` | 强制完全重新索引 |
| `gitnexus analyze --skip-embeddings` | 跳过嵌入生成（更快） |
| `gitnexus mcp` | 启动 MCP 服务器（stdio）— 服务所有索引的仓库 |
| `gitnexus serve` | 启动本地 HTTP 服务器（多仓库）供 Web UI 连接 |
| `gitnexus list` | 列出所有索引的仓库 |
| `gitnexus status` | 显示当前仓库的索引状态 |
| `gitnexus clean` | 删除当前仓库的索引 |
| `gitnexus clean --all --force` | 删除所有索引 |
| `gitnexus wiki [path]` | 从知识图谱生成仓库 Wiki |
| `gitnexus wiki --model <model>` | 使用自定义 LLM 模型生成 Wiki |
| `gitnexus wiki --base-url <url>` | 使用自定义 LLM API 基础 URL |

### AI 代理获得的能力

#### 7 个 MCP 工具

| 工具 | 功能 | repo 参数 |
|------|------|-----------|
| `list_repos` | 发现所有索引的仓库 | — |
| `query` | 进程分组混合搜索（BM25 + 语义 + RRF） | 可选 |
| `context` | 360度符号视图 —— 分类引用、进程参与 | 可选 |
| `impact` | 影响范围分析，带深度分组和置信度 | 可选 |
| `detect_changes` | Git diff 影响 —— 将变更行映射到受影响进程 | 可选 |
| `rename` | 多文件协调重命名，带图谱 + 文本搜索 | 可选 |
| `cypher` | 原始 Cypher 图谱查询 | 可选 |

> 当只有一个仓库被索引时，`repo` 参数是可选的。有多个仓库时，需指定：`query({query: "auth", repo: "my-app"})`

#### MCP 资源

| 资源 | 用途 |
|------|------|
| `gitnexus://repos` | 列出所有索引的仓库（首先读取） |
| `gitnexus://repo/{name}/context` | 代码库统计、过期检查、可用工具 |
| `gitnexus://repo/{name}/clusters` | 所有功能集群及内聚分数 |
| `gitnexus://repo/{name}/cluster/{name}` | 集群成员和详情 |
| `gitnexus://repo/{name}/processes` | 所有执行流 |
| `gitnexus://repo/{name}/process/{name}` | 完整进程追踪及步骤 |
| `gitnexus://repo/{name}/schema` | Cypher 查询的图谱模式 |

#### MCP 提示

| 提示 | 功能 |
|------|------|
| `detect_impact` | 提交前变更分析 —— 范围、受影响进程、风险等级 |
| `generate_map` | 从知识图谱生成架构文档（含 Mermaid 图表） |

#### 代理技能

自动安装到 `.claude/skills/`：

1. **Exploring** —— 使用知识图谱导航不熟悉的代码
2. **Debugging** —— 通过调用链追踪 Bug
3. **Impact Analysis** —— 变更前分析影响范围
4. **Refactoring** —— 使用依赖映射规划安全重构

### 多仓库 MCP 架构

GitNexus 使用**全局注册表**，一个 MCP 服务器可以服务多个索引仓库。无需每个项目配置 MCP —— 一次设置，处处可用。

**工作流程**：
1. 每个 `gitnexus analyze` 将索引存储在仓库内的 `.gitnexus/`（可移植、gitignored）
2. 在 `~/.gitnexus/registry.json` 中注册指针
3. AI 代理启动时，MCP 服务器读取注册表并可以服务任何索引仓库
4. KuzuDB 连接在首次查询时延迟打开，5 分钟不活动后回收（最多 5 个并发）

---

## Web UI 使用指南

### 在线使用

直接访问：**[gitnexus.vercel.app](https://gitnexus.vercel.app)**

1. 拖放 GitHub 仓库 ZIP 文件
2. 或输入 GitHub 仓库 URL
3. 开始探索知识图谱

### 本地运行

```bash
git clone https://github.com/abhigyanpatwari/gitnexus.git
cd gitnexus/gitnexus-web
npm install
npm run dev
```

### Web UI 特点

- **完全客户端**：所有处理在浏览器中完成，使用 WebAssembly
- **Tree-sitter WASM**：解析代码语法
- **KuzuDB WASM**：内存中图谱数据库
- **transformers.js**：浏览器内嵌入生成（WebGPU/WASM）
- **Sigma.js + Graphology**：WebGL 图谱可视化

### 本地后端模式

运行 `gitnexus serve` 并本地打开 Web UI —— 它会自动检测服务器并显示所有 CLI 索引的仓库，支持完整的 AI 对话功能。无需重新上传或重新索引。

---

## 核心功能详解

### 1. 影响范围分析（Impact Analysis）

```javascript
// 分析 UserService 的上游依赖（哪些代码依赖于它）
impact({target: "UserService", direction: "upstream", minConfidence: 0.8})
```

**输出示例**：
```
TARGET: Class UserService (src/services/user.ts)

UPSTREAM (依赖于此项):
  Depth 1 (将破坏):
    handleLogin [CALLS 90%] -> src/api/auth.ts:45
    handleRegister [CALLS 90%] -> src/api/auth.ts:78
    UserController [CALLS 85%] -> src/controllers/user.ts:12
  Depth 2 (可能受影响):
    authRouter [IMPORTS] -> src/routes/auth.ts
```

**参数**：
- `maxDepth`: 最大深度
- `minConfidence`: 最小置信度
- `relationTypes`: 关系类型（`CALLS`, `IMPORTS`, `EXTENDS`, `IMPLEMENTS`）
- `includeTests`: 是否包含测试

### 2. 进程分组搜索（Process-Grouped Search）

```javascript
// 搜索与认证中间件相关的代码
query({query: "authentication middleware"})
```

**输出示例**：
```yaml
processes:
  - summary: "LoginFlow"
    priority: 0.042
    symbol_count: 4
    process_type: cross_community
    step_count: 7

process_symbols:
  - name: validateUser
    type: Function
    filePath: src/auth/validate.ts
    process_id: proc_login
    step_index: 2

definitions:
  - name: AuthConfig
    type: Interface
    filePath: src/types/auth.ts
```

### 3. 360度符号视图（Context）

```javascript
// 获取 validateUser 函数的完整上下文
context({name: "validateUser"})
```

**输出示例**：
```yaml
symbol:
  uid: "Function:validateUser"
  kind: Function
  filePath: src/auth/validate.ts
  startLine: 15

incoming:
  calls: [handleLogin, handleRegister, UserController]
  imports: [authRouter]

outgoing:
  calls: [checkPassword, createSession]

processes:
  - name: LoginFlow (step 2/7)
  - name: RegistrationFlow (step 3/5)
```

### 4. 变更检测（Detect Changes）

```javascript
// 检测 Git diff 的影响范围
detect_changes({scope: "all"})
```

**输出示例**：
```yaml
summary:
  changed_count: 12
  affected_count: 3
  changed_files: 4
  risk_level: medium

changed_symbols: [validateUser, AuthService, ...]
affected_processes: [LoginFlow, RegistrationFlow, ...]
```

### 5. 重命名（Rename）

```javascript
// 多文件协调重命名
rename({symbol_name: "validateUser", new_name: "verifyUser", dry_run: true})
```

**输出示例**：
```yaml
status: success
files_affected: 5
total_edits: 8
graph_edits: 6     (高置信度)
text_search_edits: 2  (需谨慎检查)
changes: [...]
```

### 6. Cypher 查询

```cypher
-- 查找调用认证函数的高置信度调用者
MATCH (c:Community {heuristicLabel: 'Authentication'})<-[:CodeRelation {type: 'MEMBER_OF'}]-(fn)
MATCH (caller)-[r:CodeRelation {type: 'CALLS'}]->(fn)
WHERE r.confidence > 0.8
RETURN caller.name, fn.name, r.confidence
ORDER BY r.confidence DESC
```

---

## 支持的编程语言

GitNexus 支持以下语言的代码分析：

- TypeScript
- JavaScript
- Python
- Java
- Kotlin
- C
- C++
- C#
- Go
- Rust
- PHP
- Swift

---

## 实际使用场景

### 场景 1：理解不熟悉的代码库

```bash
# 1. 索引仓库
gitnexus analyze

# 2. 询问 AI 代理：
# "帮我理解这个项目的架构"
# AI 会使用知识图谱生成完整的架构文档
```

### 场景 2：安全重构

```bash
# 1. 在修改前分析影响范围
# 询问 AI："如果重命名 UserService.validate()，会影响哪些代码？"
# AI 使用 impact 工具分析并展示所有依赖

# 2. 使用重命名工具
# "将 validateUser 重命名为 verifyUser"
# AI 使用 rename 工具协调多文件修改
```

### 场景 3：提交前检查

```bash
# 修改代码后，询问 AI：
# "检查我的变更会影响哪些部分"
# AI 使用 detect_changes 工具分析 Git diff 的影响范围
```

### 场景 4：Bug 追踪

```bash
# 发现 Bug 后，询问 AI：
# "追踪这个 Bug 的调用链"
# AI 使用 context 和 query 工具追踪从入口点到问题代码的完整路径
```

### 场景 5：生成项目文档

```bash
# 生成项目 Wiki
gitnexus wiki

# 使用自定义模型
gitnexus wiki --model gpt-4o

# 使用自定义 API
gitnexus wiki --base-url https://api.anthropic.com/v1
```

---

## 故障排除

### 索引失败

**问题**：`gitnexus analyze` 失败

**解决**：
```bash
# 强制重新索引
gitnexus analyze --force

# 跳过嵌入生成（更快，但搜索功能受限）
gitnexus analyze --skip-embeddings
```

### MCP 连接问题

**问题**：AI 代理无法连接到 GitNexus

**解决**：
1. 检查 MCP 配置是否正确
2. 确保 `gitnexus mcp` 可以正常启动
3. 查看编辑器 MCP 日志

### 内存问题

**问题**：Web UI 在大仓库上崩溃

**解决**：
- Web UI 适用于 ~5000 文件以下的仓库
- 对于大仓库，使用 CLI + MCP 模式
- 或使用本地后端模式：`gitnexus serve` + Web UI

### 索引过期

**问题**：代码已修改，但分析结果未更新

**解决**：
```bash
# 检查索引状态
gitnexus status

# 更新索引
gitnexus analyze
```

---

## 工作原理

GitNexus 通过多阶段索引管道构建完整的代码知识图谱：

1. **Structure** —— 遍历文件树，映射文件夹/文件关系
2. **Parsing** —— 使用 Tree-sitter AST 提取函数、类、方法、接口
3. **Resolution** —— 使用语言感知逻辑解析跨文件的导入和函数调用
4. **Clustering** —— 将相关符号分组为功能社区
5. **Processes** —— 从入口点追踪执行流通过调用链
6. **Search** —— 构建混合搜索索引实现快速检索

---

## 隐私与安全

- **CLI**：完全在本地运行，无网络调用。索引存储在 `.gitnexus/`（gitignored）。全局注册表在 `~/.gitnexus/` 仅存储路径和元数据。
- **Web**：完全在浏览器中运行。无代码上传到任何服务器。API 密钥仅存储在 localStorage 中。
- **开源**：可自行审计代码。

---

## 参考链接

- **GitHub**: https://github.com/abhigyanpatwari/GitNexus
- **Web UI**: https://gitnexus.vercel.app
- **NPM**: https://www.npmjs.com/package/gitnexus
- **Discord**: https://discord.gg/AAsRVT6fGb

---

## 技术栈

| 层级 | CLI | Web |
|------|-----|-----|
| **运行时** | Node.js (原生) | 浏览器 (WASM) |
| **解析** | Tree-sitter 原生绑定 | Tree-sitter WASM |
| **数据库** | KuzuDB 原生 | KuzuDB WASM |
| **嵌入** | HuggingFace transformers.js | transformers.js |
| **搜索** | BM25 + 语义 + RRF | BM25 + 语义 + RRF |
| **代理接口** | MCP (stdio) | LangChain ReAct Agent |
| **可视化** | — | Sigma.js + Graphology |
| **前端** | — | React 18, TypeScript, Vite, Tailwind v4 |

---

*最后更新：2025年3月*
