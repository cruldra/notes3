# GitNexus MCP 工具详解

> 完整指南：如何在 AI 编码代理中使用 GitNexus 的 7 个 MCP 工具

---

## 工具概览

| 工具 | 用途 | 典型场景 |
|------|------|----------|
| `list_repos` | 列出所有索引仓库 | 多仓库环境 |
| `query` | 智能代码搜索 | 查找相关代码 |
| `context` | 360度符号视图 | 理解函数/类的完整上下文 |
| `impact` | 影响范围分析 | 重构前评估 |
| `detect_changes` | Git diff 影响分析 | 提交前检查 |
| `rename` | 多文件重命名 | 安全重构 |
| `cypher` | 原始图谱查询 | 复杂自定义查询 |

---

## 1. list_repos - 仓库列表

### 功能
发现所有已索引的仓库。

### 使用场景
- 多仓库环境，需要选择目标仓库
- 检查哪些项目已被索引

### 示例

```javascript
// 无参数调用
list_repos()
```

**输出示例**：
```json
{
  "repos": [
    {
      "name": "my-app",
      "path": "/home/user/projects/my-app",
      "lastIndexed": "2025-03-07T10:30:00Z",
      "fileCount": 245,
      "symbolCount": 1234
    },
    {
      "name": "backend-api",
      "path": "/home/user/projects/backend-api",
      "lastIndexed": "2025-03-06T15:20:00Z",
      "fileCount": 189,
      "symbolCount": 876
    }
  ]
}
```

### 使用提示

AI 代理通常会自动调用此工具来发现可用仓库，你无需手动使用。

---

## 2. query - 智能代码搜索

### 功能
使用混合搜索（BM25 + 语义搜索 + RRF 融合）查找代码，并按进程分组。

### 使用场景
- 查找与特定功能相关的代码
- 发现代码的执行流程
- 理解功能模块的组成

### 参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `query` | string | ✅ | 搜索查询 |
| `repo` | string | 可选 | 目标仓库名称（单仓库时可省略） |
| `limit` | number | 可选 | 返回结果数量限制 |

### 示例

#### 基本搜索

```javascript
query({
  query: "authentication middleware",
  repo: "my-app"
})
```

**输出示例**：
```yaml
processes:
  - id: "proc_login"
    summary: "LoginFlow"
    priority: 0.042
    symbol_count: 4
    process_type: cross_community
    step_count: 7
    
  - id: "proc_register"
    summary: "RegistrationFlow"
    priority: 0.038
    symbol_count: 3
    process_type: single_community
    step_count: 5

process_symbols:
  - name: validateUser
    type: Function
    filePath: src/auth/validate.ts
    process_id: proc_login
    step_index: 2
    
  - name: createSession
    type: Function
    filePath: src/auth/session.ts
    process_id: proc_login
    step_index: 3

definitions:
  - name: AuthConfig
    type: Interface
    filePath: src/types/auth.ts
    
  - name: AuthMiddleware
    type: Class
    filePath: src/middleware/auth.ts
```

#### 限制结果数量

```javascript
query({
  query: "database connection",
  repo: "my-app",
  limit: 10
})
```

### 使用提示

- 使用自然语言描述，如 "user authentication flow" 而非 "auth"
- 结果按进程分组，帮助理解代码执行流程
- 关注 `process_symbols` 中的 `step_index` 了解执行顺序

---

## 3. context - 360度符号视图

### 功能
获取指定符号（函数、类、变量等）的完整上下文，包括所有引用、调用、所属进程等。

### 使用场景
- 深入理解某个函数的实现和用法
- 分析代码变更的影响范围
- 追踪代码依赖关系

### 参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `name` | string | ✅ | 符号名称 |
| `repo` | string | 可选 | 目标仓库名称 |
| `includeTests` | boolean | 可选 | 是否包含测试文件 |

### 示例

#### 函数上下文

```javascript
context({
  name: "validateUser",
  repo: "my-app"
})
```

**输出示例**：
```yaml
symbol:
  uid: "Function:validateUser"
  kind: Function
  name: validateUser
  filePath: src/auth/validate.ts
  startLine: 15
  endLine: 45
  signature: "async function validateUser(email: string, password: string): Promise<User>"

incoming:
  calls:
    - name: handleLogin
      filePath: src/api/auth.ts
      line: 45
      confidence: 0.95
    - name: handleRegister
      filePath: src/api/auth.ts
      line: 78
      confidence: 0.95
    - name: UserController.login
      filePath: src/controllers/user.ts
      line: 12
      confidence: 0.85
  imports:
    - filePath: src/routes/auth.ts
      line: 3

outgoing:
  calls:
    - name: checkPassword
      filePath: src/auth/password.ts
      line: 22
    - name: createSession
      filePath: src/auth/session.ts
      line: 10
    - name: logAuditEvent
      filePath: src/utils/audit.ts
      line: 56
  imports:
    - name: User
      filePath: src/types/user.ts
    - name: AuthError
      filePath: src/errors/auth.ts

processes:
  - name: LoginFlow
    id: proc_login
    step: 2
    totalSteps: 7
    description: "用户登录流程"
  - name: RegistrationFlow
    id: proc_register
    step: 3
    totalSteps: 5
    description: "用户注册流程"

clusters:
  - name: Authentication
    cohesion: 0.92
    members: [validateUser, checkPassword, createSession, ...]
```

#### 类上下文

```javascript
context({
  name: "UserService",
  includeTests: false
})
```

### 使用提示

- `incoming.calls` 显示哪些代码调用了此符号（上游依赖）
- `outgoing.calls` 显示此符号调用了哪些代码（下游依赖）
- `processes` 显示此符号参与哪些执行流程及在其中的位置
- `clusters` 显示此符号所属的功能集群

---

## 4. impact - 影响范围分析

### 功能
分析修改指定符号会影响哪些代码，支持上游（依赖此符号）和下游（此符号依赖）两个方向。

### 使用场景
- 重构前评估影响范围
- 修改前了解潜在风险
- 识别需要同时修改的相关代码

### 参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `target` | string | ✅ | 目标符号名称 |
| `direction` | string | 可选 | 方向：`upstream`（依赖此项）或 `downstream`（此项依赖） |
| `maxDepth` | number | 可选 | 最大深度（默认：3） |
| `minConfidence` | number | 可选 | 最小置信度（0-1，默认：0.7） |
| `relationTypes` | array | 可选 | 关系类型：`CALLS`, `IMPORTS`, `EXTENDS`, `IMPLEMENTS` |
| `includeTests` | boolean | 可选 | 是否包含测试文件 |
| `repo` | string | 可选 | 目标仓库名称 |

### 示例

#### 分析上游依赖（哪些代码依赖此项）

```javascript
impact({
  target: "UserService",
  direction: "upstream",
  maxDepth: 2,
  minConfidence: 0.8
})
```

**输出示例**：
```yaml
TARGET: Class UserService (src/services/user.ts)

UPSTREAM (what depends on this):
  Depth 1 (WILL BREAK):
    - symbol: handleLogin
      type: Function
      filePath: src/api/auth.ts:45
      relation: CALLS
      confidence: 0.95
      
    - symbol: handleRegister
      type: Function
      filePath: src/api/auth.ts:78
      relation: CALLS
      confidence: 0.95
      
    - symbol: UserController
      type: Class
      filePath: src/controllers/user.ts:12
      relation: CALLS
      confidence: 0.85

  Depth 2 (LIKELY AFFECTED):
    - symbol: authRouter
      type: Variable
      filePath: src/routes/auth.ts:15
      relation: IMPORTS
      confidence: 0.90
      
    - symbol: AuthMiddleware
      type: Class
      filePath: src/middleware/auth.ts:8
      relation: CALLS
      confidence: 0.75

SUMMARY:
  totalAffected: 5
  highConfidence: 4
  mediumConfidence: 1
  riskLevel: high
  
RECOMMENDATIONS:
  - 4 symbols at Depth 1 will BREAK if modified
  - Review all CALLS relationships carefully
  - Consider backward compatibility
```

#### 分析下游依赖（此项依赖哪些代码）

```javascript
impact({
  target: "UserService",
  direction: "downstream",
  maxDepth: 1
})
```

**输出示例**：
```yaml
TARGET: Class UserService (src/services/user.ts)

DOWNSTREAM (what this depends on):
  Depth 1:
    - symbol: Database
      type: Class
      filePath: src/db/index.ts:10
      relation: IMPORTS
      
    - symbol: User
      type: Interface
      filePath: src/types/user.ts:5
      relation: IMPORTS
      
    - symbol: Logger
      type: Class
      filePath: src/utils/logger.ts:12
      relation: IMPORTS
```

#### 指定关系类型

```javascript
impact({
  target: "BaseController",
  direction: "upstream",
  relationTypes: ["EXTENDS", "IMPLEMENTS"],
  maxDepth: 3
})
```

### 使用提示

- **上游分析**（`direction: upstream`）：修改前使用，了解会破坏哪些代码
- **下游分析**（`direction: downstream`）：了解此代码依赖哪些底层实现
- 始终关注 `riskLevel` 和 `RECOMMENDATIONS`
- 使用 `minConfidence` 过滤低置信度的结果

---

## 5. detect_changes - 变更检测

### 功能
分析 Git diff，识别变更的符号和受影响的进程。

### 使用场景
- 提交前评估变更影响
- 代码审查时快速理解修改范围
- 识别潜在风险区域

### 参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `scope` | string | 可选 | 范围：`all`（所有变更）、`staged`（暂存区）、`unstaged`（未暂存） |
| `repo` | string | 可选 | 目标仓库名称 |

### 示例

#### 检测所有变更

```javascript
detect_changes({
  scope: "all",
  repo: "my-app"
})
```

**输出示例**：
```yaml
summary:
  changed_count: 12
  changed_files: 4
  affected_count: 8
  affected_processes: 3
  risk_level: medium
  lines_added: 145
  lines_removed: 89

changed_symbols:
  - name: validateUser
    filePath: src/auth/validate.ts
    changeType: modified
    linesChanged: 15
    
  - name: AuthService
    filePath: src/services/auth.ts
    changeType: modified
    linesChanged: 42
    
  - name: createSession
    filePath: src/auth/session.ts
    changeType: added
    linesChanged: 28

affected_processes:
  - name: LoginFlow
    id: proc_login
    affectedSymbols: [validateUser, createSession]
    risk: high
    
  - name: RegistrationFlow
    id: proc_register
    affectedSymbols: [validateUser]
    risk: medium
    
  - name: PasswordResetFlow
    id: proc_reset
    affectedSymbols: [AuthService]
    risk: low

recommendations:
  - "LoginFlow 受到高风险影响，建议仔细测试"
  - "validateUser 在多个流程中使用，确保向后兼容"
  - "考虑更新相关测试用例"
```

#### 仅检测暂存区变更

```javascript
detect_changes({
  scope: "staged"
})
```

### 使用提示

- 在提交前运行，了解变更的完整影响
- 关注 `risk_level` 和 `recommendations`
- `affected_processes` 帮助你理解哪些业务流程受到影响

---

## 6. rename - 多文件重命名

### 功能
安全地在多个文件中重命名符号，协调图谱关系和文本搜索。

### 使用场景
- 安全重构变量、函数、类名
- 确保重命名不破坏代码
- 跨文件一致性修改

### 参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `symbol_name` | string | ✅ | 当前符号名称 |
| `new_name` | string | ✅ | 新符号名称 |
| `dry_run` | boolean | 可选 | 试运行模式（默认：true） |
| `repo` | string | 可选 | 目标仓库名称 |

### 示例

#### 试运行模式（推荐先用此模式）

```javascript
rename({
  symbol_name: "validateUser",
  new_name: "verifyUserCredentials",
  dry_run: true,
  repo: "my-app"
})
```

**输出示例**：
```yaml
status: success
dry_run: true

summary:
  files_affected: 5
  total_edits: 8
  graph_edits: 6
  text_search_edits: 2
  
edits:
  graph_based:
    - file: src/auth/validate.ts
      line: 15
      column: 16
      old: "async function validateUser("
      new: "async function verifyUserCredentials("
      confidence: 0.98
      
    - file: src/api/auth.ts
      line: 45
      column: 24
      old: "await validateUser("
      new: "await verifyUserCredentials("
      confidence: 0.95
      
    - file: src/api/auth.ts
      line: 78
      column: 24
      old: "await validateUser("
      new: "await verifyUserCredentials("
      confidence: 0.95
      
  text_based:
    - file: README.md
      line: 120
      old: "`validateUser()`"
      new: "`verifyUserCredentials()`"
      confidence: 0.75
      note: "文档引用，需人工确认"
      
    - file: tests/auth.test.ts
      line: 56
      old: "describe('validateUser',"
      new: "describe('verifyUserCredentials',"
      confidence: 0.70
      note: "测试描述，需人工确认"

recommendations:
  - "6 个图谱编辑为高置信度，可自动应用"
  - "2 个文本搜索编辑为中置信度，请仔细审查"
  - "建议运行测试确保重命名未破坏功能"
```

#### 执行重命名

```javascript
rename({
  symbol_name: "validateUser",
  new_name: "verifyUserCredentials",
  dry_run: false
})
```

### 使用提示

- **始终先使用 `dry_run: true`** 预览变更
- 区分 `graph_edits`（高置信度）和 `text_search_edits`（需审查）
- 重命名后运行测试验证

---

## 7. cypher - 原始图谱查询

### 功能
使用 Cypher 查询语言直接查询知识图谱。

### 使用场景
- 执行复杂的自定义查询
- 探索图谱数据
- 构建高级分析

### 参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `query` | string | ✅ | Cypher 查询语句 |
| `repo` | string | 可选 | 目标仓库名称 |

### 示例

#### 查找高置信度的函数调用

```javascript
cypher({
  query: `
    MATCH (caller:Function)-[r:CALLS]->(callee:Function)
    WHERE r.confidence > 0.9
    RETURN caller.name, callee.name, r.confidence
    ORDER BY r.confidence DESC
    LIMIT 20
  `,
  repo: "my-app"
})
```

#### 查找认证相关的所有符号

```javascript
cypher({
  query: `
    MATCH (c:Community {heuristicLabel: 'Authentication'})
          <-[:MEMBER_OF]-(fn:Function)
    RETURN fn.name, fn.filePath
    ORDER BY fn.name
  `
})
```

#### 查找孤儿函数（未被调用的函数）

```javascript
cypher({
  query: `
    MATCH (fn:Function)
    WHERE NOT (:Function)-[:CALLS]->(fn)
      AND NOT (:Class)-[:CALLS]->(fn)
    RETURN fn.name, fn.filePath
  `
})
```

#### 查找循环依赖

```javascript
cypher({
  query: `
    MATCH path = (a)-[:IMPORTS*3..5]->(a)
    RETURN [node in nodes(path) | node.name] as cycle
    LIMIT 10
  `
})
```

#### 统计文件依赖数

```javascript
cypher({
  query: `
    MATCH (f:File)
    OPTIONAL MATCH (f)-[:IMPORTS]->(dep:File)
    WITH f, count(dep) as depCount
    RETURN f.path, depCount
    ORDER BY depCount DESC
    LIMIT 20
  `
})
```

### 图谱模式参考

查看完整图谱模式：
```javascript
// 读取资源获取图谱模式
resource: gitnexus://repo/my-app/schema
```

### 常用节点类型

- `File` - 源代码文件
- `Function` - 函数
- `Class` - 类
- `Interface` - 接口
- `Variable` - 变量
- `Community` - 功能集群
- `Process` - 执行流程

### 常用关系类型

- `CALLS` - 函数调用关系
- `IMPORTS` - 导入关系
- `EXTENDS` - 继承关系
- `IMPLEMENTS` - 实现关系
- `MEMBER_OF` - 集群成员关系
- `PART_OF` - 流程步骤关系

### 使用提示

- Cypher 查询功能强大，但需要熟悉图谱结构
- 先用简单查询验证，再构建复杂查询
- 使用 `LIMIT` 避免返回过多结果

---

## 使用模式总结

### 模式 1：理解代码

```
1. query({query: "功能描述"}) → 查找相关代码
2. context({name: "找到的符号"}) → 深入理解
```

### 模式 2：安全重构

```
1. impact({target: "要修改的符号", direction: "upstream"}) → 评估影响
2. rename({symbol_name: "旧名", new_name: "新名", dry_run: true}) → 试运行
3. rename({symbol_name: "旧名", new_name: "新名", dry_run: false}) → 执行
```

### 模式 3：提交前检查

```
detect_changes({scope: "staged"}) → 了解变更影响
```

### 模式 4：复杂分析

```
cypher({query: "复杂的 Cypher 查询"}) → 自定义分析
```

---

## 最佳实践

1. **始终先用试运行模式**：重命名前使用 `dry_run: true`
2. **分步骤分析**：先 query 发现，再 context 深入，最后 impact 评估
3. **关注置信度**：使用 `minConfidence` 过滤低质量结果
4. **定期重新索引**：代码变更后运行 `gitnexus analyze`
5. **利用进程信息**：关注结果中的 `processes` 了解执行流程

---

*掌握这些工具，让你的 AI 编码代理真正理解你的代码库！*
