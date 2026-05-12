---
sidebar_position: 8
---

**Claude Code 内置 50+ 指令，但绝大多数开发者只会反复使用其中 3~5 个。** 这份速查表把斜杠命令、CLI 启动标志、快捷键和典型工作流整理到一起，便于按场景快速翻查。

> 命令会随版本演进而变动，遇到差异以 `/help` 实际输出为准。表格中的部分高阶命令（如 `/btw`、`/fast`、`/remote-control` 等）来自社区文章梳理，可能依赖特定版本或插件，请在使用前先用 `/help` 核对是否存在于你的环境中。

## 一、核心斜杠命令（每日必用 Top 10）

| 命令 | 作用 | 使用场景 |
|------|------|---------|
| `/init` | 创建 `CLAUDE.md`，赋予项目长期记忆 | 进入新项目首次配置 |
| `/compact` | 压缩上下文，精简对话历史 | 上下文占用 70–80% 时主动执行 |
| `/clear` | 清空全部对话历史 | 切换到完全不同的任务时 |
| `/model` | 切换模型（Sonnet / Opus / Haiku） | 根据任务复杂度选模型 |
| `/cost` | 实时查看 Token 消耗与费用 | 监控本次会话花销 |
| `/context` | 查看当前上下文占用百分比 | Claude "变笨" 时排查 |
| `/diff` | 查看刚才修改的代码差异 | 提交前代码自审 |
| `/help` | 列出当前版本所有可用指令 | 忘记命令时的权威入口 |
| `/memory` | 不离开会话直接编辑 `CLAUDE.md` | 现场补充项目规则 |
| `/resume` | 加载并继续之前的会话 | 找回某段历史对话 |

### `/model` 用法示例

```bash
/model sonnet   # 切换到 Sonnet（日常利器，平衡）
/model opus     # 切换到 Opus（架构杀手，能力巅峰）
/model haiku    # 切换到 Haiku（极速，体力活专家）
```

### `/cost` 输出示例

```text
Session cost: $2.47
Input tokens: 48,392
Output tokens: 12,847
Model: claude-sonnet-4-20250514
```

### `/context` 输出示例

```text
Context usage: 67% (134,400 / 200,000 tokens)
```

### `/memory` 的快捷追加语法

在输入框直接以 `#` 开头写一行，会被自动追加到 `CLAUDE.md`：

```text
# Use async/await for all database queries
```

## 二、高阶斜杠命令

| 命令 | 作用 | 备注 |
|------|------|------|
| `/btw` | 不打断当前任务，临时插话提问 | 问完自动回到原任务 |
| `/fast` | 启用极速 API 模式 | 仅 Opus 系列可用 |
| `/plan` | 进入只读规划模式 | 看完方案确认后才动手，防事故 |
| `/todos` | 跨会话持久化任务清单 | 关闭重开不丢失 |
| `/simplify` | 多代理并行代码评审（安全 / 性能 / 规范） | 已取代旧的 `/review` |
| `/vim` | 输入框开启 Vim 键位 | 隐藏彩蛋 |
| `/remote-control` | 手机远程控制电脑里的 Claude | 通勤路上也能修 Bug |
| `/usage-report` | 生成月度使用分析报告 | 看 Token 都烧在哪 |
| `/export` | 导出当前会话归档 | 沉淀团队知识资产 |

## 三、CLI 启动标志

| 标志 | 作用 | 典型用法 |
|------|------|---------|
| `claude --print "..."` | 单次查询后立即退出 | 脚本里嵌入 Claude 调用 |
| `claude -c` | 接续当前目录上次的会话 | 一键续上工作 |
| `claude --agents '{...}'` | 启动时预设子代理 | 团队 / 项目固定代理配置 |
| `claude --dangerously-skip-permissions` | ⚠️ 跳过全部审批 | 仅限 Docker / CI 等受信任环境 |

### 一闪电战：单次查询

```bash
result=$(claude --print "Generate a random UUID")
echo $result
```

### 预设子代理启动

```bash
claude --agents '{
  "test-writer": {
    "role": "Write comprehensive Jest tests",
    "model": "claude-sonnet-4"
  }
}'
```

## 四、快捷键

| 快捷键 | 作用 |
|--------|------|
| `Shift + Tab` | 在 正常 / 自动接受 / 规划 三种模式间循环 |
| `Esc Esc` | 呼出回滚菜单（可选只回滚代码或只回滚对话） |
| `! <command>` | 在会话中直接执行 Bash，如 `! git status` |
| `@ <path>` | 文件路径自动补全 |
| `Ctrl + T` | 开关任务列表 |
| `# <内容>` | 输入框以 `#` 开头，自动追加到 `CLAUDE.md` |

## 五、CLAUDE.md 极简模板

```markdown
# CLAUDE.md

## Authentication
- Use JWT tokens, not sessions
- Store in httpOnly cookies

## Testing
- Write tests for all API endpoints
- Use Jest, not Mocha

## Error Handling
- Return structured errors: { error: string, code: number }
```

## 六、典型工作流：长达一天的复杂重构

| 阶段 | 命令组合 |
|------|---------|
| 进入规划 | `claude` → `Shift+Tab` 切到 Plan 模式 |
| 描述任务 | "将 Auth 模块从 session 改为 JWT，使用 bcrypt 加密密码" |
| 监控上下文 | `/context` → 达 70% 时 `/compact retain auth patterns and migration strategy` |
| 评审改动 | `/diff` → `/simplify` |
| 提交代码 | `! git add .` → `! git commit -m "feat: jwt migration"` |
| 归档沉淀 | `/export` 存为团队知识资产 |

## 七、记忆建议

不必一天背完 50 多个指令。**每周强制自己用熟一个新命令**，一个月就能把同行甩开一截。

优先固化为肌肉记忆的最小集合：

```text
/init   /compact   /diff   /model   /context
Shift+Tab   Esc Esc   ! git status   @ src/
```
