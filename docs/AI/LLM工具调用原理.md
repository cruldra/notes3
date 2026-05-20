---
title: LLM 工具调用底层原理
description: 拆解 LLM 工具调用从 token 生成到工具实际执行的完整调用栈, 澄清"模型训练 / vendor SDK / agent framework / runtime 缝合层" 四层职责, 并以 luffy-agent-platform code-agent 双轨读文件工具为例说明分流机制
keywords: [LLM, tool calling, function calling, agent, ReAct, deepagents, langgraph, MCP]
---

# LLM 工具调用底层原理

## 一个常见误解

> "LLM 推理本身只会吐文本 token。所谓 '工具调用能力' 是不是指 OpenAI / Anthropic
> 在做 HTTP SDK 时约定了一套协议 — 模型吐出 `<tool_call .../>` 之类的标签就表示
> 要调用外部工具，然后由用 SDK 的上层 agent 框架实现这个协议?"

**大方向对，两个细节要矫正：**

1. "协议" 不是 SDK 单方面约定的，是 vendor 在 **post-training (SFT + RLHF/DPO)**
   阶段就把"什么时候调工具 + 用什么格式表达调用"喂进了**模型权重**。模型自己
   已经会在合适时机吐出符合该格式的 token 序列，不是 SDK 强行劫持。
2. "上层 agent 框架"其实是**两层**: vendor SDK 只做协议反序列化, agent framework
   才负责实际执行 + ReAct loop + 回喂结果。

下面把完整调用栈拆开。

## 四层职责拆分

```
┌─────────────────────────────────────────────────────────────────────┐
│  ① LLM 推理服务 (OpenAI / Anthropic / 自托管 vLLM 等 HTTP API)      │
│     - 输入: messages + tools (JSON schema) + system_prompt          │
│     - 输出: 原始 token 流, 含训练好的工具调用格式                   │
│     - 是否调工具、调哪个、参数是什么, **完全由模型权重决定**        │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ raw response
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ② Vendor SDK 层 (openai-python / anthropic-python / 等)            │
│     - 解析 raw 文本里的 <tool_use> / 特殊 token / JSON 块           │
│     - surfacing 成结构化的 `tool_calls` / `content blocks` 字段     │
│     - 设置 stop_reason = "tool_use" 让调用方知道该执行工具          │
│     - **不执行任何工具**, 只做协议反序列化                          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ structured tool_calls
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ③ Agent Framework 层 (langgraph / deepagents / langchain / ADK)    │
│     - 拿到 tool_calls 后 dispatch 到本地 Python 函数 / MCP / HTTP   │
│     - 执行结果包成 `ToolMessage` / `tool_result` block              │
│     - 拼进 messages 再发下一轮请求                                  │
│     - 实现 ReAct loop, 直到模型不再发 tool_call 才 END              │
│     - 可插入 middleware (skills / memory / summarization …)         │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ tool_name + parsed_args
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ④ Runtime 缝合层 (项目自己写的, agent_dsl runtime / MCP server …)  │
│     - 注入 LLM 看不到的运行时参数 (sandbox / auth_token / cwd …)    │
│     - 把工具的 JSON schema 入参映射到真实函数签名                   │
│     - 鉴权、计费、日志、限流                                        │
└─────────────────────────────────────────────────────────────────────┘
```

**关键边界**: 模型只对① 的输入有"知情权" — 工具列表 (name + description +
input_schema) 和 messages。④ 注入的 sandbox 句柄、auth_token 这类**对模型不可见**，
也防止模型试图通过提示词工程篡改这些值。

## 一轮 tool_call 的完整生命周期

```
  user                framework         vendor SDK         LLM API           tool
  ────                ─────────         ──────────         ───────           ────
   │ "读 a.py"          │                  │                 │                │
   ├───────────────────▶│                  │                 │                │
   │                    │ messages         │                 │                │
   │                    │  + tools schema  │                 │                │
   │                    ├─────────────────▶│ POST /messages  │                │
   │                    │                  ├────────────────▶│                │
   │                    │                  │                 │ 推理: 决定调用 │
   │                    │                  │                 │ fs_read(path=  │
   │                    │                  │                 │ "a.py")        │
   │                    │                  │   raw tokens:   │                │
   │                    │                  │   <tool_use     │                │
   │                    │                  │     name="fs_   │                │
   │                    │                  │     read" .../> │                │
   │                    │                  │◀────────────────┤                │
   │                    │                  │ parse →         │                │
   │                    │  ToolUseBlock(   │ stop_reason=    │                │
   │                    │   name="fs_read",│ "tool_use"      │                │
   │                    │   input={...})   │                 │                │
   │                    │◀─────────────────┤                 │                │
   │                    │ dispatch         │                 │                │
   │                    ├──────────────────┼─────────────────┼───────────────▶│
   │                    │                  │                 │                │ ④ 注入
   │                    │                  │                 │                │ sandbox
   │                    │  result          │                 │                │ 执行
   │                    │◀─────────────────┼─────────────────┼────────────────┤
   │                    │ ToolMessage      │                 │                │
   │                    │  appended to     │                 │                │
   │                    │  messages        │                 │                │
   │                    ├─────────────────▶│ POST /messages  │                │
   │                    │                  ├────────────────▶│                │
   │                    │                  │   raw tokens:   │                │
   │                    │                  │   "文件内容是…" │                │
   │                    │                  │◀────────────────┤                │
   │                    │ assistant text   │ stop_reason=    │                │
   │                    │◀─────────────────┤ "end_turn"      │                │
   │ 文本回答           │                  │                 │                │
   │◀───────────────────┤                  │                 │                │
```

## 各厂商的工具调用格式速查

LLM 实际吐到 raw token 层的格式各家不同，SDK 把它们统一成结构化对象：

| 厂家 / 模型 | LLM 原始格式 | 备注 |
|---|---|---|
| Anthropic Claude | `<tool_use name="..." input="...">...</tool_use>` (XML 风格) | SDK 暴露 `content blocks` 列表，类型有 `text` / `tool_use` |
| OpenAI GPT-4/5 系 | 特殊 token 包裹的 JSON (raw 不公开 schema) | SDK 直接给 `message.tool_calls[]` |
| Qwen / DeepSeek / Llama-3 等开源 | `<tool_call>{"name":"...", "arguments":{...}}</tool_call>` | 各家略有差异，要看具体 chat_template |
| ReAct 风格旧模型 | 纯文本 `Action: foo\nAction Input: {...}` | 需 framework 自己写 parser |

**模型怎么知道有哪些工具可用?** 调用 API 时，SDK 会把 `tools` 列表 (每个工具的
name + description + JSON Schema) 拼进 system_prompt 或独立的 tools 字段, 模型据此
决定是否调用 + 调哪个 + 参数怎么填。一旦输出 tool_use 特殊 token，vendor API
端做后处理 — HTTP response 的 stop_reason 直接是 `tool_use`，raw 格式不会泄露
到普通文本字段里。

## 案例：luffy-agent-platform code-agent 双轨读文件工具

`code-agent` 这个智能体里有两套"读文件"工具，最容易把人绕进去 — 一套读
e2b 沙箱里的用户工作区，另一套读 backend 服务器上 `vendor/agents/code-agent/skills/`
目录。它俩是怎么不打架的？

### 工具命名空间从源头就分开

```
agent.yaml (code-agent)                           skills: [inferencesh-javascript-sdk]
  components.tools:                                       │
    - fs_read    (native, → vendor_tools.coding.fs.read)  │ 顶层声明触发
    - fs_write   (native)                                 ▼
    - fs_edit_diff                            agent-compiler-deepagents/skills.py
    - fs_ls                                   build_skills_middlewares()
    - fs_search                                 │
    - shell_run                                 ├─ _SkillsReadOnlyFsMiddleware
    - runtime_logs                              │    tools = [ls, read_file]
    - preview_start                             │   (deepagents 原生)
                                                │
                                                └─ _FilteredSkillsMiddleware
                                                     注入 skills_metadata 到
                                                     system_prompt
```

LLM 看到的工具表是两套**不同名**工具:

| 工具名 | 落点 | 来源 |
|---|---|---|
| `fs_read` / `fs_write` / `fs_edit_diff` / `fs_ls` / `fs_search` / `shell_run` … | **e2b 沙箱** (用户的 webapp 工作区) | agent.yaml 声明的 native tools |
| `ls` / `read_file` | **backend 主机** `vendor/agents/code-agent/skills/<slug>/` | deepagents `FilesystemMiddleware` |

名字不重叠 → 编译期 lint (AG013) 也不会让它们撞名。

### 两条通道的绑定方式完全不同

```
                  LLM 节点 (deepagents agent)
                          │
            ┌─────────────┴──────────────┐
            ▼                            ▼
  ┌──────────────────┐         ┌─────────────────────┐
  │ fs_read 工具      │         │ read_file 工具       │
  │ (native ref)      │         │ (deepagents 内置)    │
  └─────────┬────────┘         └──────────┬──────────┘
            │ runtime 缝合层注入            │ 闭包绑定到
            │ sandbox = get_session_       │ FilesystemBackend
            │   resources().sandbox         │   root_dir=skills_root
            │                               │   virtual_mode=True
            ▼                               ▼
  ┌──────────────────┐         ┌─────────────────────┐
  │ e2b 远端容器      │         │ backend 主机磁盘     │
  │ sandbox IPC:      │         │ Path.resolve() + 必须│
  │ download_file /   │         │ relative_to(root_dir)│
  │ exec              │         │ ".." / "~" 直接拒绝  │
  └──────────────────┘         └─────────────────────┘
```

**关键: 两套工具背后是两个完全不同的 backend 对象**, 从代码层面就不可能"读
着 skills 跑到沙箱里去"，反之亦然 — 它们闭包的是不同实例。

### 为什么不会越权 / 错乱

1. **工具名不同** → LLM 选错工具最多读不到东西，不会走错通道
2. **`virtual_mode=True`** → deepagents `FilesystemBackend._resolve_path` 把任意
   路径 (含 `/etc/passwd`) 当虚拟绝对路径强制拼到 `root_dir` 下，出根 ValueError
3. **read-only 裁剪** → `_SkillsReadOnlyFsMiddleware.__init__` 把 `self.tools` 收缩到
   `[ls, read_file]`，原生 `FilesystemMiddleware` 自带的 `write_file` / `edit_file` /
   `execute` / `glob` / `grep` 全删
4. **白名单** → `_FilteredSkillsMiddleware.before_agent` 每轮按 effective slug 重算
   `skills_metadata` 覆写 state，多 LLM 节点共享 state 也不会串台
5. **slug 校验** → agent_dsl lint 拒绝带路径分隔符的 slug，堵掉
   `skills: ["../../etc"]` 这种逃逸 implicit mapping 的小聪明

### 给 LLM 的方向引导

工具名不同 + system_prompt 提示词分工是软引导:

- skills middleware 在 system_prompt 注入 skill 的具体虚拟路径
  (`/inferencesh-javascript-sdk/SKILL.md`) 并指示用 `read_file` 读
- agent.yaml 自己的 system_prompt 教 LLM 用 `fs_read` 读用户工作区

硬保障还是工具的 backend 绑定 — 哪怕 LLM 提示词错乱把工作区路径喂给
`read_file`, 最多在 skills_root 下找不到文件返回 not found，**绝不会跨越**。

## 常见误解 / 边界

### Q1: 模型不听话直接输出 `<tool_use>` 文本怎么办?

主流闭源模型 (Claude, GPT) 不会 — vendor API 端做后处理, 模型一旦吐出
tool-call 特殊 token, HTTP response 的 `stop_reason` 直接是 `tool_use`,
普通文本字段里不会泄露 raw 格式。

**自托管开源模型** (Qwen, Llama 等) 才需要 framework 自己用正则/parser 兜底,
也是为什么 vLLM / TGI 这类推理引擎会暴露 `--tool-call-parser` 参数。

### Q2: tools 列表怎么传给 LLM?

每次请求都要带 — LLM 是**无状态的**, 它不记得上一次你声明过哪些工具。
agent framework 会把当前节点应该暴露的工具列表 (拼好 JSON Schema) 跟 messages
一起发上去。

这也是为什么 prompt cache 很关键 — Anthropic 把 tools 列表放在 cache_control
前缀里, 多轮对话能省 90%+ token。

### Q3: 工具 schema 太复杂模型搞不定?

模型对工具 schema 的理解能力跟训练数据强相关。深嵌套 / 联合类型 / 复杂枚
举的工具，弱模型容易乱填参数。实践经验:

- 工具单一职责: 一个工具只做一件事，参数尽量扁平
- description 用自然语言把"什么时候该调"讲清楚，比 schema 本身还重要
- 必填参数尽量少, 用默认值兜底
- 别让 LLM 自己拼 sandbox_id / auth_token 之类的运行时参数 — 让 ④ runtime
  缝合层注入

### Q4: MCP 跟 function calling 是什么关系?

MCP (Model Context Protocol) 是**工具传输协议**, 跟"模型怎么吐工具调用 token"
是正交的:

- function calling = ① + ② 这层 (模型怎么表达"我要调工具")
- MCP = ③ ↔ 工具实现之间的**远程调用协议** (类似 LSP 之于编辑器)

framework 可以把 MCP server 暴露的工具列表当作普通 tools 注入 LLM, 模型用
function calling 吐 tool_call, framework 通过 MCP stdio/http 转发执行。MCP 解
决的是"工具实现可插拔 + 跨进程/跨语言", 不解决"模型怎么知道要调工具"。

## 总结

| 层 | 职责 | 谁实现的 |
|---|---|---|
| ① 模型 | 训练时学会工具调用格式, 推理时吐 tool_use token | vendor 训练团队 |
| ② SDK | 把 raw token 解析成结构化 tool_calls | vendor SDK |
| ③ Framework | 执行工具 + ReAct loop + 拼回消息 | langgraph / deepagents / 等 |
| ④ Runtime 缝合 | 注入隐藏参数 + 鉴权计费 + 工具实现可插拔 | 你自己 (agent_dsl / vendor_tools) |

"协议是训练好的、SDK 负责解析、framework 负责执行+回喂、runtime 缝合层
负责注入隐藏参数" — 把这四层职责想清楚, 再去看 deepagents middleware /
MCP / 自己写 tool wrapper 就不会迷路了。
