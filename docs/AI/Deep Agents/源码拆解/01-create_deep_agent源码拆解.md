# deepagents 中 `create_deep_agent` 构建 StateGraph 的拆解

> 目标：搞清楚 `deepagents` 库的入口函数 `create_deep_agent` 是如何在内部一步步装配出一张 LangGraph `StateGraph` 的。
>
> 源码路径：`/home/cruldra/Sources/deepagents/libs/deepagents/deepagents/`

---

## 核心结论（一句话）

**deepagents 不自己 `add_node/add_edge` 构建图**，它委托给 langchain 的 `create_agent()` 生成基础 ReAct 图，然后通过**中间件链（middleware stack）**注入能力。

> 图 = 一个基础 ReAct 循环 + 一堆 middleware（注入工具/状态/hook） + 子智能体作为可调用的 `task` 工具。

扩展点是 **middleware** 和 **subagent**，而不是 `StateGraph.add_node`。

---

## 1. 主入口：装配后交给 langchain

**位置**：`graph.py:217`（函数定义） → `graph.py:602-623`（返回）

```python
return create_agent(
    model,
    system_prompt=final_system_prompt,
    tools=_tools,
    middleware=deepagent_middleware,   # ← 核心：图由 middleware 驱动
    response_format=response_format,
    context_schema=context_schema,
    checkpointer=checkpointer,
    store=store,
    debug=debug,
    name=name,
    cache=cache,
).with_config({
    "recursion_limit": 9_999,
    "metadata": {"ls_integration": "deepagents", ...},
})
```

langchain `create_agent()` 内部生成的 `StateGraph` 拓扑是**经典 ReAct**：

```
START → agent_node (LLM 推理) → tool_node (工具执行) → agent_node → END
```

这是主图的**骨架**，所有 deepagents 的花样都在这条骨架上长出来。

---

## 2. Middleware 链：主智能体的真正"定义"

**位置**：`graph.py:539-586`

```python
deepagent_middleware = [
    TodoListMiddleware(),              # 注入 write_todos 工具 + todos 状态
    SkillsMiddleware(...),             # 可选：加载 skill 文件
    FilesystemMiddleware(...),         # 注入 ls/read_file/write_file/edit_file/glob/grep
    SubAgentMiddleware(...),           # 关键：注入 task 工具 → 调用子智能体
    create_summarization_middleware(...),  # 自动总结过长输出
    PatchToolCallsMiddleware(),        # 修正工具调用格式
    AsyncSubAgentMiddleware(...),      # 可选：后台异步子智能体
    *_profile.extra_middleware,        # 模型特定的 middleware (OpenAI/OpenRouter)
    _ToolExclusionMiddleware(...),
    AnthropicPromptCachingMiddleware(),
    MemoryMiddleware(...),             # 可选：加载 AGENTS.md 记忆
    HumanInTheLoopMiddleware(...),     # 可选：人工批准节点
    _PermissionMiddleware(...),        # 权限兜底（最后）
]
```

### middleware 如何工作

每个 middleware 实现以下任一 hook（或多个）：
- `wrap_model_call` — 包装 LLM 调用（改 system prompt、注入上下文等）
- `wrap_tool_execution` — 包装工具执行（权限检查、记录 metadata 等）
- `@annotated` 字段 — 扩展 state schema（加新字段 + reducer）

**这些 hook 不改图的拓扑**，只在 `agent_node` 和 `tool_node` 内部做装饰。

---

## 3. State schema：TypedDict 的层层叠加

基础：langchain 的 `AgentState`，包含：
- `messages: List[BaseMessage]`（reducer 为 `add_messages`，追加语义）

middleware 通过 TypedDict 扩展字段：

| 字段 | 来源 | reducer 逻辑 |
|------|------|-------------|
| `todos` | `TodoListMiddleware` | 覆盖式更新 |
| `files_metadata` | `FilesystemMiddleware` | 自定义 `_file_data_reducer`，支持删除 |
| `skills_metadata` | `SkillsMiddleware` | 覆盖式 |
| `memory_contents` | `MemoryMiddleware` | 覆盖式 |
| `structured_response` | `response_format` 启用时 | 覆盖式 |

---

## 4. 子智能体：图中套图（递归 StateGraph）

**位置**：`middleware/subagents.py:515-527`

```python
specs.append({
    "name": spec["name"],
    "description": spec["description"],
    "runnable": create_agent(     # ← 递归：子智能体自己也是一张 StateGraph
        model,
        system_prompt=spec["system_prompt"],
        tools=spec["tools"],
        middleware=middleware,    # 子智能体有自己的 middleware stack
        name=spec["name"],
    ),
})
```

### `task` 工具：把子智能体包装成工具

**位置**：`middleware/subagents.py:363-376`

```python
def task(description: str, subagent_type: str, runtime: ToolRuntime) -> str | Command:
    subagent = subagent_graphs[subagent_type]
    # 1. 过滤出可以传递给子图的 state 字段（隔离）
    subagent_state = {
        k: v for k, v in runtime.state.items()
        if k not in _EXCLUDED_STATE_KEYS
    }
    # 2. 用 task 描述作为子图的初始消息
    subagent_state["messages"] = [HumanMessage(content=description)]
    # 3. 同步调用子图（subagent.invoke 也是 CompiledStateGraph.invoke）
    result = subagent.invoke(subagent_state)
    # 4. 返回 Command，把选择性字段合回父图，并生成一条 ToolMessage
    return _return_command_with_state_update(result, runtime.tool_call_id)
```

**关键特性**：
- 每个子智能体是**独立编译的 `CompiledStateGraph`**
- `_EXCLUDED_STATE_KEYS` 保证父/子 state 互不污染
- 使用 `langgraph.types.Command` 机制把子图结果合回父图，这是动态路由 + 状态更新的正确姿势

### 默认 general-purpose 子智能体

**位置**：`graph.py:430-461`

如果用户没提供名为 `"general-purpose"` 的子智能体，自动补一个：

```python
general_purpose_spec: SubAgent = {
    **GENERAL_PURPOSE_SUBAGENT,
    "model": model,
    "tools": _tools or [],
    "middleware": gp_middleware,
}
```

---

## 5. Filesystem 工具的注入方式

**位置**：`middleware/filesystem.py`

通过 `wrap_tool_execution` hook 注入一组工具：

- `ls` / `read_file` / `write_file` / `edit_file` / `glob` / `grep`
- 若 backend 支持 `SandboxBackendProtocol`，额外加 `execute`

执行工具时修改 `files_metadata` 状态，使后续工具能感知历史操作。

---

## 6. 最终拓扑结构

```
MAIN AGENT (CompiledStateGraph):

  START
    │
    ▼
  agent_node ───────────── LLM 调用（所有 middleware.wrap_model_call 生效）
    │
    ▼
  tool_node  ───────────── 工具执行（wrap_tool_execution 生效）
    │   │
    │   └── 若调用 task(...) 工具：
    │        ├─ SubAgent A       （自己的 StateGraph）
    │        ├─ SubAgent B       （自己的 StateGraph）
    │        └─ general-purpose  （默认 StateGraph）
    │            └─ 内部同样是 agent_node ⇄ tool_node 循环
    │        子图执行完 → Command 更新父 state → ToolMessage → 回父 agent_node
    │
    └─> 回到 agent_node（继续推理）
    │
    ▼
   END
```

---

## 7. 编译与运行时配置

```python
.with_config({
    "recursion_limit": 9_999,  # 允许深层递归（子智能体可能链式调用）
    "metadata": {"ls_integration": "deepagents", "versions": {"deepagents": __version__}}
})
```

返回的是 `CompiledStateGraph`，可直接 `.invoke()` / `.stream()` / `.ainvoke()`。

---

## 设计上的精髓

1. **"图拓扑固定，能力靠插拔"**：所有 deepagents 的特性（todo 追踪、文件系统、技能、权限、人工干预、子智能体）都是 middleware，主图永远是那张 ReAct 图。
2. **子智能体 = 可调用的子图**：通过 `task` 工具把子图伪装成普通工具，主 agent 的"路由到哪个子智能体"其实是 **LLM 的工具选择**，不是图层面的条件边。这让编排能力下放给了模型本身。
3. **`Command` 作为跨图状态桥梁**：子图 `invoke` 的结果通过 `langgraph.types.Command` 对象把部分字段回写父图 state，同时生成 ToolMessage，这是 langgraph 0.2+ 推崇的动态更新模式。
4. **多层 state 隔离**：`_EXCLUDED_STATE_KEYS` 过滤确保子智能体不会继承或污染父级的 todos、files_metadata 等。

---

## 关键文件速查

| 文件 | 作用 |
|------|------|
| `deepagents/graph.py:217` | `create_deep_agent` 入口 |
| `deepagents/graph.py:539-586` | 主智能体 middleware 链装配 |
| `deepagents/graph.py:430-461` | 默认 general-purpose 子智能体 |
| `deepagents/middleware/subagents.py:403-530` | `SubAgentMiddleware` 核心 |
| `deepagents/middleware/subagents.py:363-376` | `task` 工具实现 |
| `deepagents/middleware/filesystem.py` | 文件工具注入 |
| `deepagents/middleware/todos.py` | todos 状态与工具 |

---

*记录时间：2026-04-20*
