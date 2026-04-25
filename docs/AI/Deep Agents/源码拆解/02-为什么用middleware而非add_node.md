# 为什么 deepagents 用 middleware/subagent 扩展，而不是 `StateGraph.add_node`？

> 承接前一篇《deepagents-create_deep_agent源码拆解》的疑问：既然 langgraph 提供了 `add_node` / `add_edge` 这套标准扩展机制，为什么 deepagents 不用？
>
> 本文从第一性原理推导这个架构选型。

---

## 一、先设想：用 `add_node` 扩展会长什么样

假设要给一个基础 ReAct agent 加以下能力：

1. Todo 追踪
2. 文件系统工具
3. 子智能体调用
4. 权限检查
5. 人工批准
6. Anthropic prompt caching

### 用 `add_node` 的做法

每个能力 = **一个节点 + 若干条边**：

```
START → agent_node
  ├─ 条件边 → todo_update_node   → agent_node
  ├─ 条件边 → file_tool_node     → agent_node
  ├─ 条件边 → subagent_A_node    → agent_node
  ├─ 条件边 → subagent_B_node    → agent_node
  ├─ 条件边 → permission_node    → ...(允许→工具；拒绝→END)
  ├─ 条件边 → human_review_node  → interrupt → agent_node
  └─ 条件边 → END
```

问题立刻来了：

### 问题 1：拓扑爆炸

每加一个能力，就要新增节点 + 条件边，并且**已存在的条件边要重写**（因为路由分支变了）。N 个能力 → 条件边决策函数里可能要处理 2^N 个组合。

### 问题 2：拓扑是静态的，但 agent 能力本质上是动态选择

ReAct agent 的灵魂是 **"LLM 根据上下文决定下一步调用哪个工具"**。这个决策天然属于**运行时**。

如果你把每个能力都做成节点，就意味着要在图层面**复刻一份"LLM 已经做过的选择"**——条件边函数得读 `messages[-1].tool_calls[0].name`，然后路由到对应节点。

这是一层**冗余翻译**：工具调用本身就是 LLM → 工具名 → 执行的动态映射，为什么还要再做一次"工具名 → 节点"的静态映射？

### 问题 3：横切关注点放不进"节点"这个盒子

"在每次 LLM 调用前加 prompt caching"、"在每次工具执行前检查权限"、"给所有输出加总结"——这些是**横切的**（cross-cutting concerns）。它们不是"流程中的一步"，而是**包裹在每一步外面的装饰**。

节点是流程单位，装饰要用节点表达就得：
- 在 `agent_node` 前插一个 `caching_node`
- 在 `tool_node` 前插一个 `permission_node`
- 在 `tool_node` 后插一个 `summarize_node`
- ……

这本质上是用**粗粒度的"节点"去模拟细粒度的 AOP**，图会被这些装饰节点挤得密密麻麻，且每个装饰都要单独写边。

### 问题 4：复用性差

同一个能力（比如"文件系统工具"）在主 agent 和子 agent 中都要用。用 `add_node` 就得把那套节点 + 边**在每个图里重新接一遍**。

---

## 二、middleware 为什么能解决这些问题

### 洞察：图的**骨架不变**，变的是**每一步的行为和状态**

主图永远是：

```
agent_node ⇄ tool_node
```

所有的扩展都**不改拓扑**，而是通过 hook 插入到 `agent_node` 和 `tool_node` 的执行过程中：

- `wrap_model_call` → 改 system prompt、加缓存、插入记忆
- `wrap_tool_execution` → 检查权限、记录 metadata、格式修复
- `@annotated` 字段 → 扩展 state schema

这对应了**装饰器模式 / AOP** 的思想：**不改变对象的类型，而是在其方法调用的前后嵌入行为**。

### 四个问题一起解了

| 问题 | middleware 如何解决 |
|------|--------------------|
| 拓扑爆炸 | 图不变，永远是两节点 |
| 静态拓扑 vs 动态决策 | 路由靠 LLM 选工具，图不管；middleware 只管"行为装饰" |
| 横切关注点放不进节点 | middleware 本身就是横切抽象（hook 是其第一性原理） |
| 复用性差 | middleware 是对象，主 agent 和子 agent 的列表里都能塞进去 |

---

## 三、为什么子智能体不做成"子图节点"，而是做成 `task` 工具

这个选择更有意思，是第一性原理的最精彩处。

### 如果做成子图节点会怎样

```
agent_node → 条件边(LLM 说要调 SubAgentA?) → SubAgentA_subgraph → agent_node
           → 条件边(LLM 说要调 SubAgentB?) → SubAgentB_subgraph → agent_node
           → ...（N 个子智能体 N 个分支）
```

问题：
1. **用户添加/删除子智能体要改图**——开发体验差
2. **条件边要解析 LLM 输出再路由**——就是前面说的"冗余翻译"
3. **子图状态和父图状态的合并语义要特殊处理**——条件边机制不擅长这个

### 做成 `task` 工具的优雅之处

`task(description, subagent_type="...")` 就是一个**普通工具**。于是：

1. **LLM 天然知道怎么选**——它看工具列表（其中有 `task`，schema 里 `subagent_type` 是个枚举），按常规工具调用语义就能选到对应子智能体
2. **路由不需要图——LLM 就是路由器**：`task` 工具内部查表 `subagent_graphs[subagent_type]` 就完成了"分发"
3. **状态合并有明确 API**：子图 `invoke` 返回 → `Command(update={...})` 回写父 state + 生成 `ToolMessage`。这是 langgraph 推崇的动态状态更新机制
4. **子智能体独立编译**：可以单独测试、单独 `invoke`，不依赖父图

### 背后的哲学

**把"编排"的责任交给模型。**

传统图结构是"程序员定义控制流"，而 agent 的本质是"模型定义控制流"。
- 子图节点方案 = **用程序员视角**建模（"我来决定什么时候调哪个子 agent"）
- 工具方案     = **用模型视角**建模（"模型自己通过 tool-calling 决定"）

---

## 四、Web 类比（帮记忆）

| 架构 | Web 类比 |
|------|---------|
| `add_node` 堆能力 | 每加一个功能就新建一条 URL 路由，所有跨域功能（auth、logging）也都做成独立 endpoint |
| middleware 扩展 | Express.js / Django middleware 链：请求穿过一串装饰器，业务路由保持干净 |
| 子智能体 = 子图节点 | 所有子服务都嵌入主进程，编译期连线 |
| 子智能体 = `task` 工具 | 微服务架构，父服务通过 RPC 调用子服务，松耦合、独立部署 |

---

## 五、诚实地说——这个选择的代价

没有免费午餐。middleware 方案也有痛点：

1. **执行顺序是隐式的**：`deepagent_middleware` 列表的顺序决定 wrap 嵌套顺序（外层 wrap 内层），调试时不如图可视化直观
2. **hook 模型不如节点直白**：新手理解"为什么 `write_file` 会改 `files_metadata` 状态"需要先理解 middleware 机制
3. **失去图的静态可视化**：你 `draw_mermaid()` 出来永远是 `START → agent → tools → END` 四个节点，看不到"这个 agent 有 Filesystem 和 Subagent 能力"这种语义信息
4. **LLM 驱动的路由 = 非确定性**：你没法用"先走 A 再走 B"这种强约束，只能用 system prompt "劝"模型

---

## 六、什么时候该用哪种

| 场景 | 推荐 |
|------|------|
| **通用型 agent**（用户输入什么都要能接）、能力频繁增删 | middleware + subagent-as-tool（deepagents 的选择） |
| **确定性工作流**（"审单 → 风控 → 放款"流程固定） | `StateGraph.add_node` + 条件边 |
| **面向特定领域的 agent**，能力集稳定、流程需要可视化审计 | 混合：主流程用 add_node，能力装饰用 middleware |

---

## 七、一句话总结

> **deepagents 的架构判断是："agent 能力是运行时的横切装饰，不是编译时的流程节点。"**
>
> 在通用 agent 场景下这是对的；在确定性工作流场景下这是错的。选对工具用对地方。

---

## 关联笔记

- [create_deep_agent 源码拆解](./01-create_deep_agent源码拆解.md) — 具体实现细节
- [第一性原理-Graph](../../LangGraph/Graph.md) — StateGraph 本身的设计
- [第一性原理-Send](../../LangGraph/06-第一性原理-Send.md) — langgraph 动态分发的另一种机制

---

*记录时间：2026-04-20*
