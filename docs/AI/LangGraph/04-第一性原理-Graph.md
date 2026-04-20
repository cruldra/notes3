# 第一性原理拆解：Graph（图）

> 从"Graph = 节点 + 边"这个天真理解出发，戳破它，推导出 Graph 本质上是 **Pregel 的声明式前端 + 编译器**。

## 为什么挑 Graph

你接触 LangGraph 的第一行代码多半是：

```python
graph = StateGraph(State)
graph.add_node("step1", fn1)
graph.add_edge("step1", "step2")
```

"图嘛，节点 + 边，我懂"。但越用你会发现越多裂缝：

- 明明是图，运行时怎么又叫 Pregel？
- **条件边** 到底是图结构的一部分，还是运行时决策？
- `START` / `END` 这俩节点好奇怪——为啥不自动推断头尾？
- 一个 compiled graph 怎么能 **作为另一个图的节点**？
- 为什么有 `graph.compile()` 这一步，**编译** 什么？

这些裂缝拼起来，其实在告诉你：**Graph 根本不是你以为的那种"图"**。

## 天真理解：Graph = 节点 + 边

```python
graph = StateGraph(State)
graph.add_node("parse", parse_fn)
graph.add_node("search", search_fn)
graph.add_edge("parse", "search")
```

看上去就是个 DAG 数据结构：一堆 node + 一堆 `(from, to)` tuple。计算机课 101。

直到……

## 杀伤力 1：Pregel 里根本没有"边"

执行引擎是 Pregel。Pregel 的调度模型是什么？

- 顶点（= 节点）每步检查有没有新消息
- 有就跑，没就 idle
- 顶点之间靠 **通道写入** 通信

**"边"在这个模型里完全不存在**。你翻遍 `pregel/` 源码找不到任何 `Edge` 类。

那你 `add_edge("A", "B")` 被编译成了什么？答案是：

> **"节点 B 订阅节点 A 的输出通道"**

边消失了，变成了 **通道订阅关系**。

**条件边** 更夸张：

```python
graph.add_conditional_edges("A", router_fn, {"ok": "B", "fail": "C"})
```

这在 Pregel 里不是一条边——而是："A 写入后运行 `router_fn`，根据返回值往不同的目标通道 write"。路由逻辑从"图的静态结构"变成 **"A 节点的动态 write 动作"**。

> **边是人类可视化思维的拟像。Pregel 运行时里只存在通道订阅。**

第一个必然推论：**Graph 不是一个数据结构，而是在通道系统之上的一层 DSL**。

## 杀伤力 2：`START` / `END` 的存在暴露了什么

如果图真的只是节点 + 边，起点和终点完全可以 **自动推断**（入度为 0 是起点，出度为 0 是终点）。但 LangGraph 强制你写：

```python
graph.add_edge(START, "parse")
graph.add_edge("confirm", END)
```

为什么？

因为 Pregel **根本没有"起点终点"的概念**。所有节点在调度循环里对等。但框架需要知道两件事：

- **从哪里读用户传入的初始 state？** → 被 `START` 指向的节点
- **什么时候返回最终结果？** → `END` 通道被写入时

所以 `START` / `END` **不是图论意义上的特殊节点，而是 Pregel 通道命名空间里的两个保留 channel**：

- `START` channel：用户 input 写入这里，触发下游节点
- `END` channel：写入这里意味着 "本次 invocation 完成，返回结果"

你声明 `add_edge(START, "parse")`，编译后变成 "parse 订阅 `START` 通道"。调用 `graph.invoke({"msg": "hi"})` 时，`{"msg": "hi"}` 被写入 `START` 通道，parse 被激活，循环开始。

> **`START` / `END` 不是图的头尾，是通道名。它们是 Graph DSL 和 Pregel 运行时之间的"适配器"。**

## 杀伤力 3：节点是函数吗？

```python
graph.add_node("parse", parse_fn)
```

表面上 "节点 = 函数"。但 `add_node` 的完整签名是：

```python
graph.add_node(
    "parse",
    parse_fn,
    input=PartialInputSchema,  # 只读 state 的部分字段
    retry=RetryPolicy(...),    # 失败重试策略
    cache_policy=...,          # 缓存策略
    defer=True,                # 延后到兄弟节点完成
)
```

节点不是裸函数。它是：

> **函数 + 输入 schema + 输出 schema + 执行策略**

编译时这堆东西被打包成一个 `PregelNode` 对象挂到通道上。函数本身只是其中一小部分。

这跟前一篇 State 拆解里 "state 不是 dict，是 channel 集合" 是 **同一个模式**：**用户写的看似是简单对象，底下都是带行为的运行时实体**。

## 杀伤力 4：Graph 不是数据结构，是编译器

现在所有零件对齐了，重新定义 Graph：

> **Graph = "一组声明 (节点 + 边 + schema)" + "一个把声明编译成 Pregel 的编译器"**

`StateGraph` 类本身 **只是一个 builder**。你调用 `.add_node()` / `.add_edge()` 时，它在内存里 **累积一张描述**。真正的魔法在 `.compile()`：

```python
compiled = state_graph.compile(checkpointer=...)
```

这一步做的事：

1. **解析 state schema**：每个字段看 `Annotated` 的 reducer，决定用哪种 channel
2. **翻译节点**：每个 `add_node` 变成 `PregelNode`（带 triggers、channels、writes、策略）
3. **翻译静态边**：`add_edge(A, B)` → "B 订阅 A 的输出通道"
4. **翻译条件边**：`add_conditional_edges` → A 节点出口的动态路由 writes
5. **处理 START/END**：注册为保留 channel
6. **组装 Pregel 实例**：打包成一个可执行的 `Pregel` 对象

**`compile()` 不是一个小方法，它是整个 Graph DSL 的"编译器后端"**。你能跑的是 Pregel，graph 只是那段你写起来更好看的 DSL。

## 杀伤力 5：一旦 Graph 是"编译器"，所有零散 feature 都自动合理化

| 现象 | 在"数据结构"视角下 | 在"编译器"视角下 |
|---|---|---|
| `@entrypoint` 函数式 API 也能跑 | 很突兀："没有图怎么跑？" | 自然：**另一个前端**编译到同一个 Pregel 后端 |
| compiled graph 能做子图 | 奇怪："图不是结构吗，怎么还能当节点？" | 自然：Pregel 实例是 Runnable，Runnable 本来就能当节点 |
| 同一个图 compile 多次配不同 checkpointer | 混乱："图被改了？" | 自然：编译是动作，builder 不变 |
| `graph.stream()` / `.batch()` / `.ainvoke()` | 奇怪："图有这些方法？" | 自然：compile 结果是 Pregel 实例，Pregel 是 Runnable |
| 条件路由能 **动态派发 `Send`** | 很费解：这是边吗？ | 自然：运行时 write 动作，完全不是图结构 |

所有"能力"本来看起来像是零散 feature，现在都从 **"Graph 是编译器"** 这个定义里自然流出。

## 重新定义：Graph 是什么

| 层 | Graph 在这层的形态 |
|---|---|
| **表面**（你写的代码） | `StateGraph` builder：声明节点、边、schema 的集合 |
| **编译阶段** | 编译器：读 builder 产物，翻译成 Pregel 配置 |
| **运行时** | `Pregel` 实例（Runnable），跑 BSP 超步循环 |

**Graph ≠ 图。Graph = DSL + 编译器，最终产物是一个 Pregel 实例。**

这是一个经典的 **DSL + VM** 架构——和 LLVM（C → IR → 机器码）、SQL（query → plan → execution）、React（JSX → VDOM → DOM）是同一套思想。

## 反过来看源码

带这个定义打开 `libs/langgraph/langgraph/graph/state.py`：

- `StateGraph.__init__` 只接收 schema，不碰执行逻辑
- `add_node` / `add_edge` / `add_conditional_edges` 都只是 append 声明到 builder
- **`.compile()` 是整个文件最长、最关键的方法** —— 就是编译器后端
- 返回 `CompiledStateGraph`（继承自 `Pregel`）

再看 `libs/langgraph/langgraph/pregel/main.py` 的 `Pregel`：

- 它不认识 `add_edge`、`START`、`END` 这些词
- 它只认通道 / 节点 / triggers / writes
- 它是编译产物的消费者

**"Graph → compile → Pregel"这个流水线，对应着"面向人类的声明式语法" ↔ "面向机器的高效执行模型"的分离。** 一旦你看到这层分离，几乎所有"为什么要 compile"、"为什么 START 要显式声明"之类的疑问都自动消散。

## 收尾：这次拆解的产出

从 **"图就是节点 + 边"** 出发，戳破之后得到：

1. **边不存在于运行时**，编译成"通道订阅"
2. **`START` / `END` 不是图的头尾**，是保留 channel，是 DSL ↔ Pregel 的适配
3. **节点不是函数**，是"函数 + schema + 执行策略"的打包
4. **Graph 不是数据结构，是 DSL + 编译器**，产物是 Pregel 实例
5. **子图、`@entrypoint`、多次 compile、`graph.stream()`** 等能力都从"Graph 是编译器"这个定义自然流出

**关键洞察**：LangGraph 的 `graph/` 和 `pregel/` 两层，本质上是经典的 **DSL + VM 架构**。你写的是一种语言，跑的是另一种东西。这个分层让人类写得轻松、机器跑得高效。
