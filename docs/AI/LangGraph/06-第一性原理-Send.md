---
sidebar_position: 6
---
> 从"Send 是一种带数据的边"出发，戳破它，推导出 Send 是 **Pregel 消息语义的显式化**，是静态图撞上动态 map-reduce 时必然浮现的原语。

## 为什么挑 Send

`Send` 是 LangGraph 里最"不像图"的原语。用户第一次碰它往往会懵：

```python
from langgraph.types import Send

def router(state):
    return [Send("worker", {"item": x}) for x in state["items"]]

graph.add_conditional_edges("dispatcher", router)
```

"怎么回事，router 不是应该返回节点名吗？怎么返回一堆 **对象**？" "为什么一个 router 能返回多个 Send？"

这些困惑暴露了一个深刻的问题：**静态图模型和数据驱动的动态调度，本质上是冲突的**。Send 就是 LangGraph 对这个冲突的正面回应。拆它等于拆 "静态声明 vs 动态运行时" 的边界在哪里。

---

## 天真理解：Send = 一种带数据的边

"哦，普通边是 `(from, to)`，Send 多了个 payload，变成 `(from, to, data)`"。

听起来合理，但这个理解撑不了三秒。

---

## 杀伤力 1：静态图根本表达不了"动态数量"

假设你要做：**"state 里有 N 个 item，每个派给一个 worker 并行处理"**。

N 是运行时才知道的数字。你打算怎么写？

**尝试 1**：提前声明 N 个 worker 节点

```python
graph.add_node("worker_1", work_fn)
graph.add_node("worker_2", work_fn)
graph.add_node("worker_3", work_fn)
# ... N=?
```

N 可能是 1、可能是 1000。你预声明到 1000？那 1001 呢？显然死路。

**尝试 2**：`add_conditional_edges` 返回一个节点名列表

```python
def router(state):
    return ["worker"] * len(state["items"])
```

编译器能接受"去同一个节点 3 次"吗？就算接受，**每个 worker 需要不同的输入**（不同的 item），你怎么传？router 只能返回节点名，**不能附带数据**。

**尝试 3**：把所有 items 打包塞进 state，让 worker 自己挑

```python
def worker(state):
    for item in state["pending_items"]:
        process(item)
```

这就退化成串行处理了，失去了并发意义。

---

**撞墙的根本原因**：

> **图的边只能表达"去哪里"，不能携带"带什么数据"。但 map-reduce 场景下，"带什么数据"恰恰是每次派发的本质差异。**

静态图的抽象粒度不够。边没法装数据，节点没法动态实例化。

---

## 推导：需要一种"带负载的动态派发"原语

回到根本约束：我们要让 router 能说出这样的话：

> "针对 state 里的每个 item，我要求派一个 **独立执行** 的 `worker` 实例，**每个实例收到各自的 payload**。"

这三件事缺一不可：

1. **派发目标** —— 发给哪个节点
2. **独立实例** —— 同一节点可以并发多份
3. **独立 payload** —— 每份实例拿自己的数据

抽象成一个数据结构：`Send(node_name, payload)`。router 返回一个 `list[Send]`——想派几个就派几个，每个带自己的 payload。

这就是 `Send`：**把"派发 + 数据"一起打包的显式原语**。

---

## 杀伤力 2：多个 Send 去同一节点，节点实例化几次？

```python
return [
    Send("worker", {"item": "a"}),
    Send("worker", {"item": "b"}),
    Send("worker", {"item": "c"}),
]
```

普通图语义下，"worker" 是一个唯一的节点。但 Send 要求：**worker 要被独立实例化 3 次，每次收到不同的 payload，并发跑**。

这颠覆了一个暗含假设——**"节点是唯一的、一个 superstep 里只跑一次"**。Send 引入了一个新概念：

> **同一个节点声明，可以在一个超步里产生多个执行实例。每个实例是独立任务。**

Pregel 的 `prepare_next_tasks` 看到 Send 对象时，会为每个 Send 构造一个独立任务，而不是合并成一次节点执行。3 个 Send → 3 个并发任务，语义清晰。

---

## 杀伤力 3：3 个 worker 实例都写 `state["results"]`，怎么合并？

——这个问题 **Send 自己不需要解决**。

前一篇 State 拆解里我们已经推导出：**state 的合并由 channel 的 reducer 负责**。只要 `results` 字段声明为 `Annotated[list, operator.add]`，3 个 worker 的 `{"results": [x]}` 写入会被自动拼接。

**Send 和 channel 是正交的**：

- Send 负责 **派发**（一变多）
- channel 负责 **汇总**（多变一）
- Send 不碰 state，channel 不碰派发

这是个漂亮的关注点分离。没有 channel，Send 派出的多个实例就是孤岛；没有 Send，channel 再强大也派不出来。它们是 map-reduce 的两半。

---

## 杀伤力 4：Send 的 payload 不是 state 的一部分

这个很多人踩过。worker 函数拿到的输入 **不是全局 state**，而是 Send 里的那个 payload：

```python
Send("worker", {"item": "a"})

def worker(state):
    print(state)  # 打印的是 {"item": "a"}，不是主图的 state
```

为什么？因为如果 worker 读的是全局 state，3 个实例读到同一份数据，就失去了"每个实例带自己的数据"的意义。

所以 Send 做了一件反直觉的事：**在被 Send 触发的那个瞬间，worker 实例收到的"state"是 Send 的 payload，而不是主图的 state**。

这意味着 worker 的 input schema 通常和主图 state 不一样：

```python
class ItemInput(TypedDict):
    item: str

def worker(state: ItemInput) -> MainState:  # 输入是 ItemInput，输出回到 MainState
    return {"results": [process(state["item"])]}
```

Send 的 payload 是 **注入式** 的——绕过了正常的 channel 读取。它像是给这个节点实例定制了一个临时 state。

---

## 杀伤力 5：Send 本质是 Pregel 消息语义的"显式化"

回想 Pregel 论文的模型：顶点之间通过 **消息 (messages)** 通信。每个顶点在 superstep 里收到一批消息，执行 compute 函数，产出新消息发给其他顶点。

在 LangGraph 的普通图里，消息语义被 **隐藏** 了——通道订阅模式让它看起来像"共享 state + 边"。但 Send 把 Pregel 的消息语义 **重新暴露** 给用户：

| Pregel 原模型 | LangGraph 普通边 | LangGraph Send |
|---|---|---|
| 顶点可以发任意多条消息给任意顶点 | 边是静态的 "A → B" | 运行时动态决定发给谁 / 发几条 |
| 每条消息带 payload | 消息 = state 整体副本 | 消息 = 用户指定 payload |
| 同一顶点收多条消息就执行多次 | 节点每 superstep 最多跑一次 | 同一节点可被多次派发 |

**Send 不是新概念，是 LangGraph 一直在用但之前"藏起来"的东西**。它只是把隐式能力显式化——当普通边 / 条件边不够用时，用户可以直接接管消息派发。

这也解释了为什么 Send 看起来这么"底层"——因为它就是底层。它是 Pregel 消息模型的直接接口。

---

## 杀伤力 6：Send 和 checkpoint 怎么交互？

问题：一次 invocation 跑到一半挂了，恢复时 router 函数会不会产出不同的 Send 列表？

- 如果 router 是纯函数，基于 state 推导，每次返回结果一致 → 安全
- 如果 router 里带随机 / 当前时间 / 外部 API 调用 → 不一致，回放出错

LangGraph 的约束：**router 应当是 deterministic 的**，或者至少"在回放时相对确定"。派发的 Send 列表被视作 superstep 状态的一部分，写入 checkpoint。

这也暗示了 Send 的一个使用准则：**派发逻辑要纯净**，别在 router 里做副作用或非确定操作。想要非确定性就放到被派发的 worker 里——worker 的非确定性会被 checkpoint 记录下来。

---

## 重新定义：Send 是什么

三层视角：

| 层 | Send 在这层的形态 |
|---|---|
| **用户视角** | 一个 `Send(node, payload)` 对象，router 可以返回一批，表达 "动态派发 N 个任务" |
| **Pregel 视角** | 一条显式的、带 payload 的消息——Pregel 消息模型的直接暴露 |
| **实现视角** | `libs/langgraph/langgraph/types.py` 里的 `Send` dataclass，被 `_algo.prepare_next_tasks` 识别，为每个 Send 构造独立 PregelExecutableTask |

**Send = 带 payload 的、可重复派发同一节点的、动态决定数量的消息**。它和 channel 构成 map-reduce 的两半：**Send 分发，channel 汇总**。

---

## 反过来看源码

`libs/langgraph/langgraph/types.py` 里 `Send` 的定义 **极其简洁**：

```python
@dataclass
class Send:
    node: str       # 目标节点名
    arg: Any        # payload
```

就这两个字段。真正的魔法在 `pregel/_algo.py` 的 `prepare_next_tasks`：

1. 扫描上一步所有节点的 writes
2. 如果 write 是 Send 对象 → 为每个 Send 创建一个 `PregelExecutableTask`
3. 任务的 input 直接设为 Send.arg（**绕过 channel 读取**）
4. 多个 Send 指向同一节点 → 产生多个并发任务
5. 这些任务的输出通过正常 channel 机制汇总

**所以整个 Send 机制的核心实现其实只有 50 行左右代码**。因为它只需要在"准备任务"这一步做特殊处理，剩下的并发、重试、checkpoint、汇总，全都复用已有机制。

这就是好抽象的味道：**新能力不需要新架构，只需要在合适的地方开一个口子**。

---

## 收尾：这次拆解的产出

从"Send 是一种带数据的边"出发，被几个问题戳破后得到：

1. **静态图根本表达不了动态数量** —— 边装不下数据，节点不能动态实例化
2. **Send = 派发目标 + 独立实例 + 独立 payload**，三合一的原语
3. **Send 和 channel 是正交的**：前者分发，后者汇总，合起来 = map-reduce
4. **Send 的 payload 绕过 channel** 直接注入目标节点实例
5. **Send 是 Pregel 消息模型的"显式化"**，把一直存在的底层能力开放给用户
6. **router 必须 deterministic**，否则和 checkpoint / 回放冲突

**关键洞察**：Send 看起来"特殊"，其实是 LangGraph 正常机制的延伸——把本来隐藏在通道订阅背后的 Pregel 消息语义，直接暴露成一个可编程原语。它不需要新的调度器、新的 checkpoint、新的 reducer——因为这些都由已有抽象处理。Send 只做 **一件新事**：让用户显式构造消息。

这就是"好的原语"的标志：**加入它不需要改动其他任何东西**。
