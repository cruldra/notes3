---
sidebar_position: 7
---
> 从"interrupt = pause/resume"这个天真理解出发，戳破它，推导出 Interrupt 实际上是 **"回滚 + 存档 + 重试时短路"**——由 Pregel + Channel + Checkpoint + Scratchpad 组合出来的新玩法，没有引入任何新底层机制。

## 为什么挑 Interrupt

Interrupt 是 LangGraph 里最 **反直觉** 的概念。用户看到：

```python
def confirm_node(state):
    user_input = interrupt("请确认订单")
    return {"confirmed": user_input}
```

自然会以为：节点执行到 `interrupt()` 这一行，**函数就停在这里**，等外部给个值，然后继续往下跑。像 `input()` 一样。

这个想法 **完全错误**。但这个错误非常有教育意义——它能把你一路逼到 "LangGraph 的执行模型到底能表达什么" 的本质上。拆完你会发现：

> **Interrupt 根本不是"暂停"，是"回滚 + 存档 + 重试时短路"。** 这是 Pregel + checkpoint + scratchpad 三件既有能力组合出来的新玩法。

---

## 天真理解：Interrupt = pause/resume

大多数人第一次看到 `interrupt()` 会想到 `input()`、`asyncio.Event.wait()`、`coroutine.suspend()` 这类东西：

```python
# 想象中的样子
def confirm_node(state):
    x = step1()
    user_input = input()  # 进程停在这里等
    y = step2(user_input)
    return y
```

"interrupt 就是 input()，只不过外部提供值"。

这个理解在 **一个** 场景下能 work：**你的进程能一直活着**。

但只要你问一个问题，这个理解就崩了。

---

## 杀伤力 1：进程可以死透

用户调完图就关机。服务器半夜重启。Lambda 函数 15 分钟超时了。生产环境滚动发布把容器干掉了。

**没有任何运行时对象能"冻结"着等外部输入几小时甚至几天**。线程、协程、Future、Event——它们都活在进程里，进程一没了它们全蒸发。

> **唯一能跨越进程生命周期的东西，是持久化数据。**

所以 interrupt 要在"用户三天后才回来"的场景下还能 work，它 **必须把状态存盘，而不是在内存里 hold 住什么**。

那剩下的问题就是：**哪里存？以什么粒度存？怎么恢复？**

---

## 杀伤力 2：必然落回 Pregel 的 superstep 屏障

回想 Pregel 拆解里的关键结论：

> checkpoint 只能在 superstep 屏障处存。屏障之间的状态是半成品。

如果 interrupt 存档时机不落在屏障上，你存下来的是 **一致性可疑** 的半成品——恢复时不知道已完成哪些写入、哪些节点已跑、哪些没跑。

所以 interrupt 的"暂停点"必须 **对齐** superstep 屏障。这就推出了最简单的一种 interrupt——**static interrupt**：

```python
graph.compile(
    interrupt_before=["confirm"],   # 在 confirm 节点要跑的那个 superstep 之前停
    interrupt_after=["confirm"],    # 在 confirm 节点跑完的那个 superstep 之后停
)
```

两个时机都 **在屏障上**：节点还没跑 / 节点刚跑完。状态都是一致的，checkpoint 存档没问题。

static interrupt 其实就是 **Pregel 调度器在屏障处看一眼"要不要停"**，要停就落盘，然后退出循环。完全是调度层的事，节点本身毫不知情。这种 interrupt 简单、安全、无副作用——但它 **不够灵活**：你要等到"某个节点之前/之后"才能停，节点内部做不了精细控制。

---

## 杀伤力 3：Dynamic interrupt 看起来在节点中途停下，但它不可能真的这么做

```python
def confirm_node(state):
    x = do_something()
    user_input = interrupt("msg")   # ← 停在这里？
    return {"x": x, "user_input": user_input}
```

等等，这明显是在节点内部、函数执行到一半停下来。这和"只能在屏障停"的推论矛盾啊？

矛盾只是表象。实际上发生的事情完全不是字面意思：

1. 节点函数执行，跑到 `interrupt("msg")`
2. `interrupt()` 内部 **raise 一个特殊异常**（`GraphInterrupt`）
3. Pregel runner 捕获这个异常，把这个节点这次的 **全部执行视为未完成**
4. **该节点的所有写入被丢弃**（还没到屏障，根本没 commit）
5. Pregel 在屏障处 checkpoint 当前 state，并记录"这个节点在等 resume，消息是 msg"
6. 退出调度循环，返回给用户

关键：**节点不是"暂停"在 `interrupt()` 那一行——它是被整个回滚了**。执行的前半段 `x = do_something()` 的结果一并作废。

等用户调用：

```python
graph.stream(Command(resume="yes"), config=thread)
```

发生的事：

1. Pregel 从 checkpoint 恢复
2. 看到"这个节点在等 resume，值是 yes"
3. **重新完整执行** `confirm_node(state)`
4. 函数又跑到 `x = do_something()`（副作用重复！），然后再次调用 `interrupt("msg")`
5. 这次 `interrupt()` 检测到有 resume 值可用 → **直接返回 `"yes"`**，不 raise 异常
6. 函数继续往下跑到 `return {...}`，这次正常完成
7. 写入被 commit，图进入下一个 superstep

所以 dynamic interrupt 的真实语义是：

> **"抛异常 → 回滚节点 → 存档 → 恢复时重跑节点 → interrupt() 在重跑时短路返回 resume 值"**

节点从头到尾 **跑了两次**。第一次是为了"发现暂停请求"，第二次是为了"消费 resume 值继续"。

---

## 杀伤力 4：这对节点有什么要求？

既然节点会被 **整个重跑**，那 `interrupt()` 之前的代码 **必须幂等**：

- ✅ 纯计算、读 state、查数据库、调用只读 API
- ❌ 发送不可撤销的邮件、写数据库、扣钱、调用只写 API

否则 resume 一次，副作用就重复一次。

这也能推出一条实用准则：

> **把副作用放到 `interrupt()` 之后**。interrupt 之前只做 read-only 的计算和数据准备，"需要用户确认才能执行"的真正动作放到 resume 之后。

这和 Send 拆解里推出的 "router 必须 deterministic" 是同一性质的约束——**BSP 屏障对齐的设计，天然要求节点在屏障之间的执行是可重复的**。这不是 LangGraph 故意刁难，而是 BSP 模型的代价。

---

## 杀伤力 5：`interrupt()` 到底是个什么函数？

看起来它叫了一次就停了，调两次返回值就不一样了——像带状态的函数。它确实带状态，状态存在 **Pregel 的 scratchpad** 里（`pregel/_scratchpad.py`）。

scratchpad = 节点执行的临时上下文，里面记录：

- 这次执行是"首次尝试"还是"resume 后的重跑"
- 如果是重跑，resume 值是什么
- 这个节点已经调用了几次 `interrupt()`（支持多个 interrupt）

`interrupt()` 函数的伪代码：

```python
def interrupt(value):
    scratchpad = get_current_scratchpad()
    if scratchpad.has_resume_value():
        return scratchpad.consume_resume_value()
    else:
        raise GraphInterrupt(value)
```

就这么简单。它的"魔法"完全来自 **执行上下文（scratchpad）由外部调度器管理** 这一事实。

这个模式很熟悉——它叫 **algebraic effects / continuations**（代数效应 / 续延），函数式语言（Eff、Koka、Unison）里的概念。`interrupt()` 等于一次 effect request，用户的 resume 等于 effect handler 提供的值。LangGraph 悄悄地在 Python 里做了一个小型的 effect system。

---

## 杀伤力 6：Interrupt 其实没有引入"新"机制

现在回头数一数 interrupt 依赖的组件：

| 能力 | 来自哪个既有抽象 |
|---|---|
| 存档当前状态 | **Checkpoint**（来自 Pregel BSP 屏障） |
| 屏障对齐 | **Pregel** superstep 循环 |
| "节点被回滚"的能力 | **Channel** 在屏障前不 commit（节点的 writes 只有整体完成才生效） |
| 记录 resume 值 | **Scratchpad**（Pregel 本来就要管的执行上下文） |
| 用户"喂入"数据 | **图的 input**（Command 是一种特殊 input） |

**Interrupt 没有引入任何新底层机制——它只是把已有的 5 件事组合成一个 API**。

这和 Send 拆解里的结论一模一样：**好原语的标志是"加入它不需要改动其他任何东西"**。Interrupt 和 Send 都印证了这一点。

---

## 重新定义：Interrupt 是什么

三层视角：

| 层 | Interrupt 在这层的形态 |
|---|---|
| **用户视角** | "在流程中途停下来等外部输入" |
| **调度视角** | 利用 Pregel 屏障 + checkpoint，把流程冻结成持久化状态，resume 时从上次屏障重启 |
| **实现视角** | **dynamic**：raise 异常 + 回滚节点 + 重跑时短路返回值；**static**：调度层在屏障处检查标志决定是否退出循环 |

**Interrupt ≠ 暂停。Interrupt = 回滚到上个屏障 + 存档 + 唤醒时从屏障重启 + 节点重跑时 `interrupt()` 短路返回 resume 值**。

一句话：**Interrupt 是"checkpoint + 延迟回填"的 API 包装**。

---

## 反过来看源码

带这个理解打开 `libs/langgraph/langgraph/types.py` 和 `pregel/_scratchpad.py`：

- `interrupt(value)` 函数本身 **非常短**：查 scratchpad，要么返回、要么抛异常
- `GraphInterrupt` 是一个 marker exception，不是"错误"——只是一个控制流信号
- Pregel `_runner.py` 的任务执行逻辑里有一个 `except GraphInterrupt` 分支，把这个当作"节点未完成"处理
- `_loop.py` 在屏障处检查"是否有未完成的 interrupt 任务"，如果有就把它们的 value 记录到 checkpoint，然后退出
- resume 时从 checkpoint 恢复，scratchpad 里预填 resume 值，再次进入调度循环

**所有这些代码加起来不到 200 行**。因为它只是在既有调度循环上开了一个口子，其他（持久化、并发、合并、重试）全部复用。

---

## 收尾：这次拆解的产出

从"interrupt = pause/resume"的天真理解出发，戳破后得到：

1. **进程不会活着等你三天**，所以 interrupt 必须落在持久化层，不是运行时层
2. **必须对齐 Pregel 的 superstep 屏障**，否则 checkpoint 不一致
3. **Dynamic interrupt 不是暂停，是"抛异常 → 回滚 → 重跑 → 短路"** 的四步舞
4. **节点的副作用在 `interrupt()` 之前必须幂等**，否则 resume 重复执行会出问题
5. **`interrupt()` 函数本身只有几行**，它的所有魔法来自 scratchpad 和 Pregel 调度器
6. **Interrupt 没有引入新底层机制**，是 Pregel + Channel + Checkpoint + Scratchpad 的组合应用

**关键洞察**：Interrupt 是 LangGraph 里 **"API 极富，机制极省"** 的一个典范——它对外给用户一个看起来很强大的 pause/resume 能力，对内完全是用既有的调度 + 持久化机制实现的。这种"无新机制，纯组合"的设计是架构优雅的标志。

顺带一个反直觉的美学：**dynamic interrupt 节点会跑两次**。这听起来像 bug，但它其实是 "BSP 屏障对齐 + 不写持久化中间态" 这两个硬约束交集出的唯一解。一旦接受这个约束，所有看似别扭的使用规则（代码要幂等、副作用放后面）都自然而然。
