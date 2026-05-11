# Fan-out / Fan-in 并发模式详解

> Fan-out 和 Fan-in 是并发编程和分布式系统中两个基础概念，分别描述"一对多"和"多合一"的数据流模式。

---

## 一、概念定义

### Fan-out（一对多）

**一个任务/消息拆分后，**同时分发给多个子任务并行处理****。

```
       ┌── 子任务 A ──┐
主任务 ─┤  子任务 B   ├──→ 并行执行
       └── 子任务 C ──┘
```

### Fan-in（多合一）

**多个子任务的处理结果，**汇聚合并成一个最终输出****。

```
子任务 A ─┐
子任务 B ─┼──→ 合并结果 ──→ 主任务
子任务 C ─┘
```

---

## 二、生活中的类比

### Fan-out 的例子：新闻推送

一个新闻事件发生后：
- 编辑部产生一条新闻
- 系统**同时**推送给所有订阅用户（邮件、短信、App 通知）
- 每个用户收到的是同一条新闻，只是渠道不同

### Fan-in 的例子：高考阅卷

- 多个阅卷老师**同时**批改不同考生的同一道题
- 所有老师的打分**汇总**到分数统计系统
- 系统计算平均分、最高分、最低分等汇总指标

---

## 三、在代码中的实现

### Fan-out 示例：Goroutine 并行分发

```go
func fanOut(jobs []Job) []Result {
    results := make(chan Result, len(jobs))

    // Fan-out: 一个 job 启动一个 goroutine 并行处理
    for _, job := range jobs {
        go func(j Job) {
            results <- process(j) // 发送到公共通道
        }(job)
    }

    // Fan-in: 收集所有结果
    var out []Result
    for range jobs {
        out = append(out, <-results)
    }
    return out
}
```

### Fan-in 示例：Reduce 汇总

```python
from functools import reduce

# 多个 worker 返回部分计数
partial_counts = [10, 20, 15, 25]

# Fan-in: 将所有部分计数 reduce 为总数
total = reduce(lambda a, b: a + b, partial_counts)
# 结果: 70
```

---

## 四、典型应用场景

| 场景 | Fan-out | Fan-in |
|------|---------|--------|
| **消息队列** | 一个消息被投递到多个消费者 | 多个消费者处理结果汇总 |
| **MapReduce** | Map 阶段将数据分发给多个 reduce worker | Reduce 阶段汇总同键数据 |
| **工作流引擎** | 一个任务触发多个并行子任务 | 多个子任务完成后触发后续节点 |
| **微服务** | 一个请求触发多个下游服务并行调用 | 多个下游响应合并为统一响应 |
| **AI Agent** | 一个规划节点分发任务给多个工具节点 | 多个工具结果聚合成最终回复 |

---

## 五、与同步/异步的组合

Fan-out 和 Fan-in 本身**只描述数据流的拓扑形状**，不关心同步还是异步：

| 组合模式 | 说明 | 示例 |
|---------|------|------|
| **同步 Fan-out → 同步 Fan-in** | 所有子任务完成后才返回 | `Promise.all()` + `Promise.race()` |
| **异步 Fan-out → 异步 Fan-in** | 边发边收，结果通过回调或通道传递 | Go channel + goroutine |
| **动态 Fan-out** | 子任务数量在运行时决定，而非静态配置 | LangGraph 的 `Send` 机制 |

### 动态 Fan-out 示例：LangGraph

```python
from langgraph.types import Send

defContinue

def continue_to_tools(state: State):
    # 根据用户选择动态 fan-out
    selected = [tool for tool in state['selected_tools']]
    return [Send(tool, {"messages": [state['user_msg']]}) for tool in selected]

graph = builder.compile()
# 用户选 3 个工具 → 动态产生 3 条边 → 3 个节点并行执行
```

---

## 六、一句话总结

> **Fan-out = 一对多（分发），Fan-in = 多合一（汇总）。** 它们是描述数据流拓扑的基础词汇，配合同步/异步机制可以组合出丰富的并发模型。

---

## 七、拓展阅读

- [Reducer 模式](./reducer.md) — 函数式编程中典型的 Fan-in 实现
- [Pregel 模型](https://en.wikipedia.org/wiki/Pregel_(computing)) — Google 提出的分布式图处理模型，核心是 Bulk Synchronous Parallel 中的 fan-out/broadcast
- [MapReduce](https://en.wikipedia.org/wiki/MapReduce) — Hadoop 的核心理论基础，Map = Fan-out，Reduce = Fan-in
