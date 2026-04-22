---
sidebar_position: 8
---

绝大多数 LangGraph 代码里你都会看到这一行：

```python
config = {"configurable": {"thread_id": "..."}}
app.invoke(state, config=config)
```

很多人把它当成一个"放 thread_id 的盒子"就过了。但这个 `configurable` 其实是 LangGraph 里 **和 state 同等重要** 的第二条数据通路——它回答了一个 state 解决不了的问题：**"每次 invoke 都可能不同、但工作流本身不该持久化、节点又需要读到"的数据，放哪里？**

这篇把它讲清楚。

---

## 从一个实际痛点开始

你在写一个节点：

```python
def rewrite(state):
    text = state["raw_script"]
    return {"rewritten_script": doubao_llm(text)}
```

有一天你需要：

- 用户是 VIP 时调 `gpt-4`，普通用户调 `gpt-3.5`
- 按用户 locale 切换 prompt 语言
- API key 按租户区分
- 调试时塞一个"禁用缓存"的开关

这些信息 **不是工作流内部产生的**（不属于 state），而是 **每次调用由外部决定的**。你该把它们塞哪？

### 方案 A：塞进 state？

```python
class State(TypedDict):
    raw_script: str
    rewritten_script: str
    user_tier: str       # 加这里？
    api_key: str         # 还有这？
    locale: str
```

立刻有三个问题：

1. **state 会被 checkpoint** —— api_key 这种敏感信息存数据库？
2. **state 有 reducer 语义** —— `user_tier` 字段默认 `LastValue`，但它本来就不会变
3. **污染 schema** —— 所有节点都要看到这些字段，但大部分节点根本不关心

### 方案 B：用全局变量？

```python
CURRENT_USER = None

def rewrite(state):
    model = "gpt-4" if CURRENT_USER.is_vip else "gpt-3.5"
```

- **并发撞车**：两个用户同时 invoke，`CURRENT_USER` 互相覆盖
- **测试难写**：得 monkeypatch 全局
- **串子图**：子图里还是同一个全局，隔离失败

### 方案 C：闭包捕获？

```python
def make_rewrite(user):
    def rewrite(state):
        model = "gpt-4" if user.is_vip else "gpt-3.5"
        ...
    return rewrite

g.add_node("rewrite", make_rewrite(current_user))
```

- **图不能复用** —— 换个用户就要重新 compile 一整张图
- **违反图是静态定义的原则**

---

## 正确的答案：`configurable`

`configurable` 就是为这类需求设计的槽位：**per-invoke 传入、节点里可读、不进 checkpoint、不进 state**。

```python
def rewrite(state, config):
    user_tier = config["configurable"]["user_tier"]
    model = "gpt-4" if user_tier == "vip" else "gpt-3.5"
    return {"rewritten_script": llm(state["raw_script"], model=model)}


app.invoke(
    state,
    config={"configurable": {"thread_id": "s1", "user_tier": "vip"}},
)
```

用一个参数解决四件事：**每次调用不同、节点可读、不持久化、不污染 state**。

---

## `configurable` 是哪里来的？

它 **不是 LangGraph 发明的**，是 LangChain Core 的 `RunnableConfig` 里的一个标准字段。完整的 `RunnableConfig` 长这样：

```python
{
    "tags": [...],              # 追踪标签
    "metadata": {...},          # 调试元数据
    "callbacks": [...],         # 事件钩子
    "run_name": "...",          # 这次运行的名字
    "recursion_limit": 25,      # 最大递归深度
    "configurable": {...},      # ← 我们关心的
    "run_id": UUID,
    "max_concurrency": 10,
    ...
}
```

`configurable` 在 LangChain 里设计目的就是 **"运行时可变的参数"**——给 Runnable 一个口子，让它在调用时能接收外部注入的配置。LangGraph 直接继承了这个字段，同时在里面 **占用了几个自己需要的 key**。

---

## 两类用途

`configurable` 里的 key 分两类：

### ① LangGraph 框架保留 key

这些 key 是 LangGraph **硬编码识别的**，名字不能改：

| Key | 作用 |
|---|---|
| `thread_id` | checkpointer 识别会话时间线 |
| `checkpoint_ns` | 子图 checkpoint 命名空间，默认 `""` |
| `checkpoint_id` | 指定从哪个历史 checkpoint 恢复（time travel） |

**没 checkpointer 时** `thread_id` 可以不写；**有 checkpointer 时必须提供**。

### ② 用户自定义 key

除了上面三个保留名，剩下的都归你。想塞什么塞什么：

```python
{
    "configurable": {
        "thread_id": "s1",          # LangGraph 用
        "user_id": 42,              # 你自己用
        "tenant_id": "acme",
        "model_name": "gpt-4",
        "locale": "zh-CN",
        "debug_disable_cache": True,
    }
}
```

LangGraph 会把整个 `configurable` 原样传给每个节点和每个 Runnable（包括节点内部调用的 LLM）。

---

## 节点里怎么读

两种方式，选你喜欢的。

### 方式 1：在节点函数签名里加 `config` 参数

```python
from langchain_core.runnables import RunnableConfig

def rewrite(state, config: RunnableConfig):
    cfg = config["configurable"]
    model = cfg.get("model_name", "gpt-3.5")
    user_id = cfg["user_id"]
    ...
```

LangGraph 检测到函数有第二个参数且类型是 `RunnableConfig`，就会自动把当前 config 传进来。

### 方式 2：用 `get_config()` 全局函数

```python
from langgraph.config import get_config

def rewrite(state):
    cfg = get_config()["configurable"]
    model = cfg.get("model_name", "gpt-3.5")
    ...
```

不想改函数签名时用这个。底层靠 Python 的 `contextvars`——每个 invoke 有自己的上下文，**线程/协程安全**，不会串味。

两种方式底层是同一份 config，用哪个都行。

---

## `configurable` vs `state` 对比速查

| 维度 | `state` | `configurable` |
|---|---|---|
| **来源** | 工作流内部产生、节点写入 | 外部 per-invoke 注入 |
| **进 checkpoint** | ✅ 进 | ❌ 不进 |
| **reducer 合并** | ✅ 有 | ❌ 无（原样传递） |
| **每步会变化** | ✅ 经常 | ❌ 一次 invoke 内恒定 |
| **节点能写入吗** | ✅ 通过返回 dict | ❌ 只读 |
| **子图是否继承** | 取决于 schema 映射 | ✅ 自动继承 |
| **典型内容** | 消息、文案、中间结果 | user_id、model、api_key |
| **resume 时是否保留** | ✅ 从 checkpoint 恢复 | ❌ 必须重新传 |

最后一条特别重要——因为很多人踩过这个坑：

> **Checkpoint 恢复时只恢复 state，不恢复 configurable**。
> resume 调用必须重新提供 `configurable`（至少 `thread_id` 要一致）。

这个设计是 **有意的**：`configurable` 里常有 api_key、user_session 这类不应该持久化的东西。

---

## 配套：`ConfigurableField` 让 Runnable 字段也能被 configurable 驱动

LangChain 的 `ConfigurableField` 让你声明"这个 Runnable 的某个参数可以从 `configurable` 动态读":

```python
from langchain_core.runnables import ConfigurableField
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0.7).configurable_fields(
    temperature=ConfigurableField(id="temperature"),
    model=ConfigurableField(id="model_name"),
)


def rewrite(state):
    # 不需要手工读 config，LangChain 自动用 configurable 里的 temperature
    return {"rewritten_script": llm.invoke(state["raw_script"]).content}


app.invoke(
    state,
    config={
        "configurable": {
            "thread_id": "s1",
            "temperature": 0.2,          # → llm 自动采用
            "model_name": "claude-opus-4-7",
        }
    },
)
```

这是为啥 `configurable` 好用——**它是跨 LangChain + LangGraph 的统一运行时上下文通道**。节点、节点里的 LLM、节点里的工具调用，全部能在同一次 invoke 里读到同一份 config。

---

## 实战场景速览

**多租户**

```python
config = {"configurable": {"thread_id": "s1", "tenant_id": "acme"}}
# 每个节点读 tenant_id 决定查哪个数据库
```

**模型切换（VIP / 普通用户）**

```python
config = {"configurable": {"thread_id": "s1", "model": "gpt-4"}}
# LLM 节点根据 config 路由到对应模型
```

**A/B 实验**

```python
config = {"configurable": {"thread_id": "s1", "prompt_variant": "v2"}}
# 不同 variant 用不同的 system prompt
```

**调试开关**

```python
config = {"configurable": {"thread_id": "s1", "debug": True}}
# debug 模式跳过缓存、打详细日志
```

**子图传参**

主图的 `configurable` 自动传给子图。子图内节点直接能读，不需要额外处理。

---

## 常见陷阱

### ① 改了也白改

节点 **无法** 写入 `configurable`：

```python
def node(state, config):
    config["configurable"]["user_id"] = 99  # 改了也不会生效
    return {}
```

`configurable` 是只读的、一次 invoke 内恒定。想让下游节点看到不同的上下文？只能放 state 里。

### ② resume 忘了传

```python
# 第一次 invoke
app.invoke(initial, config={"configurable": {"thread_id": "s1", "user_id": 42}})

# 几小时后 resume
app.invoke(Command(resume="ok"), config={"configurable": {"thread_id": "s1"}})
#                                                                     ↑ user_id 没了！
```

节点代码如果依赖 `user_id`，resume 时会报 KeyError。**规则：每次 invoke / stream / resume 都要完整传一遍 configurable**。

### ③ 混用大小写 / 命名风格

LangGraph 只认 `thread_id` 这种 snake_case。`threadId`、`thread-id`、`ThreadId` 一律失效。

### ④ 把大对象塞 configurable

虽然技术上没限制，但 `configurable` 会被每次调用、每个子图、每个内部 Runnable 传递一遍。塞个几 MB 的对象进去会严重拖慢。**configurable 应该是轻量元数据**，重量级数据放 state（走 checkpoint）或者外部 store。

---

## LangGraph 1.x 的升级：`Runtime` + `context`

dict 类型的 `configurable` 有个明显缺点：**没有类型提示**。你写 `cfg["user_id"]` 全靠心记，IDE 不帮你。

LangGraph 1.x 引入了 `context` + `Runtime` 来解决这个：

```python
from dataclasses import dataclass
from langgraph.runtime import Runtime


@dataclass
class AppContext:
    user_id: int
    tenant_id: str
    model_name: str = "gpt-4"


def rewrite(state, runtime: Runtime[AppContext]):
    user = runtime.context.user_id     # ✅ 类型提示齐全
    model = runtime.context.model_name
    ...


graph = StateGraph(State, context_schema=AppContext)
```

然后调用：

```python
app.invoke(
    state,
    context=AppContext(user_id=42, tenant_id="acme"),
    config={"configurable": {"thread_id": "s1"}},
)
```

底层实现上 `context` 其实仍然经过 `configurable` 传递（LangGraph 内部把 dataclass 塞进 `configurable["__context__"]`），但用户侧得到了强类型体验。

**现在的最佳实践**：

- 框架保留 key（`thread_id` 等）继续放 `configurable`
- 业务上下文（user_id、tenant_id、model、api_key...）用 `context` + dataclass
- 遗留代码或者临时调试开关还是可以塞 `configurable`

---

## 总结：`configurable` 在 LangGraph 数据通路里的位置

LangGraph 每一次 invoke 涉及 **四条独立的数据通路**：

```
    ┌─────────────┐  工作流数据；节点读写；进 checkpoint
    │    state    │
    └─────────────┘

    ┌─────────────┐  per-invoke 运行时上下文；节点只读；不进 checkpoint
    │ configurable│  （+ Runtime context 强类型化的版本）
    └─────────────┘

    ┌─────────────┐  追踪/观察元数据；调试、LangSmith
    │   metadata  │
    └─────────────┘

    ┌─────────────┐  事件钩子
    │  callbacks  │
    └─────────────┘
```

四条通路各司其职。**`configurable` 的独特定位**是"每次 invoke 可变、节点需要读、但不属于工作流状态"——这是很多"看起来应该放 state，细想又不对"的信息的正确归宿。

一句话：

> **state 是工作流跑出来的数据，configurable 是每次 invoke 注入的上下文。**

分清楚这两条通路，LangGraph 的很多迷惑就消失了。
