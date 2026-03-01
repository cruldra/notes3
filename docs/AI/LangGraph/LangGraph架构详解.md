# LangGraph 架构深度解析 - 基于 Smart Sales 项目实践


> 本文档结合智能销售助手系统（Smart Sales）的实际代码，深入讲解 LangGraph 的架构设计、工作流程和核心概念。

---


## 概述

**LangGraph** 是 LangChain 生态系统中用于构建复杂多智能体工作流的库。它通过**状态图（StateGraph）**模型，让开发者能够定义、编排和执行多步骤的 AI 工作流。

在本项目中，LangGraph 被用作决策层的核心技术，实现了三个核心 Agent：
- **RouterAgent** - 事件路由与分发
- **ScoringAgent** - 多维度客户评分
- **ProfileAgent** - 客户画像构建与策略生成

---

## 📋 目录

1. [核心概念](#一核心概念)
2. [项目中的 LangGraph 实现](#二项目中的-langgraph-实现)
3. [LangGraph 高级特性](#三langgraph-高级特性)
4. [LLM 工厂模式](#四llm-工厂模式)
5. [RAG 集成](#五rag-集成)
6. [FSM 状态机集成](#六fsm-状态机集成)
7. [最佳实践总结](#七最佳实践总结)
8. [性能优化建议](#八性能优化建议)
9. [参考资源](#九参考资源)
10. [总结](#十总结)
11. [官方文档核心概念补充](#十一官方文档核心概念补充)
12. [官方资源链接](#十二官方资源链接)

---

## 一、核心概念

### 1.1 StateGraph（状态图）

StateGraph 是 LangGraph 的核心抽象，它是有向图结构，用于表示工作流的执行路径：

```text
┌─────────────────────────────────────────────────────────────┐
│                    StateGraph 结构                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────┐      ┌─────────┐      ┌─────────┐            │
│   │  Node A │ ───▶ │  Node B │ ───▶ │  Node C │            │
│   │(入口点)  │      │(处理)   │      │(出口点) │            │
│   └─────────┘      └─────────┘      └─────────┘            │
│        │                                        │           │
│        │              ┌─────────┐              │           │
│        └─────────────▶│  Node D │◀─────────────┘           │
│                       │(分支)   │                          │
│                       └─────────┘                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**关键特性：**
- **节点（Nodes）**：执行函数，接收状态、返回状态
- **边（Edges）**：节点间的连接，定义执行顺序
- **状态（State）**：贯穿整个工作流的数据载体
- **入口点（Entry Point）**：工作流的起始节点
- **结束点（End）**：工作流的终止节点

### 1.2 节点（Nodes）

节点是工作流的基本执行单元，本质是一个 Python 函数：

```python
async def node_function(state: StateType) -> StateType:
    # 1. 读取当前状态
    data = state.get("some_key")
    
    # 2. 执行业务逻辑
    result = await some_async_operation(data)
    
    # 3. 更新状态并返回
    state["new_key"] = result
    return state
```

**节点类型：**

| 类型 | 说明 | 示例 |
|------|------|------|
| 普通节点 | 顺序执行 | 数据收集、规则提取 |
| 条件节点 | 根据状态决定下一步 | 路由决策 |
| 并行节点 | 同时执行多个分支 | 多维度评分 |

### 1.3 边（Edges）

边定义了节点间的流转关系：

```python
# 线性边：A → B
workflow.add_edge("node_a", "node_b")

# 条件边：根据状态决定流向
workflow.add_conditional_edges(
    "node_a",
    decision_function,  # 返回下一个节点名称
    {"branch_1": "node_b", "branch_2": "node_c"}
)
```

**边类型：**

| 类型 | 说明 |
|------|------|
| 普通边 | 固定的流转关系 |
| 条件边 | 根据状态动态决定 |
| START | 虚拟起点 |
| END | 虚拟终点 |

### 1.4 状态（State）

状态是工作流的核心数据载体，使用 TypedDict 定义：

```python
from typing import TypedDict, Required, NotRequired

class WorkflowState(TypedDict, total=False):
    # Required: 必须存在的字段
    input_id: Required[str]
    
    # 中间计算结果
    intermediate_data: dict[str, Any]
    
    # 最终输出
    output_result: str
    
    # 错误信息
    error: str | None
```

**状态设计原则：**
1. **单一来源**：所有数据通过状态传递
2. **不可变更新**：每个节点返回新的状态对象
3. **类型安全**：使用 TypedDict 确保类型检查

---

## 二、项目中的 LangGraph 实现

### 2.1 ScoringAgent - 客户评分工作流

**工作流结构：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ScoringAgent 工作流                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  collect_data ──▶ extract_rules ──▶ llm_analyze ──▶ mix_scores     │
│                                                                      │
│       │                                    │                         │
│       ▼                                    ▼                         │
│  [从PG/Mongo 读取]               [LLM 智能分析]                     │
│  chat_messages,                   四维评分 + 文本分析               │
│  skip_llm 标志                   (价格/需求/共识/信任)              │
│                                                                      │
│  ──────────────────────────────────────────────────────────────     │
│                                                                      │
│  assign_grade ──▶ gen_summary ──▶ persist ──▶ [END]                │
│                                                                      │
│       │              │                 │                            │
│       ▼              ▼                 ▼                            │
│  [S/A/B/low 分级]  [中文摘要 +       [写入 PG 数据库]               │
│                    跟进建议]        [触发 FSM 事件]                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**核心代码解析：**

```python
class ScoringAgent:
    async def run(self, customer_batch_record_id: str) -> ScoringState:
        # 1. 构建 StateGraph
        workflow = StateGraph(ScoringState)
        
        # 2. 添加节点
        workflow.add_node("collect_data", self.collect_data_node)
        workflow.add_node("extract_rules", self.extract_rules_node)
        workflow.add_node("llm_analyze", self.llm_analyze_node)
        workflow.add_node("mix_scores", self.mix_scores_node)
        workflow.add_node("assign_grade", self.assign_grade_node)
        workflow.add_node("gen_summary", self.gen_summary_node)
        workflow.add_node("persist", self.persist_node)
        
        # 3. 定义边（线性工作流）
        workflow.set_entry_point("collect_data")
        workflow.add_edge("collect_data", "extract_rules")
        workflow.add_edge("extract_rules", "llm_analyze")
        workflow.add_edge("llm_analyze", "mix_scores")
        workflow.add_edge("mix_scores", "assign_grade")
        workflow.add_edge("assign_grade", "gen_summary")
        workflow.add_edge("gen_summary", "persist")
        
        # 4. 编译并执行
        graph = workflow.compile()
        result = await graph.ainvoke(initial_state)
        return result
```

**状态定义（ScoringState）：**

```python
class ScoringState(TypedDict, total=False):
    # 输入
    customer_batch_record_id: Required[str]
    customer_id: Required[str]
    
    # 数据收集结果
    chat_messages: list[dict[str, Any]]
    skip_llm: bool  # 是否跳过 LLM 分析
    
    # 规则提取结果
    rule_scores: dict[str, float]          # 四维评分
    rule_indicators: dict[str, dict]       # 指标详情
    
    # LLM 分析结果
    llm_scores: dict[str, float]
    llm_analysis: str
    
    # 混合评分结果
    dimension_scores: dict[str, float]     # 加权后评分
    composite_score: float                 # 综合分
    grade: str                             # S/A/B/low
    
    # FSM 事件检测
    detected_events: list[str]
    
    # 输出
    summary: str
    suggestion: str
    model_used: str
    error: str | None
```

**评分维度说明：**

| 维度 | 权重 | 规则指标 | 说明 |
|------|------|----------|------|
| price | 0.2 | 价格异议次数、询价次数 | 价格敏感度 |
| demand | 0.3 | 规划咨询次数、课程询问次数、问题数 | 需求匹配度 |
| consensus | 0.3 | 共识词频、沟通轮次 | 共识度 |
| trust | 0.2 | 信任词频、客户主动发起次数 | 信任度 |

**混合评分算法：**

```python
def mix_scores_node(self, state: ScoringState) -> ScoringState:
    # 规则评分 40% + LLM 评分 60%
    dimension_scores = {
        dimension: rule_scores[dimension] * 0.4 + llm_scores[dimension] * 0.6
        for dimension in ("price", "demand", "consensus", "trust")
    }
    
    # 加权综合分
    weights = {"price": 0.2, "demand": 0.3, "consensus": 0.3, "trust": 0.2}
    composite_score = sum(weights[d] * dimension_scores[d] for d in weights)
    
    state["dimension_scores"] = dimension_scores
    state["composite_score"] = composite_score
    return state
```

### 2.2 ProfileAgent - 画像生成工作流

**工作流结构：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ProfileAgent 工作流                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  START ──▶ build_profile ──▶ generate_strategy ──▶ generate_scripts │
│                                                                      │
│                │                     │                    │         │
│                ▼                     ▼                    ▼         │
│          [LLM 生成]           [RAG 检索 +           [RAG 检索 +      │
│          客户画像文本          LLM 生成]            LLM 生成]        │
│                               跟进策略              销售话术         │
│                               (成功案例库)          (原子事实库)     │
│                                                                      │
│  ──────────────────────────────────────────────────────────────     │
│                                                                      │
│  persist ──▶ END                                                    │
│                                                                      │
│    │                                                                │
│    ▼                                                                │
│  [写入 AIAnalysis 表]                                               │
│  [整合画像/策略/话术]                                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**核心代码解析：**

```python
class ProfileAgent:
    def _build_graph(self) -> Any:
        graph = StateGraph(ProfileState)
        
        # 节点定义
        graph.add_node("build_profile", self.build_profile_node)
        graph.add_node("generate_strategy", self.generate_strategy_node)
        graph.add_node("generate_scripts", self.generate_scripts_node)
        graph.add_node("persist", self.persist_node)
        
        # 边定义（使用 START/END 常量）
        graph.add_edge(START, "build_profile")
        graph.add_edge("build_profile", "generate_strategy")
        graph.add_edge("generate_strategy", "generate_scripts")
        graph.add_edge("generate_scripts", "persist")
        graph.add_edge("persist", END)
        
        return graph
```

**RAG 集成示例：**

```python
async def generate_strategy_node(self, state: ProfileState) -> ProfileState:
    # 1. 从 Qdrant 检索成功案例
    cases = await search_success_cases(
        query=profile, 
        grade=grade  # 按客户等级过滤
    )
    
    # 2. 构建上下文
    case_context = self._format_rag_results(cases)
    
    # 3. LLM 生成策略
    prompt = f"""
    你是销售策略助手。请根据客户画像和参考案例输出 JSON。
    
    客户画像: {profile}
    客户等级: {grade}
    参考案例:
    {case_context}
    """
    
    result = self._invoke_llm_json(prompt, fallback)
    state["follow_up_strategy"] = result["follow_up_strategy"]
    return state
```

### 2.3 RouterAgent - 路由决策工作流

**工作流结构：**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RouterAgent 工作流                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  START ──▶ route ──▶ execute ──▶ END                                │
│                                                                      │
│              │              │                                       │
│              ▼              ▼                                       │
│         [事件类型匹配]   [调用目标 Agent]                            │
│         message.*   →   ScoringAgent                                │
│         plan.*|profile.* → ProfileAgent                             │
│         payment.*|state.* → FSM 触发                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**路由规则：**

```python
def route_node(self, state: RouterState) -> RouterState:
    event_type = state["event_type"]
    prefix = event_type.split(".")[0]
    
    match prefix:
        case "message":
            state["routed_to"] = "ScoringAgent"
        case "plan" | "profile":
            state["routed_to"] = "ProfileAgent"
        case "payment" | "state":
            state["routed_to"] = "FSM"
        case _:
            state["error"] = f"unknown event type: {event_type}"
    
    return state
```

---

## 三、LangGraph 高级特性

### 3.1 异步执行

LangGraph 原生支持异步执行：

```python
# 同步执行
result = graph.invoke(initial_state)

# 异步执行（推荐）
result = await graph.ainvoke(initial_state)
```

**最佳实践：**
- 节点函数可以是 sync 或 async
- 涉及 IO 操作（数据库、API 调用）时，使用 async
- LangGraph 会自动处理协程调度

### 3.2 错误处理与降级

```python
def node_with_fallback(self, state: State) -> State:
    try:
        result = risky_operation()
        state["result"] = result
    except Exception as e:
        logger.warning(f"Operation failed: {e}")
        state["result"] = get_fallback_value()
        state["error"] = str(e)
    return state
```

**项目中的降级策略：**
- LLM 调用失败时，使用规则评分兜底
- RAG 检索失败时，使用默认话术
- 状态为空时，使用默认值

### 3.3 并发控制

使用 Redis 实现分布式锁，防止重复执行：

```python
async def run(self, customer_batch_record_id: str) -> ScoringState:
    redis = get_redis()
    lock_key = f"scoring:{customer_batch_record_id}"
    
    # 尝试获取锁
    lock = await redis.set(lock_key, "1", nx=True, ex=300)
    if not lock:
        return {"error": "already_processing"}
    
    try:
        # 执行工作流
        result = await graph.ainvoke(initial_state)
        return result
    finally:
        # 释放锁
        await redis.delete(lock_key)
```

---

## 四、LLM 工厂模式

项目使用工厂模式统一管理 LLM 实例：

```python
# llm_factory.py
from langchain_core.language_models.chat_models import BaseChatModel

def get_llm(
    *, 
    provider: str | None = None, 
    model: str | None = None, 
    **kwargs
) -> BaseChatModel:
    settings = get_settings()
    provider = (provider or settings.LLM_PROVIDER).lower()
    model = model or settings.LLM_MODEL
    
    match provider:
        case "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model, api_key=settings.OPENAI_API_KEY)
        
        case "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model, api_key=settings.CLAUDE_API_KEY)
        
        case "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model, google_api_key=settings.GEMINI_API_KEY)
        
        case "openrouter":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                base_url="https://openrouter.ai/api/v1",
                api_key=settings.OPENROUTER_API_KEY
            )
        
        case _:
            raise ValueError(f"Unsupported LLM provider: {provider}")
```

**支持的提供商：**

| 提供商 | 模型示例 | 用途 |
|--------|----------|------|
| OpenAI | gpt-4o, gpt-4o-mini | 通用任务 |
| Anthropic | claude-3-5-sonnet | 复杂推理 |
| Google | gemini-1.5-pro | 多模态任务 |
| OpenRouter | 多模型路由 | 灵活切换 |

---

## 五、RAG 集成

### 5.1 向量检索

使用 Qdrant 作为向量数据库：

```python
@dataclass
class RAGResult:
    doc_id: str
    title: str
    content: str
    score: float
    payload: dict

async def search_atomic_facts(
    query: str,
    category: str | None = None,
    top_k: int = 5,
    score_threshold: float = 0.7
) -> list[RAGResult]:
    # 1. 向量化查询
    vector = await embed_query(query)
    
    # 2. 构建过滤条件
    query_filter = None
    if category:
        query_filter = Filter(
            must=[FieldCondition(key="category", match=MatchValue(value=category))]
        )
    
    # 3. 向量检索
    response = await client.query_points(
        collection_name="atomic_facts",
        query=vector,
        limit=top_k,
        score_threshold=score_threshold,
        query_filter=query_filter,
        with_payload=True,
    )
    
    return [...]
```

### 5.2 双库设计

| 库名称 | 用途 | 数据类型 |
|--------|------|----------|
| atomic_facts | 原子事实库 | 产品信息、价格政策、FAQ |
| success_cases | 成功案例库 | 历史成交案例、最佳实践 |

---

## 六、FSM 状态机集成

LangGraph 与 FSM（有限状态机）协同工作：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LangGraph + FSM 协作模式                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   LangGraph Agent                      FSM (transitions.py)        │
│   ┌─────────────────┐                 ┌──────────────────┐          │
│   │ 节点执行中       │                 │ 状态定义表        │          │
│   │                 │                 │ (状态, 事件)→新状态│          │
│   │ detect_events   │ ──────────────▶ │ CustomerFSM      │          │
│   │ [分析聊天记录]   │   触发事件       │   .trigger()     │          │
│   │                 │                 │   .can_trigger() │          │
│   └─────────────────┘                 └──────────────────┘          │
│           │                                     │                    │
│           │         persist_node                │                    │
│           └─────────────────────────────────────▶                   │
│                  [写入新状态到数据库]                                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**客户生命周期状态：**

```
DAY0 → 已申请规划 → 已规划报价 → 讨价还价中 → 已成交
  │                                      │
  ▼                                      ▼
失去联系 ← 5天未互动              延迟购买 ← 确认延迟
  │
  ▼
低意向 → 营期结束 → Archived → Silent → 已激活 → 复购成功
```

---

## 七、最佳实践总结

### 7.1 状态设计

1. **使用 TypedDict**：确保类型安全
2. **区分 Required/Optional**：明确哪些字段是必需的
3. **包含 error 字段**：统一错误处理
4. **状态扁平化**：避免深层嵌套

### 7.2 节点设计

1. **单一职责**：每个节点只做一件事
2. **纯函数**：相同输入产生相同输出
3. **异步优先**：涉及 IO 时使用 async
4. **降级处理**：准备 fallback 逻辑

### 7.3 工作流编排

1. **线性优先**：简单场景使用线性流程
2. **条件分支**：复杂场景使用条件边
3. **避免循环**：除非确实需要迭代
4. **清晰的命名**：节点名称要表达意图

### 7.4 测试策略

```python
# 测试单个节点
def test_extract_rules_node():
    agent = ScoringAgent()
    state = {"chat_messages": [...]}
    result = agent.extract_rules_node(state)
    assert result["rule_scores"]["price"] > 50

# 测试完整工作流
@pytest.mark.asyncio
async def test_run_executes_linear_graph():
    agent = ScoringAgent()
    result = await agent.run("cbr-001")
    assert result["grade"] in ["S", "A", "B", "low"]
```

---

## 八、性能优化建议

### 8.1 缓存策略

- LLM 响应缓存（Redis）
- 向量检索结果缓存
- 规则评分结果缓存

### 8.2 并发优化

- 多维度评分并行计算
- RAG 检索并行执行
- 数据库连接池

### 8.3 降级策略

- LLM 不可用时使用规则评分
- RAG 不可时使用默认话术
- 数据库超时使用内存缓存

---

## 九、参考资源

### 项目文件位置

```
backend/src/smart_sales/decision/
├── agents/
│   ├── scoring_agent.py     # 评分 Agent
│   ├── profile_agent.py     # 画像 Agent
│   ├── router_agent.py      # 路由 Agent
│   └── __init__.py
├── rag/
│   ├── retriever.py         # RAG 检索
│   ├── embeddings.py        # 向量化
│   └── indexer.py           # 索引管理
├── fsm/
│   ├── machine.py           # 状态机引擎
│   ├── states.py            # 状态定义
│   └── transitions.py       # 转换规则
└── llm_factory.py           # LLM 工厂
```

### 测试文件位置

```
backend/tests/decision/
├── test_scoring_agent_nodes.py
├── test_profile_agent.py
└── rag/
    ├── test_retriever.py
    ├── test_embeddings.py
    └── test_indexer.py
```

---

## 十、总结

LangGraph 在项目中承担了**决策层**的核心职责，通过以下方式实现智能销售分析：

1. **工作流编排**：将复杂的多步骤分析流程（数据收集→规则提取→LLM分析→评分→持久化）分解为清晰的节点链

2. **多 Agent 协作**：RouterAgent 作为入口，协调 ScoringAgent 和 ProfileAgent 的分工执行

3. **RAG 增强**：集成向量检索，为策略生成和话术推荐提供上下文支持

4. **FSM 集成**：通过状态机驱动客户生命周期管理，实现自动状态流转

5. **容错设计**：多层降级策略（LLM→规则、RAG→默认、异常→返回错误）确保系统稳定性

这种架构的优势在于**模块化、可测试、可扩展**——每个节点可以独立开发和测试，新的分析维度可以通过添加节点实现，不同的业务场景可以通过组合节点来支持。



---

## 十一、官方文档核心概念补充

### 11.1 StateGraph 官方定义

**StateGraph** 是一种特殊的图结构，节点之间通过读写**共享状态**进行通信。

**核心特性**（来自官方文档）：
- 每个节点签名：`State -> Partial<State>`
- 每个节点接收当前状态，返回状态更新
- 状态在节点间自动传递
- 启用上下文感知决策和持久记忆

**官方代码示例**：
```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import operator

class State(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]  # 使用 reducer
    extra_field: int

# 创建 StateGraph
builder = StateGraph(State)
```

### 11.2 Reducers（状态归约器）

**Reducers 是 LangGraph 的核心机制**，指定如何应用状态更新：

**常用 Reducers**：
```python
import operator
from typing import Annotated

class State(TypedDict):
    # 追加列表
    messages: Annotated[list[str], operator.add]
    
    # 字符串拼接
    text: Annotated[str, operator.add]
    
    # 自定义 reducer
    aggregate: Annotated[list, custom_reducer]

def custom_reducer(left, right):
    """自定义归约逻辑"""
    return left + right
```

**工作原理**：
1. 节点返回 `{"messages": [new_msg]}`
2. Reducer 自动合并：`old_messages + [new_msg]`
3. 下一个节点接收更新后的完整状态

### 11.3 边的完整类型

**1. 静态边（Static Edges）**：
```python
from langgraph.graph import START, END

builder.add_edge(START, "first_node")     # 入口边
builder.add_edge("node_a", "node_b")       # 节点间边
builder.add_edge("last_node", END)         # 终止边
```

**2. 条件边（Conditional Edges）**：
```python
from typing import Literal

def should_continue(state: State) -> Literal["node_b", END]:
    """根据状态条件路由"""
    if len(state["aggregate"]) < 7:
        return "node_b"
    else:
        return END

builder.add_conditional_edges("node_a", should_continue)
```

**3. 动态路由（Send）**：
```python
from langgraph.types import Send

def continue_to_jokes(state: OverallState):
    """为每个主题创建并行任务"""
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

builder.add_conditional_edges(START, continue_to_jokes)
```

### 11.4 工具（Tools）集成模式

**使用 `@tool` 装饰器定义工具**：
```python
from langchain.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`."""
    return a * b

# 绑定到模型
model_with_tools = model.bind_tools(tools)
```

**使用 ToolNode 自动执行**：
```python
from langgraph.prebuilt import ToolNode

# 自动处理工具调用
tools_node = ToolNode(tools)
builder.add_node("tools", tools_node)
```

### 11.5 多 Agent 协作 - 子图模式

**子图是嵌套在父图中作为节点的图**：

```python
# 定义子图
subgraph_builder = StateGraph(SubState)
subgraph_builder.add_node("process", process_node)
subgraph = subgraph_builder.compile()

# 方式 1：从节点内调用
def call_subgraph(state: ParentState) -> dict:
    subgraph_input = {"data": state["data"]}
    result = subgraph.invoke(subgraph_input)
    return {"result": result["output"]}

# 方式 2：直接添加子图作为节点
parent_builder.add_node("subgraph", subgraph)
```

**使用场景**：
- 多智能体系统中每个 Agent 维护独立对话历史
- 避免状态污染
- 更细粒度的状态管理

### 11.6 持久化与人机协作

**Checkpointing（持久化）**：
```python
from langgraph.checkpoint.memory import MemorySaver

# 使用内存持久化（生产环境用数据库）
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 使用 thread_id 作为状态指针
config = {"configurable": {"thread_id": "user_123"}}

# 第一次调用
graph.invoke({"messages": [HumanMessage("Hi")]}, config=config)

# 稍后恢复同一会话
graph.invoke({"messages": [HumanMessage("Continue")]}, config=config)
```

**中断（Interrupts）- 人机协作**：

```python
from langgraph.types import interrupt, Command

def approval_node(state: State) -> Command[Literal["proceed", "cancel"]]:
    """动态暂停等待人工决策"""
    decision = interrupt({
        "question": "Approve this action?",
        "details": state["action_details"],
    })
    
    return Command(goto="proceed" if decision else "cancel")

# 执行到 interrupt
result = graph.invoke(initial_state, config=config)

# 提供人工决策恢复
resumed = graph.invoke(Command(resume=True), config=config)
```

### 11.7 LangGraph 1.0 官方定位

**LangGraph 与 LangChain 的区别**（来自 1.0 发布说明）：

| 特性 | LangChain | LangGraph |
|------|-----------|-----------|
| **定位** | 高级抽象 | 低级编排框架 |
| **适用场景** | 标准工具调用架构 | 高度定制和可控的智能体 |
| **持久化** | 可选 | 内置（Checkpointer） |
| **人机协作** | 有限 | 原生支持（Interrupts） |
| **多智能体** | 基础支持 | 子图支持 |
| **适用场景** | 快速交付、标准模式 | 复杂工作流、长运行应用 |

**使用 LangGraph 当你需要**：
- ✅ 混合确定性和智能体组件的工作流
- ✅ 长运行的业务流程自动化
- ✅ 敏感工作流需要更多监督/人机协作
- ✅ 高度定制或复杂的工作流
- ✅ 需要精确控制延迟和成本的应用

---

## 十二、官方资源链接

### 文档
- [LangGraph 官方文档](https://docs.langchain.com/oss/python/langgraph)
- [LangGraph 1.0 发布博客](https://blog.langchain.com/langchain-langgraph-1dot0/)
- [Graph API 参考](https://docs.langchain.com/oss/python/langgraph/graph-api)

### 源码
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph) - MIT License
- [StateGraph 核心实现](https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/graph/state.py)

### 示例
- [工作流与智能体示例](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [多智能体子图示例](https://github.com/ag-ui-protocol/ag-ui/blob/main/integrations/langgraph/python/examples/agents/subgraphs/agent.py)
- [SQL Agent 示例](https://docs.langchain.com/oss/python/langgraph/sql-agent)
