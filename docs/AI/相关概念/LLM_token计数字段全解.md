# LLM API 返回的 Token 计数字段全解

调用 OpenAI / Anthropic / DeepSeek 这类 LLM API 时，响应体里几乎都有一个 `usage` 字段，里面塞着各种 `*_tokens`。表面看起来都是计数，但**字段背后的语义并不一致**，搞混会直接导致计费估算错误、上下文使用率指示器画错、甚至 `max_tokens` 设小了空响应这种诡异 bug。这篇笔记把这些字段梳清楚。

## 1. 三个核心字段：input / output / total

无论哪家 provider，最底层都是三个字段：

- **`prompt_tokens`（OpenAI）/ `input_tokens`（Anthropic / langchain）**：这一次调用**模型实际读到的全部输入**的 token 数。
- **`completion_tokens`（OpenAI）/ `output_tokens`（Anthropic / langchain）**：模型这一次**生成出来**的 token 数。
- **`total_tokens`**：上面两者之和。

**容易踩坑的点：`prompt_tokens` 不只是"用户这一句话"。** 它至少包含：

1. system prompt
2. 历史对话（user/assistant 来回的所有轮次）
3. 上一轮模型自己生成的 tool_calls
4. 工具返回的 tool_result（这部分往往非常大，比如读了个 1000 行的文件）
5. 当前这一条用户消息

所以一个多轮对话里，`prompt_tokens` **每一轮都会比上一轮更大**——因为模型必须把所有历史重新"看"一遍才能续写。这是 Transformer 无状态推理的本质决定的，不是 API 设计的偷工减料。

## 2. 缓存命中字段：折扣，不是"不占上下文"

为了缓解上一节那个"历史越聊越贵"的问题，主流 provider 都加了 **prompt caching**：你前缀相同的那一大段（system + 工具定义 + 早期历史）服务端会缓存，下次命中就按折扣价收费。

命名各家不一样：

| Provider | 命中读取 | 首次写入 |
| --- | --- | --- |
| Anthropic | `cache_read_input_tokens` | `cache_creation_input_tokens` |
| OpenAI | `prompt_tokens_details.cached_tokens` | （无显式字段，按命中差额推算） |
| langchain 标准化 | `input_token_details.cache_read` | `input_token_details.cache_creation` |

**最重要的一条认知**：`cache_read_tokens` 是**计费**优惠（Anthropic 大约 10% 价、OpenAI 50% 价），**不是上下文窗口优惠**。这部分内容仍然完整地进入了模型的 attention 视野，仍然占满 200k / 128k 的窗口配额。

所以下面这种推算是**错的**：

```python
# ❌ 错误：以为命中缓存就不占上下文
remaining = ctx_window - (prompt_tokens - cache_read_tokens)
```

正确做法是直接用 `prompt_tokens` 作为分子（见第 5 节）。

## 3. 推理 token：藏在 output 里的隐形大户

OpenAI o 系列、DeepSeek-R1、Claude extended thinking 这类"会思考"的模型，会先生成一段**模型内部独白**再给最终答案。这段独白：

- **算 output**，不算 input；
- **占 `max_tokens` 预算**；
- **照常计费**（而且通常按 output 的高价计）。

字段位置：

- OpenAI：`completion_tokens_details.reasoning_tokens`
- Anthropic（thinking）：`thinking` content block 里的 token，计入 `output_tokens`
- DeepSeek：`reasoning_content` 字段单独出现在消息体里，token 数合并进 `completion_tokens`

**经典 bug**：调用 `o1-mini` 或 `claude-*-thinking` 时设 `max_tokens=64`，结果返回的 `content` 是空字符串。原因是 64 个 output 配额全被 reasoning 吃完了，留给真正答案的额度为 0。给会推理的模型，`max_tokens` 起步至少几千。

## 4. 三家 provider 字段映射表

```jsonc
// OpenAI
{
  "usage": {
    "prompt_tokens": 1523,
    "completion_tokens": 412,
    "total_tokens": 1935,
    "prompt_tokens_details": { "cached_tokens": 1024 },
    "completion_tokens_details": { "reasoning_tokens": 300 }
  }
}
```

```jsonc
// Anthropic
{
  "usage": {
    "input_tokens": 499,
    "output_tokens": 412,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 1024
  }
}
```

```jsonc
// langchain 标准化 (usage_metadata)
{
  "input_tokens": 1523,
  "output_tokens": 412,
  "total_tokens": 1935,
  "input_token_details": { "cache_read": 1024, "cache_creation": 0 }
}
```

注意 Anthropic 的 `input_tokens` 是**已扣除 cache_read 之后的**新增输入量；要还原"模型实际看到的总输入"得加回去：`真实 prompt = input_tokens + cache_read_input_tokens + cache_creation_input_tokens`。OpenAI 的 `prompt_tokens` 则已经是总和，`cached_tokens` 是其中已命中的部分。两家口径反过来的，这是最容易写错统计代码的地方。

我们项目里 `backend/src/luffy_agent/agent/runtime/translator.py` 的 `_extract_usage_metadata` 就在干这件事：把三种来源 normalize 成统一的 `input_tokens / output_tokens / total_tokens / cache_creation_tokens / cache_read_tokens` 五字段，下游 `LlmUsagePayload` 只认这一套口径。新增 provider 时照这个函数加分支即可。

## 5. 上下文窗口 vs 计费 token

这两件事字段一样、口径一致，但**目的完全不同**：

- **上下文窗口**：模型架构上限（Claude Opus 4.x = 200k，GPT-4.1 = 1M，DeepSeek V3 = 128k）。超了 API 直接 400 拒绝。
- **计费 token**：provider 账单口径，按 prompt / output / cached / reasoning 分档计价。

同一个调用，这两个口径数到的总 token 数是一致的——但一个决定**能不能调成功**，一个决定**要付多少钱**。

## 6. 工程实践：会话上下文使用率怎么算

想在 UI 上显示"当前会话已用 76% 上下文"这种指示器：

- **分子**：取最近一次成功调用的 `prompt_tokens`（OpenAI 口径）/ `input_tokens + cache_read + cache_creation`（Anthropic 口径还原）。它就是"模型下一轮将要看到的完整输入"的精确大小。
- **分母**：当前模型的 context window 上限。
- **不要做**：把每轮的 `total_tokens` 累加。`prompt_tokens` 本身已经包含了之前所有轮次的历史，累加等于把历史重复算 N 次，会得到一个远超 100% 的离谱数字。

同理，做"该不该触发上下文压缩"判断时，阈值也应该比对 `prompt_tokens / context_window`，不是累计 token。
