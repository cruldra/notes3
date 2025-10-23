# Rig - Rust 的 AI Agent 框架入门指南

## 📚 概述

**Rig** 是一个用 Rust 构建的模块化、可扩展的 LLM（大语言模型）应用框架，类似于 Python 生态中的 CrewAI，但专注于 Rust 的性能和类型安全优势。

### 核心特性

- ✅ **统一的 LLM 接口** - 支持多个 LLM 提供商（OpenAI、Anthropic、Cohere 等）
- ✅ **Rust 性能优势** - 零成本抽象和内存安全
- ✅ **高级 AI 工作流** - 内置 RAG、多 Agent 系统等预构建组件
- ✅ **类型安全** - 利用 Rust 的强类型系统确保编译时正确性
- ✅ **向量存储集成** - 内置支持 MongoDB、Qdrant 等向量数据库
- ✅ **灵活的嵌入支持** - 易用的 API 处理文本嵌入

### 官方资源

- 🌐 官网: https://rig.rs/
- 📦 GitHub: https://github.com/0xPlaygrounds/rig
- 📖 文档: https://docs.rs/rig-core/latest/rig/

---

## 🚀 快速开始

### 1. 安装

在 `Cargo.toml` 中添加依赖：

```toml
[dependencies]
rig-core = "0.5"
tokio = { version = "1", features = ["full"] }
```

### 2. 基础示例 - 简单的 LLM 调用

```rust
use rig::providers::openai;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // 创建 OpenAI 客户端
    let openai_client = openai::Client::from_env();

    // 创建 Agent
    let agent = openai_client
        .agent("gpt-4")
        .preamble("你是一个友好的助手。")
        .build();

    // 发送提示并获取响应
    let response = agent.prompt("什么是 Rust？").await?;
    
    println!("响应: {}", response);
    
    Ok(())
}
```

### 3. 环境变量配置

```bash
# 设置 OpenAI API Key
export OPENAI_API_KEY="your-api-key-here"
```

---

## 🏗️ 核心概念

### 1. Agent（代理）

Agent 是 Rig 中的核心抽象，代表一个可以与 LLM 交互的智能实体。

```rust
use rig::providers::openai;

let agent = openai_client
    .agent("gpt-4")
    .preamble("你是一个 Rust 专家。")
    .temperature(0.7)
    .max_tokens(1000)
    .build();
```

**Agent 配置选项**:
- `preamble()` - 系统提示词
- `temperature()` - 温度参数（0.0-2.0）
- `max_tokens()` - 最大生成 token 数
- `top_p()` - 核采样参数

### 2. Tools（工具）

工具允许 Agent 执行特定操作，如搜索、计算、API 调用等。

```rust
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Deserialize)]
struct CalculatorInput {
    operation: String,
    a: f64,
    b: f64,
}

#[derive(Debug, thiserror::Error)]
#[error("Calculator error")]
struct CalculatorError;

#[derive(Deserialize, Serialize)]
struct Calculator;

impl Tool for Calculator {
    const NAME: &'static str = "calculator";
    
    type Error = CalculatorError;
    type Args = CalculatorInput;
    type Output = f64;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "calculator".to_string(),
            description: "执行基本数学运算".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": "运算类型: add, subtract, multiply, divide"
                    },
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        match args.operation.as_str() {
            "add" => Ok(args.a + args.b),
            "subtract" => Ok(args.a - args.b),
            "multiply" => Ok(args.a * args.b),
            "divide" => Ok(args.a / args.b),
            _ => Err(CalculatorError),
        }
    }
}

// 使用工具
let agent = openai_client
    .agent("gpt-4")
    .preamble("你是一个数学助手。")
    .tool(Calculator)
    .build();
```

### 3. RAG（检索增强生成）

Rig 提供了强大的 RAG 支持，可以轻松构建基于文档的问答系统。

```rust
use rig::{
    embeddings::EmbeddingsBuilder,
    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
    vector_store::{in_memory_store::InMemoryVectorStore, VectorStore},
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let openai_client = Client::from_env();

    // 创建向量存储
    let mut vector_store = InMemoryVectorStore::default();

    // 准备文档
    let documents = vec![
        "Rust 是一种系统编程语言，注重安全性、并发性和性能。",
        "Rust 的所有权系统可以在编译时防止内存错误。",
        "Cargo 是 Rust 的包管理器和构建工具。",
    ];

    // 创建嵌入并存储
    let embeddings = EmbeddingsBuilder::new(openai_client.embedding_model(TEXT_EMBEDDING_ADA_002))
        .documents(documents)?
        .build()
        .await?;

    vector_store.add_documents(embeddings).await?;

    // 创建 RAG Agent
    let rag_agent = openai_client
        .agent("gpt-4")
        .preamble("根据提供的上下文回答问题。")
        .context(vector_store.vector_search("Rust 的特点", 2).await?)
        .build();

    let response = rag_agent.prompt("Rust 有什么特点？").await?;
    println!("{}", response);

    Ok(())
}
```

---

## 🔧 实用示例

### 示例 1: 文件分析 Agent

```rust
use rig::providers::openai;
use std::fs;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let openai_client = openai::Client::from_env();

    // 读取文件内容
    let file_content = fs::read_to_string("example.rs")?;

    // 创建代码分析 Agent
    let code_analyzer = openai_client
        .agent("gpt-4")
        .preamble("你是一个 Rust 代码审查专家。分析代码并提供改进建议。")
        .build();

    let analysis = code_analyzer
        .prompt(&format!("分析以下代码:\n\n{}", file_content))
        .await?;

    println!("代码分析结果:\n{}", analysis);

    Ok(())
}
```

### 示例 2: 多 Agent 协作系统

```rust
use rig::providers::openai;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = openai::Client::from_env();

    // 研究员 Agent
    let researcher = client
        .agent("gpt-4")
        .preamble("你是一个研究员，负责收集和整理信息。")
        .build();

    // 作家 Agent
    let writer = client
        .agent("gpt-4")
        .preamble("你是一个技术作家，负责将研究结果写成文章。")
        .build();

    // 编辑 Agent
    let editor = client
        .agent("gpt-4")
        .preamble("你是一个编辑，负责审查和改进文章。")
        .build();

    // 工作流程
    let topic = "Rust 的异步编程";

    // 1. 研究阶段
    let research = researcher
        .prompt(&format!("研究主题: {}", topic))
        .await?;

    // 2. 写作阶段
    let draft = writer
        .prompt(&format!("基于以下研究写一篇文章:\n{}", research))
        .await?;

    // 3. 编辑阶段
    let final_article = editor
        .prompt(&format!("审查并改进以下文章:\n{}", draft))
        .await?;

    println!("最终文章:\n{}", final_article);

    Ok(())
}
```

### 示例 3: 带工具的智能助手

```rust
use rig::providers::openai;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

// 天气查询工具
#[derive(Deserialize, Serialize)]
struct WeatherTool;

#[derive(Deserialize)]
struct WeatherArgs {
    city: String,
}

impl Tool for WeatherTool {
    const NAME: &'static str = "get_weather";
    
    type Error = anyhow::Error;
    type Args = WeatherArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "get_weather".to_string(),
            description: "获取指定城市的天气信息".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": ["city"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        // 模拟天气 API 调用
        Ok(format!("{}的天气: 晴天，温度 25°C", args.city))
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = openai::Client::from_env();

    let assistant = client
        .agent("gpt-4")
        .preamble("你是一个智能助手，可以查询天气信息。")
        .tool(WeatherTool)
        .build();

    let response = assistant
        .prompt("北京今天天气怎么样？")
        .await?;

    println!("{}", response);

    Ok(())
}
```

---

## 📊 Rig vs CrewAI 对比

| 特性 | Rig (Rust) | CrewAI (Python) |
|------|-----------|----------------|
| **语言** | Rust | Python |
| **性能** | 极高（编译型） | 中等（解释型） |
| **类型安全** | 编译时检查 | 运行时检查 |
| **内存安全** | 所有权系统 | GC |
| **并发模型** | async/await | asyncio |
| **生态成熟度** | 发展中 | 成熟 |
| **学习曲线** | 陡峭 | 平缓 |
| **适用场景** | 高性能、系统级 | 快速原型、数据科学 |

---

## 🎯 最佳实践

### 1. 错误处理

```rust
use anyhow::Result;

async fn safe_agent_call() -> Result<String> {
    let client = openai::Client::from_env();
    let agent = client.agent("gpt-4").build();
    
    let response = agent
        .prompt("你好")
        .await
        .map_err(|e| anyhow::anyhow!("Agent 调用失败: {}", e))?;
    
    Ok(response)
}
```

### 2. 配置管理

```rust
use serde::Deserialize;

#[derive(Deserialize)]
struct Config {
    model: String,
    temperature: f32,
    max_tokens: u32,
}

impl Config {
    fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&content)?)
    }
}
```

### 3. 日志和监控

```rust
use tracing::{info, error};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt::init();
    
    info!("启动 Agent 系统");
    
    let client = openai::Client::from_env();
    let agent = client.agent("gpt-4").build();
    
    match agent.prompt("测试").await {
        Ok(response) => {
            info!("Agent 响应成功");
            println!("{}", response);
        }
        Err(e) => {
            error!("Agent 调用失败: {}", e);
        }
    }
    
    Ok(())
}
```

---

## 🔗 相关资源

- [Rig 官方文档](https://docs.rs/rig-core/)
- [Rig GitHub 仓库](https://github.com/0xPlaygrounds/rig)
- [Rig 示例集合](https://github.com/0xPlaygrounds/awesome-rig)
- [Rust 异步编程](https://rust-lang.github.io/async-book/)

---

## 📝 总结

Rig 是 Rust 生态中构建 AI Agent 应用的强大框架，它结合了 Rust 的性能优势和现代 LLM 应用的需求。虽然学习曲线较陡，但对于需要高性能、类型安全的 AI 应用场景，Rig 是一个优秀的选择。

**适合使用 Rig 的场景**:
- 需要高性能的 AI 应用
- 系统级 AI 集成
- 对类型安全有严格要求
- 需要细粒度控制的场景

**推荐先学习 CrewAI 的场景**:
- 快速原型开发
- 数据科学项目
- Python 技术栈
- 团队熟悉 Python

