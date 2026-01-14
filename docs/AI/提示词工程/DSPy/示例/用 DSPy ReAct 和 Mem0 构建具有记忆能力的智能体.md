本教程演示如何结合 DSPy 的 ReAct 框架与 [Mem0](https://docs.mem0.ai/) 的记忆能力，构建能够跨交互记忆信息的智能对话智能体（Agent）。您将学习如何创建能够存储、检索和使用上下文信息以提供个性化和连贯响应的智能体。

## 您将构建的内容

在本教程结束时，您将拥有一个具有记忆能力的智能体，它可以：

- **记住用户偏好**和过去的对话
- **存储和检索**关于用户和主题的**事实信息**
- **利用记忆辅助决策**并提供个性化响应
- **处理**具有上下文感知的**复杂多轮对话**
- **管理不同类型的记忆**（事实、偏好、经历）

## 先决条件

- 对 DSPy 和 ReAct 智能体有基本了解
- 安装了 Python 3.9+
- 您首选的 LLM 提供商的 API 密钥

## 安装和设置

```bash
pip install dspy mem0ai
```

## 第 1 步：了解 Mem0 集成

Mem0 提供了一个记忆层，可以为 AI 智能体存储、搜索和检索记忆。让我们首先了解如何将其与 DSPy 集成：

```python
import dspy
from mem0 import Memory
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

# 配置环境
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# 初始化 Mem0 记忆系统
config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0.1
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small"
        }
    }
}
```

## 第 2 步：创建具有记忆感知能力的工具

让我们创建可以与记忆系统交互的工具：

```python
import datetime

class MemoryTools:
    """与 Mem0 记忆系统交互的工具。"""

    def __init__(self, memory: Memory):
        self.memory = memory

    def store_memory(self, content: str, user_id: str = "default_user") -> str:
        """将信息存储在记忆中。"""
        try:
            self.memory.add(content, user_id=user_id)
            return f"Stored memory: {content}"
        except Exception as e:
            return f"Error storing memory: {str(e)}"

    def search_memories(self, query: str, user_id: str = "default_user", limit: int = 5) -> str:
        """搜索相关记忆。"""
        try:
            results = self.memory.search(query, user_id=user_id, limit=limit)
            if not results:
                return "No relevant memories found."

            memory_text = "Relevant memories found:\n"
            for i, result in enumerate(results["results"]):
                memory_text += f"{i}. {result['memory']}\n"
            return memory_text
        except Exception as e:
            return f"Error searching memories: {str(e)}"

    def get_all_memories(self, user_id: str = "default_user") -> str:
        """获取用户的所有记忆。"""
        try:
            results = self.memory.get_all(user_id=user_id)
            if not results:
                return "No memories found for this user."

            memory_text = "All memories for user:\n"
            for i, result in enumerate(results["results"]):
                memory_text += f"{i}. {result['memory']}\n"
            return memory_text
        except Exception as e:
            return f"Error retrieving memories: {str(e)}"

    def update_memory(self, memory_id: str, new_content: str) -> str:
        """更新现有的记忆。"""
        try:
            self.memory.update(memory_id, new_content)
            return f"Updated memory with new content: {new_content}"
        except Exception as e:
            return f"Error updating memory: {str(e)}"

    def delete_memory(self, memory_id: str) -> str:
        """删除特定的记忆。"""
        try:
            self.memory.delete(memory_id)
            return "Memory deleted successfully."
        except Exception as e:
            return f"Error deleting memory: {str(e)}"

def get_current_time() -> str:
    """获取当前日期和时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

## 第 3 步：构建记忆增强型 ReAct 智能体

现在让我们创建可以使用记忆的主 ReAct 智能体：

```python
class MemoryQA(dspy.Signature):
    """
    You're a helpful assistant and have access to memory method.
    Whenever you answer a user's input, remember to store the information in memory
    so that you can use it later.
    """
    user_input: str = dspy.InputField()
    response: str = dspy.OutputField()

class MemoryReActAgent(dspy.Module):
    """使用 Mem0 记忆功能增强的 ReAct 智能体。"""

    def __init__(self, memory: Memory):
        super().__init__()
        self.memory_tools = MemoryTools(memory)

        # 为 ReAct 创建工具列表
        self.tools = [
            self.memory_tools.store_memory,
            self.memory_tools.search_memories,
            self.memory_tools.get_all_memories,
            get_current_time,
            self.set_reminder,
            self.get_preferences,
            self.update_preferences,
        ]

        # 使用我们的工具初始化 ReAct
        self.react = dspy.ReAct(
            signature=MemoryQA,
            tools=self.tools,
            max_iters=6
        )

    def forward(self, user_input: str):
        """使用具有记忆感知的推理处理用户输入。"""
        
        return self.react(user_input=user_input)

    def set_reminder(self, reminder_text: str, date_time: str = None, user_id: str = "default_user") -> str:
        """为用户设置提醒。"""
        reminder = f"Reminder set for {date_time}: {reminder_text}"
        return self.memory_tools.store_memory(
            f"REMINDER: {reminder}", 
            user_id=user_id
        )

    def get_preferences(self, category: str = "general", user_id: str = "default_user") -> str:
        """获取特定类别的用户偏好。"""
        query = f"user preferences {category}"
        return self.memory_tools.search_memories(
            query=query,
            user_id=user_id
        )

    def update_preferences(self, category: str, preference: str, user_id: str = "default_user") -> str:
        """更新用户偏好。"""
        preference_text = f"User preference for {category}: {preference}"
        return self.memory_tools.store_memory(
            preference_text,
            user_id=user_id
        )
```

## 第 4 步：运行记忆增强型智能体

让我们创建一个简单的界面来与我们的记忆增强型智能体进行交互：

```python
import time
def run_memory_agent_demo():
    """记忆增强型 ReAct 智能体演示。"""

    # 配置 DSPy
    lm = dspy.LM(model='openai/gpt-4o-mini')
    dspy.configure(lm=lm)

    # 初始化记忆系统
    memory = Memory.from_config(config)

    # 创建我们的智能体
    agent = MemoryReActAgent(memory)

    # 演示记忆能力的示例对话
    print("🧠 Memory-Enhanced ReAct Agent Demo")
    print("=" * 50)

    conversations = [
        "Hi, I'm Alice and I love Italian food, especially pasta carbonara.",
        "I'm Alice. I prefer to exercise in the morning around 7 AM.",
        "I'm Alice. What do you remember about my food preferences?",
        "I'm Alice. Set a reminder for me to go grocery shopping tomorrow.",
        "I'm Alice. What are my exercise preferences?",
        "I'm Alice. I also enjoy hiking on weekends.",
        "I'm Alice. What do you know about me so far?"
    ]

    for i, user_input in enumerate(conversations, 1):
        print(f"\n📝 User: {user_input}")

        try:
            response = agent(user_input=user_input)
            print(f"🤖 Agent: {response.response}")
            time.sleep(1)

        except Exception as e:
            print(f"❌ Error: {e}")

# 运行演示
if __name__ == "__main__":
    run_memory_agent_demo()
```

## 预期输出
```
🧠 Memory-Enhanced ReAct Agent Demo
==================================================

📝 User: Hi, I'm Alice and I love Italian food, especially pasta carbonara.

🤖 Agent: Hi Alice! It's great to meet you! Pasta carbonara is a delicious choice. Do you have a favorite place to get it, or do you enjoy making it at home?

📝 User: I'm Alice. I prefer to exercise in the morning around 7 AM.

🤖 Agent: Nice to meet you, Alice! I've noted that you prefer to exercise in the morning around 7 AM. If you have any questions or need tips related to your morning workouts, feel free to ask!

📝 User: I'm Alice. What do you remember about my food preferences?

🤖 Agent: I remember that you love Italian food, and your favorite dish is pasta carbonara!

📝 User: I'm Alice. Set a reminder for me to go grocery shopping tomorrow.

🤖 Agent: I have set a reminder for you, Alice, to go grocery shopping tomorrow.

📝 User: I'm Alice. What are my exercise preferences?

🤖 Agent: Your exercise preference is to work out in the morning around 7 AM.

📝 User: I'm Alice. I also enjoy hiking on weekends.

🤖 Agent: Nice to meet you, Alice! It's great to hear that you enjoy hiking on weekends. If you need any recommendations for trails or tips for your hikes, feel free to ask!

📝 User: I'm Alice. What do you know about me so far?

🤖 Agent: So far, I know that you enjoy hiking on weekends, prefer to exercise in the morning around 7 AM, and love Italian food, especially pasta carbonara. Additionally, you have a reminder set for grocery shopping on October 7, 2023.
```

## 下一步

- 使用数据库（PostgreSQL, MongoDB）**实现记忆持久化**
- **添加记忆分类**和标签以更好地组织
- 为数据管理**创建记忆过期策略**
- 为生产应用**构建多用户记忆隔离**
- **添加记忆分析**和洞察
- **与向量数据库集成**以增强语义搜索
- **实现记忆压缩**以提高长期存储效率

本教程展示了如何利用 Mem0 的记忆能力增强 DSPy 的 ReAct 框架，从而创建能够跨交互学习和记忆信息的智能上下文感知智能体，使其在实际应用中更加有用。
