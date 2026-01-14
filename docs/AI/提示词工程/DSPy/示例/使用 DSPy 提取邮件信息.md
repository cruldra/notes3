本教程演示了如何使用 DSPy 构建一个智能邮件处理系统。我们将创建一个系统，它可以自动从各种类型的邮件中提取关键信息，对邮件意图进行分类，并将数据结构化以便进一步处理。

## 你将构建什么

在本教程结束时，你将拥有一个由 DSPy 驱动的邮件处理系统，它可以：

- **分类邮件类型**（订单确认、支持请求、会议邀请等）
- **提取关键实体**（日期、金额、产品名称、联系方式）
- **确定紧急程度**及所需行动
- **将提取的数据结构化**为一致的格式
- **稳健地处理多种邮件格式**

## 前提条件

- 对 DSPy 模块和签名有基本了解
- 安装了 Python 3.9+
- OpenAI API 密钥（或访问其他支持的 LLM）

## 安装与设置

```bash
pip install dspy
```

<details>
<summary>推荐：设置 MLflow 追踪以了解底层运行情况。</summary>

### MLflow DSPy 集成

<a href="https://mlflow.org/">MLflow</a> 是一个 LLMOps 工具，它与 DSPy 原生集成，并提供可解释性和实验跟踪功能。在本教程中，你可以使用 MLflow 将提示词和优化过程可视化为追踪（traces），以便更好地理解 DSPy 的行为。你可以按照以下四个步骤轻松设置 MLflow。

1. 安装 MLflow

```bash
%pip install mlflow>=3.0.0
```

2. 在单独的终端中启动 MLflow UI
```bash
mlflow ui --port 5000 --backend-store-uri sqlite:///mlruns.db
```

3. 将 notebook 连接到 MLflow
```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DSPy")
```

4. 启用追踪。
```python
mlflow.dspy.autolog()
```


要了解更多关于集成的更多信息，请访问 [MLflow DSPy 文档](https://mlflow.org/docs/latest/llms/dspy/index.html)。
</details>

## 第 1 步：定义数据结构

首先，让我们定义想要从邮件中提取的信息类型：

```python
import dspy
from typing import List, Optional, Literal
from datetime import datetime
from pydantic import BaseModel
from enum import Enum

class EmailType(str, Enum):
    ORDER_CONFIRMATION = "order_confirmation"
    SUPPORT_REQUEST = "support_request"
    MEETING_INVITATION = "meeting_invitation"
    NEWSLETTER = "newsletter"
    PROMOTIONAL = "promotional"
    INVOICE = "invoice"
    SHIPPING_NOTIFICATION = "shipping_notification"
    OTHER = "other"

class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ExtractedEntity(BaseModel):
    entity_type: str
    value: str
    confidence: float
```

## 第 2 步：创建 DSPy 签名

现在让我们为邮件处理管道定义签名：

```python
class ClassifyEmail(dspy.Signature):
    """根据邮件内容对邮件类型和紧急程度进行分类。"""

    email_subject: str = dspy.InputField(desc="邮件的主题行")
    email_body: str = dspy.InputField(desc="邮件的正文内容")
    sender: str = dspy.InputField(desc="邮件发送者信息")

    email_type: EmailType = dspy.OutputField(desc="分类后的邮件类型")
    urgency: UrgencyLevel = dspy.OutputField(desc="邮件的紧急程度")
    reasoning: str = dspy.OutputField(desc="分类的简要解释")

class ExtractEntities(dspy.Signature):
    """从邮件内容中提取关键实体和信息。"""

    email_content: str = dspy.InputField(desc="包含主题和正文的完整邮件内容")
    email_type: EmailType = dspy.InputField(desc="分类后的邮件类型")

    key_entities: list[ExtractedEntity] = dspy.OutputField(desc="提取的实体列表，包含类型、值和置信度")
    financial_amount: Optional[float] = dspy.OutputField(desc="发现的任何货币金额（例如 '$99.99'）")
    important_dates: list[str] = dspy.OutputField(desc="邮件中发现的重要日期列表")
    contact_info: list[str] = dspy.OutputField(desc="提取的相关联系信息")

class GenerateActionItems(dspy.Signature):
    """根据邮件内容和提取的信息确定需要采取的行动。"""

    email_type: EmailType = dspy.InputField()
    urgency: UrgencyLevel = dspy.InputField()
    email_summary: str = dspy.InputField(desc="邮件内容的简要总结")
    extracted_entities: list[ExtractedEntity] = dspy.InputField(desc="邮件中发现的关键实体")

    action_required: bool = dspy.OutputField(desc="是否需要采取任何行动")
    action_items: list[str] = dspy.OutputField(desc="所需具体行动的列表")
    deadline: Optional[str] = dspy.OutputField(desc="行动的截止日期（如果适用）")
    priority_score: int = dspy.OutputField(desc="优先级评分，范围 1-10")

class SummarizeEmail(dspy.Signature):
    """创建一个简洁的邮件内容总结。"""

    email_subject: str = dspy.InputField()
    email_body: str = dspy.InputField()
    key_entities: list[ExtractedEntity] = dspy.InputField()

    summary: str = dspy.OutputField(desc="2-3 句话的邮件要点总结")
```

## 第 3 步：构建邮件处理模块

现在让我们创建主邮件处理模块：

```python
class EmailProcessor(dspy.Module):
    """一个使用 DSPy 的综合邮件处理系统。"""

    def __init__(self):
        super().__init__()

        # 初始化处理组件
        self.classifier = dspy.ChainOfThought(ClassifyEmail)
        self.entity_extractor = dspy.ChainOfThought(ExtractEntities)
        self.action_generator = dspy.ChainOfThought(GenerateActionItems)
        self.summarizer = dspy.ChainOfThought(SummarizeEmail)

    def forward(self, email_subject: str, email_body: str, sender: str = ""):
        """处理邮件并提取结构化信息。"""

        # 第 1 步：分类邮件
        classification = self.classifier(
            email_subject=email_subject,
            email_body=email_body,
            sender=sender
        )

        # 第 2 步：提取实体
        full_content = f"Subject: {email_subject}\n\nFrom: {sender}\n\n{email_body}"
        entities = self.entity_extractor(
            email_content=full_content,
            email_type=classification.email_type
        )

        # 第 3 步：生成总结
        summary = self.summarizer(
            email_subject=email_subject,
            email_body=email_body,
            key_entities=entities.key_entities
        )

        # 第 4 步：确定行动
        actions = self.action_generator(
            email_type=classification.email_type,
            urgency=classification.urgency,
            email_summary=summary.summary,
            extracted_entities=entities.key_entities
        )

        # 第 5 步：结构化结果
        return dspy.Prediction(
            email_type=classification.email_type,
            urgency=classification.urgency,
            summary=summary.summary,
            key_entities=entities.key_entities,
            financial_amount=entities.financial_amount,
            important_dates=entities.important_dates,
            action_required=actions.action_required,
            action_items=actions.action_items,
            deadline=actions.deadline,
            priority_score=actions.priority_score,
            reasoning=classification.reasoning,
            contact_info=entities.contact_info
        )
```

## 第 4 步：运行邮件处理系统

让我们创建一个简单的函数来测试邮件处理系统：

```python
import os
def run_email_processing_demo():
    """邮件处理系统演示。"""
    
    # 配置 DSPy
    lm = dspy.LM(model='openai/gpt-4o-mini')
    dspy.configure(lm=lm)
    os.environ["OPENAI_API_KEY"] = "<YOUR OPENAI KEY>"
    
    # 创建我们的邮件处理器
    processor = EmailProcessor()
    
    # 用于测试的示例邮件
    sample_emails = [
        {
            "subject": "Order Confirmation #12345 - Your MacBook Pro is on the way!",
            "body": """Dear John Smith,

Thank you for your order! We're excited to confirm that your order #12345 has been processed.

Order Details:
- MacBook Pro 14-inch (Space Gray)
- Order Total: $2,399.00
- Estimated Delivery: December 15, 2024
- Tracking Number: 1Z999AA1234567890

If you have any questions, please contact our support team at support@techstore.com.

Best regards,
TechStore Team""",
            "sender": "orders@techstore.com"
        },
        {
            "subject": "URGENT: Server Outage - Immediate Action Required",
            "body": """Hi DevOps Team,

We're experiencing a critical server outage affecting our production environment.

Impact: All users unable to access the platform
Started: 2:30 PM EST

Please join the emergency call immediately: +1-555-123-4567

This is our highest priority.

Thanks,
Site Reliability Team""",
            "sender": "alerts@company.com"
        },
        {
            "subject": "Meeting Invitation: Q4 Planning Session",
            "body": """Hello team,

You're invited to our Q4 planning session.

When: Friday, December 20, 2024 at 2:00 PM - 4:00 PM EST
Where: Conference Room A

Please confirm your attendance by December 18th.

Best,
Sarah Johnson""",
            "sender": "sarah.johnson@company.com"
        }
    ]
    
    # 处理每封邮件并显示结果
    print("🚀 邮件处理演示")
    print("=" * 50)
    
    for i, email in enumerate(sample_emails):
        print(f"\n📧 邮件 {i+1}: {email['subject'][:50]}...")
        
        # 处理邮件
        result = processor(
            email_subject=email["subject"],
            email_body=email["body"],
            sender=email["sender"]
        )
        
        # 显示关键结果
        print(f"   📊 类型: {result.email_type}")
        print(f"   🚨 紧急程度: {result.urgency}")
        print(f"   📝 总结: {result.summary}")
        
        if result.financial_amount:
            print(f"   💰 金额: ${result.financial_amount:,.2f}")
        
        if result.action_required:
            print(f"   ✅ 需要行动: 是")
            if result.deadline:
                print(f"   ⏰ 截止日期: {result.deadline}")
        else:
            print(f"   ✅ 需要行动: 否")

# 运行演示
if __name__ == "__main__":
    run_email_processing_demo()
```

## 预期输出
```
🚀 邮件处理演示
==================================================

📧 邮件 1: Order Confirmation #12345 - Your MacBook Pro is on...
   📊 类型: order_confirmation
   🚨 紧急程度: low
   📝 总结: 邮件确认了 John Smith 的订单 #12345，购买了一台 14 英寸深空灰 MacBook Pro，总金额为 $2,399.00，预计送达日期为 2024 年 12 月 15 日。其中包含追踪号码和客户支持的联系信息。
   💰 金额: $2,399.00
   ✅ 需要行动: 否

📧 邮件 2: URGENT: Server Outage - Immediate Action Required...
   📊 类型: other
   🚨 紧急程度: critical
   📝 总结: 网站可靠性团队报告了一个严重的服务器中断，始于东部时间下午 2:30，导致所有用户无法访问平台。他们要求 DevOps 团队立即加入紧急电话会议以解决该问题。
   ✅ 需要行动: 是
   ⏰ 截止日期: Immediately

📧 邮件 3: Meeting Invitation: Q4 Planning Session...
   📊 类型: meeting_invitation
   🚨 紧急程度: medium
   📝 总结: Sarah Johnson 邀请团队参加 2024 年 12 月 20 日下午 2:00 至 4:00（东部时间）在会议室 A 举行的 Q4 规划会议。请与会者在 12 月 18 日前确认出席。
   ✅ 需要行动: 是
   ⏰ 截止日期: December 18th
```

## 下一步

- **添加更多邮件类型**并优化分类（新闻简报、促销邮件等）
- **添加集成**与邮件提供商（Gmail API, Outlook, IMAP）
- **尝试不同的 LLM** 和优化策略
- **添加多语言支持**以处理国际邮件
- **优化**以提高程序的性能
