# BDD、TDD、DDD 深度解析：从需求到架构的完整方法论

---

## 引言

在现代软件开发中，我们经常会听到三个缩写：**BDD**、**TDD**、**DDD**。它们看似相似，实则解决的是完全不同层面的问题，但又可以协同工作，形成一套完整的软件开发方法论。

```
用户 SOP → PRD → BDD → DDD → 代码
(业务现实)  (需求)  (行为)  (模型)  (实现)
```

---

## 一、TDD（测试驱动开发）

### 1.1 什么是 TDD？

**Test-Driven Development（测试驱动开发）** 是一种**开发实践**，核心思想是：

> **先写测试，再写代码**

### 1.2 TDD 的工作循环（Red-Green-Refactor）

```
Red（红）     →    Green（绿）    →    Refactor（重构）
写一个失败的测试   →    写代码让测试通过   →    优化代码
```

**具体步骤：**

1. **Red**：编写一个会失败的测试（描述你想要的行为）
2. **Green**：编写最简代码使测试通过
3. **Refactor**：重构代码，保持测试通过
4. 循环往复

### 1.3 TDD 示例

```python
# 1. Red: 先写测试
class TestCalculator:
    def test_add(self):
        calc = Calculator()
        result = calc.add(2, 3)
        assert result == 5  # 此时还没有 Calculator 类，测试会失败

# 2. Green: 实现代码
class Calculator:
    def add(self, a, b):
        return a + b

# 3. Refactor: 优化代码（此时可能不需要）
```

### 1.4 TDD 的核心价值

| 优势 | 说明 |
|------|------|
| **设计导向** | 迫使你先思考接口设计 |
| **高覆盖率** | 保证大部分代码都被测试覆盖 |
| **安全重构** | 测试作为安全网，大胆重构 |
| **需求明确** | 测试就是需求的可执行版本 |

---

## 二、BDD（行为驱动开发）

### 2.1 什么是 BDD？

**Behavior-Driven Development（行为驱动开发）** 是 TDD 的演进版本，但更强调：

> **用业务语言描述系统行为**

BDD 是 TDD 在**业务层面**的延伸，它关注**系统应该做什么**。

### 2.2 BDD 的核心语法（Given-When-Then）

```gherkin
Feature: 用户支付功能

  Scenario: 用户成功支付订单
    Given 用户账户余额为 100 元
    And 商品价格为 30 元
    When 用户发起支付
    Then 支付应该成功
    And 账户余额应该为 70 元
    And 订单状态应该为"已支付"
```

### 2.3 BDD 的工作流程

```
产品经理         业务分析师         开发人员         测试人员
   │               │               │               │
   └───────────────┴───────────────┴───────────────┘
                     │
                     ▼
            定义验收标准（Acceptance Criteria）
                     │
                     ▼
            编写场景描述（Gherkin 语法）
                     │
                     ▼
            自动化测试（Cucumber/Behave 等）
                     │
                     ▼
            开发实现（让测试通过）
```

### 2.4 TDD vs BDD

| 维度 | TDD | BDD |
|------|-----|-----|
| **关注点** | 代码单元的正确性 | 系统行为的正确性 |
| **参与者** | 开发者 | 产品经理 + 业务 + 开发 + 测试 |
| **语言** | 技术语言（代码） | 业务语言（Given-When-Then） |
| **范围** | 单元测试 | 验收测试/集成测试 |
| **关系** | 基础实践 | TDD 的业务层扩展 |

### 2.5 BDD 示例（真实场景）

```gherkin
Feature: 智能消防告警系统

  Scenario: 烟雾报警触发视频复核
    Given 餐饮区 A 的烟感节点状态为"活跃"
    And 该节点已关联摄像头 C-01
    And 摄像头 C-01 状态为"在线"

    When 智能推理子系统接收到该节点持续 3 秒的"高浓度烟雾"数据帧

    Then 系统应立即生成一条"待复核告警"
    And 告警级别应为"紧急"
    And 前端监控大屏应自动弹出摄像头 C-01 的实时视频流
    And 应启动 10 秒倒计时提醒
    And 应记录事件发生时间为当前时间戳

  Scenario: 误报处理
    Given 存在一条待复核告警

    When 值班员点击"误报"按钮

    Then 告警状态应变为"已关闭"
    And 应记录处理人为当前登录用户
    And 应记录处理原因为"误报"
```

---

## 三、DDD（领域驱动设计）

### 3.1 什么是 DDD？

**Domain-Driven Design（领域驱动设计）** 是**软件设计思想**，解决的是：

> **复杂业务如何建模和架构**

DDD 关注的是**系统内部怎么实现业务**，让代码结构能准确反映业务领域。

### 3.2 DDD 的核心概念

```
┌─────────────────────────────────────────────────────────────┐
│                      Bounded Context                        │
│                      (限界上下文)                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────────┐                                          │
│   │  Entity      │  ← 有唯一标识，生命周期内状态可变              │
│   │  (实体)      │     例：Order、User                        │
│   └──────────────┘                                          │
│                                                              │
│   ┌──────────────┐                                          │
│   │ Value Object │  ← 无唯一标识，不可变，通过属性值判断相等      │
│   │ (值对象)     │     例：Money、Address、TelemetryData      │
│   └──────────────┘                                          │
│                                                              │
│   ┌──────────────┐                                          │
│   │ Aggregate    │  ← 一组关联对象的集合，有根实体              │
│   │ (聚合)       │     例：Order 包含 OrderItem               │
│   └──────────────┘                                          │
│                                                              │
│   ┌──────────────┐                                          │
│   │ Repository   │  ← 聚合的持久化抽象                        │
│   │ (仓储)       │     例：OrderRepository                   │
│   └──────────────┘                                          │
│                                                              │
│   ┌──────────────┐                                          │
│   │ Domain       │  ← 业务逻辑不适合放在实体时               │
│   │ Service      │     例：PaymentService                    │
│   │ (领域服务)   │                                          │
│   └──────────────┘                                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 从 BDD 到 DDD 的推导

**BDD 场景 → DDD 模型**

```gherkin
# BDD 场景
Given 餐饮区 A 的烟感节点状态为"活跃"
When 智能推理子系统接收到该节点持续 3 秒的"高浓度烟雾"数据帧
Then 系统应立即生成一条"待复核告警"
```

```python
# 推导出的 DDD 领域模型

# 实体 (Entity)
class SensorNode:
    """传感器节点 - 有唯一标识，状态可变"""
    node_id: str           # 唯一标识
    location: str          # 位置：餐饮区 A
    status: NodeStatus     # 活跃/离线/故障
    camera_id: str         # 关联摄像头
    
    def is_active(self) -> bool:
        return self.status == NodeStatus.ACTIVE

# 值对象 (Value Object)
class TelemetryData:
    """环境数据帧 - 无标识，不可变"""
    sensor_id: str
    smoke_density: float   # 烟雾浓度
    temperature: float
    timestamp: datetime
    
    def is_high_smoke(self, threshold: float = 0.8) -> bool:
        return self.smoke_density > threshold

# 聚合根 (Aggregate Root)
class FireAlert:
    """消防告警 - 聚合根"""
    alert_id: str
    sensor_node: SensorNode
    telemetry_data: TelemetryData
    status: AlertStatus    # 待复核/已确认/已关闭
    level: AlertLevel      # 紧急/一般
    created_at: datetime
    
    def confirm(self, operator: str) -> None:
        """确认告警"""
        self.status = AlertStatus.CONFIRMED
        self.record_operation(operator, "确认告警")
    
    def close(self, operator: str, reason: str) -> None:
        """关闭告警（误报）"""
        self.status = AlertStatus.CLOSED
        self.close_reason = reason
        self.record_operation(operator, f"关闭告警: {reason}")

# 领域服务
class FireAlertService:
    """告警处理服务"""
    
    def create_alert_from_telemetry(
        self, 
        sensor: SensorNode, 
        telemetry: TelemetryData
    ) -> FireAlert:
        """根据环境数据创建告警"""
        if not sensor.is_active():
            raise DomainError("传感器未激活")
        
        if not telemetry.is_high_smoke():
            return None  # 不创建告警
        
        return FireAlert(
            alert_id=generate_id(),
            sensor_node=sensor,
            telemetry_data=telemetry,
            status=AlertStatus.PENDING_CONFIRMATION,
            level=AlertLevel.EMERGENCY,
            created_at=now()
        )
```

### 3.4 限界上下文（Bounded Context）

```
┌────────────────────────────────────────────────────────────────┐
│                     智能监控系统                                 │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐                                          │
│  │ 设备接入上下文    │  ← 传感器注册、数据采集、设备管理         │
│  │ Device Context   │                                          │
│  └──────────────────┘                                          │
│            │                                                    │
│            │ Integration Event: SensorDataReceived              │
│            ▼                                                    │
│  ┌──────────────────┐                                          │
│  │ 智能推理上下文    │  ← AI模型、异常检测、风险评估             │
│  │ AI Context       │                                          │
│  └──────────────────┘                                          │
│            │                                                    │
│            │ Integration Event: AnomalyDetected                 │
│            ▼                                                    │
│  ┌──────────────────┐                                          │
│  │ 告警响应上下文    │  ← 告警生成、通知发送、处置流程           │
│  │ Alert Context    │                                          │
│  └──────────────────┘                                          │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## 四、三者关系：TDD、BDD、DDD 如何协作

### 4.1 层级关系

```
┌─────────────────────────────────────────────────────────────┐
│                      业务目标                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      BDD                                    │
│         解决：系统应该做什么行为？                             │
│         输出：验收场景（Given-When-Then）                     │
│         参与者：产品 + 业务 + 开发 + 测试                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      DDD                                    │
│         解决：业务如何建模？                                  │
│         输出：领域模型（实体/值对象/聚合/限界上下文）           │
│         参与者：业务专家 + 架构师                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      TDD                                    │
│         解决：代码如何正确实现？                              │
│         输出：单元测试 + 可运行的代码                          │
│         参与者：开发者                                       │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 完整工作流示例

**场景：开发一个订单支付功能**

**Step 1: BDD - 定义业务行为**

```gherkin
Feature: 订单支付

  Scenario: 使用账户余额支付
    Given 用户"张三"的账户余额为 100 元
    And 有一个待支付订单，金额为 30 元
    When 用户选择"余额支付"
    Then 支付应该成功
    And 订单状态应该变为"已支付"
    And 账户余额应该变为 70 元
    And 应该生成一条支付流水记录
```

**Step 2: DDD - 领域建模**

```python
# 识别领域对象
- Order (聚合根)
- Account (聚合根)  
- Payment (聚合根)
- Money (值对象)

# 确定限界上下文
- Order Context (订单上下文)
- Payment Context (支付上下文)
- Account Context (账户上下文)
```

**Step 3: TDD - 代码实现**

```python
# 测试先行
class TestPayment:
    def test_pay_with_balance(self):
        # Given
        account = Account(user_id="张三", balance=Money(100))
        order = Order(id="ORDER-001", amount=Money(30))
        service = PaymentService()
        
        # When
        result = service.pay_with_balance(account, order)
        
        # Then
        assert result.success is True
        assert order.status == OrderStatus.PAID
        assert account.balance == Money(70)
        assert len(result.transaction) == 1
```

### 4.3 关键区别总结

| 维度 | TDD | BDD | DDD |
|------|-----|-----|-----|
| **解决什么问题** | 代码怎么写正确 | 系统应该做什么 | 业务怎么建模 |
| **关注层面** | 代码单元 | 系统行为 | 领域架构 |
| **输入是什么** | 方法签名 | 业务场景 | 业务概念 |
| **输出是什么** | 测试 + 代码 | 验收标准 | 领域模型 |
| **核心思想** | 测试先行 | 业务语言 | 统一语言 |
| **常用工具** | JUnit/Pytest | Cucumber/Behave | - |
| **参与者** | 开发者 | 全员 | 业务专家+架构师 |

---

## 五、实战：完整示例

### 5.1 业务场景

开发一个**电商促销系统**，实现满减功能：
- 满 100 减 10
- 满 200 减 30
- 满 500 减 100

### 5.2 BDD 场景定义

```gherkin
Feature: 订单满减促销

  Background:
    Given 存在以下满减规则：
      | 满额条件 | 优惠金额 |
      | 100     | 10      |
      | 200     | 30      |
      | 500     | 100     |

  Scenario: 订单金额满足满 100 减 10
    Given 购物车商品总价为 150 元
    When 应用满减优惠
    Then 优惠后金额应为 140 元
    And 应用的规则应为"满 100 减 10"

  Scenario: 订单金额满足满 200 减 30
    Given 购物车商品总价为 250 元
    When 应用满减优惠
    Then 优惠后金额应为 220 元
    And 应用的规则应为"满 200 减 30"

  Scenario: 订单金额不满足任何满减条件
    Given 购物车商品总价为 50 元
    When 应用满减优惠
    Then 优惠后金额应为 50 元
    And 不应应用任何优惠规则

  Scenario: 订单金额恰好等于满减条件
    Given 购物车商品总价为 200 元
    When 应用满减优惠
    Then 优惠后金额应为 170 元
    And 应用的规则应为"满 200 减 30"
```

### 5.3 DDD 领域建模

```python
from dataclasses import dataclass
from typing import List, Optional
from decimal import Decimal
from enum import Enum

# ==================== 值对象 ====================

@dataclass(frozen=True)
class Money:
    """金额值对象 - 不可变"""
    amount: Decimal
    currency: str = "CNY"
    
    def subtract(self, other: 'Money') -> 'Money':
        return Money(self.amount - other.amount, self.currency)
    
    def __ge__(self, other: 'Money') -> bool:
        return self.amount >= other.amount

# ==================== 实体 ====================

class PromotionRule:
    """促销规则 - 实体"""
    
    def __init__(self, rule_id: str, min_amount: Money, discount: Money):
        self.rule_id = rule_id
        self.min_amount = min_amount  # 满额条件
        self.discount = discount      # 优惠金额
    
    def is_applicable(self, order_amount: Money) -> bool:
        """检查规则是否适用于该订单金额"""
        return order_amount >= self.min_amount

# ==================== 聚合根 ====================

class ShoppingCart:
    """购物车 - 聚合根"""
    
    def __init__(self, cart_id: str):
        self.cart_id = cart_id
        self.items: List[CartItem] = []
        self.applied_promotion: Optional[AppliedPromotion] = None
    
    def add_item(self, product_id: str, price: Money, quantity: int):
        self.items.append(CartItem(product_id, price, quantity))
    
    def calculate_subtotal(self) -> Money:
        """计算商品总价（未优惠前）"""
        total = sum(
            item.price.amount * item.quantity 
            for item in self.items
        )
        return Money(total)
    
    def apply_promotion(self, promotion_engine: 'PromotionEngine'):
        """应用最优促销"""
        subtotal = self.calculate_subtotal()
        best_promotion = promotion_engine.find_best_promotion(subtotal)
        
        if best_promotion:
            self.applied_promotion = AppliedPromotion(
                rule_id=best_promotion.rule_id,
                description=f"满 {best_promotion.min_amount.amount} 减 {best_promotion.discount.amount}",
                discount=best_promotion.discount,
                final_amount=subtotal.subtract(best_promotion.discount)
            )
        else:
            self.applied_promotion = None
    
    def get_final_amount(self) -> Money:
        """获取优惠后金额"""
        if self.applied_promotion:
            return self.applied_promotion.final_amount
        return self.calculate_subtotal()

@dataclass
class CartItem:
    """购物车项"""
    product_id: str
    price: Money
    quantity: int

@dataclass
class AppliedPromotion:
    """已应用的促销"""
    rule_id: str
    description: str
    discount: Money
    final_amount: Money

# ==================== 领域服务 ====================

class PromotionEngine:
    """促销引擎 - 领域服务"""
    
    def __init__(self):
        self.rules: List[PromotionRule] = []
    
    def add_rule(self, rule: PromotionRule):
        """添加促销规则"""
        self.rules.append(rule)
        # 按优惠力度降序排列，优先应用最大优惠
        self.rules.sort(key=lambda r: r.discount.amount, reverse=True)
    
    def find_best_promotion(self, order_amount: Money) -> Optional[PromotionRule]:
        """找到适用于该订单的最优促销规则"""
        applicable_rules = [
            rule for rule in self.rules 
            if rule.is_applicable(order_amount)
        ]
        
        if not applicable_rules:
            return None
        
        # 返回优惠最大的规则（已排序，第一个是最大优惠）
        return applicable_rules[0]
```

### 5.4 TDD 测试实现

```python
import pytest
from decimal import Decimal

class TestPromotionEngine:
    """满减引擎测试"""
    
    @pytest.fixture
    def promotion_engine(self):
        """设置满减规则"""
        engine = PromotionEngine()
        engine.add_rule(PromotionRule("RULE-100", Money(Decimal("100")), Money(Decimal("10"))))
        engine.add_rule(PromotionRule("RULE-200", Money(Decimal("200")), Money(Decimal("30"))))
        engine.add_rule(PromotionRule("RULE-500", Money(Decimal("500")), Money(Decimal("100"))))
        return engine
    
    def test_promotion_100_10(self, promotion_engine):
        """满 100 减 10"""
        # Given
        cart = ShoppingCart("CART-001")
        cart.add_item("P001", Money(Decimal("150")), 1)
        
        # When
        cart.apply_promotion(promotion_engine)
        
        # Then
        assert cart.get_final_amount() == Money(Decimal("140"))
        assert cart.applied_promotion.rule_id == "RULE-100"
        assert "满 100 减 10" in cart.applied_promotion.description
    
    def test_promotion_200_30(self, promotion_engine):
        """满 200 减 30"""
        # Given
        cart = ShoppingCart("CART-002")
        cart.add_item("P002", Money(Decimal("250")), 1)
        
        # When
        cart.apply_promotion(promotion_engine)
        
        # Then
        assert cart.get_final_amount() == Money(Decimal("220"))
        assert cart.applied_promotion.rule_id == "RULE-200"
    
    def test_no_applicable_promotion(self, promotion_engine):
        """不满足任何条件"""
        # Given
        cart = ShoppingCart("CART-003")
        cart.add_item("P003", Money(Decimal("50")), 1)
        
        # When
        cart.apply_promotion(promotion_engine)
        
        # Then
        assert cart.get_final_amount() == Money(Decimal("50"))
        assert cart.applied_promotion is None
    
    def test_exact_threshold(self, promotion_engine):
        """恰好等于条件"""
        # Given
        cart = ShoppingCart("CART-004")
        cart.add_item("P004", Money(Decimal("200")), 1)
        
        # When
        cart.apply_promotion(promotion_engine)
        
        # Then
        assert cart.get_final_amount() == Money(Decimal("170"))
        assert cart.applied_promotion.rule_id == "RULE-200"
    
    def test_best_promotion_selected(self, promotion_engine):
        """选择最优优惠（500元应使用满500减100，而不是满200减30）"""
        # Given
        cart = ShoppingCart("CART-005")
        cart.add_item("P005", Money(Decimal("500")), 1)
        
        # When
        cart.apply_promotion(promotion_engine)
        
        # Then
        assert cart.get_final_amount() == Money(Decimal("400"))
        assert cart.applied_promotion.rule_id == "RULE-500"
```

---

## 六、常见误区

### ❌ 误区 1：BDD 和 DDD 是替代关系

**真相**：
- BDD 解决需求沟通和行为验证
- DDD 解决复杂业务建模
- 两者互补，一起使用

### ❌ 误区 2：TDD 只写单元测试

**真相**：
- TDD 可以应用于任何测试级别
- 单元测试、集成测试、验收测试都可以 TDD
- BDD 本质上是验收级别的 TDD

### ❌ 误区 3：用了 DDD 就是把代码分成三层

**真相**：
```
❌ 错误理解：
Controller
   ↓
Service
   ↓
Repository

✅ 正确理解：
DDD = 统一语言 + 限界上下文 + 聚合设计 + 领域服务
     远不止分层！
```

### ❌ 误区 4：所有项目都需要 BDD + DDD + TDD

**真相**：
- 简单项目：TDD 即可
- 需要多方协作：加上 BDD
- 复杂业务系统：再加 DDD
- 根据项目特点选择，不要过度设计

---

## 七、最佳实践建议

### 7.1 如何选择使用哪种方法？

```
项目类型                           推荐方法
─────────────────────────────────────────────────────────
简单工具/脚本                      TDD
标准 CRUD 应用                     TDD + 少量 BDD
多团队协作项目                     TDD + BDD
复杂业务系统（金融/电商）           TDD + BDD + DDD
遗留系统改造                       BDD（先补验收测试）+ 逐步 DDD
```

### 7.2 实施顺序建议

**新项目**：
```
Week 1-2:  BDD（定义核心业务场景）
Week 2-3:  DDD（领域建模、划分限界上下文）
Week 3+:   TDD（迭代开发，红绿重构）
```

**现有项目**：
```
Phase 1: 用 BDD 补验收测试（建立安全网）
Phase 2: 识别核心业务，提取领域模型（DDD）
Phase 3: 重构代码，应用 TDD 开发新功能
```

### 7.3 团队分工

| 角色 | BDD | DDD | TDD |
|------|-----|-----|-----|
| 产品经理 | ✅ 编写场景 | ✅ 参与统一语言 | ❌ |
| 业务分析师 | ✅ 主导 | ✅ 核心参与者 | ❌ |
| 架构师 | ✅ 评审 | ✅ 主导建模 | ⚠️ 评审 |
| 开发人员 | ✅ 实现步骤定义 | ✅ 实现模型 | ✅ 主导 |
| 测试人员 | ✅ 自动化测试 | ❌ | ✅ 编写测试 |

---

## 八、总结

### 8.1 核心记忆点

```
TDD = 怎么开发（代码层面）
      ↓ 先写测试，再写代码

BDD = 系统应该做什么（业务层面）
      ↓ Given-When-Then 描述行为

DDD = 业务怎么建模（架构层面）
      ↓ 实体、值对象、聚合、限界上下文
```

### 8.2 三者关系图解

```
┌────────────────────────────────────────────────────────────────┐
│                          业务需求                               │
└────────────────────────────────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │     BDD      │   │     DDD      │   │     TDD      │
    │              │   │              │   │              │
    │ 行为驱动开发  │   │ 领域驱动设计  │   │ 测试驱动开发  │
    │              │   │              │   │              │
    │ 回答：应该    │   │ 回答：怎么    │   │ 回答：如何    │
    │ 做什么？      │   │ 建模？        │   │ 实现？        │
    │              │   │              │   │              │
    │ Given-When-  │   │ 聚合/实体/   │   │ Red-Green-   │
    │ Then         │   │ 限界上下文   │   │ Refactor     │
    └──────────────┘   └──────────────┘   └──────────────┘
            │                   │                   │
            └───────────────────┼───────────────────┘
                                ▼
                    ┌──────────────────┐
                    │   高质量软件      │
                    │  满足业务需求     │
                    │  易维护可扩展     │
                    └──────────────────┘
```

### 8.3 一句话总结

> **BDD 定义了做什么，DDD 设计了怎么做，TDD 保证了做的对。**

三者协同，构成了从业务需求到代码实现的完整方法论链。

---

## 九、与 DDD 同级的软件设计思想

除了 BDD、TDD、DDD 之外，软件工程中还有许多同样重要的设计思想和方法论。它们与 DDD 处于**同一层级**，都是为了**管理复杂系统的设计与结构**，但关注点各不相同。

```
软件工程方法论层次

设计原则层
└─ SOLID

软件建模层
└─ DDD (Domain-Driven Design)

架构模式层
├─ Clean Architecture (整洁架构)
├─ Hexagonal Architecture (六边形架构)
├─ Onion Architecture (洋葱架构)
└─ Layered Architecture (分层架构)

系统架构层
├─ Microservices Architecture (微服务架构)
└─ Event-Driven Architecture (事件驱动架构)
```

### 9.1 SOLID 设计原则

**SOLID** 是面向对象设计的五大基本原则，是所有软件设计的基础。

| 缩写 | 全称 | 中文 | 核心思想 |
|------|------|------|----------|
| **S** | Single Responsibility Principle | 单一职责原则 | 一个类只做一件事 |
| **O** | Open/Closed Principle | 开闭原则 | 对扩展开放，对修改关闭 |
| **L** | Liskov Substitution Principle | 里氏替换原则 | 子类可以替换父类而不影响程序 |
| **I** | Interface Segregation Principle | 接口隔离原则 | 客户端不应依赖它不需要的接口 |
| **D** | Dependency Inversion Principle | 依赖倒置原则 | 依赖抽象而非具体实现 |

**为什么重要？**
- DDD 的代码实践大量使用这些原则
- 是 Clean Architecture、Hexagonal Architecture 的理论基础
- 帮助写出高内聚、低耦合的代码

**示例：依赖倒置原则**

```python
# ❌ 错误：高层依赖低层具体实现
class OrderService:
    def __init__(self):
        self.mysql_repo = MySQLOrderRepository()  # 直接依赖具体实现

# ✅ 正确：依赖抽象（接口）
class OrderService:
    def __init__(self, repo: OrderRepository):  # 依赖接口
        self.repo = repo
```

### 9.2 Clean Architecture（整洁架构）

由 Robert C. Martin（Uncle Bob）提出，是 DDD 项目常用的架构模式。

**核心思想：**
```
┌───────────────────────────────────────┐
│      Frameworks & Drivers             │
│   (Web Frameworks, DB, External APIs) │
└───────────────────────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│       Interface Adapters              │
│   (Controllers, Presenters, Gateways) │
└───────────────────────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│         Use Cases                     │
│   (Application Business Rules)        │
└───────────────────────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│         Entities                      │
│   (Enterprise Business Rules)         │
└───────────────────────────────────────┘
```

**依赖规则：**
- **依赖方向只能向内**（外层依赖内层）
- 内层不知道外层的存在
- 业务逻辑（Entity、Use Case）在中心，最纯净

**解决的问题：**
- 框架耦合（UI/数据库变化不影响业务逻辑）
- 业务逻辑分散在各层
- 难以单元测试（需要启动整个应用）

**与 DDD 的关系：**
```
DDD：定义业务模型（Entities、Value Objects）
     ↓
Clean Architecture：定义代码结构（分层、依赖方向）
     ↓
合起来：业务模型位于架构中心，不依赖任何框架
```

### 9.3 Hexagonal Architecture（六边形架构 / 端口适配器架构）

由 Alistair Cockburn 提出，也叫 **Ports and Adapters Architecture**。

**核心思想：**
```
              ┌──────────────────┐
              │                  │
   ┌─────────│      Domain      │──────────┐
   │         │   (核心业务逻辑)   │          │
   │         │                  │          │
   │         └────────┬─────────┘          │
   │                  │                    │
   │       ┌──────────┴──────────┐         │
   │       │                     │         │
   ▼       ▼                     ▼         ▼
┌──────┐ ┌──────┐           ┌──────┐ ┌──────┐
│  API │ │  CLI │           │ MySQL│ │Redis │
│  Port│ │  Port│           │Adapter│Adapter│
└──────┘ └──────┘           └──────┘ └──────┘
```

**关键概念：**
- **Domain（领域）**：核心业务逻辑，位于中心
- **Port（端口）**：领域对外暴露的接口（输入/输出）
- **Adapter（适配器）**：外部系统通过适配器与领域交互

**解决的问题：**
- 外部系统（数据库、UI、第三方 API）污染业务逻辑
- 难以替换技术实现（如 MySQL → MongoDB）
- 测试困难（需要真实数据库）

**与 DDD 的关系：**
```python
# DDD 定义领域模型
class Order:
    def confirm(self): ...

# Hexagonal Architecture 定义如何暴露领域能力
class OrderServicePort:  # 端口（接口）
    @abstractmethod
    def confirm_order(self, order_id: str): ...

class OrderAPIAdapter:   # 适配器
    def __init__(self, service: OrderServicePort):
        self.service = service
    
    def post(self, order_id):  # HTTP API 适配
        self.service.confirm_order(order_id)

class OrderRepositoryPort:  # 端口
    @abstractmethod
    def save(self, order: Order): ...

class OrderMySQLAdapter:    # 适配器
    def save(self, order: Order):
        # 具体的 MySQL 存储实现
        pass
```

### 9.4 Onion Architecture（洋葱架构）

由 Jeffrey Palermo 提出，与 Clean Architecture 非常相似。

**结构：**
```
        ┌───────────────────────────────────┐
       /      Infrastructure Layer          \
      /  (DB, Web, External Services)       \
     /─────────────────────────────────────────\
    /         Application Layer               \
   /    (Use Cases, Application Services)     \
  /─────────────────────────────────────────────\
 /             Domain Layer                      \
/         (Entities, Value Objects)              \
\─────────────────────────────────────────────────/
 \            Core Layer (Domain Services)       /
  \                                             /
   └───────────────────────────────────────────┘
```

**原则：**
- 所有依赖都指向中心
- **Domain Layer** 是核心，不依赖任何外部框架
- **Application Layer** 协调用例，也不依赖框架
- **Infrastructure Layer** 包含所有技术细节

**与 DDD 的关系：**
- Onion Architecture 是 .NET DDD 项目最常用的架构
- Domain Layer = DDD 的领域模型
- 保证领域模型的纯净，不受技术选型影响

### 9.5 Microservices Architecture（微服务架构）

**核心思想：**
> 将大型应用拆分为小型、独立部署的服务

**与 DDD 的关系：**
```
DDD 概念                    微服务映射
─────────────────────────────────────────
Bounded Context    →       Microservice
                            (限界上下文)     (微服务)
Context Mapping    →       Service Integration
                            (上下文映射)     (服务间集成)
```

**示例：**
```
┌──────────────────────────────────────────────────────┐
│                    电商平台                           │
├──────────────────────────────────────────────────────┤
│                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐│
│  │ Order        │  │ Payment      │  │ Inventory    ││
│  │ Service      │  │ Service      │  │ Service      ││
│  │ (订单服务)    │  │ (支付服务)    │  │ (库存服务)    ││
│  └──────────────┘  └──────────────┘  └──────────────┘│
│         │                │                │          │
│         └────────────────┼────────────────┘          │
│                          │                           │
│              ┌───────────┴───────────┐               │
│              │   User Service        │               │
│              │   (用户服务)           │               │
│              └───────────────────────┘               │
│                                                       │
└──────────────────────────────────────────────────────┘

对应 DDD 限界上下文：
- Order Bounded Context → Order Service
- Payment Bounded Context → Payment Service
- Inventory Bounded Context → Inventory Service
```

**DDD + Microservices 的最佳实践：**
1. 先用 DDD 识别 Bounded Context
2. 每个 Bounded Context 对应一个微服务
3. 服务间通过 Context Map 定义集成方式（REST、消息队列等）
4. 每个微服务内部可以再用 Clean Architecture 或 Hexagonal Architecture

### 9.6 Event-Driven Architecture（事件驱动架构）

**核心思想：**
> 系统通过异步事件进行通信，而非直接调用

**事件流示例：**
```
用户下单
    │
    ▼
┌─────────────┐
│ OrderCreated │ ───────┐
│  (事件)      │        │
└─────────────┘        │
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│Inventory │  │ Payment  │  │ Notification│
│ Service  │  │ Service  │  │ Service     │
│(减库存)   │  │(创建支付)  │  │(发送通知)   │
└──────────┘  └──────────┘  └──────────┘
```

**与 DDD 的关系：**
- DDD 有 **Domain Event（领域事件）** 概念
- 领域事件天然适合用 Event-Driven Architecture 实现
- 服务间解耦的最佳方式

**代码示例：**
```python
# DDD 领域事件
class OrderCreatedEvent:
    def __init__(self, order_id: str, user_id: str, total: Money):
        self.order_id = order_id
        self.user_id = user_id
        self.total = total
        self.occurred_on = datetime.now()

# 领域模型发布事件
class Order:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.items = []
        self.events = []  # 领域事件列表
    
    def confirm(self):
        self.status = OrderStatus.CONFIRMED
        # 发布领域事件
        self.events.append(OrderCreatedEvent(
            order_id=self.id,
            user_id=self.user_id,
            total=self.total
        ))

# Event-Driven Architecture 消费事件
class InventoryHandler:
    """库存服务的事件处理器"""
    
    def handle_order_created(self, event: OrderCreatedEvent):
        """处理订单创建事件，扣减库存"""
        for item in self.get_order_items(event.order_id):
            self.inventory_service.decrease(item.product_id, item.quantity)
```

### 9.7 Layered Architecture（分层架构）

**传统三层架构：**
```
┌─────────────────────────────────────┐
│         Presentation Layer          │
│         (表现层 / Controller)        │
├─────────────────────────────────────┤
│          Business Logic Layer       │
│          (业务逻辑层 / Service)      │
├─────────────────────────────────────┤
│          Data Access Layer          │
│          (数据访问层 / Repository)   │
└─────────────────────────────────────┘
```

**优点：**
- 简单直观，容易理解
- 适合小型项目或 CRUD 应用

**缺点：**
- 业务逻辑容易散在各 Service 中
- 难以表达复杂领域模型
- 上层依赖下层，难以单元测试

**演进路径：**
```
Layered Architecture → DDD + Clean Architecture
      (三层架构)        (领域驱动+整洁架构)
           │                    │
           │   当业务变复杂时    │
           └──────────────────▶│
                               │
           业务逻辑集中        业务逻辑分散在实体/领域服务中
           难以应对复杂性       清晰表达领域模型
```

### 9.8 架构模式对比表

| 架构模式 | 核心思想 | 业务逻辑位置 | 依赖方向 | 适用场景 | 与 DDD 关系 |
|---------|---------|-------------|---------|---------|------------|
| **Layered** | 按层组织代码 | Service 层 | 上层依赖下层 | 简单 CRUD | DDD 可替代它 |
| **Clean** | 依赖向内指向业务 | Entity、Use Case | 向内 | 中大型应用 | DDD + Clean = 最佳实践 |
| **Hexagonal** | 端口适配器模式 | Domain（中心） | 向内 | 需要多接口适配 | DDD + Hexagonal = 最佳实践 |
| **Onion** | 与 Clean 类似 | Domain（中心） | 向内 | .NET 生态常用 | DDD + Onion = .NET 最佳实践 |
| **Microservices** | 服务拆分 | 每个服务内部 | 服务间松耦合 | 大型分布式系统 | DDD BC → Microservice |
| **Event-Driven** | 异步事件通信 | Event Handler | 无直接依赖 | 高并发、解耦 | DDD Domain Event → Event-Driven |

### 9.9 如何组合使用？

**推荐组合（根据项目规模）：**

**小型项目：**
```
SOLID + Layered Architecture
```

**中型项目：**
```
SOLID + DDD + Clean Architecture/Hexagonal Architecture
```

**大型分布式系统：**
```
SOLID + DDD + Clean Architecture + Microservices + Event-Driven
```

**示例：电商平台的完整架构**

```
用户下单流程的完整技术栈：

1. BDD（需求层面）
   └─ Given-When-Then 定义用户下单场景

2. DDD（建模层面）
   └─ 识别 Order Bounded Context
      ├─ 聚合：Order（包含 OrderItem）
      ├─ 实体：Order、Product
      ├─ 值对象：Money、Address
      └─ 领域服务：PricingService

3. Clean Architecture（代码结构）
   Order Service
   ├─ Domain Layer（Entity、Value Object）
   ├─ Application Layer（Use Case：CreateOrder）
   ├─ Interface Layer（OrderController）
   └─ Infrastructure Layer（OrderRepositoryImpl）

4. Microservices（系统架构）
   ├─ Order Service（订单服务）
   ├─ Payment Service（支付服务）
   ├─ Inventory Service（库存服务）
   └─ Notification Service（通知服务）

5. Event-Driven（服务通信）
   Order Service ──OrderCreated──▶ Payment Service
                ──OrderCreated──▶ Inventory Service
                ──OrderCreated──▶ Notification Service

6. TDD（开发实践）
   └─ Red-Green-Refactor 循环开发每个功能
```

---

## 参考资源

1. 《测试驱动开发》- Kent Beck
2. 《BDD in Action》- John Ferguson Smart
3. 《领域驱动设计》- Eric Evans
4. 《实现领域驱动设计》- Vaughn Vernon
5. [Cucumber 官方文档](https://cucumber.io/docs)
6. [Gherkin 语法参考](https://cucumber.io/docs/gherkin/)
