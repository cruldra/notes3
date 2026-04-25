---
sidebar_position: 1
---

**KISS 原则**（Keep It Simple, Stupid）是一条针对**工程实现复杂度**的判断标准：在能完成需求的前提下，**选最简单、最直白的实现**，反对炫技和过度工程。它最早由美国海军工程师 **Kelly Johnson**（洛克希德 SR-71 黑鸟侦察机的总工程师）在 1960 年代提出，原意是要求军用装备**任何普通技工拿一把扳手就能修**——后来被软件工程沿用，成为代码可读性的核心准则之一。

它的核心主张只有一句：

> **代码先要让人看懂，再要它能跑。**

这里的 "Stupid" 不是骂人，而是强调"**面向最笨的读者**"——假设接手代码的人对这块业务一无所知、注意力涣散、刚从一个长会议出来。能让这种状态的人 5 分钟看懂的代码，才算合格。

---

## 它在反对什么

KISS 反对的是**主动制造复杂度**：

- **炫技**：用罕见特性（元类、装饰器嵌套、函数式高阶组合）替代简单写法
- **过度工程**：为了"优雅"引入策略模式、责任链、事件总线，但实际只有一种策略、一条链、一个事件
- **抽象上瘾**：每个常量必须从配置读、每个函数必须传 logger、每个类必须有 interface

这些做法的特点是：**写的人爽，看的人苦**。

---

## 反例 vs 正例

### 例 1：求平均值

**❌ 违反 KISS**

```python
from functools import reduce

def calculate_average(numbers):
    return reduce(lambda x, y: x + y, numbers) / len(numbers) if numbers else 0
```

**✅ 遵循 KISS**

```python
def calculate_average(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)
```

差别：`reduce + lambda` 没有"更对"，只是"更花"——`sum()` 是内置函数，可读性高、性能不差。

### 例 2：权限检查

**❌ 违反 KISS**

```python
class PermissionChecker:
    def __init__(self, user):
        self.user = user
        self.strategies = []

    def add_strategy(self, strategy):
        self.strategies.append(strategy)

    def check(self, resource):
        for strategy in self.strategies:
            if not strategy(self.user, resource):
                return False
        return True

# 调用
checker = PermissionChecker(user)
checker.add_strategy(lambda u, r: u.id == r.owner_id)
checker.add_strategy(lambda u, r: u.role == 'admin')
result = checker.check(resource)
```

**✅ 遵循 KISS**

```python
def can_access(user, resource):
    if user.role == 'admin':
        return True
    if user.id == resource.owner_id:
        return True
    return False

result = can_access(user, resource)
```

策略模式的复杂度只有在**策略真的会动态变化、且数量多到值得抽象**的时候才划算。两条规则的简单 if-return 就能完成的事，不需要架构。

---

## 误区：什么时候不该硬上 KISS

KISS 不是"不准用任何高级特性"——它要求**复杂度与场景匹配**：

| 场景 | 简单是否仍正确 |
|---|---|
| 一个小工具脚本 | ✓ 简单到底 |
| 业务模块 | ✓ 简单优先，必要时抽象 |
| 库（被很多人用） | ✗ 此时复杂度可以前置在库内，换取使用者侧的简单 |
| 性能热点 | ✗ 必要的优化即使复杂也要做，但要写注释解释为什么 |

**真正违反 KISS 的标志**：复杂度**没换来任何东西**——既没更快，也没更通用，只是写起来"更高级"。

---

## 自检清单

写完一段代码问自己：

1. **新人 5 分钟能看懂吗**？看不懂就是失败
2. **有没有更简单的实现方式**？经常有。回头改
3. **代码注释比代码本身还长吗**？是 → 大概率代码该简化
4. **能不能去掉一层间接**？类→接口→工厂→builder 这种链路，每一层都要审视是否必要
5. **第三方读者会不会问"这里为什么要这样写"**？会 → 代码没自解释，要么改简单要么补 doc

---

## 与其他原则的关系

- [DRY](./DRY.md) 主张消除重复，但**过度 DRY 会创造抽象** → 此时 KISS 是制衡：宁可两处轻微重复，不要一处难懂的抽象
- [YAGNI](./YAGNI.md) 主张不做不需要的功能，本质上是**KISS 在功能维度的延伸**：少做事就少复杂
- [MECE](./MECE原则.md) 主张分类要不重不漏，是**信息组织维度**的判断标准；KISS 是**代码实现维度**的判断标准
- [金字塔原理](../Methodology/金字塔原理.md) 主张"结论先行、自顶向下"，本质上是**表达维度的 KISS**——让读者用最少认知成本拿到主旨

简单说：**KISS、DRY、YAGNI 三条经常打架**——KISS 与 DRY 互相制衡（消除重复 vs 别造抽象），KISS 与 YAGNI 同向（都让事情少而清楚）。落到一行代码上，要根据场景做权衡。

---

## 小结

KISS 不是反对一切复杂度，而是反对**没收益的复杂度**：

> **能用 10 行写清楚的事，不要用 50 行装聪明。**

它要求的不是技术不够好，而是**自我克制**——克制对优雅 / 对称 / 模式的追求，把"读者负担最小"放在首位。

---

## 延伸阅读

- Robert C. Martin, *Clean Code*（《代码整洁之道》）——整本书的核心论点之一就是 KISS
- Andrew Hunt & David Thomas, *The Pragmatic Programmer*（《程序员修炼之道》）——"Easier to Change" 与 KISS 同源
- Wikipedia: [KISS principle](https://en.wikipedia.org/wiki/KISS_principle)
