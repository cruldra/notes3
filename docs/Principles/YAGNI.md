---
sidebar_position: 3
---

**YAGNI 原则**（You Aren't Gonna Need It）是一条针对**功能与抽象范围**的判断标准：**只在真正需要的时候才实现**，不要为"未来可能用到"提前写代码或搭抽象。它由极限编程（XP）的提出者 **Ron Jeffries** 在 1990 年代末归纳出来，核心动机是：开发者高估自己预测未来的能力，绝大部分"为未来准备"的代码最终都没被用上，反倒成了维护负担。

它的核心主张只有一句：

> **现在不需要的，现在就不要做。**

---

## 它在反对什么

YAGNI 反对的是**预测式开发**：

- "未来可能要支持多种数据库" → 提前写 5 个 Adapter
- "用户量起来后可能要分库分表" → 提前在所有 SQL 加 sharding key
- "万一以后要做插件机制" → 提前把所有逻辑包成 plugin
- "未来可能要换 ORM" → 在数据访问层包一层不必要的抽象

这些预测的共同问题：

1. **方向常猜错**——你以为会扩展数据库，结果团队改了缓存策略
2. **过早架构 = 过早绑定**——架构早了反而限制了真实需求出现时的灵活性
3. **维护成本立刻产生，收益遥遥无期**——抽象层的复杂度从写下那一刻就开始烧时间

---

## 反例 vs 正例

### 例：数据库访问

**❌ 违反 YAGNI**

```python
# 当前只用 MySQL，但提前写一套抽象以"防止未来切换"
class DatabaseInterface:
    def connect(self): ...
    def query(self, sql): ...
    def close(self): ...

class MySQLAdapter(DatabaseInterface): ...
class PostgreSQLAdapter(DatabaseInterface): ...
class MongoDBAdapter(DatabaseInterface): ...
```

代价：

- 多 3 个类要维护
- 接口被"未来想象"绑死，等真要切的时候发现 MongoDB 根本不适合 SQL 抽象
- 团队读代码时增加心智负担

**✅ 遵循 YAGNI**

```python
import pymysql

def query(sql):
    conn = pymysql.connect(...)
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            return cursor.fetchall()
    finally:
        conn.close()
```

等真有第二种数据库要接的时候，**那时候**才知道抽象应该怎么切。事实证明，那一天往往**永远不到来**。

---

## 误区：YAGNI ≠ 短视

YAGNI 不是"只写最低能跑的代码"。下列情形**不算违反 YAGNI**：

| 情形 | 为什么不算 |
|---|---|
| **已确认下一迭代要做** | 不是"未来可能"，是已规划的近期工作 |
| **不预留扩展点 = 推倒重写** | 比如核心数据模型，改造代价远大于现在多花点心思 |
| **行业标准实践** | REST API 加版本号、日志加 trace_id——这些不是"未来才用"，是行业默认就该有 |
| **删除 / 弃用比新增更贵** | 数据库表加字段比删字段简单得多，一开始把字段设计够 |

判断标准：**不做这件事，未来真出现需求时代价是否能承受**。能 → YAGNI 适用，先不做。不能 → 这是必要的前置投资，不算违反。

---

## 与其他抗过度设计原则的对照

YAGNI 经常被混淆为"懒"或"短视"。它不是。它是与下面三条**互补**的：

| 原则 | 反对什么 |
|---|---|
| **YAGNI** | 反对"做没必要做的功能" |
| **KISS** | 反对"用过复杂的方式做必要的功能" |
| **DRY** | 反对"同一件事在多处重复表达" |
| **AHA**（Avoid Hasty Abstractions） | 反对"急着把还没看清的模式抽象出来" |

四条一起用，才能完整防御过度工程：YAGNI 控范围、KISS 控复杂度、DRY 控重复、AHA 控抽象时机。

---

## 自检清单

写代码前问自己：

1. **现在的需求真的需要这个吗**？没人提需求 → 别写
2. **不做这个，下一个真实需求来时，代价能承受吗**？能 → YAGNI 适用
3. **这是预测，还是已确认的计划**？预测 → 危险信号
4. **这层抽象现在能列出 ≥ 2 个具体使用者吗**？只有一个 → 大概率不需要抽象，直接写
5. **加了这个东西，半年后我能解释为什么加它吗**？说不清 → 不该加

---

## 与其他原则的关系

- [KISS](./KISS.md) 主张实现要简单——**与 YAGNI 同向**：少做事 + 简单做 = 总复杂度最低
- [DRY](./DRY.md) 主张消除重复——**与 YAGNI 互补**：YAGNI 控功能边界，DRY 控知识表达
- [MECE](./MECE原则.md) 主张分类不重不漏——**与 YAGNI 正交**：MECE 保证当下做的事情结构合格，YAGNI 保证不做未来的幻觉
- [第一性原理](../Methodology/第一性原理.md) 主张拆到本质——**互为校验**：第一性帮你识别什么是真正必需的，YAGNI 帮你拒绝那些"看起来必需"但其实是想象的功能

---

## 小结

YAGNI 不是反对架构，而是反对**为想象中的未来付费**：

> **架构应该跟着需求长出来，而不是为可能的需求提前搭好。**

它要求的不是技术克制，而是**对自己预测能力的诚实**——承认你不知道未来会怎么变。等真变化来了，再用 [KISS](./KISS.md)、[DRY](./DRY.md) 把当下问题解决干净。

---

## 延伸阅读

- Martin Fowler, ["YAGNI"](https://martinfowler.com/bliki/Yagni.html)——一篇很短但很经典的解释
- Andy Hunt & Dave Thomas, *The Pragmatic Programmer*——XP 实践，YAGNI 在其中是核心
- Wikipedia: [You aren't gonna need it](https://en.wikipedia.org/wiki/You_aren%27t_gonna_need_it)
