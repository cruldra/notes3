---
sidebar_position: 2
---

**DRY 原则**（Don't Repeat Yourself）是一条针对**知识与逻辑表示**的判断标准：系统中的每一项知识或决策，**必须有唯一、无歧义、权威的表示位置**——不允许散落在多处各自为政。它由 **Andy Hunt** 与 **Dave Thomas** 在 1999 年的《The Pragmatic Programmer》中正式命名，但思想可追溯到更早的"单一事实来源"（Single Source of Truth）实践。

它的核心主张只有一句：

> **同一件事，只在一个地方说一次。**

注意原文不是 "Don't Repeat Code"——是 "Don't Repeat **Yourself**"。重复的可能是代码，但更可能是**配置、文档、注释、数据、流程**。任何一项"知识"散在多处都算违反。

---

## 它在反对什么

DRY 反对的是**多份真相共存**：

- 同一段计算逻辑在 3 个 service 里 copy-paste
- 数据库连接配置在 5 个文件里硬编码
- 业务规则同时写在代码、文档、注释里，三份"真相"哪个都不全
- 同一份用户数据在 user 表、cache、profile 表里各存一份，更新时漏一处

后果不是写得多——是**维护成本指数级上升 + Bug 几率倍增**：改一处忘了另一处，系统进入不一致状态。

---

## 反例 vs 正例

### 例 1：散落的业务逻辑

**❌ 违反 DRY**

```python
class NormalUser:
    def checkout(self, items):
        subtotal = sum(item.price for item in items)
        tax = subtotal * 0.08         # 税费计算 #1
        discount = 0
        total = subtotal + tax - discount
        return self.process_payment(total)

class VIPUser:
    def checkout(self, items):
        subtotal = sum(item.price for item in items)
        tax = subtotal * 0.08         # 税费计算 #2（复制粘贴）
        discount = subtotal * 0.10
        total = subtotal + tax - discount
        return self.process_payment(total)
```

税率从 8% 改成 10% 时，**两处都要改**——漏一处就崩。

**✅ 遵循 DRY**

```python
def calculate_tax(amount):
    return amount * 0.08

class BaseUser:
    def checkout(self, items):
        subtotal = sum(item.price for item in items)
        tax = calculate_tax(subtotal)
        discount = self.calculate_discount(subtotal)
        total = subtotal + tax - discount
        return self.process_payment(total)

    def calculate_discount(self, subtotal):
        return 0

class NormalUser(BaseUser):
    pass

class VIPUser(BaseUser):
    def calculate_discount(self, subtotal):
        return subtotal * 0.10
```

税率改动只动 `calculate_tax` 一处。

### 例 2：配置重复

**❌ 违反 DRY**

```python
def connect_db():
    return pymysql.connect(host='localhost', port=3306, user='admin')

def backup_db():
    return pymysql.connect(host='localhost', port=3306, user='admin')
```

**✅ 遵循 DRY**

```python
# config.py
DB_CONFIG = {'host': 'localhost', 'port': 3306, 'user': 'admin'}

# 各处引用
from config import DB_CONFIG

def connect_db():
    return pymysql.connect(**DB_CONFIG)
```

DB 切换环境只改一份配置。

---

## DRY 的范围远不止代码

| 类型 | DRY 表现 |
|---|---|
| **代码** | 公共逻辑抽函数 / 抽基类 |
| **配置** | 集中到 config 文件 / 环境变量 |
| **文档** | 文档与代码注释别重复描述同一件事；保留权威方，另一方写指针 |
| **数据** | 不维护多份相同数据副本；用关联或缓存失效机制保证一致 |
| **构建脚本** | 多平台共享构建步骤抽到公共脚本 |
| **测试 fixture** | 公共 setup 抽出来 |

DRY 的对象是"**知识**"——任何一份会因业务变更而需要同步修改的东西。

---

## 误区：过度 DRY 比重复更糟

DRY 有副作用：**为了消除微小重复创造的复杂抽象**，比重复本身更难维护。

**❌ 过度 DRY**

```python
def execute_operation_with_logging(operation, *args, **kwargs):
    logger.info(f"Executing {operation.__name__}")
    result = operation(*args, **kwargs)
    logger.info(f"Completed {operation.__name__}")
    return result

# 调用
result = execute_operation_with_logging(calculate_sum, 1, 2)
# 而不是简单的
result = calculate_sum(1, 2)
```

为了消除两行 logger 调用的重复，引入了一个高阶函数，所有调用点都得改写——读者还得跳进 `execute_operation_with_logging` 才能搞清在干什么。

### 经验法则

| 重复次数 | 处理方式 |
|---|---|
| 1 处 | 不抽 |
| 2 处 | **不抽**——只两处时抽象往往猜错；先放着 |
| 3 处及以上 | 考虑抽象，但要确认三处是**同一份知识**而不是**长得像但语义不同** |

更准确的原则叫 **AHA**（Avoid Hasty Abstractions）：**不要急着抽象**。先观察重复，等模式稳定了再抽。

### 长得像 ≠ 同一份知识

最常见的过度 DRY 来自把**两件事抽象成一件事**：

```python
# 用户头像与商品图片都需要"上传文件 + 生成缩略图 + 存 S3"
# 看起来重复，抽到 upload_image()
# 但用户头像后来要加裁剪、商品图片要加水印
# 这时候 upload_image() 变成参数地狱
```

它们**当前**长得像，但属于不同业务、会**独立演化**。强行 DRY 会绑住手脚。

---

## 自检清单

1. **同一件事我改了几处**？> 1 → 违反 DRY
2. **抽象后调用点更难看懂了吗**？是 → 过度 DRY，回退
3. **重复的两段代码会一起变化吗**？是 → 真重复，可以抽。会独立变化 → 长得像而已，别抽
4. **抽象的命名能一句话说清吗**？说不清 → 抽象不成立
5. **配置 / 数据 / 文档有没有多份真相**？常被忽略，但同样违反 DRY

---

## 与其他原则的关系

- [KISS](./KISS.md) 主张代码要简单——**与 DRY 经常打架**：DRY 要"抽出来"，KISS 要"别造抽象"。当抽象本身比重复更复杂时，KISS 赢
- [YAGNI](./YAGNI.md) 主张不做不需要的功能——**与 DRY 间接互补**：YAGNI 让你少做（少重复），DRY 让你做得更整齐
- [MECE](./MECE原则.md) 主张分类不重不漏——**信息维度的 DRY**："不重"对应 ME，"不漏"对应 CE
- [金字塔原理](../Methodology/金字塔原理.md) 中"以上统下"——**结构维度的 DRY**：每一层只回答上一层的问题，不重复表达

---

## 小结

DRY 不是"消除一切重复"，而是**消除"同一份知识的多处表达"**：

> **代码可以重复，知识不能重复。**

判断要不要 DRY 的关键问题不是"看起来一样吗"，而是"**它们会因为同一个原因一起变化吗**"。能 → 抽。不能 → 留着，未来再说。

---

## 延伸阅读

- Andy Hunt & Dave Thomas, *The Pragmatic Programmer*（《程序员修炼之道》）——DRY 的诞生地
- Sandi Metz, ["The Wrong Abstraction"](https://sandimetz.com/blog/2016/1/20/the-wrong-abstraction)——AHA 思想的经典文章
- Wikipedia: [Don't repeat yourself](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself)
