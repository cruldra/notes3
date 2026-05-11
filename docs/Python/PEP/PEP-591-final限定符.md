---
sidebar_position: 5
---

**[PEP 591](https://peps.python.org/pep-0591/) 在 Python 3.8 中引入了 `Final` 类型限定符与 `@final` 装饰器**，用来在类型系统层面表达"这里不允许再被覆盖/继承/赋值"的意图。它不是运行时强制——执行 `class Sub(Base)` 仍然可以继承一个 `@final` 类——而是把"封闭性"作为一种**约束声明**交给类型检查器（mypy / Pyright / Pylance）去校验。

> 设计目标：把 Java 的 `final`、Kotlin 的 `open/sealed` 这类"作者可以禁止下游某些扩展"的能力，引入到 Python 的渐进类型系统中，让重构更安全、API 边界更清晰。

## 三种用法

`Final` 与 `@final` 都来自 `typing` 模块（Python 3.11 起也可从 `typing_extensions` 兼容旧版本）。它们覆盖三种位置：

```python
from typing import Final, final

# 1. Final 变量：声明一次后不可再赋值
TIMEOUT: Final = 30
PI: Final[float] = 3.14159

# 2. @final 方法：子类不能覆盖
class Base:
    @final
    def hash_id(self) -> str:
        return ...

# 3. @final 类：不能被继承
@final
class Singleton:
    ...
```

## `Final` 变量

`Final` 用在赋值的类型注解位置，告诉类型检查器"这个名字此后不允许再被赋值"。它有两种写法：

```python
MAX_RETRIES: Final = 3              # 推断为 int
MAX_RETRIES: Final[int] = 3         # 显式标注 int
```

两者等价，但**带方括号的形式更适合常量在模块或类的顶层暴露给外部读者**——类型一目了然。

约束规则：

- 同一作用域内只能赋值一次，再次赋值由 mypy 报 `[misc]` 错误。
- 不能在循环、条件分支中**仅在某一分支**赋值后又在另一分支重新赋值。
- 类属性 `x: Final = ...` 既不能在 `__init__` 之外被赋值，也不能在子类里覆盖。
- 函数局部变量也可以 `x: Final = compute()`，常用于"算一次后不再变"的场景。

```python
class Config:
    DEBUG: Final = False

    def __init__(self, port: int) -> None:
        self.port: Final = port      # 实例级 Final，仅在 __init__ 中赋值

class DevConfig(Config):
    DEBUG = True   # ❌ mypy: Cannot assign to final name "DEBUG"
```

> `Final` ≠ 不可变（immutable）。`xs: Final = [1, 2]` 表示**不能再让 `xs` 指向别的对象**，但 `xs.append(3)` 仍然合法——它只锁名字绑定，不锁对象内容。

## `@final` 方法

`@final` 装饰方法表示"子类不允许重写这个方法"。它用于保护那些**承担不变量、被基类内部依赖**的方法，避免子类无意中破坏契约。

```python
from typing import final

class Repository:
    @final
    def save(self, entity) -> None:
        self._validate(entity)
        self._persist(entity)

    def _persist(self, entity) -> None:   # 留给子类扩展
        ...

class UserRepository(Repository):
    def save(self, entity) -> None:        # ❌ Cannot override final attribute "save"
        ...
    def _persist(self, entity) -> None:    # ✅ 允许覆盖
        ...
```

经验法则：**模板方法（template method）配 `@final`**——把骨架方法封死，把可扩展的钩子方法留开，是对基类作者最友好的写法。

## `@final` 类

`@final` 装饰类表示"该类不允许被继承"。最常见的用途是：

- **数据类 / 值对象**：语义上不应有"子类化的版本"，例如 `Email`、`Money`。
- **单例 / 工厂返回的封闭类型**：保证类型检查时 `isinstance(x, Foo)` 与 `type(x) is Foo` 等价。
- **Enum 替代场景**：当一组取值有限且不希望扩展时。

```python
@final
class Money:
    def __init__(self, amount: int, currency: str) -> None:
        self.amount = amount
        self.currency = currency

class Yen(Money):  # ❌ Cannot inherit from final class "Money"
    ...
```

> 类型检查器会同时把"不允许继承"传播到协议（`Protocol`）兼容性判断上：被 `@final` 的类的实例只能匹配它自己的接口，不会被认作"某个未知子类"。

## 与运行时行为的关系

PEP 591 **明确把约束限定在静态检查阶段**——CPython 解释器不会因为 `@final` 而抛错：

```python
@final
class A: ...

class B(A): ...        # 运行时 OK
B()                    # 运行时 OK
```

如果想要运行时强制，需要自己加守卫，例如在 `__init_subclass__` 里检查：

```python
class NoSubclass:
    def __init_subclass__(cls, **kw):
        raise TypeError(f"{cls.__name__} 不能继承 NoSubclass")
```

但这通常没有必要——把检查交给 mypy / CI 已足够。运行时反射可以通过 `typing.final` 装饰后保留的 `__final__ = True` 属性识别：

```python
from typing import final

@final
def foo(): ...

print(foo.__final__)        # True
```

## `Final` vs `Constant` vs `Readonly`

Python 类型系统里有几个语义相邻的概念，容易混淆：

| 概念 | 来源 | 锁定的是 | 典型用法 |
|------|------|---------|---------|
| `Final` | PEP 591 | 名字绑定（不能再赋值） | `TIMEOUT: Final = 30` |
| `Literal[v]` | PEP 586 | 取值集合（必须等于 v） | `def f(x: Literal["a", "b"])` |
| `ReadOnly`（PEP 705） | PEP 705 | `TypedDict` 字段（不能写入） | `class T(TypedDict): id: ReadOnly[int]` |
| `Frozen dataclass` | dataclasses | 实例属性（运行时强制） | `@dataclass(frozen=True)` |

`Final` 关注"绑定关系"，`Literal` 关注"具体取值"，`ReadOnly` 关注"字典键能否写入"，`frozen=True` 才是运行时不可变。**它们正交而非互斥**——一个常量经常同时是 `Final` 与 `Literal`：

```python
LOG_LEVEL: Final[Literal["DEBUG", "INFO", "WARNING"]] = "INFO"
```

## 与 Java / Kotlin / TypeScript 对比

| 语言 | "禁止继承" | "禁止覆盖" | "禁止再赋值" |
|------|-----------|-----------|-------------|
| Java | `final class` | `final` 方法 | `final` 字段 |
| Kotlin | 默认即 final，需要 `open` 才能继承 | 同上 | `val` |
| TypeScript | 无原生关键字（用 `private constructor`+工厂模拟） | 用 `final` 注释/lint | `readonly` / `as const` |
| Python（PEP 591） | `@final` 类 | `@final` 方法 | `Final` 限定符 |

Python 的方案在三个维度上都是**类型层面的标注**而非语言级强制，与 Kotlin 默认 final 的取舍正相反——Python 更倾向"约定大于强制"。

## 何时用 / 何时不用

推荐使用：

- **公共库 API**：明确告诉下游"不要继承 / 不要覆盖 / 不要重新绑定"。
- **领域值对象**：`Money`、`Email`、`Url` 这类语义封闭的小类。
- **模块常量**：`TIMEOUT`、`API_VERSION` 等只赋值一次的值。
- **模板方法的骨架**：固定流程的方法封 `@final`，钩子方法留开放。

谨慎使用：

- 测试基础设施（mock、stub）经常需要继承或覆盖，过度 `@final` 会让测试无法编写。
- 仍处于探索阶段的内部类——过早封死会阻碍重构。

## 延伸阅读

- 官方提案：[https://peps.python.org/pep-0591/](https://peps.python.org/pep-0591/)
- `typing.final` 文档：[https://docs.python.org/3/library/typing.html#typing.final](https://docs.python.org/3/library/typing.html#typing.final)
- `typing.Final` 文档：[https://docs.python.org/3/library/typing.html#typing.Final](https://docs.python.org/3/library/typing.html#typing.Final)
- mypy `[misc]` 错误码说明：[https://mypy.readthedocs.io/en/stable/error_code_list.html](https://mypy.readthedocs.io/en/stable/error_code_list.html)
- 相邻提案 PEP 705（TypedDict ReadOnly）：[https://peps.python.org/pep-0705/](https://peps.python.org/pep-0705/)
