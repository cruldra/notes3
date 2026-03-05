---
sidebar_position: 3
---


## 1. 为什么需要 Protocol？（鸭子类型的痛点）

Python 是动态语言，我们常说：**"如果一个东西走起来像鸭子，叫起来像鸭子，那它就是鸭子。"**

在 Python 中，我们不需要对象继承某个特定的父类，只要它有特定的方法（比如 `.quack()`），我们就可以直接调用。然而，当我们引入静态类型提示（Type Hinting）和工具（如 `mypy`）时，一个问题出现了：**如何告诉类型检查器"我需要一个只要会 `quack()` 的对象就行"？**

在没有 `Protocol` 之前，你只能让类去继承一个共同的基类（比如 `class Duck(Animal)`），这叫**名义子类型（Nominal Subtyping）**。但这破坏了 Python 灵活的鸭子类型特性。

**`typing.Protocol` 就是为了给 Python 的"鸭子类型（Duck Typing）"提供静态类型检查支持。**

## 2. Protocol 是如何工作的？

`Protocol` 允许你定义一个**"协议"**：**只要你满足了这个协议里规定的条件（比如有特定的方法或属性），我就认同你是这个类型，哪怕你根本没有继承我。**

### 2.1 基础示例

假设我们需要一个会飞（`fly`）的对象：

```python
from typing import Protocol

# 1. 定义一个"协议"：任何实现了 fly 方法的类，都符合 Flyable 协议
class Flyable(Protocol):
    def fly(self) -> str:
        ...  # 用省略号定义接口，不实现代码

# 2. 定义两个毫无血缘关系（没有继承 Flyable），但都有 fly 方法的类
class Bird:
    def fly(self) -> str:
        return "鸟儿拍打翅膀飞！"

class Airplane:
    def fly(self) -> str:
        return "飞机开启发动机飞！"

# 定义一个缺少 fly 方法的类
class Dog:
    def run(self) -> str:
        return "狗狗在地上跑！"

# 3. 在函数类型提示中使用这个"协议"
def make_it_fly(item: Flyable) -> None:
    print(item.fly())

# 测试
bird = Bird()
plane = Airplane()
dog = Dog()

make_it_fly(bird)   # ✅ 类型检查通过，运行正常
make_it_fly(plane)  # ✅ 类型检查通过，运行正常

# make_it_fly(dog)  # ❌ 类型检查报错：
                    # Argument 1 to "make_it_fly" has incompatible type "Dog"; expected "Flyable"
```

### 2.2 带属性的 Protocol

Protocol 不仅可以定义方法，还可以定义属性：

```python
from typing import Protocol

class Drawable(Protocol):
    """任何具有 name 属性和 draw 方法的对象都满足此协议"""
    name: str  # 属性要求
    
    def draw(self) -> None: ...

class Circle:
    def __init__(self, name: str) -> None:
        self.name = name
    
    def draw(self) -> None:
        print(f"Drawing circle: {self.name}")

def render(item: Drawable) -> None:
    print(f"Rendering {item.name}")
    item.draw()

circle = Circle("my_circle")
render(circle)  # ✅ 通过类型检查
```

## 3. 名义子类型 vs 结构子类型

为了更好理解，可以这样对比：

| 特性 | 名义子类型（Nominal） | 结构子类型（Structural） |
|------|---------------------|------------------------|
| **类型关系** | 基于显式继承 | 基于结构（方法/属性） |
| **类比** | 必须出示北大毕业证（继承关系） | 能写出符合要求的代码就行（能力） |
| **实现方式** | 继承基类 | 满足 Protocol 定义 |
| **Python 特性** | 传统 OOP | 鸭子类型 + Protocol |

**示例对比：**

```python
from typing import Protocol

# 名义子类型：必须显式继承
class Animal:
    def speak(self) -> str: ...

class Dog(Animal):  # 显式继承
    def speak(self) -> str:
        return "Woof!"

# 结构子类型：隐式满足
class Speaker(Protocol):
    def speak(self) -> str: ...

class Cat:  # 没有继承 Speaker
    def speak(self) -> str:
        return "Meow!"

def greet(speaker: Speaker) -> None:
    print(speaker.speak())

greet(Dog())   # ✅ OK

greet(Cat())   # ✅ OK - Cat 满足 Speaker 协议，即使没继承它
```

## 4. 运行时检查：@runtime_checkable

默认情况下，Protocol 仅用于静态分析。但如果需要在运行时使用 `isinstance()` 进行验证，可以使用 `@runtime_checkable` 装饰器：

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Sized(Protocol):
    """任何具有 __len__ 方法的对象"""
    def __len__(self) -> int: ...

# 测试
my_list = [1, 2, 3]
my_int = 42

print(isinstance(my_list, Sized))  # ✅ True
print(isinstance(my_int, Sized))   # ❌ False

# 自定义类
class MyCollection:
    def __len__(self) -> int:
        return 100

print(isinstance(MyCollection(), Sized))  # ✅ True
```

**注意事项：**
- `@runtime_checkable` 会对性能有轻微影响
- 仅检查方法/属性是否存在，**不检查签名是否正确**
- 只应在真正需要运行时检查时使用

## 5. 协议组合与继承

Protocol 可以像普通类一样继承和组合：

```python
from typing import Protocol

class Readable(Protocol):
    def read(self, size: int = -1) -> bytes: ...

class Writable(Protocol):
    def write(self, data: bytes) -> int: ...

# 协议继承
class ReadWritable(Readable, Writable, Protocol):
    """同时支持读和写的协议"""
    pass

# 或者使用 & 运算符合并（Python 3.10+）
def process_stream(stream: Readable & Writable) -> None:
    """参数必须同时满足 Readable 和 Writable"""
    pass
```

## 6. 泛型 Protocol

Protocol 也可以是泛型的，支持类型参数：

```python
from typing import Protocol, TypeVar

T = TypeVar("T")

class Container(Protocol[T]):
    """一个可以包含任意类型元素的容器"""
    def get(self) -> T: ...
    def put(self, item: T) -> None: ...

class Box:
    def __init__(self) -> None:
        self._item: str = ""
    
    def get(self) -> str:
        return self._item
    
    def put(self, item: str) -> None:
        self._item = item

# Box 满足 Container[str]
def use_container(c: Container[str]) -> None:
    c.put("hello")
    print(c.get())

use_container(Box())  # ✅ OK
```

## 7. 实际应用场景

### 7.1 定义回调接口

```python
from typing import Protocol

class EventHandler(Protocol):
    def handle(self, event: str) -> None: ...

class Button:
    def __init__(self) -> None:
        self._handler: EventHandler | None = None
    
    def set_handler(self, handler: EventHandler) -> None:
        self._handler = handler
    
    def click(self) -> None:
        if self._handler:
            self._handler.handle("click")

# 任何实现了 handle 方法的类都可以作为处理器
class MyHandler:
    def handle(self, event: str) -> None:
        print(f"Handled: {event}")

button = Button()
button.set_handler(MyHandler())  # ✅ 不需要继承任何基类
button.click()
```

### 7.2 定义可迭代对象

```python
from typing import Protocol, TypeVar

T_co = TypeVar("T_co", covariant=True)

class Iterable(Protocol[T_co]):
    def __iter__(self) -> "Iterator[T_co]": ...

class Iterator(Protocol[T_co]):
    def __next__(self) -> T_co: ...
    def __iter__(self) -> "Iterator[T_co]": ...

def print_all(items: Iterable[str]) -> None:
    for item in items:
        print(item)

# 适用于任何可迭代对象
print_all(["a", "b", "c"])
print_all(("x", "y", "z"))
```

### 7.3 与现有代码库的集成

Protocol 特别适合为没有类型注解的第三方库添加类型支持：

```python
from typing import Protocol
import some_third_party_lib

# 为第三方库的对象定义协议
class ThirdPartyInterface(Protocol):
    def do_something(self, x: int) -> str: ...

def my_function(obj: ThirdPartyInterface) -> None:
    result = obj.do_something(42)
    print(result)

# 使用第三方库的对象，无需修改其源码
third_party_obj = some_third_party_lib.SomeClass()
my_function(third_party_obj)  # 只要 SomeClass 有 do_something 方法就 OK
```

## 8. 最佳实践

1. **优先使用 Protocol 而非继承**：当只需要特定接口时，使用 Protocol 比强制继承更灵活

2. **命名约定**：Protocol 类名通常以 `Supports` 或 `Has` 开头，如 `SupportsRead`、`HasName`

3. **只定义必要的方法**：Protocol 应该最小化，只包含真正需要的方法

4. **谨慎使用 @runtime_checkable**：仅在需要运行时类型检查时添加，避免不必要的性能开销

5. **文档化**：为 Protocol 添加详细的 docstring，说明实现类需要满足什么条件

6. **与其他类型系统的配合**：Protocol 可以和 `TypeVar`、`Generic`、`Union` 等类型构造器结合使用

```python
from typing import Protocol, TypeVar, runtime_checkable

@runtime_checkable
class SupportsClose(Protocol):
    """
    协议：任何实现了 close() 方法的对象。
    
    常用于上下文管理器或需要显式释放资源的场景。
    """
    def close(self) -> None: ...

def safe_close(resource: SupportsClose) -> None:
    """安全地关闭资源"""
    try:
        resource.close()
    except Exception as e:
        print(f"Error closing resource: {e}")
```

## 9. 总结

`typing.Protocol` 是 Python 类型系统中连接**动态灵活性**与**静态安全性**的桥梁：

- **核心理念**：结构化子类型 - 只关心"你有什么"，不关心"你是谁"
- **主要优势**：
  - 保留 Python 鸭子类型的灵活性
  - 获得静态类型检查的好处（IDE 提示、提前发现错误）
  - 无需修改现有代码即可添加类型支持
- **适用场景**：
  - 定义回调接口
  - 描述第三方库的接口
  - 实现依赖注入
  - 任何需要"能力而非继承"的场景

通过 Protocol，你可以在保持 Pythonic 编码风格的同时，享受到现代类型系统带来的安全性和开发效率提升。

## 参考资料

1. [Python Documentation - typing.Protocol](https://docs.python.org/3/library/typing.html#typing.Protocol)
2. [PEP 544 - Protocols: Structural subtyping (static duck typing)](https://peps.python.org/pep-0544/)
3. [Typing Documentation - Protocols](https://typing.python.org/en/latest/spec/protocol.html)
4. [Mypy Documentation - Protocols](https://mypy.readthedocs.io/en/stable/protocols.html)
