`__iter__` 是 Python 迭代器协议（Iterator Protocol）的核心魔术方法之一。它负责返回一个迭代器对象，使得容器对象（如列表、元组、自定义类）支持 `for` 循环、`in` 操作符以及 `iter()` 内置函数。

## 1. 方法定义与调用机制

当对一个对象进行迭代操作时（例如在 `for` 循环中），Python 解释器会自动调用该对象的 `__iter__` 方法。

### 1.1 函数签名

```python
object.__iter__(self)
```

*   **`self`**: 实例对象本身。
*   **返回值**: 必须返回一个**迭代器（Iterator）**对象。迭代器对象必须实现 `__next__` 方法。

### 1.2 迭代器协议 (Iterator Protocol)

Python 的迭代机制基于两个魔术方法：

1.  **`__iter__`**: 返回迭代器对象本身。
2.  **`__next__`**: 返回容器中的下一个元素。如果没有更多元素，则抛出 `StopIteration` 异常。

### 1.3 可迭代对象 (Iterable) vs 迭代器 (Iterator)

*   **可迭代对象 (Iterable)**: 实现了 `__iter__` 方法的对象。它可以被传递给 `iter()` 函数。例如：`list`, `tuple`, `dict`, `str`。
    *   **职责**: 提供获取迭代器的方法。
*   **迭代器 (Iterator)**: 实现了 `__iter__` 和 `__next__` 方法的对象。
    *   **职责**: 负责具体的迭代逻辑（记录当前位置、计算下一个值）。
    *   **注意**: 迭代器的 `__iter__` 方法通常直接返回 `self`。

## 2. 代码示例

### 2.1 简单示例：自定义范围迭代器

实现一个类似于内置 `range` 函数的类。

```python
class MyRange:
    def __init__(self, start, end):
        self.current = start
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        value = self.current
        self.current += 1
        return value

# 使用示例
for num in MyRange(1, 4):
    print(num)
# 输出:
# 1
# 2
# 3
```

### 2.2 分离可迭代对象与迭代器

为了支持多次独立迭代，通常将可迭代对象和迭代器分开实现。

```python
# 迭代器：负责状态维护
class NodeIterator:
    def __init__(self, node):
        self.current_node = node

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_node is None:
            raise StopIteration
        value = self.current_node.value
        self.current_node = self.current_node.next
        return value

# 可迭代对象：负责存储数据
class LinkedList:
    def __init__(self):
        self.head = None

    def add(self, value):
        new_node = Node(value)
        new_node.next = self.head
        self.head = new_node

    def __iter__(self):
        # 每次调用 iter() 都会返回一个新的迭代器
        return NodeIterator(self.head)

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

# 使用示例
ll = LinkedList()
ll.add(1)
ll.add(2)

# 第一次迭代
print([x for x in ll]) # 输出: [2, 1]
# 第二次迭代（互不影响）
print([x for x in ll]) # 输出: [2, 1]
```

### 2.3 使用生成器简化实现

利用 Python 的生成器（Generator），可以极大地简化 `__iter__` 的实现。生成器函数自动支持迭代器协议。

```python
class CountDown:
    def __init__(self, start):
        self.start = start

    def __iter__(self):
        num = self.start
        while num > 0:
            yield num
            num -= 1

# 使用示例
counter = CountDown(3)
for i in counter:
    print(i)
# 输出:
# 3
# 2
# 1
```

## 3. `for` 循环的底层原理

当执行 `for item in container:` 时，Python 实际上执行了以下步骤：

1.  调用 `iter(container)` 获取迭代器对象（内部调用 `container.__iter__()`）。
2.  在一个无限循环中，不断调用 `next(iterator)` 获取下一个值（内部调用 `iterator.__next__()`）。
3.  如果捕获到 `StopIteration` 异常，则终止循环。

**等价代码：**

```python
# for item in container:
#     do_something(item)

# 底层逻辑：
iterator = iter(container)
while True:
    try:
        item = next(iterator)
    except StopIteration:
        break
    do_something(item)
```

## 4. 常见陷阱与注意事项

1.  **返回非迭代器**: `__iter__` 必须返回一个实现了 `__next__` 的迭代器对象。如果返回其他类型（如 `list`），`iter()` 函数会尝试调用该返回值的 `__iter__`，但为了符合协议，最好直接返回迭代器。如果返回的对象没有 `__iter__` 方法，会抛出 `TypeError`。
2.  **迭代器耗尽**: 迭代器是单向的，一旦耗尽（抛出 `StopIteration`），就不能再次使用。如果需要多次迭代，应实现为可迭代对象（每次 `__iter__` 返回新迭代器）。
3.  **与 `__getitem__` 的关系**: 如果一个对象没有实现 `__iter__`，但实现了 `__getitem__`，Python 会创建一个默认迭代器，尝试从索引 0 开始顺序访问，直到抛出 `IndexError`。这是一种向后兼容机制，但现代 Python 代码应优先实现 `__iter__`。

## 5. 参考资料

1.  **Python 官方文档**: [3. Data model - Special method names](https://docs.python.org/3/reference/datamodel.html#object.__iter__)
2.  **Real Python**: [Iterators and Iterables in Python](https://realpython.com/python-iterators-iterables/)
3.  **Finxter**: [Python __iter__() Magic Method](https://blog.finxter.com/python-__iter__-magic-method/)
