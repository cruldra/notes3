`__getitem__` 是 Python 数据模型（Data Model）中用于实现对象索引（indexing）和切片（slicing）操作的核心魔术方法。它允许其实例通过方括号 `[]` 语法访问内部元素，是实现序列（Sequence）和映射（Mapping）协议的关键。

## 1. 方法定义与调用机制

当对一个对象使用下标访问语法 `obj[key]` 时，Python 解释器会在后台隐式调用该对象的 `__getitem__` 方法。

### 1.1 函数签名

```python
object.__getitem__(self, key)
```

-   **`self`**: 实例对象本身。
-   **`key`**: 索引键。对于序列类型，通常是整数或 `slice` 对象；对于映射类型，可以是任何可哈希（hashable）对象。

### 1.2 调用转换

Python 解释器会将语法糖转换为方法调用：

| 语法 | 内部调用 | 说明 |
| :--- | :--- | :--- |
| `instance[index]` | `type(instance).__getitem__(instance, index)` | 基础索引访问 |
| `instance[start:stop]` | `type(instance).__getitem__(instance, slice(start, stop))` | 切片访问 |
| `for x in instance:` | 依赖 `__getitem__` (若未定义 `__iter__`) | 迭代回退机制 |

## 2. 协议实现（Protocols）

`__getitem__` 的具体实现行为取决于对象是模仿**序列**还是**映射**。

### 2.1 序列协议 (Sequence Protocol)

若对象表现为序列（如 `list`, `tuple`, `str`），`__getitem__` 应遵循以下规则：

1.  **接受整数键**：`key` 为整数索引。
2.  **支持切片**：`key` 可能为 `slice` 对象。
3.  **负索引处理**：虽然 Python 解释器不会自动处理负索引，但按照惯例，实现者应将 `key < 0` 转换为 `len(self) + key`。
4.  **异常处理**：
    -   索引超出范围时，**必须**抛出 `IndexError`。这是 `for` 循环等迭代工具检测序列结束的标志。
    -   索引类型错误时，应抛出 `TypeError`。

**代码示例：自定义序列**

```python
class CustomSequence:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, index):
        # 显式处理切片
        if isinstance(index, slice):
            return self._data[index]
        
        # 显式处理整数索引
        if isinstance(index, int):
            if index < 0: # 处理负索引
                index += len(self._data)
            if index < 0 or index >= len(self._data):
                raise IndexError("CustomSequence index out of range")
            return self._data[index]
            
        raise TypeError(f"Invalid argument type: {type(index)}")

    def __len__(self):
        return len(self._data)

seq = CustomSequence([10, 20, 30])
print(seq[1])    # 输出: 20
print(seq[:2])   # 输出: [10, 20]
```

### 2.2 映射协议 (Mapping Protocol)

若对象表现为映射（如 `dict`），规则如下：

1.  **接受任意键**：`key` 可以是任何可哈希对象。
2.  **异常处理**：
    -   若键不存在，**必须**抛出 `KeyError`。
    -   这允许 `dict.get(key, default)` 等方法正常工作。

**代码示例：自定义映射**

```python
class CaseInsensitiveDict:
    def __init__(self, data):
        self._data = {k.lower(): v for k, v in data.items()}

    def __getitem__(self, key):
        return self._data[key.lower()]

cid = CaseInsensitiveDict({"Name": "Alice"})
print(cid["name"]) # 输出: Alice
```

## 3. 切片原理 (Slicing)

在 Python 3 中，切片操作不再调用已废弃的 `__getslice__`，而是将 `slice` 对象直接传递给 `__getitem__`。

`slice` 对象包含三个只读属性：`start`, `stop`, `step`。

```python
class Slicer:
    def __getitem__(self, key):
        if isinstance(key, slice):
            return f"Slice: start={key.start}, stop={key.stop}, step={key.step}"
        return f"Index: {key}"

s = Slicer()
print(s[1:5:2]) # 输出: Slice: start=1, stop=5, step=2
```

## 4. 与其他魔术方法的交互

### 4.1 迭代 (`__iter__`)

当对对象进行迭代（如 `for x in obj`）时，Python 会按以下优先级查找：
1.  `__iter__`：若存在，优先使用迭代器协议。
2.  `__getitem__`：若 `__iter__` 不存在，Python 会创建一个默认迭代器，从索引 `0` 开始尝试调用 `__getitem__`，直到捕获 `IndexError` 为止。

### 4.2 缺失键处理 (`__missing__`)

仅对 `dict` 的子类有效。如果 `__getitem__` 未找到键，且对象定义了 `__missing__`，则会调用 `__missing__(key)` 而不是直接抛出 `KeyError`。这通常用于实现 `defaultdict` 行为。

### 4.3 类型提示 (`__class_getitem__`)

从 Python 3.7 (PEP 560) 开始，引入了 `__class_getitem__` 以支持泛型类型的运行时下标访问（例如 `List[int]`）。
-   `__getitem__` 用于**实例**索引（`obj[0]`）。
-   `__class_getitem__` 用于**类**索引（`cls[int]`），主要用于类型提示系统。

## 5. 底层实现细节 (CPython)

在 CPython 层面，`__getitem__` 对应于 C 结构体 `PyTypeObject` 中的特定槽位（slots）：

-   **序列**：对应 `tp_as_sequence->sq_item`。
-   **映射**：对应 `tp_as_mapping->mp_subscript`。

当定义了 `__getitem__` 的 Python 类被创建时，解释器会根据该方法的实现填充这些槽位。如果对象同时表现得像序列和映射（极少见），`mp_subscript` 优先级通常更高。

---

## 参考资料

1.  **Python 官方文档**: [3. Data model - Special method names](https://docs.python.org/3/reference/datamodel.html#object.__getitem__)
2.  **Rafe Kettler**: [A Guide to Python's Magic Methods](https://rszalski.github.io/magicmethods/#sequence)
3.  **Real Python**: [Python's Magic Methods: Leverage Their Power in Your Classes](https://realpython.com/python-magic-methods/)
