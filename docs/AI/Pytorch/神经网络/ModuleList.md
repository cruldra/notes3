`nn.ModuleList` 是 PyTorch 中用于存储子模块（Submodules）的列表容器。它看起来和 Python 原生的 `list` 非常相似，但有一个至关重要的区别：**它能自动注册子模块**。

## 1. 为什么需要 `nn.ModuleList`？

如果你在 `nn.Module` 的 `__init__` 方法中使用普通的 Python `list` 来保存层，PyTorch 的优化器（Optimizer）将无法识别这些层的参数。

### 错误示例：使用 Python List
```python
import torch
import torch.nn as nn

class BadModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 错误：普通的 list 无法被 PyTorch 追踪
        self.layers = [nn.Linear(10, 10) for _ in range(5)]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

model = BadModel()
# 打印参数列表为空！优化器无法更新这些层。
print(list(model.parameters()))  # []
```

### 正确示例：使用 `nn.ModuleList`
```python
class GoodModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 正确：ModuleList 会自动将这些层注册为模型的子模块
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])

    def forward(self, x):
        # 像普通列表一样遍历
        for layer in self.layers:
            x = layer(x)
        return x

model = GoodModel()
# 现在可以看到参数了
print(len(list(model.parameters())))  # 10 (5个权重矩阵 + 5个偏置向量)
```

## 2. 核心特性

1.  **参数注册**：放入 `ModuleList` 的 `nn.Module` 会自动添加到整个模型的 `.parameters()` 中。
2.  **索引与迭代**：支持索引访问（`self.layers[0]`）、切片（`self.layers[:2]`）和迭代（`for layer in self.layers:`）。
3.  **没有 `forward` 方法**：与 `nn.Sequential` 不同，`ModuleList` **没有定义** `forward` 方法。这意味着你不能直接调用 `model.layers(x)`。你必须在自己的 `forward` 方法中显式地遍历它。

## 3. `nn.ModuleList` vs `nn.Sequential`

| 特性 | `nn.ModuleList` | `nn.Sequential` |
| :--- | :--- | :--- |
| **用途** | 存储一组模块，需要灵活控制前向传播逻辑 | 存储一组模块，按顺序串联执行 |
| **`forward`** | 无（需手动实现循环） | 有（自动按顺序传递数据） |
| **灵活性** | 高（可以在循环中加残差、条件判断等） | 低（只能一条路走到黑） |
| **场景** | ResNet Block、Attention Headers、多分支结构 | 简单的层堆叠（如 `Conv2d -> ReLU -> MaxPool`） |

### 场景对比

**场景 A：简单的层堆叠 -> 用 `Sequential`**
```python
self.block = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 10)
)
# forward 中只需一行：return self.block(x)
```

**场景 B：需要复用或复杂连接 -> 用 `ModuleList`**
```python
self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])

# forward 中可以做骚操作
def forward(self, x):
    for i, layer in enumerate(self.layers):
        x = layer(x)
        if i % 2 == 0:  # 偶数层加个残差
            x = x + input_x 
    return x
```

## 4. 常用方法

*   `append(module)`: 在末尾添加模块。
*   `extend(modules)`: 添加可迭代对象中的多个模块。
*   `insert(index, module)`: 在指定索引插入模块。

## 参考资料
1.  [PyTorch Docs: nn.ModuleList](https://docs.pytorch.org/docs/stable/generated/torch.nn.ModuleList.html)
2.  [StackOverflow: ModuleList vs Sequential](https://stackoverflow.com/questions/47544051/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential)
3.  [PyTorch Forums: Difference from Python List](https://discuss.pytorch.org/t/whats-the-difference-between-nn-modulelist-and-python-list/106401)
