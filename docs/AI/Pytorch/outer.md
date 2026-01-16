`torch.outer` 是 PyTorch 中用于计算两个向量**外积（Outer Product）**的函数。它在线性代数、位置编码（RoPE）以及协方差矩阵计算中有着广泛应用。

## 1. 函数签名与数学定义

### 函数签名
```python
torch.outer(input, vec2, *, out=None) → Tensor
```

### 数学定义
给定两个一维向量：
-   输入向量 $u$ (形状为 $N$)
-   输入向量 $v$ (形状为 $M$)

`torch.outer` 计算出的矩阵 $A$ (形状为 $N \times M$)，其元素 $A_{ij}$ 定义为：
$$
A_{ij} = u_i \times v_j
$$

即矩阵的每一列是向量 $u$ 的缩放，每一行是向量 $v$ 的缩放。

$$
A = u \otimes v = u v^T = 
\begin{bmatrix}
u_1 v_1 & u_1 v_2 & \dots & u_1 v_M \\
u_2 v_1 & u_2 v_2 & \dots & u_2 v_M \\
\vdots & \vdots & \ddots & \vdots \\
u_N v_1 & u_N v_2 & \dots & u_N v_M
\end{bmatrix}
$$

## 2. 核心特性

1.  **输入限制**：两个输入必须都是**一维张量（1-D Vectors）**。如果是标量或高维张量，会抛出 RuntimeError。
2.  **不进行广播（Broadcasting）**：该函数不支持广播机制，必须严格符合外积定义。
3.  **输出形状**：输出矩阵的形状始终为 `(input.size(0), vec2.size(0))`。

## 3. 典型应用场景

### 3.1 旋转位置编码 (RoPE) 中的频率计算
在 Llama 等大模型中，需要预计算旋转角度矩阵。`precompute_freqs_cis` 函数通常使用 `torch.outer` 来结合位置索引 $m$ 和频率 $\theta$。

```python
import torch

# 假设 dim=4, max_seq_len=3
# 频率 theta: [theta_0, theta_1]
freqs = torch.tensor([1.0, 0.5]) 
# 位置 t: [0, 1, 2]
t = torch.arange(3)

# 计算 m * theta 矩阵
# 结果形状: [3, 2] -> [seq_len, dim/2]
emb = torch.outer(t, freqs)
print(emb)
# tensor([[0.0000, 0.0000],
#         [1.0000, 0.5000],
#         [2.0000, 1.0000]])
```

### 3.2 生成网格或掩码
快速生成加法表或乘法表。

```python
x = torch.arange(1, 4) # [1, 2, 3]
y = torch.arange(1, 3) # [1, 2]

# 乘法表
print(torch.outer(x, y))
# tensor([[1, 2],
#         [2, 4],
#         [3, 6]])
```

## 4. 与其他函数的区别

| 函数 | 行为 | 输入维度要求 |
| :--- | :--- | :--- |
| **`torch.outer`** | 向量外积 ($u v^T$) | 仅限 1D 向量 |
| **`torch.matmul` (@)** | 矩阵乘法 | 任意维度 (支持广播) |
| **`torch.mul` (*)** | 逐元素乘法 | 支持广播 |

**注意**：`torch.outer(a, b)` 等价于 `a.unsqueeze(1) @ b.unsqueeze(0)`，但 `outer` 更直观且可能有底层优化。

## 5. 常见错误

**错误 1：输入非向量**
```python
a = torch.tensor([[1, 2]]) # 2D tensor
b = torch.tensor([3, 4])
torch.outer(a, b) 
# RuntimeError: input must be 1-D, got 2-D
```
**修正**：使用 `flatten()` 或 `view(-1)` 将其展平。

## 参考资料
1.  [PyTorch Documentation: torch.outer](https://docs.pytorch.org/docs/stable/generated/torch.outer.html)
2.  [Wikipedia: Outer Product](https://en.wikipedia.org/wiki/Outer_product)
