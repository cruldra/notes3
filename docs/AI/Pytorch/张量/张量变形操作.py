import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import torch
    return mo, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 核心前提：元素总量守恒

    无论你怎么变形状，**元素的总个数必须保持不变**。
    如果你有一个形状为$(2,3)$的张量（总共 6 个元素），你可以把它变成 $(1,6)$或$(6,1)$或 ，但绝对不能变成$(2,4)$。

    数学表达式：

    $A \times B = A' \times B'$

    # PyTorch 张量变形操作详解

    本笔记本详细介绍 PyTorch 中五个核心的张量变形 API：

    | API | 作用 | 关键特点 |
    |-----|------|----------|
    | `.view()` | 改变张量形状 | 要求内存连续，返回视图 |
    | `.reshape()` | 改变张量形状 | 自动处理非连续情况 |
    | `.contiguous()` | 确保内存连续 | 必要时创建副本 |
    | `.transpose()` | 交换两个维度 | 返回视图，不连续 |
    | `.permute()` | 任意重排所有维度 | 返回视图，不连续 |

    ## 核心概念：视图(View) vs 拷贝(Copy)

    - **视图**：与原张量共享内存，修改一个会影响另一个
    - **拷贝**：独立的内存空间，互不影响

    理解这个概念对正确使用这些 API 至关重要。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. view() - 改变张量视图

    `view(*shape)` 返回一个具有新形状的张量**视图**，与原张量共享数据。

    ### 函数签名
    ```python
    Tensor.view(*shape) -> Tensor
    ```

    ### 关键特点
    - ✅ 返回视图（共享内存）
    - ✅ 支持 `-1` 自动推断维度
    - ⚠️ **要求张量必须内存连续**
    """)
    return


@app.cell
def _(torch):
    # 基本用法：将 1D 张量变为 2D
    t1 = torch.arange(12)
    print(f"原始张量: {t1}")
    print(f"原始形状: {t1.shape}")

    t1_view = t1.view(3, 4)
    print(f"\nview(3, 4) 后:")
    print(t1_view)
    print(f"新形状: {t1_view.shape}")
    return t1, t1_view


@app.cell
def _(t1, t1_view):
    # 验证视图共享内存：修改 view 会影响原张量
    t1_view[0, 0] = 999
    print("修改 t1_view[0, 0] = 999 后:")
    print(f"t1_view:\n{t1_view}")
    print(f"原始 t1: {t1}")  # 原张量也被修改了！
    return


@app.cell
def _(torch):
    # 使用 -1 自动推断维度
    t2 = torch.arange(24)

    # -1 表示该维度由其他维度自动计算
    print(f"view(2, -1): {t2.view(2, -1).shape}")  # 自动推断为 12
    print(f"view(-1, 6): {t2.view(-1, 6).shape}")  # 自动推断为 4
    print(f"view(2, 3, -1): {t2.view(2, 3, -1).shape}")  # 自动推断为 4
    return


@app.cell
def _(torch):
    # view 的限制：必须内存连续
    t3 = torch.arange(6).view(2, 3)
    print(f"原始张量:\n{t3}")
    print(f"是否连续: {t3.is_contiguous()}")

    # 转置后不再连续
    t3_t = t3.t()  # 或 t3.transpose(0, 1)
    print(f"\n转置后:\n{t3_t}")
    print(f"是否连续: {t3_t.is_contiguous()}")
    return (t3_t,)


@app.cell
def _(t3_t):
    # 对非连续张量调用 view 会报错
    try:
        t3_t.view(6)
    except RuntimeError as e:
        print(f"❌ 错误: {e}")
        print("\n💡 解决方案: 先调用 .contiguous()，或使用 .reshape()")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. reshape() - 重塑张量形状

    `reshape(*shape)` 与 `view()` 类似，但能自动处理非连续张量。

    ### 函数签名
    ```python
    Tensor.reshape(*shape) -> Tensor
    ```

    ### 与 view() 的区别
    | 特性 | view() | reshape() |
    |------|--------|-----------|
    | 内存连续要求 | 必须连续 | 自动处理 |
    | 返回类型 | 始终是视图 | 视图或拷贝 |
    | 性能 | 更快 | 可能涉及拷贝 |

    ### 使用建议
    - 确定连续时用 `view()`（更明确）
    - 不确定时用 `reshape()`（更安全）
    """)
    return


@app.cell
def _(torch):
    # reshape 基本用法（与 view 相同）
    r1 = torch.arange(12)
    print(r1)
    r1_reshaped = r1.reshape(3, 4)
    print(f"reshape(3, 4):\n{r1_reshaped}")
    return


@app.cell
def _(torch):
    # reshape 处理非连续张量（view 做不到）
    r2 = torch.arange(6).view(2, 3)
    r2_t = r2.t()  # 转置后不连续

    print(f"转置后的张量:\n{r2_t}")
    print(f"是否连续: {r2_t.is_contiguous()}")

    # reshape 可以正常工作
    r2_flat = r2_t.reshape(6)
    print(f"\nreshape(6) 成功: {r2_flat}")
    return


@app.cell
def _(torch):
    # 判断 reshape 返回的是视图还是拷贝
    r3 = torch.arange(6).view(2, 3)

    # 连续张量：返回视图
    r3_reshaped = r3.reshape(3, 2)
    r3_reshaped[0, 0] = 999
    print("连续张量 reshape 后修改:")
    print(f"r3_reshaped:\n{r3_reshaped}")
    print(f"原始 r3:\n{r3}")  # 被修改了 = 视图

    # 非连续张量：返回拷贝
    r4 = torch.arange(6).view(2, 3).t()
    r4_reshaped = r4.reshape(6)
    r4_reshaped[0] = 888
    print(f"\n非连续张量 reshape 后修改:")
    print(f"r4_reshaped: {r4_reshaped}")
    print(f"原始 r4:\n{r4}")  # 未被修改 = 拷贝
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. contiguous() - 确保内存连续

    `contiguous()` 返回一个内存连续的张量。如果已连续，返回自身；否则创建副本。

    ### 函数签名
    ```python
    Tensor.contiguous(memory_format=torch.contiguous_format) -> Tensor
    ```

    ### 什么是内存连续？

    张量在内存中按行优先（C 顺序）存储时，称为**连续的**。

    - 创建的张量默认是连续的
    - `transpose`、`permute` 等操作会改变元素的逻辑顺序，但不移动实际数据，导致不连续
    """)
    return


@app.cell
def _(torch):
    # 理解 stride（步长）
    c1 = torch.arange(6).view(2, 3)
    print(f"张量:\n{c1}")
    print(f"形状: {c1.shape}")
    print(f"步长 stride: {c1.stride()}")
    print("解释: 沿第0维移动1步需要跳过3个元素，沿第1维移动1步需要跳过1个元素")
    return


@app.cell
def _(torch):
    # 转置后 stride 变化
    c2 = torch.arange(6).view(2, 3)
    c2_t = c2.t()

    print(f"原始张量:\n{c2}")
    print(f"stride: {c2.stride()}, 连续: {c2.is_contiguous()}")

    print(f"\n转置后:\n{c2_t}")
    print(f"stride: {c2_t.stride()}, 连续: {c2_t.is_contiguous()}")
    print("解释: 转置后沿第0维移动1步只跳过1个元素，不符合行优先顺序")
    return


@app.cell
def _(torch):
    # contiguous() 的作用
    c3 = torch.arange(6).view(2, 3).t()
    print(f"非连续张量:\n{c3}")
    print(f"stride: {c3.stride()}, 连续: {c3.is_contiguous()}")

    c3_contig = c3.contiguous()
    print(f"\n调用 contiguous() 后:\n{c3_contig}")
    print(f"stride: {c3_contig.stride()}, 连续: {c3_contig.is_contiguous()}")
    return c3, c3_contig


@app.cell
def _(c3, c3_contig):
    # contiguous() 创建的是拷贝
    c3_contig[0, 0] = 999
    print("修改 c3_contig[0, 0] = 999:")
    print(f"c3_contig:\n{c3_contig}")
    print(f"原始 c3:\n{c3}")  # 未被修改
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. transpose() - 交换两个维度

    `transpose(dim0, dim1)` 交换张量的两个指定维度。

    ### 函数签名
    ```python
    Tensor.transpose(dim0, dim1) -> Tensor
    ```

    ### 关键特点
    - ✅ 返回视图（共享内存）
    - ⚠️ 结果通常**不连续**
    - 📝 对于 2D 张量，`.t()` 是 `.transpose(0, 1)` 的简写
    """)
    return


@app.cell
def _(torch):
    # 2D 矩阵转置
    tr1 = torch.arange(6).view(2, 3)
    print(f"原始矩阵 (2x3):\n{tr1}")

    tr1_t = tr1.transpose(0, 1)  # 等价于 tr1.t()
    print(f"\ntranspose(0, 1) 后 (3x2):\n{tr1_t}")
    return


@app.cell
def _(torch):
    # 3D 张量转置
    tr2 = torch.arange(24).view(2, 3, 4)
    print(f"原始形状: {tr2.shape}")
    print(f"原始张量:\n{tr2}")

    # 交换第1维和第2维
    tr2_t = tr2.transpose(1, 2)
    print(f"\ntranspose(1, 2) 后形状: {tr2_t.shape}")
    print(f"转置后张量:\n{tr2_t}")
    return


@app.cell
def _(torch):
    # transpose 返回视图
    tr3 = torch.arange(6).view(2, 3)
    tr3_t = tr3.transpose(0, 1)

    tr3_t[0, 0] = 999
    print("修改 tr3_t[0, 0] = 999:")
    print(f"tr3_t:\n{tr3_t}")
    print(f"原始 tr3:\n{tr3}")  # 也被修改了
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. permute() - 任意重排所有维度

    `permute(*dims)` 按指定顺序重新排列张量的所有维度。

    ### 函数签名
    ```python
    Tensor.permute(*dims) -> Tensor
    ```

    ### 与 transpose 的区别
    | 特性 | transpose() | permute() |
    |------|-------------|-----------|
    | 操作维度数 | 只能交换2个 | 可重排所有 |
    | 参数 | 两个维度索引 | 新的维度顺序 |
    | 典型场景 | 简单转置 | 复杂维度变换 |

    ### 典型应用
    图像数据格式转换：`(H, W, C)` ↔ `(C, H, W)`
    """)
    return


@app.cell
def _(torch):
    # permute 基本用法
    p1 = torch.arange(24).view(2, 3, 4)
    print(f"原始形状: {p1.shape}")  # (2, 3, 4)

    # 将维度顺序从 (0, 1, 2) 变为 (2, 0, 1)
    p1_permuted = p1.permute(2, 0, 1)
    print(f"permute(2, 0, 1) 后: {p1_permuted.shape}")  # (4, 2, 3)
    return


@app.cell
def _(torch):
    # 图像格式转换：HWC -> CHW
    # 假设一张 RGB 图像，形状为 (高度=4, 宽度=5, 通道=3)
    image_hwc = torch.randn(4, 5, 3)
    print(f"HWC 格式 (高x宽x通道): {image_hwc.shape}")

    # PyTorch 卷积层需要 CHW 格式
    image_chw = image_hwc.permute(2, 0, 1)
    print(f"CHW 格式 (通道x高x宽): {image_chw.shape}")

    # 批量图像：NHWC -> NCHW
    batch_nhwc = torch.randn(8, 224, 224, 3)  # 8张 224x224 RGB图像
    batch_nchw = batch_nhwc.permute(0, 3, 1, 2)
    print(f"\n批量转换: {batch_nhwc.shape} -> {batch_nchw.shape}")
    return


@app.cell
def _(torch):
    # permute 也返回视图
    p2 = torch.arange(6).view(2, 3)
    p2_permuted = p2.permute(1, 0)  # 等价于 transpose(0, 1)

    p2_permuted[0, 0] = 999
    print("修改 p2_permuted[0, 0] = 999:")
    print(f"p2_permuted:\n{p2_permuted}")
    print(f"原始 p2:\n{p2}")  # 也被修改了
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. 综合对比与选择建议

    ### API 对比表

    | API | 返回类型 | 内存连续要求 | 典型用途 |
    |-----|----------|--------------|----------|
    | `view()` | 视图 | ✅ 必须连续 | 明确知道连续时的形状变换 |
    | `reshape()` | 视图/拷贝 | ❌ 无要求 | 不确定连续性时的形状变换 |
    | `contiguous()` | 自身/拷贝 | - | 确保内存连续 |
    | `transpose()` | 视图 | ❌ 无要求 | 交换两个维度 |
    | `permute()` | 视图 | ❌ 无要求 | 任意重排所有维度 |

    ### 选择决策树

    ```
    需要改变形状？
    ├── 是 → 张量确定连续？
    │       ├── 是 → 用 view()
    │       └── 否/不确定 → 用 reshape()
    └── 否 → 需要调整维度顺序？
            ├── 只交换2个维度 → 用 transpose()
            └── 重排多个维度 → 用 permute()

    后续需要 view()？
    └── 是 → 先调用 contiguous()
    ```
    """)
    return


@app.cell
def _(torch):
    # 实战示例1：Transformer 中的注意力机制维度变换
    batch_size, seq_len, d_model = 2, 10, 512
    num_heads, d_k = 8, 64

    # 输入: (batch, seq_len, d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"输入形状: {x.shape}")

    # 拆分多头: (batch, seq_len, num_heads, d_k)
    x_split = x.view(batch_size, seq_len, num_heads, d_k)
    print(f"拆分多头: {x_split.shape}")

    # 调整为: (batch, num_heads, seq_len, d_k)
    x_transposed = x_split.transpose(1, 2)
    print(f"转置后: {x_transposed.shape}")

    # 或者一步到位用 permute
    x_permuted = x_split.permute(0, 2, 1, 3)
    print(f"permute 结果: {x_permuted.shape}")
    return


@app.cell
def _(torch):
    # 实战示例2：卷积层输出展平为全连接层输入
    # 假设卷积输出: (batch=4, channels=64, height=7, width=7)
    conv_output = torch.randn(4, 64, 7, 7)
    print(f"卷积输出形状: {conv_output.shape}")

    # 方法1: 直接 view (连续张量)
    flat1 = conv_output.view(4, -1)  # (4, 64*7*7) = (4, 3136)
    print(f"view 展平: {flat1.shape}")

    # 方法2: reshape (更安全)
    flat2 = conv_output.reshape(4, -1)
    print(f"reshape 展平: {flat2.shape}")

    # 方法3: flatten (推荐，语义更清晰)
    flat3 = conv_output.flatten(start_dim=1)
    print(f"flatten 展平: {flat3.shape}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. 常见错误与解决方案

    | 错误 | 原因 | 解决方案 |
    |------|------|----------|
    | `view size is not compatible` | 元素总数不匹配 | 确保变形前后元素数量相同 |
    | `cannot view as non-contiguous` | 对非连续张量调用 view | 先 `.contiguous()` 或用 `.reshape()` |
    | 修改视图影响原数据 | view/transpose/permute 返回视图 | 需要独立副本时用 `.clone()` |
    """)
    return


@app.cell
def _(torch):
    # 错误1: 元素数量不匹配
    err1 = torch.arange(12)
    try:
        err1.view(3, 5)  # 12 != 3*5=15
    except RuntimeError as e:
        print(f"❌ 错误: {e}")
    return


@app.cell
def _(torch):
    # 安全创建独立副本
    original = torch.arange(6).view(2, 3)

    # 创建独立副本而非视图
    independent_copy = original.view(3, 2).clone()
    independent_copy[0, 0] = 999

    print(f"修改副本后:")
    print(f"副本:\n{independent_copy}")
    print(f"原始:\n{original}")  # 未被修改
    return


if __name__ == "__main__":
    app.run()
