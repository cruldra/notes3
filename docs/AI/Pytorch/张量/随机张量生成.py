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
    # PyTorch 随机张量生成与唯一值操作

本笔记本介绍四个常用的 PyTorch API：
    - `torch.randn` - 生成标准正态分布随机数
    - `torch.randint` - 生成随机整数
    - `torch.arange` - 生成等差数列张量
    - `torch.unique` - 获取张量中的唯一值
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. torch.randn

    `torch.randn(*size)` 返回一个由**标准正态分布**（均值为0，方差为1）中抽取的随机数填充的张量。

    ### 函数签名
    ```python
    torch.randn(*size, out=None, dtype=None, layout=torch.strided,
                device=None, requires_grad=False)
    ```

    ### 参数说明
    - `size`: 定义输出张量形状的整数序列
    - `dtype`: 返回张量的数据类型，默认为 `torch.float32`
    - `device`: 返回张量所在的设备
    - `requires_grad`: 是否需要计算梯度
    """)
    return


@app.cell
def _(torch):
    # 创建一个 3x4 的随机张量
    randn_2d = torch.randn(3, 4)
    print("2D 随机张量 (3x4):")
    print(randn_2d)
    return


@app.cell
def _(torch):
    # 创建一个 2x3x4 的三维随机张量
    randn_3d = torch.randn(2, 3, 4)
    print("3D 随机张量 (2x3x4):")
    print(randn_3d)
    print(f"{randn_3d.ndim}维")  # 输出维度数量：3
    print(randn_3d.shape)  # 输出具体形状：torch.Size([2, 3, 4])
    return


@app.cell
def _(torch):
    # 验证 randn 生成的数据符合标准正态分布
    large_tensor = torch.randn(100000)
    print(f"均值 (期望接近 0): {large_tensor.mean():.4f}")
    print(f"标准差 (期望接近 1): {large_tensor.std():.4f}")
    return


@app.cell
def _(torch):
    # 指定数据类型和设备
    randn_float64 = torch.randn(2, 3, dtype=torch.float64)
    print(f"float64 张量:\n{randn_float64}")
    print(f"数据类型: {randn_float64.dtype}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## 2. torch.randint

    `torch.randint(low, high, size)` 返回一个在 `[low, high)` 区间内均匀分布的随机整数张量。

    ### 函数签名
    ```python
    torch.randint(low=0, high, size, *, generator=None, out=None,
                  dtype=None, layout=torch.strided, device=None,
                  requires_grad=False)
    ```

    ### 参数说明
    - `low`: 最小值（包含），默认为 0
    - `high`: 最大值（不包含）
    - `size`: 输出张量的形状（元组）
    """)
    return

@app.cell
def _(torch):
    # 生成 0-9 之间的随机整数
    randint_basic = torch.randint(0, 10, (3, 4))
    print("0-9 之间的随机整数 (3x4):")
    print(randint_basic)
    return


@app.cell
def _(torch):
    # 只指定 high 参数（low 默认为 0）
    randint_simple = torch.randint(5, (2, 3))
    print("0-4 之间的随机整数 (2x3):")
    print(randint_simple)
    return


@app.cell
def _(torch):
    # 生成负数范围的随机整数
    randint_negative = torch.randint(-5, 5, (2, 4))
    print("-5 到 4 之间的随机整数:")
    print(randint_negative)
    return


@app.cell
def _(torch):
    # 模拟骰子投掷
    dice_rolls = torch.randint(1, 7, (10,))
    print(f"10次骰子投掷结果: {dice_rolls.tolist()}")
    return


@app.cell
def _(torch):
    # 指定数据类型
    randint_int8 = torch.randint(0, 100, (3, 3), dtype=torch.int8)
    print(f"int8 类型的随机整数:\n{randint_int8}")
    print(f"数据类型: {randint_int8.dtype}")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. torch.arange

    `torch.arange(start, end, step)` 返回一个等差数列张量，区间为 `[start, end)`。

    ### 函数签名
    ```python
    torch.arange(start=0, end, step=1, *, out=None, dtype=None,
                 layout=torch.strided, device=None, requires_grad=False)
    ```

    ### 参数说明
    - `start`: 起始值（包含），默认为 0
    - `end`: 结束值（不包含）
    - `step`: 步长，默认为 1
    """)
    return





@app.cell
def _(torch):
    # 基础用法：生成从 0 到 4 的等差数列
    arange_basic = torch.arange(5)
    print(f"等差数列(0-4): {arange_basic.tolist()}")
    return


@app.cell
def _(torch):
    # 指定起止与步长
    arange_step = torch.arange(2, 10, 2)
    print(f"等差数列(2-8, step=2): {arange_step.tolist()}")
    return


@app.cell
def _(torch):
    # 指定数据类型和设备
    arange_typed = torch.arange(0, 5, 0.5, dtype=torch.float64)
    print(f"float64 等差数列:\n{arange_typed}")
    print(f"数据类型: {arange_typed.dtype}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. torch.unique

    `torch.unique(input)` 返回输入张量中的唯一元素。

    ### 函数签名
    ```python
    torch.unique(input, sorted=True, return_inverse=False,
                 return_counts=False, dim=None)
    ```

    ### 参数说明
    - `input`: 输入张量
    - `sorted`: 是否对唯一值进行排序，默认 True
    - `return_inverse`: 是否返回原始张量到唯一值的索引映射
    - `return_counts`: 是否返回每个唯一值的计数
    - `dim`: 沿指定维度计算唯一值
    """)
    return


@app.cell
def _(torch):
    # 基本用法：获取唯一值
    tensor_with_duplicates = torch.tensor([1, 3, 2, 3, 1, 4, 2, 1])
    unique_values = torch.unique(tensor_with_duplicates)
    print(f"原始张量: {tensor_with_duplicates.tolist()}")
    print(f"唯一值: {unique_values.tolist()}")
    return (tensor_with_duplicates,)


@app.cell
def _(tensor_with_duplicates, torch):
    # 获取唯一值和计数
    unique_vals, counts = torch.unique(tensor_with_duplicates, return_counts=True)
    print(f"唯一值: {unique_vals.tolist()}")
    print(f"计数: {counts.tolist()}")
    return


@app.cell
def _(tensor_with_duplicates, torch):
    # 获取逆向索引（可用于重建原始张量）
    unique_v, inverse_indices = torch.unique(
        tensor_with_duplicates, return_inverse=True
    )
    print(f"唯一值: {unique_v.tolist()}")
    print(f"逆向索引: {inverse_indices.tolist()}")
    print(f"重建原始张量: {unique_v[inverse_indices].tolist()}")
    return


@app.cell
def _(torch):
    # 同时返回逆向索引和计数
    data = torch.tensor([5, 2, 8, 2, 5, 5, 8])
    unique_all, inverse_all, counts_all = torch.unique(
        data, return_inverse=True, return_counts=True
    )
    print(f"原始数据: {data.tolist()}")
    print(f"唯一值: {unique_all.tolist()}")
    print(f"逆向索引: {inverse_all.tolist()}")
    print(f"计数: {counts_all.tolist()}")
    return


@app.cell
def _(torch):
    # 2D 张量的唯一值
    tensor_2d = torch.tensor([[1, 2], [3, 1], [1, 2], [4, 5]])
    print(f"2D 张量:\n{tensor_2d}")

    # 沿 dim=0 获取唯一行
    unique_rows = torch.unique(tensor_2d, dim=0)
    print(f"\n唯一行 (dim=0):\n{unique_rows}")
    return


@app.cell
def _(torch):
    # 浮点数张量的唯一值
    float_tensor = torch.tensor([1.1, 2.2, 1.1, 3.3, 2.2])
    unique_floats = torch.unique(float_tensor)
    print(f"浮点数张量: {float_tensor.tolist()}")
    print(f"唯一浮点值: {unique_floats.tolist()}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 综合示例：组合使用

    下面展示如何组合使用这三个函数完成实际任务。
    """)
    return


@app.cell
def _(torch):
    # 示例：生成随机分类标签并统计分布
    # 生成 100 个样本的随机分类标签（0-4 共 5 个类别）
    labels = torch.randint(0, 5, (100,))

    # 获取唯一类别和每个类别的样本数
    classes, class_counts = torch.unique(labels, return_counts=True)

    print("分类标签分布统计:")
    for c, n in zip(classes.tolist(), class_counts.tolist()):
        print(f"  类别 {c}: {n} 个样本 ({n}%)")
    return


@app.cell
def _(torch):
    # 示例：生成随机特征矩阵并归一化
    # 生成 5 个样本，每个样本 3 个特征的随机矩阵
    features = torch.randn(5, 3)
    print("原始特征矩阵:")
    print(features)

    # 按行归一化（L2 范数）
    normalized = features / features.norm(dim=1, keepdim=True)
    print("\n归一化后的特征矩阵:")
    print(normalized)

    # 验证每行的范数为 1
    print(f"\n每行的范数: {normalized.norm(dim=1).tolist()}")
    return


@app.cell
def _(torch):
    # 示例：模拟批量数据的随机采样
    # 假设有 1000 条数据，随机选择 10 个不重复的索引
    total_samples = 1000
    batch_size = 10

    # 生成随机索引
    random_indices = torch.randint(0, total_samples, (batch_size * 2,))

    # 使用 unique 去重，取前 batch_size 个
    unique_indices = torch.unique(random_indices)[:batch_size]

    print(f"随机采样的 {batch_size} 个索引: {unique_indices.tolist()}")
    return


if __name__ == "__main__":
    app.run()
