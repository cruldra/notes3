import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 可视化理解 PyTorch 张量 (Tensor)

    **张量 (Tensor)** 是 PyTorch 中的核心数据结构。你可以把它想象成是一个**多维数组**。

    在这个笔记本中，我们将通过可视化的方式，从 0 维到 3 维逐步解构张量。
    """)
    return


@app.cell
def _():
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    return plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. 标量 (Scalar) - 0维张量
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    标量就是一个单纯的数字。它没有方向，只有大小。

    *   **维度 (ndim/dim)**: 0
    *   **形状 (shape)**: `torch.Size([])`
    """)
    return


@app.cell
def _(mo):
    scalar_val = mo.ui.number(start=-100, stop=100, value=42, label="改变数值")
    return (scalar_val,)


@app.cell
def _(mo, torch, scalar_val):
    def show_scalar(val):
        t = torch.tensor(val)
        return mo.md(
            f"""
            **Python 数值**: `{val}`

            **PyTorch 张量**: 
            ```python
            {t}
            ```
            *   Shape: `{t.shape}`
            *   Type: `{t.dtype}`
            """
        )

    mo.vstack([scalar_val, show_scalar(scalar_val.value)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. 向量 (Vector) - 1维张量
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    向量是一排数字。它有一个轴（Axis 0）。

    *   **维度**: 1
    *   **形状**: `torch.Size([长度])`
    """)
    return


@app.cell
def _(mo):
    vec_len = mo.ui.slider(start=1, stop=20, value=10, label="向量长度")
    return (vec_len,)


@app.cell
def _(mo, plt, torch, vec_len):
    def plot_vector(length):
        # 创建一个随机向量 (0-10之间)
        t = torch.arange(length).float()

        fig, ax = plt.subplots(figsize=(8, 1))
        # 用热力图展示
        im = ax.imshow(t.unsqueeze(0), cmap='Blues', aspect='auto')

        # 在格子里显示数值
        for i in range(length):
            ax.text(i, 0, f"{int(t[i])}", ha="center", va="center", color="black" if i < length/2 else "white")

        ax.set_yticks([])
        ax.set_xticks(range(length))
        ax.set_title(f"1D Tensor (Shape: {t.shape})")
        return fig

    mo.vstack([
        vec_len,
        plot_vector(vec_len.value)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. 矩阵 (Matrix) - 2维张量
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    矩阵是数字的网格。它有两个轴：行 (Row) 和 列 (Column)。

    *   **维度**: 2
    *   **形状**: `torch.Size([行数, 列数])`
    """)
    return


@app.cell
def _(mo):
    rows = mo.ui.slider(start=1, stop=10, value=4, label="行数 (Rows)")
    cols = mo.ui.slider(start=1, stop=10, value=5, label="列数 (Cols)")
    return (rows, cols)


@app.cell
def _(mo, plt, torch, rows, cols):
    def plot_matrix(r, c):
        # 创建一个顺序矩阵
        t = torch.arange(r * c).reshape(r, c)

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(t, cmap='viridis')

        # 显示数值
        for i in range(r):
            for j in range(c):
                val = int(t[i, j])
                # 根据背景深浅自动调整文字颜色
                text_color = "white" if val < (r*c)/2 else "black" # Viridis: low is purple(dark), high is yellow(light)
                # 其实Viridis是两头亮中间暗? 简单起见，反色处理
                # Viridis: 0(purple) -> ... -> max(yellow)
                ax.text(j, i, str(val), ha="center", va="center", color="white")

        ax.set_title(f"2D Tensor (Shape: {t.shape})")
        ax.set_xlabel("Axis 1 (Columns)")
        ax.set_ylabel("Axis 0 (Rows)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return fig

    mo.hstack([
        mo.vstack([rows, cols]),
        plot_matrix(rows.value, cols.value)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. 3维张量 (3D Tensor) - 彩色图像
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    3维张量就像是一叠矩阵。最常见的例子是 **彩色图像**。

    图像通常表示为 `[通道数, 高度, 宽度]` (PyTorch 格式: C, H, W)。

    *   **Channel 0 (Red)**: 红色通道矩阵
    *   **Channel 1 (Green)**: 绿色通道矩阵
    *   **Channel 2 (Blue)**: 蓝色通道矩阵
    """)
    return


@app.cell
def _(mo, plt, torch):
    # 生成一个简单的RGB图像数据
    def get_rgb_tensor():
        h, w = 10, 10
        # R通道：左边亮
        r = torch.zeros(h, w)
        r[:, :5] = 1.0 

        # G通道：上面亮
        g = torch.zeros(h, w)
        g[:5, :] = 1.0

        # B通道：中间亮
        b = torch.zeros(h, w)
        b[3:7, 3:7] = 1.0

        # 堆叠成 (3, H, W)
        img_tensor = torch.stack([r, g, b])
        return img_tensor

    def plot_3d_tensor():
        img_tensor = get_rgb_tensor()

        # PyTorch (C, H, W) -> Matplotlib (H, W, C)
        img_permuted = img_tensor.permute(1, 2, 0)

        fig, axes = plt.subplots(1, 4, figsize=(15, 4))

        # 显示合成图
        axes[0].imshow(img_permuted)
        axes[0].set_title(f"RGB Image\nShape: {img_tensor.shape}")
        axes[0].axis('off')

        # 显示 R 通道
        axes[1].imshow(img_tensor[0], cmap='Reds', vmin=0, vmax=1)
        axes[1].set_title("Channel 0 (Red)")
        axes[1].axis('off')

        # 显示 G 通道
        axes[2].imshow(img_tensor[1], cmap='Greens', vmin=0, vmax=1)
        axes[2].set_title("Channel 1 (Green)")
        axes[2].axis('off')

        # 显示 B 通道
        axes[3].imshow(img_tensor[2], cmap='Blues', vmin=0, vmax=1)
        axes[3].set_title("Channel 2 (Blue)")
        axes[3].axis('off')

        return fig

    mo.vstack([
        mo.md("**交互体验**：这里没有滑块，但展示了通过叠加三个 2D 矩阵形成 3D 张量的过程。"),
        plot_3d_tensor()
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. 形状变换 (Reshape / View)

    张量最强大的地方在于它可以随意改变形状，只要**元素总数**不变。

    我们可以把一个 `(4, 3)` 的矩阵（12个元素）变成 `(12,)` 的向量，或者 `(2, 6)` 的矩阵。
    """)
    return


@app.cell
def _(mo, torch):
    # 固定12个元素
    base_tensor = torch.arange(12) + 1

    # 定义合法的形状组合 (乘积为12)
    shapes = [
        (12,),
        (1, 12),
        (12, 1),
        (3, 4),
        (4, 3),
        (2, 6),
        (6, 2),
        (2, 2, 3)
    ]

    shape_labels = [str(s) for s in shapes]
    shape_selector = mo.ui.dropdown(options=shape_labels, value="(4, 3)", label="选择目标形状")
    return (shape_selector, base_tensor)


@app.cell
def _(mo, plt, shape_selector, base_tensor):
    def plot_reshape(shape_str):
        # 解析字符串回元组
        shape = eval(shape_str)

        reshaped_t = base_tensor.view(shape)

        # 无论多少维，我们都把它画成 2D 铺开来看 (除了1D)
        # 为了可视化方便，如果是 3D+，我们还是把它们平铺开

        if len(shape) == 1:
            # 1D
            disp_data = reshaped_t.unsqueeze(0)
            title = f"1D Vector: {shape}"
        elif len(shape) == 2:
            # 2D
            disp_data = reshaped_t
            title = f"2D Matrix: {shape}"
        else:
            # 3D: 平铺展示。将 (D1, D2, D3) -> (D1 * D2, D3) 或者简单的展平
            # 这里我们简单地把它展平显示，说明内存中它们是连续的
            disp_data = reshaped_t.flatten().unsqueeze(0)
            title = f"3D Tensor {shape} (Flattened for view)"

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(disp_data, cmap='coolwarm', aspect='auto')

        # 标注
        h, w = disp_data.shape
        for i in range(h):
            for j in range(w):
                ax.text(j, i, str(int(disp_data[i, j])), ha="center", va="center")

        ax.set_title(title)
        ax.set_yticks([])
        ax.set_xticks([])
        return fig

    mo.vstack([
        mo.md("原始数据: `[1, 2, ..., 12]` (Total 12 elements)"),
        shape_selector,
        plot_reshape(shape_selector.value)
    ])
    return


if __name__ == "__main__":
    app.run()
