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
    # 可视化理解：什么是“维度”？

    在深度学习和线性代数中，我们经常听到"高维空间"、"降维打击"等词汇。

    **维度 (Dimension)** 其实就是我们为了确定一个点的位置，所需要的最少**坐标数量**。

    - **0维**: 一个点。不需要坐标，它就在那里。
    - **1维**: 一条线。只需要 1 个数 ($x$) 就能定位。
    - **2维**: 一个面。需要 2 个数 ($x, y$) 就能定位。
    - **3维**: 一个体。需要 3 个数 ($x, y, z$) 就能定位。
    - **N维**: 需要 $N$ 个特征才能描述清楚的事物。

    让我们通过交互来体验一下维度的增加。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1维空间 (1D): 只有左右
    """)
    return


@app.cell
def _(mo):
    slider_1d = mo.ui.slider(start=-10, stop=10, step=1, value=0, label="X 坐标")
    return (slider_1d,)


@app.cell
def _(mo, plt, slider_1d):
    # 1D 绘图
    def plot_1d(x_val):
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.plot([-10, 10], [0, 0], 'k-', alpha=0.3) # 轨道
        ax.scatter([x_val], [0], c='red', s=100, zorder=5) # 点
        ax.set_xlim(-11, 11)
        ax.set_ylim(-1, 1)
        ax.set_yticks([]) # 隐藏 Y 轴，因为只有 1 维
        ax.set_title(f"1D Point: x = {x_val}")
        ax.grid(True, axis='x', linestyle='--')
        return fig

    mo.vstack([
        slider_1d,
        plot_1d(slider_1d.value)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2维空间 (2D): 加上前后
    """)
    return


@app.cell
def _(mo):
    slider_2d_x = mo.ui.slider(start=-10, stop=10, step=1, value=0, label="X 坐标 (左右)")
    slider_2d_y = mo.ui.slider(start=-10, stop=10, step=1, value=0, label="Y 坐标 (前后)")
    return slider_2d_x, slider_2d_y


@app.cell
def _(mo, plt, slider_2d_x, slider_2d_y):
    # 2D 绘图
    def plot_2d(x_val, y_val):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter([x_val], [y_val], c='blue', s=150, zorder=5)
        ax.set_xlim(-11, 11)
        ax.set_ylim(-11, 11)
        ax.axhline(0, color='black', alpha=0.3)
        ax.axvline(0, color='black', alpha=0.3)
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_title(f"2D Point: ({x_val}, {y_val})")
        ax.grid(True, linestyle='--')
        return fig

    mo.hstack([
        mo.vstack([slider_2d_x, slider_2d_y]),
        plot_2d(slider_2d_x.value, slider_2d_y.value)
    ], justify="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3维空间 (3D): 加上上下
    """)
    return


@app.cell
def _(mo):
    slider_3d_x = mo.ui.slider(start=-10, stop=10, step=1, value=0, label="X")
    slider_3d_y = mo.ui.slider(start=-10, stop=10, step=1, value=0, label="Y")
    slider_3d_z = mo.ui.slider(start=-10, stop=10, step=1, value=0, label="Z")
    return slider_3d_x, slider_3d_y, slider_3d_z


@app.cell
def _(mo, plt, slider_3d_x, slider_3d_y, slider_3d_z):
    # 3D 绘图
    def plot_3d(x, y, z):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 画点
        ax.scatter([x], [y], [z], c='green', s=200)

        # 画投影线帮助定位
        ax.plot([x, x], [y, y], [-10, z], 'k--', alpha=0.3)
        ax.plot([x, x], [-10, y], [z, z], 'k--', alpha=0.3)
        ax.plot([-10, x], [y, y], [z, z], 'k--', alpha=0.3)

        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-10, 10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"3D Point: ({x}, {y}, {z})")
        return fig

    mo.hstack([
        mo.vstack([slider_3d_x, slider_3d_y, slider_3d_z]),
        plot_3d(slider_3d_x.value, slider_3d_y.value, slider_3d_z.value)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. 高维空间：数据的视角

    超过 3 维，我们的眼睛就很难画出来了。但是对于计算机来说，**维度只是列表的长度**。

    比如描述一个人：
    `[身高, 体重, 年龄, 收入, 血压]` -> 这就是一个 **5维** 向量。

    在 PyTorch 中，这只是一个形状为 `(5,)` 的张量。
    """)
    return


@app.cell
def _(mo):
    text_input = mo.ui.text(
        value="175, 70, 25, 5000", 
        label="输入特征 (逗号分隔):"
    )
    return (text_input,)


@app.cell
def _(mo, text_input):
    import torch

    def explain_tensor_dim(features):
        feature_list = [float(x) for x in features.split(',')]
        dim = len(feature_list)
        tensor = torch.tensor(feature_list)

        return mo.md(
            f"""
            **输入特征**: {feature_list}

            **维度 (Dimension)**: {dim}

            **PyTorch Tensor**: 
            ```python
            {tensor}
            # Shape: {tensor.shape}
            ```
            """
        )

    mo.vstack([
        text_input,
        explain_tensor_dim(text_input.value)
    ])
    return


if __name__ == "__main__":
    app.run()
