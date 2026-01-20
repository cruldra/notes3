import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import torch
    import matplotlib.pyplot as plt
    return mo, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # PyTorch API 可视化交互指南

    本笔记本旨在通过交互式可视化帮助理解以下 PyTorch API：
    - `torch.linspace`: 创建等间距的数值序列
    - `torch.unsqueeze`: 在指定位置增加维度
    - `torch.rand`: 生成 [0, 1) 区间的均匀分布随机数
    - `torch.pow`: 计算幂次方
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. torch.linspace (线性等分)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `torch.linspace(start, end, steps)` 返回一个一维张量，包含在区间 `[start, end]` 上均匀间隔的 `steps` 个点。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    start_slider = mo.ui.slider(start=-10, stop=0, step=1, value=0, label="Start (开始值)")
    end_slider = mo.ui.slider(start=1, stop=20, step=1, value=10, label="End (结束值)")
    steps_slider = mo.ui.slider(start=3, stop=20, step=1, value=5, label="Steps (步数)")

    mo.vstack([
        mo.md("### 参数调节"),
        start_slider,
        end_slider,
        steps_slider
    ])
    return end_slider, start_slider, steps_slider


@app.cell
def _(end_slider, mo, plt, start_slider, steps_slider, torch):
    # 获取UI值
    _start = start_slider.value
    _end = end_slider.value
    _steps = steps_slider.value

    # 执行 API
    t_linspace = torch.linspace(_start, _end, steps=_steps)

    # 可视化
    fig1, ax1 = plt.subplots(figsize=(10, 2))
    ax1.plot(t_linspace.numpy(), 'o-', markersize=8, color='teal')
    ax1.set_title(f"torch.linspace({_start}, {_end}, steps={_steps})")
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Value")
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 标注数值
    for idx1, v in enumerate(t_linspace):
        ax1.text(idx1, v + 0.1, f"{v:.1f}", ha='center', va='bottom', fontsize=9)

    # 调整Y轴范围以便显示标签
    ax1.set_ylim(min(t_linspace) - 1, max(t_linspace) + 2)

    mo.vstack([
        mo.md(f"**结果张量**: `{t_linspace}`"),
        mo.as_html(fig1)
    ])
    return (t_linspace,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. torch.unsqueeze (增加维度)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `torch.unsqueeze(input, dim)` 返回一个新的张量，对 `input` 在指定位置 `dim` 插入一个大小为 1 的维度。

    这在需要广播 (broadcasting) 或者匹配模型输入形状时非常有用。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # 以前面的 linspace 结果作为输入

    dim_slider = mo.ui.slider(start=-2, stop=1, step=1, value=0, label="Dim (维度索引)")
    mo.vstack([
        mo.md("### 选择插入维度的位置"),
        dim_slider
    ])

    return (dim_slider,)


@app.cell
def _(dim_slider, mo, plt, t_linspace, torch):
    _dim = dim_slider.value

    t_unsqueezed = torch.unsqueeze(t_linspace, _dim)

    # 准备数据进行可视化
    # 确保数据是 2D 用于 imshow (unsqueeze 1D 张量结果必然是 2D)
    data_np_unsqueeze = t_unsqueezed.numpy()
    rows_u, cols_u = data_np_unsqueeze.shape

    # 动态调整画布大小
    # 基础大小 + 根据行列数调整
    fig_w = max(4, cols_u * 0.8)
    fig_h = max(2, rows_u * 0.5)

    fig_unsqueeze, ax_unsqueeze = plt.subplots(figsize=(fig_w, fig_h))

    # 绘制方格图
    im_unsqueeze = ax_unsqueeze.imshow(data_np_unsqueeze, cmap='coolwarm', aspect='equal')
    # plt.colorbar(im_unsqueeze, ax=ax_unsqueeze)

    ax_unsqueeze.set_title(f"Result Shape: {list(t_unsqueezed.shape)}\n(rows={rows_u}, cols={cols_u})", pad=20)

    # 设置坐标轴标签
    ax_unsqueeze.set_xticks(range(cols_u))
    ax_unsqueeze.set_yticks(range(rows_u))
    ax_unsqueeze.set_xlabel(f"Dimension 1 (size={cols_u})")
    ax_unsqueeze.set_ylabel(f"Dimension 0 (size={rows_u})")

    # 在格子里填数
    for i_row in range(rows_u):
        for j_col in range(cols_u):
            val_unsqueeze = data_np_unsqueeze[i_row, j_col]
            # 根据背景色深浅自动调整文字颜色
            norm_val = (val_unsqueeze - data_np_unsqueeze.min()) / (data_np_unsqueeze.max() - data_np_unsqueeze.min() + 1e-6)
            text_color = "white" if 0.2 < norm_val < 0.8 else "black"
            ax_unsqueeze.text(j_col, i_row, f"{val_unsqueeze:.1f}", ha='center', va='center', color=text_color, fontsize=9, fontweight='bold')

    info_md = mo.md(f"""
    **直观理解**:
    - **原始形状**: `{t_linspace.shape}` (1D 序列)
    - **当前操作**: `unsqueeze(dim={_dim})`

    观察下方图表形状的变化：
    - 当 `dim=0` 时，变成了 **行向量** (1 行多列)。
    - 当 `dim=1` (或 -1) 时，变成了 **列向量** (多行 1 列)。
    """)

    mo.vstack([
        info_md,
        mo.as_html(fig_unsqueeze)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. torch.rand (随机数生成)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `torch.rand(*size)` 返回一个张量，包含了从区间 `[0, 1)` 的均匀分布中抽取的随机数。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    rows_slider = mo.ui.slider(start=1, stop=8, value=4, label="Rows (行数)")
    cols_slider = mo.ui.slider(start=1, stop=8, value=4, label="Cols (列数)")

    mo.vstack([
        mo.md("### 定义形状"),
        mo.hstack([rows_slider, cols_slider], justify="start")
    ])
    return cols_slider, rows_slider


@app.cell
def _(cols_slider, mo, plt, rows_slider, torch):
    _r = rows_slider.value
    _c = cols_slider.value

    # 执行 API
    t_rand = torch.rand(_r, _c)

    # 可视化
    fig_rand, ax_rand = plt.subplots(figsize=(6, 5))
    im_rand = ax_rand.imshow(t_rand.numpy(), cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im_rand, ax=ax_rand, label="Value")
    ax_rand.set_title(f"torch.rand({_r}, {_c})")

    # 在格子里显示数值
    for i_idx in range(_r):
        for j_idx in range(_c):
            val_rand = t_rand[i_idx, j_idx].item()
            color = "white" if val_rand > 0.5 else "black"
            ax_rand.text(j_idx, i_idx, f"{val_rand:.2f}", ha="center", va="center", color=color, fontsize=10)

    ax_rand.set_xticks(range(_c))
    ax_rand.set_yticks(range(_r))

    mo.vstack([
        mo.md(f"生成了一个 `{_r}x{_c}` 的矩阵。颜色越深代表数值越大。"),
        mo.as_html(fig_rand)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. torch.pow (幂运算)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `torch.pow(input, exponent)` 对输入 `input` 的每个元素计算 `exponent` 次幂。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    exp_slider = mo.ui.slider(start=0.1, stop=5.0, step=0.1, value=2.0, label="Exponent (指数)")
    mo.vstack([
        mo.md("### 调节指数"),
        exp_slider
    ])
    return (exp_slider,)


@app.cell
def _(exp_slider, mo, plt, torch):
    _p = exp_slider.value

    # 基础数据：0 到 2 之间
    base_pow = torch.linspace(0, 2, 50)

    # 执行 API
    result_pow = torch.pow(base_pow, _p)

    fig_pow, ax_pow = plt.subplots(figsize=(8, 4))
    ax_pow.plot(base_pow.numpy(), result_pow.numpy(), 'r-', linewidth=2, label=f'y = x^{_p:.1f}')

    # 对比参照线
    ax_pow.plot(base_pow.numpy(), base_pow.numpy(), 'k--', alpha=0.3, label='y = x (线性)')
    if _p != 2:
        ax_pow.plot(base_pow.numpy(), torch.pow(base_pow, 2).numpy(), 'b--', alpha=0.3, label='y = x^2 (平方)')

    ax_pow.set_title(f"幂函数可视化: exponent={_p}")
    ax_pow.set_xlabel("Input (Base)")
    ax_pow.set_ylabel("Output (Power)")
    ax_pow.legend()
    ax_pow.grid(True, alpha=0.3)

    mo.vstack([
        mo.md(f"当前计算: $y = x^{{{_p}}}$"),
        mo.as_html(fig_pow)
    ])
    return


if __name__ == "__main__":
    app.run()
