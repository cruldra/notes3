import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PyTorch nn.Linear 详解与演示

    `torch.nn.Linear` 是 PyTorch 中实现线性变换（全连接层）的模块。
    它对输入数据 $x$ 应用线性变换：

    $$ y = xA^T + b $$

    其中：
    - $x$ 是输入
    - $A$ 是权重矩阵 (weight)，形状为 `(out_features, in_features)`
    - $b$ 是偏置向量 (bias)，形状为 `(out_features)`
    """)
    return


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    return mo, nn, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 基本定义与参数
    """)
    return


@app.cell
def _(mo, nn):
    # 定义输入特征维度和输出特征维度
    in_features = 4
    out_features = 3

    # 创建 Linear 层
    linear_layer = nn.Linear(in_features, out_features)

    mo.md(
        f"""
        创建了一个 `nn.Linear` 层:
        - 输入维度: `{in_features}`
        - 输出维度: `{out_features}`

        权重矩阵形状 `weight.shape`: `{linear_layer.weight.shape}` (注意是 `[out_features, in_features]`)
        偏置向量形状 `bias.shape`: `{linear_layer.bias.shape}`
        """
    )
    return (linear_layer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 前向传播 (Forward Pass)
    """)
    return


@app.cell
def _(linear_layer, mo, torch):
    # 创建一个随机输入张量
    # 形状通常是 (batch_size, in_features)
    batch_size = 2
    input_tensor = torch.randn(batch_size, linear_layer.in_features)

    # 前向传播
    output_tensor = linear_layer(input_tensor)

    mo.md(
        f"""
        输入张量形状: `{input_tensor.shape}`
        输出张量形状: `{output_tensor.shape}`

        **输出结果**:
        ```python
        {output_tensor}
        ```
        """
    )
    return input_tensor, output_tensor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 手动计算验证
    """)
    return


@app.cell
def _(input_tensor, linear_layer, mo, output_tensor, torch):
    # 手动实现 y = x @ W.T + b
    # 注意: nn.Linear 存储的 weight 是 [out_features, in_features]，所以计算时需要转置
    manual_output = (
        torch.matmul(input_tensor, linear_layer.weight.t()) + linear_layer.bias
    )

    # 检查是否接近
    is_close = torch.allclose(output_tensor, manual_output, atol=1e-6)

    mo.md(
        f"""
        手动计算公式: `input @ weight.T + bias`

        PyTorch 输出与手动计算是否一致: **{is_close}**

        差异值: `{torch.abs(output_tensor - manual_output).max().item()}`
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. 交互式可视化
    """)
    return


@app.cell
def _(mo):
    features_slider = mo.ui.slider(
        start=1, stop=20, value=5, label="Input Features (in)"
    )
    output_slider = mo.ui.slider(
        start=1, stop=20, value=8, label="Output Features (out)"
    )

    mo.md(
        f"""
        调整下面的滑块来改变线性层的维度，观察权重矩阵的形状变化。

        {features_slider}
        {output_slider}
        """
    )
    return features_slider, output_slider


@app.cell
def _(features_slider, mo, nn, output_slider, plt):
    # 获取滑块的值
    d_in = features_slider.value
    d_out = output_slider.value

    # 动态创建层
    layer = nn.Linear(d_in, d_out)

    # 简单的可视化
    fig, ax = plt.subplots(figsize=(6, 4))

    # 绘制权重矩阵的热图
    # weight 是 [out, in]，所以 imshow 直接画正好是 y=out, x=in
    weight_data = layer.weight.detach().numpy()

    im = ax.imshow(weight_data, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="Weight Value")

    ax.set_title(
        f"Weight Matrix Shape: {weight_data.shape}\n(Rows=Output Features, Cols=Input Features)"
    )
    ax.set_xlabel(f"Input Features ({d_in})")
    ax.set_ylabel(f"Output Features ({d_out})")

    # 确保布局紧凑
    plt.tight_layout()

    mo.vstack([mo.md(f"### 权重矩阵可视化 ({d_out} x {d_in})"), mo.as_html(fig)])
    return


if __name__ == "__main__":
    app.run()
