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
    # Linear 层的作用：语义距离调整演示

    这是一个最简短的例子，证明通过训练一个简单的 **Linear 层**（线性变换），我们可以改变向量空间，使得语义相似的词（如"苹果"和"梨子"）向量更接近，而语义无关的词（如"苹果"和"猫"）距离更远。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.mermaid(
        """
        graph LR
            A[苹果] -->|Linear| T_A[T_苹果]
            B[梨子] -->|Linear| T_B[T_梨子]
            C[猫] -->|Linear| T_C[T_猫]

            T_A <-->|距离变近| T_B
            T_A <-->|距离变远| T_C
        """
    ).center()
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    return nn, optim, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. 初始随机向量

    我们首先创建三个维度的随机向量来代表：苹果、梨子、猫。
    """)
    return


@app.cell
def _(mo, torch):
    torch.manual_seed(42)
    dim = 10  # 向量维度

    # 随机初始化三个向量
    vec_apple = torch.randn(1, dim)
    vec_pear = torch.randn(1, dim)
    vec_cat = torch.randn(1, dim)

    # 打印初始状态
    def get_dist(v1, v2):
        return torch.norm(v1 - v2).item()

    dist_ap_init = get_dist(vec_apple, vec_pear)
    dist_ac_init = get_dist(vec_apple, vec_cat)

    mo.md(f"""
    **初始距离（随机状态）：**
    * 🍎 苹果 - 🍐 梨子: `{dist_ap_init:.4f}`
    * 🍎 苹果 - 🐱 猫: `{dist_ac_init:.4f}`

    *(注意：在随机初始化的情况下，它们的距离并没有特定的语义规律)*
    """)
    return dim, dist_ac_init, dist_ap_init, vec_apple, vec_cat, vec_pear


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. 定义 Linear 层和优化目标

    我们定义一个简单的线性层 `nn.Linear`，并设定训练目标：
    1. 拉近(苹果, 梨子)的距离
    2. 推远(苹果, 猫)的距离
    """)
    return


@app.cell
def _(dim, mo, nn, optim, torch, vec_apple, vec_cat, vec_pear):
    # 定义 Linear 层 (即变换矩阵 Wx + b)
    # 不改变维度，只做空间变换
    linear = nn.Linear(dim, dim, bias=False)
    optimizer = optim.SGD(linear.parameters(), lr=0.05)

    # 训练 100 步
    steps = 100
    losses = []

    # Initialize variables for static analysis
    t_apple = t_pear = t_cat = torch.empty(0)
    d_ap = d_ac = loss = torch.tensor(0.0)
    i = 0

    for i in range(steps):
        optimizer.zero_grad()

        # 通过 Linear 层变换
        t_apple = linear(vec_apple)
        t_pear = linear(vec_pear)
        t_cat = linear(vec_cat)

        # 计算变换后的距离
        d_ap = torch.norm(t_apple - t_pear)
        d_ac = torch.norm(t_apple - t_cat)

        # Loss设计:
        # 我们希望 d_ap (苹果-梨子) 变小
        # 我们希望 d_ac (苹果-猫) 变大 (即 -d_ac 变小)
        loss = d_ap - d_ac

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    mo.md(f"✅ 训练完成！经过 {steps} 步迭代，Linear 层的参数已更新。")
    return (linear,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. 结果验证
    """)
    return


@app.cell
def _(
    dist_ac_init,
    dist_ap_init,
    linear,
    mo,
    torch,
    vec_apple,
    vec_cat,
    vec_pear,
):
    with torch.no_grad():
        final_apple = linear(vec_apple)
        final_pear = linear(vec_pear)
        final_cat = linear(vec_cat)

    dist_ap_final = torch.norm(final_apple - final_pear).item()
    dist_ac_final = torch.norm(final_apple - final_cat).item()

    # 格式化输出
    table_data = [
        {
            "关系": "🍎 苹果 - 🍐 梨子 (同类)",
            "初始距离": f"{dist_ap_init:.4f}",
            "变换后距离": f"{dist_ap_final:.4f}",
            "结果": "✅ 更近了" if dist_ap_final < dist_ap_init else "❌ 失败",
        },
        {
            "关系": "🍎 苹果 - 🐱 猫 (异类)",
            "初始距离": f"{dist_ac_init:.4f}",
            "变换后距离": f"{dist_ac_final:.4f}",
            "结果": "✅ 更远了" if dist_ac_final > dist_ac_init else "❌ 失败",
        },
    ]

    mo.ui.table(table_data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 结论

    通过一个简单的 Linear 层（本质是矩阵乘法），我们将原始向量映射到了一个新的空间。

    在这个新空间中，即使输入的原始向量是完全随机的，经过"学习"后的变换矩阵也能让**语义相似**的对象（苹果和梨子）聚集在一起，同时让**语义不同**的对象（苹果和猫）分离开。

    这就是神经网络中 Linear 层、Embedding 层以及 Attention 机制中处理语义关系的核心直觉。
    """)
    return


if __name__ == "__main__":
    app.run()
