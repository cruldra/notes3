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
    # 直观理解 Dropout：防止过拟合的利器

    Dropout 是深度学习中一种强大的正则化技术。

    **通俗的例子**：
    想象你在学习一个复杂的知识（比如做高数题）。
    - **没有 Dropout（过拟合）**：你死记硬背了每一道练习题的答案。遇到做过的题你会做，但遇到稍微变一点的题（测试集），你就懵了。
    - **有 Dropout**：学习时，你的脑子偶尔会“断片”，忘掉一部分神经元的连接。这迫使你不能依赖某一条特定的路径来解题，必须学会通过多种线索推导答案。这样学出来的知识更鲁棒，遇到新题也能举一反三。

    下面我们用一个极其简单的**回归任务**来演示这一现象。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. 制造一个容易"死记硬背"的数据集

    我们生成仅仅 20 个数据点，但是用一个很复杂的神经网络（300个隐藏神经元）去拟合它。
    """)
    return


@app.cell
def _():
    import torch
    import matplotlib.pyplot as plt

    # 设置随机种子以保证结果可复现
    torch.manual_seed(42)

    # 生成数据: y = x^2 + 噪音
    # 只有20个点，很容易过拟合
    N_SAMPLES = 20
    x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), dim=1)
    y = x.pow(2) + 0.3 * torch.rand(x.size()) # 增加一点噪音幅度

    # 测试集数据（用来画出真实的平滑曲线，作为参考）
    test_x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    return plt, test_x, torch, x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. 定义模型：过大 vs 适中（加Dropout）
    """)
    return


@app.cell
def _(torch):
    N_HIDDEN = 300 # 隐藏层神经元非常多，足以死记硬背所有点

    class Net(torch.nn.Module):
        def __init__(self, dropout_prob=0.0):
            super(Net, self).__init__()
            self.hidden = torch.nn.Linear(1, N_HIDDEN)
            self.dropout = torch.nn.Dropout(p=dropout_prob) # 核心：Dropout层
            self.predict = torch.nn.Linear(N_HIDDEN, 1)

        def forward(self, x):
            x = torch.relu(self.hidden(x))
            x = self.dropout(x) # 在激活函数后应用 dropout
            x = self.predict(x)
            return x
    return (Net,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. 训练对比

    我们要训练两个网络：
    1. **Overfit Net**: 不使用 Dropout (`p=0`)
    2. **Dropout Net**: 使用 50% 的 Dropout (`p=0.5`)

    观察它们在训练集（红点）上的拟合情况。
    """)
    return


@app.cell
def _(Net, torch, x, y):
    # 实例化两个模型
    net_overfit = Net(dropout_prob=0.0)
    net_dropout = Net(dropout_prob=0.5)

    optimizer_ofit = torch.optim.Adam(net_overfit.parameters(), lr=0.01)
    optimizer_drop = torch.optim.Adam(net_dropout.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()

    loss_ofit = None
    loss_drop = None
    pred_ofit = None
    pred_drop = None
    t = 0

    # 训练 500 步
    for t in range(500):
        # 1. Overfit Net 训练
        pred_ofit = net_overfit(x)
        loss_ofit = loss_func(pred_ofit, y)
        optimizer_ofit.zero_grad()
        loss_ofit.backward()
        optimizer_ofit.step()

        # 2. Dropout Net 训练
        pred_drop = net_dropout(x)
        loss_drop = loss_func(pred_drop, y)
        optimizer_drop.zero_grad()
        loss_drop.backward()
        optimizer_drop.step()
    return net_dropout, net_overfit


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. 结果可视化
    """)
    return


@app.cell
def _(mo, net_dropout, net_overfit, plt, test_x, x, y):
    # 预测（预测时需要把模式调为 eval，这会自动关闭 dropout）
    net_overfit.eval()
    net_dropout.eval()

    test_pred_ofit = net_overfit(test_x).data.numpy()
    test_pred_drop = net_dropout(test_x).data.numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Overfitting Plot
    ax1.scatter(x.data.numpy(), y.data.numpy(), c='red', s=50, alpha=0.5, label='train samples')
    ax1.plot(test_x.data.numpy(), test_pred_ofit, 'r-', lw=3, label='overfitting')
    ax1.set_title('Without Dropout')
    ax1.set_ylim(0, 2.0)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Dropout Plot
    ax2.scatter(x.data.numpy(), y.data.numpy(), c='red', s=50, alpha=0.5, label='train samples')
    ax2.plot(test_x.data.numpy(), test_pred_drop, 'b--', lw=3, label='dropout (p=0.5)')
    ax2.set_title('With Dropout')
    ax2.set_ylim(0, 2.0)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 使用 mo.vstack 垂直堆叠说明文本和图表
    mo.vstack([
        mo.md("左图展示了没有 Dropout 时，模型为了迎合每一个训练点，曲线可能会出现不自然的**抖动**（过拟合）。<br>右图展示了加上 Dropout 后，模型学会了更**平滑**的规律。"),
        fig
    ])
    return


if __name__ == "__main__":
    app.run()
