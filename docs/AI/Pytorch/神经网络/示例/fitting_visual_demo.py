import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 👔 拟合 (Fitting): 给数据“穿衣服”

    在机器学习中，**拟合**就像是给散乱的数据点（身材）量体裁衣。

    *   **欠拟合 (Underfitting)**: 衣服太小太紧 (XS号)，绷得紧紧的（死板的直线），完全体现不出原本的曲线。
    *   **过拟合 (Overfitting)**: 衣服太大太松 (XXL号)，连身上的褶皱（噪声）都给包进去了，看起来松松垮垮（扭曲的曲线）。
    *   **恰当拟合 (Good Fit)**: 量身定做，既舒适又合身。

    👇 **动手试试！** 拖动下面的滑块，看看不同的“尺码” (多项式次数) 对拟合效果的影响。
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell
def _(mo):
    # 数据生成控制组件
    data_gen_ui = mo.md(
        """
        ### 1. 制造“身材” (生成数据)

        这里我们生成一些带有随机波动（噪声）的数据点。

        {n_slider} 数据点数量
        {noise_slider} “褶皱”程度 (噪声)
        """
    ).batch(
        n_slider=mo.ui.slider(10, 50, value=20, label="点数"),
        noise_slider=mo.ui.slider(0.0, 1.5, step=0.1, value=0.3, label="噪声")
    )
    data_gen_ui
    return (data_gen_ui,)


@app.cell
def _(data_gen_ui, np):
    # 生成数据
    # 使用固定种子方便观察参数变化的影响
    np.random.seed(42)

    _N = data_gen_ui["n_slider"].value
    _Noise = data_gen_ui["noise_slider"].value

    # 真实曲线 (身材)
    X_raw = np.linspace(0, 2 * np.pi, _N)
    Y_true_curve = np.sin(X_raw)

    # 观测数据 (带褶皱/噪声)
    Y_observed_data = Y_true_curve + np.random.normal(0, _Noise, _N)
    return X_raw, Y_observed_data


@app.cell
def _(mo):
    # 模型控制组件
    degree_control = mo.ui.slider(1, 15, value=1, label="多项式次数 (Degree)")

    mo.md(
        f"""
        ### 2. 选择“尺码” (拟合模型)

        调整**多项式次数 (Degree)**，就像选择衣服的尺码。

        {degree_control}
        """
    )
    return (degree_control,)


@app.cell
def _(X_raw, Y_observed_data, degree_control, mo, np, plt):
    # 绘图逻辑
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

    _deg = degree_control.value

    # 1. 训练模型 (量体裁衣)
    # polyfit 返回多项式系数
    _coeffs = np.polyfit(X_raw, Y_observed_data, _deg)
    _model_fn = np.poly1d(_coeffs)

    # 2. 准备平滑曲线用于绘制
    _X_fine = np.linspace(0, 2 * np.pi, 200)
    _Y_pred = _model_fn(_X_fine)

    # 3. 计算误差 (衣服合身吗?)
    _Y_pred_on_points = _model_fn(X_raw)
    _mse_score = np.mean((Y_observed_data - _Y_pred_on_points) ** 2)

    # 判断拟合状态 (简单的启发式判断用于教学)
    _status_text = ""
    _status_color = "black"
    if _deg < 3:
        _status_text = "欠拟合 (Underfitting) - 衣服太紧了！"
        _status_color = "#E67C73" # Red-ish
    elif _deg > 10:
        _status_text = "过拟合 (Overfitting) - 衣服太松了，那是褶皱不是身材！"
        _status_color = "#F7CB4D" # Yellow-ish
    else:
        _status_text = "拟合良好 (Good Fit) - 看起来不错！"
        _status_color = "#57BB8A" # Green-ish

    # 4. 绘图
    _fig = plt.figure(figsize=(10, 6))
    _ax = plt.gca()

    # 画出真实点
    _ax.scatter(X_raw, Y_observed_data, color='blue', alpha=0.6, s=50, label=u'观测数据 (带噪声)')

    # 画出真实规律 (虚线)
    _ax.plot(_X_fine, np.sin(_X_fine), color='green', linestyle='--', alpha=0.5, label=u'真实规律 (真理)')

    # 画出拟合曲线
    _ax.plot(_X_fine, _Y_pred, color='red', linewidth=3, alpha=0.8, label=u'拟合模型 (Degree={})'.format(_deg))

    plt.title(u"拟合状态: {} (MSE误差: {:.4f})".format(_status_text.split('-')[0], _mse_score), fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-2, 2)

    # 显示结果
    mo.vstack([
        mo.md(f"### <span style='color:{_status_color}'>{_status_text}</span>"),
        _fig
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 🛠️ 关键 Numpy API 说明

    在这个示例中，我们使用了以下 Numpy 函数来处理数据和计算：

    *   `np.linspace(start, stop, num)`: 生成等差数列。
        *   **用途**: 创建 X 轴的坐标点。例如从 0 到 2π 生成 20 个点。
    *   `np.random.normal(loc, scale, size)`: 生成正态分布（高斯分布）的随机噪声。
        *   **用途**: 给完美的数据添加“杂质”，模拟真实的观测数据。
    *   `np.polyfit(x, y, deg)`: 多项式拟合的核心函数。
        *   **用途**: 根据数据点 $(x, y)$ 计算出最佳拟合多项式的**系数**。`deg` 参数决定了多项式的次数（比如 1 代表直线，2 代表抛物线）。
    *   `np.poly1d(coeffs)`: 一维多项式类。
        *   **用途**: 将 `polyfit` 算出的系数封装成一个**函数**对象。这样我们就可以直接用 `model(x)` 来预测 y 值，而不用自己手动写公式（如 $ax^2 + bx + c$）。
    *   `np.mean(array)`: 计算平均值。
        *   **用途**: 计算均方误差 (MSE)，即预测值和真实值之差的平方的平均值，用来衡量模型好坏。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 🧠 拟合与大模型 (LLM) 有什么关系？

    你可能会问，这个简单的多项式拟合和现在的 ChatGPT、Claude 这种大语言模型 (LLM) 有什么关系？

    **本质上，它们做的是同一件事：寻找规律。**

    1.  **本质相同**:
        *   **拟合**: 这里的代码在找一个函数 $f(x)$，使得 $y \approx f(x)$。
        *   **大模型**: LLM 也是在找一个超级复杂的函数 $P(\text{next\_token} | \text{context})$。它试图“拟合”人类语言的概率分布。

    2.  **规模差异**:
        *   **拟合**: 我们的多项式可能只有 2-10 个参数（系数）。
        *   **大模型**: 像 GPT-4 这样的模型拥有**万亿级**的参数。它们不再只是画一条简单的曲线，而是构建了一个能容纳人类所有知识的高维曲面。

    3.  **核心挑战一致**:
        *   **泛化 (Generalization)**: 我们不希望模型死记硬背（过拟合），而是希望它学会“举一反三”。
        *   **训练**: 我们调整多项式系数来减少 MSE 误差；大模型通过反向传播调整神经元权重来减少预测下一个词的误差。

    > **一句话总结**: 大模型就是一个超级巨大、超级复杂的“拟合器”，它拟合的不是简单的正弦波，而是人类的语言智慧。
    """)
    return


if __name__ == "__main__":
    app.run()
