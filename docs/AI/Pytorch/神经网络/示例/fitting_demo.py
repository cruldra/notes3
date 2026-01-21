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
    # 什么是"拟合" (Fitting)?

    在机器学习和统计学中，**拟合**是指构建一个数学模型（函数），使其尽可能好地描述一组观测数据。

    想象一下，你有一堆散乱的点，你想画一条线穿过它们，尽可能让这条线"代表"这些点的趋势。这就是拟合。

    *   **欠拟合 (Underfitting)**: 模型太简单，抓不住数据的规律（比如用直线去拟合曲线）。
    *   **过拟合 (Overfitting)**: 模型太复杂，把噪声也当成了规律（比如连线连得乱七八糟，甚至经过了每一个错误点）。
    *   **恰当拟合**: 找到了数据背后的真实规律。

    👇 下面是一个交互式演示。
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. 生成数据
    """)
    return


@app.cell
def _(mo):
    # 数据生成控制
    data_controls = mo.md(
        """
        **调整数据参数:**

        {n_points_slider} 数据点数量 (N)

        {noise_slider} 噪声强度 (Noise)
        """
    ).batch(
        n_points_slider=mo.ui.slider(10, 100, step=5, value=30, label="数据点数量"),
        noise_slider=mo.ui.slider(0.0, 1.0, step=0.05, value=0.2, label="噪声强度"),
    )
    data_controls
    return (data_controls,)


@app.cell
def _(data_controls, np):
    # 生成带噪声的正弦波数据
    # 使用唯一的随机种子以保证结果可复现，但允许用户观察变化
    np.random.seed(42)

    _n_points = data_controls["n_points_slider"].value
    _noise_level = data_controls["noise_slider"].value

    X_data = np.linspace(0, 2 * np.pi, _n_points)
    # 真实函数: sin(x)
    Y_true = np.sin(X_data)
    # 观测数据: sin(x) + 噪声
    Y_noise = np.random.normal(0, _noise_level, _n_points)
    Y_data = Y_true + Y_noise
    return X_data, Y_data


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. 拟合模型
    """)
    return


@app.cell
def _(mo):
    degree_slider = mo.ui.slider(1, 15, value=1, label="多项式次数 (Degree)")

    mo.md(
        f"""
        我们要用一个**多项式**来拟合上面的数据。

        试着拖动滑块，改变多项式的**次数 (Degree)**：

        {degree_slider}

        *   **Degree = 1**: 直线 (容易欠拟合)
        *   **Degree = 3~5**: 曲线 (可能比较合适)
        *   **Degree > 10**: 非常扭曲的线 (容易过拟合)
        """
    )
    return (degree_slider,)


@app.cell
def _(X_data, Y_data, degree_slider, np, plt):
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
    # 获取当前的degree
    _degree = degree_slider.value

    # 使用numpy进行多项式拟合
    # coefficients 是多项式的系数
    _coefficients = np.polyfit(X_data, Y_data, _degree)
    _polynomial_fn = np.poly1d(_coefficients)

    # 生成平滑的曲线用于绘图
    X_plot = np.linspace(0, 2 * np.pi, 200)
    Y_pred = _polynomial_fn(X_plot)

    # 计算均方误差 (MSE)
    Y_pred_on_data = _polynomial_fn(X_data)
    _mse = np.mean((Y_data - Y_pred_on_data) ** 2)

    # 绘图
    plt.figure(figsize=(10, 6))

    # 1. 画出带噪声的观测点
    plt.scatter(X_data, Y_data, color='blue', alpha=0.6, label=u'观测数据 (带噪声)')

    # 2. 画出真实的生成函数 (虚线)
    plt.plot(X_plot, np.sin(X_plot), color='green', linestyle='--', alpha=0.5, label=u'真实规律 (True Function)')

    # 3. 画出我们的拟合曲线 (红色)
    plt.plot(X_plot, Y_pred, color='red', linewidth=2, label=u'拟合模型 (Degree={})'.format(_degree))

    plt.title(u"多项式拟合演示 (MSE: {:.4f})".format(_mse))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-2, 2)

    # 返回当前的图表对象
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. 流程可视化
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # 使用 mermaid 展示拟合的思维导图
    diagram = mo.mermaid(
        """
        graph TB
            A[真实世界数据] -->|包含| B(规律 Signal)
            A -->|包含| C(噪声 Noise)
            B & C --> D[观测数据 X, Y]
            D --> E{拟合过程}
            F[模型 Model] --> E
            E -->|计算误差| G[Loss Function]
            G -->|最小化误差| H[更新参数]
            H --> F
        """
    ).center()

    diagram
    return


if __name__ == "__main__":
    app.run()
