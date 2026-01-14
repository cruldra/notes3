import marimo

__generated_with = "0.19.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 教程：分类微调

    让我们通过一个快速示例来演示如何在 DSPy 程序中微调 LM 权重。我们将应用于一个简单的 77 类分类任务。

    我们的微调程序将使用一个微小的 `Llama-3.2-1B` 语言模型，并托管在您的本地 GPU 上。为了让这更有趣，我们假设 (i) 我们没有任何 **训练标签**，但 (ii) 我们有 500 个未标记的训练示例。

    ### 安装依赖并下载数据

    请通过 `pip install -U dspy` 安装最新的 DSPy 并跟随本教程（如果您愿意，也可以使用 `uv pip`）。本教程依赖于 DSPy >= 2.6.0。您还需要运行 `pip install datasets`。

    目前本教程需要本地 GPU 进行推理，不过我们也计划支持为微调后的模型提供 ollama 服务。

    您还需要以下依赖项：
    1. 推理：我们使用 SGLang 来运行本地推理服务器。您可以按照此处的说明安装最新版本：https://docs.sglang.ai/start/install.html
    下面分享的是截至 2025 年 2 月 4 日的最新安装命令，但我们建议您通过访问安装链接并按照说明安装最新版本。
    这可以确保微调包和 `sglang` 包保持同步。
        ```shell
        > pip install --upgrade pip
        > pip install uv
        > uv pip install "sglang[all]>=0.4.4.post3" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
        ```
    1. 微调：我们使用以下包。请注意，我们指定了 transformers 包的版本，以临时修复最近的一个问题：https://github.com/huggingface/trl/issues/2338
        ```shell
        > uv pip install -U torch transformers==4.48.3 accelerate trl peft
        ```

    我们建议使用 `uv` 包管理器来加速安装。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <details>
    <summary>推荐：设置 MLflow Tracing 以了解底层发生了什么。</summary>

    ### MLflow DSPy 集成

    <a href="https://mlflow.org/">MLflow</a> 是一个 LLMOps 工具，它与 DSPy 原生集成，并提供可解释性和实验跟踪。在本教程中，您可以使用 MLflow 将提示和优化进度可视化为追踪（traces），以便更好地理解 DSPy 的行为。您可以按照以下四个步骤轻松设置 MLflow。

    ![MLflow Trace](./mlflow-tracing-classification.png)

    1. 安装 MLflow

    ```bash
    # '%pip install mlflow>=2.20' 命令在 marimo 中自动支持
    ```

    2. 在单独的终端中启动 MLflow UI
    ```bash
    mlflow ui --port 5000
    ```

    3. 将 notebook 连接到 MLflow
    ```python
    import mlflow

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("DSPy")
    ```

    4. 启用追踪。
    ```python
    mlflow.dspy.autolog()
    ```


    要了解有关集成的更多信息，请访问 [MLflow DSPy 文档](https://mlflow.org/docs/latest/llms/dspy/index.html)。
    </details>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 数据集

    在本教程中，我们将使用 Banking77 数据集。
    """)
    return


@app.cell
def _():
    import dspy
    import random
    from dspy.datasets import DataLoader
    from datasets import load_dataset

    # Load the Banking77 dataset.
    CLASSES = (
        load_dataset("PolyAI/banking77", split="train", trust_remote_code=True)
        .features["label"]
        .names
    )
    kwargs = dict(
        fields=("text", "label"),
        input_keys=("text",),
        split="train",
        trust_remote_code=True,
    )

    # Load the first 2000 examples from the dataset, and assign a hint to each *training* example.
    raw_data = [
        dspy.Example(x, label=CLASSES[x.label]).with_inputs("text")
        for x in DataLoader().from_huggingface(
            dataset_name="PolyAI/banking77", **kwargs
        )[:1000]
    ]

    random.Random(0).shuffle(raw_data)
    return CLASSES, dspy, raw_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    该数据集有 77 个不同的分类类别。让我们查看其中一些。
    """)
    return


@app.cell
def _(CLASSES):
    len(CLASSES), CLASSES[:10]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    让我们从 Banking77 中采样 500 个（未标记）查询。我们将使用这些数据进行自举微调（bootstrapped finetuning）。
    """)
    return


@app.cell
def _(dspy, raw_data):
    unlabeled_trainset = [
        dspy.Example(text=x.text).with_inputs("text") for x in raw_data[:500]
    ]

    unlabeled_trainset[0]
    return (unlabeled_trainset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### DSPy 程序

    假设我们想要一个程序，它接收 `text`（文本），逐步推理，然后从 Banking77 中选择一个类别。

    请注意，这主要用于演示，或者用于您想要检查模型推理的情况，例如为了获得少量的可解释性。换句话说，这种类型的任务不一定能从显式推理中获益太多。
    """)
    return


@app.cell
def _(CLASSES, dspy):
    from typing import Literal

    classify = dspy.ChainOfThought(f"text -> label: Literal{CLASSES}")
    return (classify,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 自举微调 (Bootstrapped finetuning)

    有很多方法可以做到这一点，例如允许模型自学，或使用推理时计算（例如集成）来识别没有标签的高置信度案例。

    也许最简单的方法是使用一个我们期望能够胜任此任务的模型作为推理和分类的教师，并将其蒸馏到我们的小模型中。所有这些模式都可以用几行代码来表达。

    让我们设置微小的 `Llama-3.2-1B-Instruct` 作为学生 LM。我们将使用 GPT-4o-mini 作为教师 LM。
    """)
    return


@app.cell
def _(dspy):
    from dspy.clients.lm_local import LocalProvider

    student_lm_name = "meta-llama/Llama-3.2-1B-Instruct"
    student_lm = dspy.LM(
        model=f"openai/local:{student_lm_name}",
        provider=LocalProvider(),
        max_tokens=2000,
    )
    teacher_lm = dspy.LM("openai/gpt-4o-mini", max_tokens=3000)
    return student_lm, teacher_lm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    现在，让我们将分类器分配给我们的 LM。
    """)
    return


@app.cell
def _(classify, student_lm, teacher_lm):
    student_classify = classify.deepcopy()
    student_classify.set_lm(student_lm)

    teacher_classify = classify.deepcopy()
    teacher_classify.set_lm(teacher_lm)
    return student_classify, teacher_classify


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    让我们现在启动自举微调。这里的“自举”（bootstrapped）一词意味着程序本身将在训练输入上被调用，并且在所有模块上看到的生成轨迹（traces）将被记录并用于微调。这是 DSPy 中各种 BootstrapFewShot 方法的权重优化变体。

    对于（未标记）训练集中的每个问题，这将调用教师程序，它将产生推理并选择一个类别。这将被追踪，然后构成学生程序中所有模块（在本例中仅为一个 CoT 模块）的训练集。

    当调用 `compile` 方法时，`BootstrapFinetune` 优化器将使用传入的教师程序（或程序列表，您可以传递一个列表！）来创建训练数据集。
    然后，它将使用此训练数据集为 `student` 程序创建 LM 的微调版本，并将其替换为训练后的 LM。
    请注意，训练后的 LM 将是一个新的 LM 实例（我们在此处实例化的 `student_lm` 对象将保持不变！）

    注意：如果您有标签，您可以将 `metric` 传递给 `BootstrapFinetune` 的构造函数。如果您想在实践中应用这一点，您可以将 `train_kwargs` 传递给构造函数以控制本地 LM 训练设置：`device`、`use_peft`、`num_train_epochs`、`per_device_train_batch_size`、`gradient_accumulation_steps`、`learning_rate`、`max_seq_length`、`packing`、`bf16` 和 `output_dir`。
    """)
    return


@app.cell
def _():
    # Optional:
    # [1] You can set `DSPY_FINETUNEDIR` environment variable to control where the directory that will be used to store the
    #     checkpoints and fine-tuning data. If this is not set, `DSPY_CACHEDIR` is used by default.
    # [2] You can set the `CUDA_VISIBLE_DEVICES` environment variable to control the GPU that will be used for fine-tuning
    #     and inference. If this is not set and the default GPU that's used by HuggingFace's `transformers` library is
    #     occupied, an OutOfMemoryError might be raised.
    #
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["DSPY_FINETUNEDIR"] = "/path/to/dir"
    return


@app.cell
def _(dspy, student_classify, teacher_classify, unlabeled_trainset):
    dspy.settings.experimental = (
        True  # fine-tuning is an experimental feature, so we set a flag to enable it
    )
    _optimizer = dspy.BootstrapFinetune(num_threads=16)
    classify_ft = _optimizer.compile(
        student_classify, teacher=teacher_classify, trainset=unlabeled_trainset
    )  # if you *do* have labels, pass metric=your_metric here!
    return (classify_ft,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    由于这是一个本地模型，我们需要显式启动它。
    """)
    return


@app.cell
def _(classify_ft):
    classify_ft.get_lm().launch()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 验证微调后的程序

    现在让我们看看这是否成功。我们可以问系统一个问题并检查其行为。
    """)
    return


@app.cell
def _(classify_ft):
    classify_ft(
        text="I didn't receive my money earlier and it says the transaction is still in progress. Can you fix it?"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    我们也可以获取一小部分黄金标签（gold labels），看看系统是否可以泛化到未见过的查询。
    """)
    return


@app.cell
def _(raw_data):
    devset = raw_data[500:600]
    devset[0]
    return (devset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    让我们在这个小型开发集上定义一个评估器，其中的度量指标忽略推理，仅检查标签是否完全正确。
    """)
    return


@app.cell
def _(devset, dspy):
    metric = lambda x, y, trace=None: x.label == y.label
    evaluate = dspy.Evaluate(
        devset=devset,
        metric=metric,
        display_progress=True,
        display_table=5,
        num_threads=16,
    )
    return evaluate, metric


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    现在，让我们评估微调后的 1B 分类器。
    """)
    return


@app.cell
def _(classify_ft, evaluate):
    evaluate(classify_ft)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <details>
    <summary>在 MLflow 实验中跟踪评估结果</summary>

    <br/>

    要随时间推移跟踪和可视化评估结果，您可以将结果记录在 MLflow 实验中。


    ```python
    import mlflow

    with mlflow.start_run(run_name="classifier_evaluation"):
        evaluate_correctness = dspy.Evaluate(
            devset=devset,
            metric=extraction_correctness_metric,
            num_threads=16,
            display_progress=True,
        )

        # Evaluate the program as usual
        result = evaluate_correctness(people_extractor)

        # Log the aggregated score
        mlflow.log_metric("exact_match", result.score)
        # Log the detailed evaluation results as a table
        mlflow.log_table(
            {
                "Text": [example.text for example in devset],
                "Expected": [example.example_label for example in devset],
                "Predicted": [output[1] for output in result.results],
                "Exact match": [output[2] for output in result.results],
            },
            artifact_file="eval_results.json",
        )
    ```

    要了解有关集成的更多信息，请访问 [MLflow DSPy 文档](https://mlflow.org/docs/latest/llms/dspy/index.html)。

    </details>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    不错，考虑到我们开始时没有该任务的标签。即使我们没有标签，您也可以使用各种策略来提高自举训练数据的质量。

    为了接下来尝试这一点，让我们通过终止微调后的 LM 来释放 GPU 内存。
    """)
    return


@app.cell
def _(classify_ft):
    classify_ft.get_lm().kill()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 针对度量指标的自举微调

    如果您有标签，通常可以大幅提升效果。为此，您可以将 `metric` 传递给 BootstrapFinetune，它将在构建微调数据之前使用该度量指标过滤程序的轨迹。
    """)
    return


@app.cell
def _(dspy, metric, raw_data, student_classify, teacher_classify):
    _optimizer = dspy.BootstrapFinetune(num_threads=16, metric=metric)
    classify_ft_1 = _optimizer.compile(
        student_classify, teacher=teacher_classify, trainset=raw_data[:500]
    )
    return (classify_ft_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    让我们现在启动并评估它。
    """)
    return


@app.cell
def _(classify_ft_1):
    classify_ft_1.get_lm().launch()
    return


@app.cell
def _(classify_ft_1, evaluate):
    evaluate(classify_ft_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    这要好得多，考虑到只有 500 个标签。事实上，这似乎比开箱即用的教师 LM 强得多！
    """)
    return


@app.cell
def _(evaluate, teacher_classify):
    evaluate(teacher_classify)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    多亏了自举（bootstrapping），模型学会了应用我们的模块来获得正确的标签，在这种情况下，是进行显式推理：
    """)
    return


@app.cell
def _(classify_ft_1, dspy):
    classify_ft_1(text="why hasnt my card come in yet?")
    dspy.inspect_history()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <details>
    <summary>在 MLflow 实验中保存微调后的程序</summary>

    <br/>

    要在生产中部署微调后的程序或与您的团队共享，您可以将其保存在 MLflow 实验中。与简单地保存到本地文件相比，MLflow 提供以下好处：

    1. **依赖管理**：MLflow 自动将冻结的环境元数据与程序一起保存，以确保可复现性。
    2. **实验跟踪**：使用 MLflow，您可以跟踪程序的性能和成本以及程序本身。
    3. **协作**：您可以通过共享 MLflow 实验与团队成员共享程序和结果。

    要将程序保存在 MLflow 中，请运行以下代码：

    ```python
    import mlflow

    # Start an MLflow Run and save the program
    with mlflow.start_run(run_name="optimized_classifier"):
        model_info = mlflow.dspy.log_model(
            classify_ft,
            artifact_path="model", # Any name to save the program in MLflow
        )

    # Load the program back from MLflow
    loaded = mlflow.dspy.load_model(model_info.model_uri)
    ```

    要了解有关集成的更多信息，请访问 [MLflow DSPy 文档](https://mlflow.org/docs/latest/llms/dspy/index.html)。

    </details>
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
