本教程演示了如何使用 MLflow 来跟踪和分析您的 DSPy 优化过程。MLflow 内置的 DSPy 集成提供了优化过程的可追溯性和可调试性。它允许您了解优化过程中的中间试验，存储优化后的程序及其结果，并为您的程序执行提供可观测性。

通过自动记录（autologging）功能，MLflow 能够跟踪以下信息：

* **优化器参数 (Optimizer Parameters)**

  + 少样本示例（few-shot examples）的数量
  + 候选数量
  + 其他配置设置
* **程序状态 (Program States)**

  + 初始指令和少样本示例
  + 优化后的指令和少样本示例
  + 优化过程中的中间指令和少样本示例
* **数据集 (Datasets)**

  + 使用的训练数据
  + 使用的评估数据
* **性能进展 (Performance Progression)**

  + 整体指标进展
  + 每个评估步骤的性能
* **追踪 (Traces)**

  + 程序执行追踪
  + 模型响应
  + 中间提示词 (Intermediate prompts)

## 入门指南

### 1. 安装 MLflow

首先，安装 MLflow（版本 2.21.1 或更高）：

```
pip install mlflow>=2.21.1

```

### 2. 启动 MLflow 跟踪服务器

让我们使用以下命令启动 MLflow 跟踪服务器。这将在 `http://127.0.0.1:5000/` 启动一个本地服务器：

```
# 使用 MLflow 追踪时，强烈建议使用 SQL 存储
mlflow server --backend-store-uri sqlite:///mydb.sqlite

```

### 3. 启用自动记录

配置 MLflow 以跟踪您的 DSPy 优化：

```
import mlflow
import dspy

# 启用所有自动记录功能
mlflow.dspy.autolog(
    log_compiles=True,    # 跟踪优化过程
    log_evals=True,       # 跟踪评估结果
    log_traces_from_compile=True  # 跟踪优化过程中的程序追踪
)

# 配置 MLflow 跟踪
mlflow.set_tracking_uri("http://localhost:5000")  # 使用本地 MLflow 服务器
mlflow.set_experiment("DSPy-Optimization")

```

### 4. 优化您的程序

以下是一个完整的示例，展示了如何跟踪数学问题求解器的优化过程：

```
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

# 配置语言模型
lm = dspy.LM(model="openai/gpt-4o")
dspy.configure(lm=lm)

# 加载数据集
gsm8k = GSM8K()
trainset, devset = gsm8k.train, gsm8k.dev

# 定义程序
program = dspy.ChainOfThought("question -> answer")

# 创建并运行带有跟踪功能的优化器
teleprompter = dspy.teleprompt.MIPROv2(
    metric=gsm8k_metric,
    auto="light",
)

# 优化过程将被自动跟踪
optimized_program = teleprompter.compile(
    program,
    trainset=trainset,
)

```

### 5. 查看结果

优化完成后，您可以通过 MLflow UI 分析结果。让我们来看看如何探索优化运行记录。

#### 步骤 1：访问 MLflow UI

在浏览器中访问 `http://localhost:5000` 以进入 MLflow 跟踪服务器 UI。

#### 步骤 2：理解实验结构

打开实验页面时，您将看到优化过程的层级视图。父运行（parent run）代表整体优化过程，而子运行（child runs）则显示优化过程中创建的每个程序中间版本。

#### 步骤 3：分析父运行

点击父运行可以查看优化过程的全貌。您将找到有关优化器配置参数的详细信息，以及评估指标随时间变化的情况。父运行还存储了最终优化后的程序，包括所使用的指令、签名定义和少样本示例。此外，您还可以查看优化过程中使用的训练数据。

#### 步骤 4：检查子运行

每个子运行都提供了特定优化尝试的详细快照。从实验页面选择子运行后，您可以探索该特定中间程序的多个方面。
在运行参数（run parameter）选项卡或工件（artifact）选项卡上，您可以查看该中间程序使用的指令和少样本示例。
最强大的功能之一是追踪（Traces）选项卡，它提供了程序执行的逐步视图。在这里，您可以确切了解 DSPy 程序如何处理输入并生成输出。

### 6. 加载模型进行推理

您可以直接从 MLflow 跟踪服务器加载优化后的程序进行推理：

```
model_path = mlflow.artifacts.download_artifacts("mlflow-artifacts:/path/to/best_model.json")
program.load(model_path)

```

## 故障排除

* 如果追踪信息没有显示，请确保设置了 `log_traces_from_compile=True`
* 对于大型数据集，请考虑设置 `log_traces_from_compile=False` 以避免内存问题
* 使用 `mlflow.get_run(run_id)` 以编程方式访问 MLflow 运行数据

欲了解更多功能，请探索 [MLflow 文档](https://mlflow.org/docs/latest/llms/dspy)。
