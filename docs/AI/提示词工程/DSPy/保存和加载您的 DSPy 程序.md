本指南演示如何保存和加载您的 DSPy 程序。从宏观上看，有两种保存 DSPy 程序的方法：

1. 仅保存程序的状态（State），类似于 PyTorch 中的仅权重保存（weights-only saving）。
2. 保存整个程序，包括架构和状态，此功能由 `dspy>=2.6.0` 支持。

## 仅保存状态 (State-only Saving)

状态（State）代表 DSPy 程序的内部状态，包括签名（signature）、演示示例（demos，即少样本示例）以及其他信息，例如程序中每个 `dspy.Predict` 使用的 `lm`。它还包括其他 DSPy 模块的可配置属性，例如 `dspy.retrievers.Retriever` 的 `k` 值。要保存程序的状态，请使用 `save` 方法并设置 `save_program=False`。您可以选择将状态保存为 JSON 文件或 Pickle 文件。我们建议将状态保存为 JSON 文件，因为它更安全且可读。但有时您的程序包含不可序列化的对象，如 `dspy.Image` 或 `datetime.datetime`，在这种情况下，您应该将状态保存为 Pickle 文件。

假设我们已经用一些数据编译了一个程序，现在想保存该程序以供将来使用：

```python
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

gsm8k = GSM8K()
gsm8k_trainset = gsm8k.train[:10]
dspy_program = dspy.ChainOfThought("question -> answer")

optimizer = dspy.BootstrapFewShot(metric=gsm8k_metric, max_bootstrapped_demos=4, max_labeled_demos=4, max_rounds=5)
compiled_dspy_program = optimizer.compile(dspy_program, trainset=gsm8k_trainset)
```

要将程序状态保存为 json 文件：

```python
compiled_dspy_program.save("./dspy_program/program.json", save_program=False)
```

要将程序状态保存为 Pickle 文件：

:::danger[安全警告：Pickle 文件可能执行任意代码]

加载 `.pkl` 文件可能会执行任意代码，这是危险的。请仅在安全环境中从受信任的来源加载 Pickle 文件。**尽可能优先使用 `.json` 文件**。如果必须使用 Pickle 文件，请确保您信任该来源，并在加载时使用 `allow_pickle=True` 参数。

:::

```python
compiled_dspy_program.save("./dspy_program/program.pkl", save_program=False)
```

要加载已保存的状态，您需要**重新创建相同的程序**，然后使用 `load` 方法加载状态。

```python
loaded_dspy_program = dspy.ChainOfThought("question -> answer") # 重新创建相同的程序。
loaded_dspy_program.load("./dspy_program/program.json")

assert len(compiled_dspy_program.demos) == len(loaded_dspy_program.demos)
for original_demo, loaded_demo in zip(compiled_dspy_program.demos, loaded_dspy_program.demos):
    # 加载的 demo 是一个字典，而原始 demo 是 dspy.Example。
    assert original_demo.toDict() == loaded_demo
assert str(compiled_dspy_program.signature) == str(loaded_dspy_program.signature)
```

或者从 Pickle 文件加载状态：

:::danger[安全警告]

请记住在加载 Pickle 文件时使用 `allow_pickle=True`，并仅从受信任的来源加载。

:::

```python
loaded_dspy_program = dspy.ChainOfThought("question -> answer") # 重新创建相同的程序。
loaded_dspy_program.load("./dspy_program/program.pkl", allow_pickle=True)

assert len(compiled_dspy_program.demos) == len(loaded_dspy_program.demos)
for original_demo, loaded_demo in zip(compiled_dspy_program.demos, loaded_dspy_program.demos):
    # 加载的 demo 是一个字典，而原始 demo 是 dspy.Example。
    assert original_demo.toDict() == loaded_demo
assert str(compiled_dspy_program.signature) == str(loaded_dspy_program.signature)
```

## 保存整个程序 (Whole Program Saving)

:::warning[安全提示：保存整个程序使用 Pickle]

保存整个程序使用 `cloudpickle` 进行序列化，这具有与 Pickle 文件相同的安全风险。请仅在安全环境中从受信任的来源加载程序。

:::

从 `dspy>=2.6.0` 开始，DSPy 支持保存整个程序，包括架构和状态。此功能由 `cloudpickle` 支持，这是一个用于序列化和反序列化 Python 对象的库。

要保存整个程序，请使用 `save` 方法并设置 `save_program=True`，并指定一个**目录路径**来保存程序，而不是文件名。我们需要一个目录路径，因为我们还保存了一些元数据，例如依赖项版本以及程序本身。

```python
compiled_dspy_program.save("./dspy_program/", save_program=True)
```

要加载保存的程序，直接使用 `dspy.load` 方法：

```python
loaded_dspy_program = dspy.load("./dspy_program/")

assert len(compiled_dspy_program.demos) == len(loaded_dspy_program.demos)
for original_demo, loaded_demo in zip(compiled_dspy_program.demos, loaded_dspy_program.demos):
    # 加载的 demo 是一个字典，而原始 demo 是 dspy.Example。
    assert original_demo.toDict() == loaded_demo
assert str(compiled_dspy_program.signature) == str(loaded_dspy_program.signature)
```

使用保存整个程序的功能，您无需重新创建程序，而是可以直接加载架构及其状态。您可以根据需要选择合适的保存方法。

### 序列化导入的模块

当使用 `save_program=True` 保存程序时，您可能需要包含程序所依赖的自定义模块。如果您的程序依赖于这些模块，但在加载时，这些模块在调用 `dspy.load` 之前尚未导入，则这是必要的。

您可以通过在调用 `save` 时将自定义模块传递给 `modules_to_serialize` 参数，来指定应随程序一起序列化的自定义模块。这确保了您的程序所依赖的任何依赖项在序列化期间都包含在内，并在以后加载程序时可用。

在底层，这使用 cloudpickle 的 `cloudpickle.register_pickle_by_value` 函数将模块注册为按值 pickle。当以这种方式注册模块时，cloudpickle 将按值而不是按引用序列化模块，确保模块内容与保存的程序一起保留。

例如，如果您的程序使用自定义模块：

```python
import dspy
import my_custom_module

compiled_dspy_program = dspy.ChainOfThought(my_custom_module.custom_signature)

# 保存包含自定义模块的程序
compiled_dspy_program.save(
    "./dspy_program/",
    save_program=True,
    modules_to_serialize=[my_custom_module]
)
```

这确保了所需的模块被正确序列化，并在以后加载程序时可用。可以将任意数量的模块传递给 `modules_to_serialize`。如果不指定 `modules_to_serialize`，则不会注册任何额外的模块进行序列化。

## 向后兼容性

对于 `dspy<3.0.0`，我们不保证已保存程序的向后兼容性。例如，如果您使用 `dspy==2.5.35` 保存程序，请在加载时确保使用相同版本的 DSPy 来加载程序，否则程序可能无法按预期工作。在不同版本的 DSPy 中加载保存的文件可能不会引发错误，但性能可能与保存程序时不同。

从 `dspy>=3.0.0` 开始，我们将保证主要版本中已保存程序的向后兼容性，即在 `dspy==3.0.0` 中保存的程序应该可以在 `dspy==3.7.10` 中加载。
