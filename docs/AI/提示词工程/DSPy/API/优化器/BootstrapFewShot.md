## `dspy.BootstrapFewShot(metric=None, metric_threshold=None, teacher_settings: dict | None = None, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1, max_errors=None)` [¶](#dspy.BootstrapFewShot "永久链接")

基类: `Teleprompter`

一个 Teleprompter 类，用于组合一组演示/示例以放入预测器的提示词中。
这些演示来自训练集中的标记示例和引导（bootstrap）演示的组合。

每个引导轮次都会在 `temperature=1.0` 下使用新的 `rollout_id` 复制 LM，以绕过缓存并收集多样化的轨迹。

参数:

| 名称 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| `metric` | `Callable` | 一个比较预期值和预测值的函数，输出比较结果。 | `None` |
| `metric_threshold` | `float` | 如果指标产生数值，则在决定是否接受引导示例时将其与此阈值进行检查。默认为 None。 | `None` |
| `teacher_settings` | `dict` | `teacher` 模型的设置。默认为 None。 | `None` |
| `max_bootstrapped_demos` | `int` | 要包含的最大引导演示数。默认为 4。 | `4` |
| `max_labeled_demos` | `int` | 要包含的最大标记演示数。默认为 16。 | `16` |
| `max_rounds` | `int` | 尝试生成所需引导示例的迭代次数。如果在 `max_rounds` 后未成功，则程序结束。默认为 1。 | `1` |
| `max_errors` | `Optional[int]` | 程序结束前的最大错误数。如果为 `None`，则继承自 `dspy.settings.max_errors`。 | `None` |

源代码位于 `dspy/teleprompt/bootstrap.py`

|  |  |
| --- | --- |
| ``` 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 ``` | ``` def __init__(     self,     metric=None,     metric_threshold=None,     teacher_settings: dict | None = None,     max_bootstrapped_demos=4,     max_labeled_demos=16,     max_rounds=1,     max_errors=None, ):     """A Teleprompter class that composes a set of demos/examples to go into a predictor's prompt.     These demos come from a combination of labeled examples in the training set, and bootstrapped demos.      Each bootstrap round copies the LM with a new ``rollout_id`` at ``temperature=1.0`` to     bypass caches and gather diverse traces.      Args:         metric (Callable): A function that compares an expected value and predicted value,             outputting the result of that comparison.         metric_threshold (float, optional): If the metric yields a numerical value, then check it             against this threshold when deciding whether or not to accept a bootstrap example.             Defaults to None.         teacher_settings (dict, optional): Settings for the `teacher` model.             Defaults to None.         max_bootstrapped_demos (int): Maximum number of bootstrapped demonstrations to include.             Defaults to 4.         max_labeled_demos (int): Maximum number of labeled demonstrations to include.             Defaults to 16.         max_rounds (int): Number of iterations to attempt generating the required bootstrap             examples. If unsuccessful after `max_rounds`, the program ends. Defaults to 1.         max_errors (Optional[int]): Maximum number of errors until program ends.             If ``None``, inherits from ``dspy.settings.max_errors``.     """     self.metric = metric     self.metric_threshold = metric_threshold     self.teacher_settings = {} if teacher_settings is None else teacher_settings      self.max_bootstrapped_demos = max_bootstrapped_demos     self.max_labeled_demos = max_... [truncated]

### 方法[¶](#dspy.BootstrapFewShot-functions "永久链接")

#### `compile(student, *, teacher=None, trainset)` [¶](#dspy.BootstrapFewShot.compile "永久链接")

源代码位于 `dspy/teleprompt/bootstrap.py`

|  |  |
| --- | --- |
| ``` 81 82 83 84 85 86 87 88 89 90 91 ``` | ``` def compile(self, student, *, teacher=None, trainset):     self.trainset = trainset      self._prepare_student_and_teacher(student, teacher)     self._prepare_predictor_mappings()     self._bootstrap()      self.student = self._train()     self.student._compiled = True      return self.student  ``` |

#### `get_params() -> dict[str, Any]` [¶](#dspy.BootstrapFewShot.get_params "永久链接")

获取 Teleprompter 的参数。

返回:

| 类型 | 描述 |
| --- | --- |
| `dict[str, Any]` | Teleprompter 的参数。 |

源代码位于 `dspy/teleprompt/teleprompt.py`

|  |  |
| --- | --- |
| ``` 25 26 27 28 29 30 31 32 ``` | ``` def get_params(self) -> dict[str, Any]:     """     Get the parameters of the teleprompter.      Returns:         The parameters of the teleprompter.     """     return self.__dict__  ``` |

:::
