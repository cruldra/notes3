**GEPA** (Genetic-Pareto) 是在 "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning" (Agrawal et al., 2025, [arxiv:2507.19457](https://arxiv.org/abs/2507.19457)) 中提出的一种反射式优化器，它自适应地进化任意系统的*文本组件*（例如提示词）。除了指标返回的标量分数外，用户还可以向 GEPA 提供文本反馈以指导优化过程。这种文本反馈使 GEPA 更清楚地了解系统为何获得该分数，然后 GEPA 可以进行内省以确定如何提高分数。这使得 GEPA 能够在极少的轮次中提出高性能的提示词。

## `dspy.GEPA(metric: GEPAFeedbackMetric, *, auto: Literal['light', 'medium', 'heavy'] | None = None, max_full_evals: int | None = None, max_metric_calls: int | None = None, reflection_minibatch_size: int = 3, candidate_selection_strategy: Literal['pareto', 'current_best'] = 'pareto', reflection_lm: LM | None = None, skip_perfect_score: bool = True, add_format_failure_as_feedback: bool = False, instruction_proposer: ProposalFn | None = None, component_selector: ReflectionComponentSelector | str = 'round_robin', use_merge: bool = True, max_merge_invocations: int | None = 5, num_threads: int | None = None, failure_score: float = 0.0, perfect_score: float = 1.0, log_dir: str | None = None, track_stats: bool = False, use_wandb: bool = False, wandb_api_key: str | None = None, wandb_init_kwargs: dict[str, Any] | None = None, track_best_outputs: bool = False, warn_on_score_mismatch: bool = True, enable_tool_optimization: bool = False, use_mlflow: bool = False, seed: int | None = 0, gepa_kwargs: dict | None = None)` [¶](#dspy.GEPA "永久链接")

基类: `Teleprompter`

GEPA 是一个进化优化器，它使用反射来进化复杂系统的文本组件。GEPA 在论文 [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457) 中提出。
GEPA 优化引擎由 `gepa` 包提供，可从 `https://github.com/gepa-ai/gepa` 获取。

GEPA 捕获 DSPy 模块执行的完整轨迹，识别与特定预测器对应的轨迹部分，并反思预测器的行为以为预测器提出新的指令。GEPA 允许用户向优化器提供文本反馈，用于指导预测器的进化。文本反馈可以在单个预测器的粒度上提供，也可以在整个系统的执行级别上提供。

要向 GEPA 优化器提供反馈，请按如下方式实现一个指标：

```
def metric(
    gold: Example,
    pred: Prediction,
    trace: Optional[DSPyTrace] = None,
    pred_name: Optional[str] = None,
    pred_trace: Optional[DSPyTrace] = None,
) -> float | ScoreWithFeedback:
    """
    此函数使用以下参数调用：
    - gold: 黄金示例（标准答案）。
    - pred: 预测输出。
    - trace: 可选。程序执行的轨迹。
    - pred_name: 可选。GEPA 当前正在优化的目标预测器的名称，正在为其请求反馈。
    - pred_trace: 可选。GEPA 正在寻求反馈的目标预测器的执行轨迹。

    注意 `pred_name` 和 `pred_trace` 参数。在优化过程中，GEPA 将调用指标以获取正在优化的单个预测器的反馈。
    GEPA 在 `pred_name` 中提供预测器的名称，在 `pred_trace` 中提供对应于预测器的子轨迹（轨迹的一部分）。
    如果在预测器级别可用，指标应返回对应于预测器的 `{'score': float, 'feedback': str}`。
    如果在预测器级别不可用，指标也可以在程序级别返回文本反馈（仅使用 gold, pred 和 trace）。
    如果未返回反馈，GEPA 将使用仅包含分数的简单文本反馈：
    `f"This trajectory got a score of {score}."`
    """
    ...

```

GEPA 还可以用作批量推理时搜索策略，方法是传递 `valset=trainset, track_stats=True, track_best_outputs=True`，并使用优化后的程序（由 `compile` 返回）的 `detailed_results` 属性来获取批次的 Pareto 前沿。`optimized_program.detailed_results.best_outputs_valset` 将包含批次中每个任务的最佳输出。

示例:

```
gepa = GEPA(metric=metric, track_stats=True)
batch_of_tasks = [dspy.Example(...) for task in tasks]
new_prog = gepa.compile(student, trainset=trainset, valset=batch_of_tasks)
pareto_frontier = new_prog.detailed_results.val_aggregate_scores
# pareto_frontier 是一个分数列表，批次中的每个任务对应一个分数。

```

参数:

| 名称 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| `metric` | `[GEPAFeedbackMetric](#dspy.teleprompt.gepa.gepa.GEPAFeedbackMetric "            dspy.teleprompt.gepa.gepa.GEPAFeedbackMetric")` | 用于反馈和评估的指标函数。 | *必填* |
| `auto` | `Literal['light', 'medium', 'heavy'] | None` | 运行使用的自动预算。选项: "light", "medium", "heavy"。 | `None` |
| `max_full_evals` | `int | None` | 要执行的最大完整评估次数。 | `None` |
| `max_metric_calls` | `int | None` | 要执行的最大指标调用次数。 | `None` |
| `reflection_minibatch_size` | `int` | 单个 GEPA 步骤中用于反射的示例数量。默认为 3。 | `3` |
| `candidate_selection_strategy` | `Literal['pareto', 'current_best']` | 候选选择策略。默认为 "pareto"，它从所有验证分数的 Pareto 前沿随机选择候选者。选项: "pareto", "current\_best"。 | `'pareto'` |
| `reflection_lm` | `[LM](../../../models/LM/#dspy.LM "              dspy.LM(model: str, model_type: Literal['chat', 'text', 'responses'] = 'chat', temperature: float | None = None, max_tokens: int | None = None, cache: bool = True, callbacks: list[BaseCallback] | None = None, num_retries: int = 3, provider: Provider | None = None, finetuning_model: str | None = None, launch_kwargs: dict[str, Any] | None = None, train_kwargs: dict[str, Any] | None = None, use_developer_role: bool = False, **kwargs) (dspy.clients.lm.LM)") | None` | 用于反射的语言模型。必填参数。GEPA 受益于强大的反射模型。考虑使用 `dspy.LM(model='gpt-5', temperature=1.0, max_tokens=32000)` 以获得最佳性能。 | `None` |
| `skip_perfect_score` | `bool` | 是否在反射期间跳过具有完美分数的示例。默认为 True。 | `True` |
| `instruction_proposer` | `ProposalFn | None` | 可选的自定义指令提议者，实现 GEPA 的 ProposalFn 协议。**默认: None（推荐大多数用户使用）** - 使用来自 [GEPA 库](https://github.com/gepa-ai/gepa) 的经过验证的 GEPA 指令提议者，该提议者实现了 [`ProposalFn`](https://github.com/gepa-ai/gepa/blob/main/src/gepa/core/adapter.py)。此默认提议者能力很强，并在 GEPA 论文和教程中报告的各种实验中得到了验证。在此处查看有关自定义指令提议者的文档 [这里](https://dspy.ai/api/optimizers/GEPA/GEPA_Advanced/#custom-instruction-proposers)。**高级功能**: 仅在特定场景下需要：- **多模态处理**: 处理 dspy.Image 输入以及文本信息 - **对约束的细微控制**: 对指令长度、格式和结构要求进行细粒度控制，超出标准反馈机制 - **特定领域知识注入**: 无法仅通过 feedback\_func 提供的专业术语或上下文 - **特定提供商提示**: 针对特定 LLM 提供商（OpenAI, Anthropic）及其独特格式偏好的优化 - **耦合组件更新**: 协调多个组件的更新，而不是独立优化 - **外部知识集成**: 运行时访问数据库、API 或知识库。默认提议者有效地处理绝大多数用例。对于视觉内容，使用 `dspy.teleprompt.gepa.instruction_proposal` 中的 `MultiModalInstructionProposer()`，或者为高度专业化的需求实现自定义 `ProposalFn`。注意：当同时设置了 `instruction_proposer` 和 `reflection_lm` 时，`instruction_proposer` 在 `reflection_lm` 上下文中调用。但是，使用自定义 `instruction_proposer` 时，`reflection_lm` 是可选的。如果需要，自定义指令提议者可以调用自己的 LLM。 | `None` |
| `component_selector` | `ReflectionComponentSelector | str` | 实现 [ReflectionComponentSelector](https://github.com/gepa-ai/gepa/blob/main/src/gepa/proposer/reflective_mutation/base.py) 协议的自定义组件选择器，或指定内置选择器策略的字符串。控制每次迭代中选择哪些组件（预测器）进行优化。默认为 'round\_robin' 策略，该策略一次循环一个组件。可用的字符串选项：'round\_robin'（按顺序循环组件），'all'（选择所有组件进行同时优化）。自定义选择器可以实现基于优化状态和轨迹的 LLM 驱动的选择逻辑。请参阅 [gepa 组件选择器](https://github.com/gepa-ai/gepa/blob/main/src/gepa/strategies/component_selector.py) 以获取可用的内置选择器，以及 ReflectionComponentSelector 协议以实现自定义选择器。 | `'round_robin'` |
| `add_format_failure_as_feedback` | `bool` | 是否将格式失败添加为反馈。默认为 False。 | `False` |
| `use_merge` | `bool` | 是否使用基于合并的优化。默认为 True。 | `True` |
| `max_merge_invocations` | `int | None` | 要执行的最大合并调用次数。默认为 5。 | `5` |
| `num_threads` | `int | None` | 使用 `Evaluate` 进行评估时使用的线程数。可选。 | `None` |
| `failure_score` | `float` | 分配给失败示例的分数。默认为 0.0。 | `0.0` |
| `perfect_score` | `float` | 指标可达到的最高分数。默认为 1.0。GEPA 使用它来确定小批量中的所有示例是否都是完美的。 | `1.0` |
| `log_dir` | `str | None` | 保存日志的目录。GEPA 在此目录中保存详细的日志以及所有候选程序。使用相同的 `log_dir` 运行 GEPA 将从上一个检查点恢复运行。 | `None` |
| `track_stats` | `bool` | 是否在优化后程序的 `detailed_results` 属性中返回详细结果和所有提议的程序。默认为 False。 | `False` |
| `use_wandb` | `bool` | 是否使用 wandb 进行记录。默认为 False。 | `False` |
| `wandb_api_key` | `str | None` | 用于 wandb 的 API 密钥。如果未提供，wandb 将使用环境变量 `WANDB_API_KEY` 中的 API 密钥。 | `None` |
| `wandb_init_kwargs` | `dict[str, Any] | None` | 传递给 `wandb.init` 的其他关键字参数。 | `None` |
| `track_best_outputs` | `bool` | 是否在验证集上跟踪最佳输出。如果 `track_best_outputs` 为 True，则 `track_stats` 必须为 True。优化后程序的 `detailed_results.best_outputs_valset` 将包含验证集中每个任务的最佳输出。 | `False` |
| `warn_on_score_mismatch` | `bool` | GEPA（当前）期望在带有和不带有 `pred_name` 的情况下调用时，指标返回相同的模块级分数。此标志（默认为 True）确定如果检测到模块级和预测器级分数不匹配，是否引发警告。 | `True` |
| `enable_tool_optimization` | `bool` | 是否启用 dspy.ReAct 模块的联合优化。启用后，GEPA 将联合优化 dspy.ReAct 模块的预测器指令和工具描述。有关何时使用此功能及其工作原理的详细信息，请参阅 [工具优化指南](https://dspy.ai/api/optimizers/GEPA/GEPA_Advanced/#tool-optimization)。默认为 False。 | `False` |
| `seed` | `int | None` | 用于可重复性的随机种子。默认为 0。 | `0` |
| `gepa_kwargs` | `dict | None` | (可选) 直接传递给 [gepa.optimize](https://github.com/gepa-ai/gepa/blob/main/src/gepa/api.py) 的其他关键字参数。用于访问未通过 DSPy 的 GEPA 接口直接公开的高级 GEPA 功能。可用参数：- batch\_sampler: 选择训练示例的策略。可以是 [BatchSampler](https://github.com/gepa-ai/gepa/blob/main/src/gepa/strategies/batch_sampler.py) 实例或字符串 ('epoch\_shuffled')。默认为 'epoch\_shuffled'。仅在 reflection\_minibatch\_size 为 None 时有效。 - merge\_val\_overlap\_floor: 在尝试合并子采样之前，父代之间所需的共享验证 ID 的最小数量。仅在使用 'full\_eval' 以外的 `val_evaluation_policy` 时相关。默认为 5。 - stop\_callbacks: 可选的停止器，当优化应停止时返回 True。可以是单个 [StopperProtocol](https://github.com/gepa-ai/gepa/blob/main/src/gepa/utils/stop_condition.py) 或 StopperProtocol 实例列表。示例: [FileStopper](https://github.com/gepa-ai/gepa/blob/main/src/gepa/utils/stop_condition.py), [TimeoutStopCondition](https://github.com/gepa-ai/gepa/blob/main/src/gepa/utils/stop_condition.py), [SignalStopper](https://github.com/gepa-ai/gepa/blob/main/src/gepa/utils/stop_condition.py), [NoImprovementStopper](https://github.com/gepa-ai/gepa/blob/main/src/gepa/utils/stop_condition.py), 或自定义停止逻辑。注意：这会覆盖默认的 max\_metric\_calls 停止条件。 - use\_cloudpickle: 使用 cloudpickle 而不是 pickle 进行序列化。当序列化状态包含动态生成的 DSPy 签名时很有用。默认为 False。 - val\_evaluation\_policy: 控制每次迭代评分哪些验证 ID 的策略。可以是 'full\_eval'（每次评估每个 ID）或 [EvaluationPolicy](https://github.com/gepa-ai/gepa/blob/main/src/gepa/strategies/eval_policy.py) 实例。默认为 'full\_eval'。 - use\_mlf... [截断]

注意

预算配置：必须提供 `auto`, `max_full_evals`, 或 `max_metric_calls` 中的一个且仅一个。
`auto` 参数提供预设配置："light" 用于快速实验，"medium" 用于平衡优化，"heavy" 用于彻底优化。

反射配置：`reflection_lm` 参数是必需的，并且应该是一个强大的语言模型。
GEPA 在使用像 `dspy.LM(model='gpt-5', temperature=1.0, max_tokens=32000)` 这样的模型时表现最佳。
反射过程分析失败的示例以生成用于程序改进的反馈。

合并配置：GEPA 可以使用 `use_merge=True` 合并成功的程序变体。
`max_merge_invocations` 参数控制优化期间进行的合并尝试次数。

评估配置：使用 `num_threads` 并行化评估。`failure_score` 和 `perfect_score` 参数帮助 GEPA 理解你的指标范围并据此进行优化。

日志配置：设置 `log_dir` 以保存详细日志并启用检查点恢复。
使用 `track_stats=True` 通过 `detailed_results` 属性访问详细的优化结果。
启用 `use_wandb=True` 进行实验跟踪和可视化。

可重复性：设置 `seed` 以确保具有相同配置的运行结果一致。

源代码位于 `dspy/teleprompt/gepa/gepa.py`

|  |  |
| --- | --- |
| ``` 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 432 433 434 435 436 437 438 439 440 441 442 443 444 ``` | ``` def __init__(     self,     metric: GEPAFeedbackMetric,     *,     # Budget configuration     auto: Literal["light", "medium", "heavy"] | None = None,     max_full_evals: int | None = None,     max_metric_calls: int | None = None,     # Reflection configuration     reflection_minibatch_size: int = 3,     candidate_selection_strategy: Literal["pareto", "current_best"] = "pareto",     reflection_lm: LM | None = None,     skip_perfect_score: bool = True,     add_format_failure_as_feedback: bool = False,     instruction_proposer: "ProposalFn | None" = None,     component_selector: "ReflectionComponentSelector | str" = "round_robin",     # Merge-based configuration     use_merge: bool = True,     max_merge_invocations: int | None = 5,     # Evaluation configuration     num_threads: int | None = None,     failure_score: float = 0.0,     perfect_score: float = 1.0,     # Logging     log_dir: str | None = None,     track_stats: bool = False,     use_wandb: bool = False,     wandb_api_key: str | None = None,     wandb_init_kwargs: dict[str, Any] | None = None,     track_best_outputs: bool = False,     warn_on_score_mismatch: bool = True,     enable_tool_optimization: bool = False,     use_mlflow: bool = False,     # Reproducibility     seed: int | None = 0,     # GEPA passthrough kwargs     gepa_kwargs: dict | None = None, ):     try:         inspect.signature(metric).bind(None, None, None, None, None)     except TypeError as e:         raise TypeError(             "GEPA metric must accept five arguments: (gold, pred, trace, pred_name, pred_trace). "         ... [truncated]

### 方法[¶](#dspy.GEPA-functions "永久链接")

#### `auto_budget(num_preds, num_candidates, valset_size: int, minibatch_size: int = 35, full_eval_steps: int = 5) -> int` [¶](#dspy.GEPA.auto_budget "永久链接")

源代码位于 `dspy/teleprompt/gepa/gepa.py`

|  |  |
| --- | --- |
| ``` 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 540 541 542 543 544 545 546 547 548 549 550 551 552 ``` | ``` def auto_budget(     self, num_preds, num_candidates, valset_size: int, minibatch_size: int = 35, full_eval_steps: int = 5 ) -> int:     import numpy as np      num_trials = int(max(2 * (num_preds * 2) * np.log2(num_candidates), 1.5 * num_candidates))     if num_trials < 0 or valset_size < 0 or minibatch_size < 0:         raise ValueError("num_trials, valset_size, and minibatch_size must be >= 0.")     if full_eval_steps < 1:         raise ValueError("full_eval_steps must be >= 1.")      V = valset_size     N = num_trials     M = minibatch_size     m = full_eval_steps      # Initial full evaluation on the default program     total = V      # Assume upto 5 trials for bootstrapping each candidate     total += num_candidates * 5      # N minibatch evaluations     total += N * M     if N == 0:         return total  # no periodic/full evals inside the loop     # Periodic full evals occur when trial_num % (m+1) == 0, where trial_num runs 2..N+1     periodic_fulls = (N + 1) // (m) + 1     # If 1 <= N < m, the code triggers one final full eval at the end     extra_final = 1 if N < m else 0      total += (periodic_fulls + extra_final) * V     return total  ``` |

#### `compile(student: Module, *, trainset: list[Example], teacher: Module | None = None, valset: list[Example] | None = None) -> Module` [¶](#dspy.GEPA.compile "永久链接")

GEPA 使用 trainset 来执行提示词的反射性更新，但使用 valset 来跟踪 Pareto 分数。
如果未提供 valset，GEPA 将对两者都使用 trainset。

参数:
- student: 要优化的学生模块。
- trainset: 用于反射性更新的训练集。
- valset: 用于跟踪 Pareto 分数的验证集。如果未提供，GEPA 将对两者都使用 trainset。

源代码位于 `dspy/teleprompt/gepa/gepa.py`

|  |  |
| --- | --- |
| ``` 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 653 654 655 656 657 658 659 660 661 662 663 664 665 666 667 668 669 670 671 672 673 674 675 676 677 678 679 680 681 682 683 684 685 686 687 688 ``` | ``` def compile(     self,     student: Module,     *,     trainset: list[Example],     teacher: Module | None = None,     valset: list[Example] | None = None, ) -> Module:     """     GEPA uses the trainset to perform reflective updates to the prompt, but uses the valset for tracking Pareto scores.     If no valset is provided, GEPA will use the trainset for both.      Parameters:     - student: The student module to optimize.     - trainset: The training set to use for reflective updates.     - valset: The validation set to use for tracking Pareto scores. If not provided, GEPA will use the trainset for both.     """     from gepa import GEPAResult, optimize      from dspy.teleprompt.gepa.gepa_utils import DspyAdapter, LoggerAdapter      assert trainset is not None and len(trainset) > 0, "Trainset must be provided and non-empty"     assert teacher is None, "Teacher is not supported in DspyGEPA yet."      if self.auto is not None:         self.max_metric_calls = self.auto_budget(             num_preds=len(student.predictors()),             num_candidates=AUTO_RUN_SETTINGS[self.auto]["n"],             valset_size=len(valset) if valset is not None else len(trainset),         )     elif self.max_full_evals is not None:         self.max_metric_calls = self.max_full_evals * (len(trainset) + (len(valset) if valset is not None else 0))     else:         assert self.max_metric_calls is not None, "Either auto, max_full_evals, or max_... [truncated]

#### `get_params() -> dict[str, Any]` [¶](#dspy.GEPA.get_params "永久链接")

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

GEPA 背后的一个关键见解是它能够利用特定领域的文本反馈。用户应提供一个反馈函数作为 GEPA 指标，其调用签名如下：

## `dspy.teleprompt.gepa.gepa.GEPAFeedbackMetric` [¶](#dspy.teleprompt.gepa.gepa.GEPAFeedbackMetric "永久链接")

基类: `Protocol`

### 方法[¶](#dspy.teleprompt.gepa.gepa.GEPAFeedbackMetric-functions "永久链接")

#### `__call__(gold: Example, pred: Prediction, trace: Optional[DSPyTrace], pred_name: str | None, pred_trace: Optional[DSPyTrace]) -> Union[float, ScoreWithFeedback]` [¶](#dspy.teleprompt.gepa.gepa.GEPAFeedbackMetric.__call__ "永久链接")

此函数使用以下参数调用：
- gold: 黄金示例。
- pred: 预测输出。
- trace: 可选。程序执行的轨迹。
- pred\_name: 可选。GEPA 当前正在优化的目标预测器的名称，正在为其请求反馈。
- pred\_trace: 可选。GEPA 正在寻求反馈的目标预测器的执行轨迹。

注意 `pred_name` 和 `pred_trace` 参数。在优化过程中，GEPA 将调用指标以获取正在优化的单个预测器的反馈。
GEPA 在 `pred_name` 中提供预测器的名称，在 `pred_trace` 中提供对应于预测器的子轨迹（轨迹的一部分）。
如果在预测器级别可用，指标应返回对应于预测器的 dspy.Prediction(score: float, feedback: str)。
如果在预测器级别不可用，指标也可以在程序级别返回文本反馈（仅使用 gold, pred 和 trace）。
如果未返回反馈，GEPA 将使用仅包含分数的简单文本反馈：
`f"This trajectory got a score of {score}."`

源代码位于 `dspy/teleprompt/gepa/gepa.py`

|  |  |
| --- | --- |
| ``` 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 ``` | ``` def __call__(     self,     gold: Example,     pred: Prediction,     trace: Optional["DSPyTrace"],     pred_name: str | None,     pred_trace: Optional["DSPyTrace"], ) -> Union[float, "ScoreWithFeedback"]:     """     This function is called with the following arguments:     - gold: The gold example.     - pred: The predicted output.     - trace: Optional. The trace of the program's execution.     - pred_name: Optional. The name of the target predictor currently being optimized by GEPA, for which         the feedback is being requested.     - pred_trace: Optional. The trace of the target predictor's execution GEPA is seeking feedback for.      Note the `pred_name` and `pred_trace` arguments. During optimization, GEPA will call the metric to obtain     feedback for individual predictors being optimized. GEPA provides the name of the predictor in `pred_name`     and the sub-trace (of the trace) corresponding to the predictor in `pred_trace`.     If available at the predictor level, the metric should return dspy.Prediction(score: float, feedback: str)     corresponding to the predictor.     If not available at the predictor level, the metric can also return a text feedback at the program level     (using just the gold, pred and trace).     If no feedback is returned, GEPA will use a simple text feedback consisting of just the score:     f"This trajectory got a score of {score}."     """     ...  ``` |

:::

当 `track_stats=True` 时，GEPA 返回有关所有提议候选者和优化运行元数据的详细结果。结果可在 GEPA 返回的优化后程序的 `detailed_results` 属性中获得，类型如下：

## `dspy.teleprompt.gepa.gepa.DspyGEPAResult(candidates: list[Module], parents: list[list[int | None]], val_aggregate_scores: list[float], val_subscores: list[list[float]], per_val_instance_best_candidates: list[set[int]], discovery_eval_counts: list[int], best_outputs_valset: list[list[tuple[int, list[Prediction]]]] | None = None, total_metric_calls: int | None = None, num_full_val_evals: int | None = None, log_dir: str | None = None, seed: int | None = None)` `dataclass` [¶](#dspy.teleprompt.gepa.gepa.DspyGEPAResult "永久链接")

与 GEPA 运行相关的其他数据。

字段:
- candidates: 提议的候选者列表 (component\_name -> component\_text)
- parents: 谱系信息；对于每个候选者 i，parents[i] 是父索引列表或 None
- val\_aggregate\_scores: 每个候选者在验证集上的综合得分（越高越好）
- val\_subscores: 每个候选者在验证集上的每实例得分 (长度 == num\_val\_instances)
- per\_val\_instance\_best\_candidates: 对于每个验证实例 t，在 t 上获得最佳分数的候选者索引集合
- discovery\_eval\_counts: 在发现每个候选者之前消耗的预算（指标调用次数 / 轮次）

* total\_metric\_calls: 整个运行过程中进行的指标调用总数
* num\_full\_val\_evals: 执行的完整验证评估次数
* log\_dir: 写入工件的位置（如果有）
* seed: 用于可重复性的 RNG 种子（如果已知）
* best\_idx: 具有最高 val\_aggregate\_scores 的候选者索引
* best\_candidate: best\_idx 的程序文本映射

### 属性[¶](#dspy.teleprompt.gepa.gepa.DspyGEPAResult-attributes "永久链接")

#### `candidates: list[Module]` `instance-attribute` [¶](#dspy.teleprompt.gepa.gepa.DspyGEPAResult.candidates "永久链接")

#### `parents: list[list[int | None]]` `instance-attribute` [¶](#dspy.teleprompt.gepa.gepa.DspyGEPAResult.parents "永久链接")

#### `val_aggregate_scores: list[float]` `instance-attribute` [¶](#dspy.teleprompt.gepa.gepa.DspyGEPAResult.val_aggregate_scores "永久链接")

#### `val_subscores: list[list[float]]` `instance-attribute` [¶](#dspy.teleprompt.gepa.gepa.DspyGEPAResult.val_subscores "永久链接")

#### `per_val_instance_best_candidates: list[set[int]]` `instance-attribute` [¶](#dspy.teleprompt.gepa.gepa.DspyGEPAResult.per_val_instance_best_candidates "永久链接")

#### `discovery_eval_counts: list[int]` `instance-attribute` [¶](#dspy.teleprompt.gepa.gepa.DspyGEPAResult.discovery_eval_counts "永久链接")

#### `best_outputs_valset: list[list[tuple[int, list[Prediction]]]] | None = None` `class-attribute` `instance-attribute` [¶](#dspy.teleprompt.gepa.gepa.DspyGEPAResult.best_outputs_valset "永久链接")

#### `total_metric_calls: int | None = None` `class-attribute` `instance-attribute` [¶](#dspy.teleprompt.gepa.gepa.DspyGEPAResult.total_metric_calls "永久链接")

#### `num_full_val_evals: int | None = None` `class-attribute` `instance-attribute` [¶](#dspy.teleprompt.gepa.gepa.DspyGEPAResult.num_full_val_evals "永久链接")

#### `log_dir: str | None = None` `class-attribute` `instance-attribute` [¶](#dspy.teleprompt.gepa.gepa.DspyGEPAResult.log_dir "永久链接")

#### `seed: int | None = None` `class-attribute` `instance-attribute` [¶](#dspy.teleprompt.gepa.gepa.DspyGEPAResult.seed "永久链接")

#### `best_idx: int` `property` [¶](#dspy.teleprompt.gepa.gepa.DspyGEPAResult.best_idx "永久链接")

#### `best_candidate: dict[str, str]` `property` [¶](#dspy.teleprompt.gepa.gepa.DspyGEPAResult.best_candidate "永久链接")

#### `highest_score_achieved_per_val_task: list[float]` `property` [¶](#dspy.teleprompt.gepa.gepa.DspyGEPAResult.highest_score_achieved_per_val_task "永久链接")

### 方法[¶](#dspy.teleprompt.gepa.gepa.DspyGEPAResult-functions "永久链接")

#### `to_dict() -> dict[str, Any]` [¶](#dspy.teleprompt.gepa.gepa.DspyGEPAResult.to_dict "永久链接")

源代码位于 `dspy/teleprompt/gepa/gepa.py`

|  |  |
| --- | --- |
| ``` 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 ``` | ``` def to_dict(self) -> dict[str, Any]:     cands = [{k: v for k, v in cand.items()} for cand in self.candidates]      return dict(         candidates=cands,         parents=self.parents,         val_aggregate_scores=self.val_aggregate_scores,         best_outputs_valset=self.best_outputs_valset,         val_subscores=self.val_subscores,         per_val_instance_best_candidates=[list(s) for s in self.per_val_instance_best_candidates],         discovery_eval_counts=self.discovery_eval_counts,         total_metric_calls=self.total_metric_calls,         num_full_val_evals=self.num_full_val_evals,         log_dir=self.log_dir,         seed=self.seed,         best_idx=self.best_idx,     )  ``` |

#### `from_gepa_result(gepa_result: GEPAResult, adapter: DspyAdapter) -> DspyGEPAResult` `staticmethod` [¶](#dspy.teleprompt.gepa.gepa.DspyGEPAResult.from_gepa_result "永久链接")

源代码位于 `dspy/teleprompt/gepa/gepa.py`

|  |  |
| --- | --- |
| ``` 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 ``` | ``` @staticmethod def from_gepa_result(gepa_result: "GEPAResult", adapter: "DspyAdapter") -> "DspyGEPAResult":     return DspyGEPAResult(         candidates=[adapter.build_program(c) for c in gepa_result.candidates],         parents=gepa_result.parents,         val_aggregate_scores=gepa_result.val_aggregate_scores,         best_outputs_valset=gepa_result.best_outputs_valset,         val_subscores=gepa_result.val_subscores,         per_val_instance_best_candidates=gepa_result.per_val_instance_best_candidates,         discovery_eval_counts=gepa_result.discovery_eval_counts,         total_metric_calls=gepa_result.total_metric_calls,         num_full_val_evals=gepa_result.num_full_val_evals,         log_dir=gepa_result.run_dir,         seed=gepa_result.seed,     )  ``` |

:::

## 使用示例[¶](#usage-examples "永久链接")

在 [GEPA 教程](../../../../tutorials/gepa_ai_program/) 中查看 GEPA 使用教程。

### 推理时搜索[¶](#inference-time-search "永久链接")

GEPA 可以充当测试时/推理搜索机制。通过将 `valset` 设置为你的*评估批次*并使用 `track_best_outputs=True`，GEPA 为每个批次元素生成在进化搜索期间找到的得分最高的输出。

```
gepa = dspy.GEPA(metric=metric, track_stats=True, ...)
new_prog = gepa.compile(student, trainset=my_tasks, valset=my_tasks)
highest_score_achieved_per_task = new_prog.detailed_results.highest_score_achieved_per_val_task
best_outputs = new_prog.detailed_results.best_outputs_valset

```

## GEPA 如何工作？[¶](#how-does-gepa-work "永久链接")

### 1. **反射性提示词变异**[¶](#1-reflective-prompt-mutation "永久链接")

GEPA 使用 LLM *反思*结构化执行轨迹（输入、输出、失败、反馈），针对选定的模块，并使用反射元提示和收集到的反馈为目标模块提出新的指令/程序文本。

### 2. **丰富的文本反馈作为优化信号**[¶](#2-rich-textual-feedback-as-optimization-signal "永久链接")

GEPA 可以利用*任何*可用的文本反馈——不仅仅是标量奖励。这包括评估日志、代码轨迹、失败的解析、约束违反、错误消息字符串，甚至隔离的子模块特定反馈。这允许可操作的、领域感知的优化。

### 3. **基于 Pareto 的候选者选择**[¶](#3-pareto-based-candidate-selection "永久链接")

GEPA 不是仅仅进化*最佳*全局候选者（这会导致局部最优或停滞），而是维护一个 Pareto 前沿：在至少一个评估实例上获得最高分数的候选者集合。在每次迭代中，从该前沿采样（概率与覆盖率成正比）下一个要变异的候选者，保证探索和稳健地保留互补策略。

### 算法摘要[¶](#algorithm-summary "永久链接")

1. **初始化** 候选池为未优化的程序。
2. **迭代**:
3. **采样一个候选者**（从 Pareto 前沿）。
4. **采样一个小批次**（从训练集中）。
5. **收集执行轨迹 + 反馈** 用于小批次上的模块轮次。
6. **选择一个模块** 进行有针对性的改进。
7. **LLM 反射:** 使用反射元提示和收集到的反馈为目标模块提出新的指令/提示词。
8. **在小批次上推出新候选者**；**如果改进，则在 Pareto 验证集上进行评估**。
9. **更新候选池/Pareto 前沿。**
10. **[可选] 系统感知合并/交叉**: 组合来自不同谱系的表现最佳的模块。
11. **继续** 直到轮次或指标预算耗尽。
12. **返回** 在验证集上具有最佳综合性能的候选者。

## 实现反馈指标[¶](#implementing-feedback-metrics "永久链接")

设计良好的指标是 GEPA 样本效率和学习信号丰富性的核心。GEPA 期望指标返回 `dspy.Prediction(score=..., feedback=...)`。GEPA 利用基于 LLM 的工作流中的自然语言轨迹进行优化，保留纯文本形式的中间轨迹和错误，而不是将其简化为数值奖励。这反映了人类诊断过程，能够更清晰地识别系统行为和瓶颈。

GEPA 友好反馈的实用秘诀：

* **利用现有工件**: 使用日志、单元测试、评估脚本和分析器输出；展示这些通常就足够了。
* **分解结果**: 将分数分解为每个目标的组件（例如，正确性、延迟、成本、安全性）并将错误归因于步骤。
* **暴露轨迹**: 标记管道阶段，报告通过/失败以及显著错误（例如，在代码生成管道中）。
* **基于检查**: 对不可验证的任务（如在 PUPA 中）使用自动验证器（单元测试、模式、模拟器）或 LLM-as-a-judge。
* **优先考虑清晰度**: 关注错误覆盖和决策点，而不是技术复杂性。

### 示例[¶](#examples "永久链接")

* **文档检索**（例如，HotpotQA）：列出正确检索、不正确或遗漏的文档，而不仅仅是 Recall/F1 分数。
* **多目标任务**（例如，PUPA）：分解总分以揭示每个目标的贡献，突出权衡（例如，质量与隐私）。
* **堆叠管道**（例如，代码生成：解析 → 编译 → 运行 → 分析 → 评估）：暴露特定阶段的失败；自然语言轨迹通常足以进行 LLM 自我修正。

## 使用 GEPA 进行工具优化[¶](#tool-optimization-with-gepa "永久链接")

当 `enable_tool_optimization=True` 时，GEPA 将 `dspy.ReAct` 模块与工具联合优化——GEPA 根据执行轨迹和反馈一起更新预测器指令和工具描述/参数描述，而不是保持工具行为固定。

有关详细信息、示例和底层设计（工具发现、命名要求以及与自定义指令提议者的交互），请参阅 [工具优化](../GEPA_Advanced/#tool-optimization)。

## 自定义指令提议[¶](#custom-instruction-proposal "永久链接")

有关 GEPA 指令提议机制的高级自定义，包括自定义指令提议者和组件选择器，请参阅 [高级功能](../GEPA_Advanced/)。

## 延伸阅读[¶](#further-reading "永久链接")

* [GEPA 论文: arxiv:2507.19457](https://arxiv.org/abs/2507.19457)
* [GEPA Github](https://github.com/gepa-ai/gepa) - 此存储库提供了 `dspy.GEPA` 优化器使用的核心 GEPA 进化管道。
* [DSPy 教程](../../../../tutorials/gepa_ai_program/)
