## `dspy.BetterTogether(metric: Callable, prompt_optimizer: Teleprompter | None = None, weight_optimizer: Teleprompter | None = None, seed: int | None = None)` [¶](#dspy.BetterTogether "Permanent link")

Bases: `Teleprompter`

Source code in `dspy/teleprompt/bettertogether.py`

|  |  |
| --- | --- |
| ``` 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 ``` | ``` def __init__(self,     metric: Callable,     prompt_optimizer: Teleprompter | None = None,     weight_optimizer: Teleprompter | None = None,     seed: int | None = None,   ):     if not dspy.settings.experimental:         raise ValueError("This is an experimental optimizer. Set `dspy.settings.experimental` to `True` to use it.")      # TODO: Note that the BetterTogether optimizer is meaningful when     # BootstrapFinetune uses a metric to filter the training data before     # fine-tuning. However, one can also choose to run this optimizer with     # a BootstrapFinetune without a metric, say, if there aren't labels     # available for the training data. Should this be noted somewhere?     # TODO: We should re-consider if the metric should be required.     self.prompt_optimizer = prompt_optimizer if prompt_optimizer else BootstrapFewShotWithRandomSearch(metric=metric)     self.weight_optimizer = weight_optimizer if weight_optimizer else BootstrapFinetune(metric=metric)      is_supported_prompt = isinstance(self.prompt_optimizer, BootstrapFewShotWithRandomSearch)     is_supported_weight = isinstance(self.weight_optimizer, BootstrapFinetune)     if not is_supported_prompt or not is_supported_weight:         raise ValueError(             "The BetterTogether optimizer only supports the following optimizers for now: BootstrapFinetune, "             "BootstrapFewShotWithRandomSearch."         )      self.rng = random.Random(seed)  ``` |

### Functions[¶](#dspy.BetterTogether-functions "Permanent link")

#### `compile(student: Module, trainset: list[Example], strategy: str = 'p -> w -> p', valset_ratio=0.1) -> Module` [¶](#dspy.BetterTogether.compile "Permanent link")

Source code in `dspy/teleprompt/bettertogether.py`

|  |  |
| --- | --- |
| ``` 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 ``` | ``` def compile(     self,     student: Module,     trainset: list[Example],     strategy: str = "p -> w -> p",     valset_ratio = 0.1, ) -> Module:     # TODO: We could record acc on a different valset to pick the best     # strategy within the provided strategy     logger.info("Validating the strategy")     parsed_strategy = strategy.lower().split(self.STRAT_SEP)      if not all(s in ["p", "w"] for s in parsed_strategy):         raise ValueError(             f"The strategy should be a sequence of 'p' and 'w' separated by '{self.STRAT_SEP}', but "             f"found: {strategy}"         )      logger.info("Preparing the student program...")     # TODO: Prepare student returns student.reset_copy(), which is what gets     # optimized. We should make this clear in the doc comments.     student = prepare_student(student)     all_predictors_have_lms(student)      # Make a shallow copy of the trainset, so that we don't change the order     # of the examples in the original trainset     trainset = trainset[:]     logger.info("Compiling the student program...")     student = self._run_strategies(parsed_strategy, student, trainset, valset_ratio)      logger.info("BetterTogether has finished compiling the student program")     return student  ``` |

#### `get_params() -> dict[str, Any]` [¶](#dspy.BetterTogether.get_params "Permanent link")

Get the parameters of the teleprompter.

Returns:

| Type | Description |
| --- | --- |
| `dict[str, Any]` | The parameters of the teleprompter. |

Source code in `dspy/teleprompt/teleprompt.py`

|  |  |
| --- | --- |
| ``` 25 26 27 28 29 30 31 32 ``` | ``` def get_params(self) -> dict[str, Any]:     """     Get the parameters of the teleprompter.      Returns:         The parameters of the teleprompter.     """     return self.__dict__  ``` |