---
sidebar_position: 2
---

**Python 自 3.12 起进入了一个高速演进期**：类型系统语法层重构、自由线程（去 GIL）实验、JIT 引入、模板字符串、子解释器、延迟注解求值等长期议题陆续落地。本文按版本梳理 3.12 / 3.13 / 3.14 中具有代表性的 PEP，作为后续深入学习的索引。

每个版本完整的 What's New 文档可在 [https://docs.python.org/3/whatsnew/](https://docs.python.org/3/whatsnew/) 查看。

## Python 3.12（2023-10）

3.12 的关键词是 **"类型语法升级 + 性能基础设施"**。

| PEP | 主题 | 影响 |
|-----|------|------|
| [PEP 695](https://peps.python.org/pep-0695/) | 类型参数语法 | 引入 `class Foo[T]`、`def f[T]()`、`type Alias[T] = ...` 等内联泛型语法 |
| [PEP 701](https://peps.python.org/pep-0701/) | f-string 语法纳入正式语法 | f-string 内可重复使用相同引号、可写注释、可跨行、嵌套层数无限 |
| [PEP 669](https://peps.python.org/pep-0669/) | 低开销监控 API | 为 profiler / debugger 提供 `sys.monitoring`，开销远低于旧的 `sys.settrace` |
| [PEP 684](https://peps.python.org/pep-0684/) | Per-Interpreter GIL | 每个子解释器拥有独立的 GIL，是真正多核并行的基础设施 |
| [PEP 688](https://peps.python.org/pep-0688/) | 让 buffer protocol 在 Python 层可用 | 新增 `collections.abc.Buffer`，可在类型注解中使用 |
| [PEP 709](https://peps.python.org/pep-0709/) | 内联推导式 | 列表/字典/集合推导式不再创建额外的栈帧，性能提升约 2 倍 |

## Python 3.13（2024-10）

3.13 的关键词是 **"去 GIL + JIT 实验 + 类型系统精修"**。

| PEP | 主题 | 影响 |
|-----|------|------|
| [PEP 703](https://peps.python.org/pep-0703/) | 可选去 GIL（free-threaded build） | 提供 `python3.13t` 实验性二进制，多线程可真并行 |
| [PEP 744](https://peps.python.org/pep-0744/) | JIT 编译 | 引入实验性 copy-and-patch JIT，构建时通过 `--enable-experimental-jit` 开启 |
| [PEP 667](https://peps.python.org/pep-0667/) | 命名空间一致视图 | `locals()` / frame `f_locals` 行为对齐，修复旧的诡异语义 |
| [PEP 702](https://peps.python.org/pep-0702/) | 用类型系统标记 deprecation | 引入 `typing.deprecated` 装饰器，IDE 与类型检查器可识别 |
| [PEP 742](https://peps.python.org/pep-0742/) | `TypeIs` 类型缩窄 | 比 `TypeGuard` 语义更直观，支持双向缩窄 |
| [PEP 696](https://peps.python.org/pep-0696/) | 类型参数默认值 | `class Box[T = int]`、`type Alias[T = str] = ...` |
| [PEP 705](https://peps.python.org/pep-0705/) | `TypedDict` 只读字段 | 通过 `ReadOnly[...]` 标记不可写键 |
| [PEP 730](https://peps.python.org/pep-0730/) | iOS 成为 Tier 3 平台 | 官方支持在 iOS 上分发 Python |
| [PEP 738](https://peps.python.org/pep-0738/) | Android 成为 Tier 3 平台 | 同上，Android 平台正式纳入支持矩阵 |

新增 / 改进的交互式解释器（来自 PyPy 的 PyREPL 移植）也是 3.13 的体感大变化，但它本身不是一个 PEP。

## Python 3.14（2025-10）

3.14 的关键词是 **"模板字符串 + 自由线程转正 + 注解求值改革"**。

| PEP | 主题 | 影响 |
|-----|------|------|
| [PEP 750](https://peps.python.org/pep-0750/) | 模板字符串 t-string | 新增 `t"..."` 字面量，得到的是结构化的 `Template` 对象，而非已拼接字符串，是构建安全 SQL/HTML/Shell 拼接 API 的基础 |
| [PEP 779](https://peps.python.org/pep-0779/) | 自由线程构建正式可用 | PEP 703 的 free-threaded 构建从实验状态升级为受支持的构建配置 |
| [PEP 649](https://peps.python.org/pep-0649/) | 注解的延迟求值 | 注解默认延迟到访问 `__annotations__` 时才求值，解决前向引用与循环依赖问题；取代 PEP 563 的 `from __future__ import annotations` 策略 |
| [PEP 768](https://peps.python.org/pep-0768/) | 安全的外部调试器接口 | 提供 `PyThreadState_RemoteDebug` 等 API，外部进程可安全注入断点 |
| [PEP 765](https://peps.python.org/pep-0765/) | 禁止 `finally` 中的 `return/break/continue` | 这些用法会吞掉异常，从此被语法层禁止 |
| [PEP 758](https://peps.python.org/pep-0758/) | `except`/`except*` 不再要求括号 | `except ValueError, TypeError:` 重新合法 |
| [PEP 734](https://peps.python.org/pep-0734/) | 标准库支持多解释器 | 新增 `concurrent.interpreters` 模块，配合 PEP 684 的 per-interpreter GIL 实现真并行 |
| [PEP 784](https://peps.python.org/pep-0784/) | 标准库引入 Zstandard | 新增 `compression.zstd`，对标 `gzip` / `bz2` / `lzma` |
| [PEP 741](https://peps.python.org/pep-0741/) | Python 配置 C API | 嵌入式场景可用稳定的 C API 控制初始化配置 |
| [PEP 661](https://peps.python.org/pep-0661/) | Sentinel 哨兵值 | 标准化的 `Sentinel("MISSING")`，替代 `object()` 和 `_MISSING = object()` 模式 |

## 学习路线建议

按主题串起来记会更顺：

- **类型系统线**：PEP 695 → PEP 696 → PEP 742 → PEP 705 → PEP 702
- **并发性能线**：PEP 684 → PEP 703 → PEP 744 → PEP 779 → PEP 734
- **语法糖与字符串线**：PEP 701 → PEP 750 → PEP 758 → PEP 765
- **运行期基础设施线**：PEP 669 → PEP 768 → PEP 649 → PEP 661
