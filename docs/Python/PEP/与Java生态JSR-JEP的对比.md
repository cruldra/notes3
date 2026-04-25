---
sidebar_position: 4
---

**直觉上 PEP 和 Java 的 JSR 都是"语言演进提案"，作用方向类似，但更准确的对应关系是 PEP ≈ JEP（JDK Enhancement Proposal），而不是 JSR**。三者在治理模式、是否规范实现分离、覆盖范围上有本质差异。

## 三者一览

| 维度 | Python PEP | Java JSR | Java JEP |
|------|-----------|----------|----------|
| 全称 | Python Enhancement Proposal | Java Specification Request | JDK Enhancement Proposal |
| 主管机构 | Python Steering Council（社区选举） | JCP（Java Community Process），由 Executive Committee 表决，历史上 Sun / Oracle 主导 | OpenJDK 项目内部 |
| 定位 | **既是规范，也是实现路线图**，CPython 是事实参考实现 | **纯规范**，要求多厂商可独立实现 | OpenJDK 的工程改进提案 |
| 范围 | Python 语言 + CPython + 标准库 | 横跨 Java SE / EE（现 Jakarta EE）/ ME / 各类 API（含 JSR-303 Bean Validation 等第三方库规范） | 仅 OpenJDK / JDK 演进 |
| 配套交付 | 直接合入 CPython | 需要参考实现（RI）+ 技术兼容性套件（TCK） | 配套 JDK 代码与 JEP 文档 |
| 文档风格 | 技术随笔，含动机、备选方案、争议 | 法律/规范文档，措辞严谨 | 介于两者之间，工程导向 |
| 官方索引 | [https://peps.python.org/](https://peps.python.org/) | [https://www.jcp.org/en/jsr/all](https://www.jcp.org/en/jsr/all) | [https://openjdk.org/jeps/](https://openjdk.org/jeps/) |

## 关键差异

### 1. 是否要求"规范与实现分离"

- **JSR**：核心精神是规范独立于实现。任何符合 JSR 的厂商都可以做自己的 JVM、自己的 Servlet 容器，只要通过 TCK 即可。这是 Java"Write Once, Run Anywhere"承诺的制度保障。
- **PEP**：Python 社区只有一个权威实现 CPython。PEP 通过即合入 CPython 即可对外发布，PyPy / MicroPython / GraalPy 等其它实现选择性跟随，不存在"通过认证才能叫 Python"这一说。
- **JEP**：和 PEP 类似，OpenJDK 内部的工程演进，没有"多厂商必须独立实现"的要求。

### 2. 治理模式

- **PEP**：BDFL 时代由 Guido 一人拍板，2018 年 Guido 卸任后改为 Steering Council 5 人投票治理；流程轻量，提案者通常就是实现者。
- **JSR**：由 JCP Executive Committee（含 Oracle、IBM、SAP、Google、阿里、社区代表等）投票通过；Spec Lead 负责组织 Expert Group 起草规范，流程较重。
- **JEP**：OpenJDK 内部走 Owner → Reviewer → Approver 流程，由 Oracle 主导，HotSpot/Lambda 等 Group Lead 拍板。

### 3. 覆盖范围

- **JSR 范围最广**：从语言核心（如 JSR-335 Lambda 表达式）到企业级 API（如 JSR-345 EJB 3.2、JSR-380 Bean Validation 2.0）都走 JSR；很多大家熟悉的"Java 标准库特性"实际上是来自第三方厂商主导的 JSR，再被纳入 Java SE/EE。
- **JEP 只管 OpenJDK**：语言、JVM、JDK 工具链、GC 算法（如 JEP 333 ZGC、JEP 439 Generational ZGC）等。
- **PEP 范围介于两者之间**：包含语言、CPython 解释器、标准库、社区流程，但不延伸到第三方库（Django/FastAPI 这些没有 PEP，只有自己的 RFC/release notes）。

### 4. 现状

- **JSR 体系明显边缘化**。Java 9 之后，新的 Java SE 特性几乎都走 JEP（如 JEP 361 Switch 表达式、JEP 440 Record Patterns、JEP 444 虚拟线程）。
- **JCP/JSR 主要还活跃在 Jakarta EE（已转交给 Eclipse 基金会，使用新流程）和一些独立规范上**。
- **PEP 体系一直是 Python 演进的唯一主轨道**，没有出现过类似的并行流程更替。

## 一一对应的例子

| 主题 | Python | Java |
|------|--------|------|
| 模式匹配 | [PEP 634](https://peps.python.org/pep-0634/) Structural Pattern Matching | [JEP 441](https://openjdk.org/jeps/441) Pattern Matching for switch |
| 类型系统增强 | [PEP 695](https://peps.python.org/pep-0695/) 类型参数语法 | [JEP 14](https://openjdk.org/jeps/14)（早期）+ JSR-14 泛型 |
| 数据类 / 记录 | [PEP 557](https://peps.python.org/pep-0557/) dataclasses | [JEP 395](https://openjdk.org/jeps/395) Records |
| 轻量并发 | [PEP 703](https://peps.python.org/pep-0703/) 自由线程 | [JEP 444](https://openjdk.org/jeps/444) Virtual Threads |
| 字符串模板 | [PEP 750](https://peps.python.org/pep-0750/) t-string | [JEP 459](https://openjdk.org/jeps/459) String Templates（已撤回，重新设计中） |

## 一句话总结

> **JSR 是合同，JEP 和 PEP 是设计文档。**
>
> JSR 强调"任何人按这份合同都能做出兼容实现"；JEP/PEP 更像是"我们打算这么改 JDK/CPython，请大家审一下"。

如果你写过 Servlet、JPA 这种依赖第三方实现的代码，会更能理解 JSR 那套规范/实现/TCK 三件套的意义；而 Python 因为只有 CPython 这一棵主干，从来没有走过那条路。
