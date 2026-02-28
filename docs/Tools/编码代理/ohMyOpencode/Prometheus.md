---
sidebar_position: 3
---

## 计划书格式

`Prometheus`会在`.sisyphus\plans`目录下生成计划书,其格式如下:

```markdown
# [项目名称] — [一句话描述核心目标]

## TL;DR

> **Quick Summary**: [用 2-3 句话总结这个计划要解决什么问题，产出什么结果]
>
> **Deliverables**:
> - [核心产出物 1，例如：XX 模块代码]
> - [核心产出物 2，例如：XX 测试脚本]
>
> **Estimated Effort**: [Small / Medium / Large]
> **Parallel Execution**: [YES / NO] — [N] waves
> **Critical Path**: Task [X] → Task [Y] → Task [Z]

---

## Context

### Original Request
[原始需求描述：用户/业务方最初提出的需求是什么]

### Interview Summary
**Key Discussions**:
- [关键沟通点 1：例如，技术选型确认]
- [关键沟通点 2：例如，前置条件确认]

**Research Findings**:
- [前期调研结论 1：现有系统的状态]
- [前期调研结论 2：缺失的模块或依赖]

### Metis Review (Gap Analysis)
**Identified Gaps**:
- [查漏补缺 1：需求与现状之间的断层，及应对方案]
- [查漏补缺 2]

---

## Work Objectives

### Core Objective
[一句话定义本次开发的最核心目的]

### Concrete Deliverables
- `[文件路径/模块名]` — [具体实现说明]
- `[文件路径/模块名]` — [具体实现说明]

### Definition of Done (DoD)
- [ ] [完成标准 1，例如：Lint 检查零报错]
- [ ] [完成标准 2，例如：某主流程跑通]

### Must Have
- [必须满足的技术规范 1，例如：全异步 IO]
- [必须满足的技术规范 2]

### Must NOT Have (Guardrails)
- ❌ [绝对不能碰的红线 1，例如：禁止修改核心数据库表结构]
- ❌ [绝对不能碰的红线 2，例如：不要引入第三方依赖]

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: [YES / NO]
- **Automated tests**: [测试策略，例如：TDD / 编写端到端脚本验证]
- **Framework**: [使用的测试框架，例如：pytest]

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

---

## Execution Strategy

### Parallel Execution Waves

```text
Wave 1 (Start Immediately):
├── Task 1: [任务名] [复杂度标签]
└── Task 2: [任务名] [复杂度标签]

Wave 2 (After Wave 1):
└── Task 3: [任务名] (depends: 1, 2) [复杂度标签]

Wave FINAL:
├── Task F1: Code quality review
└── Task F2: Scope fidelity check
```




| 章节名称 (Section) | 核心作用与大白话解释 |
| :--- | :--- |
| **## TL;DR** | **执行摘要**：项目的“太长不看”版，快速列出最终要交付什么东西、预估的工作量大小，以及决定整个项目耗时的最长任务链（关键路径）。 |
| **## Context** | **项目背景**：交代“为什么要做这个”。记录了你的原始需求、前期沟通确认的细节，以及 AI 接单前对当前代码库现状的摸底调研结果和查漏补缺。 |
| **## Work Objectives** | **目标与红线**：明确具体要产出哪些代码文件，达到什么标准才算完工（Definition of Done）。最重要的是定义了绝对不能碰的红线（Must NOT Have），防止 AI 乱改现有代码。 |
| **## Verification Strategy** | **测试方针**：定下规矩，要求“零人工干预”，所有任务必须通过自动化的脚本或终端命令来验证代码写得对不对。 |
| **## Execution Strategy** | **战术排期**：这就是我们上一问聊过的依赖矩阵所在的地方。它把任务分成不同的批次（Waves），指挥哪些 AI 助手可以同时开工，最大化提升开发速度。 |
| **## TODOs** | **开发任务书**：文档的主体部分，拆解了 7 个具体的开发任务。每个任务都详细规定了要做什么、参考哪个文件、以及具体的自动化测试用例（QA Scenarios）。 |
| **## Final Verification Wave** | **终审委员会**：代码全写完后的最后一道防线。安排了 4 个不同设定的 AI “质检员”进行并发审查，包括检查是否违规、代码质量审核、端到端模拟测试和需求还原度检查。 |
| **## Commit Strategy** | **代码提交规范**：严格规定了写完代码后，Git 提交记录（Commit Message）应该怎么写，以及每次提交包含哪些文件，保持代码库整洁。 |
| **## Success Criteria** | **最终验收标准**：给出了两行终端命令和一个清单，只要命令跑通、清单打满勾，就代表整个大工程圆满结束。 |


## 其它说明

记得在`start-work`前使用以下提示词把其中的自动提交部分删了


```text
帮我修改一下 @.sisyphus\plans\simulation-pipeline.md ,把自动提交部分删了，不要自动提交代码
```

任务结束后叫它更新相关的计划书和任务列表文件

```text
任务完成你不更新 @.sisyphus\plans\simulation-pipeline.md 和 @.sisyphus\boulder.json 吗
```