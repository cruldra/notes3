---
title: 什么是 Skills？
description: Agent Skills 是一种轻量级的开放格式，用于通过专业知识和工作流程扩展 AI 智能体的能力
---

# 什么是 Skills？

> 原文：[What are skills?](https://agentskills.io/what-are-skills)

本质上，skill 是一个包含 `SKILL.md` 文件的文件夹。该文件包含元数据（至少包括 `name` 和 `description`）和指令，告诉智能体如何执行特定任务。Skills 还可以捆绑脚本、模板和参考资料。

```
my-skill/
├── SKILL.md          # 必需：指令 + 元数据
├── scripts/          # 可选：可执行代码
├── references/       # 可选：文档
└── assets/           # 可选：模板、资源
```

## Skills 如何工作

Skills 使用**渐进式披露**来高效管理上下文：

1. **发现**：启动时，智能体只加载每个可用 skill 的名称和描述，刚好足够知道它何时可能相关。
2. **激活**：当任务与 skill 的描述匹配时，智能体将完整的 `SKILL.md` 指令读入上下文。
3. **执行**：智能体遵循指令，根据需要可选地加载引用的文件或执行捆绑的代码。

这种方法使智能体保持快速，同时按需访问更多上下文。

## SKILL.md 文件

每个 skill 都以包含 YAML frontmatter 和 Markdown 指令的 `SKILL.md` 文件开始：

```markdown
---
name: pdf-processing
description: 从 PDF 文件中提取文本和表格，填写表单，合并文档。
---

# PDF 处理

## 何时使用此 skill
当用户需要处理 PDF 文件时使用此 skill...

## 如何提取文本
1. 使用 pdfplumber 进行文本提取...

## 如何填写表单
...
```

### 必需的 frontmatter

`SKILL.md` 顶部需要以下 frontmatter：

- `name`：简短标识符
- `description`：何时使用此 skill

### Markdown 内容

Markdown 正文包含实际指令，对结构或内容没有特定限制。这种简单格式有一些关键优势：

- **自文档化**：skill 作者或用户可以阅读 `SKILL.md` 并了解它的作用，使 skills 易于审计和改进。
- **可扩展**：Skills 的复杂度可以从纯文本指令到可执行代码、资源和模板。
- **可移植**：Skills 只是文件，因此易于编辑、版本控制和共享。

## 下一步

- [查看规范](https://agentskills.io/specification) - 了解完整格式
- [为您的智能体添加 skills 支持](https://agentskills.io/integrate-skills) - 构建兼容客户端
- [查看示例 skills](https://github.com/anthropics/skills) - GitHub 上的示例
- [阅读编写最佳实践](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices) - 编写有效 skills 的指南
- [使用参考库](https://github.com/agentskills/agentskills/tree/main/skills-ref) - 验证 skills 并生成提示 XML
