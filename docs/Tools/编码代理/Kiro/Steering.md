Steering 通过 markdown 文件赋予 Kiro 关于你工作区的持久性知识。无需在每次对话中解释你的惯例，Steering 文件确保 Kiro 始终遵循你既定的模式、库和标准。

## 主要优势

**一致的代码生成** - 每个组件、API 端点或测试都遵循团队既定的模式和惯例。

**减少重复** - 无需在每次对话中解释工作区标准。Kiro 会记住你的偏好。

**团队对齐** - 所有开发者都使用相同的标准工作，无论是新加入工作区的还是资深贡献者。

**可扩展的项目知识** - 随着代码库的增长而增长的文档，捕捉项目演进过程中的决策和模式。

## Steering 文件作用域

Steering 文件可以创建为工作区作用域或全局作用域。

### 工作区 Steering

工作区 Steering 文件位于工作区根目录下的 `.kiro/steering/` 中，仅适用于该特定工作区。工作区 Steering 文件可用于告知 Kiro 适用于单个工作区的模式、库和标准。

### 全局 Steering

全局 Steering 文件位于主目录下的 `~/.kiro/steering/` 中，适用于所有工作区。全局 Steering 文件可用于告知 Kiro 适用于**所有**工作区的惯例。

如果全局和工作区 Steering 之间存在冲突指令，Kiro 将优先考虑工作区 Steering 的指令。这允许你指定通常适用于所有工作区的全局指令，同时保留为特定工作区覆盖这些指令的能力。

### 团队 Steering

全局 Steering 功能可用于定义适用于整个团队的集中式 Steering 文件。团队 Steering 文件可以通过 MDM 解决方案或组策略推送到用户的 PC，或者由用户从中央存储库下载到他们的 PC，并放置在 `~/.kiro/steering` 文件夹中。

## 基础 Steering 文件

Kiro 提供基础 Steering 文件来建立核心项目上下文。你可以按如下方式生成这些文件：

1.  导航到 Kiro 面板中的 **Steering** 部分
2.  点击 **Generate Steering Docs** 按钮，或点击 **+** 按钮并选择 **Foundation steering files** 选项
3.  Kiro 将创建三个基础文件：

**产品概述** (`product.md`) - 定义产品的用途、目标用户、关键功能和业务目标。这有助于 Kiro 理解技术决策背后的“原因”，并提出符合产品目标的解决方案。

**技术栈** (`tech.md`) - 记录你选择的框架、库、开发工具和技术约束。当 Kiro 建议实现方案时，它会优先考虑你既定的技术栈而不是其他替代方案。

**项目结构** (`structure.md`) - 概述文件组织、命名规范、导入模式和架构决策。这确保生成的代码能无缝融入你现有的代码库。

这些基础文件默认包含在每次交互中，构成了 Kiro 理解项目的基础。

## 创建自定义 Steering 文件

1.  导航到 Kiro 面板中的 **Steering** 部分
2.  点击 **+** 按钮
3.  选择 Steering 文件的一用域：工作区或全局
4.  选择一个描述性的文件名（例如，`api-standards.md`）
5.  使用标准 markdown 语法编写你的指导
6.  使用自然语言描述你的要求
7.  可选地，对于工作区 Steering 文件，你可以使用 **Refine** 按钮让 Kiro 优化你的要求

一旦创建，Steering 文件将在所有 Kiro 交互中立即生效。

## Agents.md

Kiro 支持通过 [AGENTS.md](https://agents.md/) 标准提供 Steering 指令。AGENTS.md 文件采用 markdown 格式，类似于 Kiro Steering 文件；但是，AGENTS.md 文件不支持 [包含模式](https://kiro.dev/docs/steering/#inclusion-modes) 并且总是被包含。

你可以将 AGENTS.md 文件添加到全局 Steering 文件位置 (`~/.kiro/steering/`)，或添加到工作区的根目录，Kiro 会自动获取它们。

## 包含模式

Steering 文件可以配置为根据需要及其不同时机加载。这种灵活性有助于优化性能，并确保在需要时提供相关的上下文。

通过在 Steering 文件顶部添加 front matter 来配置包含模式。Front matter 使用 YAML 语法，必须放在文件的最开头，并用三横线 (`---`) 包围。

### 始终包含 (默认)

```yaml
---
inclusion: always
---
```

这些文件会自动加载到每次 Kiro 交互中。使用此模式用于应影响所有代码生成和建议的核心标准。示例包括你的技术栈、编码规范和基本架构原则。

**最适合**：适用于全工作区的标准、技术偏好、安全策略和普遍适用的编码规范。

### 条件包含

```yaml
---
inclusion: fileMatch
fileMatchPattern: "components/**/*.tsx"
---
```

仅当处理与指定模式匹配的文件时，文件才会自动包含。这通过仅在需要时加载专门的指导，保持上下文相关性并减少干扰。

**常见模式**：

*   `"*.tsx"` - React 组件和 JSX 文件
*   `"app/api/**/*"` - API 路由和后端逻辑
*   `"**/*.test.*"` - 测试文件和测试工具
*   `"src/components/**/*"` - 组件特定的指南
*   `"*.md"` - 文档文件

**最适合**：特定领域的标准，如组件模式、API 设计规则、测试方法或仅适用于特定文件类型的部署过程。

### 手动包含

```yaml
---
inclusion: manual
---
```

通过在聊天消息中使用 `#steering-file-name` 引用它们，可以按需使用文件。这使你可以精确控制何时需要专门的上下文，而不会使每次交互都变得杂乱。

**用法**：在聊天中输入 `#troubleshooting-guide` 或 `#performance-optimization` 以在当前对话中包含该 Steering 文件。

**最适合**：专门的工作流、故障排除指南、迁移过程或仅偶尔需要的上下文繁重的文档。

## 文件引用

链接到实时工作区文件以保持 Steering 最新：

```markdown
#[[file:<relative_file_name>]]
```

示例：

*   API 规范：`#[[file:api/openapi.yaml]]`
*   组件模式：`#[[file:components/ui/button.tsx]]`
*   配置模板：`#[[file:.env.example]]`

## 最佳实践

**保持文件专注**
每个文件一个领域 - API 设计、测试或部署过程。

**使用清晰的名称**

*   `api-rest-conventions.md` - REST API 标准
*   `testing-unit-patterns.md` - 单元测试方法
*   `components-form-validation.md` - 表单组件标准

**包含上下文**
解释做出决策的原因，而不仅仅是标准是什么。

**提供示例**
使用代码片段和修改前/修改后的对比来演示标准。

**安全第一**
永远不要包含 API 密钥、密码或敏感数据。Steering 文件是你代码库的一部分。

**定期维护**

*   在 Sprint 计划和架构变更期间进行审查
*   重构后测试文件引用
*   像代码变更一样对待 Steering 变更 - 需要审查

## 常见 Steering 文件策略

**API 标准** (`api-standards.md`) - 定义 REST 惯例、错误响应格式、身份验证流程和版本控制策略。包括端点命名模式、HTTP 状态码使用和请求/响应示例。

**测试方法** (`testing-standards.md`) - 建立单元测试模式、集成测试策略、Mock 方法和覆盖率期望。记录首选的测试库、断言风格和测试文件组织。

**代码风格** (`code-conventions.md`) - 指定命名模式、文件组织、导入顺序和架构决策。包括首选代码结构、组件模式和要避免的反模式的示例。

**安全指南** (`security-policies.md`) - 记录身份验证要求、数据验证规则、输入清理标准和漏洞预防措施。包括特定于你的应用程序的安全编码实践。

**部署流程** (`deployment-workflow.md`) - 概述构建过程、环境配置、部署步骤和回滚策略。包括 CI/CD 管道细节和特定环境的要求。
