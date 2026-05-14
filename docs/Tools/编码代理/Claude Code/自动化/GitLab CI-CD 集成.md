---
sidebar_position: 5
---

# Claude Code GitLab CI/CD 集成

> 了解如何将 Claude Code 通过 GitLab CI/CD 集成到你的开发工作流中

> **注意**：Claude Code for GitLab CI/CD 目前处于 Beta 阶段。功能和特性可能会随着我们完善体验而演变。此集成由 GitLab 维护。如需支持，请参见 [GitLab Issue](https://gitlab.com/gitlab-org/gitlab/-/issues/573776)。

> **注意**：此集成构建在 [Claude Code CLI 和 Agent SDK](/en/agent-sdk/overview) 之上，支持在 CI/CD 作业和自定义自动化工作流中以编程方式使用 Claude。

## 为什么在 GitLab 中使用 Claude Code？

* **即时创建 MR**：描述你的需求，Claude 会提交一个包含更改和说明的完整 MR
* **自动实现**：通过一条命令或一次提及将 Issue 转化为可用代码
* **理解项目**：Claude 遵循你的 `CLAUDE.md` 指南和现有代码模式
* **简单配置**：只需在 `.gitlab-ci.yml` 中添加一个作业并设置一个 masked CI/CD 变量
* **企业就绪**：可选择 Claude API、Amazon Bedrock 或 Google Vertex AI，满足数据驻留和采购需求
* **默认安全**：在你的 GitLab Runner 中运行，遵循你的分支保护和审批规则

## 工作原理

Claude Code 使用 GitLab CI/CD 在隔离的作业中运行 AI 任务，并通过 MR 提交结果：

1. **事件驱动的编排**：GitLab 监听你选择的触发器（例如，在 Issue、MR 或评审讨论中提到 `@claude` 的评论）。作业从讨论和仓库中收集上下文，根据输入构建提示，然后运行 Claude Code。

2. **Provider 抽象层**：使用适合你环境的 Provider：
   * Claude API（SaaS）
   * Amazon Bedrock（基于 IAM 的访问，支持跨区域）
   * Google Vertex AI（GCP 原生，Workload Identity Federation）

3. **沙盒执行**：每次交互都在具有严格网络和文件系统规则的容器中运行。Claude Code 强制实施工作空间范围权限以限制写入。所有更改通过 MR 流转，因此评审者可以看到差异，审批规则依然适用。

选择区域终端节点以降低延迟，同时利用现有云协议满足数据主权要求。

## Claude 能做什么？

Claude Code 支持强大的 CI/CD 工作流，转变你编写代码的方式：

* 根据 Issue 描述或评论创建和更新 MR
* 分析性能回归并提出优化建议
* 直接在分支中实现功能，然后发起 MR
* 修复测试或评论中发现的 Bug 和回归
* 响应后续评论以迭代修改

## 设置

### 快速设置

最快的入门方式是向 `.gitlab-ci.yml` 中添加一个最小化作业，并将 API 密钥设置为 masked 变量。

1. **添加 masked CI/CD 变量**
   * 进入 **Settings** → **CI/CD** → **Variables**
   * 添加 `ANTHROPIC_API_KEY`（masked，根据需要设为 protected）

2. **向 `.gitlab-ci.yml` 添加 Claude 作业**

```yaml theme={null}
stages:
  - ai

claude:
  stage: ai
  image: node:24-alpine3.21
  # 调整 rules 以适配你的触发方式：
  # - 手动运行
  # - merge request 事件
  # - 当评论包含 '@claude' 时通过 web/API 触发
  rules:
    - if: '$CI_PIPELINE_SOURCE == "web"'
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'
  variables:
    GIT_STRATEGY: fetch
  before_script:
    - apk update
    - apk add --no-cache git curl bash
    - curl -fsSL https://claude.ai/install.sh | bash
  script:
    # 可选：如果你的设置提供了 GitLab MCP 服务器，启动它
    - /bin/gitlab-mcp-server || true
    # 通过 web/API 触发器配合上下文负载调用时使用 AI_FLOW_* 变量
    - echo "$AI_FLOW_INPUT for $AI_FLOW_CONTEXT on $AI_FLOW_EVENT"
    - >
      claude
      -p "${AI_FLOW_INPUT:-'Review this MR and implement the requested changes'}"
      --permission-mode acceptEdits
      --allowedTools "Bash Read Edit Write mcp__gitlab"
      --debug
```

添加作业和 `ANTHROPIC_API_KEY` 变量后，通过 **CI/CD** → **Pipelines** 手动运行作业来测试，或从 MR 触发让 Claude 在分支中提出更新并在需要时发起 MR。

> **注意**：如需在 Amazon Bedrock 或 Google Vertex AI 而非 Claude API 上运行，请参阅下方[与 Amazon Bedrock 和 Google Vertex AI 一起使用](#与-amazon-bedrock-和-google-vertex-ai-一起使用)章节，了解身份验证和环境设置。

### 手动设置（推荐用于生产环境）

如果你偏好更可控的设置或需要企业级 Provider：

1. **配置 Provider 访问**：
   * **Claude API**：创建 `ANTHROPIC_API_KEY` 并将其存储为 masked CI/CD 变量
   * **Amazon Bedrock**：**配置 GitLab** → **AWS OIDC** 并为 Bedrock 创建 IAM 角色
   * **Google Vertex AI**：**为 GitLab 配置 Workload Identity Federation** → **GCP**

2. **为 GitLab API 操作添加项目凭证**：
   * 默认使用 `CI_JOB_TOKEN`，或创建一个具有 `api` 范围的 Project Access Token
   * 如果使用 PAT，将其存储为 `GITLAB_ACCESS_TOKEN`（masked）

3. **向 `.gitlab-ci.yml` 添加 Claude 作业**（见下方示例）

4. **（可选）启用基于提及的触发**：
   * 向事件监听器添加针对"Comments (notes)"的项目 webhook（如果有的话）
   * 当评论包含 `@claude` 时，让监听器通过 Pipeline 触发器 API 调用，附带 `AI_FLOW_INPUT` 和 `AI_FLOW_CONTEXT` 等变量

## 示例用例

### 将 Issue 转化为 MR

在 Issue 评论中：

```text theme={null}
@claude 根据 Issue 描述实现此功能
```

Claude 分析 Issue 和代码库，在分支中编写更改，并发起 MR 供评审。

### 获取实现帮助

在 MR 讨论中：

```text theme={null}
@claude 请提出一个缓存此 API 调用结果的具体方案
```

Claude 提出更改建议，添加适当的缓存代码，并更新 MR。

### 快速修复 Bug

在 Issue 或 MR 评论中：

```text theme={null}
@claude 修复用户仪表板组件中的 TypeError
```

Claude 定位 Bug，实现修复，并更新分支或发起新的 MR。

## 与 Amazon Bedrock 和 Google Vertex AI 一起使用

对于企业环境，你可以在自己的云基础设施上完全运行 Claude Code，获得相同的开发者体验。

### Amazon Bedrock

#### 前置条件

在配置 Claude Code 与 Amazon Bedrock 之前，你需要：

1. 一个已启用 Amazon Bedrock 并有权访问所需 Claude 模型的 AWS 账户
2. 在 AWS IAM 中将 GitLab 配置为 OIDC 身份提供者
3. 一个具有 Bedrock 权限的 IAM 角色，其信任策略限制为你的 GitLab 项目和引用
4. 用于角色代入的 GitLab CI/CD 变量：
   * `AWS_ROLE_TO_ASSUME`（角色 ARN）
   * `AWS_REGION`（Bedrock 区域）

#### 设置说明

配置 AWS 以允许 GitLab CI 作业通过 OIDC 代入 IAM 角色（无需静态密钥）。

**必要设置：**

1. 启用 Amazon Bedrock 并请求访问你的目标 Claude 模型
2. 为 GitLab 创建 IAM OIDC 提供者（如果尚不存在）
3. 创建一个 IAM 角色，受 GitLab OIDC 提供者信任，限制为你的项目和受保护的引用
4. 附加 Bedrock 调用 API 的最小权限

**需要存储在 CI/CD 变量中的必要值：**

* `AWS_ROLE_TO_ASSUME`
* `AWS_REGION`

在 Settings → CI/CD → Variables 中添加变量：

```yaml theme={null}
# 对于 Amazon Bedrock：
- AWS_ROLE_TO_ASSUME
- AWS_REGION
```

使用上方的 Amazon Bedrock 作业示例在运行时将 GitLab 作业令牌交换为临时 AWS 凭证。

### Google Vertex AI

#### 前置条件

在配置 Claude Code 与 Google Vertex AI 之前，你需要：

1. 一个 Google Cloud 项目，具备：
   * 已启用 Vertex AI API
   * 已配置 Workload Identity Federation 以信任 GitLab OIDC
2. 一个专用的服务账号，仅拥有必要的 Vertex AI 角色
3. 用于 WIF 的 GitLab CI/CD 变量：
   * `GCP_WORKLOAD_IDENTITY_PROVIDER`（完整资源名称）
   * `GCP_SERVICE_ACCOUNT`（服务账号邮箱）

#### 设置说明

配置 Google Cloud 以允许 GitLab CI 作业通过 Workload Identity Federation 模拟服务账号。

**必要设置：**

1. 启用 IAM Credentials API、STS API 和 Vertex AI API
2. 为 GitLab OIDC 创建 Workload Identity Pool 和 Provider
3. 创建一个具有 Vertex AI 角色的专用服务账号
4. 授予 WIF 主体模拟服务账号的权限

**需要存储在 CI/CD 变量中的必要值：**

* `GCP_WORKLOAD_IDENTITY_PROVIDER`
* `GCP_SERVICE_ACCOUNT`

在 Settings → CI/CD → Variables 中添加变量：

```yaml theme={null}
# 对于 Google Vertex AI：
- GCP_WORKLOAD_IDENTITY_PROVIDER
- GCP_SERVICE_ACCOUNT
- CLOUD_ML_REGION（例如 us-east5）
```

使用上方的 Google Vertex AI 作业示例进行无密钥身份验证。

## 配置示例

以下是可直接适配到你流水线中的即用型代码片段。

### 基础 .gitlab-ci.yml（Claude API）

```yaml theme={null}
stages:
  - ai

claude:
  stage: ai
  image: node:24-alpine3.21
  rules:
    - if: '$CI_PIPELINE_SOURCE == "web"'
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'
  variables:
    GIT_STRATEGY: fetch
  before_script:
    - apk update
    - apk add --no-cache git curl bash
    - curl -fsSL https://claude.ai/install.sh | bash
  script:
    - /bin/gitlab-mcp-server || true
    - >
      claude
      -p "${AI_FLOW_INPUT:-'Summarize recent changes and suggest improvements'}"
      --permission-mode acceptEdits
      --allowedTools "Bash Read Edit Write mcp__gitlab"
      --debug
  # Claude Code 将使用 CI/CD 变量中的 ANTHROPIC_API_KEY
```

### Amazon Bedrock 作业示例（OIDC）

**前置条件：**

* Amazon Bedrock 已启用，并有权访问你选择的 Claude 模型
* 在 AWS 中配置了 GitLab OIDC，角色信任你的 GitLab 项目和引用
* 具有 Bedrock 权限的 IAM 角色（建议最小权限）

**必要的 CI/CD 变量：**

* `AWS_ROLE_TO_ASSUME`：Bedrock 访问的 IAM 角色 ARN
* `AWS_REGION`：Bedrock 区域（例如 `us-west-2`）

```yaml theme={null}
claude-bedrock:
  stage: ai
  image: node:24-alpine3.21
  rules:
    - if: '$CI_PIPELINE_SOURCE == "web"'
  before_script:
    - apk add --no-cache bash curl jq git python3 py3-pip
    - pip install --no-cache-dir awscli
    - curl -fsSL https://claude.ai/install.sh | bash
    # 将 GitLab OIDC 令牌交换为 AWS 凭证
    - export AWS_WEB_IDENTITY_TOKEN_FILE="${CI_JOB_JWT_FILE:-/tmp/oidc_token}"
    - if [ -n "${CI_JOB_JWT_V2}" ]; then printf "%s" "$CI_JOB_JWT_V2" > "$AWS_WEB_IDENTITY_TOKEN_FILE"; fi
    - >
      aws sts assume-role-with-web-identity
      --role-arn "$AWS_ROLE_TO_ASSUME"
      --role-session-name "gitlab-claude-$(date +%s)"
      --web-identity-token "file://$AWS_WEB_IDENTITY_TOKEN_FILE"
      --duration-seconds 3600 > /tmp/aws_creds.json
    - export AWS_ACCESS_KEY_ID="$(jq -r .Credentials.AccessKeyId /tmp/aws_creds.json)"
    - export AWS_SECRET_ACCESS_KEY="$(jq -r .Credentials.SecretAccessKey /tmp/aws_creds.json)"
    - export AWS_SESSION_TOKEN="$(jq -r .Credentials.SessionToken /tmp/aws_creds.json)"
  script:
    - /bin/gitlab-mcp-server || true
    - >
      claude
      -p "${AI_FLOW_INPUT:-'Implement the requested changes and open an MR'}"
      --permission-mode acceptEdits
      --allowedTools "Bash Read Edit Write mcp__gitlab"
      --debug
  variables:
    AWS_REGION: "us-west-2"
```

> **注意**：Bedrock 的模型 ID 包含特定区域前缀（例如 `us.anthropic.claude-sonnet-4-6`）。如果你的工作流支持，可通过作业配置或提示传入所需模型。

### Google Vertex AI 作业示例（Workload Identity Federation）

**前置条件：**

* GCP 项目中已启用 Vertex AI API
* 已配置 Workload Identity Federation 以信任 GitLab OIDC
* 一个具有 Vertex AI 权限的服务账号

**必要的 CI/CD 变量：**

* `GCP_WORKLOAD_IDENTITY_PROVIDER`：Provider 的完整资源名称
* `GCP_SERVICE_ACCOUNT`：服务账号邮箱
* `CLOUD_ML_REGION`：Vertex 区域（例如 `us-east5`）

```yaml theme={null}
claude-vertex:
  stage: ai
  image: gcr.io/google.com/cloudsdktool/google-cloud-cli:slim
  rules:
    - if: '$CI_PIPELINE_SOURCE == "web"'
  before_script:
    - apt-get update && apt-get install -y git && apt-get clean
    - curl -fsSL https://claude.ai/install.sh | bash
    # 通过 WIF 向 Google Cloud 认证（无需下载密钥）
    - >
      gcloud auth login --cred-file=<(cat <<EOF
      {
        "type": "external_account",
        "audience": "${GCP_WORKLOAD_IDENTITY_PROVIDER}",
        "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
        "service_account_impersonation_url": "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/${GCP_SERVICE_ACCOUNT}:generateAccessToken",
        "token_url": "https://sts.googleapis.com/v1/token"
      }
      EOF
      )
    - gcloud config set project "$(gcloud projects list --format='value(projectId)' --filter="name:${CI_PROJECT_NAMESPACE}" | head -n1)" || true
  script:
    - /bin/gitlab-mcp-server || true
    - >
      CLOUD_ML_REGION="${CLOUD_ML_REGION:-us-east5}"
      claude
      -p "${AI_FLOW_INPUT:-'Review and update code as requested'}"
      --permission-mode acceptEdits
      --allowedTools "Bash Read Edit Write mcp__gitlab"
      --debug
  variables:
    CLOUD_ML_REGION: "us-east5"
```

> **注意**：使用 Workload Identity Federation 时，你无需存储服务账号密钥。使用仓库级别的信任条件和最小权限服务账号。

## 最佳实践

### CLAUDE.md 配置

在仓库根目录创建 `CLAUDE.md` 文件，定义编码规范、评审标准和项目特定规则。Claude 在运行期间读取此文件，并在提出更改时遵循你的约定。

### 安全注意事项

**切勿将 API 密钥或云凭证提交到仓库**。始终使用 GitLab CI/CD 变量：

* 将 `ANTHROPIC_API_KEY` 添加为 masked 变量（必要时设为 protected）
* 尽可能使用 Provider 特定的 OIDC（无需长期密钥）
* 限制作业权限和网络出口
* Claude 生成的 MR 应像其他贡献者的提交一样接受评审

### 优化性能

* 保持 `CLAUDE.md` 聚焦且简洁
* 提供清晰的 Issue/MR 描述以减少来回迭代
* 配置合理的作业超时以避免失控运行
* 在 Runner 中缓存 npm 和包安装

### CI 成本

使用 Claude Code 与 GitLab CI/CD 时，请注意相关成本：

* **GitLab Runner 时间**：
  * Claude 在你的 GitLab Runner 上运行，消耗计算分钟数
  * 详情请参阅你的 GitLab 计划的 Runner 计费

* **API 成本**：
  * 每次 Claude 交互根据提示和响应大小消耗 Token
  * Token 用量因任务复杂度和代码库大小而异
  * 详情请参阅 [Anthropic 定价](https://platform.claude.com/docs/en/about-claude/pricing)

* **成本优化建议**：
  * 使用具体的 `@claude` 命令以减少不必要的轮次
  * 设置适当的 `max_turns` 和作业超时值
  * 限制并发以控制并行运行数量

## 安全与治理

* 每个作业在具有受限网络访问的隔离容器中运行
* Claude 的更改通过 MR 流转，评审者可以看到每个差异
* 分支保护和审批规则适用于 AI 生成的代码
* Claude Code 使用工作空间范围权限来限制写入
* 成本由你控制，因为你使用自己的 Provider 凭证

## 故障排查

### Claude 未响应 @claude 命令

* 验证你的流水线是否被触发（手动、MR 事件或通过评论事件监听器/webhook）
* 确保 CI/CD 变量（`ANTHROPIC_API_KEY` 或云 Provider 设置）存在且未被 masked
* 检查评论是否包含 `@claude`（而非 `/claude`），以及你的提及触发器是否已配置

### 作业无法写入评论或发起 MR

* 确保 `CI_JOB_TOKEN` 对项目具有足够权限，或使用具有 `api` 范围的 Project Access Token
* 检查 `mcp__gitlab` 工具是否在 `--allowedTools` 中启用
* 确认作业在 MR 上下文中运行，或通过 `AI_FLOW_*` 变量具有足够上下文

### 身份验证错误

* **Claude API**：确认 `ANTHROPIC_API_KEY` 有效且未过期
* **Bedrock/Vertex**：验证 OIDC/WIF 配置、角色模拟和密钥名称；确认区域和模型可用性

## 高级配置

### 常用参数和变量

Claude Code 支持以下常用输入：

* `prompt` / `prompt_file`：通过内联（`-p`）或文件提供指令
* `max_turns`：限制来回迭代的次数
* `timeout_minutes`：限制总执行时间
* `ANTHROPIC_API_KEY`：Claude API 必需（Bedrock/Vertex 不使用）
* Provider 特定环境变量：`AWS_REGION`、Vertex 的项目/区域变量

> **注意**：确切的标志和参数可能因 `@anthropic-ai/claude-code` 版本而异。在作业中运行 `claude --help` 查看支持的选项。

### 自定义 Claude 的行为

你可以通过两种主要方式引导 Claude：

1. **CLAUDE.md**：定义编码规范、安全要求和项目约定。Claude 在运行期间读取并遵循你的规则。
2. **自定义提示**：通过作业中的 `prompt`/`prompt_file` 传递特定任务的指令。为不同作业使用不同提示（例如 review、implement、refactor）。
