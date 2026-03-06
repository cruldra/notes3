# 如何使用 Vertex AI 设置 Gemini CLI

作者：[Minami Munakata](https://medium.com/@minamimunakata) · 2 分钟阅读 · 2025年6月30日

## 简介

本文是我在使用 **Vertex AI** 设置 **Gemini CLI** 时的个人笔记。我按照官方入门指南完成了整个过程。

*   **官方文档：** [README.md](https://github.com/google-gemini/gemini-cli/blob/main/README.md)

## 1. 先决条件

正如标题所示，本指南重点介绍如何在 **Vertex AI** 上使用 Gemini CLI。

使用 Vertex AI 有两种方式：

*   标准 **Vertex AI**
*   **Vertex AI Express 模式**

**Express 模式** 让你可以快速试用 Vertex AI 功能，而无需完全设置 Google Cloud 项目。只需一个 API 密钥，你就可以访问 Vertex AI Studio 和 Gemini API 的部分功能。另一方面，标准 Vertex AI 需要设置 Google Cloud 项目。

在本指南中，我选择使用**标准 Vertex AI**，而不是 Express 模式。

*   **为什么选择 Vertex AI：**
    根据[服务条款和隐私声明](https://github.com/google-gemini/gemini-cli/blob/main/docs/tos-privacy.md)，当使用个人 Google 账号时，你的提示词和代码可能会被用于改进 Google 的模型质量。相比之下，使用 Vertex AI 时，输入数据明确**不会**被用于模型训练。这就是我选择了 Vertex AI 的原因。

## 2. 为 Gemini CLI 准备 Vertex AI

### 启用 Vertex AI

你需要创建一个 Google Cloud 项目并[启用 Vertex AI](https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com)。
如果你是 Google Cloud 新手，请参考[这份指南](https://cloud.google.com/vertex-ai/docs/start/cloud-environment)。

### 设置环境变量

在你的终端中设置以下环境变量：

```bash
export GOOGLE_CLOUD_PROJECT="YOUR_PROJECT_ID"
export GOOGLE_CLOUD_LOCATION="YOUR_PROJECT_LOCATION" # 例如：us-central1
export GOOGLE_GENAI_USE_VERTEXAI=true
```

注意：这些设置在每次会话后都会重置。为了方便，请将它们添加到你的 `.zshrc` 或 `.bash_profile` 中以保持持久性。

### 设置应用默认凭证 (ADC)

运行：

```bash
gcloud auth application-default login
```

（确保你已安装 [Google Cloud CLI](https://cloud.google.com/sdk/docs/install)。）

如果未设置 ADC，你会看到如下错误：

```
[API 错误：无法加载默认凭证。请访问 https://cloud.google.com/docs/authentication/getting-started 获取更多信息。]
```

## 3. 安装 Gemini CLI

要全局使用 `gemini`，请使用 npm 安装：

```bash
npm install -g @google/gemini-cli
```

安装完成后，你可以通过以下命令启动 CLI：

```bash
gemini
```

## 4. 选择认证方式

当你第一次运行 `gemini` 时，系统会提示你选择一种认证方式。
选择 **Vertex AI 认证**。

✅ 就这样！你的 Gemini CLI 与 Vertex AI 设置已完成，可以开始使用了。

---

**标签：** #GeminiCli #VertexAI #Gemini
