# 使用 DSPy 生成用于代码文档的 llms.txt

本教程演示了如何使用 DSPy 自动为 DSPy 仓库本身生成一个 `llms.txt` 文件。`llms.txt` 标准提供了对 LLM 友好的文档，帮助 AI 系统更好地理解代码库。

## 什么是 llms.txt？

`llms.txt` 是一个提议的标准，用于提供关于项目的结构化、LLM 友好的文档。它通常包括：

- 项目概述和目的
- 关键概念和术语
- 架构和结构
- 使用示例
- 重要文件和目录

## 构建一个用于生成 llms.txt 的 DSPy 程序

让我们创建一个 DSPy 程序来分析仓库并生成全面的 `llms.txt` 文档。

### 第 1 步：定义我们的签名 (Signatures)

首先，我们将定义用于文档生成不同方面的签名：

```python
import dspy
from typing import List

class AnalyzeRepository(dspy.Signature):
    """分析仓库结构并识别关键组件。"""
    repo_url: str = dspy.InputField(desc="GitHub 仓库 URL")
    file_tree: str = dspy.InputField(desc="仓库文件结构")
    readme_content: str = dspy.InputField(desc="README.md 内容")
    
    project_purpose: str = dspy.OutputField(desc="项目的主要目的和目标")
    key_concepts: list[str] = dspy.OutputField(desc="重要概念和术语列表")
    architecture_overview: str = dspy.OutputField(desc="高层架构描述")

class AnalyzeCodeStructure(dspy.Signature):
    """分析代码结构以识别重要目录和文件。"""
    file_tree: str = dspy.InputField(desc="仓库文件结构")
    package_files: str = dspy.InputField(desc="关键包和配置文件")
    
    important_directories: list[str] = dspy.OutputField(desc="关键目录及其用途")
    entry_points: list[str] = dspy.OutputField(desc="主要入口点和重要文件")
    development_info: str = dspy.OutputField(desc="开发设置和工作流信息")

class GenerateLLMsTxt(dspy.Signature):
    """根据分析的仓库信息生成全面的 llms.txt 文件。"""
    project_purpose: str = dspy.InputField()
    key_concepts: list[str] = dspy.InputField()
    architecture_overview: str = dspy.InputField()
    important_directories: list[str] = dspy.InputField()
    entry_points: list[str] = dspy.InputField()
    development_info: str = dspy.InputField()
    usage_examples: str = dspy.InputField(desc="常见使用模式和示例")
    
    llms_txt_content: str = dspy.OutputField(desc="遵循标准格式的完整 llms.txt 文件内容")
```

### 第 2 步：创建仓库分析器模块

```python
class RepositoryAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyze_repo = dspy.ChainOfThought(AnalyzeRepository)
        self.analyze_structure = dspy.ChainOfThought(AnalyzeCodeStructure)
        self.generate_examples = dspy.ChainOfThought("repo_info -> usage_examples")
        self.generate_llms_txt = dspy.ChainOfThought(GenerateLLMsTxt)
    
    def forward(self, repo_url, file_tree, readme_content, package_files):
        # 分析仓库目的和概念
        repo_analysis = self.analyze_repo(
            repo_url=repo_url,
            file_tree=file_tree,
            readme_content=readme_content
        )
        
        # 分析代码结构
        structure_analysis = self.analyze_structure(
            file_tree=file_tree,
            package_files=package_files
        )
        
        # 生成使用示例
        usage_examples = self.generate_examples(
            repo_info=f"Purpose: {repo_analysis.project_purpose}\nConcepts: {repo_analysis.key_concepts}"
        )
        
        # 生成最终的 llms.txt
        llms_txt = self.generate_llms_txt(
            project_purpose=repo_analysis.project_purpose,
            key_concepts=repo_analysis.key_concepts,
            architecture_overview=repo_analysis.architecture_overview,
            important_directories=structure_analysis.important_directories,
            entry_points=structure_analysis.entry_points,
            development_info=structure_analysis.development_info,
            usage_examples=usage_examples.usage_examples
        )
        
        return dspy.Prediction(
            llms_txt_content=llms_txt.llms_txt_content,
            analysis=repo_analysis,
            structure=structure_analysis
        )
```

### 第 3 步：收集仓库信息

让我们创建辅助函数来提取仓库信息：

```python
import requests
import os
from pathlib import Path

os.environ["GITHUB_ACCESS_TOKEN"] = "<your_access_token>"

def get_github_file_tree(repo_url):
    """从 GitHub API 获取仓库文件结构。"""
    # 从 URL 提取 owner/repo
    parts = repo_url.rstrip('/').split('/')
    owner, repo = parts[-2], parts[-1]
    
    api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"
    response = requests.get(api_url, headers={
        "Authorization": f"Bearer {os.environ.get('GITHUB_ACCESS_TOKEN')}"
    })
    
    if response.status_code == 200:
        tree_data = response.json()
        file_paths = [item['path'] for item in tree_data['tree'] if item['type'] == 'blob']
        return '\n'.join(sorted(file_paths))
    else:
        raise Exception(f"Failed to fetch repository tree: {response.status_code}")

def get_github_file_content(repo_url, file_path):
    """从 GitHub 获取特定文件内容。"""
    parts = repo_url.rstrip('/').split('/')
    owner, repo = parts[-2], parts[-1]
    
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    response = requests.get(api_url, headers={
        "Authorization": f"Bearer {os.environ.get('GITHUB_ACCESS_TOKEN')}"
    })
    
    if response.status_code == 200:
        import base64
        content = base64.b64decode(response.json()['content']).decode('utf-8')
        return content
    else:
        return f"Could not fetch {file_path}"

def gather_repository_info(repo_url):
    """收集所有必要的仓库信息。"""
    file_tree = get_github_file_tree(repo_url)
    readme_content = get_github_file_content(repo_url, "README.md")
    
    # 获取关键包文件
    package_files = []
    for file_path in ["pyproject.toml", "setup.py", "requirements.txt", "package.json"]:
        try:
            content = get_github_file_content(repo_url, file_path)
            if "Could not fetch" not in content:
                package_files.append(f"=== {file_path} ===\n{content}")
        except:
            continue
    
    package_files_content = "\n\n".join(package_files)
    
    return file_tree, readme_content, package_files_content
```

### 第 4 步：配置 DSPy 并生成 llms.txt

```python
def generate_llms_txt_for_dspy():
    # 配置 DSPy (使用你首选的 LM)
    lm = dspy.LM(model="gpt-4o-mini")
    dspy.configure(lm=lm)
    os.environ["OPENAI_API_KEY"] = "<YOUR OPENAI KEY>"
    
    # 初始化我们的分析器
    analyzer = RepositoryAnalyzer()
    
    # 收集 DSPy 仓库信息
    repo_url = "https://github.com/stanfordnlp/dspy"
    file_tree, readme_content, package_files = gather_repository_info(repo_url)
    
    # 生成 llms.txt
    result = analyzer(
        repo_url=repo_url,
        file_tree=file_tree,
        readme_content=readme_content,
        package_files=package_files
    )
    
    return result

# 运行生成
if __name__ == "__main__":
    result = generate_llms_txt_for_dspy()
    
    # 保存生成的 llms.txt
    with open("llms.txt", "w") as f:
        f.write(result.llms_txt_content)
    
    print("Generated llms.txt file!")
    print("\nPreview:")
    print(result.llms_txt_content[:500] + "...")
```

## 预期输出结构

为 DSPy 生成的 `llms.txt` 将遵循此结构：

```
# DSPy: Programming Language Models

## Project Overview
DSPy is a framework for programming—rather than prompting—language models...

## Key Concepts
- **Modules**: Building blocks for LM programs
- **Signatures**: Input/output specifications  
- **Teleprompters**: Optimization algorithms
- **Predictors**: Core reasoning components

## Architecture
- `/dspy/`: Main package directory
  - `/adapters/`: Input/output format handlers
  - `/clients/`: LM client interfaces
  - `/predict/`: Core prediction modules
  - `/teleprompt/`: Optimization algorithms

## Usage Examples
1. **Building a Classifier**: Using DSPy, a user can define a modular classifier that takes in text data and categorizes it into predefined classes. The user can specify the classification logic declaratively, allowing for easy adjustments and optimizations.
2. **Creating a RAG Pipeline**: A developer can implement a retrieval-augmented generation pipeline that first retrieves relevant documents based on a query and then generates a coherent response using those documents. DSPy facilitates the integration of retrieval and generation components seamlessly.
3. **Optimizing Prompts**: Users can leverage DSPy to create a system that automatically optimizes prompts for language models based on performance metrics, improving the quality of responses over time without manual intervention.
4. **Implementing Agent Loops**: A user can design an agent loop that continuously interacts with users, learns from feedback, and refines its responses, showcasing the self-improving capabilities of the DSPy framework.
5. **Compositional Code**: Developers can write compositional code that allows different modules of the AI system to interact with each other, enabling complex workflows that can be easily modified and extended.
```

生成的 `llms.txt` 文件提供了 DSPy 仓库的全面、LLM 友好的概述，可以帮助其他 AI 系统更好地理解和使用该代码库。

## 下一步

- 扩展程序以分析多个仓库
- 添加对不同文档格式的支持
- 创建文档质量评估指标
- 构建用于交互式仓库分析的 Web 界面