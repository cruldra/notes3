# ComfyUI与Qwen-Image模型完整教程口播文案

## 开场白（30秒）

大家好！今天我要给大家带来一期非常实用的教程——就是在ComfyUI上使用Qwen-Image模型来生成图片。这期视频我们会从零开始，先用comfy-cli安装ComfyUI，然后详细介绍Qwen-Image模型的工作流使用方法，包括模型下载和正确的文件放置。如果你对AI图像生成感兴趣，特别是想体验阿里巴巴最新的Qwen-Image模型，相信这期视频会对你有所帮助。

## 第一部分：ComfyUI安装（3分钟）

### 什么是comfy-cli（45秒）

首先我们来了解一下[comfy-cli](https://github.com/Comfy-Org/comfy-cli)。comfy-cli是ComfyUI官方提供的命令行工具，它可以帮我们快速安装和管理ComfyUI环境。相比传统的手动安装方式，comfy-cli的优势就是我们不需要自己手动配置Python环境和依赖，整个安装过程会自动完成，非常方便。

### 安装前准备（30秒）

在开始安装之前，我们需要确保系统满足基本要求：
- Python 3.9或更高版本,可以通过`python --version`命令查看
- 至少8GB内存（推荐16GB），可以通过`free -h`命令查看
- 如果有NVIDIA显卡，确保安装了最新的显卡驱动,可以通过`nvidia-smi`命令查看

### comfy-cli安装步骤（1分45秒）

现在我们开始实际的安装过程：

**第一步：安装comfy-cli**
```powershell
pip install comfy-cli
```

安装完成后，我们可以验证一下：
```powershell
comfy --version
```

**第二步：使用comfy-cli安装ComfyUI**
```powershell
comfy install
```

这个命令会自动：
- 下载最新版本的ComfyUI
- 创建虚拟环境
- 安装所有必要的依赖
- 配置GPU支持（如果检测到NVIDIA显卡）

**第三步：启动ComfyUI**
```powershell
comfy launch -- --port 6006
```

如果一切正常，你会看到类似这样的输出：
```
To see the GUI go to: http://127.0.0.1:6006
```

现在打开浏览器，访问这个地址，你就能看到ComfyUI的界面了！

## 第二部分：Qwen-Image模型介绍（2分钟）

### 什么是Qwen-Image（1分钟）

Qwen-Image是阿里巴巴通义千问团队开发的最新图像生成模型,它最大的优势在于能够渲染包括中文在内的多种复杂文本以及精准的图像编辑能力。
具体的大家可以看下[这里](https://qwenlm.github.io/zh/blog/qwen-image/)

### ComfyUI内置支持（1分钟）

好消息是，ComfyUI已经内置了对Qwen-Image模型相关的工作流，我们只需要下载对应的模型就可以使用了

## 第三部分：模型下载与配置（2分30秒）

### 模型文件获取（1分钟）

这里我们可以看到comfyui已经内置了很多qwen-image相关的工作流，比如这个文生图、图生图、图像编辑和控制网等等，这里我们随便打开一个qwen-image工作流，然后comfyui会提示你需要哪些模型文件，并给出下载链接，我们只需要点击下载,它默认给的这个下载链接是huggingface的链接，国内有可能打不开，打不开的话我们就自己先下载到本地然后再传到comfyui即可

### 文件放置位置（1分30秒）

下载完成后，我们需要根据工作流的提示将模型文件放到ComfyUI的正确目录中。ComfyUI有一套标准的目录结构：

**目录结构说明：**
```
ComfyUI/
├── models/
│   ├── checkpoints/          # 完整模型文件
│   ├── diffusion_models/     # 扩散模型（UNET）
│   ├── text_encoders/        # 文本编码器
│   ├── vae/                  # VAE模型
│   └── clip_vision/          # 视觉编码器
```

## 第四部分：文生图工作流的简单介绍及使用（3分钟）

当我们把它需要的模型都放到正确的目录以后我们就可以运行这个工作流了，这里我们点击下方的这个运行按钮，然后稍等片刻，我们就可以看到生成的图片了
