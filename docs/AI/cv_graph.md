# 计算机视觉 (CV) 知识图谱

本文档旨在梳理计算机视觉 (CV) 生态系统中的关键概念、技术组件及其相互关系。

## 1. 基础模型 (Backbones)

基础模型是提取图像特征的骨干网络，决定了视觉任务的性能上限。

### 关键组件

```mermaid
graph TD
    Image[输入图像] --> CNN[卷积神经网络]
    Image --> Transformer[视觉 Transformer]
    
    subgraph CNNs [经典 CNN]
        ResNet
        EfficientNet
        ConvNeXt
    end
    
    subgraph ViTs [Vision Transformers]
        ViT[ViT]
        Swin[Swin Transformer]
        MAE[Masked Autoencoders]
    end
    
    CNN --> CNNs
    Transformer --> ViTs
```

### 参考链接

## 2. 核心任务 (Core Tasks)

核心任务是 CV 领域最基础的问题，包括分类、检测和分割。

### 关键组件

```mermaid
graph TD
    Backbone[骨干网络] --> Head[任务头]
    
    Head --> Classification[图像分类]
    Head --> Detection[目标检测]
    Head --> Segmentation[图像分割]
    
    subgraph Detectors [检测器]
        YOLO
        RCNN[Faster R-CNN]
        DETR
    end
    
    subgraph Segmentors [分割器]
        UNet
        MaskRCNN[Mask R-CNN]
        SAM[Segment Anything]
    end
    
    Detection --> Detectors
    Segmentation --> Segmentors
```

### 参考链接

## 3. 图像处理 (Image Processing)

关注图像质量增强、底层视觉处理及人脸/人体分析。

### 关键组件

```mermaid
graph LR
    Input[原始图像] --> Enhancement[图像增强]
    Input --> Analysis[人体/人脸分析]
    
    subgraph Restoration [复原与增强]
        SR[超分辨率]
        Denoising[去噪]
        Inpainting[修复]
    end
    
    subgraph HumanAnalysis [人体与人脸]
        Face[人脸识别/检测]
        Pose[姿态估计]
        Hand[手势识别]
        HumanLib[Human 库]
    end
    
    Enhancement --> Restoration
    Analysis --> HumanAnalysis
```

### 概念说明

*   **Human (Human Library)**: 一个基于 TensorFlow.js 的浏览器端计算机视觉库，集成了一系列轻量级 AI 模型，专注于实时 3D 人脸检测、身体姿态追踪、手势识别、虹膜分析及情绪预测等任务。

### 参考链接
- [Human 官方仓库](https://github.com/vladmandic/human)

## 4. 视频与3D (Video & 3D)

处理时序数据和三维空间信息的任务。

### 关键组件

```mermaid
graph TD
    Video[视频流] --> Tracking[目标追踪]
    Video --> Action[行为识别]
    Images[多视角图像] --> Reconstruction[3D 重建]
    
    subgraph 3DVision [3D 视觉]
        NeRF
        GaussianSplatting[3D Gaussian Splatting]
        PointClouds[点云处理]
    end
    
    Reconstruction --> 3DVision
```

### 参考链接

## 5. 生成式视觉 (Generative Vision)

利用 AI 生成或编辑图像和视频。

### 关键组件

```mermaid
graph TD
    Noise[噪声/文本] --> GenModel[生成模型]
    
    subgraph Diffusion [扩散模型]
        SD[Stable Diffusion]
        Midjourney
        ControlNet
    end
    
    subgraph GANs [生成对抗网络]
        StyleGAN
    end
    
    GenModel --> Diffusion
    GenModel --> GANs
```

### 参考链接

## 6. 评估与应用 (Evaluation & Application)

模型评估指标及实际应用场景。

### 关键组件

```mermaid
graph LR
    Model --> Metrics[评估指标]
    Model --> Deployment[部署应用]
    
    subgraph Evaluation [评估指标]
        mAP[mAP]
        IoU[IoU]
        FID[FID]
    end
    
    subgraph Apps [应用场景]
        AutoDrive[自动驾驶]
        Medical[医疗影像]
        Security[安防监控]
        WebAR[Web AR/互动]
    end
    
    Metrics --> Evaluation
    Deployment --> Apps
```

### 参考链接
