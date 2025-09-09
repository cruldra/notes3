# ComfyUI 模型路径配置指南

## 概述

ComfyUI使用灵活的路径管理系统来组织和访问各种AI模型文件。本指南详细介绍了如何配置模型路径，包括默认路径结构、自定义路径配置、多路径管理等高级功能。

## 默认路径结构

### 基础目录配置

ComfyUI的所有路径都基于`base_path`，默认为ComfyUI安装目录：

```
ComfyUI/
├── models/                    # 模型根目录
│   ├── checkpoints/          # 完整模型检查点
│   ├── diffusion_models/     # 扩散模型（UNET）
│   ├── unet/                 # UNET模型（别名）
│   ├── clip/                 # CLIP文本编码器
│   ├── text_encoders/        # 文本编码器（包含clip）
│   ├── clip_vision/          # CLIP视觉编码器
│   ├── vae/                  # VAE编码器/解码器
│   ├── loras/                # LoRA权重文件
│   ├── controlnet/           # ControlNet模型
│   ├── t2i_adapter/          # Text-to-Image适配器
│   ├── upscale_models/       # 图像放大模型
│   ├── embeddings/           # 文本嵌入
│   ├── hypernetworks/        # Hypernetwork
│   ├── style_models/         # 风格模型
│   ├── gligen/               # GLIGEN模型
│   ├── photomaker/           # PhotoMaker模型
│   ├── diffusers/            # Diffusers格式模型
│   ├── vae_approx/           # VAE近似模型
│   ├── audio_encoders/       # 音频编码器
│   ├── model_patches/        # 模型补丁
│   ├── classifiers/          # 分类器模型
│   └── configs/              # 模型配置文件
├── input/                    # 输入文件目录
├── output/                   # 输出文件目录
├── temp/                     # 临时文件目录
├── user/                     # 用户数据目录
└── custom_nodes/             # 自定义节点目录
```

### 支持的文件格式

```python
# 模型文件支持的扩展名
supported_pt_extensions = {
    '.ckpt',        # Checkpoint格式
    '.pt',          # PyTorch格式
    '.pt2',         # PyTorch 2.0格式
    '.bin',         # 二进制格式
    '.pth',         # PyTorch Hub格式
    '.safetensors', # SafeTensors格式（推荐）
    '.pkl',         # Pickle格式
    '.sft'          # SafeTensors简写
}

# 配置文件格式
config_extensions = {'.yaml'}

# 特殊格式
diffusers_format = {'folder'}  # Diffusers使用文件夹格式
```

## 路径映射详解

### 核心路径映射

```python
# 默认路径映射配置
folder_names_and_paths = {
    # 完整模型
    "checkpoints": (["models/checkpoints"], supported_pt_extensions),
    
    # 分离组件
    "diffusion_models": (["models/unet", "models/diffusion_models"], supported_pt_extensions),
    "text_encoders": (["models/text_encoders", "models/clip"], supported_pt_extensions),
    "clip_vision": (["models/clip_vision"], supported_pt_extensions),
    "vae": (["models/vae"], supported_pt_extensions),
    
    # 增强组件
    "loras": (["models/loras"], supported_pt_extensions),
    "controlnet": (["models/controlnet", "models/t2i_adapter"], supported_pt_extensions),
    "embeddings": (["models/embeddings"], supported_pt_extensions),
    "hypernetworks": (["models/hypernetworks"], supported_pt_extensions),
    
    # 特殊模型
    "upscale_models": (["models/upscale_models"], supported_pt_extensions),
    "style_models": (["models/style_models"], supported_pt_extensions),
    "gligen": (["models/gligen"], supported_pt_extensions),
    "photomaker": (["models/photomaker"], supported_pt_extensions),
    "audio_encoders": (["models/audio_encoders"], supported_pt_extensions),
    
    # 其他
    "diffusers": (["models/diffusers"], ["folder"]),
    "vae_approx": (["models/vae_approx"], supported_pt_extensions),
    "model_patches": (["models/model_patches"], supported_pt_extensions),
    "classifiers": (["models/classifiers"], [""]),
    "configs": (["models/configs"], [".yaml"]),
    "custom_nodes": (["custom_nodes"], set())
}
```

### 路径别名系统

```python
# 兼容性别名映射
legacy_mapping = {
    "unet": "diffusion_models",    # unet目录映射到diffusion_models
    "clip": "text_encoders"        # clip目录映射到text_encoders
}
```

## 自定义路径配置

### 1. 使用extra_model_paths.yaml

创建`extra_model_paths.yaml`文件来配置额外的模型路径：

```yaml
# 重命名 extra_model_paths.yaml.example 为 extra_model_paths.yaml

# Automatic1111 WebUI兼容配置
a111:
    base_path: /path/to/stable-diffusion-webui/
    checkpoints: models/Stable-diffusion
    configs: models/Stable-diffusion
    vae: models/VAE
    loras: |
         models/Lora
         models/LyCORIS
    upscale_models: |
                  models/ESRGAN
                  models/RealESRGAN
                  models/SwinIR
    embeddings: embeddings
    hypernetworks: models/hypernetworks
    controlnet: models/ControlNet

# ComfyUI标准配置
comfyui:
    base_path: /path/to/comfyui/
    is_default: true  # 标记为默认路径
    checkpoints: models/checkpoints/
    clip: models/clip/
    clip_vision: models/clip_vision/
    configs: models/configs/
    controlnet: models/controlnet/
    diffusion_models: |
                 models/diffusion_models
                 models/unet
    embeddings: models/embeddings/
    loras: models/loras/
    upscale_models: models/upscale_models/
    vae: models/vae/

# 自定义配置
custom_setup:
    base_path: /shared/ai_models/
    checkpoints: sd_models/checkpoints
    loras: sd_models/loras
    vae: sd_models/vae
    controlnet: controlnet_models
```

### 2. 多路径配置

```yaml
# 支持多个路径的配置
multi_path_config:
    base_path: /main/models/
    loras: |
         loras/character
         loras/style
         loras/concept
    checkpoints: |
               checkpoints/sd15
               checkpoints/sdxl
               checkpoints/sd3
    controlnet: |
              controlnet/sd15
              controlnet/sdxl
```

### 3. 命令行参数配置

```bash
# 使用--base-directory参数重置所有默认路径
python main.py --base-directory /custom/path/

# 使用--extra-model-paths-config指定配置文件
python main.py --extra-model-paths-config /path/to/config.yaml
```

## 高级路径管理

### 1. 动态路径添加

```python
import folder_paths

# 添加新的模型文件夹路径
folder_paths.add_model_folder_path(
    folder_name="loras",
    full_folder_path="/additional/lora/path",
    is_default=False  # 是否设为默认路径
)

# 添加为默认路径（优先级最高）
folder_paths.add_model_folder_path(
    folder_name="checkpoints", 
    full_folder_path="/priority/checkpoint/path",
    is_default=True
)
```

### 2. 路径查询和验证

```python
# 获取指定类型的所有路径
checkpoint_paths = folder_paths.get_folder_paths("checkpoints")
print(f"Checkpoint paths: {checkpoint_paths}")

# 获取文件列表
lora_files = folder_paths.get_filename_list("loras")
print(f"Available LoRAs: {lora_files}")

# 获取文件完整路径
full_path = folder_paths.get_full_path("checkpoints", "model.safetensors")
if full_path:
    print(f"Model found at: {full_path}")
else:
    print("Model not found")

# 获取文件完整路径（找不到时抛出异常）
try:
    full_path = folder_paths.get_full_path_or_raise("loras", "style.safetensors")
    print(f"LoRA found at: {full_path}")
except FileNotFoundError as e:
    print(f"Error: {e}")
```

### 3. 缓存管理

```python
# 使用缓存助手提高性能
with folder_paths.cache_helper:
    # 在此上下文中的文件列表查询会被缓存
    files1 = folder_paths.get_filename_list("checkpoints")
    files2 = folder_paths.get_filename_list("loras")
    # 重复查询会使用缓存，提高性能

# 手动清除缓存
folder_paths.filename_list_cache.clear()
```

## 特殊路径功能

### 1. 注解路径系统

ComfyUI支持在文件名中使用注解来指定特定目录：

```python
# 文件名注解格式
annotated_files = {
    "image.png[output]": "保存到output目录",
    "input.jpg[input]": "从input目录读取", 
    "temp.txt[temp]": "使用temp目录"
}

# 获取注解路径
def get_annotated_path(filename):
    return folder_paths.get_annotated_filepath(filename)

# 示例
output_path = get_annotated_path("result.png[output]")
# 返回: /path/to/comfyui/output/result.png
```

### 2. 输入输出目录管理

```python
# 获取标准目录
input_dir = folder_paths.get_input_directory()
output_dir = folder_paths.get_output_directory()
temp_dir = folder_paths.get_temp_directory()
user_dir = folder_paths.get_user_directory()

# 设置自定义目录
folder_paths.set_input_directory("/custom/input")
folder_paths.set_output_directory("/custom/output")
folder_paths.set_temp_directory("/custom/temp")
folder_paths.set_user_directory("/custom/user")

# 获取输入子文件夹列表
subfolders = folder_paths.get_input_subfolders()
print(f"Input subfolders: {subfolders}")
```

### 3. 文件过滤和搜索

```python
# 按内容类型过滤文件
import os
files = os.listdir(folder_paths.get_input_directory())

# 过滤图片文件
images = folder_paths.filter_files_content_types(files, ["image"])

# 过滤视频文件
videos = folder_paths.filter_files_content_types(files, ["video"])

# 过滤音频文件
audios = folder_paths.filter_files_content_types(files, ["audio"])

# 过滤3D模型文件
models = folder_paths.filter_files_content_types(files, ["model"])

# 按扩展名过滤
safetensors_files = folder_paths.filter_files_extensions(
    files, 
    ['.safetensors']
)
```

## 实际应用场景

### 1. 多用户环境配置

```yaml
# 为不同用户配置独立的模型路径
user_alice:
    base_path: /users/alice/ai_models/
    checkpoints: personal_models/checkpoints
    loras: personal_models/loras
    
user_bob:
    base_path: /users/bob/ai_models/
    checkpoints: work_models/checkpoints
    loras: work_models/loras

# 共享模型库
shared_models:
    base_path: /shared/ai_models/
    is_default: true
    checkpoints: public_checkpoints
    loras: public_loras
```

### 2. 网络存储配置

```yaml
# 网络附加存储(NAS)配置
nas_storage:
    base_path: /mnt/nas/ai_models/
    checkpoints: |
               stable_diffusion/checkpoints
               flux/checkpoints
               sd3/checkpoints
    loras: |
         character_loras
         style_loras
         concept_loras
    vae: vae_models
    controlnet: controlnet_models

# 本地缓存配置
local_cache:
    base_path: /local/cache/
    is_default: true  # 优先使用本地缓存
    checkpoints: cached_checkpoints
    loras: cached_loras
```

### 3. 开发环境配置

```yaml
# 开发环境
development:
    base_path: /dev/ai_models/
    checkpoints: test_models/checkpoints
    loras: test_models/loras
    
# 生产环境
production:
    base_path: /prod/ai_models/
    is_default: true
    checkpoints: |
               production_models/sd15
               production_models/sdxl
               production_models/sd3
    loras: production_loras
    
# 实验环境
experimental:
    base_path: /exp/ai_models/
    checkpoints: experimental_models
    loras: experimental_loras
```

## 性能优化建议

### 1. 存储优化

```python
# 使用SSD存储常用模型
ssd_config = {
    "base_path": "/ssd/ai_models/",
    "checkpoints": "frequently_used",
    "loras": "popular_loras"
}

# 使用HDD存储归档模型
hdd_config = {
    "base_path": "/hdd/ai_models/",
    "checkpoints": "archived_models",
    "loras": "old_loras"
}
```

### 2. 缓存策略

```python
# 启用文件列表缓存
def optimized_file_access():
    with folder_paths.cache_helper:
        # 批量查询文件列表
        checkpoints = folder_paths.get_filename_list("checkpoints")
        loras = folder_paths.get_filename_list("loras")
        vaes = folder_paths.get_filename_list("vae")
        
        return {
            "checkpoints": checkpoints,
            "loras": loras, 
            "vaes": vaes
        }
```

### 3. 路径优先级管理

```python
# 设置路径优先级
def setup_path_priority():
    # 高优先级：本地SSD
    folder_paths.add_model_folder_path(
        "checkpoints", "/ssd/models/checkpoints", is_default=True
    )
    
    # 中优先级：本地HDD
    folder_paths.add_model_folder_path(
        "checkpoints", "/hdd/models/checkpoints", is_default=False
    )
    
    # 低优先级：网络存储
    folder_paths.add_model_folder_path(
        "checkpoints", "/nas/models/checkpoints", is_default=False
    )
```

## 故障排除

### 1. 常见问题

#### 模型找不到
```python
# 检查路径配置
def debug_model_paths():
    folder_name = "checkpoints"
    paths = folder_paths.get_folder_paths(folder_name)
    print(f"Configured paths for {folder_name}: {paths}")
    
    for path in paths:
        if os.path.exists(path):
            files = os.listdir(path)
            print(f"Files in {path}: {files[:5]}...")  # 显示前5个文件
        else:
            print(f"Path does not exist: {path}")
```

#### 权限问题
```bash
# 检查目录权限
ls -la /path/to/models/

# 修复权限
chmod -R 755 /path/to/models/
chown -R user:group /path/to/models/
```

#### 路径配置错误
```yaml
# 错误配置示例
wrong_config:
    base_path: /nonexistent/path/  # 路径不存在
    checkpoints: models\checkpoints  # Windows路径分隔符错误

# 正确配置
correct_config:
    base_path: /existing/path/
    checkpoints: models/checkpoints  # 使用正斜杠
```

### 2. 调试工具

```python
def diagnose_path_issues():
    """诊断路径配置问题"""
    
    # 检查基础路径
    base_path = folder_paths.base_path
    print(f"Base path: {base_path}")
    print(f"Base path exists: {os.path.exists(base_path)}")
    
    # 检查所有配置的路径
    for folder_name, (paths, extensions) in folder_paths.folder_names_and_paths.items():
        print(f"\n{folder_name}:")
        print(f"  Extensions: {extensions}")
        
        for i, path in enumerate(paths):
            exists = os.path.exists(path)
            print(f"  Path {i}: {path} (exists: {exists})")
            
            if exists:
                try:
                    files = folder_paths.get_filename_list(folder_name)
                    print(f"    Files found: {len(files)}")
                except Exception as e:
                    print(f"    Error reading files: {e}")

# 运行诊断
diagnose_path_issues()
```

## 总结

ComfyUI的模型路径系统提供了强大的灵活性：

1. **默认结构清晰**：标准的models目录组织
2. **配置灵活**：支持YAML配置文件和命令行参数
3. **多路径支持**：可以配置多个搜索路径
4. **兼容性好**：支持A1111等其他工具的路径结构
5. **性能优化**：内置缓存和优化机制
6. **扩展性强**：支持动态添加路径和自定义配置

正确配置模型路径不仅能提高ComfyUI的使用效率，还能更好地组织和管理大量的AI模型文件。建议根据实际需求选择合适的配置方案，并定期检查和优化路径设置。
