# ComfyUI 项目目录结构

## 项目概览

ComfyUI是一个基于节点的图形化界面，用于Stable Diffusion和其他AI图像生成模型。本文档详细介绍了项目的目录结构和各部分的功能。

## 根目录文件

```
ComfyUI/
├── main.py                    # 主程序入口文件
├── server.py                  # Web服务器启动文件
├── execution.py               # 工作流执行引擎
├── nodes.py                   # 核心节点定义
├── folder_paths.py            # 路径管理模块
├── requirements.txt           # Python依赖包列表
├── pyproject.toml            # 项目配置文件
├── README.md                 # 项目说明文档
├── README_ZH.MD              # 中文说明文档
├── LICENSE                   # 开源许可证
├── CONTRIBUTING.md           # 贡献指南
└── CODEOWNERS               # 代码所有者配置
```

## 核心模块目录

### 1. comfy/ - 核心功能模块
```
comfy/
├── model_management.py      # 模型内存管理
├── model_patcher.py         # 模型补丁系统
├── sd.py                    # Stable Diffusion核心
├── clip_model.py            # CLIP文本编码器
├── clip_vision.py           # CLIP视觉编码器
├── lora.py                  # LoRA权重处理
├── controlnet.py            # ControlNet支持
├── samplers.py              # 采样器实现
├── sample.py                # 采样核心逻辑
├── model_base.py            # 模型基类
├── model_detection.py       # 模型类型检测
├── supported_models.py      # 支持的模型列表
├── latent_formats.py        # 潜在空间格式
├── diffusers_convert.py     # Diffusers格式转换
├── diffusers_load.py        # Diffusers模型加载
├── hooks.py                 # 钩子系统
├── options.py               # 配置选项
├── ops.py                   # 操作符重载
├── float.py                 # 浮点数处理
├── conds.py                 # 条件处理
├── gligen.py                # GLIGEN支持
├── checkpoint_pickle.py     # 检查点序列化
├── cli_args.py              # 命令行参数
├── context_windows.py       # 上下文窗口
├── patcher_extension.py     # 补丁扩展
├── rmsnorm.py               # RMSNorm实现
├── sampler_helpers.py       # 采样器辅助函数
├── model_sampling.py        # 模型采样配置
├── lora_convert.py          # LoRA格式转换
├── comfy_types/             # 类型定义
├── ldm/                     # Latent Diffusion Models
├── k_diffusion/             # K-Diffusion采样器
├── cldm/                    # ControlLDM
├── t2i_adapter/             # Text-to-Image适配器
├── extra_samplers/          # 额外采样器
├── audio_encoders/          # 音频编码器
├── image_encoders/          # 图像编码器
├── sd1_tokenizer/           # SD1.x分词器
└── 配置文件/                 # 各种JSON配置文件
```

### 2. comfy_extras/ - 扩展节点
```
comfy_extras/
├── nodes_advanced_samplers.py    # 高级采样器节点
├── nodes_model_merging.py         # 模型合并节点
├── nodes_controlnet.py            # ControlNet节点
├── nodes_flux.py                  # Flux模型节点
├── nodes_audio.py                 # 音频处理节点
├── nodes_images.py                # 图像处理节点
├── nodes_latent.py                # 潜在空间节点
├── nodes_mask.py                  # 遮罩处理节点
├── nodes_compositing.py           # 图像合成节点
├── nodes_cond.py                  # 条件处理节点
├── nodes_cfg.py                   # CFG相关节点
├── nodes_hypernetwork.py          # Hypernetwork节点
├── nodes_lora_extract.py          # LoRA提取节点
├── nodes_model_advanced.py        # 高级模型节点
├── nodes_model_patch.py           # 模型补丁节点
├── nodes_upscale.py               # 图像放大节点
├── nodes_video.py                 # 视频处理节点
├── nodes_3d.py                    # 3D相关节点
├── nodes_hunyuan.py               # 混元模型节点
├── nodes_mochi.py                 # Mochi模型节点
├── nodes_cosmos.py                # Cosmos模型节点
└── chainner_models/               # ChaiNNer模型支持
```

### 3. comfy_api/ - API接口
```
comfy_api/
├── version_list.py          # API版本列表
├── feature_flags.py         # 功能标志
├── generate_api_stubs.py    # API存根生成
├── util.py                  # 工具函数
├── v0_0_1/                  # API v0.0.1版本
├── v0_0_2/                  # API v0.0.2版本
├── latest/                  # 最新API版本
├── input/                   # 输入处理
├── input_impl/              # 输入实现
├── internal/                # 内部API
├── torch_helpers/           # PyTorch辅助
└── util/                    # 工具模块
```

### 4. comfy_api_nodes/ - API节点
```
comfy_api_nodes/
├── __init__.py              # 模块初始化
├── apinode_utils.py         # API节点工具
├── mapper_utils.py          # 映射工具
├── canary.py                # 金丝雀测试
├── nodes_openai.py          # OpenAI API节点
├── nodes_stability.py       # Stability AI节点
├── nodes_runway.py          # Runway ML节点
├── nodes_bfl.py             # Black Forest Labs节点
├── nodes_gemini.py          # Google Gemini节点
├── nodes_ideogram.py        # Ideogram节点
├── nodes_kling.py           # Kling节点
├── nodes_luma.py            # Luma AI节点
├── nodes_minimax.py         # MiniMax节点
├── nodes_moonvalley.py      # MoonValley节点
├── nodes_pika.py            # Pika Labs节点
├── nodes_pixverse.py        # PixVerse节点
├── nodes_recraft.py         # Recraft节点
├── nodes_rodin.py           # Rodin节点
├── nodes_tripo.py           # Tripo节点
├── nodes_veo2.py            # Google Veo2节点
├── nodes_vidu.py            # Vidu节点
├── apis/                    # API定义
├── util/                    # 工具函数
├── redocly.yaml             # API文档配置
└── redocly-dev.yaml         # 开发环境配置
```

### 5. comfy_execution/ - 执行引擎
```
comfy_execution/
├── graph.py                 # 图结构处理
├── graph_utils.py           # 图工具函数
├── caching.py               # 缓存机制
├── progress.py              # 进度跟踪
├── utils.py                 # 工具函数
└── validation.py            # 验证逻辑
```

### 6. comfy_config/ - 配置管理
```
comfy_config/
├── config_parser.py         # 配置解析器
└── types.py                 # 配置类型定义
```

## 应用层目录

### 7. app/ - 应用程序层
```
app/
├── __init__.py              # 模块初始化
├── app_settings.py          # 应用设置
├── custom_node_manager.py   # 自定义节点管理
├── frontend_management.py   # 前端管理
├── logger.py                # 日志系统
├── model_manager.py         # 模型管理器
├── user_manager.py          # 用户管理
└── database/                # 数据库相关
```

### 8. api_server/ - API服务器
```
api_server/
├── __init__.py              # 模块初始化
├── routes/                  # 路由定义
├── services/                # 服务层
└── utils/                   # 工具函数
```

## 资源和数据目录

### 9. models/ - 模型存储
```
models/
├── checkpoints/             # 完整模型检查点
├── diffusion_models/        # 扩散模型（UNET）
├── clip/                    # CLIP文本编码器
├── clip_vision/             # CLIP视觉编码器
├── vae/                     # VAE编码器/解码器
├── loras/                   # LoRA权重文件
├── controlnet/              # ControlNet模型
├── upscale_models/          # 图像放大模型
├── embeddings/              # 文本嵌入
├── hypernetworks/           # Hypernetwork
├── style_models/            # 风格模型
├── gligen/                  # GLIGEN模型
├── photomaker/              # PhotoMaker模型
├── text_encoders/           # 文本编码器
├── unet/                    # UNET模型
├── diffusers/               # Diffusers格式模型
├── vae_approx/              # VAE近似模型
├── audio_encoders/          # 音频编码器
├── model_patches/           # 模型补丁
└── configs/                 # 模型配置文件
```

### 10. input/ - 输入文件
```
input/
├── example.png              # 示例图片
└── [用户上传的图片文件]
```

### 11. output/ - 输出文件
```
output/
├── _output_images_will_be_put_here  # 输出说明
└── [生成的图片文件]
```

### 12. custom_nodes/ - 自定义节点
```
custom_nodes/
├── example_node.py.example  # 节点示例
├── websocket_image_save.py  # WebSocket图片保存
└── [第三方自定义节点]
```

## 开发和测试目录

### 13. tests/ - 集成测试
```
tests/
├── __init__.py              # 测试模块初始化
├── README.md                # 测试说明
├── conftest.py              # 测试配置
├── compare/                 # 对比测试
└── inference/               # 推理测试
```

### 14. tests-unit/ - 单元测试
```
tests-unit/
├── README.md                # 单元测试说明
├── requirements.txt         # 测试依赖
├── app_test/                # 应用层测试
├── comfy_test/              # 核心模块测试
├── comfy_api_test/          # API测试
├── comfy_api_nodes_test/    # API节点测试
├── comfy_extras_test/       # 扩展节点测试
├── execution_test/          # 执行引擎测试
├── folder_paths_test/       # 路径管理测试
├── prompt_server_test/      # 提示服务器测试
├── server/                  # 服务器测试
├── utils/                   # 工具测试
├── feature_flags_test.py    # 功能标志测试
└── websocket_feature_flags_test.py  # WebSocket功能测试
```

### 15. script_examples/ - 脚本示例
```
script_examples/
├── basic_api_example.py     # 基础API示例
├── websockets_api_example.py  # WebSocket API示例
└── websockets_api_example_ws_images.py  # WebSocket图片示例
```

### 16. utils/ - 工具模块
```
utils/
├── __init__.py              # 模块初始化
├── extra_config.py          # 额外配置
├── install_util.py          # 安装工具
└── json_util.py             # JSON工具
```

### 17. docs/ - 文档目录
```
docs/
├── ComfyUI项目目录结构.md           # 本文档
├── ComfyUI架构详解.md               # 架构详解
├── ComfyUI安装指南.md               # 安装指南
├── ComfyUI介绍口播文稿.md           # 介绍文稿
├── 端口配置指南.md                  # 端口配置
├── AI模型文件分发策略差异分析.md     # 模型分发分析
├── CheckpointLoaderSimple节点分析.md # 节点分析文档
├── CLIPTextEncode节点详细分析.md     # 节点分析文档
├── EmptyLatentImage节点详细分析.md   # 节点分析文档
├── KSampler节点详细分析.md          # 节点分析文档
├── LoraLoaderModelOnly节点详细分析.md # 节点分析文档
├── UNETLoader节点详细分析.md        # 节点分析文档
└── VAEDecode节点详细分析.md         # 节点分析文档
```

## 数据库和配置

### 18. alembic_db/ - 数据库迁移
```
alembic_db/
├── README.md                # 数据库说明
├── env.py                   # 环境配置
└── script.py.mako           # 脚本模板
```

### 19. 配置文件
```
├── alembic.ini              # Alembic配置
├── pytest.ini              # Pytest配置
├── extra_model_paths.yaml.example  # 模型路径示例
├── comfyui_version.py       # 版本信息
├── cuda_malloc.py           # CUDA内存分配
├── hook_breaker_ac10a0.py   # 钩子断路器
├── latent_preview.py        # 潜在空间预览
├── new_updater.py           # 更新器
├── node_helpers.py          # 节点辅助函数
└── protocol.py              # 协议定义
```

## 目录结构特点

### 1. 模块化设计
- **核心功能**：comfy/ 包含所有核心AI功能
- **扩展功能**：comfy_extras/ 包含额外节点
- **API层**：comfy_api/ 和 comfy_api_nodes/ 提供API接口
- **应用层**：app/ 和 api_server/ 处理应用逻辑

### 2. 清晰的职责分离
- **模型管理**：models/ 目录统一管理所有AI模型
- **输入输出**：input/ 和 output/ 分别处理输入和输出文件
- **自定义扩展**：custom_nodes/ 支持第三方节点
- **测试覆盖**：tests/ 和 tests-unit/ 提供完整测试

### 3. 开发友好
- **文档完善**：docs/ 目录包含详细文档
- **示例丰富**：script_examples/ 提供使用示例
- **工具齐全**：utils/ 提供各种工具函数
- **配置灵活**：多种配置文件支持不同需求

### 4. 扩展性强
- **插件系统**：支持自定义节点和扩展
- **API接口**：提供完整的API访问能力
- **模型支持**：支持多种AI模型格式
- **平台兼容**：跨平台设计，支持多种部署方式

这个目录结构体现了ComfyUI作为一个成熟的AI图像生成平台的设计理念：模块化、可扩展、易维护。
