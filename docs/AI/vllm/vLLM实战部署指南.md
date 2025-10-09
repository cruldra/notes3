# vLLM 实战部署指南

## 环境准备

### 硬件要求

- **GPU**：NVIDIA GPU with Compute Capability 7.0+ (V100, A100, H100, RTX 3090/4090 等)
- **显存**：根据模型大小确定
  - 7B 模型：≥16GB
  - 13B 模型：≥24GB
  - 32B 模型：≥64GB
  - 70B 模型：≥140GB (多卡)
- **内存**：建议 ≥32GB
- **存储**：SSD，根据模型大小预留空间

### 软件要求

- **操作系统**：Linux (Ubuntu 20.04/22.04 推荐)
- **Python**：3.8-3.11
- **CUDA**：11.8 或 12.1+
- **Docker**：20.10+ (可选)
- **NVIDIA Driver**：≥525.60.13

### 检查环境

```bash
# 检查 GPU
nvidia-smi

# 检查 CUDA 版本
nvcc --version

# 检查 Python 版本
python --version

# 检查 Docker 版本
docker --version
```

## 方式一：使用 pip 安装

### 1. 创建虚拟环境

```bash
# 使用 conda
conda create -n vllm python=3.10
conda activate vllm

# 或使用 venv
python -m venv vllm-env
source vllm-env/bin/activate
```

### 2. 安装 vLLM

```bash
# 安装稳定版本
pip install vllm

# 或安装最新开发版本
pip install git+https://github.com/vllm-project/vllm.git

# 验证安装
python -c "import vllm; print(vllm.__version__)"
```

### 3. 下载模型

使用 ModelScope 下载模型（国内推荐）：

```bash
# 安装 ModelScope
pip install modelscope

# 下载模型
python -c "
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='/data/models')
print(f'Model downloaded to: {model_dir}')
"
```

或使用 HuggingFace：

```bash
# 安装 huggingface-cli
pip install huggingface-hub

# 下载模型
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir /data/models/Qwen2.5-7B-Instruct
```

### 4. 启动服务

```bash
# 启动 OpenAI 兼容服务器
vllm serve /data/models/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --api-key your-secret-key
```

### 5. 测试服务

```bash
# 测试健康检查
curl http://localhost:8000/health

# 测试补全接口
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{
    "model": "/data/models/Qwen2.5-7B-Instruct",
    "prompt": "你好，请介绍一下北京",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## 方式二：使用 Docker 部署

### 1. 安装 Docker 和 NVIDIA Container Toolkit

```bash
# 安装 Docker（如果未安装）
curl -fsSL https://get.docker.com | bash

# 安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 配置 Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 测试 GPU 访问
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 2. 拉取 vLLM 镜像

```bash
# 拉取官方镜像
docker pull vllm/vllm-openai:latest

# 查看镜像
docker images | grep vllm
```

### 3. 准备模型

```bash
# 创建模型目录
mkdir -p /data/models

# 下载模型（使用 ModelScope）
docker run --rm -v /data/models:/models \
  python:3.10 bash -c "
  pip install modelscope && \
  python -c \"
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='/models')
  \"
"
```

### 4. 启动容器

```bash
docker run -d \
  --name vllm-service \
  --restart=always \
  --gpus all \
  --ipc=host \
  -v /data/models:/models \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model /models/Qwen/Qwen2___5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --api-key your-secret-key
```

### 5. 管理容器

```bash
# 查看日志
docker logs -f vllm-service

# 查看容器状态
docker ps | grep vllm

# 停止容器
docker stop vllm-service

# 启动容器
docker start vllm-service

# 重启容器
docker restart vllm-service

# 进入容器
docker exec -it vllm-service bash

# 删除容器
docker rm -f vllm-service
```

## 方式三：多卡部署

### 1. 张量并行（Tensor Parallelism）

适用于单个模型太大，需要分布到多张 GPU 上：

```bash
# 使用 4 张 GPU
vllm serve /data/models/Qwen2.5-70B-Instruct \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9
```

Docker 方式：

```bash
docker run -d \
  --name vllm-multi-gpu \
  --restart=always \
  --gpus all \
  --ipc=host \
  -v /data/models:/models \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model /models/Qwen2.5-70B-Instruct \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9
```

### 2. 流水线并行（Pipeline Parallelism）

适用于超大模型：

```bash
vllm serve /data/models/DeepSeek-R1-671B \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 4 \
  --dtype bfloat16
```

## 高级配置

### 1. 性能优化参数

```bash
vllm serve /data/models/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 256 \
  --enable-chunked-prefill \
  --disable-log-requests
```

参数说明：

- `--max-num-batched-tokens`：批处理的最大 token 数
- `--max-num-seqs`：最大并发序列数
- `--enable-chunked-prefill`：启用分块预填充
- `--disable-log-requests`：禁用请求日志（提升性能）

### 2. 长文本支持

```bash
# 使用 YaRN 扩展上下文长度
vllm serve /data/models/Qwen2.5-7B-Instruct \
  --max-model-len 32768 \
  --rope-scaling '{"type": "yarn", "factor": 4.0}'
```

### 3. 量化支持

```bash
# 使用 AWQ 量化
vllm serve /data/models/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq

# 使用 GPTQ 量化
vllm serve /data/models/Qwen2.5-7B-Instruct-GPTQ \
  --quantization gptq
```

### 4. 监控配置

```bash
# 启用 Prometheus 监控
vllm serve /data/models/Qwen2.5-7B-Instruct \
  --disable-log-stats \
  --enable-metrics
```

## 使用示例

### Python 客户端

```python
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-secret-key"
)

# 聊天补全
response = client.chat.completions.create(
    model="/data/models/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "介绍一下北京的著名景点"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)

# 流式输出
stream = client.chat.completions.create(
    model="/data/models/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "写一首关于春天的诗"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### curl 示例

```bash
# 聊天补全
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{
    "model": "/data/models/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "system", "content": "你是一个有帮助的助手。"},
      {"role": "user", "content": "介绍一下北京的著名景点"}
    ],
    "temperature": 0.7,
    "max_tokens": 500
  }'

# 流式输出
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{
    "model": "/data/models/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "user", "content": "写一首关于春天的诗"}
    ],
    "stream": true
  }'
```

## 生产环境部署建议

### 1. 使用 systemd 管理服务

创建 `/etc/systemd/system/vllm.service`：

```ini
[Unit]
Description=vLLM Service
After=network.target

[Service]
Type=simple
User=vllm
WorkingDirectory=/opt/vllm
ExecStart=/opt/vllm/venv/bin/vllm serve /data/models/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --api-key your-secret-key
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable vllm
sudo systemctl start vllm
sudo systemctl status vllm
```

### 2. 使用 Nginx 反向代理

```nginx
upstream vllm_backend {
    server localhost:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://vllm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 流式输出支持
        proxy_buffering off;
        proxy_cache off;
    }
}
```

### 3. 监控和日志

使用 Prometheus + Grafana 监控：

```bash
# 启动 vLLM 并启用监控
vllm serve /data/models/Qwen2.5-7B-Instruct \
  --enable-metrics \
  --metrics-port 9090
```

## 常见问题排查

### 1. CUDA Out of Memory

**解决方案**：
- 降低 `--gpu-memory-utilization`（如 0.8）
- 减小 `--max-model-len`
- 减小 `--max-num-seqs`
- 使用量化模型

### 2. 推理速度慢

**解决方案**：
- 增加 `--gpu-memory-utilization`
- 启用 `--enable-chunked-prefill`
- 调整 `--max-num-batched-tokens`
- 使用多卡并行

### 3. 服务启动失败

**检查步骤**：
```bash
# 检查 GPU 可用性
nvidia-smi

# 检查 CUDA 版本
nvcc --version

# 检查 vLLM 安装
python -c "import vllm; print(vllm.__version__)"

# 查看详细日志
vllm serve /data/models/Qwen2.5-7B-Instruct --log-level debug
```

## 总结

本指南涵盖了 vLLM 的三种主要部署方式：

1. **pip 安装**：适合开发和测试
2. **Docker 部署**：适合生产环境，易于管理
3. **多卡部署**：适合大模型和高性能需求

选择合适的部署方式，根据实际需求调整参数，即可快速搭建高性能的大语言模型推理服务。

