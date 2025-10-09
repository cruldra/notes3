# vLLM 学习资料

本目录包含了关于 vLLM（高性能大语言模型推理引擎）的完整学习资料。

## 📚 文档目录

### 1. [vLLM 完整指南](./vLLM完整指南.md)

**适合人群**：初学者、想要全面了解 vLLM 的开发者

**内容概要**：
- vLLM 是什么？核心特点和官方资源
- vLLM vs Ollama 详细对比分析
- 核心技术 PagedAttention 简介
- 安装部署方法（pip、Docker）
- 基本使用方法（Python API、OpenAI 兼容 API）
- 性能优化关键参数
- 常见模型显存占用参考
- 技术栈关系图
- 使用场景选择建议

**推荐阅读顺序**：⭐ 第一篇必读

---

### 2. [PagedAttention 技术详解](./PagedAttention技术详解.md)

**适合人群**：想要深入理解 vLLM 核心技术的开发者、研究人员

**内容概要**：
- KV Cache 的背景和挑战
  - 显存占用增长快
  - 内存碎片严重
  - 缓存难以复用
- PagedAttention 的解决方案
  - 分页管理机制
  - 内存共享技术
  - 写时复制（Copy-on-Write）
- 技术优势和性能对比
- 实现细节（页表结构、物理页池、注意力计算）
- 适用场景分析
- 与操作系统分页的类比

**推荐阅读顺序**：⭐⭐ 第二篇，深入理解核心技术

---

### 3. [vLLM 实战部署指南](./vLLM实战部署指南.md)

**适合人群**：需要实际部署 vLLM 服务的开发者、运维人员

**内容概要**：
- 环境准备（硬件、软件要求）
- 三种部署方式详解
  - 方式一：使用 pip 安装
  - 方式二：使用 Docker 部署
  - 方式三：多卡部署（张量并行、流水线并行）
- 高级配置
  - 性能优化参数
  - 长文本支持
  - 量化支持
  - 监控配置
- 使用示例（Python 客户端、curl）
- 生产环境部署建议
  - systemd 服务管理
  - Nginx 反向代理
  - 监控和日志
- 常见问题排查

**推荐阅读顺序**：⭐⭐⭐ 第三篇，实战必读

---

### 4. [vLLM 性能调优指南](./vLLM性能调优指南.md)

**适合人群**：需要优化 vLLM 性能的开发者、性能工程师

**内容概要**：
- 性能指标详解
  - 吞吐量（Throughput）
  - 延迟（Latency）
  - 显存利用率
  - GPU 利用率
- 核心参数调优
  - GPU 内存利用率
  - 最大模型长度
  - 最大并发序列数
  - 最大批处理 Token 数
  - 张量并行大小
  - 数据类型选择
- 高级优化技巧
  - 分块预填充
  - 禁用日志请求
  - 使用量化模型
  - 调整采样参数
- 性能监控（Prometheus + Grafana）
- 性能测试方法
- 调优流程和最佳实践
- 常见性能问题及解决方案

**推荐阅读顺序**：⭐⭐⭐⭐ 第四篇，性能优化必读

---

### 5. [安装常见错误及解决方案](./安装常见错误及解决方案.md)

**适合人群**：遇到安装问题的开发者

**内容概要**：
- 常见安装错误
- 解决方案和排查步骤

**推荐阅读顺序**：遇到问题时查阅

---

## 🎯 学习路径

### 初学者路径

1. **第一步**：阅读 [vLLM 完整指南](./vLLM完整指南.md)
   - 了解 vLLM 是什么
   - 理解 vLLM vs Ollama 的区别
   - 掌握基本使用方法

2. **第二步**：阅读 [vLLM 实战部署指南](./vLLM实战部署指南.md)
   - 选择合适的部署方式
   - 跟随指南完成部署
   - 测试基本功能

3. **第三步**：阅读 [PagedAttention 技术详解](./PagedAttention技术详解.md)
   - 深入理解核心技术
   - 了解为什么 vLLM 性能优秀

4. **第四步**：阅读 [vLLM 性能调优指南](./vLLM性能调优指南.md)
   - 根据实际需求优化性能
   - 监控和调优服务

### 进阶开发者路径

1. **快速上手**：
   - [vLLM 完整指南](./vLLM完整指南.md) - 快速浏览
   - [vLLM 实战部署指南](./vLLM实战部署指南.md) - 选择合适的部署方式

2. **深入理解**：
   - [PagedAttention 技术详解](./PagedAttention技术详解.md) - 深入学习核心技术

3. **性能优化**：
   - [vLLM 性能调优指南](./vLLM性能调优指南.md) - 系统性能优化

### 运维人员路径

1. **部署**：
   - [vLLM 实战部署指南](./vLLM实战部署指南.md) - 重点关注生产环境部署

2. **监控**：
   - [vLLM 性能调优指南](./vLLM性能调优指南.md) - 重点关注监控部分

3. **故障排查**：
   - [安装常见错误及解决方案](./安装常见错误及解决方案.md)
   - [vLLM 实战部署指南](./vLLM实战部署指南.md) - 常见问题排查部分

---

## 🔗 相关资源

### 官方资源

- **官方文档（中文）**：https://vllm.hyper.ai/docs/
- **官方文档（英文）**：https://docs.vllm.ai/en/latest/index.html
- **GitHub 仓库**：https://github.com/vllm-project/vllm
- **论文**：[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

### 社区资源

- **Discord**：https://discord.gg/vllm
- **GitHub Discussions**：https://github.com/vllm-project/vllm/discussions
- **GitHub Issues**：https://github.com/vllm-project/vllm/issues

### 相关工具

- **ModelScope**：https://modelscope.cn/ - 国内模型下载
- **HuggingFace**：https://huggingface.co/ - 国际模型下载
- **Prometheus**：https://prometheus.io/ - 监控工具
- **Grafana**：https://grafana.com/ - 可视化工具

---

## 📊 快速参考

### 常用命令

```bash
# 安装 vLLM
pip install vllm

# 启动服务（基本）
vllm serve /path/to/model

# 启动服务（完整配置）
vllm serve /path/to/model \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 4096 \
  --api-key your-secret-key

# Docker 部署
docker run -d \
  --name vllm-service \
  --gpus all \
  -v /path/to/models:/models \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model /models/your-model
```

### 推荐配置

**通用场景**：

```bash
--dtype bfloat16
--gpu-memory-utilization 0.9
--max-model-len 4096
--max-num-seqs 256
--max-num-batched-tokens 8192
--enable-chunked-prefill
```

**低延迟场景**：

```bash
--dtype bfloat16
--gpu-memory-utilization 0.85
--max-model-len 2048
--max-num-seqs 128
--max-num-batched-tokens 2048
```

**高吞吐场景**：

```bash
--dtype bfloat16
--gpu-memory-utilization 0.95
--max-model-len 8192
--max-num-seqs 512
--max-num-batched-tokens 16384
--enable-chunked-prefill
```

---

## 🤝 贡献

如果您发现文档中有错误或需要补充的内容，欢迎提出 Issue 或 Pull Request。

---

## 📝 更新日志

- **2025-01-09**：创建 vLLM 学习资料目录
  - 添加 vLLM 完整指南
  - 添加 PagedAttention 技术详解
  - 添加 vLLM 实战部署指南
  - 添加 vLLM 性能调优指南
  - 添加 README 索引文件

---

## 📄 许可证

本文档基于网络公开资料整理，仅供学习参考使用。

---

**祝您学习愉快！🎉**

