## 📌 问题现象

在 Windows 环境下使用 Docker Desktop（基于 WSL2）部署服务（例如 PostgreSQL 数据库），端口映射正常（如 `0.0.0.0:5433->5432`）。

* **可以访问**：使用本机回环地址 `127.0.0.1:5433` 连接正常。
* **无法访问**：使用本机的物理局域网 IP（例如 `192.168.1.6:5433`）连接时，一直处于转圈状态，最终报错 `Connection timed out`。局域网内的其他设备可能可以访问，但"本机访问本机的局域网 IP"会失败。

## 🔍 排查过程

遇到此类问题，通常首先排查以下两点，但如果均无问题，则极大概率是 WSL 网络模式导致的：

1. **Windows 防火墙**：确认是否已放行对应端口的入站规则（TCP）。
2. **Docker 端口映射**：确认 `docker ps` 中端口是否绑定到了 `0.0.0.0` 而非仅限 `127.0.0.1`。

## 💡 原因分析

在较新的 Windows 系统中，WSL2 引入了**镜像网络模式**（`networkingMode = mirrored`）。开启此模式后，WSL 会直接镜像 Windows 宿主机的网络接口，大大提升了网络兼容性。

但是，出于安全和路由机制的设计，**镜像网络模式默认禁止了"主机地址回环"（Host Address Loopback）**。这就导致了当 Windows 宿主机尝试通过自己的局域网 IP 访问自己时，数据包无法正确回环并转发给 WSL 内部的容器，从而导致连接超时。

## 🛠️ 解决方案

要解决这个问题，只需要在 WSL 的配置文件中显式开启"主机地址回环"功能即可。

### 步骤 1：修改 `.wslconfig` 文件

1. 使用记事本或任何文本编辑器打开 WSL 的配置文件，路径通常位于：`C:\Users\<你的用户名>\.wslconfig`。
2. 在文件中添加或修改 `[experimental]` 节点，配置 `hostAddressLoopback=true`。

完整的配置参考如下（如果你之前有其他配置，保留即可）：

```ini
[wsl2]
networkingMode = mirrored
autoProxy = true

[experimental]
hostAddressLoopback=true
```

### 步骤 2：重启 WSL 和 Docker 服务

配置文件修改保存后，必须彻底重启 WSL 才能生效。

1. 打开 PowerShell 或 CMD（不需要管理员权限），执行以下命令彻底关闭 WSL：

```powershell
wsl --shutdown
```

2. 等待几秒钟，随后**重启 Docker Desktop**。
3. 待 Docker 重新启动并拉起容器后，再次使用局域网 IP（如 `192.168.1.6`）连接数据库，即可成功连通。

## 📎 参考

- [共享宿主机的代理配置](./共享宿主机的代理配置.md)
- [启用root登录](./启用root登录.md)
