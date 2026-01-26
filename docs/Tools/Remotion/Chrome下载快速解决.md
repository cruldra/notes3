---
sidebar_position: 10
---
如果遇到 `ECONNRESET` 错误，使用以下命令手动下载：

```bash
# 1. 创建目录
mkdir -p "node_modules/.remotion/chrome-headless-shell/win64"

# 2. 下载 (使用代理)
curl -x http://127.0.0.1:7890 -L -o chrome-headless-shell-win64.zip \
  "https://storage.googleapis.com/chrome-for-testing-public/134.0.6998.35/win64/chrome-headless-shell-win64.zip" \
  --connect-timeout 30 --max-time 300

# 3. 解压
unzip -q chrome-headless-shell-win64.zip -d "node_modules/.remotion/chrome-headless-shell/win64/"

# 4. 验证
node_modules/.remotion/chrome-headless-shell/win64/chrome-headless-shell-win64/chrome-headless-shell.exe --version

# 5. 清理
rm chrome-headless-shell-win64.zip

# 6. 测试
npx remotion compositions
```

**预期输出**:
```
✅ Google Chrome for Testing 134.0.6998.35
✅ remotion-intro    30      1920x1080      9150 (305.00 sec)
```