---
sidebar_position: 8
---

Remotion 会自动将 "Chrome Headless Shell" 安装到您的 `node_modules` 中，以便渲染视频。

## 支持的平台

支持以下平台：

- macOS (x64 和 arm64)
- Windows (x64)
- Linux x64 - [安装 Linux 依赖](https://www.remotion.dev/docs/miscellaneous/linux-dependencies)
- Linux arm64 - 仅支持 "Chrome Headless Shell"

## 确保 Chrome 已安装

有两种方式确保 Chrome Headless Shell 已安装：

- 在命令行使用 [`npx remotion browser ensure`](https://www.remotion.dev/docs/cli/browser/ensure)
- 作为 Node.js / Bun API 使用 [`ensureBrowser()`](https://www.remotion.dev/docs/renderer/ensure-browser)

如果您进行服务端渲染，建议调用这些函数。这样，当有请求需要渲染视频时，浏览器已经下载好并准备就绪。

## 使用 Chrome for Testing 替代

Remotion 可以使用两种模式来渲染视频：

- **Chrome Headless Shell**：在 CPU 密集型视频渲染中更快，需要的依赖和设置更少。
- **Chrome for Testing**：在 GPU 密集型视频渲染中更快。模拟显示表面，因此需要更多依赖和资源。

只有当您想在 Linux 上设置 [GPU 加速渲染环境](https://www.remotion.dev/docs/miscellaneous/cloud-gpu)时，才应使用 Chrome for Testing。

要使用 Chrome for Testing：

**在 CLI 中**：向 `npx remotion render`、`npx remotion benchmark`、`npx remotion compositions`、`npx remotion still`、`npx remotion gpu` 和 `npx remotion browser ensure` 传递 `--chrome-mode="chrome-for-testing"`。

**通过 API**：在 `renderFrames()`、`renderMedia()`、`renderStill()`、`selectComposition()`、`getComposition()` 和 `ensureBrowser()` 调用中传递 `chromeMode: 'chrome-for-testing'`。

**在 Studio 中**：在 "Advanced" 选项卡的 "Chrome Mode" 下拉菜单中选择 "Chrome for Testing"。

**在配置文件中（仅适用于 CLI 和 Studio）**：在 `remotion.config.ts` 文件中设置 `Config.setChromeMode('chrome-for-testing')`。

**在 Lambda 和 Cloud Run 中**：不适用，因为不支持 Chrome for Testing。

## 下载位置

Chrome Headless Shell 将下载到此文件夹：

```
node_modules/.remotion/chrome-headless-shell/[platform]/chrome-headless-shell-[platform]
```

将创建可执行文件 `./chrome-headless-shell`（在 Windows 上为 `.\\chrome-headless-shell.exe`）。

---

Chrome for Testing 将下载到此文件夹：

```
node_modules/.remotion/chrome-for-testing/[platform]
```

此文件夹内的文件结构因操作系统而异。

---

`platform` 可以是 `mac-arm64`、`mac-x64`、`linux64`、`linux-arm64` 或 `win64` 之一。

## 使用自己的二进制文件

如果您不想安装 Chrome Headless Shell 或 Chrome for Testing，或者您的平台不受支持，您需要指定自己的基于 Chromium 的浏览器：

- 在配置文件中使用 [`setBrowserExecutable()`](https://www.remotion.dev/docs/config#setbrowserexecutable) 选项（用于 CLI）
- 在 [`renderMedia()`](https://www.remotion.dev/docs/renderer/render-media) 和其他 SSR API 中使用 [`browserExecutable`](https://www.remotion.dev/docs/renderer/render-media) 选项

在 [Lambda](https://www.remotion.dev/docs/lambda) 和 [Cloud Run](https://www.remotion.dev/docs/cloudrun) 中，已经安装了 Chrome 版本，因此您无需执行任何操作。

:::note
在未来的 Chrome 版本中，桌面浏览器中的无头模式将停止支持，您将需要使用 Chrome Headless Shell。
:::

## 为什么 Remotion 要管理 Chrome？

Remotion 之前使用许多用户已经安装的桌面版 Chrome。此工作流程在某个时候中断了，因为 Chrome 移除了无头模式并将其提取到 "Chrome Headless Shell" 中。

## 最佳实践

为确保您的项目不会因即将到来的 Chrome 更改而中断，您应该使用 Remotion 机制来使用并固定 Chrome Headless Shell 的版本。

- 使用 Remotion v4.0.208 或更高版本，不使用外部安装的浏览器。
- 使用 [`npx remotion browser ensure`](https://www.remotion.dev/docs/cli/browser/ensure) 确保 Chrome Headless Shell 可用。
- 不要在 Dockerfile 中下载 Chrome，但如果使用 Linux，请安装 [Linux 依赖](https://www.remotion.dev/docs/miscellaneous/linux-dependencies)。
- 不要使用 `--browser-executable`、`browserExecutable` 或 `setBrowserExecutable()` 选项用不兼容的 Chrome 版本覆盖 Headless Shell。

:::warning
注意：大多数 Linux 发行版不允许您固定 Chrome 包。如果您使用低于 v4.0.208 的 Remotion 版本，您面临 Chrome 自动升级到不附带无头模式的版本的风险。
:::

## 什么是 Chrome Headless Shell？

Chrome 过去附带 `--headless` 标志，Remotion 会使用它。

从 Chrome 123 开始，无头模式分为：

- `--headless=old`，适合截图（因此适合 Remotion）
- `--headless=new`，适合浏览器测试

`--headless=old` 将在未来版本的 Chrome 中停止工作。旧的无头模式正在被提取到 "Chrome Headless Shell" 中。

因此，我们鼓励您使用 Chrome Headless Shell 来为您的 Remotion 应用程序做好准备。

## 版本

Remotion 将下载经过充分测试的 Chrome 版本：

| Remotion 版本 | Chrome 版本 |
|--------------|-------------|
| 从 4.0.315 开始 | 134.0.6998.35 |
| 从 4.0.274 开始 | 133.0.6943.141 |
| 从 4.0.245 开始 | 123.0.6312.86 |

:::note
在 Lambda 上，升级到 134.0.6998.35 尚未可用。
:::

升级可能在补丁版本中发生，并将在此处列出。

## 更改 Chrome 版本

在 Lambda 上，无法更改 Chrome 版本。

如果您使用 [服务端渲染 API](https://www.remotion.dev/docs/ssr)，请使用 [`ensureBrowser()`](https://www.remotion.dev/docs/renderer/ensure-browser#onbrowserdownload) 中的回调来更改 Chrome 版本。

## 在 Lambda 和 Cloud Run 上

如果您使用 [Remotion Lambda](https://www.remotion.dev/docs/lambda) 或 [Cloud Run](https://www.remotion.dev/docs/cloudrun)，您无需担心安装浏览器 - 它已包含在运行时中。

## 以前的更改

### 添加了安装 Chrome for Testing 的选项 (v4.0.247)

为了在 Linux 上启用 GPU 加速工作负载，Remotion 现在允许您安装 Chrome for Testing。

### 迁移到 Chrome Headless Shell

由于 Chrome 从桌面浏览器中移除了无头模式，Remotion 现在使用 Chrome Headless Shell。

### Thorium (v4.0.18 - v4.0.135)

在这些版本中，如果找不到本地浏览器，将下载 [Thorium](https://thorium.rocks/) 实例。

Thorium 是从 Chromium 分叉的免费开源浏览器，其中包括渲染视频所需的编解码器。

### Chromium (v4.0.18 之前)

在以前的版本中，Remotion 会下载免费版本的 Chromium，其中不包括专有 H.264 和 H.265 编解码器的编解码器。这通常会在使用 [`<Html5Video>`](https://www.remotion.dev/docs/html5-video) 标签时导致问题。

## 另请参阅

- [`ensureBrowser()`](https://www.remotion.dev/docs/renderer/ensure-browser)
- [媒体播放错误](https://www.remotion.dev/docs/media-playback-error)
