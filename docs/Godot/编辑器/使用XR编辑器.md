---
sidebar_position: 7
---

在 2024 年，我们推出了 [Godot XR 编辑器](https://godotengine.org/article/godot-editor-horizon-store-early-access-release/)，这是 Godot 编辑器的一个版本，**旨在原生运行于 XR 设备上**，支持直接在设备上创建、开发和导出 2D、3D 以及 **XR** 应用程序和游戏。

该应用可以从 [Google Play 商店](https://play.google.com/store/apps/details?id=org.godotengine.editor.v4)、[Meta Horizon 商店](https://www.meta.com/experiences/godot-game-engine/7713660705416473/) 或 [Godot 下载页面](https://godotengine.org/download/preview/) 下载。

> **注意**
>
> XR 编辑器目前处于早期接入（early access）阶段，我们仍在不断优化体验。请参阅下文的 [限制与已知问题](#限制与已知问题)。

## XR 设备支持 (XR devices support)

目前，Godot XR 编辑器仅适用于 Android XR 设备，以及运行 **Meta Horizon OS v69 或更高版本** 的以下 [Meta Quest](https://www.meta.com/quest/) 设备：

*   Meta Quest 2
*   Meta Quest 3
*   Meta Quest 3s
*   Meta Quest Pro

> **注意**
>
> 我们正在努力增加对更多 XR 设备的支持，包括 PCVR 设备。

## 运行时权限 (Runtime Permissions)

*   **[所有文件访问权限 (All files access permission)](https://developer.android.com/training/data-storage/manage-all-files#all-files-access)**：允许编辑器在设备上的任何位置创建、导入和读取项目文件。如果没有此权限，编辑器虽然仍能运行，但对设备文件和目录的访问将受到限制。
*   **[REQUEST_INSTALL_PACKAGES](https://developer.android.com/reference/android/Manifest.permission#REQUEST_INSTALL_PACKAGES)**：允许编辑器安装导出的项目 APK。
*   **[RECORD_AUDIO](https://developer.android.com/reference/android/Manifest.permission#RECORD_AUDIO)**：当启用 [audio/driver/enable_input](https://docs.godotengine.org/en/stable/classes/class_projectsettings.html#class-projectsettings-property-audio-driver-enable-input) 项目设置时会请求此权限。
*   **[USE_SCENE (仅限 META)](https://developers.meta.com/horizon/documentation/native/native-spatial-data-perm/)**：运行 XR 项目时，启用并访问场景 API 所需。

## 技巧与建议 (Tips & Tricks)

### 输入 (Input)

*   为了获得最佳体验和高生产力，建议连接 **蓝牙键盘和鼠标** 与 XR 编辑器进行交互。XR 编辑器支持所有 [常用的快捷键和键位映射](https://docs.godotengine.org/en/stable/tutorials/editor/default_key_mapping.html)。
*   使用追踪控制器或追踪手势进行交互时，可以开启 `interface/touchscreen/enable_long_press_as_right_click` 编辑器设置，以实现 **长按触发右键**。
*   使用追踪控制器或追踪手势时，可以通过 `interface/touchscreen/increase_scrollbar_touch_area` 编辑器设置来 **增加滚动条的触摸区域**。

### Quest 上的多任务处理 (Multi-tasking on Quest)

*   **[剧院视图 (Theater View)](https://www.meta.com/blog/quest/meta-quest-v67-update-new-window-layout-creator-content-horizon-feed/)** 可用于将 *编辑器窗口* 全屏显示。
*   在 Quest 的 *实验性设置* 中启用 **[无缝多任务处理 (Seamless Multitasking)](https://www.uploadvr.com/seamless-multitasking-experimental-quest/)**，可以实现正在运行的 XR 项目与 *编辑器窗口* 之间的快速切换。
*   开发非 XR 项目时，通过使用 Quest 的 *应用菜单 (App menu)* 功能，Godot 编辑器的应用图标可以在 *编辑器窗口* 和 *运行窗口*（激活时）之间切换。
*   在开发和运行 XR 项目时，你可以通过以下方式找回 *编辑器窗口*：
    *   按下 *Meta* 键唤出菜单栏。
    *   点击 Godot 编辑器应用图标召唤 *应用菜单*，然后选择 *编辑器窗口 (Editor window)* 磁贴。

### 项目同步 (Projects sync)

*   通过 Git 同步项目可以通过下载 Android 版 Git 客户端完成。我们推荐使用 **[Termux 终端](https://termux.dev/en/)**，这是一个 Android 终端模拟器，提供了对 Git 和 SSH 等常用终端工具的访问。
    *   **注意**：要在 Termux 终端中使用 Git，你需要授予终端 *写入 (WRITE)* 权限。可以通过在终端内运行以下命令来完成：`termux-setup-storage`

### 插件 (Plugins)

*   GDExtension 插件可以按预期工作，但需要插件开发者提供原生 Android 二进制文件。

## 限制与已知问题 (Limitations & known issues)

以下是 XR 编辑器的已知限制和问题：

*   不支持 C#/Mono。
*   不支持外部脚本编辑器。
*   虽然可以使用，但不建议使用 *Vulkan Forward+* 渲染器，因为存在严重的性能问题。
