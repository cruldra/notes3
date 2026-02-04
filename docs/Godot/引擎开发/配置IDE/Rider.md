---
sidebar_position: 1
---

# JetBrains Rider

[JetBrains Rider](https://www.jetbrains.com/rider/) 是 JetBrains 推出的一款用于 C++、C# 和 GDScript 的商业 IDE，它使用与 Visual Studio 相同的解决方案系统。

:::note 注意
本文档用于指导如何为游戏引擎本身的开发做贡献，**而不是**如何使用 JetBrains Rider 作为 C# 或 GDScript 编辑器。如果您需要使用外部编辑器编写 C# 或 GDScript 代码，请参阅 [配置外部编辑器指南](https://docs.godotengine.org/en/stable/tutorials/scripting/c_sharp/c_sharp_basics.html#doc-c-sharp-setup-external-editor)。
:::

## 导入项目

:::tip 提示
如果您已经使用 Visual Studio 作为主要 IDE，可以在 Rider 中使用相同的解决方案文件。Rider 和 Visual Studio 使用相同的解决方案格式，因此您可以在两个 IDE 之间切换而无需重新构建解决方案文件。但在从一个 IDE 切换到另一个 IDE 时，可能需要更改调试配置。
:::

如果您从零开始，请遵循 [编译指南](https://docs.godotengine.org/en/stable/contributing/development/compiling/index.html#doc-compiling-index)，具体步骤如下：

*   安装所有依赖项。
*   确定用于针对特定平台编译的 scons 命令。

为 scons 提供额外参数以生成解决方案文件：

*   在 scons 命令中添加 `vsproj=yes dev_build=yes`。

`vsproj` 参数指示生成 Visual Studio 解决方案。`dev_build` 参数确保包含调试符号，以便可以使用断点单步调试代码。

*   在 Rider 中打开生成的 `godot.sln`。

:::note 注意
确保在 Rider 工具栏中选择了合适的解决方案配置（Solution configuration）。它会影响 SDK 的解析、代码分析、构建、运行等。
:::

## 编译与调试项目

Rider 内置了调试器，可用于调试 Godot 项目。您可以点击屏幕顶部的 **Debug** 图标启动调试器，但这仅适用于项目管理器（Project Manager）。如果您想调试编辑器本身，需要先配置调试器。

![Rider 运行调试](https://docs.godotengine.org/en/stable/_images/rider_run_debug.webp)

*   点击屏幕顶部的 **Godot > Edit Configurations** 选项。

![Rider 配置](https://docs.godotengine.org/en/stable/_images/rider_configurations.webp)

*   确保 **C++ Project** 运行配置的值如下：

    > *   Exe Path : `$(LocalDebuggerCommand)`
    > *   Program Arguments: `-e --path <Godot项目路径>`
    > *   Working Directory: `$(LocalDebuggerWorkingDirectory)`
    > *   Before Launch: 设置为 "Build Project"

这将告诉可执行文件调试指定的项目，而不打开项目管理器。请使用项目文件夹的根路径，而不是 `project.godot` 文件路径。

![Rider 配置变更](https://docs.godotengine.org/en/stable/_images/rider_configurations_changed.webp)

*   最后点击 "Apply" 和 "OK" 保存更改。
*   当点击屏幕顶部的 **Debug** 图标时，JetBrains Rider 将启动带有调试器的 Godot 编辑器。

或者，您可以使用 **Run > Attach to Process** 将调试器附加到正在运行的 Godot 实例。

![Rider 附加到进程](https://docs.godotengine.org/en/stable/_images/rider_attach_to_process.webp)

*   您可以搜索 `godot.editor` 找到 Godot 实例，然后点击 `Attach with LLDB`。

![Rider 附加到进程对话框](https://docs.godotengine.org/en/stable/_images/rider_attach_to_process_dialog.webp)

## 调试可视化

调试可视化器（Debug visualizers）用于自定义调试期间复杂数据结构的显示方式。对于 Windows，Godot 内置的 "natvis"（"Native Visualization" 的缩写）会自动被使用。对于其他操作系统，可以手动设置类似的功能。

请关注 [RIDER-123535](https://youtrack.jetbrains.com/issue/RIDER-123535/nix-Debug-Godot-Cpp-from-Rider-pretty-printers-usability)。

## 单元测试

利用 Rider 的 [doctest](https://docs.godotengine.org/en/stable/contributing/architecture/unit_testing.html#doc-unit-testing) 支持。请参阅 [相关说明](https://github.com/JetBrains/godot-support/wiki/Godot-doctest-Unit-Tests)。

## 性能分析

请参阅 [使用 dotTrace 或 JetBrains Rider 分析 Godot 引擎（本机代码）的说明](https://github.com/JetBrains/godot-support/wiki/Profiling-Godot-engine-(native-code)-with-dotTrace-or-JetBrains-Rider)。

有关 JetBrains IDE 的具体信息，请查阅 [JetBrains Rider 文档](https://www.jetbrains.com/rider/documentation/)。

## 已知问题

调试 Windows MinGV 构建版本时符号未加载。已报告 [RIDER-106816](https://youtrack.jetbrains.com/issue/RIDER-106816/Upgrade-LLDB-to-actual-version)。
