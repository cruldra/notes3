---
sidebar_position: 10
---

*最后修改日期：2025年11月12日*

Rider 为 GDScript 提供了全面的支持，包括原生的 GDScript 支持以及 LSP 集成。这为代码分析、导航、类型提示和符号解析等功能提供了高精度和高覆盖率。LSP（语言服务器）和 DAP（调试适配器协议）均会自动配置。

JetBrains Rider 著名的 [代码分析](https://www.jetbrains.com/help/rider/Code_Analysis__Index.html)、[编码辅助](https://www.jetbrains.com/help/rider/Coding_Assistance__Index.html) 和 [代码导航](https://www.jetbrains.com/help/rider/Navigation_and_Search__Index.html) 功能在 Godot 项目的 C# 和 GDScript 代码中均可使用。

Godot 支持基于 Rider 捆绑的两个开源插件：[Godot Support](https://github.com/JetBrains/godot-support)（负责 Godot 的整体功能）和 [GdScript](https://github.com/JetBrains/godot-support/tree/master/gdscript)（提供 [GDScript 语言的原生支持](#working-with-gdscript-code)）。

## 快速开始

只要您的机器上安装了 [Godot 引擎](https://godotengine.org/download/windows/)，JetBrains Rider 就已准备好处理 Godot 项目。

可以通过以下方式在 JetBrains Rider 中打开 Godot 项目：

*   如果项目仅使用 GDScript，请从主菜单选择 **File | Open | Open**，选择项目文件夹，然后点击 **Select Folder**。
*   如果项目使用 C#，请通过 **File | Open | Open** 打开位于项目文件夹中的 `.sln` 解决方案文件。
*   要将 JetBrains Rider 设置为 Godot 编辑器中 C# 脚本的默认编辑器，请在 Godot 编辑器菜单中导航至 **Editor | Editor Settings | Dotnet | Editor**，并在 **External Editor** 下拉菜单中选择 **JetBrains Rider**。
*   同样，您可以在 **Text Editor | External** 设置页面将 JetBrains Rider 设置为默认的 GDScript 编辑器。

    要查找 JetBrains Rider 可执行文件的位置，请切换到 Rider，转到 **Help | Diagnostic Tools | Special Files and Folders**，并查找 **Installation home directory**。在 Windows 上，您需要指向 `bin/Rider64.exe`。在 macOS 上，您需要指向 Rider 的 `.app` 文件夹。

    对于较旧的 Godot 编辑器版本，您可能需要在 **Exec Flags** 中指定以下内容：`{project} --line {line} {file}`。

## 为 Rider 优化 Godot 编辑器

以下是配置 Godot 编辑器以实现与作为外部编辑器的 Rider 最佳集成的推荐设置。要访问这些设置，请确保在 **Editor Settings** 对话框顶部启用了 **Advanced Settings** 开关。

*   **Editor Settings | Text Editor | Behavior | Auto Reload Scripts on External Change**
*   **Editor Settings | Interface | Editor | Save on Focus Loss**
*   **Editor Settings | Interface | Editor | Import Resources When Unfocused**

## 使用 GDScript 代码

JetBrains Rider 可以通过两种不同的方式分析 GDScript 文件：

*   通过 LSP 连接到 Godot 语言服务（通常是打开了同一项目的 Godot 编辑器）。当您安装 [Godot 引擎](https://godotengine.org/download/windows/) 时，Godot 语言服务即可在您的机器上使用。
*   使用捆绑的 GdScript 插件提供的原生 Godot 解析器。

当 Rider 连接到 Godot LSP 服务器时，状态栏上会出现一个连接指示器。如果 LSP 连接成功，它将被用于分析，并且通过插件进行的原生分析将被禁用。

![JetBrains Rider: 状态栏上的 Godot LSP 指示器](https://resources.jetbrains.com/help/img/rider/2025.3/gdscript_status_bar_indicator.png)

默认情况下，JetBrains Rider 会与正在运行的 Godot 编辑器建立 LSP 连接。但是，如果您不想让 Godot 编辑器运行，您可以让 Rider 在需要时自动在后台启动 Godot LSP 服务。为此，请在 **Languages & Frameworks | Godot Engine** 设置页面的 **LSP server connection:** 中选择 **Automatically start headless LSP server**。

### GDScript 代码补全

JetBrains Rider 为变量、常量、方法、信号、枚举、类、注解、节点、输入、组、元字段和资源提供代码补全：

*   自动补全
*   中间名匹配（Middle-name-matching）
*   节点
*   字符串中的输入、组、资源

![JetBrains Rider: GDScript 补全](https://resources.jetbrains.com/help/img/rider/2025.3/gdscript_autocompletion.png)
![JetBrains Rider: GDScript 补全](https://resources.jetbrains.com/help/img/rider/2025.3/gdscript_autocompletion_functions.png)
![JetBrains Rider: GDScript 补全](https://resources.jetbrains.com/help/img/rider/2025.3/gdscript_autocompletion_nodes.png)
![JetBrains Rider: GDScript 补全](https://resources.jetbrains.com/help/img/rider/2025.3/gdscript_string_completion.png)

默认情况下，以下划线开头的变量和方法不会出现在补全列表中。如果您想更改此行为，请在 JetBrains Rider 设置的 **Languages & Frameworks | GDScript** 页面上取消选中 **Hide _private members from completion** 复选框。

### GDScript 跳转到声明/用法

您可以使用 **Go to Declaration**（或 Ctrl-点击）跳转到符号和文件资源的用法。

如果您在符号的声明处，**Find Usages** 将帮助您找到其所有引用。

![JetBrains Rider: GDScript 查找用法](https://resources.jetbrains.com/help/img/rider/2025.3/gdscript_usages.png)

> **注意**
>
> 当前版本的 GDScript 不支持 **Show Usages** 命令。

### GDScript 文件模板

向项目添加新的 GDScript 文件时，可以使用从 Godot 源代码导入的文件模板：

![JetBrains Rider: GDScript 文件模板](https://resources.jetbrains.com/help/img/rider/2025.3/gdscript_file_template.png)

### GDScript 快速文档

您可以查看从注释生成的 **Quick Documentation**：

![JetBrains Rider: GDScript 快速文档](https://resources.jetbrains.com/help/img/rider/2025.3/gdscript_documentation.png)

### GDScript 参数信息

在 GDScript 中编写或研究函数调用时，使用 **Parameter Information** 查看可用方法签名的详细信息：

![JetBrains Rider: GDScript 参数信息](https://resources.jetbrains.com/help/img/rider/2025.3/gdscript_parameter_information.png)

### 嵌入提示（Inlay Hints）

GDScript 文件中也有专用的 **Inlay hints**：

![JetBrains Rider: GDScript 嵌入提示](https://resources.jetbrains.com/help/img/rider/2025.3/gdscript_inlay_hints.png)

### GDScript 操作指示器

JetBrains Rider 在左侧边缘添加了许多不同的操作指示器。除了常见的指示器（如快速修复灯泡 ![快速修复图标](https://resources.jetbrains.com/help/img/rider/2025.3/app.expui.codeInsight.intentionBulb.png)）外，您还可以使用以下 GDScript 特有的指示器：

#### 父类（super）方法

![JetBrains Rider: GDScript 操作指示器 - Super 方法](https://resources.jetbrains.com/help/img/rider/2025.3/gdscript_super_method.png)

#### 运行当前场景

![JetBrains Rider: GDScript 操作指示器 - 运行当前场景](https://resources.jetbrains.com/help/img/rider/2025.3/gdscript_run_marker.png)

#### 资源使用情况

![JetBrains Rider: GDScript 操作指示器 - 资源使用情况](https://resources.jetbrains.com/help/img/rider/2025.3/gdscript_resource_usage.png)

#### 已连接信号

![JetBrains Rider: GDScript 操作指示器 - 已连接信号](https://resources.jetbrains.com/help/img/rider/2025.3/gdscript_connected_signal.png)

#### 继承场景

![JetBrains Rider: GDScript 操作指示器 - 继承场景](https://resources.jetbrains.com/help/img/rider/2025.3/gdscript_inherited_scene.png)

#### 颜色选择器

![JetBrains Rider: GDScript 操作指示器 - 颜色选择器](https://resources.jetbrains.com/help/img/rider/2025.3/gdscript_color_picker.png)

### GDScript 代码格式化

代码风格的一个重要方面是如何格式化代码，即如何使用空格、制表符和换行符来排列代码元素，是否以及如何使用制表符进行缩进，是否以及如何换行长行等等。

JetBrains Rider 广泛的代码格式化规则集具有默认配置，其中考虑了众多最佳实践。您可以配置格式化规则的每个细节，并在代码中强制执行这些规则。当 JetBrains Rider 通过代码补全和代码生成功能生成新代码、应用代码模板和执行重构时，会应用这些规则。格式化规则也可以应用于当前选区、当前文件或直至整个解决方案的现有代码。

您可以在 JetBrains Rider 设置的 **Editor | Code Style | GDScript** 页面配置 GDScript 的格式化风格。

### GDScript 支持的当前限制

*   `get_node`、`get_parent` 等不会解析为实际的节点，而只是通用的 `Node` 类型。
*   `get_window` 方法（以及可能的其他少数方法）根据上下文返回不同的类（`SubViewport`、`Window` 等），但返回值被解析为基类 `Viewport`，因此要获得代码补全，您必须手动指定类型。
*   动态节点和运行时添加的节点在设计时无法预测，因此无法为它们提供代码补全。

## C# 支持

JetBrains Rider 为用 C# 编写的 Godot 项目提供全面支持。这包括：

*   专为 Godot 框架定制的导航、代码补全和重构工具
*   带有快速修复建议的代码检查
*   MSBuild 集成和对 Godot C# 项目模型的支持
*   调试功能，包括断点、单步执行和变量检查

## 场景预览窗口

在 JetBrains Rider 中，您可以在专用的场景预览窗口中查看所选 `.tscn` 的节点树，或与给定 `.gd` 文件关联的所有 `.tscn` 文件：

![JetBrains Rider: Godot 支持 - 场景预览窗口](https://resources.jetbrains.com/help/img/rider/2025.3/gdscript_scene_preview.png)

您可以将节点从场景预览窗口拖放到编辑器中的脚本中：

![JetBrains Rider: Godot 支持 - 场景预览窗口. 拖放](https://resources.jetbrains.com/help/img/rider/2025.3/gdscript_node_drag.png)

## 运行和调试

当您打开 Godot 项目时，JetBrains Rider 会自动创建一个或多个运行配置，您可以使用这些配置来运行和调试游戏。Godot 项目的调试通过调试适配器协议（DAP）进行。

运行仅包含 C# 代码的 Godot 项目与包含 GDScript 或混合 GDScript/C# 代码的项目有一些区别。

### 运行和调试包含 C# 代码的 Godot 项目

1.  如果要调试项目，请在必要处设置断点准备会话。
2.  使用 **Player** 运行配置启动项目。此配置在您打开 Godot 项目时自动创建。

    ![JetBrains Rider: 启动 Godot 项目](https://resources.jetbrains.com/help/img/rider/2025.3/godot_run_config_player.png)

3.  或者，您可以从 Godot/C# 项目启动单个场景。为此，右键单击场景文件 `.tscn` 并选择 **Debug '[scene name]'**。

包含 GDScript 代码的 Godot 项目只能在调试器下从 JetBrains Rider 启动。要在不调试的情况下运行此类项目，请使用 Godot 编辑器。

### 调试包含 GDScript 代码的 Godot 项目

1.  确保 Godot 编辑器正在运行并打开了同一项目。
    如果未运行，您可以从 JetBrains Rider 启动它，方法是点击工具栏上的 Godot 图标并选择 **Start Godot Editor**：

    ![JetBrains Rider: 启动 Godot 编辑器](https://resources.jetbrains.com/help/img/rider/2025.3/godot_start_editor.png)

2.  如果要调试项目，请在必要处设置断点准备会话。
3.  使用 **Player GDScript** 运行配置启动项目。此配置在您打开 Godot 项目时自动创建。

### 同时调试 GDScript 和 C#

1.  如上所述使用 **Player GDScript** 配置启动调试会话。
2.  按相关快捷键或从主菜单选择 **Run | Attach to Process**。
3.  在列表中选择所需的 Godot 进程，然后点击 **Attach with .NET Debugger**。

## C++ 支持

对于使用 Godot 引擎源代码的开发人员，JetBrains Rider 提供全面的 C++ 开发支持。这包括：

*   具有代码补全和导航等功能的智能代码洞察
*   代码检查和重构
*   用于修改和扩展引擎的跨平台兼容性
*   与构建系统和调试工具的集成

> **注意**
>
> 有关 C++ 开发支持的更多信息，请参阅 [Godot 引擎文档](https://docs.godotengine.org/en/stable/engine_details/development/configuring_an_ide/rider.html)。

## 测试 C# 项目

如果您在 C# Godot 项目中使用 [gdUnit4Net](https://github.com/MikeSchulze/gdUnit4Net) 单元测试框架，则可以使用广泛的单元测试功能。

## 分析 C# 项目

要分析 C# Godot 项目的性能和内存，请点击自动创建的 **Player** 运行配置旁边的三点菜单，然后选择所需的分析类型：

![JetBrains Rider: 分析 Godot 项目](https://resources.jetbrains.com/help/img/rider/2025.3/godot_profiling.png)

更多信息，请参阅 Godot 引擎中的 [原始拉取请求](https://github.com/godotengine/godot/pull/34382)。
