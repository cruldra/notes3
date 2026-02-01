本文旨在帮助您判断 Godot 是否适合您。我们将介绍引擎的一些广泛特性，让您了解可以用它实现什么，并回答诸如“开始使用需要了解什么？”等问题。

这绝不是一个详尽的概述。我们将在这个入门系列中介绍更多特性。

## 什么是 Godot？

Godot 是一个通用的 2D 和 3D 游戏引擎，旨在支持各种类型的项目。您可以用它来创建游戏或应用程序，然后发布到桌面、移动设备以及 Web 平台。

您也可以用它创建主机游戏，不过这要么需要很强的编程技能，要么需要开发人员为您移植游戏。

> **注意**
>
> 关于主机支持的信息，请参阅 [Godot 网站](https://godotengine.org/consoles/)。

## 引擎能做什么？

Godot 最初是由一家阿根廷游戏工作室内部开发的。它的开发始于 2001 年，自 2014 年开源发布以来，引擎已被重写并得到了巨大的改进。

用 Godot 创建的游戏示例包括 Cassette Beasts、PVKK 和 Usagi Shima。在应用程序方面，开源像素画绘制程序 Pixelorama 是由 Godot 驱动的，体素 RPG 制作工具 RPG in a Box 也是如此。您可以在 [官方展示](https://godotengine.org/showcase/) 中找到更多示例。

![Usagi Shima](https://docs.godotengine.org/en/stable/_images/introduction_usagi_shima.webp)
*Usagi Shima*

![Cassette Beasts](https://docs.godotengine.org/en/stable/_images/introduction_cassette_beasts.webp)
*Cassette Beasts*

![PVKK](https://docs.godotengine.org/en/stable/_images/introduction_pvkk.webp)
*PVKK: Planetenverteidigungskanonenkommandant*

![RPG in a Box](https://docs.godotengine.org/en/stable/_images/introduction_rpg_in_a_box.webp)
*RPG in a Box*

## 它如何工作以及看起来如何？

Godot 配备了一个功能齐全的游戏编辑器，集成了满足最常见需求的工具。它包括代码编辑器、动画编辑器、TileMap 编辑器、着色器编辑器、调试器、分析器等。

![Godot Editor](https://docs.godotengine.org/en/stable/_images/introduction_editor.webp)

团队致力于提供一个功能丰富且用户体验一致的游戏编辑器。虽然总有改进的空间，但用户界面一直在不断完善。

当然，如果您愿意，也可以使用外部程序。我们官方支持导入在 [Blender](https://www.blender.org/) 中设计的 3D 场景，并维护在 [VSCode](https://github.com/godotengine/godot-vscode-plugin) 和 [Emacs](https://github.com/godotengine/emacs-gdscript-mode) 中编写 GDScript 和 C# 的插件。我们还支持在 Windows 上使用 Visual Studio 进行 C# 开发。

![VSCode Integration](https://docs.godotengine.org/en/stable/_images/introduction_vscode.png)

## 编程语言

让我们谈谈可用的编程语言。

您可以使用 [GDScript](https://docs.godotengine.org/en/stable/tutorials/scripting/gdscript/index.html#doc-gdscript) 编写游戏，这是一种 Godot 专用且紧密集成的语言，语法轻量；或者使用 [C#](https://docs.godotengine.org/en/stable/tutorials/scripting/c_sharp/index.html#doc-c-sharp)，这在游戏行业很流行。这是我们支持的两种主要脚本语言。

通过 [GDExtension](https://docs.godotengine.org/en/stable/tutorials/scripting/gdextension/what_is_gdextension.html#doc-what-is-gdextension) 技术，您还可以使用 [C++](https://docs.godotengine.org/en/stable/tutorials/scripting/cpp/index.html#doc-godot-cpp) 或 [其他语言](https://docs.godotengine.org/en/stable/tutorials/scripting/other_languages.html#doc-scripting-languages) 编写游戏逻辑或高性能算法，而无需重新编译引擎。您可以使用此技术在引擎中集成第三方库和其他软件开发工具包 (SDK)。

当然，您也可以直接向引擎添加模块和特性，因为它是完全免费和开源的。

## 使用 Godot 需要了解什么？

Godot 是一个功能丰富的游戏引擎。它有成千上万的特性，有很多东西要学。为了充分利用它，您需要良好的编程基础。虽然我们努力使引擎易于上手，但首先学会像程序员一样思考会让您受益匪浅。

Godot 依赖于面向对象的编程范式。熟悉类和对象等概念将帮助您在其中高效地编码。

如果您完全是编程新手，GDQuest 的 *Learn GDScript From Zero* 是一个免费的开源交互式教程，适合绝对初学者学习使用 Godot 的 GDScript 语言进行编程。它可以作为 [桌面应用程序](https://gdquest.itch.io/learn-godot-gdscript) 或 [在浏览器中](https://gdquest.github.io/learn-gdscript) 使用。

我们将在 [学习新特性](https://docs.godotengine.org/en/stable/getting_started/introduction/learning_new_features.html#doc-learning-new-features) 中为您提供更多 Godot 特定的学习资源。

在下一部分，您将了解引擎的基本概念概览。

[上一页](https://docs.godotengine.org/en/stable/getting_started/introduction/index.html) [下一页](https://docs.godotengine.org/en/stable/getting_started/introduction/learn_to_code_with_gdscript.html)
