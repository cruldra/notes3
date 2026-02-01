---
sidebar_position: 5
---

Godot 提供了一项在项目根目录自动加载节点的功能，允许你全局访问它们，这可以履行单例（Singleton）的角色：[单例 (自动加载)](https://docs.godotengine.org/en/stable/tutorials/scripting/singletons_autoload.html)。当你使用代码通过 [SceneTree.change_scene_to_file](https://docs.godotengine.org/en/stable/classes/class_scenetree.html) 切换场景时，这些自动加载的节点不会被释放。

在本指南中，你将学习何时使用自动加载功能，以及可以用来避免使用它的技术。

## 音频中断问题 (The cutting audio issue)

其他引擎可能会鼓励创建“管理类（manager classes）”，即通过单例将大量功能组织到一个全局可访问的对象中。由于有了节点树和信号，Godot 提供了许多避免全局状态的方法。

例如，假设我们正在制作一个平台游戏，并希望在收集硬币时播放音效。有一个专门的节点可以实现：[AudioStreamPlayer](https://docs.godotengine.org/en/stable/classes/class_audiostreamplayer.html)。但是，如果我们在 `AudioStreamPlayer` 已经在播放声音时再次调用它，新声音会中断第一个。

一种解决方案是编写一个全局的、自动加载的音频管理类。它生成一个 `AudioStreamPlayer` 节点池，随着每个新的音效请求进来而循环使用。假设我们将该类命名为 `Sound`，你可以通过在项目中任何地方调用 `Sound.play("coin_pickup.ogg")` 来使用它。这在短期内解决了问题，但会引发更多问题：

1.  **全局状态**：现在一个对象负责所有对象的数据。如果 `Sound` 类出现错误或没有可用的 AudioStreamPlayer，所有调用它的节点都可能崩溃。
2.  **全局访问**：既然任何对象都可以从任何地方调用 `Sound.play(sound_path)`，就再也没有简单的方法来查找 Bug 的根源了。
3.  **全局资源分配**：从一开始就存储一个 `AudioStreamPlayer` 节点池，你可能会因为节点太少而面临 Bug，或者因为节点太多而占用不必要的内存。

> **注意**
>
> 关于全局访问，问题在于任何地方的代码都可能向我们示例中的 `Sound` 自动加载传递错误数据。因此，修复该 Bug 需要探索的领域涵盖了整个项目。

当你在场景内部保留代码时，可能只有一两个脚本涉及音频。

相比之下，让每个场景根据需要在自身内部保留尽可能多的 `AudioStreamPlayer` 节点，所有这些问题都会消失：

1.  每个场景管理自己的状态信息。如果数据有问题，只会对该场景产生影响。
2.  每个场景仅访问自己的节点。现在，如果出现 Bug，很容易找到是哪个节点出了错。
3.  每个场景恰好分配它所需的资源量。

## 管理共享功能或数据 (Managing shared functionality or data)

使用自动加载的另一个原因可能是你希望在许多场景中重用相同的方法或数据。

对于函数，你可以创建一个新的 `Node` 类型，并在 GDScript 中使用 [class_name](https://docs.godotengine.org/en/stable/tutorials/scripting/gdscript/gdscript_basics.html) 关键字为单个场景提供该功能。

对于数据，你可以：

1.  创建一个新类型的 [Resource](https://docs.godotengine.org/en/stable/classes/class_resource.html) (资源) 来共享数据。
2.  将数据存储在一个每个节点都能访问的对象中，例如使用 `owner` 属性来访问场景的根节点。

## 何时应该使用自动加载 (When you should use an Autoload)

GDScript 支持使用 `static func` 创建静态（static）函数。结合 `class_name`，可以创建辅助函数库，而无需创建实例即可调用它们。静态函数的局限在于它们不能引用成员变量、非静态函数或 `self`。

从 Godot 4.1 开始，GDScript 还支持使用 `static var` 创建静态变量。这意味着你现在可以跨类的实例共享变量，而无需创建单独的自动加载。

尽管如此，自动加载节点可以简化大范围作用域系统的代码。如果自动加载节点正在管理自己的信息，且不侵占其他对象的数据，那么它是创建处理宽泛任务系统的绝佳方式。例如，任务系统或对话系统。

> **注意**
>
> 自动加载节点 **不一定** 是单例。没有什么能阻止你实例化自动加载节点的副本。自动加载仅是一种工具，它使节点无论你的游戏节点结构如何或你运行哪个场景（例如通过按 F6 键），都会自动作为场景树根节点的子节点加载。
>
> 因此，你可以通过调用 `get_node("/root/Sound")` 来获取自动加载节点，例如一个名为 `Sound` 的自动加载项。
