---
sidebar_position: 2
---

Godot 引擎提供了两种创建可重用对象的主要方式：脚本（Scripts）和场景（Scenes）。从技术层面上看，这两者在底层都没有定义真正的“类”。

尽管如此，使用 Godot 的许多最佳实践都涉及将面向对象编程（OOP）原则应用于构成游戏的脚本和场景。这就是为什么将它们视为“类”是有益的。

本指南简要解释了脚本和场景在引擎核心中的工作方式，以帮助你理解它们的底层原理。

## 脚本在引擎中的工作方式 (How scripts work in the engine)

引擎提供了诸如 [Node](https://docs.godotengine.org/en/stable/classes/class_node.html) 之类的内置类。你可以通过脚本扩展这些类以创建派生类型。

从技术上讲，这些脚本并不是类。相反，它们是一种 **资源 (Resource)**，告诉引擎对引擎内置类之一执行一系列初始化操作。

Godot 的内部类拥有将类数据注册到 [ClassDB](https://docs.godotengine.org/en/stable/classes/class_classdb.html) 的方法。该数据库提供对类信息的运行时访问。`ClassDB` 包含有关类的信息，例如：

*   属性 (Properties)
*   方法 (Methods)
*   常量 (Constants)
*   信号 (Signals)

当对象执行诸如访问属性或调用方法之类的操作时，就会检查这个 `ClassDB`。它会检查数据库记录以及对象的基类型记录，以查看该对象是否支持该操作。

为对象附加一个 [Script](https://docs.godotengine.org/en/stable/classes/class_script.html) 会扩展 `ClassDB` 中可用的方法、属性和信号。

> **注意**
>
> 即使是不使用 `extends` 关键字的脚本，也会隐式继承自引擎的基类 [RefCounted](https://docs.godotengine.org/en/stable/classes/class_refcounted.html)。因此，你可以通过代码实例化不带 `extends` 的脚本。但由于它们扩展了 `RefCounted`，你不能将它们附加到 [Node](https://docs.godotengine.org/en/stable/classes/class_node.html) 上。

## 场景 (Scenes)

场景的行为与类有许多相似之处，因此将场景视为一个类是有意义的。场景是可重用、可实例化且可继承的节点组。创建一个场景类似于编写一个创建节点并使用 `add_child()` 将其添加为子节点的脚本。

我们通常会将一个场景与其根节点上的脚本配对，该脚本会利用场景中的节点。因此，脚本通过命令式代码为场景添加行为，从而扩展了场景。

场景的内容有助于定义：

*   脚本可以使用哪些节点。
*   这些节点是如何组织的。
*   它们是如何初始化的。
*   它们彼此之间有哪些信号连接。

为什么这些对场景组织很重要？因为场景的实例 **就是** 对象。因此，许多适用于编写代码的面向对象原则也适用于场景：单一职责、封装等。

场景 **始终是附加到其根节点的脚本的延伸**，因此你可以将其解释为类的一部分。

本最佳实践系列中解释的大多数技术都建立在这一点之上。
