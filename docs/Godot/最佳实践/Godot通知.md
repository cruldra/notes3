---
sidebar_position: 8
---

Godot 中的每个对象都实现了一个 [_notification](https://docs.godotengine.org/en/stable/classes/class_object.html#class-object-private-method-notification) 方法。其目的是允许该对象响应与其相关的各种引擎级回调。例如，如果引擎告诉一个 [CanvasItem](https://docs.godotengine.org/en/stable/classes/class_canvasitem.html) “绘制（draw）”，它将调用 `_notification(NOTIFICATION_DRAW)`。

其中一些通知（如绘制）在脚本中进行重写非常有用。以至于 Godot 为其中的许多通知公开了专用函数：

*   `_ready()`: `NOTIFICATION_READY`
*   `_enter_tree()`: `NOTIFICATION_ENTER_TREE`
*   `_exit_tree()`: `NOTIFICATION_EXIT_TREE`
*   `_process(delta)`: `NOTIFICATION_PROCESS`
*   `_physics_process(delta)`: `NOTIFICATION_PHYSICS_PROCESS`
*   `_draw()`: `NOTIFICATION_DRAW`

用户可能 *没有* 意识到的是，通知不仅存在于 Node 类型中，例如：

*   [Object::NOTIFICATION_POSTINITIALIZE](https://docs.godotengine.org/en/stable/classes/class_object.html#class-object-constant-notification-postinitialize)：在对象初始化期间触发的回调。脚本无法访问。
*   [Object::NOTIFICATION_PREDELETE](https://docs.godotengine.org/en/stable/classes/class_object.html#class-object-constant-notification-predelete)：在引擎删除对象之前触发的回调，即“析构函数”。

而且 Node 中存在的许多回调并没有专门的方法，但仍然非常有用：

*   [Node::NOTIFICATION_PARENTED](https://docs.godotengine.org/en/stable/classes/class_node.html#class-node-constant-notification-parented)：每当将子节点添加到另一个节点时触发的回调。
*   [Node::NOTIFICATION_UNPARENTED](https://docs.godotengine.org/en/stable/classes/class_node.html#class-node-constant-notification-unparented)：每当从另一个节点移除子节点时触发的回调。

可以通过通用的 `_notification()` 方法访问所有这些自定义通知。

> **注意**
>
> 文档中标记为“virtual”的方法也旨在由脚本重写。
>
> 一个经典的例子是 Object 中的 [_init](https://docs.godotengine.org/en/stable/classes/class_object.html#class-object-private-method-init) 方法。虽然它没有对应的 `NOTIFICATION_*`，但引擎仍然会调用该方法。大多数语言（C# 除外）都将其作为构造函数。

那么，在何种情况下应该使用这些通知或虚函数呢？

## _process vs. _physics_process vs. *_input

当需要帧率相关的帧间 delta 时间时，请使用 `_process()`。如果更新对象数据的代码需要尽可能频繁地更新，这里就是正确的地方。经常在这里执行循环逻辑检查和数据缓存，但这取决于需要更新评估的频率。如果它们不需要每帧执行，那么实现一个 Timer 超时循环是另一种选择。

```gdscript
# 允许执行不需要每帧（甚至不需要每个固定帧）
# 触发脚本逻辑的循环操作。
func _ready():
	var timer = Timer.new()
	timer.autostart = true
	timer.wait_time = 0.5
	add_child(timer)
	timer.timeout.connect(func():
		print("此代码块每 0.5 秒运行一次")
	)
```

当需要帧率无关的帧间 delta 时间时，请使用 `_physics_process()`。如果代码需要随时间进行一致的更新，无论时间流逝快慢，这里就是正确的地方。循环的运动学和对象变换操作应在此处执行。

虽然可以实现，但为了获得最佳性能，应避免在这些回调中进行输入检查。`_process()` 和 `_physics_process()` 将在每个机会触发（默认情况下它们不会“休息”）。相比之下，`*_input()` 回调仅在引擎实际检测到输入的帧中触发。

在输入回调中检查输入动作也是一样的。如果想使用 delta 时间，可以根据需要在相关的 delta 时间方法中获取它。

```gdscript
# 即使引擎未检测到输入，也会每帧调用。
func _process(delta):
	if Input.is_action_just_pressed("ui_select"):
		print(delta)

# 在每次输入事件期间调用。
func _unhandled_input(event):
	match event.get_class():
		"InputEventKey":
			if Input.is_action_just_pressed("ui_accept"):
				print(get_process_delta_time())
```

## _init vs. 初始化 vs. export

如果脚本初始化其自己的节点子树（不使用场景），该代码应在 `_init()` 中执行。其他属性或独立于 SceneTree 的初始化也应在此处运行。

> **注意**
>
> 与 GDScript 的 `_init()` 方法对应的 C# 等效项是构造函数。

`_init()` 在 `_enter_tree()` 或 `_ready()` 之前触发，但在脚本创建并初始化其属性之后触发。实例化场景时，属性值的设置将遵循以下顺序：

1.  **初始值分配**：属性被分配其初始值，如果未指定则为默认值。如果存在 setter，则不会使用它。
2.  `_init()` **分配**：属性的值被 `_init()` 中所做的任何分配所替换，触发 setter。
3.  **导出值分配**：导出属性的值再次被检查器中设置的任何值所替换，触发 setter。

```gdscript
# test 被初始化为 "one"，不触发 setter。
@export var test: String = "one":
	set(value):
		test = value + "!"

func _init():
	# 触发 setter，将 test 的值从 "one" 更改为 "two!"。
	test = "two"

# 如果有人从检查器将 test 设置为 "three"，它将触发
# setter，将 test 的值从 "two!" 更改为 "three!"。
```

因此，实例化脚本与实例化场景可能会影响初始化过程以及引擎调用 setter 的次数。

## _ready vs. _enter_tree vs. NOTIFICATION_PARENTED

当实例化连接到第一个执行场景的场景时，Godot 将向下遍历树实例化节点（调用 `_init()`）并从根节点向下构建树。这导致 `_enter_tree()` 调用向下级联。一旦树构建完成，叶节点将调用 `_ready`。一个节点将在其所有子节点完成调用后调用此方法。这随后会导致一个反向级联，向上回到树的根部。

当实例化脚本或独立场景时，节点在创建时不会添加到 SceneTree，因此不会触发 `_enter_tree()` 回调。相反，仅发生 `_init()` 调用。当场景被添加到 SceneTree 时，才会发生 `_enter_tree()` 和 `_ready()` 调用。

如果需要触发发生在节点设为另一个节点的子节点时的行为，无论它是作为主/活动场景的一部分发生，还是在其他情况下，可以使用 [PARENTED](https://docs.godotengine.org/en/stable/classes/class_node.html#class-node-constant-notification-parented) 通知。例如，下面是一个将节点的代码连接到父节点上的自定义信号且不会失败的代码片段。这对于在运行时创建的数据中心节点很有用。

```gdscript
extends Node

var parent_cache

func connection_check():
	return parent_cache.has_user_signal("interacted_with")

func _notification(what):
	match what:
		NOTIFICATION_PARENTED:
			parent_cache = get_parent()
			if connection_check():
				parent_cache.interacted_with.connect(_on_parent_interacted_with)
		NOTIFICATION_UNPARENTED:
			if connection_check():
				parent_cache.interacted_with.disconnect(_on_parent_interacted_with)

func _on_parent_interacted_with():
	print("我正在对我父级的交互做出反应！")
```
