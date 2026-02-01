---
sidebar_position: 7
---

通常，我们需要编写依赖于其他对象提供功能的脚本。这一过程包含两个部分：

1.  **获取引用**：获取一个（据推测）具有所需功能的对象的引用。
2.  **访问数据或逻辑**：从该对象中获取数据或调用逻辑。

本教程的其余部分将概述实现上述操作的各种方式。

## 获取对象引用 (Acquiring object references)

对于所有的 [Object](https://docs.godotengine.org/en/stable/classes/class_object.html)，最基本的引用方式是从另一个已获取的实例中获取现有对象的引用。

```gdscript
var obj = node.object # 属性访问。
var obj = node.get_object() # 方法访问。
```

同样的原则也适用于 [RefCounted](https://docs.godotengine.org/en/stable/classes/class_refcounted.html) 对象。虽然用户经常以这种方式访问 [Node](https://docs.godotengine.org/en/stable/classes/class_node.html) 和 [Resource](https://docs.godotengine.org/en/stable/classes/class_resource.html)，但还有其他替代措施。

除了通过属性或方法访问，还可以通过“加载访问”来获取资源（Resources）。

```gdscript
# 如果你需要一个“导出常量变量（export const var）”（该语法不存在），
# 请为工具脚本使用一个条件 setter，检查其是否在编辑器中执行。
# `@tool` 注解必须放置在脚本顶部。
@tool

# 在场景加载期间加载资源。
var preres = preload(path)
# 当程序运行到该语句时加载资源。
var res = load(path)

# 注意：按照惯例，用户通常使用 PascalCase（类名命名法）
# 加载场景和脚本，通常加载到常量中。
const MyScene = preload("my_scene.tscn") # 静态加载
const MyScript = preload("my_script.gd")

# 这种类型的值是变化的，即它是一个变量，因此使用 snake_case（蛇形命名法）。
@export var script_type: Script

# 必须从编辑器配置，默认为 null。
@export var const_script: Script:
	set(value):
		if Engine.is_editor_hint():
			const_script = value

# 如果值未设置，则警告用户。
func _get_configuration_warnings():
	if not const_script:
		return ["必须初始化属性 'const_script'。"]

	return []
```

注意以下几点：

1.  一个语言有多种加载此类资源的方式。
2.  在设计对象如何访问数据时，不要忘记也可以将资源作为引用进行传递。
3.  请记住，加载资源会获取由引擎维护的缓存资源实例。要获得一个新对象，必须 [克隆 (duplicate)](https://docs.godotengine.org/en/stable/classes/class_resource.html#class-resource-method-duplicate) 现有引用，或者使用 `new()` 从头开始实例化一个。

节点（Nodes）同样有一个替代访问点：SceneTree。

```gdscript
extends Node

# 慢。
func dynamic_lookup_with_dynamic_nodepath():
	print(get_node("Child"))

# 较快。仅限 GDScript。
func dynamic_lookup_with_cached_nodepath():
	print($Child)

# 最快。即使节点以后移动了也不会失效。
# 注意：`@onready` 注解仅限 GDScript。
@onready var child = $Child
func lookup_and_cache_for_future_access():
	print(child)

# 最快。在场景树面板中移动节点也不会失效。
# 必须在检查器中选择该节点，因为它是一个导出属性。
@export var child: Node
func lookup_and_cache_for_future_access():
	print(child)

# 将引用分配委托给外部源。
# 缺点：需要执行验证检查。
# 优点：节点对其外部结构没有要求。'prop' 可以来自任何地方。
var prop
func call_me_after_prop_is_initialized_by_parent():
	# 以三种方式之一验证 prop。

	# 静默失败。
	if not prop:
		return

	# 打印错误消息后失败。
	if not prop:
		printerr("'prop' 未初始化")
		return

	# 失败并终止程序。
	# 注意：从发布版导出模板运行的脚本不会运行 `assert`。
	assert(prop, "'prop' 未初始化")

# 使用自动加载（autoload）。
# 对于典型节点很危险，但对于管理自身数据且不干涉其他对象的真正单例节点很有用。
func reference_a_global_autoloaded_variable():
	print(globals)
	print(globals.prop)
	print(globals.my_getter())
```

## 从对象访问数据或逻辑 (Accessing data or logic from an object)

Godot 的脚本 API 是 **鸭子类型 (duck-typed)** 的。这意味着如果一个脚本执行了一项操作，Godot 并不会根据 **类型** 来验证它是否支持该操作。相反，它会检查该对象是否 **实现** 了该特定方法。

例如，[CanvasItem](https://docs.godotengine.org/en/stable/classes/class_canvasitem.html) 类有一个 `visible` 属性。所有暴露给脚本 API 的属性实际上都是绑定到名称的 setter 和 getter 对。如果有人尝试访问 `CanvasItem.visible`，那么 Godot 会按顺序执行以下检查：

*   如果对象附加了脚本，它将尝试通过脚本设置属性。这为脚本提供了通过重写该属性的 setter 方法来重写基对象中定义的属性的机会。
*   如果脚本没有该属性，它会在 ClassDB 中针对 `CanvasItem` 类及其所有继承类型执行 HashMap 查找，查找 “visible” 属性。如果找到，它将调用绑定的 setter 或 getter。
*   如果未找到，它会进行显式检查，查看用户是否想要访问 “script” 或 “meta” 属性。
*   如果没有，它会在 `CanvasItem` 及其继承类型中检查 `_set`/`_get` 实现（取决于访问类型）。这些方法可以执行逻辑，给人一种该对象拥有某个属性的印象。`_get_property_list` 方法也是如此。
    *   注意，即使是非法的符号名称（如以数字开头或包含斜杠的名称），也会发生这种情况。

因此，这个鸭子类型系统可以在脚本、对象的类或该对象继承的任何类中定位属性，但仅限于继承自 `Object` 的事物。

Godot 为这些访问执行运行时检查提供了多种选项：

*   **鸭子类型属性访问**。这些将是属性检查（如上所述）。如果对象不支持该操作，执行将停止。

    ```gdscript
    # 所有对象都具有鸭子类型的 get、set 和 call 包装方法。
    get_parent().set("visible", false)

    # 在方法调用中使用符号访问器而非字符串，
    # 将隐式调用 `set` 方法，进而通过属性查找序列调用绑定到该属性的 setter 方法。
    get_parent().visible = false

    # 注意：如果定义了描述属性存在的 `_set` 和 `_get`，
    # 但该属性未在任何 `_get_property_list` 方法中被识别，
    # 那么 set() 和 get() 方法将起作用，但符号访问（点号语法）将声称找不到该属性。
    ```

*   **方法检查**。在 `CanvasItem.visible` 的情况下，可以像访问任何其他方法一样访问 `set_visible` 和 `is_visible` 方法。

    ```gdscript
    var child = get_child(0)

    # 动态查找。
    child.call("set_visible", false)

    # 基于符号的动态查找。
    # GDScript 在幕后将其起别名为 'call' 方法。
    child.set_visible(false)

    # 动态查找，先检查方法是否存在。
    if child.has_method("set_visible"):
    	child.set_visible(false)

    # 类型转换检查，然后进行动态查找。
    # 当你确定该类实现了所有方法，从而进行多次“安全”调用时很有用。无需重复检查。
    # 如果对用户定义的类型执行转换检查，可能会比较棘手，因为它会强制增加依赖项。
    if child is CanvasItem:
    	child.set_visible(false)
    	child.show_on_top = true

    # 如果不想在不通知用户的情况下使这些检查失败，可以使用断言。
    # 如果不为真，这些将立即触发运行时错误。
    assert(child.has_method("set_visible"))
    assert(child.is_in_group("offer"))
    assert(child is CanvasItem)

    # 还可以使用对象标签来暗指一个接口，即假设它实现了某些方法。
    # 有两种类型，且都仅存在于节点：名称（Names）和分组（Groups）。

    # 假设...
    # 存在一个 "Quest" 对象，且 1) 它可以“完成（complete）”或“失败（fail）”，
    # 且在每个状态前后都有可用的文本...

    # 1. 使用名称。
    var quest = $Quest
    print(quest.text)
    quest.complete() # 或 quest.fail()
    print(quest.text) # 暗指新的文本内容

    # 2. 使用分组。
    for a_child in get_children():
    	if a_child.is_in_group("quest"):
    		print(a_child.text)
    		a_child.complete() # 或 a_child.fail()
    		print(a_child.text) # 暗指新的文本内容

    # 注意：这些接口是团队定义的项目特定约定（这意味着需要文档！但可能值得？）。
    # 任何符合文档记录的名称或分组“接口”的脚本都可以取而代之。
    ```

*   **将访问外包给 [Callable](https://docs.godotengine.org/en/stable/classes/class_callable.html)**。在需要最大程度摆脱依赖项的情况下，这些非常有用。在这种情况下，依靠外部上下文来设置该方法。

    ```gdscript
    # child.gd
    extends Node
    var fn = null

    func my_method():
    	if fn:
    		fn.call()

    # parent.gd
    extends Node

    @onready var child = $Child

    func _ready():
    	child.fn = print_me
    	child.my_method()

    func print_me():
    	print(name)
    ```

这些策略构成了 Godot 灵活的设计。通过这些策略，用户拥有了广泛的工具来满足其特定需求。
