`@onready` 是 GDScript（Godot 的脚本语言）中一个非常实用且高频使用的**注解（Annotation）**。

一句话解释：**它是“语法糖”，让你能在定义变量的同时给它赋值，但这个赋值操作会等到节点“准备好”（_ready）之后才真正执行。**

它的核心作用是**方便地获取子节点**。

---

### 1. 为什么要用它？（解决什么问题）

在 Godot 中，当一个脚本被实例化（`_init` 阶段）时，它的子节点还没有被添加到场景树中。

如果你试图在脚本的最顶端直接获取一个子节点，会报错：

```gdscript
extends Node2D

# ❌ 错误写法！
# 此时脚本刚创建，场景树还没建好，"Sprite2D" 还不存在。
# 结果：报错或为 null
var my_sprite = $Sprite2D 

func _ready():
    pass

```

在没有 `@onready` 之前（或者在其他编程语言中），你必须这样写：

```gdscript
extends Node2D

# 1. 先声明变量（初始为空）
var my_sprite

func _ready():
    # 2. 等到 _ready 时（节点都在树上了），再手动赋值
    my_sprite = $Sprite2D
    my_sprite.modulate = Color.RED

```

这种写法很繁琐，如果我有 10 个子节点要引用，`_ready` 函数里就会堆满赋值语句。

---

### 2. `@onready` 的写法

使用 `@onready`，我们可以将声明和赋值合并只有一行，且安全无误：

```gdscript
extends Node2D

# ✅ 正确写法
# 意思：声明 my_sprite 变量，但先别急着赋值。
# 等到 _ready() 被调用前的那一瞬间，再去执行 $Sprite2D 并赋值给它。
@onready var my_sprite = $Sprite2D
@onready var timer_label = $UI/TimerLabel

func _ready():
    # 现在可以直接用了
    my_sprite.modulate = Color.RED
    print(timer_label.text)

```

---

### 3. 主要使用场景

#### A. 获取节点引用 (最常用)

这是 90% 的用途。配合 `$` 符号（`get_node` 的简写）使用。

```gdscript
@onready var health_bar = $CanvasLayer/HealthBar
@onready var attack_timer = $Timers/AttackTimer

```

#### B. 复杂的初始化计算

如果一个变量的初始值依赖于节点的位置或其他需要在 `_ready` 才能确定的数据。

```gdscript
# ❌ 错误：在初始化时 position 还没确定，可能是 (0,0)
var start_pos = position 

# ✅ 正确：等到节点进入场景、位置确定后，再记录初始位置
@onready var start_pos = position 

```

#### C. 预加载资源并立即实例化

虽然不常用，但有时会这么写：

```gdscript
# 预加载子弹场景，并等待 ready 后实例化（这里只是举例逻辑，通常不会直接赋值实例）
@onready var bullet_scene = preload("res://bullet.tscn")

```

---

### 4. 执行顺序

理解这个有助于 debug：

1. **`_init()`**: 脚本被创建（变量声明了，但 `@onready` 的赋值还没跑）。
2. **`_enter_tree()`**: 节点进入场景树。
3. **`@onready` 变量赋值**: 从上到下依次执行。
4. **`_ready()`**: 最终执行 `_ready()` 函数体内的代码。

**注意：** 因为 `@onready` 实际上是在 `_ready` 阶段执行的，所以你**不能**在 `_init` 函数里使用 `@onready` 定义的变量（那时候它们还是 null）。

### 总结

* **没有 `@onready**`: 必须分开写“声明”和“赋值”，代码又长又乱。
* **有 `@onready**`: 声明变量时直接赋值节点路径，代码整洁，自动处理时序问题。

只要你看到代码里有 **`$`** 符号（获取节点），前面通常都应该加上 **`@onready`**。