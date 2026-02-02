在 Godot 4 中，`Vector2i` 是一个非常基础但也非常重要的数据类型。

**一句话解释：**
它是 **`Vector2` 的“整数版”**。其中的 `i` 代表 **Integer（整数）**。

它只包含两个整数成员：`x` 和 `y`。它**不能**存储小数（浮点数）。

---

### 1. 核心区别：Vector2 vs Vector2i

为了理解它，我们必须把它和大哥 `Vector2` 做对比：

| 特性 | **Vector2** (标准版) | **Vector2i** (整数版) |
| --- | --- | --- |
| **数据类型** | 浮点数 (Float) | 整数 (Integer) |
| **示例** | `(1.5, 3.14)` | `(1, 3)` |
| **主要用途** | 物理移动、世界坐标、速度、方向 | 网格坐标(TileMap)、数组索引、像素点 |
| **精度** | 平滑连续 | 离散（一格一格的） |
| **Unity 对应** | `Vector2` | `Vector2Int` |

### 2. 为什么要专门搞一个 Vector2i？

在游戏开发中，有两种截然不同的坐标系需求：

1. **世界也是平滑的**：玩家的位置是 `(102.5, 50.3)`。这时候必须用 `Vector2`。
2. **网格是离散的**：你在下棋，或者在做瓦片地图（TileMap）。你只能说“我在第 3 行第 5 列”，你不能说“我在第 3.5 行”。

**`Vector2i` 就是专门为了解决第 2 种情况而生的。**

### 3. 最常见的应用场景：TileMap (瓦片地图)

这是你以后用到 `Vector2i` 最多的地方。

在 Godot 4 中，**瓦片地图的坐标（Grid Coordinates）强制使用 `Vector2i**`。

* **世界坐标 (Global Position)**：指屏幕上的像素位置，如 `(100.5, 200.0)` —— **类型是 Vector2**。
* **地图坐标 (Map Coordinates)**：指格子在第几行第几列，如 `(3, 5)` —— **类型是 Vector2i**。

**代码示例：**

```gdscript
extends Node2D

@onready var tile_map = $TileMapLayer

func _input(event):
    if event is InputEventMouseButton and event.pressed:
        # 1. 获取鼠标点击的世界位置 (Vector2)
        var global_pos = get_global_mouse_position() 
        print("世界坐标: ", global_pos) # 输出: (150.5, 203.2)
        
        # 2. 将世界坐标转换为地图网格坐标 (Vector2 -> Vector2i)
        # 注意：local_to_map 接收 Vector2，返回 Vector2i
        var grid_pos = tile_map.local_to_map(global_pos)
        print("格子坐标: ", grid_pos)   # 输出: (3, 5)
        
        # 3. 如果要修改这个格子，必须用 Vector2i
        tile_map.set_cell(grid_pos, 0, Vector2i(0, 0))

```

### 4. 重要的注意事项（坑点）

#### A. 自动取整（丢弃小数）

当你把 `Vector2` 强转为 `Vector2i` 时，小数部分会被**直接切掉**（不是四舍五入，是向下取整）。

```gdscript
var v_float = Vector2(1.9, 2.9)
var v_int = Vector2i(v_float)
print(v_int) # 输出 (1, 2)，而不是 (2, 3)

```

#### B. 数学运算的区别

`Vector2i` 参与除法运算时，结果也是整数。

```gdscript
var a = Vector2i(5, 5)
var b = a / 2
print(b) # 输出 (2, 2)，因为 2.5 的小数部分被丢弃了

```

### 总结

* 如果你在处理**移动、旋转、物理、UI位置** -> 用 `Vector2`。
* 如果你在处理**TileMap格子、二维数组索引、像素点遍历** -> 用 `Vector2i`。
* 如果你以前用 Unity，它就是 `Vector2Int`。