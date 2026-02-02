在 Godot 中，`Rect2` 是 **2D 轴对齐矩形 (2D Axis-Aligned Bounding Box, AABB)** 的数学表示。

对于你的技术背景，最直接的定义是：
**`Rect2` = `Vector2` (位置) + `Vector2` (大小)**

它是一个由 **4 个浮点数** 组成的值类型结构体。

---

### 1. 核心数据结构

它的内存布局非常简单，本质上就是两个向量：

1. **`position`**: 矩形**左上角**的坐标 (x, y)。
2. **`size`**: 矩形的**宽度和高度** (width, height)。

```gdscript
var rect = Rect2(100, 100, 50, 50)
# 内存中等于：
# position.x = 100
# position.y = 100
# size.x = 50
# size.y = 50

```

> **注意：** 在 Godot 的 2D 坐标系中，**Y 轴是向下的**。所以 `position` 通常是矩形最“上方”且最“左侧”的点。

---

### 2. 为什么叫“轴对齐” (Axis-Aligned)？

这是 `Rect2` 最重要的限制，也是很多从 Unity 转过来的开发者容易踩的坑：

* **Rect2 不能旋转。**
* 它的边永远平行于 X 轴和 Y 轴。

如果你有一个旋转了 45 度的物体，它的 `Rect2` 并不是跟着旋转的矩形，而是能够**完全包裹住这个旋转物体的最小外接矩形（AABB）**。

---

### 3. 常用场景

虽然它不能旋转，但它在以下场景中是**性能之王**（因为计算通过简单的 `<` `>` 比较即可完成，无需三角函数）：

#### A. UI 布局与点击检测

所有的 UI 控件（Control 节点）底层都由 `Rect2` 定义边界。判断鼠标是否点中按钮，本质就是：

```gdscript
# has_point 是 Rect2 最常用的方法
if button_rect.has_point(mouse_position):
    print("点击到了！")

```

#### B. 屏幕可见性检测 (VisibilityEnabler)

判断一个怪物是否在屏幕内（相机视野内）：

```gdscript
var viewport_rect = get_viewport_rect() # 获取屏幕的 Rect2
if viewport_rect.intersects(monster_rect):
    monster.process_mode = Node.PROCESS_MODE_INHERIT # 唤醒怪物
else:
    monster.process_mode = Node.PROCESS_MODE_DISABLED # 休眠怪物

```

#### C. 简单的碰撞预判

在进行复杂的物理引擎运算前，先用 `Rect2` 做一次粗略筛选（Broad-phase）。如果两个物体的 AABB 都不相交，就根本不需要计算精确的多边形碰撞。

---

### 4. 常用属性与方法

除了 `position` 和 `size`，它还有几个很方便的计算属性：

* **`end`**: 矩形右下角的坐标。
* *公式：* `position + size`


* **`center`**: 矩形的中心点坐标。
* *公式：* `position + (size / 2)`


* **`area`**: 面积。

**常用 API：**

* `has_point(point)`: 点是否在矩形内。
* `intersects(rect)`: 两个矩形是否重叠。
* `encloses(rect)`: 一个矩形是否完全包含另一个矩形。
* `expand(point)`: 返回一个新的矩形，这个新矩形扩大到包含了指定的点（常用于动态计算包围盒）。
* `grow(amount)`: 向四周扩大指定数值。

---

### 5. Rect2 vs Rect2i

就像 `Vector2` 有整数版一样，`Rect2` 也有整数版 **`Rect2i`**。

* **Rect2**: 使用浮点数。用于世界物理对象、UI、相机。
* **Rect2i**: 使用整数。**专用于网格操作**。比如你需要截取 TileMap 上的一块区域（例如从 (0,0) 到 (10,10) 的所有格子），你会使用 `Rect2i`。

### 总结

* **它是矩形。**
* **它不能旋转。**
* **它由“左上角位置”和“宽高”定义。**
* 它是 UI 系统和基础物理检测的基石。