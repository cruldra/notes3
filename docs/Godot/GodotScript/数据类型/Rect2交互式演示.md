---
hide_table_of_contents: true
---
import GodotRect2Demo from '@site/src/components/GodotRect2Demo';

**操作指南：**

1. **左侧面板**：模拟 Godot 的 **Inspector (检查器)**。你可以拖动滑块修改蓝色矩形的 `position` (x, y) 和 `size` (width, height)。
2. **中间视口**：
* **蓝色矩形**：你的 `Rect2` 对象。
* **红色/绿色矩形**：一个干扰项，用来测试碰撞。拖动它来测试 `intersects()`。
* **鼠标**：移动鼠标来测试 `has_point()`。


3. **底部代码**：实时显示当前状态对应的 GDScript 代码。

### 这个演示展示了什么？

1. **Rect2 的本质**：
* 你可以清楚地看到，`Rect2` 就是简单的 `(x, y)` 加上 `(width, height)`。
* **Computed Properties（计算属性）**：注意观察面板里的 `end` 和 `center`。在 Godot 中，当你修改 `size` 时，`center` 会变；当你修改 `position` 时，`end` 也会变。这些不是存储在内存里的变量，而是实时算出来的属性。


2. **轴对齐 (Axis-Aligned)**：
* 无论你怎么拖动参数，这个矩形永远是横平竖直的。你无法在这个 API 里找到 `rotation`（旋转）属性。这就是 AABB（Axis-Aligned Bounding Box）的含义。


3. **常用 API 的可视化**：
* **`has_point(point)`**：当你把鼠标移入蓝色区域，底部的代码会变成 `true`，矩形背景也会变亮。这模拟了按钮点击检测或鼠标悬停检测。
* **`intersects(rect)`**：当你拖动那个虚线的“其他矩形”撞上蓝色矩形时，两者都会变红。这是物理引擎最基础的“Broad-phase”（粗略阶段）碰撞检测。

<GodotRect2Demo />