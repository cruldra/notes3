---
sidebar_position: 11
---
:::info 另请参阅
本页假设你已经创建或下载了一个 TileSet（瓦片集）。如果没有，请先阅读[使用 TileSets](./使用_TileSets.md)，因为创建 TileMap 需要瓦片集。
:::

## 简介

Tilemap（瓦片地图）是用于创建游戏布局的瓦片网格。使用 [TileMapLayer](https://docs.godotengine.org/en/stable/classes/class_tilemaplayer.html#class-tilemaplayer) 节点设计关卡有几个好处。首先，它们允许你通过在网格上“绘制”瓦片来绘制布局，这比一个一个放置 [Sprite2D](https://docs.godotengine.org/en/stable/classes/class_sprite2d.html#class-sprite2d) 节点要快得多。其次，它们支持更大的关卡，因为它们针对绘制大量瓦片进行了优化。最后，你可以为瓦片添加碰撞、遮挡和导航形状，从而为 TileMap 增添更多功能。

## 在 TileMapLayer 中指定 TileSet

如果你已经阅读了上一页关于[使用 TileSets](./使用_TileSets.md)的内容，你应该已经有了一个内置在 TileMapLayer 节点中的 TileSet 资源。这对于原型设计很有用，但在实际项目中，你通常会有多个关卡复用同一个瓦片集。

在多个 TileMapLayer 节点中复用同一个 TileSet 的推荐方式是将 TileSet 保存为外部资源。为此，点击 TileSet 资源旁边的下拉菜单并选择 **Save（保存）**：

![将内置 TileSet 资源保存到外部资源文件](https://docs.godotengine.org/en/stable/_images/using_tilemaps_save_tileset_to_resource.webp)

## 多个 TileMapLayer 及其设置

在使用瓦片地图时，通常建议在合适的情况下使用多个 TileMapLayer 节点。使用多个图层有很多优势，例如，这允许你区分前景瓦片和背景瓦片，以便更好地组织。你可以在给定位置的每个图层上放置一个瓦片，如果你有多个图层，这允许你将多个瓦片重叠在一起。

每个 TileMapLayer 节点都有几个可以调节的属性：

- **Enabled（启用）：** 如果为 `true`，则该图层在编辑器中和运行项目时可见。
- **TileSet（瓦片集）：** TileMapLayer 节点使用的瓦片集。

### Rendering（渲染）

- **Y Sort Origin（Y 排序原点）：** 每个瓦片用于 Y 排序的垂直偏移（以像素为单位）。仅当 CanvasItem 设置下的 **Y Sort Enabled（启用 Y 排序）** 为 `true` 时有效。
- **X Draw Order Reversed（X 绘制顺序反转）：** 反转瓦片在 X 轴上的绘制顺序。需要 CanvasItem 设置下的 **Y Sort Enabled（启用 Y 排序）** 为 `true`。
- **Rendering Quadrant Size（渲染象限大小）：** 象限是出于优化目的而在单个 CanvasItem 上一起绘制的一组瓦片。此设置定义了地图坐标系中正方形边长的长度。象限大小不适用于按 Y 排序的 TileMapLayer，因为在这种情况下瓦片是按 Y 位置分组的。

### Physics（物理）

- **Collision Enabled（启用碰撞）：** 启用或禁用碰撞。
- **Use Kinematic Bodies（使用运动学身体）：** 为 true 时，TileMapLayer 的碰撞形状将作为运动学身体实例化。
- **Collision Visibility Mode（碰撞可见性模式）：** TileMapLayer 的碰撞形状是否可见。如果设置为默认，则取决于显示碰撞调试设置。

### Navigation（导航）

- **Navigation Enabled（启用导航）：** 导航区域是否启用。
- **Navigation Visible（导航可见）：** TileMapLayer 的导航网格是否可见。如果设置为默认，则取决于显示导航调试设置。

:::tip 提示
TileMap 的内置导航有许多实际限制，会导致路径搜索性能和路径跟随质量下降。

设计完 TileMap 后，考虑使用 [NavigationRegion2D](https://docs.godotengine.org/en/stable/classes/class_navigationregion2d.html#class-navigationregion2d) 或 [NavigationServer2D](https://docs.godotengine.org/en/stable/classes/class_navigationserver2d.html#class-navigationserver2d) 将其烘焙到更优化的导航网格中（并禁用 TileMap 的 NavigationLayer）。有关更多信息，请参阅[使用导航网格](https://docs.godotengine.org/en/stable/tutorials/navigation/navigation_using_navigationmeshes.html#doc-navigation-using-navigationmeshes)。
:::

:::warning 警告
2D 导航网格不能像视觉效果或物理形状那样“分层”或堆叠。尝试在同一个导航地图上堆叠导航网格将导致合并和逻辑错误，从而破坏路径搜索。
:::

### 重新排列图层

你可以通过在“场景”选项卡中拖放图层节点来重新排列图层。你还可以使用 TileMap 编辑器右上角的按钮在正在操作的 TileMapLayer 节点之间进行切换。

> **注意**：你可以在以后创建、重命名或重新排列图层，而不会影响现有的瓦片。但要小心，因为*移除*一个图层也将移除放置在该图层上的所有瓦片。

## 打开 TileMap 编辑器

选中 TileMapLayer 节点，然后打开编辑器底部的 TileMap 面板：

![打开编辑器底部的 TileMap 面板。必须先选中 TileMapLayer 节点。](https://docs.godotengine.org/en/stable/_images/using_tilemaps_open_tilemap_editor.webp)

## 选择用于绘制的瓦片

首先，如果你在上方创建了额外的图层，请确保你已经选中了你想要绘制的图层：

![在 TileMap 编辑器中选择要绘制的图层](https://docs.godotengine.org/en/stable/_images/using_tilemaps_select_layer.webp)

:::tip 提示
在 2D 编辑器中，在你处于 TileMap 编辑器期间，你当前未编辑的来自同一个 TileMapLayer 节点的图层将显示为灰色。你可以通过点击图层选择菜单旁边的图标（工具提示为 **Highlight Selected TileMap Layer（高亮选中的 TileMap 图层）**）来禁用此行为。
:::

如果你没有创建额外的图层，可以跳过上述步骤，因为进入 TileMap 编辑器时会自动选中第一个图层。

在 2D 编辑器中放置瓦片之前，你必须在位于编辑器底部的 TileMap 面板中选中一个或多个瓦片。为此，请点击 TileMap 面板中的一个瓦片，或按住鼠标按钮选择多个瓦片：

![点击 TileMap 编辑器中的一个瓦片来选中它](https://docs.godotengine.org/en/stable/_images/using_tilemaps_select_single_tile_from_tileset.webp)

:::tip 提示
与在 2D 和 TileSet 编辑器中一样，你可以使用鼠标中键或右键在 TileMap 面板上平移，并使用鼠标滚轮或左上角的按钮进行缩放。
:::

你还可以按住 Shift 键将其追加到当前选区。选中多个瓦片时，每次执行绘制操作都会放置多个瓦片。这可用于单击一次即可绘制由多个瓦片组成的结构（如大型平台或树木）。

最终的选区不必是连续的：如果选中的瓦片之间有空隙，在 2D 编辑器中绘制的模式中这些空隙也将保持为空白。

![按住鼠标左键在 TileMap 编辑器中选中多个瓦片](https://docs.godotengine.org/en/stable/_images/using_tilemaps_select_multiple_tiles_from_tileset.webp)

如果你在瓦片集中创建了替代瓦片，你可以选择它们并在基础瓦片的右侧进行绘制：

![在 TileMap 编辑器中选中一个替代瓦片](https://docs.godotengine.org/en/stable/_images/using_tilemaps_use_alternative_tile.webp)

最后，如果你在瓦片集中创建了*场景集合*，你可以在 TileMap 中放置场景瓦片：

![使用 TileMap 编辑器放置一个包含粒子的场景瓦片](https://docs.godotengine.org/en/stable/_images/using_tilemaps_placing_scene_tiles.webp)

## 绘制模式和工具

使用 TileMap 编辑器顶部的工具栏，你可以在几种绘制模式和工具之间进行选择。这些模式影响在 2D 编辑器中点击时的操作，**不**影响 TileMap 面板本身。

从左到右，你可以选择的绘制模式和工具包括：

### Selection（选择）

通过点击单个瓦片来选中瓦片，或者在 2D 编辑器中按住鼠标左键通过矩形框选多个瓦片。注意空地不能被选中：如果你进行矩形框选，只有非空瓦片会被选中。

要追加到当前选区，请按住 Shift 键然后选择瓦片。要从当前选区中移除，请按住 Ctrl 键然后选择瓦片。

选区随后可用于任何其他绘制模式，以快速创建已放置模式的副本。

你可以通过按 Del 键从 TileMap 中移除选中的瓦片。

在绘制模式下，你可以通过按住 Ctrl 键并执行选择来临时切换到此模式。

:::tip 提示
你可以通过执行选择、按 Ctrl + C 然后按 Ctrl + V 来复制并粘贴已经放置的瓦片。粘贴将在左键点击后执行。你可以再次按 Ctrl + V 以这种方式执行更多复制。右键点击或按 Escape 键取消粘贴。
:::

### Paint（绘制）

标准的绘制模式允许你通过点击或按住鼠标左键来放置瓦片。

如果你右键点击，当前选中的瓦片将从瓦片地图中擦除。换句话说，它将被替换为空地。

如果你在 TileMap 中或使用选择工具选中了多个瓦片，每次你点击或在按住鼠标左键的同时拖动鼠标时，它们都会被放置。

:::tip 提示
在绘制模式下，你可以通过在按住鼠标左键*之前*按住 Shift 键，然后将鼠标拖动到线的终点来画一条线。这与使用下文所述的直线工具相同。

你也可以通过在按住鼠标左键*之前*按住 Ctrl 和 Shift 键，然后将鼠标拖动到矩形的终点来画一个矩形。这与使用下文所述的矩形工具相同。

最后，你可以通过按住 Ctrl 键然后点击一个瓦片（或按住并拖动鼠标）来拾取 2D 编辑器中现有的瓦片。这将把你当前绘制的瓦片切换为你刚刚点击的瓦片。这与使用下文所述的拾取器工具相同。
:::

### Line（直线）

选择直线绘制模式后，你可以画一条始终为 1 个瓦片厚的直线（无论其方向如何）。

如果在直线绘制模式下右键点击，你将在线上擦除。

如果你在 TileMap 中或使用选择工具选中了多个瓦片，你可以将它们以重复模式放置在整条线上。

在绘制或擦除模式下，你可以通过按住 Shift 键然后绘图来临时切换到此模式。

![在选中两个瓦片后使用直线工具斜着绘制平台](https://docs.godotengine.org/en/stable/_images/using_tilesets_line_tool_multiple_tiles.webp)

### Rectangle（矩形）

选择矩形绘制模式后，你可以画一个轴对齐的矩形。

如果在矩形绘制模式下右键点击，你将在一个轴对齐的矩形内擦除。

如果你在 TileMap 中或使用选择工具选中了多个瓦片，你可以将它们以重复模式放置在矩形内。

在绘制或擦除模式下，你可以通过按住 Ctrl 和 Shift 键然后绘图来临时切换到此模式。

### Bucket Fill（油漆桶填充）

选择油漆桶填充模式后，你可以通过切换工具栏右侧出现的 **Contiguous（连续）** 复选框来选择绘制是否应仅限于连续区域。

如果你启用 **Contiguous（连续）**（默认），则只有与当前选区接触的匹配瓦片才会被替换。这种连续检查是在水平和垂直方向上进行的，但*不是*对角线方向。

如果你禁用 **Contiguous（连续）**，整个 TileMap 中所有具有相同 ID 的瓦片都将被当前选中的瓦片替换。如果在未勾选 **Contiguous（连续）** 的情况下选择一个空瓦片，那么包含 TileMap 有效区域的矩形内的所有瓦片都将被替换。

如果在油漆桶填充模式下右键点击，你将用空瓦片替换匹配的瓦片。

如果你在 TileMap 中或使用选择工具选中了多个瓦片，你可以将它们以重复模式放置在填充区域内。

![使用油漆桶填充工具](https://docs.godotengine.org/en/stable/_images/using_tilemaps_bucket_fill.webp)

### Picker（拾取器）

选择拾取器模式后，你可以通过按住 Ctrl 键然后点击瓦片来拾取 2D 编辑器中现有的瓦片。这将把你当前绘制的瓦片切换为你刚刚点击的瓦片。你还可以通过按住鼠标左键并形成矩形选区来一次拾取多个瓦片。只有非空瓦片可以被拾取。

在绘制模式下，你可以通过按住 Ctrl 键然后点击或拖动鼠标来临时切换到此模式。

### Eraser（擦除器）

此模式可与任何其他绘制模式（绘制、直线、矩形、油漆桶填充）结合使用。启用擦除器模式后，左键点击时瓦片将被空瓦片替换，而不是绘制新线。

在任何其他模式下，你都可以通过右键点击而不是左键点击来临时切换到此模式。

## 使用散射进行随机绘制

在绘制时，你可以选择启用*随机化（randomization）*。启用后，在绘制时将从所有当前选中的瓦片中随机选择一个。绘制、直线、矩形和油漆桶填充工具都支持此功能。为了获得有效的绘制随机化，你必须在 TileMap 编辑器中选中多个瓦片或使用散布（scattering）（这两种方法可以结合使用）。

如果 **Scattering（散布）** 设置为大于 0 的值，则在绘制时有可能不会放置任何瓦片。这可用于向大区域添加偶然的、不重复的细节（例如在大型顶视 TileMap 上添加草丛或碎屑）。

使用绘制模式时的示例：

![从多个时间中选择以随机挑选，然后通过按住鼠标左键进行绘制](https://docs.godotengine.org/en/stable/_images/using_tilemaps_scatter_tiles.webp)

使用油漆桶填充模式时的示例：

![在仅使用单个瓦片但启用了随机化和散布的情况下使用油漆桶填充工具](https://docs.godotengine.org/en/stable/_images/using_tilemaps_bucket_fill_scatter.webp)

> **注意**：擦除器模式不考虑随机化和散布。选区内的所有瓦片总是会被移除。

## 使用瓦片模式保存和加载预制的瓦片放置

虽然你可以在选择模式下复制和粘贴瓦片，但你可能希望保存预制的瓦片*模式（patterns）*，以便一次性放置在一起。这可以通过选择 TileMap 编辑器的 **Patterns（模式）** 选项卡在每个 TileMap 的基础上完成。

要创建一个新模式，请切换到选择模式，执行选择并按 Ctrl + C。点击模式选项卡内的空白区域（空白区域周围应该会出现一个蓝色的聚焦矩形），然后按 Ctrl + V：

![从 TileMap 编辑器中的选区创建一个新模式](https://docs.godotengine.org/en/stable/_images/using_tilemaps_create_pattern.webp)

要使用现有模式，请在 **Patterns（模式）** 选项卡中点击其图像，切换到任何绘制模式，然后在 2D 编辑器中的某处左键点击：

![使用 TileMap 编辑器放置一个现有模式](https://docs.godotengine.org/en/stable/_images/using_tilemaps_use_pattern.webp)

与多瓦片选区一样，如果与直线、矩形或油漆桶填充绘制模式一起使用，模式将会重复。

> **注意**：尽管是在 TileMap 编辑器中编辑的，但模式存储在 TileSet 资源中。这允许在加载保存到外部文件的 TileSet 资源后，在不同的 TileMapLayer 节点中复用模式。

## 使用地形自动处理瓦片连接

要使用地形，TileMapLayer 节点必须包含至少一个地形集以及该地形集内的一个地形。如果你还没有为瓦片集创建地形集，请参阅[创建地形集（自动铺砖）](./使用_TileSets.md#创建地形集自动铺砖)。

地形连接有 3 种绘制模式：

- **Connect（连接）**，瓦片与同一个 TileMapLayer 上周围的瓦片连接。
- **Path（路径）**，瓦片与在同一次笔画中绘制的瓦片连接（直到松开鼠标按钮）。
- 用于解决冲突或处理地形系统未涵盖的情况的特定瓦片覆盖。

Connect 模式更容易使用，但 Path 更灵活，因为它允许在绘制过程中进行更多的艺术家控制。例如，Path 可以允许道路直接相邻而不相互连接，而 Connect 将强制两条道路连接在一起。

![在 TileMap 编辑器的“地形”选项卡中选择 Connect 模式](https://docs.godotengine.org/en/stable/_images/using_tilemaps_terrain_select_connect_mode.webp)

![在 TileMap 编辑器的“地形”选项卡中选择 Path 模式](https://docs.godotengine.org/en/stable/_images/using_tilemaps_terrain_select_path_mode.webp)

最后，你可以从地形中选择特定的瓦片来解决某些情况下的冲突：

![在 TileMap 编辑器的“地形”选项卡中使用特定瓦片绘图](https://docs.godotengine.org/en/stable/_images/using_tilemaps_terrain_paint_specific_tiles.webp)

任何至少有一个位设置为对应地形 ID 值的瓦片都将出现在可选瓦片列表中。

## 处理缺失的瓦片

如果你在瓦片集中移除了 TileMap 中引用的瓦片，TileMap 将显示一个占位符，指示放置了一个无效的瓦片 ID：

![由于 TileSet 引用断开，TileMap 编辑器中显示缺失的瓦片](https://docs.godotengine.org/en/stable/_images/using_tilemaps_missing_tiles.webp)

这些占位符在运行的项目中是**不可见**的，但瓦片数据仍然持久保存在磁盘上。这允许你安全地关闭并重新打开此类场景。一旦你重新添加一个具有匹配 ID 的瓦片，这些瓦片将以新瓦片的外观出现。

> **注意**：在你选中 TileMapLayer 节点并打开 TileMap 编辑器之前，缺失瓦片的占位符可能不可见。
