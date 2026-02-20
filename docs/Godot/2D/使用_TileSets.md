---
sidebar_position: 10
---

Tilemap（瓦片地图）是用于创建游戏布局的瓦片网格。使用 [TileMapLayer](https://docs.godotengine.org/en/stable/classes/class_tilemaplayer.html#class-tilemaplayer) 节点设计关卡有几个好处。首先，它们允许你通过在网格上“绘制”瓦片来绘制布局，这比一个一个放置 [Sprite2D](https://docs.godotengine.org/en/stable/classes/class_sprite2d.html#class-sprite2d) 节点要快得多。其次，它们支持更大的关卡，因为它们针对绘制大量瓦片进行了优化。最后，它们允许你通过碰撞、遮挡和导航形状为瓦片添加更强大的功能。

要使用 TileMapLayer 节点，你需要先创建一个 TileSet（瓦片集）。TileSet 是可以放置在 TileMapLayer 节点中的瓦片集合。创建 TileSet 后，你将能够[使用 TileMap 编辑器](https://docs.godotengine.org/en/stable/tutorials/2d/using_tilemaps.html#doc-using-tilemaps)来放置它们。

要遵循本指南，你需要一张包含瓦片的图像，其中每个瓦片的大小都相同（大型对象可以拆分为多个瓦片）。这种图像被称为 *tilesheet*（瓦片集贴图）。瓦片不必是正方形的：它们可以是矩形、六边形或等轴测（伪 3D 透视）。

## 创建新的 TileSet

### 使用瓦片集贴图 (tilesheet)

本演示将使用取自 [Kenney 的 “Abstract Platformer” 包](https://kenney.nl/assets/abstract-platformer) 的瓦片。我们将使用该集合中的这张 *tilesheet*：

![包含 64×64 瓦片的瓦片集贴图示例](https://docs.godotengine.org/en/stable/_images/using_tilesets_kenney_abstract_platformer_tile_sheet.webp)

包含 64×64 瓦片的瓦片集贴图。来源：[Kenney](https://kenney.nl/assets/abstract-platformer)

创建一个新的 **TileMapLayer** 节点，然后选中它并在检查器中创建一个新的 TileSet 资源：

![在 TileMapLayer 节点内创建新的 TileSet 资源](https://docs.godotengine.org/en/stable/_images/using_tilesets_create_new_tileset.webp)

创建 TileSet 资源后，点击该值在检查器中展开它。默认的瓦片形状是 Square（正方形），但你也可以选择 Isometric（等轴测）、Half-Offset Square（半偏移正方形）或 Hexagon（六边形）（取决于你的瓦片图像形状）。如果使用正方形以外的瓦片形状，你可能还需要调整 **Tile Layout**（瓦片布局）和 **Tile Offset Axis**（瓦片偏移轴）属性。最后，如果你希望瓦片被其瓦片坐标裁剪，启用 **Rendering > UV Clipping**（渲染 > UV 裁剪）属性可能会很有用。这可以确保瓦片不会绘制在瓦片集贴图上分配给它们的区域之外。

在检查器中将瓦片大小设置为 64×64，以匹配示例瓦片集贴图：

![将瓦片大小设置为 64×64 以匹配示例瓦片集贴图](https://docs.godotengine.org/en/stable/_images/using_tilesets_specify_size_then_edit.webp)

如果你依赖于自动创建瓦片（就像我们即将在这里做的那样），你必须在创建 *atlas*（图集）**之前**设置瓦片大小。图集将决定瓦片集贴图中的哪些瓦片可以添加到 TileMapLayer 节点（因为图像的每个部分不一定都是有效的瓦片）。

打开编辑器底部的 **TileSet** 面板，然后点击并将瓦片集贴图图像拖动到面板上。系统会询问是否自动创建瓦片。选择 **Yes**（是）：

![根据瓦片集贴图图像内容自动创建瓦片](https://docs.godotengine.org/en/stable/_images/using_tilesets_create_tiles_automatically.webp)

这将根据你之前在 TileSet 资源中指定的瓦片大小自动创建瓦片。这极大地加快了初始瓦片设置。

> **注意**：当使用基于图像内容的自动瓦片生成时，瓦片集贴图中*完全*透明的部分将不会生成瓦片。

如果瓦片集贴图中有你不希望出现在图集中的瓦片，请选择瓦片集预览顶部的 Eraser（橡皮擦）工具，然后点击你希望移除的瓦片：

![使用橡皮擦工具从 TileSet 图集中移除不需要的瓦片](https://docs.godotengine.org/en/stable/_images/using_tilesets_eraser_tool.webp)

你也可以右键点击瓦片并选择 **Delete**（删除），作为橡皮擦工具的替代方案。

> **提示**：与在 2D 和 TileMap 编辑器中一样，你可以使用鼠标中键或右键在 TileSet 面板上平移，并使用鼠标滚轮或左上角的按钮进行缩放。

如果你希望从多个瓦片集贴图图像中获取单个 TileSet 的瓦片，请创建额外的图集并为每个图集分配纹理。通过这种方式，也可以每个瓦片使用一张图像（尽管为了更好的易用性，建议使用瓦片集贴图）。

你可以在中间列调整图集的属性：

![在专用检查器（TileSet 面板的一部分）中调整 TileSet 图集属性](https://docs.godotengine.org/en/stable/_images/using_tilesets_properties.webp)

可以在图集上调整以下属性：

*   **ID：** 标识符（在此 TileSet 中唯一），用于排序。
*   **Name（名称）：** 图集的人类可读名称。在此处使用描述性名称以便于组织（如 “terrain”、“decoration” 等）。
*   **Margins（边距）：** 图像边缘不应被选为瓦片的边距（以像素为单位）。如果你下载的瓦片集贴图图像边缘有边距（例如为了标注出处），增加此项可能会很有用。
*   **Separation（间距）：** 图集上每个瓦片之间的间距（以像素为单位）。如果你使用的瓦片集贴图图像包含引导线（例如每个瓦片之间的轮廓线），增加此项可能会很有用。
*   **Texture Region Size（纹理区域大小）：** 图集上每个瓦片的大小（以像素为单位）。在大多数情况下，这应与 TileMapLayer 属性中定义的瓦片大小相匹配（尽管这并非严格必要）。
*   **Use Texture Padding（使用纹理填充）：** 如果勾选，则在每个瓦片周围添加 1 像素的透明边缘，以防止启用过滤时出现纹理渗漏（texture bleeding）。建议保持启用状态，除非你因纹理填充而遇到渲染问题。

请注意，更改纹理边距、间距和区域大小可能会导致瓦片丢失（因为其中一些瓦片将位于图集图像坐标之外）。要从瓦片集贴图重新自动生成瓦片，请使用 TileSet 编辑器顶部的三个垂直点菜单按钮，并选择 **Create Tiles in Non-Transparent Texture Regions**（在非透明纹理区域中创建瓦片）：

![在更改图集属性后自动重新创建瓦片](https://docs.godotengine.org/en/stable/_images/using_tilesets_recreate_tiles_automatically.webp)

### 使用场景集合

你还可以将实际的*场景*作为瓦片放置。这允许你将任何节点集合用作瓦片。例如，你可以使用场景瓦片放置游戏元素，例如玩家可以与之交互的商店。你还可以使用场景瓦片放置 AudioStreamPlayer2D（用于环境音效）、粒子效果等。

> **警告**：与图集相比，场景瓦片的性能开销更大，因为每个放置的瓦片都会单独实例化一个场景。
>
> 建议仅在必要时使用场景瓦片。要在瓦片中绘制精灵而不需要任何形式的高级操作，请[改为使用图集](https://docs.godotengine.org/en/stable/tutorials/2d/using_tilesets.html#doc-creating-tilesets-using-tilesheet)。

对于本示例，我们将创建一个包含 CPUParticles2D 根节点的场景。将此场景保存为场景文件（与包含 TileMapLayer 的场景分开），然后切换回包含 TileMapLayer 节点的场景。打开 TileSet 编辑器，并在左侧列中创建一个新的 **Scenes Collection**（场景集合）：

![在 TileSet 编辑器中创建场景集合](https://docs.godotengine.org/en/stable/_images/using_tilesets_creating_scene_collection.webp)

创建场景集合后，如果愿意，可以在中间列为该场景集合输入描述性名称。选择此场景集合，然后创建一个新的场景插槽：

![在 TileSet 编辑器中选择场景集合后创建场景瓦片](https://docs.godotengine.org/en/stable/_images/using_tilesets_scene_collection_create_scene_tile.webp)

在右侧列中选择此场景插槽，然后使用 **Quick Load**（快速加载）或 **Load**（加载）来加载包含粒子的场景文件：

![创建一个场景插槽，然后在 TileSet 编辑器中将场景文件加载到其中](https://docs.godotengine.org/en/stable/_images/using_tilesets_adding_scene_tile.webp)

你现在在 TileSet 中就有了一个场景瓦片。一旦你切换到 TileMap 编辑器，你将能够从场景集合中选择它并像任何其他瓦片一样进行绘制。

## 将多个图集合并为单个图集

在单个 TileSet 资源中使用多个图集有时很有用，但在某些情况下也可能很麻烦（特别是如果你每个瓦片使用一张图像）。Godot 允许你将多个图集合并为单个图集，以便于组织。

为此，你必须在 TileSet 资源中创建了多个图集。使用位于图集列表底部的“三个垂直点”菜单按钮，然后选择 **Open Atlas Merging Tool**（打开图集合并工具）：

![创建多个图集后打开图集合并工具](https://docs.godotengine.org/en/stable/_images/using_tilesets_open_atlas_merging_tool.webp)

这将打开一个对话框，你可以在其中通过按住 Shift 或 Ctrl 并点击多个元素来选择多个图集：

![使用图集合并工具对话框](https://docs.godotengine.org/en/stable/_images/using_tilesets_atlas_merging_tool_dialog.webp)

选择 **Merge**（合并）将选定的图集合并为单个图集图像（这会转换为 TileSet 中的单个图集）。未合并的图集将从 TileSet 中移除，但*原始瓦片集贴图图像将保留在文件系统中*。如果你不希望未合并的图集从 TileSet 资源中移除，请选择 **Merge (Keep Original Atlases)**（合并（保留原始图集））代替。

> **提示**：TileSet 具有*瓦片代理（tile proxies）*系统。瓦片代理是一个映射表，允许通知使用给定 TileSet 的 TileMap，一组给定的瓦片标识符应被另一组替换。
>
> 合并不同图集时会自动设置瓦片代理，但也可以使用上述“三个垂直点”菜单中的 **Manage Tile Proxies**（管理瓦片代理）对话框手动设置。
>
> 当你更改了图集 ID 或想要用另一个图集的瓦片替换一个图集的所有瓦片时，手动创建瓦片代理可能会很有用。请注意，在编辑 TileMap 时，你可以将所有单元格替换为其对应的映射值。

## 为 TileSet 添加碰撞、导航和遮挡

我们现在已成功创建了一个基本的 TileSet。我们现在可以开始在 TileMapLayer 节点中使用它，但它目前缺乏任何形式的碰撞检测。这意味着玩家和其他对象可以直接穿过地板或墙壁。

如果你使用 [2D 导航](https://docs.godotengine.org/en/stable/tutorials/navigation/navigation_introduction_2d.html#doc-navigation-overview-2d)，你还需要为瓦片定义导航多边形，以生成代理可以用于寻路的导航网格。

最后，如果你使用 [2D 灯光和阴影](https://docs.godotengine.org/en/stable/tutorials/2d/2d_lights_and_shadows.html#doc-2d-lights-and-shadows) 或 GPUParticles2D，你可能还希望你的 TileSet 能够投射阴影并与粒子发生碰撞。这需要在 TileSet 上为“实体”瓦片定义遮挡多边形（occluder polygons）。

为了能够为每个瓦片定义碰撞、导航和遮挡形状，你需要先为 TileSet 资源创建一个物理、导航或遮挡层。为此，选择 TileMapLayer 节点，在检查器中点击 TileSet 属性值进行编辑，然后展开 **Physics Layers**（物理层）并选择 **Add Element**（添加元素）：

![在 TileSet 资源检查器中（TileMapLayer 节点内）创建物理层](https://docs.godotengine.org/en/stable/_images/using_tilesets_create_physics_layer.webp)

如果你还需要导航支持，现在是创建导航层的好时机：

![在 TileSet 资源检查器中（TileMapLayer 节点内）创建导航层](https://docs.godotengine.org/en/stable/_images/using_tilesets_create_navigation_layer.webp)

如果你需要支持灯光多边形遮挡器，现在是创建遮挡层的好时机：

![在 TileSet 资源检查器中（TileMapLayer 节点内）创建遮挡层](https://docs.godotengine.org/en/stable/_images/using_tilesets_create_occlusion_layer.webp)

> **注意**：本教程的后续步骤是专门为创建碰撞多边形量身定制的，但导航和遮挡的过程非常相似。它们各自的多边形编辑器以相同的方式工作，因此为了简洁起见，不再重复这些步骤。
>
> 唯一的注意事项是，瓦片的遮挡多边形属性是图集检查器中 **Rendering**（渲染）子部分的一部分。确保展开此部分，以便编辑多边形。

创建物理层后，你可以在 TileSet 图集检查器中访问 **Physics Layer** 部分：

![在选择模式下打开碰撞编辑器](https://docs.godotengine.org/en/stable/_images/using_tilesets_selecting_collision_editor.webp)

在聚焦 TileSet 编辑器时按下 `F` 键可以快速创建矩形碰撞形状。如果键盘快捷键不起作用，请尝试点击多边形编辑器周围的空白区域以聚焦它：

![通过按下 F 键使用默认矩形碰撞形状](https://docs.godotengine.org/en/stable/_images/using_tilesets_using_default_rectangle_collision.webp)

在这个瓦片碰撞编辑器中，你可以访问所有的 2D 多边形编辑工具：

*   使用多边形上方的工具栏在创建新多边形、编辑现有多边形和移除多边形点之间切换。“三个垂直点”菜单按钮提供额外选项，例如旋转和翻转多边形。
*   通过点击并拖动两点之间的线来创建新点。
*   通过右键点击一个点（或使用上述移除工具并点击左键）来移除该点。
*   通过点击鼠标中键或右键在编辑器中平移。（右键平移仅能在附近没有点的地方使用。）

你可以通过移除其中一个点，使用默认的矩形形状快速创建一个三角形碰撞形状：

![通过右键点击其中一个角将其移除，从而创建三角形碰撞形状](https://docs.godotengine.org/en/stable/_images/using_tilesets_creating_triangle_collision.webp)

你也可以通过添加更多的点，将矩形作为更复杂形状的基础：

![为复杂的瓦片形状绘制自定义碰撞](https://docs.godotengine.org/en/stable/_images/using_tilesets_drawing_custom_collision.webp)

> **提示**：如果你有一个大型瓦片集，分别为每个瓦片指定碰撞可能会花费很多时间。特别是 TileMap 往往有许多具有共同碰撞模式的瓦片（如实心块或 45 度斜坡）。要快速将类似的碰撞形状应用于多个瓦片，请使用[一次为多个瓦片分配属性](https://docs.godotengine.org/en/stable/tutorials/2d/using_tilesets.html#doc-using-tilemaps-assigning-properties-to-multiple-tiles)的功能。

## 为 TileSet 的瓦片分配自定义元数据

你可以使用 *custom data layers*（自定义数据层）为每个瓦片分配自定义数据。这对于存储特定于你的游戏的信息很有用，例如瓦片在玩家接触时应造成的伤害，或者瓦片是否可以使用武器破坏。

数据与 TileSet 中的瓦片相关联：放置的瓦片的所有实例都将使用相同的自定义数据。如果你需要创建一个具有不同自定义数据的瓦片变体，可以通过[创建替代瓦片](https://docs.godotengine.org/en/stable/tutorials/2d/using_tilesets.html#doc-using-tilesets-creating-alternative-tiles)并仅更改替代瓦片的自定义数据来实现。

![在 TileSet 资源检查器中（TileMapLayer 节点内）创建自定义数据层](https://docs.godotengine.org/en/stable/_images/using_tilesets_create_custom_data_layer.webp)

![配置了游戏特定属性的自定义数据层示例](https://docs.godotengine.org/en/stable/_images/using_tilesets_custom_data_layers_example.webp)

你可以重新排序自定义数据而不会破坏现有的元数据：重新排序自定义数据属性后，TileSet 编辑器会自动更新。

在上面显示的自定义数据层示例中，我们将一个瓦片的 `damage_per_second` 元数据设置为 `25`，并将 `destructible` 元数据设置为 `false`：

![在选择模式下的 TileSet 编辑器中编辑自定义数据](https://docs.godotengine.org/en/stable/_images/using_tilesets_edit_custom_data.webp)

[瓦片属性绘制](https://docs.godotengine.org/en/stable/tutorials/2d/using_tilesets.html#doc-using-tilemaps-using-tile-property-painting) 也可以用于自定义数据：

![使用瓦片属性绘制在 TileSet 编辑器中分配自定义数据](https://docs.godotengine.org/en/stable/_images/using_tilesets_paint_custom_data.webp)

## 创建地形集（自动铺砖）

> **注意**：此功能在 Godot 3.x 中以 *autotiling*（自动铺砖）的不同形式实现。地形（Terrains）本质上是自动铺砖的一种更强大的替代方案。与自动铺砖不同，地形可以支持从一种地形到另一种地形的过渡，因为一个瓦片可以同时定义多个地形。
>
> 与以前自动铺砖是一种特定类型的瓦片不同，地形只是分配给图集瓦片的一组属性。然后，这些属性由专用的 TileMap 绘制模式使用，该模式以智能方式选择具有地形数据的瓦片。这意味着任何地形瓦片既可以作为地形绘制，也可以像任何其他瓦片一样作为单个瓦片绘制。

一个“精致”的瓦片集通常包含一些变体，你应该在平台的角落或边缘、地板等位置使用它们。虽然这些可以手动放置，但这很快就会变得乏味。对于程序化生成的关卡，处理这种情况也可能很困难且需要大量代码。

Godot 提供 *terrains*（地形）来自动执行此类瓦片连接。这允许你自动使用“正确”的瓦片变体。

地形被分组为地形集。每个地形集都被分配了一个模式，包括 **Match Corners and Sides**（匹配角和边）、**Match Corners**（匹配角）和 **Match sides**（匹配边）。它们定义了地形集中地形如何相互匹配。

> **注意**：上述模式对应于 Godot 3.x 中自动铺砖使用的先前位掩码模式：2×2、3×3 或 3×3 minimal。这与 [Tiled](https://www.mapeditor.org/) 编辑器的功能也类似。

选择 TileMapLayer 节点，转到检查器并在 TileSet *资源* 中创建一个新的地形集：

![在 TileSet 资源检查器中（TileMapLayer 节点内）创建地形集](https://docs.godotengine.org/en/stable/_images/using_tilesets_create_terrain_set.webp)

创建地形集后，你 **必须** 在地形集中创建一个或多个地形：

![在地形集中创建地形](https://docs.godotengine.org/en/stable/_images/using_tilesets_create_terrain.webp)

在 TileSet 编辑器中，切换到 Select（选择）模式并点击一个瓦片。在中间列中，展开 **Terrains** 部分，然后为该瓦片分配地形集 ID 和地形 ID。`-1` 表示“无地形集”或“无地形”，这意味着你必须先将 **Terrain Set** 设置为 `0` 或更大，然后才能将 **Terrain** 设置为 `0` 或更大。

> **注意**：地形集 ID 和地形 ID 相互独立。它们也从 `0` 而不是 `1` 开始。

![在 TileSet 编辑器的选择模式下配置单个瓦片上的地形](https://docs.godotengine.org/en/stable/_images/using_tilesets_configure_terrain_on_tile.webp)

完成后，你现在可以配置 **Terrain Peering Bits** 部分，该部分在中间列中变得可见。窥视位决定了根据相邻瓦片放置哪个瓦片。`-1` 是一个特殊值，指代空白区域。

例如，如果一个瓦片的所有位都设置为 `0` 或更大，则只有在*所有* 8 个相邻瓦片都使用具有相同地形 ID 的瓦片时，它才会出现。如果一个瓦片的位设置为 `0` 或更大，但左上、上方和右上位设置为 `-1`，则只有在其上方有空白区域（包括对角线方向）时，它才会出现。

![在 TileSet 编辑器的选择模式下配置单个瓦片上的地形窥视位](https://docs.godotengine.org/en/stable/_images/using_tilesets_configure_terrain_peering_bits.webp)

完整瓦片集贴图的一个示例配置可能如下所示：

![侧向滚动游戏的完整瓦片集贴图示例](https://docs.godotengine.org/en/stable/_images/using_tilesets_terrain_example_tilesheet.webp)

![侧向滚动游戏的完整瓦片集贴图示例，带有可见的地形窥视位](https://docs.godotengine.org/en/stable/_images/using_tilesets_terrain_example_tilesheet_configuration.webp)

## 一次为多个瓦片分配属性

有两种方法可以一次为多个瓦片分配属性。根据你的用例，一种方法可能比另一种更快：

### 使用多瓦片选择

如果你希望一次在多个瓦片上配置各种属性，请选择 TileSet 编辑器顶部的 **Select**（选择）模式：

完成此操作后，你可以通过按住 Shift 并点击瓦片在右侧列中选择多个瓦片。你还可以通过按住鼠标左键并拖动鼠标来执行矩形选择。最后，你可以通过按住 Shift 并点击已选中的瓦片来取消选择该瓦片（不影响其余选中的瓦片）。

然后，你可以使用 TileSet 编辑器中间列的检查器分配属性。只有在此处更改的属性才会应用于所有选中的瓦片。与编辑器的检查器一样，选中瓦片上不同的属性将保持不同，直到你编辑它们。

对于数值和颜色属性，在编辑属性后，你还将在图集中的所有瓦片上看到该属性值的预览：

![使用选择模式选择多个瓦片，然后应用属性](https://docs.godotengine.org/en/stable/_images/using_tilesets_select_and_set_tile_properties.webp)

### 使用瓦片属性绘制

如果你希望一次将单个属性应用于多个瓦片，可以使用 *property painting*（属性绘制）模式。

在中间列配置要绘制的属性，然后点击右侧列中的瓦片（或按住鼠标左键）以将属性“绘制”到瓦片上。

![使用 TileSet 编辑器绘制瓦片属性](https://docs.godotengine.org/en/stable/_images/using_tilesets_paint_tile_properties.webp)

瓦片属性绘制对于设置耗时的属性（如碰撞形状）特别有用：

![绘制碰撞多边形，然后点击左键应用到瓦片上](https://docs.godotengine.org/en/stable/_images/using_tilesets_paint_tile_properties_collision.webp)

## 创建替代瓦片

有时，你希望使用单个瓦片图像（在图集中仅出现一次），但以不同的方式配置。例如，你可能想使用相同的瓦片图像，但旋转、翻转或调制不同的颜色。这可以使用 *alternative tiles*（替代瓦片）来完成。

> **提示**：自 Godot 4.2 起，你不再需要通过创建替代瓦片来旋转或翻转瓦片了。在 TileMap 编辑器中放置任何瓦片时，你可以使用 TileMap 编辑器工具栏中的旋转/翻转按钮进行旋转。

要创建替代瓦片，请在 TileSet 编辑器显示的图集中右键点击一个基础瓦片，然后选择 **Create an Alternative Tile**（创建替代瓦片）：

![通过在 TileSet 编辑器中右键点击基础瓦片来创建替代瓦片](https://docs.godotengine.org/en/stable/_images/using_tilesets_create_alternative_tile.webp)

如果当前处于选择模式，替代瓦片将已被选中以便编辑。如果当前未处于选择模式，你仍然可以创建替代瓦片，但你需要切换到选择模式并选中该替代瓦片才能编辑它。

如果你没看到替代瓦片，请向右平移图集图像，因为在 TileSet 编辑器中，替代瓦片总是出现在给定图集的基础瓦片右侧：

![在 TileSet 编辑器中点击替代瓦片后对其进行配置](https://docs.godotengine.org/en/stable/_images/using_tilesets_configure_alternative_tile.webp)

选中替代瓦片后，你可以像在基础瓦片上一样使用中间列更改任何属性。但是，公开的属性列表与基础瓦片相比有所不同：

*   **Alternative ID：** 此替代瓦片的唯一数字标识符。更改它会破坏现有的 TileMap，所以要小心！此 ID 还控制编辑器中显示的替代瓦片列表的排序。
*   **Rendering > Flip H（渲染 > 水平翻转）：** 如果为 `true`，则瓦片被水平翻转。
*   **Rendering > Flip V（渲染 > 垂直翻转）：** 如果为 `true`，则瓦片被垂直翻转。
*   **Rendering > Transpose（渲染 > 转置）：** 如果为 `true`，则瓦片会*逆时针*旋转 90 度，然后垂直翻转。实际上，这意味着要在不翻转的情况下顺时针旋转瓦片 90 度，你应该启用 **Flip H** 和 **Transpose**。要顺时针旋转瓦片 180 度，启用 **Flip H** 和 **Flip V**。要顺时针旋转瓦片 270 度，启用 **Flip V** 和 **Transpose**。
*   **Rendering > Texture Origin（渲染 > 纹理原点）：** 用于绘制瓦片的元点。这可以用来使瓦片视觉上相对于基础瓦片发生偏移。
*   **Rendering > Modulate（渲染 > 调制）：** 渲染瓦片时使用的颜色乘数。
*   **Rendering > Material（渲染 > 材质）：** 用于此瓦片的材质。这可以用来对单个瓦片应用不同的混合模式或自定义着色器。
*   **Z Index（Z 索引）：** 此瓦片的排序顺序。较高的值将使瓦片渲染在同一层其他瓦片的前面。
*   **Y Sort Origin（Y 排序原点）：** 根据瓦片的 Y 坐标进行瓦片排序时使用的垂直偏移量（以像素为单位）。这允许像顶视游戏那样使用图层，就好像它们在不同的高度一样。调整此项可以帮助缓解某些瓦片的排序问题。仅当 TileMapLayer 节点下 **CanvasItem > Ordering** 中的 **Y Sort Enabled** 为 `true` 时有效。

你可以通过点击替代瓦片旁边的巨大 “+” 图标来创建额外的替代瓦片变体。这相当于选中基础瓦片并再次右键点击选择 **Create an Alternative Tile**。

> **注意**：创建替代瓦片时，不会继承基础瓦片的任何属性。如果你希望基础瓦片和替代瓦片的属性相同，必须在替代瓦片上再次设置这些属性。
