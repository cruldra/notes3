---
sidebar_position: 2
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

本教程将介绍使用 [ArrayMesh](https://docs.godotengine.org/en/stable/classes/class_arraymesh.html#class-arraymesh) 的基础知识。

我们将使用 [add_surface_from_arrays()](https://docs.godotengine.org/en/stable/classes/class_arraymesh.html#class-arraymesh-method-add-surface-from-arrays) 函数，它最多接受五个参数。前两个是必需的，后三个是可选的。

第一个参数是 `PrimitiveType`（图元类型），这是一个 OpenGL 概念，用于指示 GPU 如何根据给定的顶点安排图元，即它们是代表三角形、直线、点等。有关可用选项，请参阅 [Mesh.PrimitiveType](https://docs.godotengine.org/en/stable/classes/class_mesh.html#enum-mesh-primitivetype)。

第二个参数 `arrays` 是实际存储网格信息的 Array（数组）。该数组是一个普通的 Godot 数组，使用空括号 `[]` 构造。它为构建表面所用的每种信息类型存储一个 `Packed**Array`（例如 PackedVector3Array、PackedInt32Array 等）。

`arrays` 的常用元素如下表所示，以及它们在 `arrays` 中必须所处的位置。有关完整列表，请参阅 [Mesh.ArrayType](https://docs.godotengine.org/en/stable/classes/class_mesh.html#enum-mesh-arraytype)。

| 索引 | Mesh.ArrayType 枚举 | 数组类型 |
| --- | --- | --- |
| 0   | `ARRAY_VERTEX` | [PackedVector3Array](https://docs.godotengine.org/en/stable/classes/class_packedvector3array.html#class-packedvector3array)<br /> 或 [PackedVector2Array](https://docs.godotengine.org/en/stable/classes/class_packedvector2array.html#class-packedvector2array) |
| 1   | `ARRAY_NORMAL` | [PackedVector3Array](https://docs.godotengine.org/en/stable/classes/class_packedvector3array.html#class-packedvector3array) |
| 2   | `ARRAY_TANGENT` | [PackedFloat32Array](https://docs.godotengine.org/en/stable/classes/class_packedfloat32array.html#class-packedfloat32array)<br /> 或 [PackedFloat64Array](https://docs.godotengine.org/en/stable/classes/class_packedfloat64array.html#class-packedfloat64array)<br />，每组 4 个浮点数。前 3 个浮点数确定切线，最后一个浮点数确定副法线方向（为 -1 或 1）。 |
| 3   | `ARRAY_COLOR` | [PackedColorArray](https://docs.godotengine.org/en/stable/classes/class_packedcolorarray.html#class-packedcolorarray) |
| 4   | `ARRAY_TEX_UV` | [PackedVector2Array](https://docs.godotengine.org/en/stable/classes/class_packedvector2array.html#class-packedvector2array)<br /> 或 [PackedVector3Array](https://docs.godotengine.org/en/stable/classes/class_packedvector3array.html#class-packedvector3array) |
| 5   | `ARRAY_TEX_UV2` | [PackedVector2Array](https://docs.godotengine.org/en/stable/classes/class_packedvector2array.html#class-packedvector2array)<br /> 或 [PackedVector3Array](https://docs.godotengine.org/en/stable/classes/class_packedvector3array.html#class-packedvector3array) |
| 10  | `ARRAY_BONES` | [PackedFloat32Array](https://docs.godotengine.org/en/stable/classes/class_packedfloat32array.html#class-packedfloat32array)<br />（每组 4 个浮点数）或 [PackedInt32Array](https://docs.godotengine.org/en/stable/classes/class_packedint32array.html#class-packedint32array)<br />（每组 4 个整数）。每组列出影响给定顶点的 4 个骨骼的索引。 |
| 11  | `ARRAY_WEIGHTS` | [PackedFloat32Array](https://docs.godotengine.org/en/stable/classes/class_packedfloat32array.html#class-packedfloat32array)<br /> 或 [PackedFloat64Array](https://docs.godotengine.org/en/stable/classes/class_packedfloat64array.html#class-packedfloat64array)<br />，每组 4 个浮点数。每个浮点数列出 `ARRAY_BONES` 中对应骨骼对给定顶点的权重。 |
| 12  | `ARRAY_INDEX` | [PackedInt32Array](https://docs.godotengine.org/en/stable/classes/class_packedint32array.html#class-packedint32array) |

在创建网格的大多数情况下，我们通过顶点位置来定义它。因此，通常需要顶点数组（索引 0），而索引数组（索引 12）是可选的，仅在包含时使用。也可以创建只有索引数组而没有顶点数组的网格，但这超出了本教程的范围。

所有其他数组都带有关于顶点的信息。它们是可选的，仅在包含时使用。其中一些数组（例如 `ARRAY_COLOR`）为每个顶点使用一个条目来提供关于顶点的额外信息。它们的大小必须与顶点数组相同。其他数组（例如 `ARRAY_TANGENT`）使用四个条目来描述单个顶点。这些数组的大小必须正好是顶点数组的四倍。

对于普通用法，[add_surface_from_arrays()](https://docs.godotengine.org/en/stable/classes/class_arraymesh.html#class-arraymesh-method-add-surface-from-arrays) 的最后三个参数通常留空。

## 设置 ArrayMesh

在编辑器中，创建一个 [MeshInstance3D](https://docs.godotengine.org/en/stable/classes/class_meshinstance3d.html#class-meshinstance3d)，并在检查器中为其添加一个 [ArrayMesh](https://docs.godotengine.org/en/stable/classes/class_arraymesh.html#class-arraymesh)。通常在编辑器中添加 ArrayMesh 没有多大用处，但在本例中，它允许我们在不创建 ArrayMesh 的情况下从代码中访问它。

接下来，为 MeshInstance3D 添加一个脚本。

在 `_ready()` 下，创建一个新的 Array。

<Tabs groupId="language">
<TabItem value="gdscript" label="GDScript">

```gdscript
var surface_array = []
```

</TabItem>
<TabItem value="csharp" label="C#">

```csharp
Godot.Collections.Array surfaceArray = [];
```

</TabItem>
</Tabs>

这将是我们保存表面信息的数组——它将包含表面所需的所有数据数组。Godot 期望它的大小为 `Mesh.ARRAY_MAX`，因此请相应地调整其大小。

<Tabs groupId="language">
<TabItem value="gdscript" label="GDScript">

```gdscript
var surface_array = []
surface_array.resize(Mesh.ARRAY_MAX)
```

</TabItem>
<TabItem value="csharp" label="C#">

```csharp
Godot.Collections.Array surfaceArray = [];
surfaceArray.Resize((int)Mesh.ArrayType.Max);
```

</TabItem>
</Tabs>

接下来为您将使用的每种数据类型创建数组。

<Tabs groupId="language">
<TabItem value="gdscript" label="GDScript">

```gdscript
var verts = PackedVector3Array()
var uvs = PackedVector2Array()
var normals = PackedVector3Array()
var indices = PackedInt32Array()
```

</TabItem>
<TabItem value="csharp" label="C#">

```csharp
List<Vector3> verts = [];
List<Vector2> uvs = [];
List<Vector3> normals = [];
List<int> indices = [];
```

</TabItem>
</Tabs>

填充完几何数据数组后，您可以通过将每个数组添加到 `surface_array` 然后提交到网格来创建网格。

<Tabs groupId="language">
<TabItem value="gdscript" label="GDScript">

```gdscript
surface_array[Mesh.ARRAY_VERTEX] = verts
surface_array[Mesh.ARRAY_TEX_UV] = uvs
surface_array[Mesh.ARRAY_NORMAL] = normals
surface_array[Mesh.ARRAY_INDEX] = indices

# 不使用混合形状 (blendshapes)、层级细节 (lods) 或压缩。
mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, surface_array)
```

</TabItem>
<TabItem value="csharp" label="C#">

```csharp
surfaceArray[(int)Mesh.ArrayType.Vertex] = verts.ToArray();
surfaceArray[(int)Mesh.ArrayType.TexUV] = uvs.ToArray();
surfaceArray[(int)Mesh.ArrayType.Normal] = normals.ToArray();
surfaceArray[(int)Mesh.ArrayType.Index] = indices.ToArray();

var arrMesh = Mesh as ArrayMesh;
if (arrMesh != null)
{
    // 不使用混合形状 (blendshapes)、层级细节 (lods) 或压缩。
    arrMesh.AddSurfaceFromArrays(Mesh.PrimitiveType.Triangles, surfaceArray);
}
```

</TabItem>
</Tabs>

:::note 注意
在本例中，我们使用了 `Mesh.PRIMITIVE_TRIANGLES`，但您可以使用网格中可用的任何图元类型。
:::

综上所述，完整代码如下：

<Tabs groupId="language">
<TabItem value="gdscript" label="GDScript">

```gdscript
extends MeshInstance3D

func _ready():
	var surface_array = []
	surface_array.resize(Mesh.ARRAY_MAX)

	# 用于构建网格的 PackedVector**Arrays。
	var verts = PackedVector3Array()
	var uvs = PackedVector2Array()
	var normals = PackedVector3Array()
	var indices = PackedInt32Array()

	#######################################
	## 在此处插入生成网格的代码           ##
	#######################################

	# 将数组分配给 surface_array。
	surface_array[Mesh.ARRAY_VERTEX] = verts
	surface_array[Mesh.ARRAY_TEX_UV] = uvs
	surface_array[Mesh.ARRAY_NORMAL] = normals
	surface_array[Mesh.ARRAY_INDEX] = indices

	# 从网格数组创建网格表面。
	# 不使用混合形状、层级细节或压缩。
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, surface_array)
```

</TabItem>
<TabItem value="csharp" label="C#">

```csharp
using System.Collections.Generic;
using Godot;

public partial class MyMeshInstance3D : MeshInstance3D
{
    public override void _Ready()
    {
        Godot.Collections.Array surfaceArray = [];
        surfaceArray.Resize((int)Mesh.ArrayType.Max);

        // C# 数组无法调整大小或扩展，因此使用 List 来创建几何体。
        List<Vector3> verts = [];
        List<Vector2> uvs = [];
        List<Vector3> normals = [];
        List<int> indices = [];

        /*************************************
        * 在此处插入生成网格的代码。
        *************************************/

        // 将 List 转换为数组并分配给 surfaceArray
        surfaceArray[(int)Mesh.ArrayType.Vertex] = verts.ToArray();
        surfaceArray[(int)Mesh.ArrayType.TexUV] = uvs.ToArray();
        surfaceArray[(int)Mesh.ArrayType.Normal] = normals.ToArray();
        surfaceArray[(int)Mesh.ArrayType.Index] = indices.ToArray();

        var arrMesh = Mesh as ArrayMesh;
        if (arrMesh != null)
        {
            // 从网格数组创建网格表面
            // 不使用混合形状、层级细节或压缩。
            arrMesh.AddSurfaceFromArrays(Mesh.PrimitiveType.Triangles, surfaceArray);
        }
    }
}
```

</TabItem>
</Tabs>

中间的代码可以是您想要的任何内容。下面我们将介绍一些生成形状的代码示例，从矩形开始。

## 生成矩形

由于我们使用 `Mesh.PRIMITIVE_TRIANGLES` 进行渲染，我们将用三角形构建一个矩形。

一个矩形由两个三角形共享四个顶点组成。在我们的示例中，我们将创建一个矩形，其左上点位于 `(0, 0, 0)`，宽度和长度均为 1，如下图所示：

![由共享四个顶点的两个三角形组成的矩形](https://docs.godotengine.org/en/stable/_images/array_mesh_rectangle_as_triangles.webp)

要绘制此矩形，请在 `verts` 数组中定义每个顶点的坐标。

<Tabs groupId="language">
<TabItem value="gdscript" label="GDScript">

```gdscript
verts = PackedVector3Array([
		Vector3(0, 0, 0),
		Vector3(0, 0, 1),
		Vector3(1, 0, 0),
		Vector3(1, 0, 1),
	])
```

</TabItem>
<TabItem value="csharp" label="C#">

```csharp
verts.AddRange(new Vector3[]
{
    new Vector3(0, 0, 0),
    new Vector3(0, 0, 1),
    new Vector3(1, 0, 0),
    new Vector3(1, 0, 1),
});
```

</TabItem>
</Tabs>

`uvs` 数组用于描述纹理的各个部分应如何放置到网格上。取值范围从 0 到 1。根据您的纹理，您可能需要更改这些值。

<Tabs groupId="language">
<TabItem value="gdscript" label="GDScript">

```gdscript
uvs = PackedVector2Array([
		Vector2(0, 0),
		Vector2(1, 0),
		Vector2(0, 1),
		Vector2(1, 1),
	])
```

</TabItem>
<TabItem value="csharp" label="C#">

```csharp
uvs.AddRange(new Vector2[]
{
    new Vector2(0, 0),
    new Vector2(1, 0),
    new Vector2(0, 1),
    new Vector2(1, 1),
});
```

</TabItem>
</Tabs>

`normals` 数组用于描述顶点面向的方向，并在光照计算中使用。在本例中，我们将默认使用 `Vector3.UP` 方向。

<Tabs groupId="language">
<TabItem value="gdscript" label="GDScript">

```gdscript
normals = PackedVector3Array([
		Vector3.UP,
		Vector3.UP,
		Vector3.UP,
		Vector3.UP,
	])
```

</TabItem>
<TabItem value="csharp" label="C#">

```csharp
normals.AddRange(new Vector3[]
{
    Vector3.Up,
    Vector3.Up,
    Vector3.Up,
    Vector3.Up,
});
```

</TabItem>
</Tabs>

`indices` 数组定义顶点的绘制顺序。Godot 以 *顺时针* 方向渲染，这意味着我们必须按顺时针顺序指定要绘制的三角形顶点。

例如，要绘制第一个三角形，我们将希望按顺序绘制顶点 `(0, 0, 0)`、`(1, 0, 0)` 和 `(0, 0, 1)`。这与在 `verts` 数组中绘制 `vert[0]`、`vert[2]` 和 `vert[1]`（即索引 0, 2, 和 1）相同。这些索引值就是 `indices` 数组所定义的。

| 索引 | `verts[索引]` | `uvs[索引]` | `normals[索引]` |
| --- | --- | --- | --- |
| 0   | (0, 0, 0) | (0, 0) | Vector3.UP |
| 1   | (0, 0, 1) | (1, 0) | Vector3.UP |
| 2   | (1, 0, 0) | (0, 1) | Vector3.UP |
| 3   | (1, 0, 1) | (1, 1) | Vector3.UP |

<Tabs groupId="language">
<TabItem value="gdscript" label="GDScript">

```gdscript
indices = PackedInt32Array([
		0, 2, 1, # 绘制第一个三角形。
		2, 3, 1, # 绘制第二个三角形。
	])
```

</TabItem>
<TabItem value="csharp" label="C#">

```csharp
indices.AddRange(new int[]
{
    0, 2, 1, // 绘制第一个三角形。
    2, 3, 1, // 绘制第二个三角形。
});
```

</TabItem>
</Tabs>

综上所述，矩形生成代码如下：

<Tabs groupId="language">
<TabItem value="gdscript" label="GDScript">

```gdscript
extends MeshInstance3D

func _ready():

  # 在此处插入 PackedVector**Arrays 的设置代码。

  verts = PackedVector3Array([
		  Vector3(0, 0, 0),
		  Vector3(0, 0, 1),
		  Vector3(1, 0, 0),
		  Vector3(1, 0, 1),
	  ])

  uvs = PackedVector2Array([
		  Vector2(0, 0),
		  Vector2(1, 0),
		  Vector2(0, 1),
		  Vector2(1, 1),
	  ])

  normals = PackedVector3Array([
		  Vector3.UP,
		  Vector3.UP,
		  Vector3.UP,
		  Vector3.UP,
	  ])

  indices = PackedInt32Array([
		  0, 2, 1,
		  2, 3, 1,
	  ])

  # 在此处插入提交到 ArrayMesh 的代码。
```

</TabItem>
<TabItem value="csharp" label="C#">

```csharp
using System.Collections.Generic;
using Godot;

public partial class MyMeshInstance3D : MeshInstance3D
{
  public override void _Ready()
  {
      // 在此处插入表面数组和 List 的设置代码。

      verts.AddRange(new Vector3[]
      {
          new Vector3(0, 0, 0),
          new Vector3(0, 0, 1),
          new Vector3(1, 0, 0),
          new Vector3(1, 0, 1),
      });

      uvs.AddRange(new Vector2[]
      {
          new Vector2(0, 0),
          new Vector2(1, 0),
          new Vector2(0, 1),
          new Vector2(1, 1),
      });

      normals.AddRange(new Vector3[]
      {
          Vector3.Up,
          Vector3.Up,
          Vector3.Up,
          Vector3.Up,
      });

      indices.AddRange(new int[]
      {
          0, 2, 1,
          2, 3, 1,
      });

      // 在此处插入提交到 ArrayMesh 的代码。
  }
}
```

</TabItem>
</Tabs>

更复杂的示例请参阅下面的球面生成部分。

## 生成球面

以下是生成球面的示例代码。虽然代码是以 GDScript 呈现的，但生成方法并没有 Godot 特有的内容。此实现与 ArrayMeshes 没有特别的关系，只是生成球面的通用方法。如果您在理解上遇到困难或想了解更多关于过程几何的知识，可以使用在线找到的任何教程。

<Tabs groupId="language">
<TabItem value="gdscript" label="GDScript">

```gdscript
extends MeshInstance3D

var rings = 50
var radial_segments = 50
var radius = 1

func _ready():

	# 在此处插入 PackedVector**Arrays 的设置代码。

	# 顶点索引。
	var thisrow = 0
	var prevrow = 0
	var point = 0

	# 循环遍历环 (rings)。
	for i in range(rings + 1):
		var v = float(i) / rings
		var w = sin(PI * v)
		var y = cos(PI * v)

		# 循环遍历环中的线段 (segments)。
		for j in range(radial_segments + 1):
			var u = float(j) / radial_segments
			var x = sin(u * PI * 2.0)
			var z = cos(u * PI * 2.0)
			var vert = Vector3(x * radius * w, y * radius, z * radius * w)
			verts.append(vert)
			normals.append(vert.normalized())
			uvs.append(Vector2(u, v))
			point += 1

			# 使用索引在环中创建三角形。
			if i > 0 and j > 0:
				indices.append(prevrow + j - 1)
				indices.append(prevrow + j)
				indices.append(thisrow + j - 1)

				indices.append(prevrow + j)
				indices.append(thisrow + j)
				indices.append(thisrow + j - 1)

		prevrow = thisrow
		thisrow = point

  # 在此处插入提交到 ArrayMesh 的代码。
```

</TabItem>
<TabItem value="csharp" label="C#">

```csharp
using System.Collections.Generic;
using Godot;

public partial class MyMeshInstance3D : MeshInstance3D
{
    private int _rings = 50;
    private int _radialSegments = 50;
    private float _radius = 1;

    public override void _Ready()
    {
        // 在此处插入表面数组和 List 的设置代码。

        // 顶点索引。
        var thisRow = 0;
        var prevRow = 0;
        var point = 0;

        // 循环遍历环 (rings)。
        for (var i = 0; i < _rings + 1; i++)
        {
            var v = ((float)i) / _rings;
            var w = Mathf.Sin(Mathf.Pi * v);
            var y = Mathf.Cos(Mathf.Pi * v);

            // 循环遍历环中的线段 (segments)。
            for (var j = 0; j < _radialSegments + 1; j++)
            {
                var u = ((float)j) / _radialSegments;
                var x = Mathf.Sin(u * Mathf.Pi * 2);
                var z = Mathf.Cos(u * Mathf.Pi * 2);
                var vert = new Vector3(x * _radius * w, y * _radius, z * _radius * w);
                verts.Add(vert);
                normals.Add(vert.Normalized());
                uvs.Add(new Vector2(u, v));
                point += 1;

                // 使用索引在环中创建三角形。
                if (i > 0 && j > 0)
                {
                    indices.Add(prevRow + j - 1);
                    indices.Add(prevRow + j);
                    indices.Add(thisRow + j - 1);

                    indices.Add(prevRow + j);
                    indices.Add(thisRow + j);
                    indices.Add(thisRow + j - 1);
                }
            }

            prevRow = thisRow;
            thisRow = point;
        }

        // 在此处插入提交到 ArrayMesh 的代码。
    }
}
```

</TabItem>
</Tabs>

## 保存

最后，我们可以使用 [ResourceSaver](https://docs.godotengine.org/en/stable/classes/class_resourcesaver.html#class-resourcesaver) 类来保存 ArrayMesh。这在您想要生成网格，然后稍后使用它而无需重新生成时非常有用。

<Tabs groupId="language">
<TabItem value="gdscript" label="GDScript">

```gdscript
# 将网格保存为启用压缩的 .tres 文件。
ResourceSaver.save(mesh, "res://sphere.tres", ResourceSaver.FLAG_COMPRESS)
```

</TabItem>
<TabItem value="csharp" label="C#">

```csharp
// 将网格保存为启用压缩的 .tres 文件。
ResourceSaver.Save(Mesh, "res://sphere.tres", ResourceSaver.SaverFlags.Compress);
```

</TabItem>
</Tabs>
