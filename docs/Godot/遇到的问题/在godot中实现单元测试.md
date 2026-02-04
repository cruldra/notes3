这是一个为你整理的 **Godot 单元测试学习笔记**。你可以把它保存在你的知识库中，方便随时查阅。

---

# Godot 单元测试学习笔记

## 1. 为什么 Godot 需要专门的测试框架？

Godot 引擎本身没有像 VS Code 或 PyCharm 那样内置“点击函数旁边的绿色箭头运行测试”的功能。虽然可以用 `extends SceneTree` 写简单的脚本（像你之前生成的那个），但在大型项目中，我们需要：

* **可视化界面：** 能看到红/绿条，一眼看出哪个测试挂了。
* **断言库 (Assertions)：** 比如 `assert_eq(a, b)`，比 `if a != b: print("Error")` 更专业。
* **生命周期管理：** 在测试前后自动创建/销毁节点，防止内存泄漏。
* **模拟 (Mocking)：** 比如测试“玩家受伤”时，不需要真的加载一个子弹场景，而是模拟一个子弹对象。

---

## 2. 主流框架选择

目前 Godot 4 主要有两个选择，**推荐新手使用 GUT**。

| 特性 | **GUT (Godot Unit Test)** | **GdUnit4** |
| --- | --- | --- |
| **流行度** | ⭐⭐⭐⭐⭐ (社区标准，文档最全) | ⭐⭐⭐ (较新，但增长快) |
| **易用性** | 容易上手，有专门的编辑器面板 | 类似 Java JUnit 风格，更严谨 |
| **语言支持** | GDScript 最好 | GDScript + C# (C#支持极好) |
| **核心特点** | 功能丰富，支持 Double/Mock/Spy | 运行速度快，能够并发测试 |

---

## 3. GUT 框架快速上手 (Quick Start)

### 第一步：安装

1. 在 Godot 编辑器点击 **AssetLib**。
2. 搜索 **GUT** 并下载安装。
3. 进入 **Project Settings -> Plugins**，勾选 **Enable**。
4. 底部会出现一个 **GUT** 面板。

### 第二步：配置目录

1. 在 `res://` 根目录下新建文件夹 `tests`。
2. 在 `tests` 目录下新建脚本 `test_example.gd`。

### 第三步：编写第一个测试

GUT 的测试脚本必须继承自 `GutTest`，且测试函数名必须以 `test_` 开头。

```gdscript
# res://tests/test_calculator.gd
extends GutTest

# 测试前准备 (比如实例化要测试的类)
var validator

func before_each():
    # 假设你有一个 MoveValidator 类
    validator = MoveValidator.new()

# 测试后清理
func after_each():
    validator.free()

# --- 测试用例 ---

func test_addition():
    # 简单的断言
    assert_eq(1 + 1, 2, "数学应该是对的")

func test_chess_move_valid():
    # 结合你的象棋项目
    var from = Vector2i(0, 0)
    var to = Vector2i(0, 1)
    # 假设 validate_general 返回 true
    var result = validator.validate_general(from, to, Constants.Side.BLACK)
    
    assert_true(result, "黑将应该能向下移动一格")

func test_chess_move_invalid():
    # 测试非法移动
    var from = Vector2i(0, 0)
    var to = Vector2i(0, 5) # 飞太远了
    var result = validator.validate_general(from, to, Constants.Side.BLACK)
    
    assert_false(result, "黑将不能一次走5格")

```

### 第四步：运行

1. 点击底部 GUT 面板。
2. 点击 **"Run All"**。
3. 看结果：全是绿色即为通过。

---

## 4. 核心概念：断言 (Assertions)

不要用 `print` 和 `if`，请使用 GUT 提供的断言方法：

* `assert_eq(a, b)`: 期待 a 等于 b。
* `assert_ne(a, b)`: 期待 a 不等于 b。
* `assert_true(condition)`: 期待条件为真。
* `assert_false(condition)`: 期待条件为假。
* `assert_null(obj)` / `assert_not_null(obj)`: 检查对象是否存在。
* `assert_file_exists(path)`: 检查文件路径。

---

## 5. 进阶技巧：如何测试场景 (Scene)？

很多时候你需要测试一个节点（比如 Player）的功能，但它依附于场景树。

```gdscript
extends GutTest

var PlayerScene = preload("res://player.tscn")
var player

func test_player_health():
    # 1. 实例化场景
    player = PlayerScene.instantiate()
    
    # 2. 关键：必须把节点加到 GUT 的测试树里，_ready 才会执行！
    add_child_autofree(player) 
    # 注意：add_child_autofree 是 GUT 特有方法，测试完会自动删除，防止内存泄漏
    
    # 3. 执行操作
    player.take_damage(10)
    
    # 4. 断言
    assert_eq(player.current_health, 90, "玩家掉血后应该是90")

```

---

## 6. 避坑指南 (常见错误)

1. **红色的类名报错 (`class_name`)**
* **问题**：测试脚本里写 `var p = Player.new()` 报错。
* **解决**：确保你的 `player.gd` 顶部写了 `class_name Player`。如果没写，只能用 `load("res://player.gd").new()`，非常麻烦。


2. **`_ready` 不执行**
* **问题**：你 `new()` 了一个节点，但发现里面的变量是空的。
* **原因**：Godot 的节点只有被 `add_child()` 到场景树后，`_ready()` 才会跑。
* **解决**：在 GUT 中使用 `add_child_autofree(node)`。


3. **循环依赖**
* 测试代码引用了游戏代码，游戏代码不要反向引用测试代码。保持单向依赖。



---

## 7. 附：关于你之前的脚本 (命令行运行)

你之前 AI 生成的脚本是 `extends SceneTree`，这种属于 **"Custom CLI Script"**。

* **适用场景**：CI/CD 自动化流水线（比如在 GitHub Actions 里跑测试），或者极简单的纯逻辑测试。
* **如何运行**：
在终端输入：`godot -s tests/test_move_validator.gd`
* **缺点**：没有 UI，调试困难，无法测试依赖 `_process` 帧更新的动画逻辑。

**总结建议**：
在开发阶段，**强烈建议使用 GUT 插件**进行交互式测试；当你需要把代码上传到服务器进行自动检查时，再考虑命令行模式。