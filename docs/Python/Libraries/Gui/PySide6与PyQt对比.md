# PySide6 与 PyQt6 对比分析

## 一、背景介绍

### 1.1 为什么有两个库?

**PyQt** 由 Riverbank Computing Ltd. 的 Phil Thompson 开发,历史悠久,支持 Qt 的多个版本(从 2.x 开始)。

**PySide** 于 2009 年由诺基亚(当时拥有 Qt)创建,目的是提供更宽松的 LGPL 许可证。名称来源于芬兰语"side",意为"绑定"。

### 1.2 发展历程

- **PyQt5** 于 2016 年中期发布,支持 Qt 5
- **PySide2** 在 PyQt5 发布 2 年后才推出稳定版
- **Qt 6** 发布后,两个库几乎同时推出了对应版本:
  - **PyQt6**: 2021年1月首次稳定版
  - **PySide6**: 2020年12月首次稳定版

目前 Qt 项目已正式采用 PySide 作为官方的 [Qt for Python](https://www.qt.io/qt-for-python) 发布版本。

## 二、核心差异对比

### 2.1 基本信息对比

| 对比项 | PyQt6 | PySide6 |
|--------|-------|---------|
| **开发者** | Riverbank Computing Ltd. | Qt 公司 |
| **首次稳定版** | 2021年1月 | 2020年12月 |
| **许可证** | GPL 或 商业许可 | LGPL |
| **Python版本** | Python 3 | Python 3 |
| **API相似度** | 99.9% 相同 | 99.9% 相同 |
| **官方支持** | 第三方 | Qt 官方 |

### 2.2 许可证差异(重要!)

这是两个库最重要的区别:

**PyQt6**:
- 双许可模式: GPL 或 商业许可
- 如果你的软件采用 GPL 许可,可免费使用
- 如果要分发闭源商业软件,需要购买商业许可证

**PySide6**:
- LGPL 许可证
- 可以在商业软件中免费使用
- 不需要开源你的应用代码
- 更适合商业项目

> **建议**: 如果开发商业闭源软件,优先选择 PySide6;如果是开源项目或学习用途,两者皆可。

## 三、API 差异详解

虽然两个库 99.9% 相同,但仍有一些细微差异需要注意:

### 3.1 枚举和标志的命名

**PyQt6** 要求使用完全限定名称:
```python
# PyQt6 (必须使用完整路径)
Qt.ItemDataRole.DisplayRole
Qt.Alignment.AlignLeft
```

**PySide6** 同时支持长短两种形式:
```python
# PySide6 (两种都支持)
Qt.DisplayRole  # 短形式
Qt.ItemDataRole.DisplayRole  # 长形式

Qt.AlignLeft  # 短形式
Qt.AlignmentFlag.AlignLeft  # 长形式
```

**注意**: 标志组的命名也有差异:
- PyQt6: `Qt.Alignment`
- PySide6: `Qt.AlignmentFlag`

### 3.2 UI 文件加载

**PyQt6** 使用 `uic` 子模块:
```python
from PyQt6 import QtWidgets, uic

app = QtWidgets.QApplication(sys.argv)
window = uic.loadUi("mainwindow.ui")
window.show()
app.exec()
```

**PySide6** 需要创建 `QUiLoader` 对象:
```python
from PySide6 import QtWidgets
from PySide6.QtUiTools import QUiLoader

loader = QUiLoader()
app = QtWidgets.QApplication(sys.argv)
window = loader.load("mainwindow.ui", None)
window.show()
app.exec_()
```

### 3.3 UI 文件转 Python 代码

**PyQt6** 使用 `pyuic6` 命令:
```bash
pyuic6 mainwindow.ui -o MainWindow.py
```

**PySide6** 使用 `pyside6-uic` 命令:
```bash
pyside6-uic mainwindow.ui -o MainWindow.py
```

转换后的使用方式相同:
```python
from MainWindow import Ui_MainWindow

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
```

### 3.4 exec() vs exec_()

**PyQt6**:
```python
app.exec()  # Python 3 中 exec 不再是关键字
```

**PySide6**:
```python
app.exec_()  # 保留了旧的命名方式
```

### 3.5 信号和槽

**PyQt6**:
```python
from PyQt6.QtCore import pyqtSignal, pyqtSlot

my_signal = pyqtSignal()
my_signal_with_arg = pyqtSignal(int)

@pyqtSlot
def my_slot():
    pass
```

**PySide6**:
```python
from PySide6.QtCore import Signal, Slot

my_signal = Signal()
my_signal_with_arg = Signal(int)

@Slot
def my_slot():
    pass
```

**兼容性写法** (在 PyQt6 中使用 PySide6 风格):
```python
from PyQt6.QtCore import pyqtSignal as Signal, pyqtSlot as Slot
```

### 3.6 QMouseEvent

**PyQt6** 移除了快捷方法:
```python
# PyQt6 (必须使用 position())
pos = event.position()
x = pos.x()
y = pos.y()
```

**PySide6** 两种方式都支持:
```python
# PySide6 (推荐使用 position())
pos = event.position()
x = pos.x()
y = pos.y()

# 或者使用旧方法
x = event.x()
y = event.y()
```

## 四、PySide6 独有特性

### 4.1 Python 特性标志

PySide6 支持两个 `__feature__` 标志,使代码更 Pythonic:

**传统写法**:
```python
table = QTableWidget()
table.setColumnCount(2)

button = QPushButton("Add")
button.setEnabled(False)

layout = QVBoxLayout()
layout.addWidget(table)
layout.addWidget(button)
```

**启用 `snake_case` 和 `true_property` 后**:
```python
from __feature__ import snake_case, true_property

table = QTableWidget()
table.column_count = 2  # 属性访问

button = QPushButton("Add")
button.enabled = False  # 属性访问

layout = QVBoxLayout()
layout.add_widget(table)  # 蛇形命名
layout.add_widget(button)
```

> **注意**: 这些特性在 PyQt6 中不支持,会影响代码的可移植性。

## 五、Qt5 vs Qt6 的重要变化

### 5.1 性能提升

Qt6 相比 Qt5 有显著的性能改进:

- **QML 渲染**: 2D/3D 渲染性能大幅提升
- **Quick3D**: 支持实例化渲染,模型较多时性能提升明显
- **启动时间**: 减少约 30%
- **内存占用**: 降低约 20%

### 5.2 图形系统变化

- **RHI (Rendering Hardware Interface)**: 统一的渲染抽象层
- 支持多种后端: Vulkan / OpenGL / DirectX / Metal
- 着色器需要提前编译(不再支持字符串形式的 GLSL)
- 使用 Qt 自创的着色器语法

### 5.3 平台支持

**Qt6 不再支持**:
- Windows 7
- Windows 8

**Qt5** 仍然支持这些旧平台。

### 5.4 构建系统

- **Qt6**: CMake 成为首选构建工具
- **Qt5**: 主要使用 qmake

## 六、选择建议

### 6.1 选择 PySide6 的理由

✅ 开发商业闭源软件  
✅ 不想购买商业许可证  
✅ 需要 Qt 官方支持  
✅ 想使用 Python 特性标志(`snake_case`, `true_property`)  
✅ 文档和社区资源更新及时  

### 6.2 选择 PyQt6 的理由

✅ 开发 GPL 开源软件  
✅ 已有 PyQt5 项目需要升级  
✅ 更成熟的第三方库生态  
✅ 更多的历史教程和示例  

### 6.3 学习建议

对于初学者:
1. **两个库都值得了解**,因为它们 99.9% 相同
2. **可以混用教程**: PyQt6 的教程可以用于 PySide6,反之亦然
3. **重点关注差异**: 主要是许可证、命名约定和少数 API 差异
4. **新项目推荐 PySide6**: 官方支持 + LGPL 许可更灵活

## 七、兼容性代码示例

如果要编写同时支持两个库的代码:

```python
import sys

# 检测使用的库
if 'PyQt6' in sys.modules:
    from PyQt6 import QtGui, QtWidgets, QtCore
    from PyQt6.QtCore import pyqtSignal as Signal, pyqtSlot as Slot
else:
    from PySide6 import QtGui, QtWidgets, QtCore
    from PySide6.QtCore import Signal, Slot

# 处理 exec 差异
def _exec(obj):
    if hasattr(obj, 'exec'):
        return obj.exec()
    else:
        return obj.exec_()

# 处理枚举差异
def _enum(obj, name):
    parent, child = name.split('.')
    result = getattr(obj, child, False)
    if result:
        return result
    obj = getattr(obj, parent)
    return getattr(obj, child)

# 使用示例
app = QtWidgets.QApplication(sys.argv)
window = QtWidgets.QMainWindow()
window.show()
_exec(app)
```

## 八、总结

| 方面 | PyQt6 | PySide6 | 推荐 |
|------|-------|---------|------|
| **许可证** | GPL/商业 | LGPL | PySide6 (商业项目) |
| **官方支持** | 第三方 | Qt 官方 | PySide6 |
| **API 完整度** | 99.9% | 99.9% | 平手 |
| **文档质量** | 丰富 | 官方完善 | PySide6 |
| **社区资源** | 更多历史资源 | 快速增长 | PyQt6 (历史) |
| **Python 特性** | 无 | snake_case/true_property | PySide6 |
| **学习曲线** | 平缓 | 平缓 | 平手 |

**最终建议**:
- 🎯 **新项目**: 优先选择 **PySide6**
- 🔄 **旧项目迁移**: 根据许可证需求决定
- 📚 **学习**: 两者都了解,重点掌握差异
- 💼 **商业项目**: **PySide6** (避免许可证问题)
- 🆓 **开源项目**: 两者皆可,**PySide6** 更推荐

无论选择哪个,学到的知识都可以轻松应用到另一个库中! 🚀

