# PySide6 常用组件详解

## 一、QtWidgets 模块组件

### 1. QApplication

**作用**: Qt 应用程序的核心类，管理整个应用程序的控制流和主要设置。

**关键特性**:
- 每个 PySide6 应用程序必须有且只有一个 QApplication 实例
- 处理应用程序的初始化和清理
- 管理事件循环
- 处理命令行参数

**基本用法**:
```python
import sys
from PySide6.QtWidgets import QApplication

# 创建应用程序实例
app = QApplication(sys.argv)

# ... 创建窗口和其他组件 ...

# 启动事件循环
sys.exit(app.exec_())
```

**常用方法**:
```python
# 获取应用程序实例
app = QApplication.instance()

# 退出应用程序
app.quit()

# 设置应用程序名称
app.setApplicationName("我的应用")

# 设置应用程序版本
app.setApplicationVersion("1.0.0")

# 设置组织名称
app.setOrganizationName("我的公司")

# 处理所有待处理事件
app.processEvents()
```

---

### 2. QMainWindow

**作用**: 提供主应用程序窗口的框架，包含菜单栏、工具栏、状态栏和中央部件区域。

**窗口结构**:
```
┌─────────────────────────────────┐
│      菜单栏 (Menu Bar)           │
├─────────────────────────────────┤
│      工具栏 (Tool Bar)           │
├─────────────────────────────────┤
│  │                         │    │
│停│   中央部件              │停  │
│靠│   (Central Widget)      │靠  │
│区│                         │区  │
│  │                         │    │
├─────────────────────────────────┤
│      状态栏 (Status Bar)         │
└─────────────────────────────────┘
```

**基本用法**:
```python
from PySide6.QtWidgets import QMainWindow, QWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 设置窗口标题
        self.setWindowTitle("主窗口")
        
        # 设置窗口大小
        self.resize(800, 600)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建菜单栏
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件")
        file_menu.addAction("打开")
        file_menu.addAction("保存")
        
        # 创建工具栏
        toolbar = self.addToolBar("主工具栏")
        toolbar.addAction("新建")
        
        # 创建状态栏
        statusbar = self.statusBar()
        statusbar.showMessage("就绪")
```

**常用方法**:
```python
# 设置中央部件
main_window.setCentralWidget(widget)

# 获取菜单栏
menubar = main_window.menuBar()

# 添加工具栏
toolbar = main_window.addToolBar("工具栏名称")

# 获取状态栏
statusbar = main_window.statusBar()

# 添加停靠窗口
main_window.addDockWidget(Qt.LeftDockWidgetArea, dock_widget)

# 设置窗口图标
main_window.setWindowIcon(QIcon("icon.png"))

# 最大化/最小化/全屏
main_window.showMaximized()
main_window.showMinimized()
main_window.showFullScreen()
main_window.showNormal()
```

---

### 3. QWidget

**作用**: 所有用户界面对象的基类，可以作为容器或独立窗口。

**关键特性**:
- 是所有 UI 组件的基类
- 可以包含其他部件（作为父容器）
- 可以独立显示为窗口
- 支持布局管理

**基本用法**:
```python
from PySide6.QtWidgets import QWidget, QLabel

# 作为独立窗口
widget = QWidget()
widget.setWindowTitle("独立窗口")
widget.resize(400, 300)
widget.show()

# 作为容器
container = QWidget()
label = QLabel("Hello", parent=container)
```

**常用方法**:
```python
# 设置大小
widget.resize(400, 300)
widget.setFixedSize(400, 300)  # 固定大小
widget.setMinimumSize(200, 150)
widget.setMaximumSize(800, 600)

# 设置位置
widget.move(100, 100)
widget.setGeometry(100, 100, 400, 300)  # x, y, width, height

# 显示/隐藏
widget.show()
widget.hide()
widget.setVisible(True)

# 启用/禁用
widget.setEnabled(True)
widget.setDisabled(True)

# 设置样式
widget.setStyleSheet("background-color: white;")

# 设置布局
widget.setLayout(layout)

# 获取父部件
parent = widget.parentWidget()

# 查找子部件
child = widget.findChild(QLabel, "label_name")
```

---

### 4. QVBoxLayout (垂直布局)

**作用**: 将子部件垂直排列的布局管理器。

**布局示意**:
```
┌─────────────┐
│   Widget 1  │
├─────────────┤
│   Widget 2  │
├─────────────┤
│   Widget 3  │
└─────────────┘
```

**基本用法**:
```python
from PySide6.QtWidgets import QVBoxLayout, QPushButton

# 创建垂直布局
layout = QVBoxLayout()

# 添加部件
layout.addWidget(QPushButton("按钮 1"))
layout.addWidget(QPushButton("按钮 2"))
layout.addWidget(QPushButton("按钮 3"))

# 添加间距
layout.addSpacing(20)

# 添加弹性空间
layout.addStretch()

# 应用到容器
widget = QWidget()
widget.setLayout(layout)
```

**常用方法**:
```python
# 添加部件
layout.addWidget(widget)
layout.addWidget(widget, stretch=1)  # 带拉伸因子

# 插入部件
layout.insertWidget(0, widget)  # 在索引 0 处插入

# 添加布局
layout.addLayout(another_layout)

# 添加间距
layout.addSpacing(10)  # 固定间距
layout.addStretch(1)   # 弹性空间

# 设置边距
layout.setContentsMargins(10, 10, 10, 10)  # 左、上、右、下
layout.setContentsMargins(0, 0, 0, 0)      # 无边距

# 设置部件间距
layout.setSpacing(5)

# 移除部件
layout.removeWidget(widget)

# 设置对齐方式
layout.setAlignment(Qt.AlignTop)
```

---

### 5. QHBoxLayout (水平布局)

**作用**: 将子部件水平排列的布局管理器。

**布局示意**:
```
┌─────────┬─────────┬─────────┐
│Widget 1 │Widget 2 │Widget 3 │
└─────────┴─────────┴─────────┘
```

**基本用法**:
```python
from PySide6.QtWidgets import QHBoxLayout, QPushButton

# 创建水平布局
layout = QHBoxLayout()

# 添加部件
layout.addWidget(QPushButton("左"))
layout.addWidget(QPushButton("中"))
layout.addWidget(QPushButton("右"))

# 应用到容器
widget = QWidget()
widget.setLayout(layout)
```

**常用方法**: (与 QVBoxLayout 相同)
```python
# 添加部件
layout.addWidget(widget)
layout.addWidget(widget, stretch=2)  # 拉伸因子为 2

# 添加弹性空间
layout.addStretch()

# 设置间距
layout.setSpacing(10)

# 设置边距
layout.setContentsMargins(5, 5, 5, 5)
```

**组合使用示例**:
```python
# 创建复杂布局
main_layout = QVBoxLayout()

# 顶部水平布局
top_layout = QHBoxLayout()
top_layout.addWidget(QLabel("标题"))
top_layout.addStretch()
top_layout.addWidget(QPushButton("关闭"))

# 中间内容
content = QLabel("内容区域")

# 底部水平布局
bottom_layout = QHBoxLayout()
bottom_layout.addStretch()
bottom_layout.addWidget(QPushButton("确定"))
bottom_layout.addWidget(QPushButton("取消"))

# 组合
main_layout.addLayout(top_layout)
main_layout.addWidget(content)
main_layout.addStretch()
main_layout.addLayout(bottom_layout)

widget = QWidget()
widget.setLayout(main_layout)
```

---

### 6. QTabWidget (选项卡部件)

**作用**: 提供选项卡式的页面切换界面。

**界面示意**:
```
┌─────┬─────┬─────┐
│Tab1 │Tab2 │Tab3 │
├─────┴─────┴─────┴──────┐
│                         │
│    当前选项卡的内容      │
│                         │
└─────────────────────────┘
```

**基本用法**:
```python
from PySide6.QtWidgets import QTabWidget, QWidget, QLabel

# 创建选项卡部件
tab_widget = QTabWidget()

# 创建页面
page1 = QWidget()
page1_layout = QVBoxLayout()
page1_layout.addWidget(QLabel("这是第一页"))
page1.setLayout(page1_layout)

page2 = QWidget()
page2_layout = QVBoxLayout()
page2_layout.addWidget(QLabel("这是第二页"))
page2.setLayout(page2_layout)

# 添加选项卡
tab_widget.addTab(page1, "首页")
tab_widget.addTab(page2, "设置")

# 显示
tab_widget.show()
```

**常用方法**:
```python
# 添加选项卡
index = tab_widget.addTab(widget, "标签文本")
index = tab_widget.addTab(widget, QIcon("icon.png"), "带图标")

# 插入选项卡
tab_widget.insertTab(0, widget, "插入的标签")

# 移除选项卡
tab_widget.removeTab(index)

# 获取/设置当前选项卡
current_index = tab_widget.currentIndex()
tab_widget.setCurrentIndex(2)
current_widget = tab_widget.currentWidget()
tab_widget.setCurrentWidget(widget)

# 获取选项卡数量
count = tab_widget.count()

# 获取指定索引的部件
widget = tab_widget.widget(index)

# 设置选项卡文本
tab_widget.setTabText(index, "新文本")

# 设置选项卡图标
tab_widget.setTabIcon(index, QIcon("icon.png"))

# 设置选项卡工具提示
tab_widget.setTabToolTip(index, "这是提示信息")

# 启用/禁用选项卡
tab_widget.setTabEnabled(index, False)

# 设置选项卡位置
tab_widget.setTabPosition(QTabWidget.North)   # 顶部（默认）
tab_widget.setTabPosition(QTabWidget.South)   # 底部
tab_widget.setTabPosition(QTabWidget.West)    # 左侧
tab_widget.setTabPosition(QTabWidget.East)    # 右侧

# 设置选项卡形状
tab_widget.setTabShape(QTabWidget.Rounded)    # 圆角
tab_widget.setTabShape(QTabWidget.Triangular) # 三角形

# 设置是否可移动
tab_widget.setMovable(True)

# 设置是否可关闭
tab_widget.setTabsClosable(True)
```

**信号**:
```python
# 当前选项卡改变时
tab_widget.currentChanged.connect(lambda index: print(f"切换到索引: {index}"))

# 选项卡关闭请求
tab_widget.tabCloseRequested.connect(lambda index: tab_widget.removeTab(index))

# 选项卡栏点击
tab_widget.tabBarClicked.connect(lambda index: print(f"点击了索引: {index}"))

# 选项卡栏双击
tab_widget.tabBarDoubleClicked.connect(lambda index: print(f"双击了索引: {index}"))
```

---

### 7. QLabel (标签)

**作用**: 显示文本或图像的只读标签。

**基本用法**:
```python
from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap

# 文本标签
label = QLabel("Hello World")

# HTML 文本
label = QLabel("<h1>标题</h1><p>段落</p>")

# 图像标签
label = QLabel()
pixmap = QPixmap("image.png")
label.setPixmap(pixmap)

# 数字标签
label = QLabel()
label.setNum(42)
```

**常用方法**:
```python
# 设置文本
label.setText("新文本")
label.setNum(123)

# 获取文本
text = label.text()

# 设置对齐方式
label.setAlignment(Qt.AlignCenter)
label.setAlignment(Qt.AlignLeft | Qt.AlignTop)

# 设置文本格式
label.setTextFormat(Qt.PlainText)  # 纯文本
label.setTextFormat(Qt.RichText)   # 富文本
label.setTextFormat(Qt.AutoText)   # 自动检测

# 设置自动换行
label.setWordWrap(True)

# 设置缩进
label.setIndent(10)

# 设置图像
label.setPixmap(QPixmap("image.png"))

# 缩放图像
label.setScaledContents(True)

# 设置链接可点击
label.setOpenExternalLinks(True)
label.setText('<a href="https://example.com">点击这里</a>')

# 设置选择行为
label.setTextInteractionFlags(Qt.TextSelectableByMouse)
```

---

### 8. QPushButton (按钮)

**作用**: 可点击的按钮控件，用于触发操作。

**基本用法**:
```python
from PySide6.QtWidgets import QPushButton
from PySide6.QtGui import QIcon

# 创建按钮
button = QPushButton("点击我")

# 带图标的按钮
button = QPushButton(QIcon("icon.png"), "保存")

# 带父容器的按钮
button = QPushButton("确定", parent=widget)
```

**常用方法**:
```python
# 设置文本
button.setText("新文本")

# 获取文本
text = button.text()

# 设置图标
button.setIcon(QIcon("icon.png"))

# 设置图标大小
button.setIconSize(QSize(32, 32))

# 设置快捷键
button.setShortcut("Ctrl+S")

# 设置为默认按钮（按回车触发）
button.setDefault(True)

# 设置为自动默认
button.setAutoDefault(True)

# 设置扁平样式
button.setFlat(True)

# 设置可选中（切换按钮）
button.setCheckable(True)

# 检查是否选中
if button.isChecked():
    print("按钮已选中")

# 设置选中状态
button.setChecked(True)

# 启用/禁用
button.setEnabled(False)
```

**信号**:
```python
# 点击信号（最常用）
button.clicked.connect(lambda: print("按钮被点击"))

# 按下信号
button.pressed.connect(lambda: print("按钮被按下"))

# 释放信号
button.released.connect(lambda: print("按钮被释放"))

# 切换信号（仅当 checkable=True）
button.toggled.connect(lambda checked: print(f"切换状态: {checked}"))
```

**样式示例**:
```python
# 设置样式表
button.setStyleSheet("""
    QPushButton {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
    }
    QPushButton:hover {
        background-color: #45a049;
    }
    QPushButton:pressed {
        background-color: #3d8b40;
    }
    QPushButton:disabled {
        background-color: #cccccc;
    }
""")
```

---

### 9. QTextEdit (多行文本编辑器)

**作用**: 多行富文本编辑器，支持格式化文本。

**基本用法**:
```python
from PySide6.QtWidgets import QTextEdit

# 创建文本编辑器
text_edit = QTextEdit()

# 设置初始文本
text_edit = QTextEdit("初始内容")

# 设置为只读
text_edit = QTextEdit()
text_edit.setReadOnly(True)
```

**常用方法**:
```python
# 设置文本
text_edit.setText("纯文本")
text_edit.setPlainText("纯文本")
text_edit.setHtml("<h1>HTML 文本</h1>")

# 获取文本
text = text_edit.toPlainText()  # 纯文本
html = text_edit.toHtml()       # HTML 格式

# 追加文本
text_edit.append("新的一行")

# 插入文本（在光标位置）
text_edit.insertPlainText("插入的文本")
text_edit.insertHtml("<b>粗体文本</b>")

# 清空内容
text_edit.clear()

# 设置只读
text_edit.setReadOnly(True)

# 撤销/重做
text_edit.undo()
text_edit.redo()

# 剪切/复制/粘贴
text_edit.cut()
text_edit.copy()
text_edit.paste()

# 全选
text_edit.selectAll()

# 设置占位符文本
text_edit.setPlaceholderText("请输入内容...")

# 设置自动换行
text_edit.setLineWrapMode(QTextEdit.WidgetWidth)  # 按部件宽度换行
text_edit.setLineWrapMode(QTextEdit.NoWrap)       # 不换行

# 设置接受富文本
text_edit.setAcceptRichText(True)
```

**信号**:
```python
# 文本改变
text_edit.textChanged.connect(lambda: print("文本已改变"))

# 光标位置改变
text_edit.cursorPositionChanged.connect(lambda: print("光标移动"))

# 选择改变
text_edit.selectionChanged.connect(lambda: print("选择改变"))
```

---

### 10. QLineEdit (单行文本输入框)

**作用**: 单行文本输入框，用于输入简短文本。

**基本用法**:
```python
from PySide6.QtWidgets import QLineEdit

# 创建输入框
line_edit = QLineEdit()

# 设置初始文本
line_edit = QLineEdit("初始文本")

# 设置占位符
line_edit = QLineEdit()
line_edit.setPlaceholderText("请输入用户名...")
```

**常用方法**:
```python
# 设置/获取文本
line_edit.setText("新文本")
text = line_edit.text()

# 清空文本
line_edit.clear()

# 设置最大长度
line_edit.setMaxLength(20)

# 设置只读
line_edit.setReadOnly(True)

# 设置输入掩码
line_edit.setInputMask("000.000.000.000")  # IP 地址
line_edit.setInputMask("(999) 999-9999")   # 电话号码
line_edit.setInputMask(">AAAAA")           # 5个大写字母

# 设置回显模式（密码输入）
line_edit.setEchoMode(QLineEdit.Normal)    # 正常显示
line_edit.setEchoMode(QLineEdit.Password)  # 密码模式
line_edit.setEchoMode(QLineEdit.NoEcho)    # 不显示

# 设置对齐方式
line_edit.setAlignment(Qt.AlignCenter)

# 设置验证器
from PySide6.QtGui import QIntValidator, QDoubleValidator
int_validator = QIntValidator(0, 100)  # 只允许 0-100 的整数
line_edit.setValidator(int_validator)

# 撤销/重做
line_edit.undo()
line_edit.redo()

# 剪切/复制/粘贴
line_edit.cut()
line_edit.copy()
line_edit.paste()

# 全选
line_edit.selectAll()

# 设置选择范围
line_edit.setSelection(0, 5)  # 选择前5个字符

# 获取选中文本
selected = line_edit.selectedText()
```

**信号**:
```python
# 文本改变
line_edit.textChanged.connect(lambda text: print(f"文本: {text}"))

# 编辑完成（失去焦点或按回车）
line_edit.editingFinished.connect(lambda: print("编辑完成"))

# 按下回车
line_edit.returnPressed.connect(lambda: print("按下回车"))

# 光标位置改变
line_edit.cursorPositionChanged.connect(lambda old, new: print(f"光标: {old} -> {new}"))

# 选择改变
line_edit.selectionChanged.connect(lambda: print("选择改变"))
```

---

### 11. QCheckBox (复选框)

**作用**: 复选框控件，提供选中/未选中的二态或三态选择。

**基本用法**:
```python
from PySide6.QtWidgets import QCheckBox

# 创建复选框
checkbox = QCheckBox("同意条款")

# 设置初始状态
checkbox = QCheckBox("记住密码")
checkbox.setChecked(True)
```

**常用方法**:
```python
# 设置文本
checkbox.setText("新文本")

# 获取文本
text = checkbox.text()

# 设置选中状态
checkbox.setChecked(True)

# 获取选中状态
if checkbox.isChecked():
    print("已选中")

# 切换状态
checkbox.toggle()

# 设置三态模式
checkbox.setTristate(True)

# 获取状态（三态）
state = checkbox.checkState()
# Qt.Unchecked (0) - 未选中
# Qt.PartiallyChecked (1) - 部分选中
# Qt.Checked (2) - 选中

# 设置状态（三态）
checkbox.setCheckState(Qt.PartiallyChecked)
```

**信号**:
```python
# 状态改变（bool）
checkbox.stateChanged.connect(lambda state: print(f"状态: {state}"))

# 切换信号（bool）
checkbox.toggled.connect(lambda checked: print(f"选中: {checked}"))

# 点击信号
checkbox.clicked.connect(lambda: print("被点击"))
```

**示例**:
```python
# 创建一组复选框
layout = QVBoxLayout()

options = ["选项 1", "选项 2", "选项 3"]
checkboxes = []

for option in options:
    cb = QCheckBox(option)
    cb.stateChanged.connect(lambda state, opt=option: print(f"{opt}: {state}"))
    checkboxes.append(cb)
    layout.addWidget(cb)

# 获取所有选中的选项
selected = [cb.text() for cb in checkboxes if cb.isChecked()]
```

---

### 12. QComboBox (下拉列表框)

**作用**: 下拉列表框，允许用户从列表中选择一项。

**基本用法**:
```python
from PySide6.QtWidgets import QComboBox

# 创建下拉框
combo = QComboBox()

# 添加项目
combo.addItem("选项 1")
combo.addItem("选项 2")
combo.addItem("选项 3")

# 批量添加
combo.addItems(["苹果", "香蕉", "橙子"])

# 带数据的项目
combo.addItem("显示文本", userData="实际数据")
```

**常用方法**:
```python
# 添加项目
combo.addItem("文本")
combo.addItem(QIcon("icon.png"), "带图标")
combo.addItems(["项目1", "项目2", "项目3"])

# 插入项目
combo.insertItem(0, "插入到开头")

# 移除项目
combo.removeItem(index)

# 清空所有项目
combo.clear()

# 获取/设置当前索引
index = combo.currentIndex()
combo.setCurrentIndex(2)

# 获取/设置当前文本
text = combo.currentText()
combo.setCurrentText("选项 2")

# 获取项目数量
count = combo.count()

# 获取指定索引的文本
text = combo.itemText(index)

# 设置指定索引的文本
combo.setItemText(index, "新文本")

# 获取/设置项目数据
data = combo.itemData(index)
combo.setItemData(index, "自定义数据")

# 查找项目
index = combo.findText("查找的文本")
index = combo.findData("查找的数据")

# 设置可编辑
combo.setEditable(True)

# 设置最大可见项目数
combo.setMaxVisibleItems(10)

# 设置插入策略（可编辑时）
combo.setInsertPolicy(QComboBox.InsertAtTop)
combo.setInsertPolicy(QComboBox.InsertAtBottom)
combo.setInsertPolicy(QComboBox.InsertAlphabetically)
```

**信号**:
```python
# 当前索引改变
combo.currentIndexChanged.connect(lambda index: print(f"索引: {index}"))

# 当前文本改变
combo.currentTextChanged.connect(lambda text: print(f"文本: {text}"))

# 激活（用户选择）
combo.activated.connect(lambda index: print(f"激活索引: {index}"))

# 文本激活
combo.textActivated.connect(lambda text: print(f"激活文本: {text}"))

# 编辑文本改变（可编辑时）
combo.editTextChanged.connect(lambda text: print(f"编辑: {text}"))
```

**示例**:
```python
# 创建带数据的下拉框
combo = QComboBox()

fruits = [
    ("苹果", "apple"),
    ("香蕉", "banana"),
    ("橙子", "orange")
]

for display, data in fruits:
    combo.addItem(display, userData=data)

# 获取选中项的数据
def on_selection_changed(index):
    text = combo.currentText()
    data = combo.currentData()
    print(f"选择了: {text}, 数据: {data}")

combo.currentIndexChanged.connect(on_selection_changed)
```

---

### 13. QSplitter (分割器)

**作用**: 可调整大小的分割器，允许用户拖动调整子部件的大小。

**基本用法**:
```python
from PySide6.QtWidgets import QSplitter
from PySide6.QtCore import Qt

# 创建水平分割器
splitter = QSplitter(Qt.Horizontal)

# 创建垂直分割器
splitter = QSplitter(Qt.Vertical)

# 添加部件
splitter.addWidget(widget1)
splitter.addWidget(widget2)
splitter.addWidget(widget3)
```

**常用方法**:
```python
# 添加部件
splitter.addWidget(widget)

# 插入部件
splitter.insertWidget(index, widget)

# 获取部件数量
count = splitter.count()

# 获取指定索引的部件
widget = splitter.widget(index)

# 设置方向
splitter.setOrientation(Qt.Horizontal)  # 水平
splitter.setOrientation(Qt.Vertical)    # 垂直

# 设置各部件的大小
splitter.setSizes([100, 200, 300])

# 获取各部件的大小
sizes = splitter.sizes()

# 设置是否可折叠
splitter.setCollapsible(0, False)  # 第一个部件不可折叠

# 设置分割条宽度
splitter.setHandleWidth(5)

# 设置子部件拉伸因子
splitter.setStretchFactor(0, 1)  # 第一个部件拉伸因子为1
splitter.setStretchFactor(1, 2)  # 第二个部件拉伸因子为2

# 设置是否显示子部件
splitter.widget(0).setVisible(False)
```

**信号**:
```python
# 分割器移动
splitter.splitterMoved.connect(lambda pos, index: print(f"位置: {pos}, 索引: {index}"))
```

**示例**:
```python
# 创建三栏布局
splitter = QSplitter(Qt.Horizontal)

left_panel = QTextEdit("左侧面板")
center_panel = QTextEdit("中间面板")
right_panel = QTextEdit("右侧面板")

splitter.addWidget(left_panel)
splitter.addWidget(center_panel)
splitter.addWidget(right_panel)

# 设置初始大小比例
splitter.setSizes([200, 400, 200])

# 嵌套分割器
main_splitter = QSplitter(Qt.Horizontal)
right_splitter = QSplitter(Qt.Vertical)

main_splitter.addWidget(QTextEdit("左侧"))
main_splitter.addWidget(right_splitter)

right_splitter.addWidget(QTextEdit("右上"))
right_splitter.addWidget(QTextEdit("右下"))
```

---

### 14. QScrollArea (滚动区域)

**作用**: 为内容提供滚动功能的容器。

**基本用法**:
```python
from PySide6.QtWidgets import QScrollArea, QLabel

# 创建滚动区域
scroll_area = QScrollArea()

# 创建内容部件
content = QLabel("很长的内容..." * 100)

# 设置部件
scroll_area.setWidget(content)

# 设置部件可调整大小
scroll_area.setWidgetResizable(True)
```

**常用方法**:
```python
# 设置内容部件
scroll_area.setWidget(widget)

# 获取内容部件
widget = scroll_area.widget()

# 设置部件可调整大小
scroll_area.setWidgetResizable(True)

# 设置水平滚动条策略
scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)   # 始终显示
scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 始终隐藏
scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)   # 需要时显示

# 设置垂直滚动条策略
scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

# 获取滚动条
h_scrollbar = scroll_area.horizontalScrollBar()
v_scrollbar = scroll_area.verticalScrollBar()

# 设置滚动位置
scroll_area.horizontalScrollBar().setValue(100)
scroll_area.verticalScrollBar().setValue(200)

# 滚动到顶部/底部
scroll_area.verticalScrollBar().setValue(0)  # 顶部
scroll_area.verticalScrollBar().setValue(
    scroll_area.verticalScrollBar().maximum()
)  # 底部

# 确保部件可见
scroll_area.ensureWidgetVisible(child_widget)
```

**示例**:
```python
# 创建包含大量内容的滚动区域
scroll_area = QScrollArea()
scroll_area.setWidgetResizable(True)

# 创建内容容器
content_widget = QWidget()
content_layout = QVBoxLayout(content_widget)

# 添加大量部件
for i in range(100):
    content_layout.addWidget(QPushButton(f"按钮 {i+1}"))

scroll_area.setWidget(content_widget)
```

---

### 15. QFrame (框架)

**作用**: 带边框的容器部件，可以作为视觉分隔或分组。

**基本用法**:
```python
from PySide6.QtWidgets import QFrame

# 创建框架
frame = QFrame()

# 设置框架样式
frame.setFrameShape(QFrame.Box)
frame.setFrameShadow(QFrame.Raised)
```

**框架形状**:
```python
# 设置形状
frame.setFrameShape(QFrame.NoFrame)      # 无框架
frame.setFrameShape(QFrame.Box)          # 矩形框
frame.setFrameShape(QFrame.Panel)        # 面板
frame.setFrameShape(QFrame.StyledPanel)  # 样式化面板
frame.setFrameShape(QFrame.HLine)        # 水平线
frame.setFrameShape(QFrame.VLine)        # 垂直线
```

**框架阴影**:
```python
# 设置阴影
frame.setFrameShadow(QFrame.Plain)   # 平面
frame.setFrameShadow(QFrame.Raised)  # 凸起
frame.setFrameShadow(QFrame.Sunken)  # 凹陷
```

**常用方法**:
```python
# 设置线宽
frame.setLineWidth(2)

# 设置中线宽度
frame.setMidLineWidth(1)

# 获取框架矩形
rect = frame.frameRect()

# 设置框架矩形
frame.setFrameRect(QRect(0, 0, 200, 100))

# 获取框架宽度
width = frame.frameWidth()
```

**示例**:
```python
# 创建分隔线
h_line = QFrame()
h_line.setFrameShape(QFrame.HLine)
h_line.setFrameShadow(QFrame.Sunken)

v_line = QFrame()
v_line.setFrameShape(QFrame.VLine)
v_line.setFrameShadow(QFrame.Sunken)

# 创建带边框的容器
container = QFrame()
container.setFrameShape(QFrame.StyledPanel)
container.setFrameShadow(QFrame.Raised)
container.setLineWidth(2)

layout = QVBoxLayout(container)
layout.addWidget(QLabel("内容"))
layout.addWidget(QPushButton("按钮"))
```

---

### 16. QGroupBox (分组框)

**作用**: 带标题的分组容器，用于逻辑分组相关控件。

**基本用法**:
```python
from PySide6.QtWidgets import QGroupBox

# 创建分组框
group_box = QGroupBox("设置")

# 创建带布局的分组框
group_box = QGroupBox("选项")
layout = QVBoxLayout()
layout.addWidget(QCheckBox("选项 1"))
layout.addWidget(QCheckBox("选项 2"))
group_box.setLayout(layout)
```

**常用方法**:
```python
# 设置标题
group_box.setTitle("新标题")

# 获取标题
title = group_box.title()

# 设置对齐方式
group_box.setAlignment(Qt.AlignLeft)
group_box.setAlignment(Qt.AlignCenter)
group_box.setAlignment(Qt.AlignRight)

# 设置是否可选中
group_box.setCheckable(True)

# 设置选中状态
group_box.setChecked(True)

# 获取选中状态
if group_box.isChecked():
    print("分组框已选中")

# 设置扁平样式
group_box.setFlat(True)
```

**信号**:
```python
# 选中状态改变（仅当 checkable=True）
group_box.toggled.connect(lambda checked: print(f"选中: {checked}"))

# 点击信号
group_box.clicked.connect(lambda: print("分组框被点击"))
```

**示例**:
```python
# 创建设置面板
settings_group = QGroupBox("显示设置")
settings_group.setCheckable(True)
settings_group.setChecked(True)

layout = QVBoxLayout()
layout.addWidget(QCheckBox("显示工具栏"))
layout.addWidget(QCheckBox("显示状态栏"))
layout.addWidget(QCheckBox("全屏模式"))

settings_group.setLayout(layout)

# 当分组框未选中时，内部控件自动禁用
settings_group.toggled.connect(lambda checked: print(f"设置{'启用' if checked else '禁用'}"))

# 创建多个分组框
main_layout = QVBoxLayout()

# 个人信息组
personal_group = QGroupBox("个人信息")
personal_layout = QVBoxLayout()
personal_layout.addWidget(QLabel("姓名:"))
personal_layout.addWidget(QLineEdit())
personal_layout.addWidget(QLabel("邮箱:"))
personal_layout.addWidget(QLineEdit())
personal_group.setLayout(personal_layout)

# 偏好设置组
preference_group = QGroupBox("偏好设置")
preference_layout = QVBoxLayout()
preference_layout.addWidget(QCheckBox("接收通知"))
preference_layout.addWidget(QCheckBox("自动更新"))
preference_group.setLayout(preference_layout)

main_layout.addWidget(personal_group)
main_layout.addWidget(preference_group)
```

---

### 17. QListWidget (列表部件)

**作用**: 显示和管理项目列表的部件，支持单选或多选。

**基本用法**:
```python
from PySide6.QtWidgets import QListWidget, QListWidgetItem

# 创建列表部件
list_widget = QListWidget()

# 添加项目
list_widget.addItem("项目 1")
list_widget.addItem("项目 2")
list_widget.addItem("项目 3")

# 批量添加
list_widget.addItems(["苹果", "香蕉", "橙子"])

# 添加自定义项目
item = QListWidgetItem("自定义项目")
list_widget.addItem(item)
```

**常用方法**:
```python
# 添加项目
list_widget.addItem("文本")
list_widget.addItem(QListWidgetItem(QIcon("icon.png"), "带图标"))
list_widget.addItems(["项目1", "项目2", "项目3"])

# 插入项目
list_widget.insertItem(0, "插入到开头")

# 移除项目
list_widget.takeItem(index)  # 移除并返回项目

# 清空列表
list_widget.clear()

# 获取项目数量
count = list_widget.count()

# 获取指定索引的项目
item = list_widget.item(index)

# 获取项目文本
text = list_widget.item(index).text()

# 设置项目文本
list_widget.item(index).setText("新文本")

# 获取当前项目
current_item = list_widget.currentItem()
current_row = list_widget.currentRow()

# 设置当前项目
list_widget.setCurrentRow(2)
list_widget.setCurrentItem(item)

# 获取选中的项目
selected_items = list_widget.selectedItems()

# 设置选择模式
list_widget.setSelectionMode(QListWidget.SingleSelection)    # 单选
list_widget.setSelectionMode(QListWidget.MultiSelection)     # 多选
list_widget.setSelectionMode(QListWidget.ExtendedSelection)  # 扩展选择（Ctrl/Shift）
list_widget.setSelectionMode(QListWidget.NoSelection)        # 不可选择

# 排序
list_widget.sortItems(Qt.AscendingOrder)   # 升序
list_widget.sortItems(Qt.DescendingOrder)  # 降序

# 查找项目
items = list_widget.findItems("搜索文本", Qt.MatchContains)

# 滚动到项目
list_widget.scrollToItem(item)
```

**QListWidgetItem 方法**:
```python
# 创建项目
item = QListWidgetItem("文本")

# 设置图标
item.setIcon(QIcon("icon.png"))

# 设置工具提示
item.setToolTip("这是提示信息")

# 设置状态提示
item.setStatusTip("状态栏提示")

# 设置复选框
item.setCheckState(Qt.Checked)
item.setCheckState(Qt.Unchecked)
item.setCheckState(Qt.PartiallyChecked)

# 获取复选框状态
state = item.checkState()

# 设置是否可选择
item.setFlags(item.flags() | Qt.ItemIsSelectable)
item.setFlags(item.flags() & ~Qt.ItemIsSelectable)

# 设置是否可编辑
item.setFlags(item.flags() | Qt.ItemIsEditable)

# 设置是否启用
item.setFlags(item.flags() | Qt.ItemIsEnabled)

# 设置背景色
item.setBackground(QColor(255, 200, 200))

# 设置前景色（文字颜色）
item.setForeground(QColor(0, 0, 255))

# 设置字体
item.setFont(QFont("Arial", 12, QFont.Bold))

# 设置文本对齐
item.setTextAlignment(Qt.AlignCenter)

# 存储自定义数据
item.setData(Qt.UserRole, {"id": 123, "name": "数据"})

# 获取自定义数据
data = item.data(Qt.UserRole)
```

**信号**:
```python
# 当前项目改变
list_widget.currentItemChanged.connect(
    lambda current, previous: print(f"从 {previous.text()} 切换到 {current.text()}")
)

# 当前行改变
list_widget.currentRowChanged.connect(lambda row: print(f"当前行: {row}"))

# 项目点击
list_widget.itemClicked.connect(lambda item: print(f"点击: {item.text()}"))

# 项目双击
list_widget.itemDoubleClicked.connect(lambda item: print(f"双击: {item.text()}"))

# 项目激活（双击或回车）
list_widget.itemActivated.connect(lambda item: print(f"激活: {item.text()}"))

# 项目改变（编辑后）
list_widget.itemChanged.connect(lambda item: print(f"改变: {item.text()}"))

# 选择改变
list_widget.itemSelectionChanged.connect(lambda: print("选择改变"))
```

**实用示例**:
```python
# 创建带复选框的列表
list_widget = QListWidget()

tasks = ["任务 1", "任务 2", "任务 3"]
for task in tasks:
    item = QListWidgetItem(task)
    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
    item.setCheckState(Qt.Unchecked)
    list_widget.addItem(item)

# 获取所有选中的任务
def get_checked_tasks():
    checked = []
    for i in range(list_widget.count()):
        item = list_widget.item(i)
        if item.checkState() == Qt.Checked:
            checked.append(item.text())
    return checked

# 创建可编辑列表
list_widget = QListWidget()
for i in range(5):
    item = QListWidgetItem(f"项目 {i+1}")
    item.setFlags(item.flags() | Qt.ItemIsEditable)
    list_widget.addItem(item)

# 创建带图标和自定义数据的列表
list_widget = QListWidget()

files = [
    {"name": "文档.txt", "icon": "doc.png", "size": 1024},
    {"name": "图片.jpg", "icon": "img.png", "size": 2048},
    {"name": "视频.mp4", "icon": "video.png", "size": 4096}
]

for file_info in files:
    item = QListWidgetItem(QIcon(file_info["icon"]), file_info["name"])
    item.setData(Qt.UserRole, file_info)
    list_widget.addItem(item)

# 获取选中项的数据
def on_item_clicked(item):
    data = item.data(Qt.UserRole)
    print(f"文件: {data['name']}, 大小: {data['size']} 字节")

list_widget.itemClicked.connect(on_item_clicked)
```

---

### 18. QStackedWidget (堆叠部件)

**作用**: 堆叠多个部件，一次只显示一个，常用于实现多页面切换。

**基本用法**:
```python
from PySide6.QtWidgets import QStackedWidget, QWidget, QLabel

# 创建堆叠部件
stacked_widget = QStackedWidget()

# 创建页面
page1 = QWidget()
page1_layout = QVBoxLayout(page1)
page1_layout.addWidget(QLabel("这是第一页"))

page2 = QWidget()
page2_layout = QVBoxLayout(page2)
page2_layout.addWidget(QLabel("这是第二页"))

# 添加页面
stacked_widget.addWidget(page1)
stacked_widget.addWidget(page2)

# 显示第一页
stacked_widget.setCurrentIndex(0)
```

**常用方法**:
```python
# 添加部件
index = stacked_widget.addWidget(widget)

# 插入部件
stacked_widget.insertWidget(index, widget)

# 移除部件
stacked_widget.removeWidget(widget)

# 获取部件数量
count = stacked_widget.count()

# 获取当前索引
current_index = stacked_widget.currentIndex()

# 设置当前索引
stacked_widget.setCurrentIndex(2)

# 获取当前部件
current_widget = stacked_widget.currentWidget()

# 设置当前部件
stacked_widget.setCurrentWidget(widget)

# 获取指定索引的部件
widget = stacked_widget.widget(index)

# 获取部件的索引
index = stacked_widget.indexOf(widget)
```

**信号**:
```python
# 当前索引改变
stacked_widget.currentChanged.connect(lambda index: print(f"切换到页面: {index}"))

# 部件移除
stacked_widget.widgetRemoved.connect(lambda index: print(f"移除页面: {index}"))
```

**与 QListWidget 配合使用**:
```python
from PySide6.QtWidgets import QHBoxLayout

# 创建主布局
main_layout = QHBoxLayout()

# 创建导航列表
nav_list = QListWidget()
nav_list.addItems(["首页", "设置", "关于"])
nav_list.setMaximumWidth(150)

# 创建内容堆叠
content_stack = QStackedWidget()

# 创建页面
home_page = QWidget()
home_layout = QVBoxLayout(home_page)
home_layout.addWidget(QLabel("欢迎来到首页"))
home_layout.addWidget(QPushButton("开始使用"))

settings_page = QWidget()
settings_layout = QVBoxLayout(settings_page)
settings_layout.addWidget(QLabel("设置页面"))
settings_layout.addWidget(QCheckBox("启用通知"))
settings_layout.addWidget(QCheckBox("自动更新"))

about_page = QWidget()
about_layout = QVBoxLayout(about_page)
about_layout.addWidget(QLabel("关于页面"))
about_layout.addWidget(QLabel("版本: 1.0.0"))

# 添加到堆叠部件
content_stack.addWidget(home_page)
content_stack.addWidget(settings_page)
content_stack.addWidget(about_page)

# 连接导航和内容
nav_list.currentRowChanged.connect(content_stack.setCurrentIndex)

# 添加到主布局
main_layout.addWidget(nav_list)
main_layout.addWidget(content_stack)

# 应用到窗口
widget = QWidget()
widget.setLayout(main_layout)
```

**与 QTabWidget 的区别**:
```python
# QTabWidget: 自带选项卡导航
tab_widget = QTabWidget()
tab_widget.addTab(page1, "页面1")  # 自动显示选项卡
tab_widget.addTab(page2, "页面2")

# QStackedWidget: 需要自己实现导航
stacked_widget = QStackedWidget()
stacked_widget.addWidget(page1)  # 没有内置导航
stacked_widget.addWidget(page2)

# 需要配合其他控件（如按钮、列表）实现导航
button1.clicked.connect(lambda: stacked_widget.setCurrentIndex(0))
button2.clicked.connect(lambda: stacked_widget.setCurrentIndex(1))
```

**完整示例 - 设置界面**:
```python
class SettingsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("设置")
        self.resize(800, 600)

        # 中央部件
        central = QWidget()
        self.setCentralWidget(central)

        # 主布局
        main_layout = QHBoxLayout(central)

        # 左侧导航
        self.nav_list = QListWidget()
        self.nav_list.addItems([
            "通用",
            "外观",
            "账户",
            "隐私",
            "高级"
        ])
        self.nav_list.setCurrentRow(0)
        self.nav_list.setMaximumWidth(200)

        # 右侧内容
        self.content_stack = QStackedWidget()

        # 创建各个设置页面
        self.content_stack.addWidget(self.create_general_page())
        self.content_stack.addWidget(self.create_appearance_page())
        self.content_stack.addWidget(self.create_account_page())
        self.content_stack.addWidget(self.create_privacy_page())
        self.content_stack.addWidget(self.create_advanced_page())

        # 连接导航
        self.nav_list.currentRowChanged.connect(self.content_stack.setCurrentIndex)

        # 添加到主布局
        main_layout.addWidget(self.nav_list)
        main_layout.addWidget(self.content_stack, 1)  # 拉伸因子为1

    def create_general_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<h2>通用设置</h2>"))
        layout.addWidget(QCheckBox("启动时自动运行"))
        layout.addWidget(QCheckBox("最小化到系统托盘"))
        layout.addStretch()
        return page

    def create_appearance_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<h2>外观设置</h2>"))

        theme_group = QGroupBox("主题")
        theme_layout = QVBoxLayout()
        theme_layout.addWidget(QComboBox())
        theme_group.setLayout(theme_layout)

        layout.addWidget(theme_group)
        layout.addStretch()
        return page

    def create_account_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<h2>账户设置</h2>"))
        layout.addWidget(QLabel("用户名:"))
        layout.addWidget(QLineEdit())
        layout.addWidget(QLabel("邮箱:"))
        layout.addWidget(QLineEdit())
        layout.addStretch()
        return page

    def create_privacy_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<h2>隐私设置</h2>"))
        layout.addWidget(QCheckBox("允许收集使用数据"))
        layout.addWidget(QCheckBox("发送崩溃报告"))
        layout.addStretch()
        return page

    def create_advanced_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("<h2>高级设置</h2>"))
        layout.addWidget(QCheckBox("启用开发者模式"))
        layout.addWidget(QCheckBox("显示调试信息"))
        layout.addStretch()
        return page
```

---

## 二、QtGui 模块组件

### 1. QFont (字体)

**作用**: 定义文本的字体属性。

**基本用法**:
```python
from PySide6.QtGui import QFont

# 创建字体
font = QFont()

# 指定字体族和大小
font = QFont("Arial", 12)

# 完整参数
font = QFont("微软雅黑", 14, QFont.Bold, italic=True)
```

**常用方法**:
```python
# 设置字体族
font.setFamily("宋体")

# 设置字体大小
font.setPointSize(12)      # 点大小
font.setPixelSize(16)      # 像素大小

# 设置粗细
font.setWeight(QFont.Normal)   # 正常
font.setWeight(QFont.Bold)     # 粗体
font.setWeight(QFont.ExtraBold) # 特粗
font.setBold(True)             # 快捷方式

# 设置斜体
font.setItalic(True)

# 设置下划线
font.setUnderline(True)

# 设置删除线
font.setStrikeOut(True)

# 设置字距
font.setLetterSpacing(QFont.AbsoluteSpacing, 2)

# 设置大小写
font.setCapitalization(QFont.AllUppercase)  # 全大写
font.setCapitalization(QFont.AllLowercase)  # 全小写
font.setCapitalization(QFont.SmallCaps)     # 小型大写

# 应用到部件
label = QLabel("文本")
label.setFont(font)
```

**字体权重常量**:
```python
QFont.Thin        # 100
QFont.ExtraLight  # 200
QFont.Light       # 300
QFont.Normal      # 400
QFont.Medium      # 500
QFont.DemiBold    # 600
QFont.Bold        # 700
QFont.ExtraBold   # 800
QFont.Black       # 900
```

---

## 三、完整示例

```python
import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, 
    QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, QPushButton
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

class DemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PySide6 组件演示")
        self.resize(800, 600)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 标题
        title = QLabel("PySide6 组件演示")
        title_font = QFont("微软雅黑", 16, QFont.Bold)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        # 创建选项卡
        tab_widget = QTabWidget()
        
        # 第一个选项卡
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        tab1_layout.addWidget(QLabel("这是第一个选项卡"))
        tab1_layout.addWidget(QPushButton("按钮 1"))
        tab1_layout.addStretch()
        
        # 第二个选项卡
        tab2 = QWidget()
        tab2_layout = QHBoxLayout(tab2)
        tab2_layout.addWidget(QLabel("左侧"))
        tab2_layout.addStretch()
        tab2_layout.addWidget(QLabel("右侧"))
        
        # 添加选项卡
        tab_widget.addTab(tab1, "垂直布局")
        tab_widget.addTab(tab2, "水平布局")
        
        main_layout.addWidget(tab_widget)
        
        # 底部按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(QPushButton("确定"))
        button_layout.addWidget(QPushButton("取消"))
        
        main_layout.addLayout(button_layout)
        
        # 状态栏
        self.statusBar().showMessage("就绪")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DemoWindow()
    window.show()
    sys.exit(app.exec_())
```

## 四、组件对比总结

| 组件 | 用途 | 特点 |
|------|------|------|
| **QApplication** | 应用程序管理 | 必须有且只有一个实例 |
| **QMainWindow** | 主窗口 | 包含菜单栏、工具栏、状态栏 |
| **QWidget** | 基础部件/容器 | 所有UI组件的基类 |
| **QVBoxLayout** | 垂直布局 | 自动垂直排列子部件 |
| **QHBoxLayout** | 水平布局 | 自动水平排列子部件 |
| **QTabWidget** | 选项卡 | 多页面切换界面 |
| **QLabel** | 标签 | 显示文本或图像 |
| **QFont** | 字体 | 定义文本样式 |

---

## 五、QtCore 模块组件

### 1. Signal (信号)

**作用**: 实现对象间的通信机制，是 Qt 信号-槽机制的核心。

**关键概念**:
- **信号 (Signal)**: 事件发生时发出的通知
- **槽 (Slot)**: 响应信号的函数
- **连接 (Connect)**: 将信号与槽关联起来

**基本用法**:
```python
from PySide6.QtCore import QObject, Signal

class MyObject(QObject):
    # 定义信号（必须在类级别定义）
    my_signal = Signal()                    # 无参数信号
    value_changed = Signal(int)             # 单参数信号
    data_ready = Signal(str, int)           # 多参数信号
    type_signal = Signal([int], [str])      # 重载信号（支持多种类型）

    def __init__(self):
        super().__init__()

    def do_something(self):
        # 发射信号
        self.my_signal.emit()
        self.value_changed.emit(42)
        self.data_ready.emit("Hello", 100)

# 使用信号
obj = MyObject()

# 连接到普通函数
def on_signal():
    print("信号被触发了！")

obj.my_signal.connect(on_signal)

# 连接到 lambda
obj.value_changed.connect(lambda x: print(f"值改变为: {x}"))

# 连接到另一个对象的方法
class Receiver(QObject):
    def handle_data(self, text, number):
        print(f"收到数据: {text}, {number}")

receiver = Receiver()
obj.data_ready.connect(receiver.handle_data)

# 触发信号
obj.do_something()
```

**信号类型定义**:
```python
from PySide6.QtCore import Signal

class SignalExample(QObject):
    # 基本类型
    int_signal = Signal(int)
    str_signal = Signal(str)
    bool_signal = Signal(bool)
    float_signal = Signal(float)

    # 多个参数
    multi_signal = Signal(int, str, bool)

    # 重载信号（同一个信号支持不同参数类型）
    overloaded = Signal([int], [str], [int, str])

    # 自定义类型
    custom_signal = Signal(object)  # 任意 Python 对象
    list_signal = Signal(list)
    dict_signal = Signal(dict)

    def emit_examples(self):
        self.int_signal.emit(42)
        self.str_signal.emit("Hello")
        self.multi_signal.emit(1, "test", True)

        # 重载信号的不同用法
        self.overloaded[int].emit(100)
        self.overloaded[str].emit("text")
        self.overloaded[int, str].emit(200, "data")
```

**连接和断开**:
```python
# 连接信号
signal.connect(slot_function)

# 断开特定连接
signal.disconnect(slot_function)

# 断开所有连接
signal.disconnect()

# 临时阻塞信号
obj.blockSignals(True)   # 阻塞
obj.my_signal.emit()     # 不会触发
obj.blockSignals(False)  # 恢复

# 检查是否有连接
if signal.receivers() > 0:
    print("有槽函数连接到此信号")
```

**实际应用示例**:
```python
from PySide6.QtWidgets import QPushButton, QLabel
from PySide6.QtCore import Signal, QObject

class DataProcessor(QObject):
    # 定义进度信号
    progress_updated = Signal(int)
    processing_finished = Signal(str)
    error_occurred = Signal(str)

    def process_data(self, data):
        try:
            for i in range(100):
                # 处理数据...
                self.progress_updated.emit(i + 1)

            self.processing_finished.emit("处理完成！")
        except Exception as e:
            self.error_occurred.emit(str(e))

# 在 UI 中使用
processor = DataProcessor()

# 连接到 UI 组件
progress_label = QLabel("0%")
processor.progress_updated.connect(
    lambda value: progress_label.setText(f"{value}%")
)

status_label = QLabel()
processor.processing_finished.connect(status_label.setText)
processor.error_occurred.connect(
    lambda err: status_label.setText(f"错误: {err}")
)

# 按钮点击触发处理
button = QPushButton("开始处理")
button.clicked.connect(lambda: processor.process_data([1, 2, 3]))
```

**内置信号示例**:
```python
from PySide6.QtWidgets import QPushButton, QLineEdit, QSlider

# QPushButton 的信号
button = QPushButton("点击我")
button.clicked.connect(lambda: print("按钮被点击"))
button.pressed.connect(lambda: print("按钮被按下"))
button.released.connect(lambda: print("按钮被释放"))

# QLineEdit 的信号
line_edit = QLineEdit()
line_edit.textChanged.connect(lambda text: print(f"文本改变: {text}"))
line_edit.returnPressed.connect(lambda: print("按下回车"))
line_edit.editingFinished.connect(lambda: print("编辑完成"))

# QSlider 的信号
slider = QSlider()
slider.valueChanged.connect(lambda value: print(f"值改变: {value}"))
slider.sliderPressed.connect(lambda: print("滑块被按下"))
slider.sliderReleased.connect(lambda: print("滑块被释放"))
```

---

### 2. QThread (线程)

**作用**: 提供平台无关的线程管理，用于在后台执行耗时操作，避免阻塞 UI。

**为什么需要 QThread**:
- GUI 应用运行在主线程（UI 线程）
- 耗时操作会冻结界面
- QThread 可以在后台执行任务，保持 UI 响应

**基本用法 - 方法一：继承 QThread**:
```python
from PySide6.QtCore import QThread, Signal
import time

class WorkerThread(QThread):
    # 定义信号
    progress = Signal(int)
    finished = Signal(str)

    def __init__(self):
        super().__init__()
        self._is_running = True

    def run(self):
        """线程的主要工作在这里执行"""
        for i in range(100):
            if not self._is_running:
                break

            # 模拟耗时操作
            time.sleep(0.1)

            # 发送进度信号
            self.progress.emit(i + 1)

        # 完成时发送信号
        self.finished.emit("任务完成！")

    def stop(self):
        """停止线程"""
        self._is_running = False

# 使用线程
thread = WorkerThread()

# 连接信号
thread.progress.connect(lambda value: print(f"进度: {value}%"))
thread.finished.connect(lambda msg: print(msg))

# 启动线程
thread.start()

# 等待线程完成
thread.wait()  # 阻塞等待

# 或者停止线程
thread.stop()
thread.wait()
```

**基本用法 - 方法二：使用 QObject + moveToThread (推荐)**:
```python
from PySide6.QtCore import QObject, QThread, Signal, Slot

class Worker(QObject):
    """工作对象（不继承 QThread）"""
    progress = Signal(int)
    finished = Signal(str)

    def __init__(self):
        super().__init__()
        self._is_running = True

    @Slot()
    def do_work(self):
        """执行工作"""
        for i in range(100):
            if not self._is_running:
                break

            time.sleep(0.1)
            self.progress.emit(i + 1)

        self.finished.emit("完成！")

    @Slot()
    def stop(self):
        self._is_running = False

# 创建线程和工作对象
thread = QThread()
worker = Worker()

# 将工作对象移到线程
worker.moveToThread(thread)

# 连接信号
thread.started.connect(worker.do_work)  # 线程启动时开始工作
worker.finished.connect(thread.quit)     # 工作完成时退出线程
worker.finished.connect(worker.deleteLater)
thread.finished.connect(thread.deleteLater)

# 连接进度信号
worker.progress.connect(lambda v: print(f"进度: {v}%"))

# 启动线程
thread.start()
```

**完整的 UI 集成示例**:
```python
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QProgressBar, QLabel
from PySide6.QtCore import QThread, Signal, QObject, Slot
import sys
import time

class Worker(QObject):
    progress = Signal(int)
    finished = Signal()
    status = Signal(str)

    def __init__(self):
        super().__init__()
        self._is_running = True

    @Slot()
    def do_work(self):
        self.status.emit("开始处理...")

        for i in range(100):
            if not self._is_running:
                self.status.emit("已取消")
                return

            # 模拟耗时操作
            time.sleep(0.05)
            self.progress.emit(i + 1)

        self.status.emit("处理完成！")
        self.finished.emit()

    @Slot()
    def stop(self):
        self._is_running = False
        self.status.emit("正在取消...")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QThread 示例")
        self.resize(400, 200)

        # 创建 UI
        layout = QVBoxLayout(self)

        self.status_label = QLabel("就绪")
        self.progress_bar = QProgressBar()
        self.start_button = QPushButton("开始")
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)

        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        # 创建线程和工作对象
        self.thread = None
        self.worker = None

        # 连接按钮
        self.start_button.clicked.connect(self.start_work)
        self.stop_button.clicked.connect(self.stop_work)

    def start_work(self):
        # 创建新线程和工作对象
        self.thread = QThread()
        self.worker = Worker()
        self.worker.moveToThread(self.thread)

        # 连接信号
        self.thread.started.connect(self.worker.do_work)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.on_finished)

        # 清理
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # 更新 UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)

        # 启动线程
        self.thread.start()

    def stop_work(self):
        if self.worker:
            self.worker.stop()

    def on_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
```

**QThread 常用方法**:
```python
# 启动线程
thread.start()

# 等待线程结束
thread.wait()                    # 无限等待
thread.wait(5000)                # 等待最多 5 秒

# 检查线程状态
if thread.isRunning():
    print("线程正在运行")

if thread.isFinished():
    print("线程已完成")

# 退出线程
thread.quit()                    # 请求退出
thread.terminate()               # 强制终止（不推荐）

# 获取线程优先级
priority = thread.priority()

# 设置线程优先级
thread.setPriority(QThread.HighPriority)
```

**线程优先级**:
```python
QThread.IdlePriority          # 空闲时运行
QThread.LowestPriority        # 最低优先级
QThread.LowPriority           # 低优先级
QThread.NormalPriority        # 正常优先级（默认）
QThread.HighPriority          # 高优先级
QThread.HighestPriority       # 最高优先级
QThread.TimeCriticalPriority  # 时间关键优先级
QThread.InheritPriority       # 继承优先级
```

**QThread 内置信号**:
```python
# 线程启动时
thread.started.connect(lambda: print("线程已启动"))

# 线程完成时
thread.finished.connect(lambda: print("线程已完成"))
```

**最佳实践**:

✅ **推荐做法**:
```python
# 1. 使用 QObject + moveToThread（更灵活）
worker = Worker()
thread = QThread()
worker.moveToThread(thread)

# 2. 使用信号槽通信
worker.finished.connect(thread.quit)

# 3. 正确清理资源
worker.finished.connect(worker.deleteLater)
thread.finished.connect(thread.deleteLater)

# 4. 使用 @Slot 装饰器
@Slot()
def do_work(self):
    pass
```

❌ **避免做法**:
```python
# 1. 不要在线程中直接操作 UI
def run(self):
    label.setText("错误！")  # ❌ 不要这样做

# 正确做法：使用信号
def run(self):
    self.update_text.emit("正确！")  # ✅

# 2. 不要使用 terminate()
thread.terminate()  # ❌ 可能导致资源泄漏

# 3. 不要忘记调用 wait()
thread.quit()
# thread.wait()  # ❌ 忘记等待可能导致问题
```

**线程安全注意事项**:
```python
from PySide6.QtCore import QMutex, QMutexLocker

class ThreadSafeCounter(QObject):
    def __init__(self):
        super().__init__()
        self._value = 0
        self._mutex = QMutex()

    def increment(self):
        # 使用互斥锁保护共享数据
        locker = QMutexLocker(self._mutex)
        self._value += 1
        # locker 离开作用域时自动解锁

    def get_value(self):
        locker = QMutexLocker(self._mutex)
        return self._value
```

---

## 六、信号与线程综合示例

```python
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QTextEdit, QProgressBar
from PySide6.QtCore import QThread, Signal, QObject, Slot
import sys
import time

class DataFetcher(QObject):
    """数据获取工作对象"""
    progress = Signal(int, str)      # 进度和消息
    data_ready = Signal(str)         # 数据就绪
    error = Signal(str)              # 错误信号
    finished = Signal()              # 完成信号

    @Slot()
    def fetch_data(self):
        try:
            self.progress.emit(0, "开始获取数据...")
            time.sleep(1)

            self.progress.emit(25, "连接服务器...")
            time.sleep(1)

            self.progress.emit(50, "下载数据...")
            time.sleep(1)

            self.progress.emit(75, "处理数据...")
            time.sleep(1)

            self.progress.emit(100, "完成！")
            self.data_ready.emit("这是获取到的数据内容")

        except Exception as e:
            self.error.emit(f"错误: {str(e)}")
        finally:
            self.finished.emit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("信号与线程综合示例")
        self.resize(500, 400)

        # 中央部件
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # UI 组件
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.progress_bar = QProgressBar()
        self.fetch_button = QPushButton("获取数据")

        layout.addWidget(self.text_edit)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.fetch_button)

        # 连接按钮
        self.fetch_button.clicked.connect(self.start_fetching)

        self.thread = None
        self.fetcher = None

    def start_fetching(self):
        # 创建线程和工作对象
        self.thread = QThread()
        self.fetcher = DataFetcher()
        self.fetcher.moveToThread(self.thread)

        # 连接信号
        self.thread.started.connect(self.fetcher.fetch_data)
        self.fetcher.progress.connect(self.update_progress)
        self.fetcher.data_ready.connect(self.display_data)
        self.fetcher.error.connect(self.show_error)
        self.fetcher.finished.connect(self.thread.quit)
        self.fetcher.finished.connect(self.on_finished)

        # 清理
        self.fetcher.finished.connect(self.fetcher.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # 禁用按钮
        self.fetch_button.setEnabled(False)
        self.text_edit.clear()

        # 启动
        self.thread.start()

    @Slot(int, str)
    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.text_edit.append(f"[{value}%] {message}")

    @Slot(str)
    def display_data(self, data):
        self.text_edit.append(f"\n数据内容:\n{data}")

    @Slot(str)
    def show_error(self, error_msg):
        self.text_edit.append(f"\n❌ {error_msg}")

    @Slot()
    def on_finished(self):
        self.fetch_button.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
```

---

## 七、组件分类总结

### 7.1 按功能分类

**应用程序管理**:
- QApplication - 应用程序核心

**窗口和容器**:
- QMainWindow - 主窗口框架
- QWidget - 基础部件/容器
- QTabWidget - 选项卡容器
- QStackedWidget - 堆叠部件容器
- QSplitter - 可调整分割器
- QScrollArea - 滚动区域
- QFrame - 框架容器
- QGroupBox - 分组框

**布局管理**:
- QVBoxLayout - 垂直布局
- QHBoxLayout - 水平布局

**输入控件**:
- QLineEdit - 单行文本输入
- QTextEdit - 多行文本编辑
- QCheckBox - 复选框
- QComboBox - 下拉列表框
- QPushButton - 按钮

**显示和列表控件**:
- QLabel - 文本/图像标签
- QListWidget - 列表部件

**样式和外观**:
- QFont - 字体定义

**通信和并发**:
- Signal - 信号机制
- QThread - 线程管理

### 7.2 完整组件对比表

| 组件 | 模块 | 类型 | 用途 | 特点 |
|------|------|------|------|------|
| **QApplication** | QtWidgets | 应用管理 | 应用程序核心 | 必须有且只有一个实例 |
| **QMainWindow** | QtWidgets | 窗口 | 主窗口框架 | 包含菜单栏、工具栏、状态栏 |
| **QWidget** | QtWidgets | 容器 | 基础部件/容器 | 所有UI组件的基类 |
| **QVBoxLayout** | QtWidgets | 布局 | 垂直布局 | 自动垂直排列子部件 |
| **QHBoxLayout** | QtWidgets | 布局 | 水平布局 | 自动水平排列子部件 |
| **QTabWidget** | QtWidgets | 容器 | 选项卡 | 多页面切换，自带导航 |
| **QLabel** | QtWidgets | 显示 | 标签 | 显示文本或图像 |
| **QPushButton** | QtWidgets | 输入 | 按钮 | 可点击触发操作 |
| **QTextEdit** | QtWidgets | 输入 | 多行文本编辑器 | 支持富文本格式 |
| **QLineEdit** | QtWidgets | 输入 | 单行文本输入框 | 支持验证器和掩码 |
| **QCheckBox** | QtWidgets | 输入 | 复选框 | 二态或三态选择 |
| **QComboBox** | QtWidgets | 输入 | 下拉列表框 | 从列表中选择 |
| **QListWidget** | QtWidgets | 列表 | 列表部件 | 显示和管理项目列表 |
| **QStackedWidget** | QtWidgets | 容器 | 堆叠部件 | 多页面切换，需自定义导航 |
| **QSplitter** | QtWidgets | 容器 | 分割器 | 可调整子部件大小 |
| **QScrollArea** | QtWidgets | 容器 | 滚动区域 | 为内容提供滚动 |
| **QFrame** | QtWidgets | 容器 | 框架 | 带边框的容器 |
| **QGroupBox** | QtWidgets | 容器 | 分组框 | 带标题的分组容器 |
| **QFont** | QtGui | 样式 | 字体 | 定义文本样式 |
| **Signal** | QtCore | 通信 | 信号 | 对象间通信机制 |
| **QThread** | QtCore | 并发 | 线程 | 后台执行耗时操作 |

### 7.3 常用组合模式

**基本窗口结构**:
```python
QApplication
└── QMainWindow
    └── QWidget (中央部件)
        └── QVBoxLayout
            ├── QLabel
            ├── QLineEdit
            └── QPushButton
```

**选项卡界面**:
```python
QMainWindow
└── QTabWidget
    ├── QWidget (Tab 1)
    │   └── QVBoxLayout
    ├── QWidget (Tab 2)
    │   └── QHBoxLayout
    └── QWidget (Tab 3)
```

**导航+内容界面（使用 QListWidget + QStackedWidget）**:
```python
QMainWindow
└── QHBoxLayout
    ├── QListWidget (导航列表)
    └── QStackedWidget (内容区域)
        ├── QWidget (页面 1)
        ├── QWidget (页面 2)
        └── QWidget (页面 3)
```

**分割面板**:
```python
QMainWindow
└── QSplitter
    ├── QTextEdit (左侧)
    ├── QSplitter (右侧垂直分割)
    │   ├── QTextEdit (右上)
    │   └── QTextEdit (右下)
```

**表单布局**:
```python
QGroupBox ("用户信息")
└── QVBoxLayout
    ├── QHBoxLayout
    │   ├── QLabel ("姓名:")
    │   └── QLineEdit
    ├── QHBoxLayout
    │   ├── QLabel ("邮箱:")
    │   └── QLineEdit
    └── QCheckBox ("接收通知")
```

---

## 八、快速参考

### 8.1 信号连接速查

```python
# 按钮
button.clicked.connect(function)

# 文本输入
line_edit.textChanged.connect(function)
line_edit.returnPressed.connect(function)
text_edit.textChanged.connect(function)

# 复选框
checkbox.stateChanged.connect(function)
checkbox.toggled.connect(function)

# 下拉框
combo.currentIndexChanged.connect(function)
combo.currentTextChanged.connect(function)

# 列表
list_widget.currentRowChanged.connect(function)
list_widget.itemClicked.connect(function)
list_widget.itemDoubleClicked.connect(function)

# 选项卡
tab_widget.currentChanged.connect(function)

# 堆叠部件
stacked_widget.currentChanged.connect(function)

# 分割器
splitter.splitterMoved.connect(function)

# 分组框
group_box.toggled.connect(function)

# 自定义信号
my_signal.connect(function)
```

### 8.2 布局技巧速查

```python
# 添加部件
layout.addWidget(widget)

# 添加拉伸空间
layout.addStretch()

# 添加固定间距
layout.addSpacing(20)

# 设置边距
layout.setContentsMargins(10, 10, 10, 10)

# 设置间距
layout.setSpacing(5)

# 设置对齐
layout.setAlignment(Qt.AlignCenter)
```

### 8.3 样式设置速查

```python
# 设置样式表
widget.setStyleSheet("background-color: white;")

# 设置字体
widget.setFont(QFont("Arial", 12))

# 设置大小
widget.resize(400, 300)
widget.setFixedSize(400, 300)
widget.setMinimumSize(200, 150)

# 设置启用/禁用
widget.setEnabled(False)

# 设置可见/隐藏
widget.setVisible(False)
```

这些是 PySide6 中最基础和常用的组件，掌握它们可以构建大部分桌面应用程序界面！🚀

