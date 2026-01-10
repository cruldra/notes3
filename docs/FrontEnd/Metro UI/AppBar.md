App Bar 组件提供了一个响应式导航栏，可以放置在应用程序的顶部。它支持移动设备的汉堡菜单，并可在较大屏幕上展开，为应用程序导航提供灵活的解决方案。

## 用法

[标题为“用法”的章节](#usage)

查看示例：由 [Serhii Pimenov](https://pimenov.com.ua) 在 [CodePen](https://codepen.io) 上展示。

### 基本用法

[标题为“基本用法”的章节](#basic-usage)

```html
1
<div data-role="app-bar">
2
<a href="#" class="brand">
3
<span class="caption">Brand</span>
4
</a>
5
<ul class="app-bar-menu">
6
<li><a href="#">Home</a></li>
7
<li><a href="#">About</a></li>
8
<li><a href="#">Contact</a></li>
9
</ul>
10
</div>
```

### 带有汉堡菜单和图标

[标题为“带有汉堡菜单和图标”的章节](#with-hamburger-menu-and-icon)

```html
1
<div data-role="app-bar">
2
<a href="#" class="brand">
3
<span class="icon"><img src="path/to/logo.png"></span>
4
<span class="caption">Brand</span>
5
</a>
6
<ul class="app-bar-menu">
7
<li><a href="#">Home</a></li>
8
<li><a href="#">Products</a></li>
9
<li><a href="#">Services</a></li>
10
<li><a href="#">About</a></li>
11
<li><a href="#">Contact</a></li>
12
</ul>
13
</div>
```

### 编程式创建

[标题为“编程式创建”的章节](#programmatic-creation)

```javascript
// 在现有元素上初始化 app-bar
Metro.makePlugin("#myAppBar", "app-bar");

// 获取 app-bar 对象
const appBar = Metro.getPlugin("#myAppBar", "app-bar");
```

## 插件参数

[标题为“插件参数”的章节](#plugin-parameters)

| 参数 | 类型 | 默认值 | 描述 |
| --- | --- | --- | --- |
| `appbarDeferred` | number | 0 | 延迟初始化时间（毫秒） |
| `expand` | boolean | false | 如果为 true，应用栏将始终展开 |
| `expandPoint` | string | null | 应用栏展开的媒体查询字符串（例如 "md", "lg"） |
| `duration` | number | 100 | 菜单打开/关闭动画持续时间（毫秒） |
| `checkHamburgerColor` | boolean | false | 如果为 true，汉堡图标颜色将取决于应用栏背景颜色 |

## 事件

[标题为“事件”的章节](#events)

| 事件 | 描述 |
| --- | --- |
| `onMenuOpen` | 菜单打开时触发 |
| `onMenuClose` | 菜单关闭时触发 |
| `onBeforeMenuOpen` | 菜单打开前触发 |
| `onBeforeMenuClose` | 菜单关闭前触发 |
| `onMenuCollapse` | 菜单折叠时触发 |
| `onMenuExpand` | 菜单展开时触发 |
| `onAppBarCreate` | 应用栏创建时触发 |

## API 方法

[标题为“API 方法”的章节](#api-methods)

* `open()` - 打开应用栏菜单。
* `close()` - 关闭应用栏菜单。
* `destroy()` - 销毁应用栏组件并移除所有事件监听器。

```javascript
// 打开应用栏菜单
const appBar = Metro.getPlugin('#myAppBar', 'app-bar');
appBar.open();
```

## 使用 CSS 变量样式化

[标题为“使用 CSS 变量样式化”的章节](#styling-with-css-variables)

| 变量 | 默认值 (Light) | Dark Mode | 描述 |
| --- | --- | --- | --- |
| `--appbar-border-radius` | 4px | 4px | 应用栏项目的边框半径 |
| `--appbar-z-index` | @zindex-fixed | @zindex-fixed | 应用栏的 Z-index |
| `--appbar-background` | #ffffff | #1e1f22 | 应用栏的背景颜色 |
| `--appbar-color` | #191919 | #dbdfe7 | 应用栏的文本颜色 |
| `--appbar-item-background` | transparent | transparent | 应用栏项目的背景颜色 |
| `--appbar-item-color` | #191919 | #dbdfe7 | 应用栏项目的文本颜色 |
| `--appbar-item-color-disabled` | #ccc | #a8a8a8 | 禁用状态应用栏项目的文本颜色 |
| `--appbar-item-color-hover` | #000000 | #ffffff | 悬停时应用栏项目的文本颜色 |
| `--appbar-item-background-hover` | #e8e8e8 | #2b2d30 | 悬停时应用栏项目的背景颜色 |

### 自定义样式示例

[标题为“自定义样式示例”的章节](#example-of-custom-styling)

```css
/* 特定应用栏的自定义样式 */
#myCustomAppBar {
--appbar-background: #3498db;
--appbar-color: #ffffff;
--appbar-item-color: #ffffff;
--appbar-item-background-hover: rgba(255, 255, 255, 0.2);
--appbar-item-color-hover: #ffffff;
}
```

## 可用的 CSS 类

[标题为“可用的 CSS 类”的章节](#available-css-classes)

### 基础类

[标题为“基础类”的章节](#base-classes)

* `.app-bar` - 主容器类（自动添加）
* `.app-bar-menu` - 菜单容器
* `.app-bar-item` - 应用栏内的项目
* `.app-bar-item-static` - 没有悬停效果的静态项目
* `.brand` - 用于品牌元素
* `.app-bar-button` - 按钮样式
* `.hamburger` - 汉堡菜单按钮

### 状态类

[标题为“状态类”的章节](#state-classes)

* `.collapsed` - 当菜单折叠时应用
* `.opened` - 当菜单打开时应用
* `.app-bar-expand` - 当应用栏展开时应用

## 响应式行为

[标题为“响应式行为”的章节](#responsive-behavior)

App Bar 组件会自动适应不同的屏幕尺寸：

1. 在小屏幕上，菜单会折叠并通过汉堡按钮访问
2. 在大屏幕上（或当 `expand` 为 true 时），菜单水平展开

您可以使用 `expandPoint` 参数控制应用栏展开的断点。

## 无障碍性

[标题为“无障碍性”的章节](#accessibility)

App Bar 组件包含 ARIA 属性以提高无障碍性：

* 汉堡按钮具有 `aria-label`、`aria-expanded` 和 `aria-controls` 属性
* 菜单项具有适当的角色（roles）和 tabindex 属性
* 支持键盘导航（Enter/Space 切换菜单）

[Edit page](https://github.com/olton/metroui-docs/edit/master/src/content/docs/components/app-bar.mdx)
