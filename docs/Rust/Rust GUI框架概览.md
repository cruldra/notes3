# Rust GUI 框架概览

## 概述

Rust 作为一门系统级编程语言，在 GUI 开发领域有多种选择。虽然生态系统还在发展中，但已经有许多成熟和新兴的框架可供选择。本文整理了 2025 年 Rust GUI 开发的主流技术和框架。

## 主流框架分类

### 1. Web 技术栈（Webview 方案）

#### Tauri
- **官网**: https://tauri.app/
- **仓库**: https://github.com/tauri-apps/tauri
- **特点**:
  - 使用系统原生 WebView（Windows 上使用 WebView2，macOS 使用 WebKit，Linux 使用 WebKitGTK）
  - 前端可使用任何 Web 框架（React、Vue、Svelte 等）
  - 后端使用 Rust 处理业务逻辑
  - 打包体积小，性能优秀
  - 2024 年 10 月发布 2.0 稳定版
- **优势**:
  - 成熟稳定，生产环境可用
  - 丰富的插件生态
  - 跨平台支持好
  - 可以利用现有的 Web 开发技能
- **劣势**:
  - 前后端通信缺乏类型安全（使用字符串传递命令）
  - 需要同时掌握 Web 技术和 Rust

#### Dioxus
- **官网**: https://dioxuslabs.com/
- **仓库**: https://github.com/DioxusLabs/dioxus
- **特点**:
  - React 风格的声明式 UI 框架
  - 支持桌面、Web、移动端、SSR 等多个平台
  - 桌面端基于 Tauri 的 WebView
  - 2024 年 12 月发布 0.6 版本
  - 纯 Rust 编写，无需 JavaScript
- **优势**:
  - 类 React 的开发体验，学习曲线平缓
  - 全栈 Rust 解决方案
  - 屏幕阅读器支持良好
  - IME（输入法）支持完善
- **劣势**:
  - 桌面端仍依赖 WebView
  - 相对较新，生态系统还在发展

### 2. 原生渲染框架

#### egui
- **仓库**: https://github.com/emilk/egui
- **特点**:
  - 即时模式（Immediate Mode）GUI 库
  - 纯 Rust 实现
  - 支持 Web（通过 WASM）和原生平台
  - 简单易用，上手快
- **优势**:
  - API 简洁直观
  - 性能优秀
  - 适合游戏开发和工具类应用
  - 屏幕阅读器支持
- **劣势**:
  - IME 支持有问题（Tab 键被拦截）
  - 默认字体不支持 CJK 字符
  - 即时模式可能不适合所有场景

#### Iced
- **仓库**: https://github.com/iced-rs/iced
- **特点**:
  - 受 Elm 启发的声明式 GUI 框架
  - 渲染器无关设计
  - 类型安全的响应式架构
  - System76 的 COSMIC 桌面环境使用该框架
- **优势**:
  - 优雅的 Elm 架构
  - 跨平台支持
  - 活跃的社区
- **劣势**:
  - 目前缺乏屏幕阅读器支持
  - IME 支持不完善
  - 相对年轻，API 可能变化

#### Slint
- **官网**: https://slint.dev/
- **仓库**: https://github.com/slint-ui/slint
- **特点**:
  - 声明式 GUI 工具包
  - 支持 Rust、C++、JavaScript 和 Python
  - 提供可视化设计工具
  - 2023 年发布 1.0 版本
  - 三重许可：GPLv3、免版税和商业许可
- **优势**:
  - 成熟稳定，生产环境可用
  - 丰富的工具支持（VSCode 扩展、实时预览、图形编辑器）
  - 良好的屏幕阅读器支持
  - IME 基本可用（临时状态显示有小问题）
  - 适合嵌入式和桌面应用
- **劣势**:
  - 需要学习 Slint 标记语言
  - 商业使用需要考虑许可证

### 3. 原生平台绑定

#### GTK 4
- **仓库**: https://github.com/gtk-rs/gtk4-rs
- **特点**:
  - GNOME 工具包的 Rust 绑定
  - 成熟的跨平台 GUI 库
  - 支持 CSS 样式
- **优势**:
  - 功能完整，组件丰富
  - Linux 平台首选
  - 良好的文档
- **劣势**:
  - Windows 上的屏幕阅读器支持不佳
  - Windows 上的外观不够原生
  - 依赖较多

#### Relm4
- **仓库**: https://github.com/Relm4/Relm4
- **特点**:
  - 基于 GTK4 的惯用 Rust GUI 库
  - 受 Elm 启发
  - 提供更符合 Rust 习惯的 API
- **优势**:
  - 比直接使用 GTK4 更符合 Rust 风格
  - 支持 CSS 样式
  - 跨平台
- **劣势**:
  - 继承了 GTK 的一些限制
  - 屏幕阅读器支持不完善

#### FLTK
- **仓库**: https://github.com/fltk-rs/fltk-rs
- **特点**:
  - FLTK C++ 库的 Rust 绑定
  - 轻量级，可静态链接
  - 快速的二进制文件
- **优势**:
  - 打包体积小
  - 启动快速
  - IME 支持良好
- **劣势**:
  - 布局系统不够现代
  - API 设计有些过时
  - 默认无屏幕阅读器支持（需要额外的 crate）

### 4. 实验性/新兴框架

#### Xilem
- **仓库**: https://github.com/linebender/xilem
- **特点**:
  - 基于 Masonry 构建
  - Druid 的继承者
  - 类似 Elm 和 SwiftUI 的架构
- **优势**:
  - 现代化的架构设计
  - 纯 Rust 实现
  - 有潜力成为主流
- **劣势**:
  - 仍在开发中，API 不稳定
  - 缺少版本发布
  - 屏幕阅读器支持有问题

#### Freya
- **仓库**: https://github.com/marc2332/freya
- **特点**:
  - 基于 Dioxus 和 Skia
  - 使用 Dioxus 的逻辑，但自己渲染
  - 避免了 WebView 的依赖
- **优势**:
  - Dioxus 的开发体验
  - 原生渲染性能
  - 跨平台
- **劣势**:
  - 屏幕阅读器支持不完整
  - IME 支持有限
  - 相对不成熟

#### Makepad
- **仓库**: https://github.com/makepad/makepad
- **特点**:
  - 新型 VR、Web 和原生渲染 UI 框架
  - 使用 WebGPU
  - 独特的 DSL
- **优势**:
  - 现代化的渲染技术
  - 支持 VR
- **劣势**:
  - 文档缺乏
  - 主要为 Makepad 团队自用
  - 缺乏无障碍支持

#### Vizia
- **仓库**: https://github.com/vizia/vizia
- **特点**:
  - 声明式桌面 GUI 框架
  - 基于 winit
- **优势**:
  - 现代化的架构
  - 纯 Rust
- **劣势**:
  - 屏幕阅读器只能看到结构，看不到内容
  - 还不够成熟

### 5. 跨语言方案

#### Flutter Rust Bridge
- **仓库**: https://github.com/fzyzcjy/flutter_rust_bridge
- **特点**:
  - Flutter/Dart 与 Rust 的桥接
  - UI 用 Flutter/Dart 编写
  - 业务逻辑用 Rust 编写
- **优势**:
  - 利用 Flutter 成熟的 UI 生态
  - Rust 处理性能关键部分
- **劣势**:
  - 仍需要学习 Dart
  - 状态管理复杂
  - IME 支持有问题

## 2025 年推荐选择

根据 2025 年的调查和社区反馈，以下是不同场景的推荐：

### 生产环境首选

1. **Tauri** - 如果你熟悉 Web 技术，需要快速开发跨平台应用
2. **Slint** - 如果需要成熟稳定的纯 Rust 方案，特别是嵌入式场景
3. **Dioxus** - 如果想要全 Rust 栈，且接受 WebView 方案

### 游戏和工具开发

1. **egui** - 即时模式，适合游戏内 UI 和开发工具
2. **Iced** - 如果喜欢 Elm 架构

### 值得关注的未来之星

1. **Xilem** - Linebender 团队的新作，架构优秀
2. **Freya** - Dioxus + 原生渲染的组合
3. **Dioxus** - 持续快速发展，社区活跃

## 关键考虑因素

### 无障碍支持
- **良好**: Dioxus, Slint, egui, Tauri
- **缺失**: Iced, Floem, Makepad, Vizia

### IME（输入法）支持
- **完善**: Dioxus, Slint, GTK4, FLTK
- **部分支持**: egui（Tab 键问题）, Freya（显示问题）
- **不支持**: Iced, Floem

### 跨平台支持
- **优秀**: Tauri, Dioxus, Slint, egui, Iced
- **Linux 优先**: GTK4, Relm4
- **Windows 专用**: WinSafe

### 开发体验
- **简单易用**: egui, Dioxus
- **需要学习曲线**: Slint（DSL）, Iced（Elm 架构）
- **复杂**: Tauri（需要 Web 技术）

## 参考资源

- [Are We GUI Yet?](https://areweguiyet.com/) - Rust GUI 生态系统追踪
- [2025 Survey of Rust GUI Libraries](https://www.boringcactus.com/2025/04/13/2025-survey-of-rust-gui-libraries.html) - 详细的框架对比
- [Tauri vs Iced vs egui 性能对比](http://lukaskalbertodt.github.io/2023/02/03/tauri-iced-egui-performance-comparison.html)

## 总结

Rust GUI 生态系统在 2025 年已经相对成熟，有多个可用于生产环境的选择。选择框架时需要考虑：

- **项目需求**: Web 技术栈 vs 原生渲染
- **目标平台**: 桌面、移动、Web 还是嵌入式
- **团队技能**: 是否熟悉 Web 开发
- **性能要求**: 是否需要极致性能
- **无障碍需求**: 是否需要屏幕阅读器支持
- **国际化需求**: 是否需要完善的 IME 支持

对于大多数桌面应用，**Tauri**、**Dioxus** 和 **Slint** 是最安全的选择。如果是游戏或工具开发，**egui** 是不错的选项。对于追求前沿技术的开发者，可以关注 **Xilem** 和 **Freya** 的发展。

