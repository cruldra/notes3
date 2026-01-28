在 Docusaurus 中，默认情况下页面右侧会显示文章的目录（Table of Contents, TOC）。在某些场景下，例如展示超宽表格、大型架构图或者希望提供更沉浸的阅读体验时，我们可能需要隐藏这个侧边栏并让正文区域自动扩展。

以下是实现这一目标的几种主要方法：

## 方法一：使用 Front Matter（推荐用于单页面）

这是最简单且最常用的方法，适用于只需要在特定页面隐藏 TOC 的场景。

在 Markdown 或 MDX 文件的顶部 Front Matter 区域添加 `hide_table_of_contents: true`：

```markdown
---
title: 我的超宽页面
hide_table_of_contents: true
---

这里是页面的正文内容，右侧的目录现在已经消失了。
```

## 方法二：使用全局 CSS（适用于全局调整）

如果你希望全局隐藏 TOC，或者在隐藏 TOC 的同时增加正文的显示宽度，可以通过修改全局 CSS 文件实现。

### 1. 隐藏 TOC
在 `src/css/custom.css` 中添加以下代码：

```css
/* 隐藏右侧目录栏 */
.theme-doc-toc-desktop {
  display: none;
}

/* 兼容移动端：隐藏移动端顶部的 TOC 按钮（可选） */
.theme-doc-toc-mobile {
  display: none;
}
```

### 2. 增加正文容器宽度
默认情况下，Docusaurus 的正文宽度是有上限的。隐藏 TOC 后，你可能希望正文占满剩余空间：

```css
:root {
  /* 调整容器的最大宽度，默认通常是 1140px 或更小 */
  --ifm-container-width: 100%; 
}

/* 强制文档内容区域占满可用空间 */
.docItemContainer {
  max-width: 100% !important;
}

/* 如果希望正文更加居中且宽度适中，可以设置具体像素 */
/* 
.container {
  max-width: 1400px;
}
*/
```

## 方法三：通过配置隐藏（针对特定插件）

如果你使用的是经典主题，可以在 `docusaurus.config.ts` (或 `.js`) 中通过 `themeConfig` 进行一些基础限制，虽然它不能直接“开关”隐藏，但可以限制显示级别：

```typescript
// docusaurus.config.ts
export default {
  themeConfig: {
    tableOfContents: {
      minHeadingLevel: 2,
      maxHeadingLevel: 2, // 设为相同或无效级别可减少显示内容
    },
  },
};
```

## 方法四：Swizzle TOC 组件（高级自定义）

如果你需要根据复杂的业务逻辑（例如根据用户权限或屏幕状态）动态决定是否显示 TOC，可以使用 Docusaurus 的 Swizzle 功能替换默认组件：

```bash
pnpm docusaurus swizzle @docusaurus/theme-classic TOC -- --eject
```

**警告**：`eject` 会导致你脱离主题的官方更新维护，仅在上述 CSS 和配置无法满足需求时使用。

## 总结建议

1. **单页面需求**：首选 **Front Matter**，简单、安全、不影响其他页面。
2. **全局或大范围需求**：结合 **CSS 变量覆盖**，既能隐藏 TOC 也能优化整体布局宽度。
3. **响应式处理**：在修改宽度时，请务必检查在移动端和小屏幕上的显示效果。
