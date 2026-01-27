---
title: 在 Docusaurus v3 中添加 Tailwind v4
description: Tailwind CSS v4 集成到 Docusaurus v3 的基本设置
---

# 在 Docusaurus v3 中添加 Tailwind v4

> 原文：[Adding Tailwind v4 to Docusaurus v3](https://dev.to/michalwrzosek/adding-tailwind-v4-to-docusaurus-v3-3poa)

这是一个非常基础的 Tailwind v4 在 Docusaurus v3 中的设置。

## 安装步骤

### 1. 安装 Tailwind 和 Postcss

```bash
npm i --save-dev tailwindcss postcss @tailwindcss/postcss
```

### 2. 创建 Docusaurus 插件

在 `src/plugins/tailwind-config.js` 创建插件：

```javascript
module.exports = function tailwindPlugin(context, options) {
  return {
    name: "tailwind-plugin",
    configurePostCss(postcssOptions) {
      postcssOptions.plugins = [require("@tailwindcss/postcss")];
      return postcssOptions;
    },
  };
};
```

### 3. 添加插件到配置

在 `docusaurus.config.ts` 中添加：

```typescript
const config: Config = {
  //...
  plugins: ["./src/plugins/tailwind-config.js"],
  //...
};
```

### 4. 更新 CSS 文件

在 `src/css/custom.css` 文件中添加：

```css
@import "tailwindcss";

@custom-variant dark (&:is([data-theme="dark"] *));
```

完成！从现在开始，Tailwind 类应该可以被 Docusaurus 识别，并且深色主题也会正确同步。

## 注意事项

- **不需要** `tailwind.config.js` 文件
- **不需要** 独立的 `postcss.config.js` 文件
- 使用 `@import "tailwindcss"` 而不是 `@tailwind base/components/utilities`
- `@custom-variant dark` 确保深色模式与 Docusaurus 主题同步
