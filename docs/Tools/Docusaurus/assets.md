# 资源 (Assets)

有时你想从 Markdown 文件直接链接到资源（例如 docx 文件、图片...），将资源与使用它的 Markdown 文件放在一起会很方便。

让我们设想以下文件结构：

```text
# 你的文档
/website/docs/myFeature.mdx
# 一些你想使用的资源
/website/docs/assets/docusaurus-asset-example-banner.png
/website/docs/assets/docusaurus-asset-example.docx
```

## 图像 (Images)

你可以通过三种不同的方式显示图像：Markdown 语法、CommonJS require 或 ES imports 语法。

使用简单的 Markdown 语法显示图像：

```md
![Example banner](./assets/docusaurus-asset-example-banner.png)
```

在 JSX image 标签中使用内联 CommonJS `require` 显示图像：

```jsx
<img src={require('./assets/docusaurus-asset-example-banner.png').default} alt="Example banner" />
```

在 JSX image 标签中使用 ES `import` 语法显示图像：

```jsx
import myImageUrl from './assets/docusaurus-asset-example-banner.png';

<img src={myImageUrl} alt="Example banner" />;
```

:::note

如果你正在使用 [`@docusaurus/plugin-ideal-image`](https://docusaurus.io/docs/api/plugins/@docusaurus/plugin-ideal-image)，你需要按照文档使用专用的 image 组件。

:::

## 文件 (Files)

同样，你可以通过 `require` 链接到现有资源，并在 `video`、`a` 锚点链接等标签中使用返回的 URL。

```jsx
# 我的 Markdown 页面

<a target="_blank" href={require('./assets/docusaurus-asset-example.docx').default}> Download this docx </a>

或者

[Download this docx using Markdown](./assets/docusaurus-asset-example.docx)
```

**Markdown 链接始终是文件路径**

如果你使用 Markdown 图像或链接语法，所有资源路径都将被 Docusaurus 解析为文件路径，并自动转换为 `require()` 调用。除非你使用 JSX 语法（需要自己处理），否则不需要在 Markdown 中使用 `require()`。

## 内联 SVG (Inline SVGs)

Docusaurus 开箱即支持内联 SVG。

```jsx
import DocusaurusSvg from './docusaurus.svg';

<DocusaurusSvg />;
```

如果你想通过 CSS 改变 SVG 图像的一部分，这非常有用。例如，你可以根据当前主题更改 SVG 颜色之一。

```jsx
import DocusaurusSvg from './docusaurus.svg';

<DocusaurusSvg className="themedDocusaurus" />;
```

```css
[data-theme='light'] .themedDocusaurus [fill='#FFFF50'] {
  fill: greenyellow;
}

[data-theme='dark'] .themedDocusaurus [fill='#FFFF50'] {
  fill: seagreen;
}
```

## 主题图片 (Themed Images)

Docusaurus 支持主题图片：`ThemedImage` 组件（包含在主题中）允许你根据当前主题切换图片源。

```jsx
import useBaseUrl from '@docusaurus/useBaseUrl';
import ThemedImage from '@theme/ThemedImage';

<ThemedImage
  alt="Docusaurus themed image"
  sources={{
    light: useBaseUrl('/img/docusaurus_light.svg'),
    dark: useBaseUrl('/img/docusaurus_dark.svg'),
  }}
/>;
```

### GitHub 风格的主题图片

GitHub 使用其自己的[图片主题化方法](https://github.blog/changelog/2021-11-24-specify-theme-context-for-images-in-markdown/)（使用路径片段），你可以轻松地自己实现。

要使用路径片段切换图片的可见性（对于 GitHub，是 `#gh-dark-mode-only` 和 `#gh-light-mode-only`），请将以下内容添加到你的自定义 CSS 中（如果你不想与 GitHub 耦合，也可以使用自己的后缀）：

`src/css/custom.css`

```css
[data-theme='light'] img[src$='#gh-dark-mode-only'],
[data-theme='dark'] img[src$='#gh-light-mode-only'] {
  display: none;
}
```

```md
![Docusaurus themed image](/img/docusaurus_keytar.svg#gh-light-mode-only)
![Docusaurus themed image](/img/docusaurus_speed.svg#gh-dark-mode-only)
```

## 静态资源 (Static assets)

如果 Markdown 链接或图像具有绝对路径，则该路径将被视为文件路径，并将从静态目录中解析。例如，如果你已将[静态目录](https://docusaurus.io/docs/static-assets)配置为 `['public', 'static']`，那么对于以下图像：

`my-doc.md`

```md
![An image from the static](/img/docusaurus.png)
```

Docusaurus 将尝试在 `static/img/docusaurus.png` 和 `public/img/docusaurus.png` 中查找它。链接随后将被转换为 `require()` 调用，而不是保留为 URL。这在两个方面是可取的：

1.  你不必担心基本 URL (base URL)，Docusaurus 在提供资源时会处理它；
2.  图像进入 Webpack 的构建管道，其名称将附加哈希值，这使得浏览器能够积极缓存图像并提高站点性能。

如果你打算编写 URL，可以使用 `pathname://` 协议来禁用自动资源链接。

```md
![banner](pathname:///img/docusaurus-asset-example-banner.png)
```

此链接将生成为 `<img src="/img/docusaurus-asset-example-banner.png" alt="banner" />`，不做任何处理或文件存在性检查。
