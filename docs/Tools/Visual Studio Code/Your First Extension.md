# 你的第一个扩展

在本主题中，我们将教你构建扩展的基本概念。确保你已安装 [Node.js](https://nodejs.org) 和 [Git](https://git-scm.com/)。

首先，使用 [Yeoman](https://yeoman.io/) 和 [VS Code 扩展生成器](https://www.npmjs.com/package/generator-code) 来搭建一个 TypeScript 或 JavaScript 项目，为开发做好准备。

- 如果你不想为以后的使用安装 Yeoman，请运行以下命令：
    
    ```
    npx --package yo --package generator-code -- yo code
    ```
    
- 如果你想全局安装 Yeoman 以方便重复运行，请执行以下命令：
    
    ```
    npm install --global yo generator-code
    
    yo code
    ```
    

对于 TypeScript 项目，填写以下字段：

```
# ? 你想创建什么类型的扩展？New Extension (TypeScript)
# ? 你的扩展名称是什么？HelloWorld
### 按 <Enter> 键为下面所有选项选择默认值 ###

# ? 你的扩展标识符是什么？helloworld
# ? 你的扩展描述是什么？留空
# ? 初始化 git 仓库吗？Y
# ? 使用哪个打包工具？unbundled
# ? 使用哪个包管理器？npm

# ? 你想用 Visual Studio Code 打开新文件夹吗？Open with `code`
```

在编辑器中，打开 `src/extension.ts` 并按 F5 或从命令面板 (⇧⌘P (Windows、Linux Ctrl+Shift+P)) 运行 **Debug: Start Debugging** 命令。这将在一个新的 **Extension Development Host** 窗口中编译并运行扩展。

在新窗口的命令面板 (⇧⌘P (Windows、Linux Ctrl+Shift+P)) 中运行 **Hello World** 命令：

你应该会看到显示 `Hello World from HelloWorld!` 的通知。成功！

如果你无法在调试窗口中看到 **Hello World** 命令，请检查 `package.json` 文件，确保 `engines.vscode` 版本与安装的 VS Code 版本兼容。

## 开发扩展

让我们对消息做一个更改：

1. 在 `extension.ts` 中将消息从 "Hello World from HelloWorld!" 改为 "Hello VS Code"。
2. 在新窗口中运行 **Developer: Reload Window**。
3. 再次运行 **Hello World** 命令。

你应该会看到更新后的消息显示出来。

以下是一些你可以尝试的创意：

- 在命令面板中为 **Hello World** 命令取一个新名称。
- [贡献](/api/references/contribution-points) 另一个命令，在信息消息中显示当前时间。贡献点是在 `package.json` [扩展清单](/api/references/extension-manifest) 中进行的静态声明，用于扩展 VS Code，例如向你的扩展添加命令、菜单或键绑定。
- 将 `vscode.window.showInformationMessage` 替换为另一个 [VS Code API](/api/references/vscode-api) 调用来显示警告消息。

## 调试扩展

VS Code 的内置调试功能使调试扩展变得简单。通过单击行旁边的空白处来设置断点，VS Code 就会在断点处停下。你可以将鼠标悬停在编辑器中的变量上，或使用左侧的 **Run and Debug** 视图来检查变量的值。调试控制台允许你计算表达式。

你可以在 [Node.js 调试主题](/docs/nodejs/nodejs-debugging) 中了解更多关于在 VS Code 中调试 Node.js 应用的信息。

## 下一步

在下一个主题 [Extension Anatomy](/api/get-started/extension-anatomy) 中，我们将仔细查看 `Hello World` 示例的源代码，并解释关键概念。

你可以在以下地址找到本教程的源代码：[https://github.com/microsoft/vscode-extension-samples/tree/main/helloworld-sample](https://github.com/microsoft/vscode-extension-samples/tree/main/helloworld-sample)。[Extension Guides](/api/extension-guides/overview) 主题包含其他示例，每个示例都说明了不同的 VS Code API 或贡献点，并遵循了我们的 [UX 指南](/api/ux-guidelines/overview) 中的建议。

### 使用 JavaScript

在本指南中，我们主要介绍如何使用 TypeScript 开发 VS Code 扩展，因为我们相信 TypeScript 为开发 VS Code 扩展提供了最佳体验。但是，如果你更喜欢 JavaScript，你仍然可以使用 [helloworld-minimal-sample](https://github.com/microsoft/vscode-extension-samples/tree/main/helloworld-minimal-sample) 进行开发。

### UX 指南

现在也是回顾我们的 [UX 指南](/api/ux-guidelines/overview) 的好时机，这样你就可以开始设计你的扩展用户界面，遵循 VS Code 的最佳实践。

---

*原文日期：2026/3/18*

*原文链接：[Your First Extension | Visual Studio Code Extension API](https://code.visualstudio.com/api/get-started/your-first-extension)*
