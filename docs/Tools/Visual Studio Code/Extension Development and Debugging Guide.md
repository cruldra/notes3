# VSCode 插件开发与调试完全指南

## 目录

1. [环境准备](#环境准备)
2. [创建第一个插件](#创建第一个插件)
3. [插件架构解析](#插件架构解析)
4. [调试技巧](#调试技巧)
5. [测试插件](#测试插件)
6. [常见问题](#常见问题)
7. [进阶资源](#进阶资源)

---

## 环境准备

### 必需工具

- **VSCode**: 确保已安装最新版本
- **Node.js**: 推荐 v16.x 或更高版本
- **Git**: 用于版本控制
- **TypeScript**: VSCode 扩展开发的首选语言

### 推荐工具

- **Yeoman**: 项目脚手架工具
- **VS Code Extension Generator**: 官方扩展生成器

安装方式（二选一）：

```bash
# 方式1: 使用 npx（推荐，无需全局安装）
npx --package yo --package generator-code -- yo code

# 方式2: 全局安装
npm install --global yo generator-code
yo code
```

---

## 创建第一个插件

### 步骤1: 使用脚手架创建项目

运行 `yo code` 后，根据提示选择：

```
? What type of extension do you want to create? New Extension (TypeScript)
? What's the name of your extension? HelloWorld
? What's the identifier of your extension? helloworld
? What's the description of your extension? [留空]
? Initialize a git repository? Yes
? Which bundler to use? unbundled
? Which package manager to use? npm
? Do you want to open the new folder with Visual Studio Code? Open with `code`
```

### 步骤2: 理解核心文件

#### package.json（扩展清单）

```json
{
  "name": "smart-developer-tools",
  "displayName": "Smart Developer Tools",
  "description": "Your intelligent coding companion",
  "version": "0.0.1",
  "engines": {
    "vscode": "^1.74.0"
  },
  "categories": ["Other"],
  "activationEvents": [],
  "main": "./out/extension.js",
  "contributes": {
    "commands": [
      {
        "command": "smart-developer-tools.helloWorld",
        "title": "Smart Developer: Hello World"
      }
    ]
  },
  "scripts": {
    "vscode:prepublish": "npm run compile",
    "compile": "tsc -p ./",
    "watch": "tsc -watch -p ./"
  },
  "devDependencies": {
    "@types/vscode": "^1.74.0",
    "@types/node": "16.x",
    "typescript": "^4.9.4"
  }
}
```

关键字段说明：
- `name`: 扩展的唯一标识符
- `displayName`: 显示在 Marketplace 的名称
- `engines.vscode`: 兼容的 VSCode 最低版本
- `main`: 入口文件路径
- `contributes`: 向 VSCode 贡献的功能（命令、菜单、键绑定等）
- `activationEvents`: 激活事件，决定何时加载扩展

#### tsconfig.json（TypeScript 配置）

```json
{
  "compilerOptions": {
    "module": "commonjs",
    "target": "ES2021",
    "outDir": "out",
    "rootDir": "src",
    "lib": ["ES2021"],
    "sourceMap": true,
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true
  },
  "exclude": ["node_modules", ".vscode-test"]
}
```

推荐配置：
- `strict: true`: 启用严格类型检查，提前发现 bug
- `sourceMap: true`: 生成 source map，便于调试
- `outDir`: 编译输出目录

#### src/extension.ts（主代码）

```typescript
import * as vscode from 'vscode';

// 激活函数 - 扩展启动时调用
export function activate(context: vscode.ExtensionContext) {
    console.log('扩展已激活! 🚀');

    // 注册命令
    let disposable = vscode.commands.registerCommand('smart-developer-tools.helloWorld', () => {
        vscode.window.showInformationMessage('Hello from Smart Developer Tools! 🎯');
    });

    // 将命令添加到订阅列表，便于自动清理
    context.subscriptions.push(disposable);
}

// 停用函数 - 扩展关闭时调用
export function deactivate() {
    // 清理资源
}
```

---

## 插件架构解析

### 扩展生命周期

```
激活 (Activation) → 命令注册 → 用户交互 → 清理 (Cleanup)
```

1. **激活阶段**: VSCode 读取 `package.json`，根据 `activationEvents` 决定何时调用 `activate()`
2. **命令注册**: 扩展向 VSCode 注册可用命令
3. **用户交互**: 用户运行命令时，VSCode 路由到注册的回调函数
4. **清理阶段**: 扩展停用或卸载时调用 `deactivate()` 进行资源清理

### 构建流程

```
src/extension.ts → [TypeScript Compiler] → out/extension.js → [VSCode 加载]
```

### 设计模式

- **关注点分离**: 配置（package.json）与逻辑（extension.ts）分离
- **类型安全**: TypeScript 编译时捕获错误
- **资源管理**: Disposable 模式防止内存泄漏
- **热重载**: 开发时支持实时重新加载

---

## 调试技巧

### 基础调试配置

创建 `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Run Extension",
      "type": "extensionHost",
      "request": "launch",
      "args": [
        "--extensionDevelopmentPath=${workspaceFolder}"
      ],
      "outFiles": [
        "${workspaceFolder}/out/**/*.js"
      ]
    }
  ]
}
```

### 启动调试

1. **编译代码**: `npm run compile`
2. **启动调试**: 按 `F5` 或从命令面板运行 "Debug: Start Debugging"
3. **Extension Development Host**: 会打开一个新的 VSCode 窗口，你的扩展已加载

### 调试功能

- **设置断点**: 单击代码行左侧空白处
- **单步执行**: F10（逐过程）、F11（逐语句）
- **查看变量**: 鼠标悬停或使用左侧 "Run and Debug" 视图
- **调试控制台**: 计算表达式和查看输出 (`Ctrl+Shift+Y`)
- **输出面板**: 查看扩展的 console.log 输出

### 热重载

修改代码后：

1. TypeScript 编译器（watch 模式）自动编译
2. 在 Extension Development Host 窗口按 `Ctrl+R` (Mac: `Cmd+R`) 重新加载
3. 或点击调试工具栏的重启按钮

### 高级调试技巧

#### 1. 禁用其他扩展

在 `launch.json` 中添加 `--disable-extensions`:

```json
{
  "args": [
    "--disable-extensions",
    "--extensionDevelopmentPath=${workspaceFolder}"
  ]
}
```

#### 2. 指定特定版本的 VSCode

```javascript
// .vscode-test.js
const { defineConfig } = require('@vscode/test-cli');

module.exports = defineConfig({
  files: 'out/test/**/*.test.js',
  version: 'insiders'  // 或 'stable', '1.74.0' 等
});
```

#### 3. 调试测试

创建测试调试配置：

```json
{
  "name": "Extension Tests",
  "type": "extensionHost",
  "request": "launch",
  "runtimeExecutable": "${execPath}",
  "args": [
    "--extensionDevelopmentPath=${workspaceFolder}",
    "--extensionTestsPath=${workspaceFolder}/out/test/suite/index"
  ],
  "outFiles": ["${workspaceFolder}/out/test/**/*.js"]
}
```

#### 4. 使用 VSCode Insiders 进行开发

由于稳定版限制，如果要在命令行运行集成测试，建议：
- 使用 **VSCode Stable** 运行测试
- 使用 **VSCode Insiders** 进行日常开发

这样避免 "Running extension tests from the command line is currently only supported if no other instance of Code is running" 错误。

---

## 测试插件

### 测试 CLI 快速配置

安装测试依赖：

```bash
npm install --save-dev @vscode/test-cli @vscode/test-electron
```

在 `package.json` 中添加脚本：

```json
{
  "scripts": {
    "test": "vscode-test"
  }
}
```

创建 `.vscode-test.js` 配置文件：

```javascript
const { defineConfig } = require('@vscode/test-cli');

module.exports = defineConfig({
  files: 'out/test/**/*.test.js',
  mocha: {
    ui: 'tdd',
    timeout: 20000
  }
});
```

### 编写测试

示例测试文件 `src/test/suite/extension.test.ts`:

```typescript
import * as assert from 'assert';
import * as vscode from 'vscode';

suite('Extension Test Suite', () => {
  suiteTeardown(() => {
    vscode.window.showInformationMessage('All tests done!');
  });

  test('Sample test', () => {
    assert.strictEqual(-1, [1, 2, 3].indexOf(5));
    assert.strictEqual(-1, [1, 2, 3].indexOf(0));
  });

  test('Command test', async () => {
    // 测试命令是否注册
    const commands = await vscode.commands.getCommands();
    assert.ok(commands.includes('smart-developer-tools.helloWorld'));
  });
});
```

### 运行测试

```bash
# 运行所有测试
npm test

# 使用 Extension Test Runner 扩展
# 在 VSCode 中运行 "Test: Run All Tests"

# 调试测试
# 在 VSCode 中运行 "Test: Debug All Tests"
```

---

## 常见问题

### Q1: 为什么看不到我的命令？

**A**: 检查以下几点：
1. `package.json` 中 `engines.vscode` 版本是否与安装的 VSCode 兼容
2. 命令是否正确注册在 `contributes.commands` 中
3. TypeScript 是否编译成功（查看终端输出）
4. Extension Development Host 是否已重新加载

### Q2: 如何查看扩展输出？

**A**: 
- 打开 "输出" 面板 (`Ctrl+Shift+U`)
- 从下拉菜单选择你的扩展名称
- 或使用 `console.log()` 输出到调试控制台

### Q3: 调试时断点不生效？

**A**:
1. 确保 `tsconfig.json` 中 `sourceMap: true`
2. 检查 `launch.json` 中的 `outFiles` 路径是否正确
3. 重新编译代码: `npm run compile`
4. 重启调试会话

### Q4: 如何发布扩展？

**A**:
1. 安装 vsce 工具: `npm install -g @vscode/vsce`
2. 创建 publisher: `vsce create-publisher <publisher-name>`
3. 登录: `vsce login <publisher-name>`
4. 打包: `vsce package`
5. 发布: `vsce publish`

### Q5: 扩展冲突如何排查？

**A**:
1. 使用 `--disable-extensions` 参数启动，只加载你的扩展
2. 逐个启用其他扩展，找出冲突源
3. 检查扩展的 `activationEvents` 是否有冲突
4. 查看 VSCode 开发者工具 (`Help > Toggle Developer Tools`)

---

## 进阶资源

### 官方文档

- [Your First Extension](https://code.visualstudio.com/api/get-started/your-first-extension) - 官方入门教程
- [Extension API](https://code.visualstudio.com/api/references/vscode-api) - 完整 API 文档
- [Contribution Points](https://code.visualstudio.com/api/references/contribution-points) - 配置扩展点
- [UX Guidelines](https://code.visualstudio.com/api/ux-guidelines/overview) - 用户体验设计指南

### 示例代码

- [官方示例仓库](https://github.com/microsoft/vscode-extension-samples) - 包含各种扩展示例
- [Hello World Sample](https://github.com/microsoft/vscode-extension-samples/tree/main/helloworld-sample) - 基础示例
- [Test Sample](https://github.com/microsoft/vscode-extension-samples/tree/main/helloworld-test-sample) - 测试示例

### 扩展类型

VSCode 支持多种扩展类型：

- **命令扩展**: 添加自定义命令
- **主题扩展**: 颜色和图标主题
- **语言扩展**: 语法高亮、代码补全、LSP
- **调试器扩展**: 自定义调试器适配器
- **Webview 扩展**: 自定义 UI 面板
- **树视图扩展**: 侧边栏树形视图
- **AI 扩展**: 语言模型、Chat 参与者

### 下一步

掌握基础后，可以探索：

- [Tree View API](https://code.visualstudio.com/api/extension-guides/tree-view) - 创建自定义侧边栏视图
- [Webview API](https://code.visualstudio.com/api/extension-guides/webview) - 构建富 UI 界面
- [Language Server Extension](https://code.visualstudio.com/api/language-extensions/language-server-extension-guide) - 实现语言支持
- [Chat Participant](https://code.visualstudio.com/api/extension-guides/ai/chat) - 集成 AI 助手

---

## 总结

VSCode 扩展开发的关键要点：

1. **从简单开始** - 每个复杂扩展都从 "Hello World" 起步
2. **package.json 是核心** - 它定义了扩展与 VSCode 的契约
3. **TypeScript 是最佳实践** - 类型安全避免运行时错误
4. **调试体验优秀** - VSCode 让扩展开发非常愉快
5. **测试很重要** - 使用集成测试确保扩展质量

开始你的 VSCode 扩展开发之旅吧！🚀

---

*最后更新: 2026年3月20日*

*参考资源:*
- [Your First Extension | Visual Studio Code Extension API](https://code.visualstudio.com/api/get-started/your-first-extension)
- [Testing Extensions | Visual Studio Code Extension API](https://code.visualstudio.com/api/working-with-extensions/testing-extension)
- [Debugging extensions - vscode-docs](https://vscode-docs.readthedocs.io/en/stable/extensions/debugging-extensions/)
- [Build Your First VSCode Extension in 15 Minutes](https://dev.to/dr_rvinobchander_ac6a/build-your-first-vscode-extension-in-15-minutes-complete-beginners-guide-4ljj)
