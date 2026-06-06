# pre-commit

> 本文翻译自 [pre-commit 官方文档](https://pre-commit.com/)，目标是把整篇教程系统地介绍一遍。文档本身分四块：介绍与安装、配置（`.pre-commit-config.yaml` 写法）、用法与命令、进阶特性。原文按章节锚点 `#intro` / `#installation` / `#quick-start` / `#usage` / `#advanced-features` 组织，译文保留同名小节便于回查。

---

## 简介

Git hook 脚本的价值在于"在提交前"就把简单问题拦下来。我们在每次 `git commit` 时自动跑 hook，揪出诸如缺分号、行尾空白、调试语句之类的低级问题。把这些琐碎的事前解决，code reviewer 才能把精力集中在变更的架构与逻辑上，而不是被格式 nit 耗掉。

随着维护的开源库和项目越来越多，我们发现一个痛点：**跨项目复用 pre-commit hook 极其痛苦**。每换一个项目就复制粘贴一段 bash 脚本，再针对不同项目结构手改一遍，脆弱到不行。

我们坚信"该用业界最好的 linter"。但现实是，最强的 linter 往往不是用你项目的语言写的，甚至机器上没装。比如 `scss-lint` 是 Ruby 写的 SCSS 检查器；你在 Node 项目里想用它作为 hook，难道还得给项目加一个 `Gemfile`，弄明白怎么把 `scss-lint` 装上吗？

`pre-commit` 就是为了解决这一摊事而生的：它是一个**多语言的 hook 包管理器**。你只需要在配置文件里列出想要的 hook，pre-commit 负责安装和执行——不管 hook 本身用什么语言写。它的一个关键设计是**不需要 root 权限**。比如某个开发者没装 Node，但他改了一个 JS 文件，pre-commit 会自动下载并构建一份 Node 出来跑 eslint，全程不需要 sudo。

## 安装

跑 hook 之前先把 pre-commit 包管理器装上。

**pip 装：**

```bash
pip install pre-commit
```

**Python 项目里**则推荐写到 `requirements.txt`（或 `requirements-dev.txt`）里：

```text
pre-commit
```

**零依赖 zipapp：** 如果不想污染环境，可以直接用发布包里的 `.pyz` 文件：

1. 去 [GitHub releases](https://github.com/pre-commit/pre-commit/releases) 下载对应版本的 `.pyz`
2. 用 `python pre-commit-#.#.#.pyz ...` 代替 `pre-commit ...` 来调用

## 快速开始

### 1. 安装 pre-commit

按上文安装，然后验证：

```bash
$ pre-commit --version
pre-commit 4.6.0
```

### 2. 写一份配置

在仓库根目录创建 `.pre-commit-config.yaml`：

- 也可以用 [`pre-commit sample-config`](#pre-commit-sample-config-options) 生成一份最简骨架
- 配置项全表见 [`.pre-commit-config.yaml` 完整说明](#pre-commit-configyaml-的配置)

下面这个示例用了 Python 的格式化器，但 pre-commit **不挑语言**，任何语言的 hook 都能跑。

```yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v2.3.0
    hooks:
    -   id: check-yaml
    -   id: end-of-file-fixer
    -   id: trailing-whitespace
-   repo: https://github.com/psf/black
    rev: 22.10.0
    hooks:
    -   id: black
```

### 3. 安装 git hook 脚本

```bash
$ pre-commit install
pre-commit installed at .git/hooks/pre-commit
```

此后 `git commit` 就会自动触发 pre-commit。

### 4. （可选）先对全量文件跑一遍

加新 hook 时建议先对整个仓库跑一次，因为 pre-commit 在 git hook 上下文里默认只跑**本次改动**的文件：

```bash
$ pre-commit run --all-files
[INFO] Initializing environment for https://github.com/pre-commit/pre-commit-hooks.
[INFO] Initializing environment for https://github.com/psf/black.
[INFO] Installing environment for https://github.com/pre-commit/pre-commit-hooks.
[INFO] Once installed this environment will be reused.
[INFO] This may take a few minutes...
[INFO] Installing environment for https://github.com/psf/black.
[INFO] Once installed this environment will be reused.
[INFO] This may take a few minutes...
Check Yaml...............................................................Passed
Fix End of Files.........................................................Passed
Trim Trailing Whitespace.................................................Failed
- hook id: trailing-whitespace
- exit code: 1
Files were modified by this hook. Additional output:
Fixing sample.py
black....................................................................Passed
```

上面这次 `Trim Trailing Whitespace` 失败了，并且**自动改了文件**——这正是 pre-commit 的典型行为：能修就修，修不了才报错退出。这种"修一下再来"的工作流建议配合 CI 一起跑，作为"机器做兜底"。

## 把 pre-commit 加到项目里

装好 pre-commit 之后，怎么用 hook 全靠仓库根目录的 `.pre-commit-config.yaml`。

### `.pre-commit-config.yaml` —— 顶层字段

| 字段 | 说明 |
| --- | --- |
| [`repos`](#pre-commit-configyaml--repos) | 仓库映射列表 |
| [`default_install_hook_types`](#pre-commit-install-options) | （可选，默认 `[pre-commit]`）`pre-commit install` 时默认安装的 `--hook-type` 列表 |
| [`default_language_version`](#top-level-default_language_version) | （可选，默认 `{}`）按语言给出默认 `language_version`，会被未显式设置的 hook 继承 |
| [`default_stages`](#confining-hooks-to-run-at-certain-stages) | （可选，默认全 stage）所有 hook 的默认 `stages` |
| [`files`](#top-level-files) | （可选，默认 `''`）全局文件包含模式 |
| `exclude` | （可选，默认 `^$`）全局文件排除模式 |
| `fail_fast` | （可选，默认 `false`）第一个 hook 失败就停 |
| `minimum_pre_commit_version` | （可选，默认 `'0'`）声明最低 pre-commit 版本要求 |

一个典型的顶层片段：

```yaml
exclude: '^$'
fail_fast: false
repos:
-   ...
```

### `.pre-commit-config.yaml` —— `repos` 项

`repos` 列表里的每一项告诉 pre-commit"这个 hook 从哪拉"。

| 字段 | 说明 |
| --- | --- |
| `repo` | `git clone` 用的仓库地址；也可以是特殊值 [`local`](#repository-local-hooks) / [`meta`](#meta-hooks) |
| `rev` | 要 clone 的 tag / 分支 / SHA |
| `hooks` | 该仓库下要启用的 [hook 列表](#pre-commit-configyaml--hooks) |

```yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v1.2.3
    hooks:
    -   ...
```

### `.pre-commit-config.yaml` —— `hooks` 项

| 字段 | 说明 |
| --- | --- |
| `id` | 仓库里的 hook id |
| `alias` | （可选）`pre-commit run <hookid>` 时可以用的别名 |
| `name` | （可选）覆盖执行时显示的 hook 名字 |
| `language_version` | （可选）覆盖该 hook 的语言版本 |
| `files` | （可选）覆盖默认的匹配文件模式 |
| `exclude` | （可选）文件排除模式 |
| `types` | （可选）覆盖默认文件类型（AND 关系） |
| `types_or` | （可选）覆盖默认文件类型（OR 关系） |
| `exclude_types` | （可选）要排除的文件类型 |
| `args` | （可选）传给 hook 的额外参数 |
| `stages` | （可选）限定该 hook 在哪些 git hook 触发时跑 |
| `additional_dependencies` | （可选）该 hook 环境下额外安装的依赖，例如 eslint 插件 |
| `always_run` | （可选，默认 `false`）即使没匹配到文件也跑 |
| `verbose` | （可选，默认 `false`）即使 hook 通过也强制打印输出 |
| `log_file` | （可选）失败或 verbose 时把输出额外写到文件 |

完整示例：

```yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v1.2.3
    hooks:
    -   id: trailing-whitespace
```

含义：clone pre-commit-hooks 仓库，只启用 `trailing-whitespace` 这一个 hook。

### 自动更新 hook 版本

```bash
pre-commit autoupdate
```

默认会把所有 `rev` 升到该仓库默认分支的**最新 tag**。配合定期跑（或者 CI 跑）可以避免 hook 长期停留在老版本。

## 用法

跑 `pre-commit install` 把脚本装到 git hooks 里之后，每次 `commit` 都会自动触发 pre-commit。**克隆任何带 pre-commit 的项目后，第一件事就是 `pre-commit install`**。

- 想手动全量跑：`pre-commit run --all-files`
- 想跑单个 hook：`pre-commit run <hook_id>`

第一次跑某个 hook 时 pre-commit 会自动下载、安装并执行。注意**首次会慢**——比如机器上没 Node，pre-commit 就会去下载并构建一份 Node。

```bash
$ pre-commit install
pre-commit installed at /home/asottile/workspace/pytest/.git/hooks/pre-commit
$ git commit -m "Add super awesome feature"
black....................................................................Passed
blacken-docs.........................................(no files to check)Skipped
Trim Trailing Whitespace.................................................Passed
Fix End of Files.........................................................Passed
Check Yaml...........................................(no files to check)Skipped
Debug Statements (Python)................................................Passed
Flake8...................................................................Passed
Reorder python imports...................................................Passed
pyupgrade................................................................Passed
rst \`\`code\`\` is two backticks........................(no files to check)Skipped
rst..................................................(no files to check)Skipped
changelog filenames..................................(no files to check)Skipped
[main 146c6c2c] Add super awesome feature
 1 file changed, 1 insertion(+)
```

## 自己写 hook

pre-commit 支持[许多语言](#支持的脚本语言)。只要你的 git 仓库能打包成可安装的包（gem、npm、pypi 等），或者暴露一个可执行文件，就能成为 pre-commit hook。一个 git 仓库可以容纳任意多种语言、任意多个 hook。

> 自 2.5.0 起，pre-commit 在 hook 执行时会设置环境变量 `PRE_COMMIT=1`。

hook 要么以非零状态码表示失败，要么直接修改文件。

提供 hook 的 git 仓库必须包含一个 `.pre-commit-hooks.yaml` 文件，告诉 pre-commit 这个仓库暴露了哪些 hook：

| 字段 | 说明 |
| --- | --- |
| `id` | hook id，会出现在 `pre-commit-config.yaml` 中 |
| `name` | 执行时显示的名字 |
| `entry` | 入口可执行文件；可以带不会被覆盖的固定参数，例如 `entry: autopep8 -i` |
| `language` | hook 语言，pre-commit 据此决定怎么安装 |
| `files` | （可选，默认 `''`）文件匹配模式 |
| `exclude` | （可选，默认 `^$`）从 `files` 匹配结果中再排除的模式 |
| `types` | （可选，默认 `[file]`）文件类型列表（AND） |
| `types_or` | （可选，默认 `[]`）文件类型列表（OR） |
| `exclude_types` | （可选，默认 `[]`）要排除的文件类型 |
| `always_run` | （可选，默认 `false`）无文件匹配也跑 |
| `fail_fast` | （可选，默认 `false`）该 hook 失败就停 |
| `verbose` | （可选，默认 `false`）即使通过也打印输出 |
| `pass_filenames` | （可选，默认 `true`）设为 `false` 则不传文件名 |
| `require_serial` | （可选，默认 `false`）设为 `true` 时单进程顺序跑 |
| `description` | （可选）描述，仅作元数据 |
| `language_version` | （可选）见 [Overriding language version](#overriding-language-version) |
| `minimum_pre_commit_version` | （可选）最低 pre-commit 版本要求 |
| `args` | （可选，默认 `[]`）传给 hook 的额外参数 |
| `stages` | （可选，默认全 stage）限制在哪些 git hook 触发 |

示例：

```yaml
-   id: trailing-whitespace
    name: Trim Trailing Whitespace
    description: This hook trims trailing whitespace.
    entry: trailing-whitespace-fixer
    language: python
    types: [text]
```

## 交互式开发 hook

`.pre-commit-config.yaml` 里的 `repo` 字段可以指向任何 `git clone` 能解析的地址——所以开发时**直接指本地目录**通常最方便。`pre-commit try-repo` 把这个流程包了一下。

> 用 `prepare-commit-msg` / `commit-msg` 类型的 hook 时，可能需要额外提供 `--commit-msg-filename`。

注意 `try-repo` 在本地目录上**不需要 commit**——pre-commit 会自动 clone 那些已 tracked 的未提交改动。

```bash
~/work/hook-repo $ git checkout origin/main -b feature
# ... make some changes

# 另一个终端
~/work/other-repo $ pre-commit try-repo ../hook-repo foo --verbose --all-files
===============================================================================
Using config:
===============================================================================
repos:
-   repo: ../hook-repo
    rev: 84f01ac09fcd8610824f9626a590b83cfae9bcbd
    hooks:
    -   id: foo
===============================================================================
[INFO] Initializing environment for ../hook-repo.
Foo......................................................................Passed
- hook id: foo
- duration: 0.02s

Hello from foo hook!
```

## 支持的脚本语言

> 完整逐语言细节见 [官方文档 Supported languages](https://pre-commit.com/#supported-languages)。下面给每个语言一个**人话简介 + 适用场景**。

- **python** —— hook 仓库需要能 `pip install .`（一般是 `setup.py` / `pyproject.toml`）。支持 `additional_dependencies`，可以作为 `repo: local` 用。**零系统级依赖**。
- **node** —— 仓库需要 `package.json`，pre-commit 跑 `npm install .`。**零系统级依赖**。
- **ruby** —— 需要 `*.gemspec`，pre-commit 跑 `gem build && gem install`。**零系统级依赖**。
- **rust** —— 走 `cargo install --bins`。**预装 Rust 时**零系统级依赖；pre-commit 也可以自动 bootstrap rust。支持 `{pkg}:{ver}` 和 `cli:{pkg}:{ver}` 两种 `additional_dependencies` 语法。
- **golang** —— hook 仓库里放 Go 源码，pre-commit 跑 `go install ./...`，会为每个 hook 创建独立 `GOPATH`。支持 `additional_dependencies`。**预装 Go 时**零系统级依赖；pre-commit 也可以自动 bootstrap go。3.0.0 起支持 `language_version`。
- **conda** —— 仓库里放 `environment.yml`，pre-commit 跑 `conda env create`。可通过 `PRE_COMMIT_USE_MAMBA=1` / `PRE_COMMIT_USE_MICROMAMBA=1` 切到 mamba / micromamba。需系统装好 conda。
- **coursier** —— 仓库里放 `.pre-commit-channel` 目录，里面是 coursier 的 application descriptor。需要系统装好 `cs` 或 `coursier`。
- **dart** —— 仓库里放 `pubspec.yaml`，里面有 `executables` 段。pre-commit 跑 `dart compile exe`。需要系统装好 Dart SDK。`additional_dependencies` 可用 `pkg:ver` 语法指定版本。
- **docker** —— 仓库根目录有 `Dockerfile`，pre-commit 跑 `docker build .`。**需要宿主机有运行中的 Docker engine**。pre-commit 会自动把仓库以 `-v $PWD:/src:rw,Z` 挂进容器，工作目录设成 `/src`。注意：`boot2docker` 下的 Docker hook **不能修改文件**。
- **docker_image** —— 比 `docker` 更轻量：直接用现成的 docker 镜像，无需 `Dockerfile`。`entry` 字段写 docker tag，可选覆盖 entrypoint。**典型用法就是配合 `repo: local`**。
- **dotnet** —— 仓库是个 dotnet CLI 工具，能 `pack` + `install`。**当前不支持 `additional_dependencies`**。
- **fail** —— 极轻量 hook，按文件名直接 fail。`entry` 字段就是失败时打印的信息。**最适合 `repo: local`**。例：禁止 changelog 目录里出现非 `.rst` 文件。
- **haskell** —— 仓库里有 `*.cabal`，`executable` 段里暴露可执行文件。需要系统装好 `cabal`。
- **julia** —— `entry` 是相对仓库根的 julia 源文件路径。hook 在仓库里 `Project.toml` 定义的隔离环境下跑；没有 `Project.toml` 时跑在空环境里。`additional_dependencies` 用 Pkg REPL 模式语法（如 `'ExtraDepA@1'`）。需要系统装好 `julia`。
- **lua** —— 用 Luarocks 装。需要系统装好 Luarocks。
- **perl** —— 用系统 `cpan` 装。仓库一般是 `Makefile.PL` 或 `Build.PL`。`additional_dependencies` 用 `cpan` 能理解的格式。**需要系统装好 Perl + cpan**。
- **r** —— 仓库里有 `renv.lock`，pre-commit 跑 `renv::restore()`。`entry` 写法是 `Rscript -e {expression}` 或 `Rscript path/relative/to/hook/root`。R 启动流程会跳过（等同于 `--vanilla`），所有配置应通过 `args` 显式传入。需要系统装好 R。
- **swift** —— 仓库里有 `Package.swift`，pre-commit 跑 `swift build -c release`。需要系统装好 swift。
- **pygrep** —— Python 版的 `grep`。`entry` 直接写正则，支持 Python 正则的全部特性。`(?i)` 前缀 = 大小写不敏感；`args: [--multiline]` = 多行匹配；`args: [--negate]` = 必须全部匹配。**全平台支持**。
- **unsupported**（4.4.0 起；原名 `system`）—— 跑系统级可执行文件，**不提供虚拟环境**，需要什么依赖消费者自己装。典型场景：pylint 这种依赖项目虚拟环境的工具。
- **unsupported_script**（4.4.0 起；原名 `script`）—— 同上，但 `entry` 写仓库内脚本路径而不是系统命令。**不提供虚拟环境**。

## 命令行接口

所有 pre-commit 命令都接受以下全局选项：

- `--color {auto,always,never}`：是否彩色输出，默认 `auto`。可被 `PRE_COMMIT_COLOR=...` 或 `TERM=dumb` 覆盖。
- `-c CONFIG` / `--config CONFIG`：指定备选配置文件。
- `-h` / `--help`：帮助。

退出码：

- `1`：检测到 / 预期的错误
- `3`：意外错误
- `130`：被 `^C` 终止

### `pre-commit autoupdate [options]`

把 `rev` 升到各 hook 仓库的最新版本。

- `--bleeding-edge`：升到默认分支的 bleeding edge（commit SHA），而不是最新 tag。
- `--freeze`：用 commit SHA + 注释（`# frozen: v2.4.0`）锁定版本，而不是 tag。
- `--repo REPO`：只更新指定仓库（可重复指定）。
- `-j` / `--jobs`（3.3.0 起）：并发数，默认 1。

```bash
# 默认：升到最新 tag
$ pre-commit autoupdate
Updating https://github.com/pre-commit/pre-commit-hooks ... updating v2.1.0 -> v2.4.0.
Updating https://github.com/asottile/pyupgrade ... updating v1.25.0 -> v1.25.2.

# 只升一个仓库到 bleeding edge
$ pre-commit autoupdate --bleeding-edge --repo https://github.com/pre-commit/pre-commit-hooks
Updating https://github.com/pre-commit/pre-commit-hooks ... updating v2.1.0 -> 5df1a4bf6f04a1ed3a643167b38d502575e29aef.

# 升到 frozen 版本
$ pre-commit autoupdate --freeze
Updating https://github.com/pre-commit/pre-commit-hooks ... updating v2.1.0 -> v2.4.0 (frozen).
```

> 出现同名 tag 时，pre-commit 会**优先选带 `.` 的**（如 `v1.2.3` 而不是 `latest`）。

### `pre-commit clean`

清掉所有缓存的 pre-commit 文件。无额外选项。

### `pre-commit gc`

清掉缓存里**没人用**的 hook 仓库。`pre-commit` 的缓存目录会越积越大，定期 gc 可以瘦身。无额外选项。

### `pre-commit init-templatedir DIRECTORY [options]`

把 hook 脚本装到 `git config init.templateDir` 指向的目录里，实现**新 clone 的仓库自动启用 hook**。

- `-t HOOK_TYPE, --hook-type HOOK_TYPE`：指定 hook 类型。

```bash
git config --global init.templateDir ~/.git-template
pre-commit init-templatedir ~/.git-template
```

Windows cmd.exe 用 `%HOMEPATH%` 代替 `~`；PowerShell 用 `$HOME`。

### `pre-commit install [options]`

把 pre-commit 脚本装到 git hooks。

- `-f` / `--overwrite`：覆盖已有 hook。
- `--install-hooks`：立刻把所有 hook 的环境也装好（而不是第一次跑时才装）。等同 `pre-commit install-hooks`。
- `-t HOOK_TYPE, --hook-type HOOK_TYPE`：装到具体哪个 hook。
- `--allow-missing-config`：允许仓库里没有 `.pre-commit-config.yaml`，缺失时静默跳过。

常用调用：

- `pre-commit install`：默认调用，和已有 hook 共存。
- `pre-commit install --install-hooks --overwrite`：幂等地把 hook 脚本替换为 pre-commit，并提前装好所有 hook 环境。

未传 `--hook-type` 时，pre-commit 会按顶层 [`default_install_hook_types`](#pre-commit-configyaml-的配置) 安装。

### `pre-commit install-hooks [options]`

只把缺失的 hook 环境装齐。**不会装 git hook 脚本**——如果想一步到位，用 `pre-commit install --install-hooks`。

### `pre-commit migrate-config [options]`

把 list 形式的旧配置迁移成 map 形式的新格式。

### `pre-commit run [hook-id] [options]`

跑 hook。

- `[hook-id]`：指定单个 hook id。
- `-a` / `--all-files`：对全仓库跑。
- `--files [FILES ...]`：只对指定文件跑。
- `--from-ref FROM_REF` + `--to-ref TO_REF`：对 `FROM_REF...TO_REF` 之间的变更文件跑。
- `--hook-stage STAGE`：选 stage 跑。
- `--show-diff-on-failure`：失败时自动跑 `git diff`。
- `-v` / `--verbose`：无论成功失败都打印输出，并附上 hook id。

典型用法：

```bash
pre-commit run                                  # 默认：跑所有 hook，针对本次暂存文件
pre-commit run --all-files                     # 跑所有 hook，针对全仓库（CI 常用）
pre-commit run flake8                          # 只跑 flake8，针对本次暂存文件
git ls-files -- '*.py' | xargs pre-commit run --files
pre-commit run --from-ref HEAD^^^ --to-ref HEAD  # 跑最近 3 次 commit 改过的文件
```

### `pre-commit sample-config [options]`

生成一份最简的 `.pre-commit-config.yaml` 骨架。

### `pre-commit try-repo REPO [options]`

临时把指定仓库当作 hook 源跑一遍，适合开发新 hook 或试用新仓库。`try-repo` 会先打印它生成的配置，再实际跑。

- `REPO`：必填，本地路径或可 clone 的 URL。
- `--ref REF`：手动指定要跑的 ref，不传默认 `HEAD`。

支持所有 [`pre-commit run`](#pre-commit-run-hook-id-options) 的选项。

```bash
pre-commit try-repo https://github.com/pre-commit/pre-commit-hooks
pre-commit try-repo ../path/to/repo
pre-commit try-repo ../pre-commit-hooks flake8   # 只跑其中一个 hook
```

### `pre-commit uninstall [options]`

卸掉 pre-commit 脚本。

- `-t HOOK_TYPE, --hook-type HOOK_TYPE`：卸掉指定 hook。

### `pre-commit validate-config [filenames ...]`

校验 `.pre-commit-config.yaml` 是否合法。

### `pre-commit validate-manifest [filenames ...]`

校验 `.pre-commit-hooks.yaml` 是否合法。

## 进阶特性

### 迁移模式运行

默认情况下，如果你已经有 hook 脚本，`pre-commit install` 会进入**迁移模式**：你的旧 hook 和 pre-commit 的 hook 一起跑。不想这样就给 `install` 传 `-f` / `--overwrite` 直接覆盖。哪天不用 pre-commit 了，`pre-commit uninstall` 会把 hook 恢复到安装前的状态。

### 临时禁用 hook

不是所有 hook 都完美，有时候需要跳过某一个——pre-commit 提供 `SKIP` 环境变量，是一个 hook id 的逗号分隔列表。比 `--no-verify` 更精细，可以只跳过单个 hook 而不是整个 commit。

```bash
SKIP=flake8 git commit -m "foo"
```

### 限定 hook 运行的 stage

pre-commit 不止支持 `pre-commit` 这一个 git hook，常见的 git hook 都能挂。

hook 提供方可以通过 `.pre-commit-hooks.yaml` 里的 `stages` 字段指定自己跑在哪些 git hook 上；用户也可以在 `.pre-commit-config.yaml` 里通过 `stages` 覆盖。如果两边都没设，则取顶层 [`default_stages`](#pre-commit-configyaml-的配置)，默认**全 stage**。

> 3.2.0 起 `stages` 的值和 git hook 名字直接对齐。旧值 `commit` / `push` / `merge-commit` 已重命名为 `pre-commit` / `pre-push` / `pre-merge-commit`。

`manual` 是一个特殊 stage：**任何 git hook 都不会自动触发它**。配置成 `stages: [manual]` 后，只能通过 `pre-commit run --hook-stage manual [hookid]` 手动调用——适合放那些"不该自动跑、但需要时可以跑"的工具。

写 hook 时给个合理的 `stages` 是好习惯，比如 linter / formatter 通常写：

```yaml
stages: [pre-commit, pre-merge-commit, pre-push, manual]
```

想给多个 git hook 装 pre-commit，可以传多次 `--hook-type`：

```bash
$ pre-commit install --hook-type pre-commit --hook-type pre-push
pre-commit installed at .git/hooks/pre-commit
pre-commit installed at .git/hooks/pre-push
```

也可以在顶层用 `default_install_hook_types` 一次性指定默认集合：

```yaml
default_install_hook_types: [pre-commit, pre-push, commit-msg]
```

#### 支持的 git hooks

- [commit-msg](https://git-scm.com/docs/githooks#_commit_msg) —— 接收一个文件名（commit message 文件），非零退出即 abort commit。
- [post-checkout](https://git-scm.com/docs/githooks#_post_checkout) —— `checkout` 之后跑。**不操作文件**，所以必须设 `always_run: true`。环境变量：`PRE_COMMIT_FROM_REF` / `PRE_COMMIT_TO_REF` / `PRE_COMMIT_CHECKOUT_TYPE`。
- [post-commit](https://git-scm.com/docs/githooks#_post_commit) —— commit 完成之后跑，**无法阻止 commit**。`always_run: true`。
- [post-merge](https://git-scm.com/docs/githooks#_post_merge) —— `git merge` 成功后跑。`always_run: true`。环境变量：`PRE_COMMIT_IS_SQUASH_MERGE`。
- [post-rewrite](https://git-scm.com/docs/githooks#_post_rewrite) —— 改写历史的命令（`commit --amend` / `rebase`）之后跑。`always_run: true`。环境变量：`PRE_COMMIT_REWRITE_COMMAND`。
- [pre-commit](https://git-scm.com/docs/githooks#_pre_commit) —— commit 提交前跑。**pre-commit 默认只对暂存内容跑**：未暂存的改动会被临时 stash 起来，避免漏报 / 误报。
- [pre-merge-commit](https://git-scm.com/docs/githooks#_pre_merge_commit) —— merge 成功、merge commit 创建前跑，对 merge 进来的所有暂存文件生效。**需要 git ≥ 2.24**。
- [pre-push](https://git-scm.com/docs/githooks#_pre_push) —— `git push` 触发。环境变量：`PRE_COMMIT_FROM_REF` / `PRE_COMMIT_TO_REF` / `PRE_COMMIT_REMOTE_NAME` / `PRE_COMMIT_REMOTE_URL` / `PRE_COMMIT_REMOTE_BRANCH` / `PRE_COMMIT_LOCAL_BRANCH`。
- [pre-rebase](https://git-scm.com/docs/githooks#_pre_rebase) —— rebase 之前跑，失败可中止 rebase。`always_run: true`。环境变量：`PRE_COMMIT_PRE_REBASE_UPSTREAM` / `PRE_COMMIT_PRE_REBASE_BRANCH`。
- [prepare-commit-msg](https://git-scm.com/docs/githooks#_prepare_commit_msg) —— 接收一个文件名（commit message 草稿，可能为空），hook 可以修改其内容。退出非零即 abort。**建议检查 `GIT_EDITOR=:`** 来判断是否会启动编辑器。环境变量：`PRE_COMMIT_COMMIT_MSG_SOURCE` / `PRE_COMMIT_COMMIT_OBJECT_NAME`。

### 给 hook 传参数

有些 hook 需要参数才能正确工作。在 `.pre-commit-config.yaml` 里用 `args` 字段传**静态参数**：

```yaml
-   repo: https://github.com/PyCQA/flake8
    rev: 4.0.1
    hooks:
    -   id: flake8
        args: [--max-line-length=131]
```

这会等价于 `flake8 --max-line-length=131 ...`。

#### hook 接收参数的模式

如果是你**自己写的 hook**，hook 脚本应当接收 `args` 字段的值再接暂存文件列表。假设配置是：

```yaml
-   repo: https://github.com/path/to/your/hook/repo
    rev: badf00ddeadbeef
    hooks:
    -   id: my-hook-script-id
        args: [--myarg1=1, --myarg1=2]
```

下次跑 pre-commit 时，pre-commit 实际会这样调用：

```bash
path/to/script-or-system-exe --myarg1=1 --myarg1=2 dir/file1 dir/file2 file3
```

如果 `args` 为空或没设：

```bash
path/to/script-or-system-exe dir/file1 dir/file2 file3
```

**写 local hook 时**不要把命令参数塞 `args`——local hook 没有"用户级覆盖"的概念，参数会被固定住没法调。**直接写到 `entry` 里**：

```yaml
-   repo: local
    hooks:
    -   id: check-requirements
        name: check requirements files
        language: unsupported
        entry: python -m scripts.check_requirements --compare
        files: ^requirements\.*\.txt$
```

### 仓库内 local hook

local hook 适合这些场景：

- hook 脚本和仓库强耦合，希望跟代码一起分发。
- hook 需要仓库构建产物才能跑（比如 pylint 依赖项目虚拟环境）。
- 某个 linter 的官方仓库没有 pre-commit 元数据。

用 `local` 这个 sentinel 声明 local hook：

```yaml
-   repo: local
```

local hook 可以用任何支持 `additional_dependencies` 或 `docker_image` / `fail` / `pygrep` / `unsupported` / `unsupported_script` 的语言——以前需要专门建一个"空镜像仓库"才能装的依赖，现在直接在 local 里写就行。

`local` hook 必须显式指定 `id` / `name` / `language` / `entry`，以及 `files` / `types`（参见 [Creating new hooks](#自己写-hook)）。

示例配置：

```yaml
-   repo: local
    hooks:
    -   id: pylint
        name: pylint
        entry: pylint
        language: unsupported
        types: [python]
        require_serial: true
    -   id: check-x
        name: Check X
        entry: ./bin/check-x.sh
        language: unsupported_script
        files: \.x$
    -   id: scss-lint
        name: scss-lint
        entry: scss-lint
        language: ruby
        language_version: 2.1.5
        types: [scss]
        additional_dependencies: ['scss_lint:0.52.0']
```

### meta hooks

pre-commit 自带几个用来**检查 pre-commit 配置本身**的 hook，通过 `repo: meta` 启用：

```yaml
-   repo: meta
    hooks:
    -   id: ...
```

| id | 说明 |
| --- | --- |
| `check-hooks-apply` | 确认配置的 hook 至少能匹配到仓库里的一个文件 |
| `check-useless-excludes` | 确认 `exclude` 规则能匹配到至少一个文件（避免写错的"白名单"） |
| `identity` | 一个打印自己所有参数的简单 hook，调试用 |

### `pre-commit hazmat`

"hazardous materials"——给一些特殊场景用的 `entry` 前缀。**强调：用这些通常是坏主意**。

> hazmat 助手**不能**和那些自己会改 `entry` 的语言一起用：`docker` / `docker_image` / `fail` / `julia` / `pygrep` / `r` / `unsupported_script`。

#### `pre-commit hazmat cd`

> 4.5.0 起。

monorepo 场景下切到子目录跑 hook。`entry` 写 `pre-commit hazmat cd <subdir> <real-entry>`，文件名参数会被自动调整。

```yaml
# minimum_pre_commit_version: 4.5.0
repos:
-   repo: ...
    rev: ...
    hooks:
    -   id: example
        alias: example-repo1
        name: example (repo1)
        files: ^repo1/
        # 注意：以 `--` 结尾
        # 把 args: [...] 复制到 entry，然后把 args 留空
        entry: pre-commit hazmat cd repo1 example-bin --arg1 --
        args: []

    -   id: example
        alias: example-repo2
        name: example (repo2)
        files: ^repo2/
        entry: pre-commit hazmat cd repo2 example-bin --arg1 --
        args: []
```

#### `pre-commit hazmat ignore-exit-code`

> 4.5.0 起。

让 hook 忽略非零退出码——用来加 warning 噪声通常不好，但确实需要时它是一条退路。**记得配 `verbose: true`**，否则输出永远会被吞掉。

```yaml
# minimum_pre_commit_version: 4.5.0
repos:
-   repo: ...
    rev: ...
    hooks:
    -   id: example
        # 把 args: [...] 复制到 entry，然后把 args 留空
        entry: pre-commit hazmat ignore-exit-code example-bin --arg1 --
        args: []
        verbose: true
```

#### `pre-commit hazmat n1`

> 4.5.0 起。

有些 hook 只接受一个文件名参数。`n1` 强制一次只传一个文件（**会很慢**）。

```yaml
# minimum_pre_commit_version: 4.5.0
repos:
-   repo: ...
    rev: ...
    hooks:
    -   id: example
        # 注意：以 `--` 结尾
        # 把 args: [...] 复制到 entry，然后把 args 留空
        entry: pre-commit hazmat n1 example-bin --arg1 --
        args: []
```

### git 2.54+ 的 config-based hook

> 4.6.0 起：pre-commit 改进了对 `git config` 配置 hook 的支持。后续版本会把 `pre-commit install` 默认改成这种新方式。

[git 2.54](https://github.blog/open-source/git/highlights-from-git-2-54/#h-config-based-hooks) 引入了一种新的 hook 启用方式：通过 `git config`。

基本模式：

```bash
git config set hook.<name>.event pre-push
git config set hook.<name>.command 'some command here'
```

用 pre-commit 启用时：

```bash
# "hook" 名字格式：<tool>.<event>
# 这里就是 pre-commit.pre-commit
git config set hook.pre-commit.pre-commit.event pre-commit
git config set hook.pre-commit.pre-commit.command 'pre-commit hook-impl --hook-type pre-commit --'

# pre-push 同理
# git config set hook.pre-commit.pre-push.event pre-push
# git config set hook.pre-commit.pre-push.command 'pre-commit hook-impl --hook-type pre-push --'
```

**注意按 `<tool>.<event>` 命名以保持 `pre-commit install` 的兼容性**。

`pre-commit hook-impl` 是一个"隐藏"实现命令，选项：

- `--hook-type ...`：指定 [hook 类型](#支持的-git-hooks)
- `--config ...`：（可选）`.pre-commit-config.yaml` 路径
- `--skip-on-missing-config`：配置文件缺失时静默放过

#### "全局"安装 pre-commit

配合 `git config set --global ...`，可以让所有仓库自动启用 pre-commit：

```bash
git config set --global hook.pre-commit.pre-commit.event pre-commit
git config set --global hook.pre-commit.pre-commit.command 'pre-commit hook-impl --hook-type pre-commit --skip-on-missing-config --'
```

- **这种用法不推荐**：clone 一个不可信仓库时，hook 会被自动触发，可能执行意料之外的操作。
- `--skip-on-missing-config` 强烈建议加上，因为不是所有仓库都有 `.pre-commit-config.yaml`。

#### 总是对全量文件跑某个 hook

因为 `git config` 可以随便重复设 hook，所以可以让一个 hook 永远跑、永远跑全量文件：

```bash
git config set hook.pre-commit.pre-commit-always.event pre-commit
git config set hook.pre-commit.pre-commit-always.command 'pre-commit run hookid --hook-stage pre-commit --all-files'
```

> 同样**不推荐**：又慢又背离 pre-commit 的"只在暂存文件上跑"的预期。

### 让新仓库自动启用 pre-commit

> 如果你 git 版本够新，**优先用上面 [git 2.54+ 的 config-based hook](#git-254-的-config-based-hook)**。

`pre-commit init-templatedir` 可以给 `git` 的 `init.templateDir` 选项准备一个模板目录——这样**新 clone 的仓库会自动带好 hook**，不用每次手动跑 `pre-commit install`。

```bash
$ git config --global init.templateDir ~/.git-template
$ pre-commit init-templatedir ~/.git-template
pre-commit installed at /home/asottile/.git-template/hooks/pre-commit
```

之后 clone 任何带 pre-commit 的项目都会自动配置好：

```bash
$ git clone -q git@github.com:asottile/pyupgrade
$ cd pyupgrade
$ git commit --allow-empty -m 'Hello world!'
Check docstring is first.............................(no files to check)Skipped
Check Yaml...........................................(no files to check)Skipped
Debug Statements (Python)............................(no files to check)Skipped
...
```

`init-templatedir` 内部用了 `pre-commit install` 的 `--allow-missing-config`，所以没配 `.pre-commit-config.yaml` 的仓库会被静默跳过：

```bash
$ git init sample
Initialized empty Git repository in /tmp/sample/.git/
$ cd sample
$ git commit --allow-empty -m 'Initial commit'
`.pre-commit-config.yaml` config file not found. Skipping `pre-commit`.
[main (root-commit) d1b39c1] Initial commit
```

如果想"允许用户不主动 `pre-commit install`，但要提醒他装"，可以在 `~/.git-template/hooks/pre-commit` 里塞个这样的脚本：

```bash
#!/usr/bin/env bash
if [ -f .pre-commit-config.yaml ]; then
    echo 'pre-commit configuration detected, but `pre-commit install` was never run' 1>&2
    exit 1
fi
```

这样漏装 `pre-commit install` 时 commit 会报错：

```bash
$ git clone -q https://github.com/asottile/pyupgrade
$ cd pyupgrade/
$ git commit -m 'foo'
pre-commit configuration detected, but `pre-commit install` was never run
```

### 用 `types` 过滤文件

相比 `files` 的正则匹配，`types` 有几个明显优点：

- 不用写易错正则
- 能按 shebang 匹配（即使没扩展名）
- 符号链接 / 子模块可以方便地排除

`types` 是按 hook 配的字符串数组，标签由 [identify](https://github.com/pre-commit/identify) 这个小型 Python 库启发式地推断出来。

常见标签：

- `file`
- `symlink`
- `directory`（在 pre-commit 上下文里就是子模块）
- `executable`（可执行位）
- `text` / `binary`（文本 / 二进制）
- [按扩展名 / 命名约定的标签](https://github.com/pre-commit/identify/blob/main/identify/extensions.py)
- [按 shebang (`#!`) 的标签](https://github.com/pre-commit/identify/blob/main/identify/interpreters.py)

不确定某文件是什么类型时，可以用 identify 自带的 cli：

```bash
$ identify-cli setup.py
["file", "non-executable", "python", "text"]
$ identify-cli some-random-file
["file", "non-executable", "text"]
$ identify-cli --filename-only some-random-file; echo $?
1
```

> 自家项目用了不识别的扩展名，欢迎去 [identify 提 PR](https://github.com/pre-commit/identify)。

`types` / `types_or` / `files` 三者之间是 **AND**；`types` 内部的标签也是 AND；`types_or` 内部是 OR。

```yaml
files: ^foo/
types: [file, python]
```

匹配 `foo/1.py`，不匹配 `setup.py`。

```yaml
files: ^foo/
types_or: [javascript, jsx, ts, tsx]
```

匹配 `foo/bar.js` / `foo/bar.jsx` / `foo/bar.ts` / `foo/bar.tsx`，不匹配 `baz.js`。

想让 `check-json` 检查非 json 扩展名的文件时，可以把 `types` 覆盖掉：

```yaml
-   id: check-json
    types: [file]  # 覆盖原配置里的 types: [json]
    files: \.(json|myext)$
```

`types: python` 也能匹配到 `exe` 文件——只要 shebang 写的是 `#!/usr/bin/env python3`。

和 `files` / `exclude` 一样，也可以用 `exclude_types` 排除类型。

### 正则表达式

`files` 和 `exclude` 用的都是 Python 的 [`re`](https://docs.python.org/3/library/re.html#regular-expression-syntax)，匹配方式为 [`re.search`](https://docs.python.org/3/library/re.html#re.search)，所以 Python 正则支持的全套语法都能用。

排除 / 包含项一长起来正则就会很丑——这种时候可以用 `re.VERBOSE` 模式 + YAML 的多行字面量，配上 `(?x)` 标志：

```yaml
# ...
    -   id: my-hook
        exclude: |
            (?x)^(
                path/to/file1.py|
                path/to/file2.py|
                path/to/file3.py
            )$
```

### 覆盖语言版本

有时你只想让 hook 跑在某个特定的语言版本上。默认情况下，每种语言会**使用系统装的版本**（比如你系统是 python3.7，hook 指定 `python`，pre-commit 就用 python3.7 跑）。想覆盖时给 hook 设 `language_version`：

```yaml
-   repo: https://github.com/pre-commit/mirrors-scss-lint
    rev: v0.54.0
    hooks:
    -   id: scss-lint
        language_version: 2.1.5
```

这会让 pre-commit 用 ruby 2.1.5 跑 `scss-lint`。

各语言的有效值：

- **python**：系统装的所有 python 解释器。该参数会作为 `-p` 传给 `virtualenv`。Windows 上 [PEP 394](https://www.python.org/dev/peps/pep-0394/) 名字会翻译成 py launcher，所以仍然写 `python3`（`py -3`）或 `python3.6`（`py -3.6`）。
- **node**：见 [nodeenv](https://github.com/ekalinin/nodeenv#advanced)。
- **ruby**：见 [ruby-build](https://github.com/sstephenson/ruby-build/tree/master/share/ruby-build)。
- **rust**：传给 `rustup`。
- **golang**（3.0.0 起）：用 [go.dev/dl](https://go.dev/dl/) 的版本号，如 `1.19.5`。

也可以在配置顶层用 `default_language_version` 给所有未显式指定的 hook 设置默认版本：

```yaml
default_language_version:
    # 强制所有未指定的 python hook 用 python3
    python: python3
    # 强制所有未指定的 ruby hook 用 ruby 2.1.5
    ruby: 2.1.5
```

### 给仓库加 badge

可以给仓库加一个徽章，告诉贡献者 / 用户你在用 pre-commit：

```markdown
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
```

**Markdown：**

```markdown
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
```

**HTML：**

```html
<a href="https://github.com/pre-commit/pre-commit"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit" alt="pre-commit" style="max-width:100%;"></a>
```

**reStructuredText：**

```rst
.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
```

**AsciiDoc：**

```asciidoc
image:https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit[pre-commit, link=https://github.com/pre-commit/pre-commit]
```

### 在 CI 里跑

pre-commit 也可以作为 CI 工具用。简单一句 `pre-commit run --all-files` 当 CI 步骤就能保证仓库一直处于"健康"状态。想跑快一点只检查变更文件：

```bash
pre-commit run --from-ref origin/HEAD --to-ref HEAD
```

#### CI 缓存管理

pre-commit 默认把仓库缓存放在 `~/.cache/pre-commit`，有两种方式改：

- `PRE_COMMIT_HOME`：优先使用这个环境变量指向的位置。
- `XDG_CACHE_HOME`：按 [XDG Base Directory 规范](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html) 用 `$XDG_CACHE_HOME/pre-commit`。

##### pre-commit.ci

接入 [pre-commit.ci](https://pre-commit.ci/) **零额外配置**。它的额外好处：

- 比多数免费 CI 方案更快
- 会**自动修复** PR（autofix）
- 会**定期自动更新**你的 hook 版本

##### AppVeyor

```yaml
cache:
- '%USERPROFILE%\.cache\pre-commit'
```

##### Azure Pipelines

> 注意：Azure Pipelines 的 cache 是**不可变**的，所以 cache key 必须包含 python 版本和 `.pre-commit-config.yaml` 的 hash。完整模板见 [asottile@job--pre-commit.yml](https://github.com/asottile/azure-pipeline-templates/blob/main/job--pre-commit.yml)。

```yaml
jobs:
- job: precommit
  # ...
  variables:
    PRE_COMMIT_HOME: $(Pipeline.Workspace)/pre-commit-cache

  steps:
  # ...
  - script: echo "##vso[task.setvariable variable=PY]$(python -VV)"
  - task: CacheBeta@0
    inputs:
      key: pre-commit | .pre-commit-config.yaml | "$(PY)"
      path: $(PRE_COMMIT_HOME)
```

##### CircleCI

> 和 Azure Pipelines 一样，cache 是不可变的。

```yaml
  steps:
  - run:
    command: |
      cp .pre-commit-config.yaml pre-commit-cache-key.txt
      python --version --version >> pre-commit-cache-key.txt
  - restore_cache:
    keys:
    - v1-pc-cache-{{ checksum "pre-commit-cache-key.txt" }}
  # ...
  - save_cache:
    key: v1-pc-cache-{{ checksum "pre-commit-cache-key.txt" }}
    paths:
      - ~/.cache/pre-commit
```

> 来源：[@chriselion](https://github.com/Unity-Technologies/ml-agents/pull/3094/files#diff-1d37e48f9ceff6d8030570cd36286a61)

##### GitHub Actions

> 推荐用 [官方 pre-commit GitHub Action](https://github.com/pre-commit/action)。

```yaml
    - name: set PY
      run: echo "PY=$(python -VV | sha256sum | cut -d' ' -f1)" >> $GITHUB_ENV
    - uses: actions/cache@v3
      with:
        path: ~/.cache/pre-commit
        key: pre-commit|${{ env.PY }}|${{ hashFiles('.pre-commit-config.yaml') }}
```

##### GitLab CI

参考 [GitLab 缓存最佳实践](https://docs.gitlab.com/ee/ci/caching/#good-caching-practices) 调整 cache 粒度。

```yaml
my_job:
  variables:
    PRE_COMMIT_HOME: ${CI_PROJECT_DIR}/.cache/pre-commit
  cache:
    paths:
      - ${PRE_COMMIT_HOME}
```

> pre-commit 的 cache 需要在不同 build 之间**保持路径不变**。GitLab k8s runner 默认不是这样——如果遇到 `InvalidManifestError`，把 `[[runner]]` 里的 `builds_dir` 改成静态路径，比如 `builds_dir = "/builds"`。

##### Travis CI

```yaml
cache:
  directories:
  - $HOME/.cache/pre-commit
```

### 和 tox 一起用

[tox](https://tox.readthedocs.io/) 经常被用来配置测试 / CI 工具，包括 pre-commit。`tox>=2` 的一个特性是会**清理环境变量**让测试更可复现——pre-commit 在某些情况下需要几个环境变量，所以得显式放行。

通过 ssh 克隆仓库（`repo: git@github.com:...`）时，git 需要 `SSH_AUTH_SOCK`：

```ini
[testenv]
passenv = SSH_AUTH_SOCK
```

否则会报：

```text
[INFO] Initializing environment for git@github.com:pre-commit/pre-commit-hooks.
An unexpected error has occurred: CalledProcessError: command: ('/usr/bin/git', 'fetch', 'origin', '--tags')
return code: 128
expected return code: 0
stdout: (none)
stderr:
    git@github.com: Permission denied (publickey).
    fatal: Could not read from remote repository.
    Please make sure that you have the correct access rights
    and the repository exists.
Check the log at /home/asottile/.cache/pre-commit/pre-commit.log
```

通过 http(s) 克隆（`repo: https://github.com:...`）时，公司内网代理环境下需要 `http_proxy` / `https_proxy` / `no_proxy`：

```ini
[testenv]
passenv = http_proxy https_proxy no_proxy
```

### 仓库"最新版"的使用

pre-commit 的设计目标是**可重复 + 快速**，因此**故意不提供**"unpinned latest"（不锁版本用最新版）的选项。

要升级到最新版本就用 [`pre-commit autoupdate`](#pre-commit-autoupdate-options)。要"绝对最新"（不是最新 tag），传 `--bleeding-edge`——这会升到该仓库默认分支的 HEAD commit。

`pre-commit` 假设 `rev` 是**不可变**的引用（tag 或 SHA），并基于此缓存。用分支名（或者 `HEAD`）作为 `rev` **不**被支持——它只代表**安装时**那个可变 ref 的状态，**不会自动更新**。

## 贡献

项目希望更多人参与贡献——尤其是支持更多语言 / 版本。也希望给流行 linter 加 `.pre-commit-hooks.yaml`，这样就不必再维护 fork / mirror 了。

欢迎提 bug 报告、PR 和功能请求。

## 赞助

如果你或你的公司愿意支持 pre-commit 的开发，可以：

- [GitHub Sponsors（asottile）](https://github.com/sponsors/asottile)
- [Open Collective](https://opencollective.com/pre-commit)

## 获取帮助

- 在 [Stack Overflow 上用 pre-commit.com 标签提问](https://stackoverflow.com/questions/tagged/pre-commit.com)
- 在 [pre-commit/pre-commit](https://github.com/pre-commit/pre-commit/issues/) 提 issue
- 在 [asottile 的 Twitch Discord](https://discord.gg/xDKGPaW) 的 `#pre-commit` 频道

## 贡献者

- 网站：[Molly Finkle](https://github.com/mfnkl)
- 创建者：[Anthony Sottile](https://github.com/asottile)
- 核心开发者：[Ken Struys](https://github.com/struys)、[Chris Kuehl](https://github.com/chriskuehl)
- [框架贡献者](https://github.com/pre-commit/pre-commit/graphs/contributors)
- [核心 hook 贡献者](https://github.com/pre-commit/pre-commit-hooks/graphs/contributors)
- 还有用户朋友们
