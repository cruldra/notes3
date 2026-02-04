---
sidebar_position: 2
---

给定一个 `llms.txt` 文件，本工具提供了一个 CLI 和 Python API 来解析该文件并从中创建一个 XML 上下文文件。输入文件应遵循此格式：

```markdown
# FastHTML

> FastHTML is a python library which...

When writing FastHTML apps remember to:

- Thing to remember

## Docs

- [Surreal](https://host/README.md): Tiny jQuery alternative with Locality of Behavior
- [FastHTML quick start](https://host/quickstart.html.md): An overview of FastHTML features

## Examples

- [Todo app](https://host/adv_app.py)

## Optional

- [Starlette docs](https://host/starlette-sml.md): A subset of the Starlette docs
```

## 安装

```bash
pip install llms-txt
```

## 如何使用

### CLI

安装后，`llms_txt2ctx` 将在您的终端中可用。

获取 CLI 帮助：

```bash
llms_txt2ctx -h
```

将 `llms.txt` 文件转换为 XML 上下文并保存为 `llms.md`：

```bash
llms_txt2ctx llms.txt > llms.md
```

传递 `--optional True` 以添加输入文件的“optional”（可选）部分。

### Python 模块

```python
from llms_txt import *

samp = Path('llms-sample.txt').read_text()
```

使用 `parse_llms_file` 创建一个包含 llms.txt 文件各部分的数据结构（如果需要，您也可以添加 `optional=True`）：

```python
parsed = parse_llms_file(samp)
list(parsed)

['title', 'summary', 'info', 'sections']

parsed.title,parsed.summary

('FastHTML',
 'FastHTML is a python library which brings together Starlette, Uvicorn, HTMX, and fastcore\'s `FT` "FastTags" into a library for creating server-rendered hypermedia applications.')

list(parsed.sections)

['Docs', 'Examples', 'Optional']

parsed.sections.Optional[0]

{ 'desc': 'A subset of the Starlette documentation useful for FastHTML '
          'development.',
  'title': 'Starlette full documentation',
  'url': 'https://gist.githubusercontent.com/jph00/809e4a4808d4510be0e3dc9565e9cbd3/raw/9b717589ca44cedc8aaf00b2b8cacef922964c0f/starlette-sml.md'}
```

使用 `create_ctx` 创建一个包含 XML 部分的 LLM 上下文文件，适用于 Claude 等系统（这正是 CLI 在幕后调用的功能）。

```python
ctx = create_ctx(samp)

print(ctx[:300])

<project title="FastHTML" summary='FastHTML is a python library which brings together Starlette, Uvicorn, HTMX, and fastcore&#39;s `FT` "FastTags" into a library for creating server-rendered hypermedia applications.'>
Remember:

- Use `serve()` for running uvicorn (`if __name__ == "__main__"` is not
```

### 实现与测试

为了展示解析 `llms.txt` 文件是多么简单，这里提供了一个少于 20 行代码且无依赖的完整解析器：

```python
from pathlib import Path
import re,itertools

def chunked(it, chunk_sz):
    it = iter(it)
    return iter(lambda: list(itertools.islice(it, chunk_sz)), [])

def parse_llms_txt(txt):
    "Parse llms.txt file contents in `txt` to a `dict`"
    def _p(links):
        link_pat = '-\\s*\\[(?P<title>[^\\]]+)\\]\\((?P<url>[^\\)]+)\\)(?::\\s*(?P<desc>.*))?'
        return [re.search(link_pat, l).groupdict()\
                for l in re.split(r'\\n+', links.strip()) if l.strip()]

    start,*rest = re.split(fr'^##\\s*(.*?$)', txt, flags=re.MULTILINE)
    sects = {k: _p(v) for k,v in dict(chunked(rest, 2)).items()}
    pat = '^#\\s*(?P<title>.+?$)\\n+(?:^>\\s*(?P<summary>.+?$)$)?\\n+(?P<info>.*)'
    d = re.search(pat, start.strip(), (re.MULTILINE|re.DOTALL)).groupdict()
    d['sections'] = sects
    return d
```

我们在 `tests/test-parse.py` 中提供了一个测试套件，并确认该实现通过了所有测试。
