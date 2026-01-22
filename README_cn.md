# AIBaton

[English](README.md)

**Headless Agent自动化** — codex/claude CLI 的 Python 封装，基于可编程提示词实现任意复杂的agent自动化/工作流。

## 特性

- **面向提示词编程** — 自然语言提示词即python代码，和agent无缝交互
- **skill支持** — 使用skill将意图转为aibaton代码，复杂任务更可控
- **循环检测** — 内置ralph迭代优化(`loop_max`)
- **响应解析** — agent结果使用`parse(options)`语义选项解析

## 适用场景

- **重复性任务** — 可重复使用一套提示词流程和agent交互执行的批量任务
- **上下文爆炸任务** — 上下文非常多，agent忽略太多细节，需要更精细化控制的任务
- **自动化工作流** — 直接基于agent实现工作流，简单python语法实现任意复杂逻辑

## 不适用场景

- **非重复式任务** — 不需要写成脚本重复使用
- **复杂任务逐步确认** — 难以通过脚本预先设定所有情况，需人工确认Agent每个阶段结果

## 示例

使用codex安装aibaton SKILL，然后输入下面提示词：
```text
$aibaton 帮我查找项目中超过300行的go代码文件并优化，提取冗余代码为子函数，遵循DRY原则，并修改其他文件中引用。
```

输出可执行脚本：
```python
from aibaton import run, set_default
import os

set_default(provider="codex", dangerous_permissions=True)

prompt = """审查当前文件代码并优化，提取冗余代码为子函数，遵循DRY原则，并修改其他文件中引用。"""

def get_go_files(root_dir: str, min_lines: int = 0) -> list[str]:
    go_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith('.go'):
                fpath = os.path.join(dirpath, f)
                if min_lines <= 0 or sum(1 for _ in open(fpath)) > min_lines:
                    go_files.append(fpath)
    return go_files

if __name__ == '__main__':
    par_dir = '/app/banbot'
    for fpath in get_go_files(par_dir, min_lines=300):
        run(f"文件路径: {fpath}\n\n{prompt}", cwd=par_dir, add_dirs=[par_dir], stream=True)
```
按提示运行脚本即可。

### 更复杂的任务示例
[一个复杂的工作流提示词](examples_cn/work_flow_prompt.md)

经过codex使用aibaton skill，输出可执行脚本：[work_flow.py](examples_cn/work_flow.py)

## 安装

```bash
pip install aibaton
# 或
uv pip install aibaton
```

## 许可证

Apache 2.0
