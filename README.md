# AIBaton

[中文文档](README_cn.md)

**Headless Agent Automation** — Python wrapper for codex/claude CLI, enabling arbitrarily complex agent automation/workflows through programmable prompts.

## Features

- **Prompt-Oriented Programming** — Natural language prompts as Python code, seamlessly interact with agents
- **Skill Support** — Use skills to convert intent into aibaton code, making complex tasks more controllable
- **Loop Detection** — Built-in iterative optimization (`loop_max`)
- **Response Parsing** — Parse agent results with semantic options using `parse(options)`

## Use Cases

- **Repetitive Tasks** — Batch tasks that reuse the same prompt workflow for agent interaction
- **Context Explosion Tasks** — Tasks with extensive context where agents miss too many details, requiring finer-grained control
- **Automated Workflows** — Build workflows directly on agents, implement arbitrarily complex logic with simple Python syntax

## Not Suitable For

- **Non-Repetitive Tasks** — No need to script for repeated use
- **Complex Tasks Requiring Step-by-Step Confirmation** — Difficult to pre-define all scenarios via script, requires human confirmation at each agent stage

## Example

Install the aibaton SKILL using codex, then enter the following prompt:
```text
$aibaton Find Go code files over 300 lines in the project and optimize them, extract redundant code into sub-functions following DRY principles, and update references in other files.
```

Output executable script:
```python
from aibaton import run, set_default
import os

set_default(provider="codex", dangerous_permissions=True)

prompt = """Review the current file code and optimize, extract redundant code into sub-functions following DRY principles, and update references in other files."""

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
        run(f"File path: {fpath}\n\n{prompt}", cwd=par_dir, add_dirs=[par_dir], stream=True)
```
Run the script as prompted.

### More Complex Task Example
[A complex workflow prompt](examples_en/work_flow_prompt.md)

After codex uses aibaton skill, output executable script: [work_flow.py](examples_en/work_flow.py)

## Installation

```bash
pip install aibaton
# or
uv pip install aibaton
```

## License

Apache 2.0
