# AIBaton - AI Agent Orchestrator

基于 Python 调用 codex/claude CLI，实现可编程提示词，支持复杂工作流。

## 1. 核心能力

- **可行**：codex 与 claude CLI 支持非交互模式和结构化事件输出（`--json` / `--output-format stream-json`）
- **可控**：一次 `run()` 对应一次 CLI 执行（可通过`loop_max`实现ralph循环）
- **可扩展**：Provider 适配层 + 统一事件协议

## 2. 总体架构

```
┌────────────────────────────────────────────┐
│           runner.py (run)         │
└───────────────────┬────────────────────────┘
                    │
┌───────────────────▼────────────────────────┐
│     Provider Adapter (codex.py/claude.py)   │
└───────────────────┬────────────────────────┘
                    │
┌───────────────────▼────────────────────────┐
│      Process Exec + Event Parser            │
└───────────────────┬────────────────────────┘
                    │
┌───────────────────▼────────────────────────┐
│   Progress/Logger/Session/Storage           │
└─────────────────────────────────────────────┘
```

## 3. 核心接口

### 3.1 `run()`

- **入口**: `run(prompt)` 单次执行
- **返回**: `AgentRes` 含 `text/events/status/usage/provider/elapsed_ms`
- **方法**: `select(tag)` 提取标签内容，`parse(options)` 分析匹配选项

**关键参数**:
- `loop_max` - 循环次数，>1 时自动注入 `<promise>DONE</promise>` 检测
- `provider` - codex | claude
- `dangerous_permissions` - 绕过沙箱审批
- `options` - 自动调用子任务分析响应并匹配最佳选项

### 3.2 `set_default()`

全局默认参数设置：`provider`, `dangerous_permissions`, `cwd`, `add_dirs`

### 3.3 `start_process()`

独立子进程管理，返回 `ProcessHandle` 支持 `poll_events()/iter_events()/watch()/kill()`

## 4. Provider 适配

### 4.1 codex
- 命令: `codex exec --json --cd <cwd> --add-dir <dir> --sandbox workspace-write`
- prompt 走 stdin（遇错重试为 argv）

### 4.2 claude
- 命令: `claude <prompt> --print --output-format stream-json --add-dir <dir>`
- prompt 走 argv

### 4.3 重试逻辑
- 检测 `too many arguments` 切 prompt_as_arg
- 检测 `permission denied` 设 `HOME=cwd`

## 5. 事件与进度

### 5.1 事件归一化
`normalize_event(raw, source)` 统一为 `{type, ts, payload, source}`

### 5.2 文本提取
`extract_text(raw)` 从 `text/message/content/item/delta` 提取文本，支持:
- Codex `item.type=agent_message`
- Anthropic `content_block_delta.delta.text`

### 5.3 进度显示
`ProgressPrinter` 用 spinner + 状态栏显示活动细节:
- thinking/calling/writing/streaming
- 识别 reasoning/tool_call/exec.spawn/file 操作

## 6. 存档与会话

### 6.1 存档路径
`~/.aibaton/workspaces/<sha1[:12]>/sessions/<session_id>/runs/<run_id>/`

### 6.2 存档文件
- `events.jsonl` - 原始事件流
- `output.txt` - 拼接后的文本
- `run.json` - 摘要与元信息

### 6.3 会话恢复
- `get_or_resume_session(cwd)` 按 cwd 恢复 `status=open` 会话
- `update_session()` 记录 run 并在检测到 `<promise>DONE</promise>` 时关闭
