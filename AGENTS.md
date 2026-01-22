# AGENTS

- **目标**: 基于 Python 调用 codex/claude CLI, 实现可编程提示词，支持复杂工作流。
- **入口**: `run(prompt)`, `set_default()`, `start_process(cmd)`.
- **返回**: `AgentRes` 含 `text/events/status/usage/provider/elapsed_ms`, 方法: `select(tag)`, `parse(options)`.
- **执行模型**: `run()` 单进程 + selector 轮询 stdout/stderr; `loop_max>1` 自动注入 `<promise>DONE</promise>` 检测循环;
- **Provider**:
  - `codex`: `codex exec --json --cd <cwd> --add-dir <dir> --sandbox workspace-write`, prompt 走 stdin (遇错重试为 argv).
  - `claude`: `claude <prompt> --print --output-format stream-json --add-dir <dir>`, prompt 走 argv.
  - 重试逻辑: 检测 `too many arguments` 切 prompt_as_arg; 检测 `permission denied` 设 `HOME=cwd`.
- **事件流**: `normalize_event(raw, source)` 统一为 `{type, ts, payload, source}`; `extract_text(raw)` 从 `text/message/content/item/delta` 提取文本; 支持 Codex `item.type=agent_message` 和 Anthropic `content_block_delta.delta.text`.
- **进度**: `ProgressPrinter` 用 spinner + 状态栏显示活动细节 (thinking/calling/writing/streaming); `_extract_activity()` 识别 reasoning/tool_call/exec.spawn/file 操作; 流式输出逐行打印, error 走 stderr.
- **归档**: `log_dir=None` 时自动用 `~/.aibaton/workspaces/<sha1[:12]>/sessions/<session_id>/runs/<run_id>/`, 写 `events.jsonl`, `output.txt`, `run.json`; session 含 `status=open/closed`, `resume_count`, `runs[]`.
- **会话**: `get_or_resume_session(cwd)` 按 cwd 恢复 `status=open` 会话; `update_session()` 记录 run 并在检测到 `<promise>DONE</promise>` 时关闭.
- **选项**: `options=[...]` 自动调用子任务分析响应并匹配最佳选项, 缓存到 `AgentRes.option`; `parse(options)` 按需解析.
- **进程**: `start_process(cmd)` 返回 `ProcessHandle` 支持异步 `poll_events()/iter_events()/watch()/kill()`; 独立于 agent runner, 用于通用子进程管理.
- **配置**: `set_default(provider, dangerous_permissions, cwd, add_dirs)` 全局默认参数; `dangerous_permissions=True` 用 `--dangerously-bypass-approvals-and-sandbox`.
- **失败/超时**: `timeout_s` 到期 kill 进程标记 `status=timeout`; `returncode!=0` 标记 `error`; selector 0.1s 超时轮询.