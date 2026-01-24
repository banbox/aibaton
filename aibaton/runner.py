import os
import re
import selectors
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .events import extract_text, normalize_event
from .logger import logger
from .progress import ProgressPrinter
from .storage import make_run_dir, write_events, write_text, write_summary
from .utils import now_ms, safe_json_loads
from .providers.codex import CodexProvider
from .providers.claude import ClaudeProvider
from .session import get_or_resume_session, update_session


def extract_trailing_tag(text: str, tag: str) -> Optional[str]:
    """从 text 右侧查找 <tag>...</tag>，若 </tag> 后无有效文本则返回中间内容，否则返回 None。"""
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    idx = text.rfind(open_tag)
    if idx < 0:
        return None
    end_idx = text.find(close_tag, idx)
    if end_idx < 0:
        return None
    after = text[end_idx + len(close_tag):]
    if re.search(r'[\w\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', after):
        return None
    return text[idx + len(open_tag):end_idx]


def _build_option_prompt(text: str, options: List[str]) -> str:
    """Build a prompt to analyze text and pick the best matching option."""
    if not options:
        return "output: no match"
    items = "\n".join([f"<option>{opt}</option>" for opt in options])
    return f"""\
<rawtext>{text}</rawtext>

The above is raw text. Please select the option that best matches its meaning:
{items}
Return the result as <option>option</option> at the end of the response. dont output if not match option."""


end_loop_tip = "\nOnce confirmed that all requirements are met without error or omission, output <promise>DONE</promise> at the end of your response."


@dataclass
class AgentRes:
    text: str
    events: List[Dict[str, Any]]
    status: str               # success | timeout | error
    usage: Optional[Dict[str, Any]]
    artifacts: Optional[Dict[str, Any]]
    provider: str             # codex | claude
    model: Optional[str]
    elapsed_ms: int
    option: Optional[str] = None  # cached select result

    def __str__(self) -> str:
        return self.text

    def select(self, tag: str = "option") -> str:
        if tag != "option":
            return extract_trailing_tag(self.text, tag) or ''
        if self.option is not None:
            return self.option
        self.option = extract_trailing_tag(self.text, tag) or ''
        return self.option.strip()

    def parse(self, options: List[str]) -> str:
        res = run(_build_option_prompt(self.text, options))
        self.option = res.select()
        return self.option


def _get_provider(name: str):
    if name == "codex":
        return CodexProvider()
    if name == "claude":
        return ClaudeProvider()
    logger.error("unknown provider: %s", name)
    raise ValueError(f"unknown provider: {name}")


_DEFAULT_PROVIDER = "codex"
_DEFAULT_DANGEROUS_PERMISSIONS = False
_DEFAULT_CWD: Optional[str] = None
_DEFAULT_ADD_DIRS: Optional[List[str]] = None


def set_default(
    provider: Optional[str] = None,
    dangerous_permissions: Optional[bool] = None,
    cwd: Optional[str] = None,
    add_dirs: Optional[List[str]] = None,
) -> None:
    """Set default values for run() parameters."""
    global _DEFAULT_PROVIDER, _DEFAULT_DANGEROUS_PERMISSIONS, _DEFAULT_CWD, _DEFAULT_ADD_DIRS
    if provider is not None:
        _get_provider(provider)
        _DEFAULT_PROVIDER = provider
    if dangerous_permissions is not None:
        _DEFAULT_DANGEROUS_PERMISSIONS = bool(dangerous_permissions)
    if cwd is not None:
        _DEFAULT_CWD = cwd
    if add_dirs is not None:
        _DEFAULT_ADD_DIRS = list(add_dirs)


def _should_retry_prompt_arg(events: List[Dict[str, Any]]) -> bool:
    for ev in events:
        if ev.get("type") in ("thread.started", "turn.started", "turn.completed", "item.completed"):
            return False
    for ev in events:
        if ev.get("type") != "error":
            continue
        payload = ev.get("payload") or {}
        msg = payload.get("message") or payload.get("error") or payload.get("text") or ""
        if not isinstance(msg, str):
            continue
        low = msg.lower()
        if (
            "too many arguments" in low
            or "unexpected argument" in low
            or "found argument" in low
            or "usage: codex exec" in low
        ):
            return True
    return False


def _build_env(cwd: Optional[str]) -> Dict[str, str]:
    env = dict(os.environ)
    if cwd:
        if env.get("AIBATON_FORCE_HOME"):
            env["HOME"] = cwd
        else:
            env.setdefault("HOME", cwd)
    return env


def _should_retry_stdin(events: List[Dict[str, Any]]) -> bool:
    for ev in events:
        if ev.get("type") in ("thread.started", "turn.started", "turn.completed", "item.completed"):
            return False
    for ev in events:
        if ev.get("type") != "error":
            continue
        payload = ev.get("payload") or {}
        msg = payload.get("message") or payload.get("error") or payload.get("text") or ""
        if not isinstance(msg, str):
            continue
        low = msg.lower()
        if "reading prompt from stdin" in low or "stdin" in low:
            return True
    return False


def _should_retry_home_fallback(events: List[Dict[str, Any]]) -> bool:
    for ev in events:
        if ev.get("type") != "error":
            continue
        payload = ev.get("payload") or {}
        msg = payload.get("message") or payload.get("error") or payload.get("text") or ""
        if not isinstance(msg, str):
            continue
        low = msg.lower()
        if "permission denied" in low and ("/.codex" in low or "codex" in low):
            return True
    return False


def _run_once(
    prompt: str,
    provider: str,
    model: Optional[str],
    cwd: Optional[str],
    add_dirs: Optional[List[str]],
    timeout_s: Optional[int],
    json_mode: bool,
    stream: bool,
    dangerous_permissions: bool,
    log_dir: Optional[str],
    session_meta: Optional[Dict[str, Any]],
    prompt_as_arg: bool = False,
    env_override: Optional[Dict[str, str]] = None,
) -> AgentRes:
    prov = _get_provider(provider)
    cmd, stdin_data = prov.build_command(
        prompt, json_mode, cwd, add_dirs, dangerous_permissions
    )
    if provider == "codex" and prompt_as_arg:
        cmd = cmd + [prompt]
        stdin_data = None

    start = time.monotonic()
    run_id = f"{now_ms()}_{os.getpid()}"
    run_dir = make_run_dir(log_dir, run_id)
    logger.debug("run_once start: provider=%s run_id=%s cwd=%s", provider, run_id, cwd)

    progress = ProgressPrinter(stream_tokens=stream)
    progress.start(f"{provider} run")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if stdin_data is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=cwd,
        env=env_override,
    )

    if stdin_data is not None and proc.stdin is not None:
        try:
            proc.stdin.write(stdin_data)
            proc.stdin.close()
        except Exception:
            pass

    selector = selectors.DefaultSelector()
    if proc.stdout:
        selector.register(proc.stdout, selectors.EVENT_READ)
    if proc.stderr:
        selector.register(proc.stderr, selectors.EVENT_READ)

    events: List[Dict[str, Any]] = []
    text_parts: List[str] = []
    status = "success"

    while True:
        if timeout_s is not None and (time.monotonic() - start) > timeout_s:
            status = "timeout"
            logger.warning("run timeout after %ds: run_id=%s", timeout_s, run_id)
            try:
                proc.kill()
            except Exception:
                pass
            break

        if not selector.get_map() and proc.poll() is not None:
            break

        for key, _ in selector.select(timeout=0.1):
            line = key.fileobj.readline()
            if not line:
                selector.unregister(key.fileobj)
                continue
            line = line.rstrip("\n")

            if key.fileobj is proc.stdout:
                if json_mode:
                    raw = safe_json_loads(line)
                    if raw is None:
                        ev = normalize_event({"type": "message", "text": line}, provider)
                        events.append(ev)
                        text_parts.append(line + "\n")
                        progress.on_event(ev)
                    else:
                        ev = normalize_event(raw, provider)
                        events.append(ev)
                        txt = extract_text(raw)
                        if txt:
                            text_parts.append(txt)
                        progress.on_event(ev)
                else:
                    ev = normalize_event({"type": "message", "text": line}, provider)
                    events.append(ev)
                    text_parts.append(line + "\n")
                    progress.on_event(ev)
            else:
                ev = normalize_event({"type": "error", "message": line}, provider)
                events.append(ev)
                progress.on_event(ev)

    rc = proc.wait(timeout=1) if proc.poll() is None else proc.returncode
    if status != "timeout" and rc not in (0, None):
        status = "error"
        logger.error("run error: run_id=%s returncode=%s", run_id, rc)

    elapsed_ms = int((time.monotonic() - start) * 1000)
    text = "".join(text_parts).strip()

    if run_dir:
        summary = {
            "provider": provider,
            "model": model,
            "status": status,
            "elapsed_ms": elapsed_ms,
            "cwd": cwd,
            "add_dirs": add_dirs or [],
            "json_mode": json_mode,
            "stream": stream,
            "dangerous_permissions": dangerous_permissions,
            "prompt": prompt,
            "run_id": run_id,
        }
        if session_meta:
            summary["session"] = dict(session_meta)
        write_events(run_dir, events)
        write_text(run_dir, text)
        write_summary(run_dir, summary)

    progress.done(status, elapsed_ms)
    if text and not stream:
        logger.info("response:\n%s", text)
    logger.info("run_once done: run_id=%s status=%s elapsed_ms=%d", run_id, status, elapsed_ms)

    artifacts = {"run_dir": run_dir, "run_id": run_id} if run_dir else None
    return AgentRes(
        text=text,
        events=events,
        status=status,
        usage=None,
        artifacts=artifacts,
        provider=provider,
        model=model,
        elapsed_ms=elapsed_ms,
    )


def run(
    prompt: str,
    loop_max: int = 1,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    cwd: Optional[str] = None,
    add_dirs: Optional[List[str]] = None,
    timeout_s: Optional[int] = None,
    json_mode: bool = True,
    stream: bool = True,
    dangerous_permissions: Optional[bool] = None,
    log_dir: Optional[str] = None,
    options: Optional[List[str]] = None,
) -> AgentRes:
    if provider is None:
        provider = _DEFAULT_PROVIDER
    if dangerous_permissions is None:
        dangerous_permissions = _DEFAULT_DANGEROUS_PERMISSIONS
    if cwd is None:
        cwd = _DEFAULT_CWD
    if add_dirs is None:
        add_dirs = _DEFAULT_ADD_DIRS

    cwd_eff = cwd or os.getcwd()
    session = None
    session_meta = None
    base_log_dir = log_dir
    logger.info("run start: provider=%s cwd=%s loop_max=%d", provider, cwd_eff, loop_max)
    if base_log_dir is None:
        session, resumed = get_or_resume_session(cwd_eff)
        if resumed:
            logger.debug("session resumed: id=%s", session.get("session_id"))
        base_log_dir = session.get("runs_dir")
        session_meta = {
            "id": session.get("session_id"),
            "cwd": session.get("cwd"),
            "resumed": resumed,
            "status": session.get("status"),
        }
    elif base_log_dir:
        os.makedirs(base_log_dir, exist_ok=True)

    last_res: Optional[AgentRes] = None
    env_base = _build_env(cwd_eff)
    real_loop = max(loop_max, 1)
    if real_loop > 1 and "<promise>DONE</promise>" not in prompt:
        prompt += end_loop_tip
    for _ in range(real_loop):
        last_res = _run_once(
            prompt=prompt,
            provider=provider,
            model=model,
            cwd=cwd_eff,
            add_dirs=add_dirs,
            timeout_s=timeout_s,
            json_mode=json_mode,
            stream=stream,
            dangerous_permissions=dangerous_permissions,
            log_dir=base_log_dir,
            session_meta=session_meta,
            env_override=env_base,
        )
        if provider == "codex":
            needs_prompt_retry = _should_retry_prompt_arg(last_res.events) or (
                _should_retry_stdin(last_res.events) and not last_res.text
            )
            if needs_prompt_retry:
                logger.debug("retrying with prompt as arg")
                last_res = _run_once(
                    prompt=prompt,
                    provider=provider,
                    model=model,
                    cwd=cwd_eff,
                    add_dirs=add_dirs,
                    timeout_s=timeout_s,
                    json_mode=json_mode,
                    stream=stream,
                    dangerous_permissions=dangerous_permissions,
                    log_dir=base_log_dir,
                    session_meta=session_meta,
                    prompt_as_arg=True,
                    env_override=env_base,
                )
            needs_home_retry = _should_retry_home_fallback(last_res.events) and (
                last_res.status == "error" or not last_res.text
            )
            if needs_home_retry:
                logger.debug("retrying with HOME fallback to cwd")
                env_fallback = dict(env_base)
                env_fallback["HOME"] = cwd_eff
                last_res = _run_once(
                    prompt=prompt,
                    provider=provider,
                    model=model,
                    cwd=cwd_eff,
                    add_dirs=add_dirs,
                    timeout_s=timeout_s,
                    json_mode=json_mode,
                    stream=stream,
                    dangerous_permissions=dangerous_permissions,
                    log_dir=base_log_dir,
                    session_meta=session_meta,
                    prompt_as_arg=True,
                    env_override=env_fallback,
                )
        done_flag = extract_trailing_tag(last_res.text, "promise") == "DONE"
        if session and last_res.artifacts and last_res.artifacts.get("run_id"):
            update_session(cwd_eff, session, last_res.artifacts["run_id"], last_res.status, done_flag)
            if session_meta is not None:
                session_meta["status"] = session.get("status")
        if loop_max > 1 and done_flag:
            logger.debug("loop exit: DONE flag detected")
            break

    assert last_res is not None
    logger.info("run complete: status=%s elapsed_ms=%d", last_res.status, last_res.elapsed_ms)

    # If options provided, run analysis to pick best option and cache in option field
    if options and last_res.text:
        analysis_res = _run_once(
            prompt=_build_option_prompt(last_res.text, options),
            provider=provider,
            model=model,
            cwd=cwd_eff,
            add_dirs=add_dirs,
            timeout_s=timeout_s,
            json_mode=json_mode,
            stream=stream,
            dangerous_permissions=dangerous_permissions,
            log_dir=base_log_dir,
            session_meta=session_meta,
            env_override=env_base,
        )
        last_res.option = analysis_res.select()

    return last_res
