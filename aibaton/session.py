import hashlib
import json
import os
from threading import Lock
from typing import Any, Dict, Optional, Tuple

from .logger import logger
from .utils import ensure_dir, now_ms

_session_lock = Lock()


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _agent_root() -> str:
    home = os.path.expanduser("~") or os.getcwd()
    root = os.path.join(home, ".aibaton")
    ensure_dir(root)
    return root


def _workspace_id(cwd: str) -> str:
    digest = hashlib.sha1(cwd.encode("utf-8")).hexdigest()
    return digest[:12]


def _workspace_dir(cwd: str) -> str:
    ws_dir = os.path.join(_agent_root(), "workspaces", _workspace_id(cwd))
    ensure_dir(ws_dir)
    return ws_dir


def _session_public(session: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(session)
    data.pop("runs_dir", None)
    return data


def get_or_resume_session(cwd: str) -> Tuple[Dict[str, Any], bool]:
    cwd_abs = os.path.abspath(cwd)
    ws_dir = _workspace_dir(cwd_abs)

    workspace_meta = os.path.join(ws_dir, "workspace.json")
    if not os.path.exists(workspace_meta):
        _write_json(
            workspace_meta,
            {"cwd": cwd_abs, "created_at": now_ms(), "updated_at": now_ms()},
        )

    current_path = os.path.join(ws_dir, "session.json")
    with _session_lock:
        session = _read_json(current_path)
        resumed = False

        if session and session.get("status") == "open":
            resumed = True
            session["resume_count"] = int(session.get("resume_count", 0)) + 1
            session["updated_at"] = now_ms()
            logger.debug("session resumed: id=%s resume_count=%d", session.get("session_id"), session["resume_count"])
        else:
            session = {
                "session_id": f"{now_ms()}_{os.getpid()}",
                "status": "open",
                "cwd": cwd_abs,
                "created_at": now_ms(),
                "updated_at": now_ms(),
                "resume_count": 0,
                "runs": [],
            }
            logger.debug("session created: id=%s cwd=%s", session["session_id"], cwd_abs)

        session_dir = os.path.join(ws_dir, "sessions", session["session_id"])
        runs_dir = os.path.join(session_dir, "runs")
        ensure_dir(runs_dir)
        session["runs_dir"] = runs_dir

        _write_json(current_path, _session_public(session))
        _write_json(os.path.join(session_dir, "session.json"), _session_public(session))

    return session, resumed


def update_session(
    cwd: str,
    session: Dict[str, Any],
    run_id: str,
    status: str,
    done: bool,
) -> None:
    cwd_abs = os.path.abspath(cwd)
    ws_dir = _workspace_dir(cwd_abs)
    current_path = os.path.join(ws_dir, "session.json")

    with _session_lock:
        session["last_run_id"] = run_id
        session["updated_at"] = now_ms()
        session.setdefault("runs", []).append(
            {"run_id": run_id, "status": status, "ts": now_ms()}
        )
        if done:
            session["status"] = "closed"
            logger.debug("session closed: id=%s", session.get("session_id"))

        session_dir = os.path.join(ws_dir, "sessions", session["session_id"])
        _write_json(current_path, _session_public(session))
        _write_json(os.path.join(session_dir, "session.json"), _session_public(session))
