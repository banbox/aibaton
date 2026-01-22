import json
import os
from typing import Any, Dict, List, Optional

from .logger import logger
from .utils import ensure_dir


def make_run_dir(base_dir: Optional[str], run_id: str) -> Optional[str]:
    if not base_dir:
        return None
    run_dir = os.path.join(base_dir, run_id)
    ensure_dir(run_dir)
    logger.debug("run_dir created: %s", run_dir)
    return run_dir


def write_events(run_dir: Optional[str], events: List[Dict[str, Any]]) -> None:
    if not run_dir:
        return
    path = os.path.join(run_dir, "events.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    logger.debug("events written: %s count=%d", path, len(events))


def write_text(run_dir: Optional[str], text: str) -> None:
    if not run_dir:
        return
    path = os.path.join(run_dir, "output.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.debug("output written: %s len=%d", path, len(text))


def write_summary(run_dir: Optional[str], summary: Dict[str, Any]) -> None:
    if not run_dir:
        return
    path = os.path.join(run_dir, "run.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.debug("summary written: %s", path)
