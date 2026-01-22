import json
import os
import time
from typing import Any, Dict, Optional


def now_ms() -> int:
    return int(time.time() * 1000)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(line)
    except Exception:
        return None
