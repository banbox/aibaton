import codecs
import locale
import os
import queue
import selectors
import subprocess
import threading
import time
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from .events import normalize_event
from .logger import logger


_QUEUE_DONE = object()


@dataclass
class ProcessResult:
    cmd: Union[str, List[str]]
    status: str  # success | timeout | killed | error
    returncode: Optional[int]
    output: str
    stdout: str
    stderr: str
    events: List[Dict[str, Any]]
    elapsed_ms: int
    pid: Optional[int]


class ProcessHandle:
    def __init__(
        self,
        cmd: Union[str, Sequence[str]],
        timeout_s: Optional[float],
        cwd: Optional[str],
        env: Optional[Dict[str, str]],
        shell: Optional[bool],
        encoding: Optional[str],
        errors: str,
    ) -> None:
        if isinstance(cmd, str):
            self._cmd = cmd
        else:
            self._cmd = list(cmd)
        if shell is None:
            shell = isinstance(self._cmd, str)
        popen_cmd = self._cmd
        if shell and isinstance(popen_cmd, list):
            popen_cmd = " ".join(popen_cmd)

        self._start = time.monotonic()
        self._timeout_s = timeout_s
        self._events: List[Dict[str, Any]] = []
        self._stdout_parts: List[str] = []
        self._stderr_parts: List[str] = []
        self._merged_parts: List[str] = []
        self._text_buffers: Dict[str, str] = {"stdout": "", "stderr": ""}
        self._status = "success"
        self._returncode: Optional[int] = None
        self._elapsed_ms = 0
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._queue: "queue.Queue[object]" = queue.Queue()

        enc = encoding or locale.getpreferredencoding(False)
        self._decoders = {
            "stdout": codecs.getincrementaldecoder(enc)(errors=errors),
            "stderr": codecs.getincrementaldecoder(enc)(errors=errors),
        }

        self._proc = subprocess.Popen(
            popen_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            cwd=cwd,
            env=env,
            shell=shell,
        )
        logger.debug("process started: pid=%s cmd=%s", self._proc.pid, popen_cmd)

        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def returncode(self) -> Optional[int]:
        with self._lock:
            return self._returncode

    def is_running(self) -> bool:
        return self._proc.poll() is None and not self._done.is_set()

    def kill(self) -> None:
        self._set_status("killed")
        logger.debug("process killed: pid=%s", self._proc.pid)
        try:
            self._proc.kill()
        except Exception:
            pass

    def terminate(self) -> None:
        self._set_status("killed")
        logger.debug("process terminated: pid=%s", self._proc.pid)
        try:
            self._proc.terminate()
        except Exception:
            pass

    def wait(self, timeout: Optional[float] = None) -> Optional[ProcessResult]:
        if timeout is not None and not self._done.wait(timeout):
            return None
        self._done.wait()
        return self.result()

    def result(self) -> ProcessResult:
        self._done.wait()
        with self._lock:
            return ProcessResult(
                cmd=self._cmd if isinstance(self._cmd, list) else str(self._cmd),
                status=self._status,
                returncode=self._returncode,
                output="".join(self._merged_parts),
                stdout="".join(self._stdout_parts),
                stderr="".join(self._stderr_parts),
                events=list(self._events),
                elapsed_ms=self._elapsed_ms,
                pid=self._proc.pid,
            )

    def poll_events(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        while True:
            try:
                ev = self._queue.get_nowait()
            except queue.Empty:
                break
            if ev is _QUEUE_DONE:
                self._queue.put(_QUEUE_DONE)
                break
            if isinstance(ev, dict):
                items.append(ev)
        return items

    def iter_events(self, timeout: Optional[float] = None) -> Iterator[Dict[str, Any]]:
        while True:
            try:
                ev = self._queue.get(timeout=timeout)
            except queue.Empty:
                if self._done.is_set():
                    break
                continue
            if ev is _QUEUE_DONE:
                self._queue.put(_QUEUE_DONE)
                break
            if isinstance(ev, dict):
                yield ev

    def watch(self, stream: bool = True) -> ProcessResult:
        for ev in self.iter_events():
            if stream:
                _print_event(ev)
        return self.result()

    def _set_status(self, status: str) -> None:
        with self._lock:
            if self._status in ("timeout", "killed"):
                return
            self._status = status

    def _record(self, stream: str, line: str, raw_line: str) -> None:
        if stream == "stdout":
            raw = {"type": "message", "text": line, "stream": "stdout", "pid": self._proc.pid}
            ev = normalize_event(raw, "process")
            with self._lock:
                self._events.append(ev)
                self._stdout_parts.append(raw_line)
                self._merged_parts.append(raw_line)
            self._queue.put(ev)
        else:
            raw = {"type": "error", "message": line, "stream": "stderr", "pid": self._proc.pid}
            ev = normalize_event(raw, "process")
            with self._lock:
                self._events.append(ev)
                self._stderr_parts.append(raw_line)
                self._merged_parts.append(raw_line)
            self._queue.put(ev)

    def _process_text(self, stream: str, text: str) -> None:
        if not text:
            return
        buf = self._text_buffers[stream] + text
        while True:
            idx = buf.find("\n")
            if idx < 0:
                break
            line = buf[:idx]
            self._record(stream, line, line + "\n")
            buf = buf[idx + 1:]
        self._text_buffers[stream] = buf

    def _flush_buffers(self) -> None:
        for stream in ("stdout", "stderr"):
            extra = self._decoders[stream].decode(b"", final=True)
            if extra:
                self._process_text(stream, extra)
            remaining = self._text_buffers[stream]
            if remaining:
                self._record(stream, remaining, remaining)
                self._text_buffers[stream] = ""

    def _reader_loop(self) -> None:
        selector = selectors.DefaultSelector()
        if self._proc.stdout:
            selector.register(self._proc.stdout, selectors.EVENT_READ)
        if self._proc.stderr:
            selector.register(self._proc.stderr, selectors.EVENT_READ)

        while True:
            if self._timeout_s is not None and (time.monotonic() - self._start) > self._timeout_s:
                self._set_status("timeout")
                logger.warning("process timeout: pid=%s timeout_s=%s", self._proc.pid, self._timeout_s)
                self.kill()

            if not selector.get_map() and self._proc.poll() is not None:
                break

            for key, _ in selector.select(timeout=0.1):
                try:
                    data = os.read(key.fileobj.fileno(), 4096)
                except BlockingIOError:
                    continue
                if not data:
                    selector.unregister(key.fileobj)
                    continue
                stream = "stdout" if key.fileobj is self._proc.stdout else "stderr"
                text = self._decoders[stream].decode(data)
                self._process_text(stream, text)

        self._flush_buffers()
        rc = self._proc.wait(timeout=1) if self._proc.poll() is None else self._proc.returncode
        elapsed_ms = int((time.monotonic() - self._start) * 1000)
        with self._lock:
            self._returncode = rc
            self._elapsed_ms = elapsed_ms
            if self._status not in ("timeout", "killed"):
                if rc not in (0, None):
                    self._status = "error"
                    logger.error("process error: pid=%s returncode=%s", self._proc.pid, rc)
                else:
                    self._status = "success"
        logger.debug("process done: pid=%s status=%s elapsed_ms=%d", self._proc.pid, self._status, elapsed_ms)

        self._done.set()
        self._queue.put(_QUEUE_DONE)


def _print_event(ev: Dict[str, Any]) -> None:
    payload = ev.get("payload", {}) if isinstance(ev, dict) else {}
    text = payload.get("text") or payload.get("message") or ""
    if ev.get("type") == "error" or payload.get("stream") == "stderr":
        print(text, file=sys.stderr, flush=True)
    else:
        print(text, file=sys.stdout, flush=True)


def start_process(
    cmd: Union[str, Sequence[str]],
    *,
    timeout_s: Optional[float] = None,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    shell: Optional[bool] = None,
    encoding: Optional[str] = None,
    errors: str = "replace",
) -> ProcessHandle:
    return ProcessHandle(
        cmd=cmd,
        timeout_s=timeout_s,
        cwd=cwd,
        env=env,
        shell=shell,
        encoding=encoding,
        errors=errors,
    )
