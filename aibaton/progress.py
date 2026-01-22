import sys
import time
import threading
from typing import Dict, Any, Optional, List


def _extract_activity(raw: Dict[str, Any]) -> Optional[str]:
    """Extract human-readable activity description from event payload."""
    etype = raw.get("type") or raw.get("event") or ""
    
    # Codex reasoning/thinking
    if etype == "reasoning":
        return "thinking..."
    if etype in ("thinking", "thought"):
        return "thinking..."
    
    # Tool/function calls
    if etype in ("tool_call", "function_call", "tool_use"):
        name = raw.get("name") or raw.get("tool") or ""
        if name:
            return f"calling {name}..."
        return "calling tool..."
    
    # Codex exec events
    if etype == "exec.spawn":
        cmd = raw.get("command") or raw.get("cmd") or ""
        if isinstance(cmd, list):
            cmd = " ".join(cmd[:3])
        if cmd:
            short = cmd[:40] + "..." if len(cmd) > 40 else cmd
            return f"running: {short}"
        return "running command..."
    if etype == "exec.output":
        return "command output..."
    
    # File operations
    if etype in ("file.write", "file.create", "write_file"):
        path = raw.get("path") or raw.get("file") or ""
        if path:
            fname = path.rsplit("/", 1)[-1]
            return f"writing {fname}..."
        return "writing file..."
    if etype in ("file.read", "read_file"):
        path = raw.get("path") or raw.get("file") or ""
        if path:
            fname = path.rsplit("/", 1)[-1]
            return f"reading {fname}..."
        return "reading file..."
    if etype in ("patch.apply", "edit", "apply_patch"):
        path = raw.get("path") or raw.get("file") or ""
        if path:
            fname = path.rsplit("/", 1)[-1]
            return f"editing {fname}..."
        return "editing file..."
    
    # Item events (codex)
    item = raw.get("item")
    if isinstance(item, dict):
        item_type = item.get("type") or ""
        if item_type == "reasoning":
            return "thinking..."
        if item_type in ("tool_call", "function_call"):
            name = item.get("name") or ""
            if name:
                return f"calling {name}..."
            return "calling tool..."
        if item_type in ("agent_message", "assistant_message", "message"):
            return "responding..."
    
    # Turn/thread events
    if etype == "thread.started":
        return "started..."
    if etype == "turn.started":
        return "processing..."
    if etype == "turn.completed":
        return "turn done"
    if etype == "item.started":
        return "working..."
    if etype == "item.completed":
        return "item done"
    
    # Claude specific
    if etype == "content_block_start":
        cb = raw.get("content_block", {})
        if cb.get("type") == "tool_use":
            name = cb.get("name") or ""
            if name:
                return f"calling {name}..."
            return "calling tool..."
        return "generating..."
    if etype == "content_block_delta":
        return "streaming..."
    if etype == "message_start":
        return "responding..."
    if etype == "message_delta":
        stop = raw.get("delta", {}).get("stop_reason")
        if stop:
            return f"stopped: {stop}"
    
    return None


class ProgressPrinter:
    """
    Terminal progress display with refreshable status bar at bottom.
    
    - Status bar (spinner + status + activity) refreshes in-place at terminal bottom
    - Agent conversation messages are printed above, appended line by line
    - Activity details extracted from events show what the agent is doing
    """
    
    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    SPINNER_INTERVAL = 0.1
    
    def __init__(self, stream_tokens: bool = True):
        self.stream_tokens = stream_tokens
        self._label: str = ""
        self._status: str = ""
        self._activity: str = ""  # detailed activity info
        self._spinner_idx: int = 0
        self._start_time: float = 0.0
        self._token_count: int = 0
        self._event_count: int = 0
        self._running: bool = False
        self._spinner_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._is_tty = sys.stderr.isatty()
        self._current_line: str = ""  # current line content before newline
        self._has_content: bool = False  # whether any content was printed
        self._recent_activities: List[str] = []  # track recent activities
    
    def _format_elapsed(self) -> str:
        elapsed = time.monotonic() - self._start_time
        if elapsed < 60:
            return f"{elapsed:.1f}s"
        minutes = int(elapsed // 60)
        secs = elapsed % 60
        return f"{minutes}m{secs:.0f}s"
    
    def _format_status_line(self) -> str:
        spinner = self.SPINNER_FRAMES[self._spinner_idx % len(self.SPINNER_FRAMES)]
        elapsed = self._format_elapsed()
        parts = [f"{spinner} {self._label}"]
        # Show detailed activity if available, otherwise status
        if self._activity:
            parts.append(self._activity)
        elif self._status:
            parts.append(self._status)
        parts.append(f"[{elapsed}]")
        if self._token_count > 0:
            parts.append(f"{self._token_count} tok")
        if self._event_count > 0:
            parts.append(f"{self._event_count} ev")
        return " ".join(parts)
    
    def _write_status_line(self) -> None:
        """Write status on a new line below content, can be overwritten."""
        if not self._is_tty:
            return
        line = self._format_status_line()
        max_width = 80
        if len(line) > max_width:
            line = line[:max_width - 3] + "..."
        # Save cursor, go to new line, write status, restore cursor
        # Using simpler approach: just overwrite current line position
        sys.stderr.write(f"\r\033[K{line}")
        sys.stderr.flush()
    
    def _spinner_loop(self) -> None:
        while self._running:
            with self._lock:
                if self._running:
                    self._spinner_idx += 1
                    self._write_status_line()
            time.sleep(self.SPINNER_INTERVAL)
    
    def _flush_line(self, text: str, is_error: bool = False) -> None:
        """Print a complete line of text, preserving it above status bar."""
        with self._lock:
            if self._is_tty:
                # Clear current status, print text line, then redraw status
                sys.stderr.write("\r\033[K")
                sys.stderr.write(text)
                if not text.endswith("\n"):
                    sys.stderr.write("\n")
                sys.stderr.flush()
                self._has_content = True
                if self._running:
                    self._write_status_line()
            else:
                stream = sys.stderr if is_error else sys.stdout
                stream.write(text)
                if not text.endswith("\n"):
                    stream.write("\n")
                stream.flush()
    
    def _update_streaming(self, text: str) -> None:
        """Handle streaming text - accumulate until newline."""
        self._current_line += text
        self._token_count += len(text.split())
        
        # Check for complete lines
        while "\n" in self._current_line:
            line, self._current_line = self._current_line.split("\n", 1)
            if line:
                self._flush_line(line)
    
    def start(self, label: str) -> None:
        self._label = label
        self._status = "starting..."
        self._activity = ""
        self._start_time = time.monotonic()
        self._token_count = 0
        self._event_count = 0
        self._spinner_idx = 0
        self._running = True
        self._current_line = ""
        self._has_content = False
        self._recent_activities = []
        
        if self._is_tty:
            self._write_status_line()
            self._spinner_thread = threading.Thread(target=self._spinner_loop, daemon=True)
            self._spinner_thread.start()
        else:
            print(f"[agent] {label}", file=sys.stderr)
    
    def set_status(self, status: str) -> None:
        """Update the status message in the status bar."""
        with self._lock:
            self._status = status
            if self._is_tty and self._running:
                self._write_status_line()
    
    def set_activity(self, activity: str) -> None:
        """Update the activity detail in the status bar."""
        with self._lock:
            self._activity = activity
            # Keep track of recent activities
            if activity and (not self._recent_activities or self._recent_activities[-1] != activity):
                self._recent_activities.append(activity)
                if len(self._recent_activities) > 10:
                    self._recent_activities.pop(0)
            if self._is_tty and self._running:
                self._write_status_line()
    
    def on_event(self, event: Dict[str, Any]) -> None:
        self._event_count += 1
        etype = event.get("type")
        payload = event.get("payload", {})
        
        # Extract detailed activity from payload
        if isinstance(payload, dict):
            activity = _extract_activity(payload)
            if activity:
                self.set_activity(activity)
        
        if etype == "error":
            msg = payload.get("message") or payload.get("error") or str(payload)
            self._flush_line(f"[error] {msg}", is_error=True)
            self.set_status("error")
            self.set_activity("")
        elif self.stream_tokens and isinstance(payload, dict):
            text = payload.get("text")
            if isinstance(text, str) and text:
                self._update_streaming(text)
                if not self._activity:  # don't override detailed activity
                    self.set_status("streaming...")
        
        # Update status based on event type (fallback if no activity)
        if not self._activity:
            if etype in ("thread.started", "turn.started"):
                self.set_status("processing...")
            elif etype == "turn.completed":
                self.set_status("turn completed")
            elif etype == "item.completed":
                self.set_status("item completed")
    
    def done(self, status: str, elapsed_ms: int) -> None:
        self._running = False
        if self._spinner_thread:
            self._spinner_thread.join(timeout=0.2)
            self._spinner_thread = None
        
        # Flush any remaining content
        if self._current_line:
            self._flush_line(self._current_line)
            self._current_line = ""
        
        # Clear status line
        if self._is_tty:
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()
        
        # Print final status
        icon = "✓" if status == "success" else "✗" if status == "error" else "◷"
        elapsed_s = elapsed_ms / 1000
        final_msg = f"{icon} {self._label} {status} ({elapsed_s:.1f}s)"
        if self._has_content or not self._is_tty:
            print(final_msg, file=sys.stderr)
        else:
            print(f"\n{final_msg}", file=sys.stderr)
