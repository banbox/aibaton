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
        self._item_line_open: bool = False  # whether an item line is waiting for completion
    
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
        # Don't overwrite when an item line is open waiting for completion
        if self._item_line_open:
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
        self._item_line_open = False
        
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
    
    def _get_item_summary(self, item: Dict[str, Any], max_len: int = 60) -> str:
        """Extract detailed summary from item payload."""
        item_type = item.get("type", "")
        
        def truncate(s: str, limit: int = 40) -> str:
            return (s[:limit] + "...") if len(s) > limit else s
        
        # Tool call / function call - show name and key arguments
        if item_type in ("tool_call", "function_call", "tool_use"):
            name = item.get("name") or item.get("tool") or "tool"
            args = item.get("arguments") or item.get("input") or item.get("args") or {}
            detail = ""
            if isinstance(args, dict):
                # Extract key info: path, command, query, etc.
                for key in ("path", "file", "command", "cmd", "query", "url", "pattern"):
                    if key in args:
                        val = args[key]
                        if isinstance(val, list):
                            val = " ".join(str(v) for v in val[:3])
                        detail = truncate(str(val), 35)
                        break
            if detail:
                return truncate(f"{name}({detail})", max_len)
            return name
        
        # Reasoning - try to extract summary or text
        if item_type in ("reasoning", "thinking"):
            summary = item.get("summary") or ""
            if summary:
                return f"thinking: {truncate(summary, 50)}"
            # Codex reasoning has text field
            text = item.get("text") or ""
            if text:
                # Extract first sentence or line
                first = text.split("\n")[0].split(".")[0].strip()
                if first.startswith("**") and "**" in first[2:]:
                    # Remove markdown bold
                    first = first.replace("**", "")
                return f"thinking: {truncate(first, 50)}"
            return "reasoning"
        
        # Message - show brief content preview
        if item_type in ("agent_message", "assistant_message", "message"):
            text = item.get("text") or ""
            if not text:
                content = item.get("content")
                if isinstance(content, list) and content:
                    first = content[0]
                    if isinstance(first, dict):
                        text = first.get("text", "")
                    elif isinstance(first, str):
                        text = first
            if text:
                # First line only, truncated
                first_line = text.split("\n")[0].strip()
                return f"message: {truncate(first_line, 50)}"
            return "message"
        
        # Exec - show command
        if item_type == "exec":
            cmd = item.get("command") or item.get("cmd") or ""
            if isinstance(cmd, list):
                cmd = " ".join(str(c) for c in cmd[:4])
            return f"exec: {truncate(str(cmd), 50)}" if cmd else "exec"
        
        # File operations
        if item_type in ("file_read", "read_file"):
            path = item.get("path") or item.get("file") or ""
            fname = path.rsplit("/", 1)[-1] if path else ""
            return f"read: {fname}" if fname else "read_file"
        if item_type in ("file_write", "write_file", "patch", "apply_patch"):
            path = item.get("path") or item.get("file") or ""
            fname = path.rsplit("/", 1)[-1] if path else ""
            return f"write: {fname}" if fname else "write_file"
        
        # Shell/command execution (including codex command_execution)
        if item_type in ("shell", "bash", "command", "command_execution", "local_shell_call"):
            # Command is directly at item.command for codex
            cmd = item.get("command") or item.get("cmd") or item.get("input") or ""
            if isinstance(cmd, list):
                cmd = " ".join(str(c) for c in cmd[:3])
            if cmd:
                # Extract the actual command from bash -lc "..." wrapper
                if "-lc" in cmd and '"' in cmd:
                    # Extract content after -lc "
                    start = cmd.find('-lc')
                    if start >= 0:
                        rest = cmd[start + 3:].strip().strip('"').strip("'")
                        cmd = rest
                return f"$ {truncate(str(cmd), 55)}"
            return item_type
        
        # Fallback: try to extract any useful info from the item
        # Search for command/path/file in any nested dict
        def find_detail(d: Dict[str, Any], depth: int = 0) -> str:
            if depth > 2:
                return ""
            for key in ("command", "cmd", "cmdline", "path", "file", "name", "input"):
                val = d.get(key)
                if isinstance(val, str) and val:
                    return truncate(val, 45)
                if isinstance(val, list) and val:
                    return truncate(" ".join(str(v) for v in val[:3]), 45)
            for v in d.values():
                if isinstance(v, dict):
                    result = find_detail(v, depth + 1)
                    if result:
                        return result
            return ""
        
        detail = find_detail(item)
        if detail:
            return f"{item_type}: {detail}" if item_type else detail
        
        return item_type or "item"

    def _log_item_started(self, payload: Dict[str, Any]) -> None:
        """Print item.started as new line prefix (without newline), waiting for completion."""
        elapsed = self._format_elapsed()
        item = payload.get("item", {})
        summary = self._get_item_summary(item) if isinstance(item, dict) else "item"
        
        with self._lock:
            # Close previous item line if still open
            if self._item_line_open:
                sys.stderr.write("\n")
                self._item_line_open = False
            
            if self._is_tty:
                sys.stderr.write(f"\r\033[K[{elapsed}] ▶ {summary} ")
            else:
                sys.stderr.write(f"[{elapsed}] ▶ {summary} ")
            sys.stderr.flush()
            self._item_line_open = True
            self._has_content = True

    def _log_item_completed(self, payload: Dict[str, Any]) -> None:
        """Complete the current item line with result."""
        elapsed = self._format_elapsed()
        item = payload.get("item", {})
        summary = self._get_item_summary(item) if isinstance(item, dict) else "item"
        
        with self._lock:
            if self._item_line_open:
                # Append completion to current line
                sys.stderr.write(f"✓ ({elapsed})\n")
                self._item_line_open = False
            else:
                # No open line, print full line
                if self._is_tty:
                    sys.stderr.write(f"\r\033[K[{elapsed}] ✓ {summary}\n")
                else:
                    sys.stderr.write(f"[{elapsed}] ✓ {summary}\n")
            sys.stderr.flush()

    def _log_turn_event(self, etype: str) -> None:
        """Log turn start/complete events."""
        elapsed = self._format_elapsed()
        with self._lock:
            # Close any open item line first
            if self._item_line_open:
                sys.stderr.write("\n")
                self._item_line_open = False
            
            if etype == "turn.started":
                marker = "──── turn started ────"
            else:
                marker = "──── turn completed ────"
            
            if self._is_tty:
                sys.stderr.write(f"\r\033[K[{elapsed}] {marker}\n")
            else:
                sys.stderr.write(f"[{elapsed}] {marker}\n")
            sys.stderr.flush()
            self._has_content = True

    def _log_claude_tool_use(self, payload: Dict[str, Any]) -> None:
        """Handle Claude assistant events with tool_use in message.content."""
        message = payload.get("message", {})
        content = message.get("content", [])
        if not isinstance(content, list):
            return
        
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "tool_use":
                continue
            
            elapsed = self._format_elapsed()
            name = item.get("name", "tool")
            inp = item.get("input", {})
            
            # Extract detail based on tool type
            detail = ""
            if isinstance(inp, dict):
                # Common fields to look for
                for key in ("command", "pattern", "file_path", "path", "glob", "description"):
                    val = inp.get(key)
                    if val:
                        detail = str(val)[:45]
                        if len(str(val)) > 45:
                            detail += "..."
                        break
            
            with self._lock:
                if self._item_line_open:
                    sys.stderr.write("\n")
                    self._item_line_open = False
                
                if detail:
                    line = f"[{elapsed}] ▶ {name}: {detail}"
                else:
                    line = f"[{elapsed}] ▶ {name}"
                
                if self._is_tty:
                    sys.stderr.write(f"\r\033[K{line} ")
                else:
                    sys.stderr.write(f"{line} ")
                sys.stderr.flush()
                self._item_line_open = True
                self._has_content = True

    def _log_claude_tool_result(self, payload: Dict[str, Any]) -> None:
        """Handle Claude user events with tool_result."""
        message = payload.get("message", {})
        content = message.get("content", [])
        if not isinstance(content, list):
            return
        
        has_result = any(
            isinstance(item, dict) and item.get("type") == "tool_result"
            for item in content
        )
        if not has_result:
            return
        
        elapsed = self._format_elapsed()
        # Check if error
        is_error = any(
            isinstance(item, dict) and item.get("is_error")
            for item in content
        )
        
        with self._lock:
            if self._item_line_open:
                mark = "✗" if is_error else "✓"
                sys.stderr.write(f"{mark} ({elapsed})\n")
                self._item_line_open = False
            sys.stderr.flush()

    def _log_event_brief(self, etype: str, payload: Dict[str, Any], activity: Optional[str]) -> None:
        """Log important events briefly (non-TTY fallback for other events)."""
        # Skip noisy/unimportant events
        skip_types = {"message", "content_block_delta", "message_delta"}
        if etype in skip_types:
            return
        
        # item/turn events and claude events are handled separately
        if etype in ("item.started", "item.completed", "turn.started", "turn.completed", "assistant", "user"):
            return
        
        elapsed = self._format_elapsed()
        
        # Use activity if available, otherwise format based on event type
        if activity:
            print(f"[{elapsed}] {activity}", file=sys.stderr)
            return
        
        # Format based on event type for important state transitions
        if etype in ("exec.spawn", "tool_call", "function_call"):
            name = payload.get("name") or payload.get("command") or payload.get("tool") or ""
            if isinstance(name, list):
                name = " ".join(name[:2])
            short = (name[:30] + "...") if len(str(name)) > 30 else name
            print(f"[{elapsed}] {etype}: {short}", file=sys.stderr)
        elif etype.startswith("file.") or etype in ("read_file", "write_file", "apply_patch"):
            path = payload.get("path") or payload.get("file") or ""
            fname = path.rsplit("/", 1)[-1] if path else ""
            print(f"[{elapsed}] {etype}: {fname}", file=sys.stderr)
    
    def on_event(self, event: Dict[str, Any]) -> None:
        self._event_count += 1
        etype = event.get("type")
        payload = event.get("payload", {})
        
        # Extract detailed activity from payload
        activity = None
        if isinstance(payload, dict):
            activity = _extract_activity(payload)
            if activity:
                self.set_activity(activity)
        
        # Log item/turn events as persistent lines (both TTY and non-TTY)
        # Codex events
        if etype == "item.started" and isinstance(payload, dict):
            self._log_item_started(payload)
        elif etype == "item.completed" and isinstance(payload, dict):
            self._log_item_completed(payload)
        elif etype in ("turn.started", "turn.completed"):
            self._log_turn_event(etype)
        # Claude events - assistant with tool_use, user with tool_result
        elif etype == "assistant" and isinstance(payload, dict):
            self._log_claude_tool_use(payload)
        elif etype == "user" and isinstance(payload, dict):
            self._log_claude_tool_result(payload)
        elif not self._is_tty and etype:
            # Log other important events in non-TTY mode only
            self._log_event_brief(etype, payload, activity)
        
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
        
        # Close any open item line
        with self._lock:
            if self._item_line_open:
                sys.stderr.write("\n")
                self._item_line_open = False
        
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
