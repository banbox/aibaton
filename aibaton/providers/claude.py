from typing import List, Optional, Tuple

from ..logger import logger


class ClaudeProvider:
    name = "claude"

    def build_command(
        self,
        prompt: str,
        json_mode: bool,
        cwd: Optional[str],
        add_dirs: Optional[List[str]],
        dangerous_permissions: bool,
    ) -> Tuple[List[str], Optional[str]]:
        cmd = ["claude", prompt, "--print"]
        if json_mode:
            cmd.extend(["--output-format", "stream-json"])
        if add_dirs:
            for d in add_dirs:
                cmd.extend(["--add-dir", d])
        if dangerous_permissions:
            cmd.append("--dangerously-skip-permissions")

        # Claude prompt passed as argument
        logger.debug("claude command: %s", " ".join(cmd[:3]) + " ...")
        return cmd, None
