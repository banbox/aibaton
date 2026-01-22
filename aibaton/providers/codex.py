from typing import List, Optional, Tuple

from ..logger import logger


class CodexProvider:
    name = "codex"

    def build_command(
        self,
        prompt: str,
        json_mode: bool,
        cwd: Optional[str],
        add_dirs: Optional[List[str]],
        dangerous_permissions: bool,
    ) -> Tuple[List[str], Optional[str]]:
        cmd = ["codex", "exec", "--skip-git-repo-check"]
        if json_mode:
            cmd.append("--json")
        if cwd:
            cmd.extend(["--cd", cwd])
        if add_dirs:
            for d in add_dirs:
                cmd.extend(["--add-dir", d])
        if dangerous_permissions:
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            cmd.extend(["--sandbox", "workspace-write"])

        # Codex CLI typically reads prompt from stdin for exec
        logger.debug("codex command: %s", " ".join(cmd))
        return cmd, prompt
