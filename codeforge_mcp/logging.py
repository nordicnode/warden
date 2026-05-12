"""Structured JSON logging to stderr for production observability.

Logs tool calls, subagent spawns, errors, and timing information.
"""

from __future__ import annotations

import json
import re
import sys
import time
from typing import Any


# Patterns that match secrets in command strings.  Each (regex, replacement)
# pair redacts the secret value while preserving the key/header name so the
# log entry remains useful for debugging.
# All patterns are applied with re.IGNORECASE.
_SECRET_PATTERNS: list[tuple[str, str]] = [
    # curl -H 'Authorization: Bearer <token>'  or  --header="Authorization: Basic <base64>"
    # Also matches --header='...' (with =) and bare -H'...' (no space).
    (
        r"(-H|--header)[\s=]+([\"']?)Authorization:\s*(Bearer|Basic)\s+(\S+)(\2)",
        r"\1 \2Authorization: \3 [REDACTED]\2",
    ),
    # curl -H Authorization: Bearer <token>  (no quotes at all)
    (
        r"(-H|--header)[\s=]+Authorization:\s*(Bearer|Basic)\s+(\S+)",
        r"\1 Authorization: \2 [REDACTED]",
    ),
    # curl -H 'Authorization: <raw-token>'  (no Bearer/Basic scheme)
    (
        r"(-H|--header)[\s=]+([\"']?)Authorization:\s+(\S+)(\2)",
        r"\1 \2Authorization: [REDACTED]\2",
    ),
    # curl -H 'X-API-Key: <value>' / -H 'api-key: <value>'
    (
        r"(-H|--header)[\s=]+([\"']?)(x-api-key|apikey):\s*(\S+)(\2)",
        r"\1 \2\3: [REDACTED]\2",
    ),
    # Environment variable assignments: TOKEN=abc123 SECRET=xyz PASSWORD=foo
    (
        r"\b(TOKEN|SECRET|PASSWORD|API_KEY|AUTH_TOKEN|ACCESS_TOKEN|PASS)\s*=\s*(\S+)",
        r"\1=[REDACTED]",
    ),
    # URL query parameters: ?access_token=abc123 &token=xyz
    (
        r"([?&])(access_token|token|api_key|apikey)\s*=\s*(\S+)",
        r"\1\2=[REDACTED]",
    ),
]


def sanitize_command(cmd: str) -> str:
    """Redact secrets (tokens, keys, passwords) from a shell command string.

    Preserves the structure of the command and the secret key names so
    that the log entry remains debuggable without exposing the raw value.
    """
    for pattern, replacement in _SECRET_PATTERNS:
        cmd = re.sub(pattern, replacement, cmd, flags=re.IGNORECASE)
    return cmd


def _log(level: str, message: str, **kwargs: Any) -> None:
    """Write a structured log line to stderr."""
    entry = {
        "ts": time.time(),
        "level": level,
        "msg": message,
        **kwargs,
    }
    print(json.dumps(entry, default=str), file=sys.stderr, flush=True)


def info(message: str, **kwargs: Any) -> None:
    _log("INFO", message, **kwargs)


def warn(message: str, **kwargs: Any) -> None:
    _log("WARN", message, **kwargs)


def error(message: str, **kwargs: Any) -> None:
    _log("ERROR", message, **kwargs)


def tool_call(name: str, args: dict[str, Any], duration_ms: float, status: str = "ok") -> None:
    _log("TOOL", name, args=args, duration_ms=duration_ms, status=status)


def subagent(role: str, task: str, duration_ms: float, success: bool) -> None:
    _log("SUBAGENT", task[:80], role=role, duration_ms=duration_ms, success=success)


def audit_dangerous_execution(
    cmd: str,
    dangers: list[str],
    cwd: str,
    outcome: str,
    exit_code: int | None = None,
    duration_ms: float | None = None,
) -> None:
    """Log a dangerous command execution attempt for audit trail.

    Args:
        cmd: The command that was run (or attempted).
        dangers: List of dangerous pattern reasons detected.
        cwd: Working directory at time of execution.
        outcome: One of "blocked", "completed", "error".
        exit_code: Process exit code (None if blocked).
        duration_ms: Execution wall-clock ms (None if blocked).
    """
    _log(
        "AUDIT",
        f"dangerous_cmd: {outcome}",
        command=sanitize_command(cmd),
        dangers=dangers,
        cwd=cwd,
        outcome=outcome,
        exit_code=exit_code,
        duration_ms=duration_ms,
    )


def startup(project_root: str) -> None:
    _log("STARTUP", "codeforge-mcp starting", project=project_root)


def shutdown() -> None:
    _log("SHUTDOWN", "codeforge-mcp shutting down")
