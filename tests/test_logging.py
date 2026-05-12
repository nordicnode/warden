"""Tests for structured JSON logging — info, warn, error, tool_call, subagent, startup, shutdown.

Verifies:
- Logs are written as JSON lines to stderr
- Each log entry has required fields (ts, level, msg)
- Extra kwargs appear as top-level keys in the JSON
- tool_call and subagent include role/task/duration fields
"""

import io
import json
import sys
from unittest.mock import patch

from codeforge_mcp.logging import (
    info,
    warn,
    error,
    tool_call,
    subagent,
    startup,
    shutdown,
)


def _capture_stderr(fn, *args, **kwargs):
    """Run fn and return stderr as parsed JSON lines."""
    buf = io.StringIO()
    with patch.object(sys, 'stderr', buf):
        fn(*args, **kwargs)
    lines = [line for line in buf.getvalue().strip().split("\n") if line]
    return [json.loads(line) for line in lines]


class TestInfoWarnError:
    """Verify info, warn, error log structured JSON to stderr."""

    def test_info_logs_json(self) -> None:
        entries = _capture_stderr(info, "server started", port=8080)
        assert len(entries) == 1
        e = entries[0]
        assert e["level"] == "INFO"
        assert e["msg"] == "server started"
        assert "ts" in e
        assert isinstance(e["ts"], float)
        assert e["port"] == 8080

    def test_warn_logs_json(self) -> None:
        entries = _capture_stderr(warn, "low memory", available_mb=128)
        assert len(entries) == 1
        e = entries[0]
        assert e["level"] == "WARN"
        assert e["msg"] == "low memory"
        assert e["available_mb"] == 128

    def test_error_logs_json(self) -> None:
        entries = _capture_stderr(error, "connection failed", reason="timeout")
        assert len(entries) == 1
        e = entries[0]
        assert e["level"] == "ERROR"
        assert e["msg"] == "connection failed"
        assert e["reason"] == "timeout"

    def test_multiple_logs(self) -> None:
        buf = io.StringIO()
        with patch.object(sys, 'stderr', buf):
            info("one")
            warn("two")
            error("three")
        lines = [json.loads(line) for line in buf.getvalue().strip().split("\n") if line]
        assert len(lines) == 3
        assert [l["level"] for l in lines] == ["INFO", "WARN", "ERROR"]

    def test_no_extra_kwargs(self) -> None:
        entries = _capture_stderr(info, "simple message")
        e = entries[0]
        # Only ts, level, msg — no extra keys
        assert set(e.keys()) == {"ts", "level", "msg"}

    def test_complex_value_serialized(self) -> None:
        entries = _capture_stderr(info, "data", items=[1, 2, 3], nested={"a": 1})
        e = entries[0]
        assert e["items"] == [1, 2, 3]
        assert e["nested"] == {"a": 1}


class TestToolCall:
    """Verify tool_call logs with duration and status."""

    def test_tool_call_success(self) -> None:
        entries = _capture_stderr(tool_call, "read_file",
                                   args={"path": "main.py"}, duration_ms=12.5)
        assert len(entries) == 1
        e = entries[0]
        assert e["level"] == "TOOL"
        assert e["msg"] == "read_file"
        assert e["args"] == {"path": "main.py"}
        assert e["duration_ms"] == 12.5
        assert e["status"] == "ok"

    def test_tool_call_failure(self) -> None:
        entries = _capture_stderr(tool_call, "write_file",
                                   args={"path": "/etc/shadow"}, duration_ms=1.0, status="error")
        e = entries[0]
        assert e["status"] == "error"


class TestSubagent:
    """Verify subagent logs with role, task, duration, success."""

    def test_subagent_success(self) -> None:
        entries = _capture_stderr(subagent, "bash-runner", "Run pytest",
                                   duration_ms=3400.0, success=True)
        e = entries[0]
        assert e["level"] == "SUBAGENT"
        assert e["role"] == "bash-runner"
        assert e["msg"] == "Run pytest"  # task[:80]
        assert e["duration_ms"] == 3400.0
        assert e["success"] is True

    def test_subagent_failure(self) -> None:
        entries = _capture_stderr(subagent, "code-searcher", "Find bug",
                                   duration_ms=500.0, success=False)
        e = entries[0]
        assert e["success"] is False

    def test_subagent_truncates_long_task(self) -> None:
        long_task = "x" * 100
        entries = _capture_stderr(subagent, "researcher", long_task,
                                   duration_ms=1.0, success=True)
        e = entries[0]
        # Truncated to 80 chars
        assert len(e["msg"]) <= 80
        assert e["msg"] == long_task[:80]


class TestStartupShutdown:
    """Verify startup and shutdown log messages."""

    def test_startup(self) -> None:
        entries = _capture_stderr(startup, "/home/user/project")
        e = entries[0]
        assert e["level"] == "STARTUP"
        assert "starting" in e["msg"]
        assert e["project"] == "/home/user/project"

    def test_shutdown(self) -> None:
        entries = _capture_stderr(shutdown)
        e = entries[0]
        assert e["level"] == "SHUTDOWN"
        assert "shutting" in e["msg"]
