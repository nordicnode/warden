"""Tests for patch_file tool (GAP-2)."""

import os
import tempfile
import threading
import time
from pathlib import Path

import pytest

from codeforge_mcp.tools.patch import patch_file


@pytest.fixture
def project(tmp_path):
    """Create a small project with a Python file."""
    f = tmp_path / "hello.py"
    f.write_text("def greet(name):\n    return f'Hello, {name}!'\n\n\ndef main():\n    print(greet('World'))\n")
    return tmp_path


class TestPatchFile:
    def test_basic_patch(self, project):
        result = patch_file(str(project), "hello.py", 2, 2, "    return f'Hi, {name}!'")
        assert result["success"] is True
        assert result["data"]["written"] is True
        assert result["data"]["lines_removed"] == 1
        assert result["data"]["lines_added"] == 1
        content = (project / "hello.py").read_text()
        assert "Hi, {name}!" in content

    def test_multi_line_patch(self, project):
        result = patch_file(str(project), "hello.py", 1, 2, "def greet(name, greeting='Hey'):\n    return f'{greeting}, {name}!'")
        assert result["success"] is True
        content = (project / "hello.py").read_text()
        assert "greeting='Hey'" in content

    def test_hash_verification_passes(self, project):
        import xxhash
        content = (project / "hello.py").read_bytes()
        h = xxhash.xxh3_64(content).hexdigest()
        result = patch_file(str(project), "hello.py", 2, 2, "    return 'patched'", expected_hash=h)
        assert result["success"] is True

    def test_hash_verification_fails(self, project):
        result = patch_file(str(project), "hello.py", 2, 2, "    return 'patched'", expected_hash="badhash")
        assert result["success"] is False
        assert result["error_code"] == "PATCH_HASH_MISMATCH"

    def test_path_traversal_denied(self, project):
        result = patch_file(str(project), "../../../etc/passwd", 1, 1, "hacked")
        assert result["success"] is False
        assert result["error_code"] == "FILE_TRAVERSAL_DENIED"

    def test_protected_path(self, project):
        (project / ".git").mkdir()
        (project / ".git" / "config").write_text("test")
        result = patch_file(str(project), ".git/config", 1, 1, "hacked")
        assert result["success"] is False
        assert result["error_code"] == "FILE_PROTECTED"

    def test_file_not_found(self, project):
        result = patch_file(str(project), "nonexistent.py", 1, 1, "test")
        assert result["success"] is False
        assert result["error_code"] == "FILE_NOT_FOUND"

    def test_retries_briefly_for_new_file(self, project):
        target = project / "new_file.py"

        def delayed_create() -> None:
            time.sleep(0.03)
            target.write_text("value = 1\n")

        writer = threading.Thread(target=delayed_create)
        writer.start()
        try:
            result = patch_file(str(project), "new_file.py", 1, 1, "value = 2")
        finally:
            writer.join()

        assert result["success"] is True
        assert target.read_text() == "value = 2\n"

    def test_invalid_line_range(self, project):
        result = patch_file(str(project), "hello.py", 5, 2, "test")
        assert result["success"] is False
        assert result["error_code"] == "PATCH_LINE_MISMATCH"

    def test_start_beyond_file(self, project):
        result = patch_file(str(project), "hello.py", 999, 999, "test")
        assert result["success"] is False
        assert result["error_code"] == "PATCH_LINE_MISMATCH"

    def test_diff_preview_included(self, project):
        result = patch_file(str(project), "hello.py", 2, 2, "    return 'changed'")
        assert result["success"] is True
        assert "diff_preview" in result["data"]
        assert "---" in result["data"]["diff_preview"]
        assert "+++" in result["data"]["diff_preview"]

    def test_syntax_validation_reports_errors(self, project):
        result = patch_file(str(project), "hello.py", 1, 1, "def greet(name")
        assert result["success"] is True  # writes anyway
        assert result["data"]["written"] is True
        # syntax_valid may be False if tree-sitter is available
        assert "syntax_valid" in result["data"]


class TestResponseModels:
    def test_tool_response_ok(self):
        from codeforge_mcp.tools.responses import ToolResponse
        resp = ToolResponse.ok(path="test.py", lines=42)
        assert resp.success is True
        assert resp.data["path"] == "test.py"
        assert resp.error_code.value == "NONE"

    def test_tool_response_error(self):
        from codeforge_mcp.tools.responses import ToolResponse, ErrorCode
        resp = ToolResponse.error(ErrorCode.FILE_NOT_FOUND, "Not found")
        assert resp.success is False
        assert resp.error_code == ErrorCode.FILE_NOT_FOUND
        assert "Not found" in resp.error_message


class TestContextBudget:
    def test_budget_tracking(self):
        from codeforge_mcp.subagents.orchestrator import ContextBudget
        budget = ContextBudget(max_tokens=1000)
        assert budget.remaining == 1000
        budget.consume(300)
        assert budget.remaining == 700
        assert budget.call_count == 1
        assert not budget.would_exceed(500)
        assert budget.would_exceed(800)

    def test_budget_warning(self):
        from codeforge_mcp.subagents.orchestrator import ContextBudget
        budget = ContextBudget(max_tokens=100)
        budget.consume(85)
        summary = budget.summary()
        assert summary["budget_warning"] is True
        assert summary["utilization_pct"] == 85.0

    def test_budget_summary(self):
        from codeforge_mcp.subagents.orchestrator import ContextBudget
        budget = ContextBudget(max_tokens=16000)
        summary = budget.summary()
        assert summary["used_tokens"] == 0
        assert summary["max_tokens"] == 16000
        assert summary["remaining"] == 16000
        assert summary["tool_call_count"] == 0
        assert summary["subagent_call_count"] == 0
        assert summary["budget_warning"] is False
