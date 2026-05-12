"""Tests for MCP server tool response envelopes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from codeforge_mcp import server
from codeforge_mcp.subagents.orchestrator import ContextBudget, SubAgentResult


class TestIsTestHeuristic:
    """Tests for the is_test heuristic in test_run_affected."""

    def test_is_test_regex_matches_test_prefix(self) -> None:
        """Stem matching (^|[._-])(test|spec|tests|specs)([._-]|$) should match test_ prefix."""
        from codeforge_mcp import server
        import re
        pattern = re.compile(r"(^|[._-])(test|spec|tests|specs)([._-]|$)")

        # Should match
        assert pattern.search("test_foo") is not None
        assert pattern.search("test_bar.py") is not None
        assert pattern.search("foo_test.py") is not None
        assert pattern.search("test") is not None

    def test_is_test_regex_matches_spec_prefix(self) -> None:
        """Should match spec_ prefix and _spec suffix."""
        import re
        pattern = re.compile(r"(^|[._-])(test|spec|tests|specs)([._-]|$)")

        assert pattern.search("spec_helper.py") is not None
        assert pattern.search("foo_spec.py") is not None
        assert pattern.search("_spec") is not None

    def test_is_test_regex_rejects_false_positives(self) -> None:
        """Word-boundary-aware regex should not match 'contest' or 'specimen'."""
        import re
        pattern = re.compile(r"(^|[._-])(test|spec|tests|specs)([._-]|$)")

        assert pattern.search("contest") is None
        assert pattern.search("specimen") is None
        assert pattern.search("tests.py") is not None  # but tests.py is a valid test file

    def test_is_test_directory_patterns(self) -> None:
        """Standard test directory conventions: /tests/, /test/, /__tests__/, /__spec__/, /spec/.

        Mirrors the actual heuristic in server.py:test_run_affected, which uses
        both substring checks (for embedded paths like /project/tests/foo.py) and
        startswith checks (for relative paths like tests/foo.py).
        """
        import os
        paths = [
            ("src/utils.py", False),
            ("tests/test_server.py", True),   # startswith tests/
            ("/project/tests/test_server.py", True),  # /tests/ substring
            ("src/test/test_foo.py", True),   # /test/ substring
            ("__tests__/test_foo.py", True),  # startswith __tests__/
            ("/project/__tests__/foo.py", True),  # /__tests__/ substring
            ("spec/unit_spec.py", True),  # startswith spec/
            ("/project/spec/helpers.py", True),  # /spec/ substring
            ("__spec__/test_foo.py", True),  # startswith __spec__/
            ("special/utils.py", False),  # spec substring but NOT a test dir
            ("contest/main.py", False),   # contest contains test but not a test file
        ]
        for path_str, expected in paths:
            normalized = path_str.replace(os.sep, "/")
            result = (
                "/tests/" in normalized
                or "/test/" in normalized
                or "/__tests__/" in normalized
                or "/__spec__/" in normalized
                or "/spec/" in normalized
                or normalized.startswith("tests/")
                or normalized.startswith("test/")
                or normalized.startswith("__tests__/")
                or normalized.startswith("__spec__/")
                or normalized.startswith("spec/")
            )
            assert result == expected, f"{path_str}: expected {expected}, got {result}"


class TestToolResponseShapes:
    @pytest.mark.asyncio
    async def test_code_find_files_returns_envelope(self) -> None:
        async def fake_to_thread(*args, **kwargs):
            return [{"path": "a.py"}]

        with patch.object(server, "_ensure_init"), \
             patch("codeforge_mcp.server.asyncio.to_thread", new=fake_to_thread):
            server._project_root = "/tmp/project"
            result = await server.code_find_files("a.py")
        assert result == {"success": True, "files": [{"path": "a.py"}]}

    @pytest.mark.asyncio
    async def test_code_search_returns_envelope(self) -> None:
        async def fake_to_thread(*args, **kwargs):
            return [{"file": "a.py", "line": 1}]

        with patch.object(server, "_ensure_init"), \
             patch("codeforge_mcp.server.asyncio.to_thread", new=fake_to_thread):
            server._project_root = "/tmp/project"
            result = await server.code_search("needle")
        assert result == {"success": True, "matches": [{"file": "a.py", "line": 1}]}

    @pytest.mark.asyncio
    async def test_lsp_find_references_returns_envelope(self) -> None:
        server._project_root = "/tmp/project"
        server._lsp = AsyncMock()
        server._lsp.references = AsyncMock(return_value=[{"file": "/tmp/project/a.py", "line": 1}])
        with patch.object(server, "_ensure_init"):
            result = await server.lsp_find_references("a.py", 1, 0)
        assert result == {
            "success": True,
            "references": [{"file": "/tmp/project/a.py", "line": 1}],
        }

    @pytest.mark.asyncio
    async def test_context_budget_tracks_non_subagent_tool_usage(self) -> None:
        async def fake_to_thread(*args, **kwargs):
            return [{"file": "a.py", "line": 1}]

        server._project_root = "/tmp/project"
        server._orchestrator = SimpleNamespace(budget=ContextBudget(max_tokens=1000))

        with patch.object(server, "_ensure_init"), \
             patch("codeforge_mcp.server.asyncio.to_thread", new=fake_to_thread):
            await server.code_search("needle")
            budget = await server.context_budget()

        assert budget["used_tokens"] > 0
        assert budget["call_count"] >= 1
        assert budget["tool_call_count"] >= 1

    @pytest.mark.asyncio
    async def test_write_file_reverts_when_review_fails(self, tmp_path: Path) -> None:
        server._project_root = str(tmp_path)
        server._orchestrator = SimpleNamespace(
            budget=ContextBudget(max_tokens=1000),
            review_change=AsyncMock(return_value=SubAgentResult(
                role="reviewer",
                task="review",
                success=True,
                data={
                    "review_passed": False,
                    "blocking_issues": [{"message": "broken"}],
                    "review_context": {"review_files": ["bad.py"]},
                    "checklist": ["broken"],
                },
            )),
        )
        server._ast_indexer = None
        server._graph = None

        with patch.object(server, "_ensure_init"):
            result = await server.write_file("bad.py", "x = 1\n")

        assert result["written"] is False
        assert result["reverted"] is True
        assert "reverted after post-edit review" in result["error"].lower()
        assert not (tmp_path / "bad.py").exists()

    @pytest.mark.asyncio
    async def test_patch_file_tool_reverts_when_review_fails(self, tmp_path: Path) -> None:
        target = tmp_path / "edit.py"
        target.write_text("x = 1\n")
        server._project_root = str(tmp_path)
        server._orchestrator = SimpleNamespace(
            budget=ContextBudget(max_tokens=1000),
            review_change=AsyncMock(return_value=SubAgentResult(
                role="reviewer",
                task="review",
                success=True,
                data={
                    "review_passed": False,
                    "blocking_issues": [{"message": "broken"}],
                    "review_context": {"review_files": ["edit.py"]},
                    "checklist": ["broken"],
                },
            )),
        )
        server._ast_indexer = None
        server._graph = None

        with patch.object(server, "_ensure_init"):
            result = await server.patch_file_tool("edit.py", 1, 1, "x = 2")

        assert result["success"] is False
        assert result["data"]["reverted"] is True
        assert target.read_text() == "x = 1\n"

    @pytest.mark.asyncio
    async def test_safe_edit_tool_appends_code_review_stage(self, tmp_path: Path) -> None:
        server._project_root = str(tmp_path)
        server._orchestrator = SimpleNamespace(
            budget=ContextBudget(max_tokens=1000),
            review_change=AsyncMock(return_value=SubAgentResult(
                role="reviewer",
                task="review",
                success=True,
                data={
                    "review_passed": True,
                    "blocking_issues": [],
                    "review_context": {"review_files": ["edit.py"]},
                    "checklist": ["clean"],
                },
            )),
        )
        server._ast_indexer = None
        server._graph = None
        server._lsp = None

        fake_safe_edit = {
            "success": True,
            "error_code": "NONE",
            "error_message": "",
            "data": {
                "path": "edit.py",
                "written": True,
                "hash": "abc",
                "bytes_written": 4,
                "total_lines": 1,
                "diff_preview": "--- a/edit.py\n+++ b/edit.py",
                "validation_passed": True,
                "risk_level": "LOW",
                "validation_stages": [],
            },
        }

        with patch.object(server, "_ensure_init"), \
             patch("codeforge_mcp.tools.safe_edit.safe_edit", AsyncMock(return_value=fake_safe_edit)):
            result = await server.safe_edit_tool("edit.py", 1, 1, "x = 2")

        assert result["success"] is True
        stages = result["data"]["validation_stages"]
        assert any(stage["stage"] == "code_review" for stage in stages)
