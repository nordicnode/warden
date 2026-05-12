"""Tests for MCP server tool response envelopes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from codeforge_mcp import server
from codeforge_mcp.subagents.orchestrator import ContextBudget, SubAgentResult


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
