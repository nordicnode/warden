"""Tests for safe_edit rollback behavior."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codeforge_mcp.tools.safe_edit import safe_edit


@pytest.mark.asyncio
async def test_safe_edit_reverts_invalid_syntax(tmp_path: Path) -> None:
    target = tmp_path / "hello.py"
    original = "def greet(name):\n    return f'Hello, {name}'\n"
    target.write_text(original)

    def fake_patch_file(*args: object, **kwargs: object) -> dict[str, object]:
        target.write_text("def greet(name\n")
        return {
            "success": True,
            "data": {
                "path": "hello.py",
                "written": True,
                "hash": "brokenhash",
                "bytes_written": len("def greet(name\n"),
                "total_lines": 1,
                "diff_preview": "--- a/hello.py\n+++ b/hello.py",
                "syntax_valid": False,
                "diagnostics": ["Line 1: syntax error"],
            },
        }

    with patch("codeforge_mcp.tools.safe_edit.patch_file", side_effect=fake_patch_file):
        result = await safe_edit(
            project_root=str(tmp_path),
            path="hello.py",
            start_line=1,
            end_line=1,
            new_content="def greet(name",
            run_diagnostics=False,
            lsp_multiplexer=AsyncMock(),
            ast_indexer=MagicMock(),
            graph=MagicMock(),
        )

    assert result["success"] is True
    assert result["data"]["validation_passed"] is False
    assert result["data"]["risk_level"] == "CRITICAL"
    assert result["data"]["reverted"] is True
    assert result["data"]["written"] is False
    assert target.read_text() == original
