"""Tests for safe_edit rollback behavior and diagnostic regression detection."""

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


@pytest.mark.asyncio
async def test_safe_edit_detects_new_lsp_diagnostics(tmp_path: Path) -> None:
    """Simulate an edit that introduces a new error and verify it appears in the response.

    The multiplexer._handle_notification flattens diagnostics to {line, severity,
    message, source} — NOT the nested {range: {start: {line, character}}} shape.
    This test verifies the regression detection uses the flattened shape correctly.
    """
    target = tmp_path / "broken.py"
    target.write_text("x = 1\n")

    # Pre-edit diagnostics: no errors
    pre_diags = []

    # Post-edit diagnostics: one new error introduced by the edit
    # Note: this is the FLATTENED shape from multiplexer._handle_notification
    post_diags = [
        {"line": 1, "severity": 1, "message": "Cannot assign 'int' to 'str'", "source": "pyright"},
        {"line": 2, "severity": 2, "message": "Unused variable 'x'", "source": "pyright"},
    ]

    mock_lsp = AsyncMock()
    # First call → pre diagnostics (empty), second call → post diagnostics
    mock_lsp.diagnostics = AsyncMock(side_effect=[pre_diags, post_diags])

    def fake_patch_file(*args: object, **kwargs: object) -> dict[str, object]:
        target.write_text("x: str = 1\n")
        return {
            "success": True,
            "data": {
                "path": "broken.py",
                "written": True,
                "hash": "newhash",
                "bytes_written": len("x: str = 1\n"),
                "total_lines": 1,
                "diff_preview": "--- a/broken.py\n+++ b/broken.py",
                "syntax_valid": True,
                "diagnostics": [],
            },
        }

    with patch("codeforge_mcp.tools.safe_edit.patch_file", side_effect=fake_patch_file):
        result = await safe_edit(
            project_root=str(tmp_path),
            path="broken.py",
            start_line=1,
            end_line=1,
            new_content="x: str = 1",
            run_diagnostics=True,
            lsp_multiplexer=mock_lsp,
            ast_indexer=MagicMock(),
            graph=MagicMock(),
        )

    assert result["success"] is True

    # Find the LSP diagnostics stage
    stages = result["data"].get("validation_stages", [])
    lsp_stage = next((s for s in stages if s["stage"] == "lsp_diagnostics"), None)
    assert lsp_stage is not None, "lsp_diagnostics stage not found in validation_stages"

    # Both new errors should be detected (not skipped due to shape mismatch)
    assert lsp_stage["error_count"] == 1   # severity=1 → error
    assert lsp_stage["warning_count"] == 1  # severity=2 → warning
    assert lsp_stage["passed"] is False      # new errors = failed

    # Verify details use flattened line/character (not nested range)
    details = lsp_stage["details"]
    assert len(details) == 2
    error_detail = next(d for d in details if d["severity"] == "error")
    assert error_detail["line"] == 1                     # d.get("line", 0) — not d["range"]["start"]["line"] + 1
    assert error_detail["message"] == "Cannot assign 'int' to 'str'"
    warning_detail = next(d for d in details if d["severity"] == "warning")
    assert warning_detail["line"] == 2

    # Overall validation should fail due to new errors
    assert result["data"]["validation_passed"] is False
    assert result["data"]["risk_level"] == "HIGH"
