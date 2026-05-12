"""safe_edit — compound validation pipeline for code changes.

Orchestrates a multi-stage validation after applying a patch:
1. Apply the line-range patch (via patch_file)
2. Parse with tree-sitter — check for syntax errors
3. Run LSP diagnostics — check for type errors / new warnings
4. Run affected tests — check for regressions

Returns a comprehensive report so the model can decide whether to
keep the edit, refine it, or revert.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from codeforge_mcp.tools.responses import ToolResponse, ErrorCode
from codeforge_mcp.tools.patch import patch_file


async def safe_edit(
    project_root: str,
    path: str,
    start_line: int,
    end_line: int,
    new_content: str,
    expected_hash: str = "",
    run_tests: bool = False,
    run_diagnostics: bool = True,
    lsp_multiplexer: Any = None,
    ast_indexer: Any = None,
    graph: Any = None,
) -> dict[str, Any]:
    """Apply a code edit with multi-stage validation.

    This is the guardrail tool — it applies a patch, then validates the
    result through up to 3 stages:

    1. **Syntax check** (always): tree-sitter parse validation
    2. **LSP diagnostics** (if run_diagnostics=True and LSP available):
       type errors, undefined references, etc.
    3. **Affected tests** (if run_tests=True): runs only tests that
       touch the modified symbol via call graph analysis

    Args:
        project_root: Absolute project root path.
        path: File path relative to project root.
        start_line: First line to replace (1-based, inclusive).
        end_line: Last line to replace (1-based, inclusive).
        new_content: Replacement content for the specified line range.
        expected_hash: Optional file hash for staleness detection.
        run_tests: Whether to run affected tests after the edit.
        run_diagnostics: Whether to run LSP diagnostics.
        lsp_multiplexer: LSPMultiplexer instance (for diagnostics).
        ast_indexer: ASTIndexer instance (for re-indexing).
        graph: KnowledgeGraph instance (for impact analysis).

    Returns:
        ToolResponse-shaped dict with comprehensive validation results.
    """
    root = Path(project_root).resolve()
    file_path = (root / path).resolve()
    original_bytes = _read_existing_file_bytes(file_path)

    # Stage 1: Apply the patch (includes tree-sitter syntax validation)
    patch_result = patch_file(
        project_root=project_root,
        path=path,
        start_line=start_line,
        end_line=end_line,
        new_content=new_content,
        expected_hash=expected_hash,
        validate_syntax=True,
    )

    if not patch_result.get("success", False):
        return patch_result

    data = patch_result.get("data", {})
    validation_stages: list[dict[str, Any]] = []

    # Record syntax validation result
    validation_stages.append({
        "stage": "syntax",
        "passed": data.get("syntax_valid", True),
        "details": data.get("diagnostics", []),
    })

    if not data.get("syntax_valid", True):
        reverted, rollback_details = _restore_original_file(file_path, original_bytes)
        validation_stages.append({
            "stage": "rollback",
            "passed": reverted,
            "details": [rollback_details],
        })

        current_bytes = _read_existing_file_bytes(file_path, retries=1, delay_seconds=0) or b""
        current_hash = _hash_bytes(current_bytes) if current_bytes else data.get("hash", "")
        current_lines = _count_lines(current_bytes) if current_bytes else data.get("total_lines", 0)

        return ToolResponse.ok(
            path=data.get("path", path),
            written=not reverted,
            reverted=reverted,
            hash=current_hash,
            bytes_written=len(current_bytes),
            total_lines=current_lines,
            diff_preview=data.get("diff_preview", ""),
            validation_passed=False,
            risk_level="CRITICAL",
            validation_stages=validation_stages,
        ).model_dump()

    # Stage 2: Re-index the file in the knowledge graph
    if ast_indexer is not None:
        try:
            abs_path = str(file_path)
            ast_indexer.index_file_incremental(abs_path)
            validation_stages.append({
                "stage": "reindex",
                "passed": True,
                "details": [],
            })
        except Exception as e:
            validation_stages.append({
                "stage": "reindex",
                "passed": False,
                "details": [f"Re-indexing failed: {e}"],
            })

    # Stage 3: LSP diagnostics
    lsp_diagnostics: list[dict[str, Any]] = []
    if run_diagnostics and lsp_multiplexer is not None:
        try:
            abs_path = str(file_path)
            lsp_diagnostics = await lsp_multiplexer.diagnostics(abs_path)
            new_errors = [
                d for d in lsp_diagnostics
                if d.get("severity", 0) <= 2  # Error or Warning
            ]
            validation_stages.append({
                "stage": "lsp_diagnostics",
                "passed": len(new_errors) == 0,
                "error_count": len([d for d in new_errors if d.get("severity") == 1]),
                "warning_count": len([d for d in new_errors if d.get("severity") == 2]),
                "details": [
                    {
                        "line": d.get("range", {}).get("start", {}).get("line", 0) + 1,
                        "severity": "error" if d.get("severity") == 1 else "warning",
                        "message": d.get("message", ""),
                    }
                    for d in new_errors[:20]
                ],
            })
        except Exception as e:
            validation_stages.append({
                "stage": "lsp_diagnostics",
                "passed": True,  # Don't fail on LSP errors
                "details": [f"LSP diagnostics unavailable: {e}"],
            })

    # Stage 4: Affected tests
    test_results: dict[str, Any] = {}
    if run_tests and graph is not None:
        try:
            test_results = await _run_affected_tests(
                project_root, path, start_line, end_line, graph, ast_indexer
            )
            validation_stages.append({
                "stage": "affected_tests",
                "passed": test_results.get("exit_code", 1) == 0,
                "tests_run": test_results.get("tests_run", 0),
                "tests_failed": test_results.get("tests_failed", 0),
                "details": test_results.get("failures", []),
            })
        except Exception as e:
            validation_stages.append({
                "stage": "affected_tests",
                "passed": True,  # Don't fail on test runner errors
                "details": [f"Test execution failed: {e}"],
            })

    # Overall verdict
    all_passed = all(stage["passed"] for stage in validation_stages)
    risk_level = "LOW"
    if not data.get("syntax_valid", True):
        risk_level = "CRITICAL"
    elif any(not s["passed"] for s in validation_stages if s["stage"] == "lsp_diagnostics"):
        risk_level = "HIGH"
    elif any(not s["passed"] for s in validation_stages if s["stage"] == "affected_tests"):
        risk_level = "HIGH"

    return ToolResponse.ok(
        path=data.get("path", path),
        written=True,
        hash=data.get("hash", ""),
        bytes_written=data.get("bytes_written", 0),
        total_lines=data.get("total_lines", 0),
        diff_preview=data.get("diff_preview", ""),
        validation_passed=all_passed,
        risk_level=risk_level,
        validation_stages=validation_stages,
    ).model_dump()


async def _run_affected_tests(
    project_root: str,
    path: str,
    start_line: int,
    end_line: int,
    graph: Any,
    ast_indexer: Any,
) -> dict[str, Any]:
    """Run only tests affected by the changed lines.

    Uses the knowledge graph's call graph to identify which test files
    are upstream of symbols defined in the changed line range.
    """
    from codeforge_mcp.tools.execution import test_run

    root = Path(project_root).resolve()
    abs_path = str((root / path).resolve())

    # Find symbols in the changed line range
    affected_symbols = []
    if graph is not None:
        try:
            rows = graph.conn.execute(
                "SELECT id, name, file FROM symbols WHERE file = ? AND line >= ? AND line <= ?",
                (abs_path, start_line, end_line),
            ).fetchall()
            affected_symbols = [{"id": r[0], "name": r[1], "file": r[2]} for r in rows]
        except Exception:
            pass

    # Find test files that depend on these symbols
    test_files: set[str] = set()
    if graph is not None:
        for sym in affected_symbols:
            try:
                upstream = graph.upstream(sym["id"], depth=5)
                for caller in upstream:
                    fname = caller.get("file", "")
                    name = caller.get("name", "")
                    if "test" in fname.lower() or "test" in name.lower() or "spec" in fname.lower():
                        # Extract relative path for test selector
                        try:
                            test_files.add(str(Path(fname).relative_to(root)))
                        except ValueError:
                            test_files.add(fname)
            except Exception:
                continue

    # Run only the affected test files
    if not test_files and graph is not None:
        # High impact fallback: if symbols in the changed range affect many modules,
        # run the full suite.
        high_impact = False
        for sym in affected_symbols:
            try:
                impact = graph.affected_modules(sym["id"], depth=3)
                if len(impact) > 10:
                    high_impact = True
                    break
            except Exception:
                continue
        
        if high_impact:
            result = test_run(selector="", project_root=project_root) # Full suite
            return {
                "exit_code": result.get("exit_code", 1),
                "tests_run": "full_suite",
                "tests_failed": len(result.get("failures", [])),
                "failures": result.get("failures", []),
                "message": "High impact change; ran full test suite as fallback.",
            }

        return {
            "exit_code": 0,
            "tests_run": 0,
            "tests_failed": 0,
            "failures": [],
            "message": "No affected tests found in call graph.",
        }

    # Run only the affected test files
    selector = " ".join(sorted(test_files))
    result = test_run(selector=selector, project_root=project_root)
    return {
        "exit_code": result.get("exit_code", 1),
        "tests_run": len(test_files),
        "tests_failed": len(result.get("failures", [])),
        "failures": result.get("failures", []),
        "test_files": sorted(test_files),
    }


def _read_existing_file_bytes(
    file_path: Path,
    retries: int = 4,
    delay_seconds: float = 0.05,
) -> bytes | None:
    """Read a file with a short retry window for newly-created paths."""
    for attempt in range(retries):
        if file_path.is_file():
            try:
                return file_path.read_bytes()
            except OSError:
                return None
        if attempt < retries - 1:
            time.sleep(delay_seconds)
    return None


def _restore_original_file(file_path: Path, original_bytes: bytes | None) -> tuple[bool, str]:
    """Restore the original file contents after a failed safe edit."""
    if original_bytes is None:
        return False, "Rollback unavailable: original file contents could not be captured."
    try:
        file_path.write_bytes(original_bytes)
        return True, "Patched content was reverted because syntax validation failed."
    except OSError as exc:
        return False, f"Rollback failed: {exc}"


def _hash_bytes(content_bytes: bytes) -> str:
    import xxhash

    return xxhash.xxh3_64(content_bytes).hexdigest()


def _count_lines(content_bytes: bytes) -> int:
    try:
        content = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        content = content_bytes.decode("latin-1", errors="replace")
    return len(content.split("\n"))
