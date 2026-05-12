"""Codeforge MCP Server — persistent 5-layer knowledge graph for AI code navigation.

Entry point: `uv run codeforge-mcp --project /path/to/project`
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from codeforge_mcp.graph import KnowledgeGraph
from codeforge_mcp.ast import ASTIndexer
from codeforge_mcp.lsp import LSPMultiplexer
from codeforge_mcp.subagents import SubAgentOrchestrator
from codeforge_mcp.indexer import index_project, find_project_root
from codeforge_mcp import logging as log
from codeforge_mcp.logging import sanitize_command

# ── Globals ──────────────────────────────────────────────────────────

_project_root: str = ""
_graph: KnowledgeGraph | None = None
_ast_indexer: ASTIndexer | None = None
_lsp: LSPMultiplexer | None = None
_orchestrator: SubAgentOrchestrator | None = None
_watcher: Any = None  # FileWatcher, lazy-imported
_initialized: bool = False
_reindex_on_startup: bool = False
_watching_disabled: bool = False
_init_lock: threading.Lock = threading.Lock()

# ── FastMCP server ───────────────────────────────────────────────────

mcp = FastMCP("codeforge")


def _ensure_init() -> None:
    """Lazily initialize the graph and indexer on first tool call."""
    global _project_root, _graph, _ast_indexer, _lsp, _orchestrator, _initialized, _reindex_on_startup

    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

        root = find_project_root(_project_root or os.getcwd())
        _project_root = str(root)
        db_path = root / ".codeforge" / "knowledge.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        log.startup(str(root))

        _graph = KnowledgeGraph(str(db_path))
        _ast_indexer = ASTIndexer(_graph)
        _lsp = LSPMultiplexer(str(root))
        _orchestrator = SubAgentOrchestrator(
            graph=_graph,
            ast_indexer=_ast_indexer,
            lsp_multiplexer=_lsp,
            project_root=str(root),
        )
        _initialized = True

        # ── --reindex flag: force a full rebuild on startup ───────────
        if _reindex_on_startup:
            t0 = time.time()
            log.info("startup", reindex="full", reason="--reindex flag")
            _reindex_on_startup = False  # one-shot
            symbols_removed = _graph.clear_symbols()
            result = index_project(_project_root, _graph, _ast_indexer, full=True)
            result["symbols_removed"] = symbols_removed
            log.info("startup", reindex="done", duration_ms=int((time.time() - t0) * 1000), **result)


def _tool_timing(name: str, start: float, args: dict[str, Any]) -> None:
    log.tool_call(name, args, (time.time() - start) * 1000)
    if _orchestrator is None or name in {"context_budget", "spawn_subagent", "spawn_subagents"}:
        return
    try:
        payload = json.dumps(args, sort_keys=True, default=str)
    except TypeError:
        payload = repr(args)
    token_estimate = max(1, (len(name) + len(payload)) // 4)
    _orchestrator.budget.consume(token_estimate, source="tool")


def _capture_file_state(path: str) -> tuple[bool, bytes | None]:
    """Capture whether a file existed and its original bytes before mutation."""
    abs_path = (Path(_project_root).resolve() / path).resolve()
    if not abs_path.exists():
        return False, None
    try:
        return True, abs_path.read_bytes()
    except OSError:
        return True, None


def _restore_file_state(path: str, existed_before: bool, original_bytes: bytes | None) -> tuple[bool, str]:
    """Restore a file to its pre-edit state, deleting new files when needed."""
    abs_path = (Path(_project_root).resolve() / path).resolve()
    try:
        if existed_before:
            if original_bytes is None:
                return False, "Rollback unavailable: original file contents could not be captured."
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_bytes(original_bytes)
            return True, "Original file contents restored after failed review."
        if abs_path.exists():
            abs_path.unlink()
        return True, "New file removed after failed review."
    except OSError as exc:
        return False, f"Rollback failed: {exc}"


def _current_file_metadata(path: str) -> dict[str, Any]:
    """Return current on-disk metadata for a file after rollback."""
    abs_path = (Path(_project_root).resolve() / path).resolve()
    if not abs_path.is_file():
        return {"hash": "", "bytes_written": 0, "total_lines": 0}
    content_bytes = abs_path.read_bytes()
    try:
        import xxhash
        file_hash = xxhash.xxh3_64(content_bytes).hexdigest()
    except Exception:
        file_hash = ""
    try:
        content = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        content = content_bytes.decode("latin-1", errors="replace")
    return {
        "hash": file_hash,
        "bytes_written": len(content_bytes),
        "total_lines": len(content.split("\n")),
    }


async def _reindex_after_edit(path: str) -> None:
    """Keep the graph in sync after a mutation or rollback."""
    if _ast_indexer is None or _graph is None:
        return
    abs_path = str((Path(_project_root).resolve() / path).resolve())
    if Path(abs_path).exists():
        await asyncio.to_thread(_ast_indexer.index_file_incremental, abs_path)
        return
    rel_path = path
    await asyncio.to_thread(_graph.delete_symbols_in_file, abs_path)
    if rel_path != abs_path:
        await asyncio.to_thread(_graph.delete_symbols_in_file, rel_path)


async def _review_edit(
    path: str,
    start_line: int,
    end_line: int,
    diff_preview: str = "",
) -> dict[str, Any]:
    """Run the post-edit reviewer with blast-radius context."""
    if _orchestrator is None:
        return {"review_available": False, "review_passed": True, "review": None}

    await _reindex_after_edit(path)
    review_result = await _orchestrator.review_change(
        path=path,
        start_line=start_line,
        end_line=end_line,
        diff_preview=diff_preview,
    )
    review_data = review_result.data if isinstance(review_result.data, dict) else {}
    review_passed = bool(review_result.success and review_data.get("review_passed", True))
    return {
        "review_available": True,
        "review_passed": review_passed,
        "review": review_data,
        "review_error": review_result.error,
    }


def _review_stage_from_payload(review_payload: dict[str, Any]) -> dict[str, Any]:
    """Convert reviewer output into a validation stage entry."""
    review = review_payload.get("review") or {}
    return {
        "stage": "code_review",
        "passed": review_payload.get("review_passed", True),
        "details": review.get("checklist", []),
        "blocking_issues": review.get("blocking_issues", []),
        "files_reviewed": review.get("review_context", {}).get("review_files", []),
    }


# ── Navigation Tools ─────────────────────────────────────────────────

@mcp.tool()
async def brief(force: bool = False) -> dict[str, Any]:
    """Return a summary of the codebase: file count, symbol count, knowledge score.

    Indexes the project on first call, or when force=True.
    When force=True, does an incremental reindex (changed and new files only;
    symbols from deleted files may persist). For a full clean rebuild use
    the reindex() tool instead.

    Args:
        force: If true, re-index changed and new files even if already indexed.
    """
    t0 = time.time()
    _ensure_init()

    if _graph is None:
        return {"file_count": 0, "symbol_count": 0, "knowledge_score": 0.0}

    symbol_count = await asyncio.to_thread(_graph.symbol_count)

    if symbol_count == 0 or force:
        # Use full=True on the very first index so edges get created.
        # When force=True, keep incremental behavior as documented.
        is_first_index = symbol_count == 0
        result = await asyncio.to_thread(index_project, _project_root, _graph, _ast_indexer, full=is_first_index)
        file_count = await asyncio.to_thread(_graph.file_count)
        symbol_count = await asyncio.to_thread(_graph.symbol_count)
        knowledge_score = await asyncio.to_thread(_graph.knowledge_score)
        out = {
            **result,
            "file_count": file_count,
            "symbol_count": symbol_count,
            "knowledge_score": knowledge_score,
        }
    else:
        from codeforge_mcp.tools.memory import brief as memory_brief
        out = await asyncio.to_thread(memory_brief, _graph)

    _tool_timing("brief", t0, {"force": force})
    return out


@mcp.tool()
async def reindex() -> dict[str, Any]:
    """Force a full rebuild of the knowledge graph from source files.

    Clears all symbols and edges, then re-parses every source file
    with tree-sitter using a two-pass strategy (symbols, then edges).

    Use this after making edits when the file watcher is not running,
    or when the graph appears stale.

    Returns:
        {files_indexed, symbols_added, edges_created, duration_ms, symbols_removed}
    """
    t0 = time.time()
    _ensure_init()

    if _graph is None:
        return {"error": "Knowledge graph not initialized", "files_indexed": 0}

    symbols_removed = await asyncio.to_thread(_graph.clear_symbols)

    result = await asyncio.to_thread(index_project, _project_root, _graph, _ast_indexer, full=True)
    result["symbols_removed"] = symbols_removed

    # Snapshot the diff baseline so _synthetic_diff can detect changes
    # that happen after this manual reindex.
    baseline_count = await asyncio.to_thread(_graph.snapshot_diff_baseline)
    result["baseline_snapshot"] = baseline_count

    _tool_timing("reindex", t0, {})
    return result


@mcp.tool()
async def code_find_files(pattern: str, file_type: str = "") -> dict[str, Any]:
    """Find files by name pattern using fd or fallback glob.

    Args:
        pattern: File name pattern (glob or substring).
        file_type: Optional file extension filter, e.g. '.py'.
    """
    t0 = time.time()
    _ensure_init()
    from codeforge_mcp.tools.navigation import code_find_files as cff
    result = await asyncio.to_thread(cff, _project_root, pattern, file_type)
    _tool_timing("code_find_files", t0, {"pattern": pattern, "file_type": file_type})
    return {"success": True, "files": result}


@mcp.tool()
async def code_search(query: str, regex: bool = False, context: int = 3) -> dict[str, Any]:
    """Search code using ripgrep, returning matches with context lines."""
    t0 = time.time()
    _ensure_init()
    from codeforge_mcp.tools.navigation import code_search as cs
    try:
        result = await asyncio.to_thread(cs, _project_root, query, regex, context)
    except asyncio.CancelledError:
        return {
            "success": False,
            "matches": [],
            "error": "Operation cancelled by client",
        }
    _tool_timing("code_search", t0, {"query": query, "regex": regex, "context": context})
    return {"success": True, "matches": result}


@mcp.tool()
async def symbol_lookup(name: str) -> dict[str, Any]:
    """Look up a symbol via LSP workspace/symbol, falling back to the graph.

    Returns a dict with the symbol's details and ``found: true``, or
    ``{found: false, name: ...}`` when the symbol cannot be resolved.
    """
    t0 = time.time()
    _ensure_init()
    from codeforge_mcp.tools.navigation import symbol_lookup as sl
    try:
        result = await sl(_lsp, _graph, name)
    except Exception as e:
        _tool_timing("symbol_lookup", t0, {"name": name})
        return {"found": False, "name": name, "error": str(e)}
    _tool_timing("symbol_lookup", t0, {"name": name})
    if result is None:
        return {"found": False, "name": name}
    return {"found": True, **result}


@mcp.tool()
async def lsp_workspace_symbols(query: str) -> dict[str, Any]:
    """Search for symbols across the entire workspace via all LSP servers.

    Args:
        query: Symbol name or prefix to search for.
    """
    t0 = time.time()
    _ensure_init()
    if _lsp is None:
        return {"success": False, "symbols": [], "error": "LSP not initialized"}
    try:
        result = await _lsp.workspace_symbols(query)
    except Exception as e:
        _tool_timing("lsp_workspace_symbols", t0, {"query": query})
        return {"success": False, "symbols": [], "error": str(e)}
    _tool_timing("lsp_workspace_symbols", t0, {"query": query})
    return {"success": True, "symbols": result}


# ── Understanding Tools ──────────────────────────────────────────────

@mcp.tool()
async def ast_query(file: str, xpath: str) -> dict[str, Any]:
    """Run a tree-sitter query against a file.

    Supports keyword mode ('function', 'class', 'variable', 'all') and
    tree-sitter S-expression patterns ('(function_definition) @func').
    """
    t0 = time.time()
    _ensure_init()
    from codeforge_mcp.tools.understanding import ast_query as aq
    full_path = Path(_project_root) / file
    try:
        matches = await asyncio.to_thread(aq, _ast_indexer, str(full_path), xpath)
    except Exception as e:
        _tool_timing("ast_query", t0, {"file": file, "xpath": xpath})
        return {"success": False, "matches": [], "error": str(e)}
    _tool_timing("ast_query", t0, {"file": file, "xpath": xpath})

    # The underlying run_ast_query encodes error states (bad xpath,
    # unsupported language, missing grammar, …) as a single dict
    # ``{"type": "error", "message": "..."}``. Surface those as a real
    # failure instead of pretending the query succeeded with zero matches —
    # otherwise callers can't distinguish "no matches" from "bad query".
    if (
        isinstance(matches, list)
        and len(matches) == 1
        and isinstance(matches[0], dict)
        and matches[0].get("type") == "error"
    ):
        return {
            "success": False,
            "matches": [],
            "error": matches[0].get("message", "ast_query failed"),
        }
    return {"success": True, "matches": matches}


@mcp.tool()
async def call_graph(
    function: str,
    file: str = "",
    direction: str = "both",
    depth: int = 2,
) -> dict[str, Any]:
    """Traverse the call graph from a symbol."""
    t0 = time.time()
    _ensure_init()
    from codeforge_mcp.tools.understanding import call_graph as cg
    file_path = str(Path(_project_root) / file) if file else None
    try:
        result = await asyncio.to_thread(cg, _graph, function, file_path, direction, depth)
    except Exception as e:
        _tool_timing("call_graph", t0, {"function": function, "file": file, "direction": direction, "depth": depth})
        return {"upstream": [], "downstream": [], "cycles": [], "error": str(e)}
    # ── Truncate large outputs to prevent context budget depletion ──
    max_items = 50
    for key in ("upstream", "downstream"):
        if key in result and len(result[key]) > max_items:
            result[f"{key}_truncated"] = True
            result[f"{key}_total"] = len(result[key])
            result[key] = result[key][:max_items]

    _tool_timing("call_graph", t0, {"function": function, "file": file, "direction": direction, "depth": depth})
    return result


@mcp.tool()
async def impact_analysis(target: str, file: str = "") -> dict[str, Any]:
    """Analyze the blast radius of changing a symbol."""
    t0 = time.time()
    _ensure_init()
    from codeforge_mcp.tools.understanding import impact_analysis as ia
    file_path = str(Path(_project_root) / file) if file else None
    result = await asyncio.to_thread(ia, _graph, _ast_indexer, target, file_path)
    _tool_timing("impact_analysis", t0, {"target": target, "file": file})
    return result


# ── AST Dependency Graph ─────────────────────────────────────────────

@mcp.tool()
async def cognition_map(include_data_flow: bool = True) -> dict[str, Any]:
    """Generate a Codebase Cognition Map — a dense, model-friendly overview."""
    t0 = time.time()
    _ensure_init()

    # Check resource cache (expired entries are evicted on read)
    cached = _cache_get("cognition_map", 30)
    if cached is not None:
        return json.loads(cached)

    from codeforge_mcp.tools.cognition import cognition_map as cm
    try:
        result = await asyncio.to_thread(cm, _graph, _ast_indexer, _project_root, include_data_flow)
    except asyncio.CancelledError:
        return {"error": "Operation cancelled by client", "layers": [], "data_flows": []}

    # Cache the result
    _cache_resource("cognition_map", json.dumps(result, default=str))

    _tool_timing("cognition_map", t0, {"include_data_flow": include_data_flow})
    return result


# ── AST Dependency Graph ─────────────────────────────────────────────

@mcp.tool()
async def ast_dependency_graph(focus_file: str = "", max_files: int = 200) -> dict[str, Any]:
    """Build a module-level dependency graph from import statements."""
    t0 = time.time()
    _ensure_init()
    from codeforge_mcp.tools.dependency import ast_dependency_graph as adg
    try:
        result = await asyncio.to_thread(adg, _ast_indexer, _project_root, focus_file or None, max_files)
    except asyncio.CancelledError:
        return {"error": "Operation cancelled by client", "nodes": [], "edges": []}
    _tool_timing("ast_dependency_graph", t0, {"focus_file": focus_file, "max_files": max_files})
    return result


# ── LSP Tools ────────────────────────────────────────────────────────

@mcp.tool()
async def lsp_goto_definition(file: str, line: int, col: int = 0) -> dict[str, Any]:
    """Go to the definition of a symbol at the given file position.

    Returns ``{found: true, ...}`` with the definition location, or
    ``{found: false}`` when no definition can be resolved.

    Args:
        file: File path relative to project root.
        line: Line number (1-based).
        col: Column number (0-based).
    """
    t0 = time.time()
    _ensure_init()
    if _lsp is None:
        return {"found": False, "error": "LSP not initialized"}
    full_path = str(Path(_project_root) / file)
    try:
        result = await _lsp.goto_definition(full_path, line, col)
    except Exception as e:
        return {"found": False, "error": str(e)}
    _tool_timing("lsp_goto_definition", t0, {"file": file, "line": line, "col": col})
    if result is None:
        return {"found": False, "file": file, "line": line, "col": col}
    return {"found": True, **result}


@mcp.tool()
async def lsp_find_references(file: str, line: int, col: int = 0) -> dict[str, Any]:
    """Find all references to a symbol at the given file position.

    Args:
        file: File path relative to project root.
        line: Line number (1-based).
        col: Column number (0-based).
    """
    t0 = time.time()
    _ensure_init()
    if _lsp is None:
        return {"success": False, "references": [], "error": "LSP not initialized"}
    full_path = str(Path(_project_root) / file)
    try:
        result = await _lsp.references("", full_path, line, col)
    except Exception as e:
        return {"success": False, "references": [], "error": str(e)}
    _tool_timing("lsp_find_references", t0, {"file": file, "line": line, "col": col})
    return {"success": True, "references": result}


@mcp.tool()
async def lsp_hover(file: str, line: int, col: int = 0) -> dict[str, Any]:
    """Get hover information (type, documentation) at a file position.

    Returns ``{found: true, ...}`` with hover details, or
    ``{found: false}`` when no hover information is available.

    Args:
        file: File path relative to project root.
        line: Line number (1-based).
        col: Column number (0-based).
    """
    t0 = time.time()
    _ensure_init()
    if _lsp is None:
        return {"found": False, "error": "LSP not initialized"}
    full_path = str(Path(_project_root) / file)
    try:
        result = await _lsp.hover(full_path, line, col)
    except Exception as e:
        return {"found": False, "error": str(e)}
    _tool_timing("lsp_hover", t0, {"file": file, "line": line, "col": col})
    if result is None:
        return {"found": False, "file": file, "line": line, "col": col}
    return {"found": True, **result}


@mcp.tool()
async def lsp_diagnostics(file: str) -> dict[str, Any]:
    """Run LSP diagnostics on a file via publishDiagnostics notifications.

    Opens the file, waits for the server to publish diagnostics, returns them.
    """
    t0 = time.time()
    _ensure_init()
    if _lsp is None:
        return {"success": False, "diagnostics": [], "error": "LSP not initialized"}
    full_path = str(Path(_project_root) / file)
    try:
        result = await _lsp.diagnostics(full_path)
    except Exception as e:
        _tool_timing("lsp_diagnostics", t0, {"file": file})
        return {"success": False, "diagnostics": [], "error": str(e)}
    _tool_timing("lsp_diagnostics", t0, {"file": file})
    return {
        "success": True,
        "file": file,
        "diagnostics": result,
        "diagnostic_count": len(result),
        "summary": f"{len(result)} diagnostic issue(s) found" if result else "No diagnostic issues found",
    }


# ── File Operation Tools ─────────────────────────────────────────────

@mcp.tool()
async def read_file(path: str, start_line: int = 0, end_line: int = 0) -> dict[str, Any]:
    """Read a file from the project, optionally with line range."""
    t0 = time.time()
    _ensure_init()
    from codeforge_mcp.tools.file_ops import read_file as rf
    result = await asyncio.to_thread(rf, _project_root, path, start_line, end_line)
    _tool_timing("read_file", t0, {"path": path, "start_line": start_line, "end_line": end_line})
    return result


@mcp.tool()
async def write_file(path: str, content: str) -> dict[str, Any]:
    """Write content to a file within the project."""
    t0 = time.time()
    _ensure_init()
    existed_before, original_bytes = _capture_file_state(path)
    from codeforge_mcp.tools.file_ops import write_file as wf
    result = wf(_project_root, path, content)
    if result.get("written"):
        line_count = max(1, len(content.split("\n")))
        review_payload = await _review_edit(path, 1, line_count)
        result["review_passed"] = review_payload["review_passed"]
        result["review"] = review_payload["review"]
        if not review_payload["review_passed"]:
            reverted, rollback_message = _restore_file_state(path, existed_before, original_bytes)
            await _reindex_after_edit(path)
            result.update(_current_file_metadata(path))
            result["written"] = False
            result["reverted"] = reverted
            result["error"] = (
                "Edit reverted after post-edit review found blocking issues. "
                f"{rollback_message}"
            )
    _tool_timing("write_file", t0, {"path": path, "content_len": len(content)})
    return result


@mcp.tool()
async def list_directory(path: str = "", depth: int = 2, show_hidden: bool = False) -> dict[str, Any]:
    """List files and directories in a project subdirectory."""
    t0 = time.time()
    _ensure_init()
    from codeforge_mcp.tools.file_ops import list_directory as ld
    result = await asyncio.to_thread(ld, _project_root, path, depth, show_hidden)
    _tool_timing("list_directory", t0, {"path": path, "depth": depth, "show_hidden": show_hidden})
    return result


@mcp.tool()
async def git_diff(base: str = "HEAD", head: str = "") -> dict[str, Any]:
    """Get git diff information including both staged and unstaged changes.

    For non-git workspaces, falls back to a synthetic diff computed by
    comparing on-disk symbol hashes to the knowledge-graph's last indexed
    state (Phase 4).
    """
    t0 = time.time()
    _ensure_init()
    from codeforge_mcp.tools.file_ops import git_diff as gd
    result = await asyncio.to_thread(gd, _project_root, base, head, _graph, _ast_indexer)
    _tool_timing("git_diff", t0, {"base": base, "head": head})
    return result


# ── Execution Tools ──────────────────────────────────────────────────

@mcp.tool()
async def bash_run(cmd: str, cwd: str = "", timeout: int = 30, confirmed: bool = False) -> dict[str, Any]:
    """Run a shell command, optionally sandboxed via bwrap (set CODEFORGE_SANDBOX=1).

    Dangerous commands (rm, sudo, chmod, git push --force, etc.) require
    confirmed=true. The tool will return a warning if the command looks dangerous.

    Args:
        cmd: The shell command to run.
        cwd: Working directory relative to project root.
        timeout: Maximum execution time in seconds (default 30).
        confirmed: Set to true to confirm execution of dangerous commands.
    """
    t0 = time.time()
    _ensure_init()
    from codeforge_mcp.tools.execution import bash_run as br
    work_dir = str(Path(_project_root) / cwd) if cwd else _project_root
    result = await br(cmd, cwd=work_dir, timeout=timeout, confirmed=confirmed)
    _tool_timing("bash_run", t0, {"cmd": sanitize_command(cmd[:100]), "timeout": timeout, "confirmed": confirmed})
    return result


@mcp.tool()
async def test_run(selector: str = "", summary_only: bool = False) -> dict[str, Any]:
    """Run tests using auto-detected test runner (pytest/vitest/jest/cargo test).

    Args:
        selector: Optional test selector (file, function, or filter).
        summary_only: When True, omit per-test PASSED markers and return only
            the runner header, failure section, and final summary line. Use
            for large suites where the full stdout would consume too many
            context tokens.
    """
    t0 = time.time()
    _ensure_init()
    from codeforge_mcp.tools.execution import test_run as tr
    result = await asyncio.to_thread(tr, selector, _project_root, summary_only)
    _tool_timing("test_run", t0, {"selector": selector, "summary_only": summary_only})
    return result


# ── Memory Tools ─────────────────────────────────────────────────────

@mcp.tool()
async def decision_record(title: str, why: str, files: str = "") -> dict[str, Any]:
    """Record a design decision in .codeforge/decisions.md."""
    t0 = time.time()
    _ensure_init()
    from codeforge_mcp.tools.memory import decision_record as dr
    file_list = [f.strip() for f in files.split(",") if f.strip()] if files else []
    result = await asyncio.to_thread(dr, _project_root, _graph, title, why, file_list)
    _tool_timing("decision_record", t0, {"title": title})
    return result


# ── Subagent Tools ───────────────────────────────────────────────────

@mcp.tool()
async def spawn_subagent(
    role: str = "",
    task: str = "",
    files: str = "",
    capabilities: list[str] | None = None,
) -> dict[str, Any]:
    """Spawn a specialized subagent with a restricted, deterministic toolset.

    Roles (aliases for capability lists):
        file_finder, code_searcher, reviewer, test_impact, diagnose,
        refactor_advisor, security_auditor, doc_generator

    Capabilities:
        search, ast, symbol, lsp, graph, graph_upstream

    Args:
        role: Subagent role (preset capability list).
        task: Natural-language task description.
        files: Optional comma-separated file paths for context.
        capabilities: Explicit list of capability strings (dynamic mode).
    """
    t0 = time.time()
    _ensure_init()
    file_list = [f.strip() for f in files.split(",") if f.strip()] if files else []

    if _orchestrator is None:
        return {"role": role, "task": task, "success": False, "data": None, "error": "Orchestrator not initialized", "token_estimate": 0}

    result = await _orchestrator.spawn_subagent(
        role=role, task=task, files=file_list, capabilities=capabilities
    )

    log.subagent(result.role, task, (time.time() - t0) * 1000, result.success)
    _tool_timing("spawn_subagent", t0, {"role": result.role, "task": task[:100]})

    return {
        "role": result.role,
        "task": result.task,
        "success": result.success,
        "data": result.data,
        "error": result.error,
        "token_estimate": result.token_estimate,
    }


@mcp.tool()
async def spawn_subagents(specs: str) -> list[dict[str, Any]]:
    """Run multiple subagents in parallel and return all results.

    Args:
        specs: JSON array of {role?, task, files?, capabilities?} objects.
    """
    t0 = time.time()
    _ensure_init()

    try:
        spec_list = json.loads(specs)
    except json.JSONDecodeError as e:
        return [{"role": "error", "task": "", "success": False, "data": None, "error": f"Invalid JSON: {e}", "token_estimate": 0}]

    if not isinstance(spec_list, list):
        return [{"role": "error", "task": "", "success": False, "data": None, "error": "specs must be a JSON array", "token_estimate": 0}]

    if _orchestrator is None:
        return [{"role": "error", "task": "", "success": False, "data": None, "error": "Orchestrator not initialized", "token_estimate": 0}]

    # ── Upfront spec validation ───────────────────────────────────
    # Validate each spec before delegation so the caller gets a clear,
    # actionable error for malformed inputs instead of a confusing
    # downstream failure like "No target files available for review".
    ROLE_CAPS = {"file_finder", "code_searcher", "reviewer", "test_impact",
                 "diagnose", "refactor_advisor", "security_auditor",
                 "doc_generator", "decompose"}
    # Aliases mirror SubAgentOrchestrator.ROLE_ALIASES — kept here so that
    # the server-level validation accepts the same intuitive names that the
    # orchestrator silently rewrites later.
    ROLE_ALIASES = {
        "researcher": "code_searcher",
        "investigator": "code_searcher",
        "search": "code_searcher",
        "auditor": "security_auditor",
        "security": "security_auditor",
        "review": "reviewer",
        "doc": "doc_generator",
        "docs": "doc_generator",
        "refactor": "refactor_advisor",
        "tests": "test_impact",
        "diag": "diagnose",
    }
    # Roles that require the 'files' array to be non-empty.
    # NOTE: 'reviewer' is intentionally excluded — it has auto-gather logic
    # in _run_reviewer that discovers files when none are provided.
    FILE_REQUIRED_ROLES = {"refactor_advisor", "doc_generator",
                           "security_auditor"}

    validation_errors: list[dict[str, Any]] = []
    valid_specs: list[dict[str, Any]] = []

    for i, spec in enumerate(spec_list):
        if not isinstance(spec, dict):
            validation_errors.append({
                "role": "error", "task": "", "success": False,
                "data": None, "token_estimate": 0,
                "error": f"spec[{i}]: expected an object, got {type(spec).__name__}",
            })
            continue

        # 'task' is required
        if "task" not in spec or not isinstance(spec.get("task"), str) or not spec["task"].strip():
            validation_errors.append({
                "role": spec.get("role", "unknown"), "task": "", "success": False,
                "data": None, "token_estimate": 0,
                "error": f"spec[{i}]: 'task' is required and must be a non-empty string",
            })
            continue

        # Either 'role' or 'capabilities' must be provided
        role = spec.get("role", "")
        caps = spec.get("capabilities", [])
        if not role and not caps:
            validation_errors.append({
                "role": "unknown", "task": spec["task"], "success": False,
                "data": None, "token_estimate": 0,
                "error": f"spec[{i}]: either 'role' or 'capabilities' must be provided",
            })
            continue

        # Resolve aliases before validation so users can use intuitive
        # names like 'researcher' (→ code_searcher).
        if role in ROLE_ALIASES:
            role = ROLE_ALIASES[role]
            spec["role"] = role

        # Validate role name
        if role and role not in ROLE_CAPS:
            validation_errors.append({
                "role": role, "task": spec["task"], "success": False,
                "data": None, "token_estimate": 0,
                "error": (f"spec[{i}]: unknown role '{role}'. "
                          f"Available: {sorted(ROLE_CAPS)}. "
                          f"Aliases: {sorted(ROLE_ALIASES)}"),
            })
            continue

        # Validate 'files' type
        files_val = spec.get("files")
        if files_val is not None:
            if isinstance(files_val, str):
                # Auto-convert comma-separated strings to list
                spec["files"] = [f.strip() for f in files_val.split(",") if f.strip()]
            elif not isinstance(files_val, list):
                validation_errors.append({
                    "role": role, "task": spec["task"], "success": False,
                    "data": None, "token_estimate": 0,
                    "error": f"spec[{i}]: 'files' must be a list of strings or a comma-separated string",
                })
                continue

        # Role-specific validation: some roles require files
        effective_files = spec.get("files", [])
        if role in FILE_REQUIRED_ROLES and not effective_files:
            validation_errors.append({
                "role": role, "task": spec["task"], "success": False,
                "data": None, "token_estimate": 0,
                "error": (f"spec[{i}]: role '{role}' requires a non-empty 'files' array. "
                          f"Provide file paths relative to the project root."),
            })
            continue

        valid_specs.append(spec)

    # Run the valid specs
    out: list[dict[str, Any]] = list(validation_errors)  # start with errors

    if valid_specs:
        swarm = len(valid_specs) > 1 or any(bool(spec.get("swarm")) for spec in valid_specs)
        results = await _orchestrator.spawn_multiple(valid_specs, swarm=swarm)
        for r in results:
            log.subagent(r.role, r.task, (time.time() - t0) * 1000, r.success)
            out.append({
                "role": r.role,
                "task": r.task,
                "success": r.success,
                "data": r.data,
                "error": r.error,
                "token_estimate": r.token_estimate,
            })

    _tool_timing("spawn_subagents", t0, {"count": len(spec_list), "valid": len(valid_specs), "invalid": len(validation_errors)})
    return out


@mcp.tool()
async def patch_file_tool(
    path: str,
    start_line: int,
    end_line: int,
    new_content: str,
    expected_hash: str = "",
    validate_syntax: bool = True,
) -> dict[str, Any]:
    """Apply a line-range patch with hash verification and syntax validation."""
    _ensure_init()
    existed_before, original_bytes = _capture_file_state(path)
    from codeforge_mcp.tools.patch import patch_file as pf
    result = pf(
        project_root=_project_root,
        path=path,
        start_line=start_line,
        end_line=end_line,
        new_content=new_content,
        expected_hash=expected_hash,
        validate_syntax=validate_syntax,
    )
    if result.get("success"):
        data = result.setdefault("data", {})
        review_payload = await _review_edit(
            path=path,
            start_line=start_line,
            end_line=end_line,
            diff_preview=str(data.get("diff_preview", "")),
        )
        data["review_passed"] = review_payload["review_passed"]
        data["review"] = review_payload["review"]
        if not review_payload["review_passed"]:
            reverted, rollback_message = _restore_file_state(path, existed_before, original_bytes)
            await _reindex_after_edit(path)
            data.update(_current_file_metadata(path))
            result["success"] = False
            result["error_code"] = "VALIDATION_ERROR"
            result["error_message"] = (
                "Edit reverted after post-edit review found blocking issues. "
                f"{rollback_message}"
            )
            data["written"] = False
            data["reverted"] = reverted
    return result


@mcp.tool()
async def safe_edit_tool(
    path: str,
    start_line: int,
    end_line: int,
    new_content: str,
    expected_hash: str = "",
    run_tests: bool = False,
    run_diagnostics: bool = True,
) -> dict[str, Any]:
    """Apply a code edit with multi-stage validation pipeline."""
    _ensure_init()
    existed_before, original_bytes = _capture_file_state(path)
    from codeforge_mcp.tools.safe_edit import safe_edit as se
    result = await se(
        project_root=_project_root,
        path=path,
        start_line=start_line,
        end_line=end_line,
        new_content=new_content,
        expected_hash=expected_hash,
        run_tests=run_tests,
        run_diagnostics=run_diagnostics,
        lsp_multiplexer=_lsp,
        ast_indexer=_ast_indexer,
        graph=_graph,
    )
    if result.get("success"):
        data = result.setdefault("data", {})
        if data.get("written", False):
            review_payload = await _review_edit(
                path=path,
                start_line=start_line,
                end_line=end_line,
                diff_preview=str(data.get("diff_preview", "")),
            )
            validation_stages = data.setdefault("validation_stages", [])
            validation_stages.append(_review_stage_from_payload(review_payload))
            data["review_passed"] = review_payload["review_passed"]
            data["review"] = review_payload["review"]
            if not review_payload["review_passed"]:
                reverted, rollback_message = _restore_file_state(path, existed_before, original_bytes)
                await _reindex_after_edit(path)
                data.update(_current_file_metadata(path))
                result["success"] = False
                result["error_code"] = "VALIDATION_ERROR"
                result["error_message"] = (
                    "Edit reverted after post-edit review found blocking issues. "
                    f"{rollback_message}"
                )
                data["written"] = False
                data["reverted"] = reverted
                data["validation_passed"] = False
                data["risk_level"] = "CRITICAL"
    return result


@mcp.tool()
async def test_run_affected(
    target: str,
    file_path: str = "",
    summary_only: bool = False,
) -> dict[str, Any]:
    """Run only tests that are affected by changes to a specific symbol.

    Uses the knowledge graph's call graph to find test files that
    transitively depend on the given symbol, then runs only those tests.

    Args:
        target: Symbol name to analyze (e.g. 'process_payment').
        file_path: Optional file path to disambiguate the symbol.
        summary_only: When True, return a condensed test runner output (drops
            per-test PASSED markers).
    """
    t0 = time.time()
    _ensure_init()

    if _graph is None:
        return {"error": "Knowledge graph not initialized", "tests_run": 0}

    sym = await asyncio.to_thread(_graph.get_symbol, target, file_path or None)
    if sym is None:
        results = await asyncio.to_thread(_graph.search_symbols, target, limit=1)
        if not results:
            return {"error": f"Symbol not found: {target}", "tests_run": 0}
        sym = results[0]

    # Find test files via upstream call graph
    upstream = await asyncio.to_thread(_graph.upstream, sym["id"], depth=5)
    test_files: set[str] = set()
    root_resolved = Path(_project_root).resolve()
    for caller in upstream:
        fname = caller.get("file", "")
        if not fname:
            continue
        
        # Heuristic for test files
        p = Path(fname)
        is_test = (
            "tests" in p.parts or 
            "spec" in p.parts or
            p.name.startswith("test_") or 
            p.name.endswith("_test.py") or
            "test" in p.name.lower() and p.suffix in (".py", ".ts", ".js", ".rs", ".go") and any(x in p.name.lower() for x in ("test", "spec"))
        )
        if is_test:
            try:
                # File paths from the graph are now absolute (resolved).
                # Convert to project-relative for the test runner.
                test_files.add(str(Path(fname).resolve().relative_to(root_resolved)))
            except ValueError:
                test_files.add(fname)

    # ── Convention-based fallback (Phase 5 fix) ───────────────────
    # When the call graph has no upstream edges to test files (common
    # for entry-point / initializer functions that are called by
    # framework dispatch rather than direct source-level calls), try
    # to locate test files by naming convention:
    #   symbol in  "codeforge_mcp/server.py" → look for "tests/test_server.py"
    #   symbol in  "codeforge_mcp/indexer.py" → look for "tests/test_indexer.py"
    if not test_files:
        sym_file = sym.get("file", "")
        if sym_file:
            sym_path = Path(sym_file)
            # Derive the module basename (e.g. "server" from "server.py")
            module_stem = sym_path.stem
            # Check common test file naming patterns
            candidate_names = [
                f"test_{module_stem}.py",
                f"{module_stem}_test.py",
            ]
            for candidate in candidate_names:
                # Search in tests/ directory and project root
                for tests_dir in [root_resolved / "tests", root_resolved]:
                    test_candidate = tests_dir / candidate
                    if test_candidate.exists():
                        try:
                            test_files.add(str(test_candidate.relative_to(root_resolved)))
                        except ValueError:
                            test_files.add(str(test_candidate))

    # Fallback: if no affected tests found but the symbol has many callers,
    # it might be a generic utility where specific tests are hard to find.
    if not test_files:
        # Check total callers to estimate impact risk
        impact = await asyncio.to_thread(_graph.affected_modules, sym["id"], depth=3)
        if len(impact) > 10: # High impact fallback
             _tool_timing("test_run_affected", t0, {"target": target, "fallback": "full_suite"})
             from codeforge_mcp.tools.execution import test_run
             result = await asyncio.to_thread(test_run, "", _project_root) # Run full suite
             return {
                 "tests_run": "full_suite",
                 "test_files": [],
                 "exit_code": result.get("exit_code", 1),
                 "failures": result.get("failures", []),
                 "stdout": result.get("stdout", "")[:5000],
                 "message": f"High impact symbol ({len(impact)} modules); ran full test suite as fallback.",
             }

        _tool_timing("test_run_affected", t0, {"target": target})
        return {"tests_run": 0, "message": "No affected tests found in call graph."}

    from codeforge_mcp.tools.execution import test_run
    selector = " ".join(sorted(test_files))
    result = await asyncio.to_thread(test_run, selector, _project_root)

    _tool_timing("test_run_affected", t0, {"target": target, "files": len(test_files)})
    return {
        "tests_run": len(test_files),
        "test_files": sorted(test_files),
        "exit_code": result.get("exit_code", 1),
        "failures": result.get("failures", []),
        "stdout": result.get("stdout", "")[:5000],
    }


@mcp.tool()
async def context_budget() -> dict[str, Any]:
    """Return current context budget utilization."""
    _ensure_init()
    if _orchestrator is None:
        return {"error": "Orchestrator not initialized"}
    return _orchestrator.budget.summary()


# ── Atlas:// MCP Resources ──────────────────────────────────────────
# Resources stream indexed data directly to the client without tool calls.
# They are read-only snapshots of the current knowledge graph state.
# Cache TTL: workspace/structure and dependencies are cached for 10 seconds
# after a watcher event, since they walk the full filesystem/project.

_resource_cache: dict[str, tuple[float, str]] = {}
_cache_lock: threading.Lock = threading.Lock()

# Size limits for resource cache entries.  cognition_map can produce
# multi-MB Markdown blobs; without caps, repeated regeneration or a
# large project can blow out the process RSS.
_MAX_CACHE_VALUE_BYTES = 5 * 1024 * 1024   # 5 MB per entry
_MAX_CACHE_TOTAL_BYTES = 20 * 1024 * 1024  # 20 MB across all entries

# TTLs in seconds — the single source of truth for both cache readers
# and eviction priority in _cache_resource.
_CACHE_TTL_SECONDS: dict[str, int] = {
    "cognition_map": 30,
    "workspace/structure": 10,
    "dependencies": 10,
}


def _cache_get(key: str, default_ttl: int) -> str | None:
    """Return cached value if fresh; return None if missing or expired.

    Expired entries are removed immediately so a subsequent
    _cache_resource() failure doesn't leave stale data in the cache.
    """
    with _cache_lock:
        entry = _resource_cache.get(key)
        if entry is None:
            return None
        ts, data = entry
        ttl = _CACHE_TTL_SECONDS.get(key, default_ttl)
        if time.time() - ts < ttl:
            return data
        # Expired — remove now; _cache_resource may refuse the replacement
        del _resource_cache[key]
        return None


def _cache_resource(key: str, value: str) -> None:
    """Insert a string value into _resource_cache with size enforcement.

    - Refuses to cache entries larger than _MAX_CACHE_VALUE_BYTES.
    - Evicts oldest entries (by timestamp) until total size <= _MAX_CACHE_TOTAL_BYTES.
    - Expired entries (past their TTL) are pruned during eviction so stale
      data never crowds out fresh data.
    """
    val_size = len(value.encode("utf-8"))
    if val_size > _MAX_CACHE_VALUE_BYTES:
        return  # too large to cache at all

    now = time.time()

    with _cache_lock:
        # Avoid double-counting the key being overwritten: the old entry's
        # size will be freed when we reassign _resource_cache[key] below.
        old_size = len(_resource_cache[key][1].encode("utf-8")) if key in _resource_cache else 0

        def _sort_key(item: tuple[str, tuple[float, str]]) -> tuple[int, float]:
            _k, (_ts, _v) = item
            # Expired entries go first (0); within each group, oldest timestamp next.
            return (0 if now - _ts > _CACHE_TTL_SECONDS.get(_k, 10) else 1, _ts)

        total = val_size
        for _k, (_ts, _v) in _resource_cache.items():
            total += len(_v.encode("utf-8"))
        total -= old_size

        if total > _MAX_CACHE_TOTAL_BYTES:
            # Calculate how much space we need to free.  Only evict if we can
            # make enough room; otherwise refuse the insert and leave the
            # cache untouched (prevents data loss when the new value is too
            # large even after clearing everything else).
            needed = total - _MAX_CACHE_TOTAL_BYTES
            candidates = sorted(_resource_cache.items(), key=_sort_key)
            freed = 0
            victims: list[str] = []
            for ck, _cv in candidates:
                if ck == key:
                    continue  # don't evict the key we're about to overwrite
                if freed >= needed:
                    break
                freed += len(_cv[1].encode("utf-8"))
                victims.append(ck)

            if freed < needed:
                return  # can't free enough, refuse to cache

            for ck in victims:
                del _resource_cache[ck]
            total -= freed

        # If eviction couldn't free enough room (e.g., the new entry alone
        # exceeds the total cap), refuse to cache.
        if total > _MAX_CACHE_TOTAL_BYTES:
            return

        _resource_cache[key] = (now, value)


def _clear_resource_cache() -> None:
    """Clear resource caches that depend on file structure (called by watcher).

    Clears every key registered in _CACHE_TTL_SECONDS — add a new TTL
    entry there and it is automatically evicted on watcher events.
    """
    with _cache_lock:
        for key in _CACHE_TTL_SECONDS:
            _resource_cache.pop(key, None)

    # Also clear tsconfig path cache
    try:
        from codeforge_mcp.tools.dependency import clear_tsconfig_cache
        clear_tsconfig_cache()
    except ImportError:
        pass


@mcp.resource("atlas://workspace/structure")
def resource_workspace_structure() -> str:
    """Return the project file tree as a resource.

    Streaming resource that returns discovered source files organized
    by directory, suitable for the host to display as a tree.
    Cached for 10 seconds to avoid repeated filesystem walks.
    """
    _ensure_init()
    import json

    cached = _cache_get("workspace/structure", 10)
    if cached is not None:
        return cached

    from codeforge_mcp.indexer import discover_files

    files = discover_files(_project_root)

    # Build a directory tree
    tree: dict[str, Any] = {}
    for f in files:
        parts = f.split(os.sep)
        current = tree
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = None  # leaf

    result = json.dumps({
        "project": str(Path(_project_root).name),
        "total_files": len(files),
        "tree": tree,
    }, indent=2, default=str)
    _cache_resource("workspace/structure", result)
    return result


@mcp.resource("atlas://symbols/{language}")
def resource_symbols_by_language(language: str) -> str:
    """Return all symbols for a given language (python, javascript, rust, etc.).

    Args:
        language: Language name (e.g., 'python', 'typescript', 'rust', 'go', 'c', 'cpp').
    """
    _ensure_init()
    import json

    from codeforge_mcp.ast.indexer import EXT_TO_LANG

    # Find all file extensions that map to this language
    matching_exts = [ext for ext, lang in EXT_TO_LANG.items() if lang == language]
    if not matching_exts:
        return json.dumps({"language": language, "count": 0, "symbols": []})

    if _graph is None:
        return json.dumps({"language": language, "count": 0, "symbols": []})

    results: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    try:
        for ext in matching_exts:
            rows = _graph.conn.execute(
                "SELECT id, name, kind, file, line, signature, doc FROM symbols "
                "WHERE file GLOB ? ORDER BY file, line LIMIT 500",
                (f"*{ext}",),
            ).fetchall()
            cols = ["id", "name", "kind", "file", "line", "signature", "doc"]
            for r in rows:
                d = dict(zip(cols, r))
                if d["id"] not in seen_ids:
                    seen_ids.add(d["id"])
                    del d["id"]
                    results.append(d)
    except Exception as e:
        return json.dumps({
            "language": language,
            "count": 0,
            "symbols": [],
            "error": f"Query failed: {e}",
        }, indent=2, default=str)

    return json.dumps({
        "language": language,
        "count": len(results),
        "symbols": results[:500],
    }, indent=2, default=str)


@mcp.resource("atlas://dependencies")
def resource_dependencies() -> str:
    """Return the full module-level dependency graph.

    Streams the import graph (nodes + edges) for the entire project.
    For a focused view of one file, use the ast_dependency_graph tool instead.
    Cached for 10 seconds to avoid repeated full-project parsing.
    """
    _ensure_init()
    import json

    cached = _cache_get("dependencies", 10)
    if cached is not None:
        return cached

    from codeforge_mcp.tools.dependency import ast_dependency_graph as adg

    graph_data = adg(_ast_indexer, _project_root, focus_file=None, max_files=300)
    result = json.dumps(graph_data, indent=2, default=str)
    _cache_resource("dependencies", result)
    return result


@mcp.resource("atlas://decisions")
def resource_decisions() -> str:
    """Return recent design decisions from the knowledge graph.

    Streams the last 50 decisions, including title, date, why, and
    affected files.
    """
    _ensure_init()
    import json

    if _graph is None:
        return json.dumps([])

    decisions = _graph.recent_decisions(limit=50)
    return json.dumps(decisions, indent=2, default=str)


@mcp.resource("atlas://brief")
def resource_brief() -> str:
    """Return a brief summary of the codebase state.

    Includes file count, symbol count, knowledge score, and
    language breakdown. Same data as the brief() tool but
    available as a stateless resource.
    """
    _ensure_init()
    import json

    if _graph is None:
        return json.dumps({"file_count": 0, "symbol_count": 0, "knowledge_score": 0.0})

    brief_data = _graph.brief()

    # Add language breakdown
    from codeforge_mcp.ast.indexer import EXT_TO_LANG
    lang_counts: dict[str, int] = {}
    for ext, lang in EXT_TO_LANG.items():
        count = _graph.conn.execute(
            "SELECT COUNT(*) FROM symbols WHERE file GLOB ?", (f"*{ext}",)
        ).fetchone()[0]
        if count > 0:
            if lang not in lang_counts:
                lang_counts[lang] = 0
            lang_counts[lang] += count

    brief_data["languages"] = dict(
        sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)
    )

    return json.dumps(brief_data, indent=2, default=str)


# ── File Watcher Tools ───────────────────────────────────────────────

@mcp.tool()
async def start_watcher(debounce_ms: int = 300) -> dict[str, Any]:
    """Start the file watcher to auto-reindex changed files.

    Watches for file changes using inotify and re-indexes modified files
    in the AST indexer and knowledge graph automatically.

    Args:
        debounce_ms: Debounce delay in milliseconds (default 300).
    """
    global _watcher
    t0 = time.time()
    _ensure_init()

    if _watching_disabled:
        return {"success": False, "error": "File watching disabled at startup (--no-watch)"}

    try:
        from codeforge_mcp.watcher import FileWatcher
    except ImportError:
        return {"success": False, "error": "watchfiles not installed"}

    if _watcher is not None and _watcher.running:
        return {"success": True, "status": "already running", "stats": _watcher.stats}

    _watcher = FileWatcher(
        project_root=_project_root,
        graph=_graph,
        ast_indexer=_ast_indexer,
        lsp_multiplexer=_lsp,
        cache_clear_callback=_clear_resource_cache,
    )
    await _watcher.start(debounce_ms=debounce_ms)

    _tool_timing("start_watcher", t0, {"debounce_ms": debounce_ms})
    return {"success": True, "status": "started", "debounce_ms": debounce_ms}


@mcp.tool()
async def stop_watcher() -> dict[str, Any]:
    """Stop the file watcher if it's running."""
    global _watcher
    t0 = time.time()

    if _watcher is None or not _watcher.running:
        return {"success": True, "status": "not running"}

    await _watcher.stop()
    _tool_timing("stop_watcher", t0, {})
    return {"success": True, "status": "stopped", "stats": _watcher.stats}


@mcp.tool()
def watcher_status() -> dict[str, Any]:
    """Check if the file watcher is running and get its stats."""
    global _watcher
    if _watcher is None:
        return {"running": False, "stats": {}}
    return {"running": _watcher.running, "stats": _watcher.stats}


# ── CLI Entry Point ──────────────────────────────────────────────────

def _cleanup() -> None:
    """Synchronous cleanup: kill LSP processes, stop watcher, close graph.

    Idempotent — safe to call multiple times (atexit + signal + finally).
    Sets globals to None after cleanup so repeat calls are no-ops.
    """
    global _lsp, _graph, _watcher

    # Kill LSP child processes (prevents pyright/rust-analyzer/clangd leaks)
    if _lsp is not None:
        lsp = _lsp
        _lsp = None
        if hasattr(lsp, '_states'):
            for state in list(lsp._states.values()):
                try:
                    if state.reader_task is not None:
                        state.reader_task.cancel()
                except Exception:
                    pass
                try:
                    if state.proc is not None and state.proc.returncode is None:
                        state.proc.kill()
                except Exception:
                    pass

    # Close the knowledge graph SQLite connection
    if _graph is not None:
        graph = _graph
        _graph = None
        try:
            graph.close()
        except Exception:
            pass

    # File watcher cleanup: the watcher task dies with the process;
    # watchfiles will release inotify watches on interpreter exit.
    _watcher = None


def main() -> None:
    """Entry point: codeforge-mcp --project /path/to/project."""
    import argparse
    import atexit

    parser = argparse.ArgumentParser(
        description="Codeforge MCP Server — persistent knowledge graph for AI code navigation",
    )
    parser.add_argument(
        "--project",
        "-p",
        default=os.getcwd(),
        help="Project root directory (default: current working directory)",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force a full knowledge-graph rebuild on startup",
    )
    parser.add_argument(
        "--no-watch",
        action="store_true",
        help="Disable the file watcher (start_watcher tool returns an error)",
    )

    args = parser.parse_args()
    global _project_root, _reindex_on_startup, _watching_disabled
    
    # If --project is not provided, auto-detect root from CWD by walking up
    start_path = args.project if args.project else os.getcwd()
    _project_root = str(find_project_root(start_path))
    
    _reindex_on_startup = args.reindex
    _watching_disabled = args.no_watch

    # Register cleanup for normal exit and signals
    atexit.register(_cleanup)

    def _sig_handler(signum: int, frame: Any) -> None:
        _cleanup()
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        mcp.run()
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
