"""Main indexer — discovers files, respects .gitignore + .codeforgeignore,
parses with tree-sitter, populates the knowledge graph.

Uses `pathspec` for correct gitignore semantics (handles **, negations,
directory patterns like node_modules/, etc.).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Sequence

import pathspec


def _load_gitignore_spec(project_root: Path) -> pathspec.PathSpec:
    """Load .gitignore and .codeforgeignore patterns as a pathspec object.

    Returns a PathSpec that correctly handles all gitignore semantics:
    directory patterns (node_modules/), wildcards (*.pyc), double-star (**),
    negations (!pattern), and escaped characters.
    """
    patterns: list[str] = []

    # Always-ignore patterns
    patterns.extend([
        ".git/",
        "__pycache__/",
        "*.pyc",
        "node_modules/",
        "target/",
        "build/",
        "dist/",
        ".codeforge/",
        "*.egg-info/",
        ".venv/",
        "venv/",
        ".tox/",
        ".mypy_cache/",
        ".pytest_cache/",
        ".*/",  # hidden directories (e.g. .hidden/, .idea/, .vscode/)
        "*.so",
        "*.o",
        "*.a",
    ])

    # Read .gitignore
    gitignore = project_root / ".gitignore"
    if gitignore.exists():
        try:
            patterns.extend(
                line.strip()
                for line in gitignore.read_text().splitlines()
                if line.strip() and not line.startswith("#")
            )
        except (OSError, PermissionError):
            pass

    # Read .codeforgeignore (overrides/extensions)
    cfignore = project_root / ".codeforgeignore"
    if cfignore.exists():
        try:
            patterns.extend(
                line.strip()
                for line in cfignore.read_text().splitlines()
                if line.strip() and not line.startswith("#")
            )
        except (OSError, PermissionError):
            pass

    return pathspec.PathSpec.from_lines("gitignore", patterns)


def find_project_root(start_path: str | Path) -> Path:
    """Find the project root by walking up from start_path.

    Looks for markers like .git, pyproject.toml, package.json, etc.
    Falls back to start_path if no root is found.
    """
    markers = {
        ".git",
        ".codeforge",
        "pyproject.toml",
        "package.json",
        "Cargo.toml",
        "go.mod",
        "requirements.txt",
        "setup.py",
        "Makefile",
    }
    current = Path(start_path).resolve()
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent
    return current


def discover_files(
    project_root: str | Path,
    extensions: Sequence[str] | None = None,
) -> list[str]:
    """Walk the project tree and return supported source files.

    Respects .gitignore and .codeforgeignore using pathspec for
    correct gitignore semantics (**, negations, directory matching).

    Args:
        project_root: Project directory.
        extensions: Filter to specific extensions (from EXT_TO_LANG).

    Returns:
        Sorted list of relative file paths.
    """
    root = Path(project_root).resolve()
    spec = _load_gitignore_spec(root)

    from codeforge_mcp.ast.indexer import EXT_TO_LANG

    if extensions is None:
        extensions = list(EXT_TO_LANG.keys())

    files: list[str] = []
    for dirpath_str, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath_str)

        # Filter directories: remove ignored dirs from os.walk traversal
        # so we don't waste time walking into node_modules, etc.
        keep_dirs: list[str] = []
        for d in dirnames:
            rel_path = os.path.relpath(dirpath / d, root)
            if spec.match_file(rel_path) or spec.match_file(rel_path + "/"):
                continue
            keep_dirs.append(d)
        dirnames[:] = keep_dirs

        for fname in filenames:
            rel_path = os.path.relpath(dirpath / fname, root)

            # Check if ignored
            if spec.match_file(rel_path):
                continue

            # Check extension
            for ext in extensions:
                if fname.endswith(ext):
                    files.append(rel_path)
                    break

    return sorted(files)


def index_project(
    project_root: str | Path,
    graph: Any,
    ast_indexer: Any,
    full: bool = False,
) -> dict[str, Any]:
    """Index (or re-index) the entire project into the knowledge graph.

    Uses a two-pass strategy when full=True:
      1. Pass 1 — index all symbols (so every symbol exists in the graph).
      2. Pass 2 — re-index each file that has symbols to create call edges.
         (On the second pass symbols are unchanged so only edges are added.)

    For incremental mode (full=False) this does a single pass per file;
    cross-file call edges may be incomplete but will be filled in as
    other files are touched by the watcher.

    Args:
        project_root: Project directory.
        graph: KnowledgeGraph instance.
        ast_indexer: ASTIndexer instance.
        full: If True, do a full re-index. Otherwise, incremental.

    Returns:
        {files_indexed, symbols_added, edges_created, duration_ms, parse_errors}
    """
    start = time.time()
    files = discover_files(project_root)
    symbols_added = 0
    parse_errors: list[dict[str, str]] = []

    root = Path(project_root)

    for fpath in files:
        try:
            if full:
                count = ast_indexer.index_file(root / fpath)
            else:
                count = ast_indexer.index_file_incremental(root / fpath)
            symbols_added += count
        except Exception as e:
            parse_errors.append({"file": fpath, "error": str(e)})

    # ── Second pass for full re-index: recreate call edges ──────
    # On the first pass cross-file callees may not exist yet.
    # Running with create_edges_only=True skips wasteful UPSERTs;
    # it only looks up existing symbol IDs and creates edges.
    edges_created = 0
    if full and hasattr(ast_indexer, "_parse_and_index"):
        before_edges = graph.conn.execute(
            "SELECT COUNT(*) FROM edges"
        ).fetchone()[0]
        for fpath in files:
            try:
                ast_indexer._parse_and_index(
                    root / fpath, clear_first=False, create_edges_only=True
                )
            except Exception as e:
                parse_errors.append({"file": fpath, "error": str(e), "phase": "edges"})
                continue
        after_edges = graph.conn.execute(
            "SELECT COUNT(*) FROM edges"
        ).fetchone()[0]
        edges_created = after_edges - before_edges

    duration_ms = int((time.time() - start) * 1000)

    return {
        "files_indexed": len(files),
        "symbols_added": symbols_added,
        "edges_created": edges_created,
        "duration_ms": duration_ms,
        "parse_errors": parse_errors,
    }
