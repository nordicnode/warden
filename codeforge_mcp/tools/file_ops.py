"""Direct file operation tools — read_file, write_file, list_directory, git_diff.

These are the fundamental primitives that let an AI agent view and edit code.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any


_SKIPPED_LIST_DIRS: frozenset[str] = frozenset({
    "__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox", ".eggs", ".codeforge", ".gemini", ".venv", "node_modules",
})
_SKIPPED_LIST_SUFFIXES: frozenset[str] = frozenset({".pyc", ".pyo", ".pyd"})


def read_file(
    project_root: str,
    path: str,
    start_line: int = 0,
    end_line: int = 0,
) -> dict[str, Any]:
    """Read a file from the project, optionally with line range.

    Args:
        project_root: Absolute project root path.
        path: File path relative to project root.
        start_line: Starting line (1-based, inclusive). 0 = from beginning.
        end_line: Ending line (1-based, inclusive). 0 = to end.

    Returns:
        {path, content, total_lines, start_line, end_line, hash}
    """
    root = Path(project_root).resolve()
    file_path = (root / path).resolve()

    # Security: ensure the file is within the project root
    if not file_path.is_relative_to(root):
        return {"error": f"Path traversal denied: {path}", "content": "", "total_lines": 0}

    if not file_path.is_file():
        return {"error": f"File not found: {path}", "content": "", "total_lines": 0}

    try:
        content_bytes = file_path.read_bytes()
        # Try UTF-8 first, fall back to latin-1
        try:
            content = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content = content_bytes.decode("latin-1", errors="replace")

        import xxhash
        file_hash = xxhash.xxh3_64(content_bytes).hexdigest()

        lines = content.split("\n")
        total_lines = len(lines)

        if start_line > 0 or end_line > 0:
            s = max(0, start_line - 1) if start_line > 0 else 0
            e = min(len(lines), end_line) if end_line > 0 else len(lines)
            lines = lines[s:e]

        return {
            "path": path,
            "content": "\n".join(lines),
            "total_lines": total_lines,
            "start_line": start_line or 1,
            "end_line": end_line or total_lines,
            "hash": file_hash,
        }
    except Exception as e:
        return {"error": str(e), "content": "", "total_lines": 0}


def write_file(
    project_root: str,
    path: str,
    content: str,
    create_dirs: bool = True,
    validate_syntax: bool = True,
) -> dict[str, Any]:
    """Write content to a file within the project.

    Args:
        project_root: Absolute project root path.
        path: File path relative to project root.
        content: File content to write.
        create_dirs: Create parent directories if they don't exist.
        validate_syntax: If True (default), run tree-sitter syntax
            validation for supported code files before writing.
            Warnings are returned but the write is NOT blocked.

    Returns:
        {path, written, bytes_written, hash, syntax_warnings?}
    """
    root = Path(project_root).resolve()
    file_path = (root / path).resolve()

    # Security: ensure the file is within the project root
    if not file_path.is_relative_to(root):
        return {"error": f"Path traversal denied: {path}", "written": False}

    # Don't write to .git or .codeforge internals
    rel = str(file_path.relative_to(root))
    rel_parts = Path(rel).parts
    if rel_parts and rel_parts[0] in (".git", ".codeforge"):
        return {"error": f"Cannot write to protected path: {rel}", "written": False}

    # ── Escape warning: detect high density of literal \n sequences ─
    # This catches the case where a client sends JSON with double backslash-n
    # (e.g. "def test():\n pass\n") instead of actual newlines.
    literal_backslash_n_count = content.count("\\n")
    if literal_backslash_n_count > 2:
        # Heuristic: if more than 2 occurrences of \n and the content
        # looks like code (has at least one colon), warn the user.
        has_code_chars = any(c in content for c in ":{}()=#")
        if has_code_chars:
            # Return a warning but still proceed with the write.
            # Use a sentinel that callers can check.
            _escape_warn = (
                f"Content contains {literal_backslash_n_count} literal "
                f"'\\n' sequences. Did you forget to use actual newlines? "
                f"The file will be written as-is."
            )
        else:
            _escape_warn = None
    else:
        _escape_warn = None

    # ── Optional syntax validation (non-blocking) ─────────────────
    syntax_warnings: list[str] = []
    if validate_syntax:
        try:
            from codeforge_mcp.ast.indexer import _supported_ext, _get_language, _ensure_tree_sitter
            lang_name = _supported_ext(file_path)
            if lang_name is not None:
                language = _get_language(lang_name)
                if language is not None:
                    ts = _ensure_tree_sitter()
                    parser = ts.Parser()
                    parser.language = language
                    tree = parser.parse(content.encode("utf-8"))
                    # Walk root children for ERROR nodes
                    _collect_errors(tree.root_node, syntax_warnings, max_errors=5)
        except Exception:
            # Don't let validation failure block the write
            pass

    try:
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        content_bytes = content.encode("utf-8")
        file_path.write_bytes(content_bytes)

        import xxhash
        file_hash = xxhash.xxh3_64(content_bytes).hexdigest()

        result: dict[str, Any] = {
            "path": path,
            "written": True,
            "bytes_written": len(content_bytes),
            "hash": file_hash,
        }
        if syntax_warnings:
            result["syntax_warnings"] = syntax_warnings
        if _escape_warn:
            result["escape_warning"] = _escape_warn
        return result
    except Exception as e:
        return {"error": str(e), "written": False}


def _collect_errors(node: Any, errors: list[str], max_errors: int = 5) -> None:
    """Walk tree-sitter AST and collect ERROR / MISSING node messages."""
    if len(errors) >= max_errors:
        return
    if node.type == "ERROR" or node.is_missing:
        line = node.start_point[0] + 1
        col = node.start_point[1]
        kind = "syntax error" if node.type == "ERROR" else "missing token"
        errors.append(f"Line {line}:{col}: {kind}")
    for child in node.children:
        _collect_errors(child, errors, max_errors)


def list_directory(
    project_root: str,
    path: str = "",
    depth: int = 2,
    show_hidden: bool = False,
) -> dict[str, Any]:
    """List files and directories in a project subdirectory.

    Args:
        project_root: Absolute project root path.
        path: Directory path relative to project root ("" for root).
        depth: Maximum recursion depth (1 = direct children only).
        show_hidden: Whether to include hidden files/dirs.

    Returns:
        {path, files: [{name, path, type: "file"|"dir", size}], total_files, total_dirs}
    """
    root = Path(project_root).resolve()
    target = (root / path).resolve() if path else root

    # Security
    if not target.is_relative_to(root):
        return {"error": f"Path traversal denied: {path}", "files": []}

    if not target.exists():
        return {"error": f"Directory not found: {path}", "files": []}

    files: list[dict[str, Any]] = []
    seen: set[str] = set()  # Full path dedup
    total_files = 0
    total_dirs = 0

    def _skip_entry(entry: Path) -> bool:
        if entry.name in _SKIPPED_LIST_DIRS:
            return True
        if entry.suffix.lower() in _SKIPPED_LIST_SUFFIXES:
            return True
        return False

    def _walk(current: Path, current_depth: int) -> None:
        nonlocal total_files, total_dirs
        if current_depth > depth:
            return
        try:
            entries = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return
        for entry in entries:
            # Resolve symlinks and verify the entry stays within the project root.
            # This prevents reporting file metadata or descending into directories
            # that live outside the project via symlink traversal.
            try:
                resolved = entry.resolve()
            except (OSError, RuntimeError) as e:
                # Could not resolve (broken symlink, permission error, etc.) — skip.
                logging.getLogger(__name__).debug(
                    "Skipping unresolvable entry %s: %s", entry, e
                )
                continue

            # Verify the resolved path is inside the project root.
            # This also catches symlinks that escape via '..' components or
            # symlinks to absolute paths outside the root.
            try:
                resolved.relative_to(root)
            except ValueError:
                # Symlink resolves outside project root — skip it and log.
                logging.getLogger(__name__).info(
                    "Skipping symlink that escapes project root: %s -> %s",
                    entry, resolved,
                )
                continue

            # Use the original (unresolved) entry path for dedup, so that
            # in-project symlinks to files are listed as their own entry even
            # if they resolve to the same on-disk inode as an existing file.
            # The traversal check above already ensured `resolved` is safe.
            entry_key = str(entry)
            if entry_key in seen:
                continue
            seen.add(entry_key)
            name = entry.name
            if not show_hidden and name.startswith("."):
                continue
            if _skip_entry(entry):
                continue
            # Check type via the resolved (safe) path to correctly handle symlinks.
            is_dir = resolved.is_dir()
            is_file = resolved.is_file()
            if is_dir:
                total_dirs += 1
                files.append({
                    "name": name,
                    "path": str(entry.relative_to(root)),
                    "type": "dir",
                    "size": 0,
                })
                if current_depth < depth:
                    _walk(entry, current_depth + 1)
            elif is_file:
                total_files += 1
                try:
                    size = resolved.stat().st_size
                except OSError:
                    size = 0
                files.append({
                    "name": name,
                    "path": str(entry.relative_to(root)),
                    "type": "file",
                    "size": size,
                })

    if target.is_dir():
        _walk(target, 1)
        files = files[:200]

    return {
        "path": path or ".",
        "files": files[:200],
        "total_files": total_files,
        "total_dirs": total_dirs,
    }


def git_diff(
    project_root: str,
    base: str = "HEAD",
    head: str = "",
    graph: Any = None,
    ast_indexer: Any = None,
) -> dict[str, Any]:
    """Get git diff information including both staged and unstaged changes.

    When called with defaults (base="HEAD", head=""):
    - Returns both unstaged changes (git diff) and staged changes (git diff --cached)
    - Shows which files are changed, added, or deleted

    If the project is not a git repository AND a knowledge graph + ast indexer
    are supplied, returns a "synthetic diff" computed by re-parsing each
    indexed file and comparing per-symbol hashes against the graph's last
    indexed state (Phase 4 fix — provides diff-like functionality in
    git-less workspaces).

    Args:
        project_root: Absolute project root path.
        base: Base ref (default: "HEAD" to diff against working tree).
        head: Optional head ref for explicit comparison.
        graph: Optional KnowledgeGraph for synthetic-diff fallback.
        ast_indexer: Optional ASTIndexer for re-parsing files.

    Returns:
        {files_changed, staged_changes, unstaged_changes, diff_summary, raw_diff (truncated)}
    """
    root = Path(project_root)

    # Check if it's a valid git repo — always verify with rev-parse
    # regardless of whether .git directory exists, because an empty or
    # corrupted .git directory (e.g. from a prior failed init) will
    # pass the existence check but break all git commands.
    is_git_repo = False
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=5
        )
        is_git_repo = proc.returncode == 0 and proc.stdout.strip() == "true"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        is_git_repo = False

    if not is_git_repo:
        # Diagnose *why* git rejected the directory so callers (and humans
        # reading logs) can immediately see the root cause instead of
        # silently falling back to a synthetic diff that says "no changes
        # detected" — which is what happened in our E2E test against an
        # empty .git directory.
        git_dir = root / ".git"
        if git_dir.exists():
            try:
                contents = list(git_dir.iterdir()) if git_dir.is_dir() else []
            except OSError:
                contents = []
            if not contents:
                git_status = "empty_git_directory"
                git_status_note = (
                    f"{git_dir} exists but is empty — git considers this "
                    "*not* a repository. Run `git init` to (re)initialise."
                )
            else:
                git_status = "broken_git_directory"
                git_status_note = (
                    f"{git_dir} exists but `git rev-parse --is-inside-work-tree` "
                    "failed. The repository may be corrupted; try `git fsck` or "
                    "re-initialise with `git init`."
                )
        else:
            git_status = "no_git_directory"
            git_status_note = (
                f"No .git directory found at {root}. Run `git init` to enable "
                "git-aware diffs."
            )

        # Phase 4: synthetic diff against the knowledge graph.
        # Returns added/removed/modified symbols compared to last index.
        if graph is not None and ast_indexer is not None:
            synthetic = _synthetic_diff(root, graph, ast_indexer)
            synthetic["git_status"] = git_status
            synthetic["git_status_note"] = git_status_note
            return synthetic
        return {
            "error": "No git repository found and no knowledge graph available for synthetic diff.",
            "note": "Initialize a git repo (git init) or ensure the server has indexed the project for synthetic diff support.",
            "git_status": git_status,
            "git_status_note": git_status_note,
            "files_changed": [],
            "staged_changes": [],
            "unstaged_changes": [],
            "diff_summary": "This project is not a git repository and no indexed state is available for comparison.",
            "raw_diff": "",
        }

    def _run_git(args: list[str]) -> tuple[str, str]:
        """Run git command, return (stdout, stderr)."""
        try:
            proc = subprocess.run(
                ["git", "-C", str(root)] + args,
                capture_output=True, text=True, timeout=15,
            )
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, ["git"] + args, output=proc.stdout, stderr=proc.stderr)
            return proc.stdout, proc.stderr
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return "", ""

    files_changed: list[str] = []
    staged_files: list[str] = []
    unstaged_files: list[str] = []
    raw_diff = ""
    summary = ""

    unstaged_stat = ""
    staged_stat = ""

    if head:
        # Explicit comparison between two refs
        stat_out, _ = _run_git(["diff", "--stat", base, head])
        summary = stat_out.strip()
        raw_diff, _ = _run_git(["diff", base, head])
    elif base != "HEAD":
        # Explicit ref → working tree diff
        stat_out, _ = _run_git(["diff", "--stat", base])
        summary = stat_out.strip()
        raw_diff, _ = _run_git(["diff", base])
    else:
        # Default: show both staged and unstaged changes
        # Unstaged (working tree vs index)
        unstaged_stat, _ = _run_git(["diff", "--stat"])
        # Staged (index vs HEAD)
        staged_stat, _ = _run_git(["diff", "--staged", "--stat"])

        if unstaged_stat.strip():
            summary += f"── Unstaged changes ──\n{unstaged_stat.strip()}\n"
        if staged_stat.strip():
            summary += f"── Staged changes ──\n{staged_stat.strip()}\n"

        if not summary:
            stub_out, _ = _run_git(["status", "--short"])
            if stub_out.strip():
                summary = f"── Working tree status ──\n{stub_out.strip()}"

        # Raw diff: combine unstaged + staged, truncated
        unstaged_raw, _ = _run_git(["diff"])
        staged_raw, _ = _run_git(["diff", "--staged"])
        raw_diff = (unstaged_raw + staged_raw)[:20000]

    # Parse file list from summary
    for line in summary.split("\n"):
        line = line.strip()
        if line and "|" in line and not line.startswith("──"):
            fname = line.split("|")[0].strip()
            files_changed.append(fname)

    # Parse staged vs unstaged files
    if unstaged_stat:
        for line in unstaged_stat.split("\n"):
            line = line.strip()
            if line and "|" in line:
                fname = line.split("|")[0].strip()
                unstaged_files.append(fname)
    if staged_stat:
        for line in staged_stat.split("\n"):
            line = line.strip()
            if line and "|" in line:
                fname = line.split("|")[0].strip()
                staged_files.append(fname)

    return {
        "files_changed": files_changed,
        "staged_changes": staged_files,
        "unstaged_changes": unstaged_files,
        "diff_summary": summary[:5000],
        "raw_diff": raw_diff,
    }


def _synthetic_diff(root: Path, graph: Any, ast_indexer: Any) -> dict[str, Any]:
    """Synthesise a diff by comparing on-disk file state to the knowledge graph.

    For each file that has indexed symbols in the graph:
      - Re-parses the current file (if it still exists) to extract symbols.
      - Compares the per-symbol hash to the hash stored in the graph at
        index time.
      - Reports added / removed / modified symbols per file.

    Files that are in the graph but missing on disk are reported as deleted.

    Args:
        root: Project root.
        graph: KnowledgeGraph instance (must expose `conn` + `hashes_for_file`).
        ast_indexer: ASTIndexer instance (must expose `parse_file` or use the
                     module-level `parse_file`).

    Returns:
        Same shape as `git_diff`, with `synthetic=True`.
    """
    from codeforge_mcp.ast.indexer import parse_file  # local import — heavy module

    files_changed: list[str] = []
    per_file_diffs: list[str] = []
    raw_lines: list[str] = []

    # Distinct files known to the graph
    indexed_files: list[str] = [
        row[0] for row in graph.conn.execute(
            "SELECT DISTINCT file FROM symbols"
        ).fetchall()
    ]

    for file in indexed_files:
        file_path = Path(file)
        if not file_path.is_absolute():
            file_path = root / file
        indexed_hashes = graph.baseline_hashes_for_file(file)

        # Fall back to live hashes if no baseline has been captured yet
        # (e.g. the project was never manually reindexed).
        if not indexed_hashes:
            indexed_hashes = graph.hashes_for_file(file)

        if not file_path.exists():
            files_changed.append(file)
            per_file_diffs.append(f"D {file}  ({len(indexed_hashes)} symbols deleted)")
            raw_lines.append(f"--- a/{file}\n+++ /dev/null  (file deleted)")
            continue

        try:
            current_symbols = parse_file(file_path)
        except Exception as exc:
            per_file_diffs.append(f"? {file}  (parse error: {exc})")
            continue

        current_hashes: dict[str, str] = {
            str(s["name"]): str(s["hash"]) for s in current_symbols if s.get("name")
        }

        added = sorted(set(current_hashes) - set(indexed_hashes))
        removed = sorted(set(indexed_hashes) - set(current_hashes))
        modified = sorted(
            name for name in set(current_hashes) & set(indexed_hashes)
            if current_hashes[name] != indexed_hashes[name]
        )

        if not (added or removed or modified):
            continue

        files_changed.append(file)
        parts: list[str] = []
        if added:
            parts.append(f"+{len(added)}")
        if removed:
            parts.append(f"-{len(removed)}")
        if modified:
            parts.append(f"~{len(modified)}")
        per_file_diffs.append(f"M {file}  symbols: {' '.join(parts)}")

        diff_block = [f"--- a/{file}  (indexed)", f"+++ b/{file}  (current)"]
        for name in added:
            diff_block.append(f"+ symbol added:    {name}")
        for name in removed:
            diff_block.append(f"- symbol removed:  {name}")
        for name in modified:
            diff_block.append(f"~ symbol changed:  {name}")
        raw_lines.append("\n".join(diff_block))

    # Files on disk but not in graph → new files.  Use `discover_files`
    # for complete project coverage (not just siblings of indexed files)
    # so files created in new directories or at the project root are
    # detected.  Skip files that parse to 0 symbols (e.g. empty
    # __init__.py) — they are known-empty, not genuinely "new" (Phase 4
    # fix — eliminates false "new file" reports).
    indexed_file_set = {Path(f).resolve() if Path(f).is_absolute() else (root / f).resolve()
                        for f in indexed_files}

    try:
        from codeforge_mcp.indexer import discover_files as _discover
        all_project_files = _discover(root)
    except Exception:
        all_project_files = []

    new_files: list[str] = []
    for rel in all_project_files:
        abs_path = (root / rel).resolve()
        if abs_path in indexed_file_set:
            continue
        # Guard: only flag as "new" if the file actually contains symbols.
        # Empty __init__.py, stub files, etc. produce 0 symbols and
        # should not be reported — they were likely present during the
        # last index but were rightfully skipped.
        try:
            parsed = parse_file(abs_path)
            if not parsed:
                continue
        except Exception:
            continue
        new_files.append(rel)

    for nf in sorted(set(new_files)):
        files_changed.append(nf)
        per_file_diffs.append(f"A {nf}  (new file, not yet indexed)")
        raw_lines.append(f"--- /dev/null\n+++ b/{nf}  (new file)")

    summary_lines = ["── Synthetic diff (vs. last indexed state) ──"]
    if per_file_diffs:
        summary_lines.extend(per_file_diffs)
    else:
        summary_lines.append("No changes detected since last indexing.")

    return {
        "files_changed": files_changed,
        "staged_changes": [],
        "unstaged_changes": files_changed,
        "diff_summary": "\n".join(summary_lines)[:5000],
        "raw_diff": "\n\n".join(raw_lines)[:20000],
        "synthetic": True,
    }
