"""patch_file — line-level file editing with guardrails.

Provides a safe, deterministic file patching primitive that:
1. Verifies file hash before patching (prevents stale-data edits)
2. Validates line-range expectations (prevents off-by-one overwrites)
3. Optionally validates the result parses without syntax errors (tree-sitter)
4. Returns a structured diff of what changed

This is the PRIMARY tool weak models should use instead of write_file,
because it minimises the chance of introducing bugs through full-file overwrites.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from codeforge_mcp.tools.responses import ToolResponse, ErrorCode


_FILE_NOT_FOUND_RETRIES = 4
_FILE_NOT_FOUND_DELAY_SECONDS = 0.05


def patch_file(
    project_root: str,
    path: str,
    start_line: int,
    end_line: int,
    new_content: str,
    expected_hash: str = "",
    validate_syntax: bool = True,
) -> dict[str, Any]:
    """Apply a line-range patch to a file with safety checks.

    Args:
        project_root: Absolute project root path.
        path: File path relative to project root.
        start_line: First line to replace (1-based, inclusive).
        end_line: Last line to replace (1-based, inclusive).
        new_content: Replacement content for the specified line range.
        expected_hash: Optional xxhash of the file before patching.
                       If provided, the patch is rejected if the file
                       has been modified since the hash was obtained.
        validate_syntax: If True, validate the patched file parses
                         without tree-sitter syntax errors.

    Returns:
        ToolResponse-shaped dict with:
          - path, written, bytes_written, hash (new hash)
          - diff_preview: a unified-diff-style preview of the change
          - syntax_valid: whether the result is syntactically valid
          - diagnostics: list of syntax errors if validation was requested
    """
    root = Path(project_root).resolve()
    file_path = (root / path).resolve()

    # Security: path traversal
    if not file_path.is_relative_to(root):
        return ToolResponse.error(
            ErrorCode.FILE_TRAVERSAL_DENIED,
            f"Path traversal denied: {path}",
        ).model_dump()

    # Security: protected paths
    rel = str(file_path.relative_to(root))
    if rel.startswith(".git") or rel.startswith(".codeforge"):
        return ToolResponse.error(
            ErrorCode.FILE_PROTECTED,
            f"Cannot write to protected path: {rel}",
        ).model_dump()

    # Read current file. A short retry window makes sequential mutation
    # tools more resilient when a preceding create/write is still settling.
    if not _wait_for_file(file_path):
        return ToolResponse.error(
            ErrorCode.FILE_NOT_FOUND,
            f"File not found: {path}",
        ).model_dump()

    try:
        content_bytes = file_path.read_bytes()
    except (OSError, PermissionError) as e:
        return ToolResponse.error(
            ErrorCode.FILE_READ_ERROR,
            f"Cannot read file: {e}",
        ).model_dump()

    # Hash verification
    if expected_hash:
        import xxhash
        actual_hash = xxhash.xxh3_64(content_bytes).hexdigest()
        if actual_hash != expected_hash:
            return ToolResponse.error(
                ErrorCode.PATCH_HASH_MISMATCH,
                f"File has been modified since hash was obtained. "
                f"Expected {expected_hash}, got {actual_hash}. "
                f"Re-read the file to get the current hash.",
                current_hash=actual_hash,
            ).model_dump()

    # Decode and split lines
    try:
        content = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        content = content_bytes.decode("latin-1", errors="replace")

    lines = content.split("\n")
    total_lines = len(lines)

    # Validate line range
    if start_line < 1 or end_line < start_line:
        return ToolResponse.error(
            ErrorCode.PATCH_LINE_MISMATCH,
            f"Invalid line range: {start_line}-{end_line}. "
            f"start_line must be >= 1 and end_line >= start_line.",
        ).model_dump()

    if start_line > total_lines:
        return ToolResponse.error(
            ErrorCode.PATCH_LINE_MISMATCH,
            f"start_line {start_line} exceeds file length ({total_lines} lines).",
            total_lines=total_lines,
        ).model_dump()

    # Apply the patch
    s = start_line - 1  # 0-based
    e = min(end_line, total_lines)  # clamp to file length

    old_lines = lines[s:e]
    new_lines = new_content.split("\n")

    # Build diff preview
    diff_preview = _build_diff_preview(path, old_lines, new_lines, start_line)

    # Apply
    patched_lines = lines[:s] + new_lines + lines[e:]
    patched_content = "\n".join(patched_lines)
    patched_bytes = patched_content.encode("utf-8")

    # Syntax validation
    syntax_valid = True
    diagnostics: list[str] = []
    if validate_syntax:
        syntax_valid, diagnostics = _validate_syntax(file_path, patched_bytes)
        # Don't block the write on syntax errors — just report them.
        # The model should decide whether to proceed.

    # Write
    try:
        file_path.write_bytes(patched_bytes)
    except (OSError, PermissionError) as e:
        return ToolResponse.error(
            ErrorCode.FILE_WRITE_ERROR,
            f"Cannot write file: {e}",
        ).model_dump()

    import xxhash
    new_hash = xxhash.xxh3_64(patched_bytes).hexdigest()

    return ToolResponse.ok(
        path=path,
        written=True,
        bytes_written=len(patched_bytes),
        hash=new_hash,
        total_lines=len(patched_lines),
        lines_removed=len(old_lines),
        lines_added=len(new_lines),
        diff_preview=diff_preview,
        syntax_valid=syntax_valid,
        diagnostics=diagnostics,
    ).model_dump()


def _build_diff_preview(
    path: str,
    old_lines: list[str],
    new_lines: list[str],
    start_line: int,
) -> str:
    """Build a human-readable unified-diff-style preview."""
    lines: list[str] = []
    lines.append(f"--- a/{path}")
    lines.append(f"+++ b/{path}")
    lines.append(
        f"@@ -{start_line},{len(old_lines)} +{start_line},{len(new_lines)} @@"
    )
    for old in old_lines:
        lines.append(f"-{old}")
    for new in new_lines:
        lines.append(f"+{new}")
    return "\n".join(lines)


def _wait_for_file(
    file_path: Path,
    retries: int = _FILE_NOT_FOUND_RETRIES,
    delay_seconds: float = _FILE_NOT_FOUND_DELAY_SECONDS,
) -> bool:
    """Wait briefly for a just-created file to become visible."""
    for attempt in range(retries):
        if file_path.is_file():
            return True
        if attempt < retries - 1:
            time.sleep(delay_seconds)
    return file_path.is_file()


def _validate_syntax(file_path: Path, content_bytes: bytes) -> tuple[bool, list[str]]:
    """Validate that the patched file has no syntax errors via tree-sitter.

    Returns (is_valid, list_of_error_descriptions).
    """
    ext = file_path.suffix
    lang_map = {
        ".py": "python", ".js": "javascript", ".mjs": "javascript",
        ".ts": "typescript", ".tsx": "tsx", ".jsx": "javascript",
        ".rs": "rust", ".go": "go",
        ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp",
    }
    lang_name = lang_map.get(ext)
    if lang_name is None:
        return True, []  # unsupported language — skip validation

    try:
        import tree_sitter
        # Dynamically load tree-sitter language grammar
        parser = tree_sitter.Parser()
        try:
            lang_mod = __import__(f"tree_sitter_{lang_name}")
            lang = tree_sitter.Language(lang_mod.language())
        except (ImportError, AttributeError):
            return True, []  # grammar not installed — skip
        parser.language = lang
        tree = parser.parse(content_bytes)
        errors = _collect_errors(tree.root_node, max_errors=10)
        return len(errors) == 0, errors
    except Exception:
        return True, []  # if tree-sitter isn't available, skip


def _collect_errors(node: Any, max_errors: int = 10) -> list[str]:
    """Walk the tree and collect ERROR/MISSING node descriptions."""
    errors: list[str] = []

    def walk(n: Any) -> None:
        if len(errors) >= max_errors:
            return
        if n.type == "ERROR" or n.is_missing:
            line = n.start_point[0] + 1
            col = n.start_point[1]
            text = n.text.decode(errors="replace")[:80] if n.text else ""
            kind = "syntax error" if n.type == "ERROR" else "missing node"
            errors.append(f"Line {line}:{col} — {kind}: {text}")
        for child in n.children:
            walk(child)

    walk(node)
    return errors
