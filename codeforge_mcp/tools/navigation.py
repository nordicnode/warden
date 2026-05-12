"""Navigation tools — code_find_files, code_search, symbol_lookup."""

from __future__ import annotations

import subprocess
import os
import re
from pathlib import Path
from typing import Any


# Directories that are never source code — skipped in glob fallback AND
# passed as exclusion globs to external `rg` and `fd` commands so the same
# rules apply across all search paths (Phase 1 fix).
_SKIPPED_DIRS: frozenset[str] = frozenset({
    ".git", "node_modules", ".venv", "venv", "env", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".eggs",
    "dist", "build", "target", ".next", ".turbo",
    ".expo", ".docusaurus", "coverage", ".cache",
    ".idea", ".vscode", ".gradle", ".svelte-kit",
    # Project-specific hidden dirs the indexer already skips via ".*/ "
    ".codeforge", ".exoskeleton", ".gemini",
})

# File extensions that are binary or build-artefacts and should never
# appear in code search / file-finder output (Phase 2 fix).
_SKIPPED_FILE_GLOBS: tuple[str, ...] = (
    "*.pyc", "*.pyo", "*.pyd",
    "*.o", "*.a", "*.so", "*.dylib", "*.dll", "*.exe",
    "*.class", "*.jar",
    "*.wasm",
    "*.min.js", "*.min.css",
    "*.lock",
)

_SOURCE_FILE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".pyi",
    ".js", ".jsx", ".mjs",
    ".ts", ".tsx",
    ".rs", ".go",
    ".c", ".h", ".cpp", ".cc", ".hpp",
    ".sh", ".bash", ".zsh",
    ".json", ".toml", ".yaml", ".yml", ".ini",
    ".sql",
})

_SOURCE_FILENAMES: frozenset[str] = frozenset({
    "Dockerfile",
    "Makefile",
    "Justfile",
    "CMakeLists.txt",
})


def _ignore_globs_for_rg() -> list[str]:
    """Build --glob '!pattern' arguments for ripgrep covering skipped dirs/files.

    Also adds a blanket hidden-directory exclusion (``!.*/**``) to match
    the indexer's ``.*/ `` gitignore pattern.  This prevents rg from
    crawling into ``.exoskeleton``, ``.gemini``, ``.codeforge``, etc.
    and timing out on large hidden trees.
    """
    args: list[str] = []
    # Blanket hidden-directory exclusion (matches the indexer's ".*/ " rule)
    args.extend(["--glob", "!.*"])
    for d in sorted(_SKIPPED_DIRS):
        args.extend(["--glob", f"!{d}/", "--glob", f"!**/{d}/**"])
    for g in _SKIPPED_FILE_GLOBS:
        args.extend(["--glob", f"!{g}"])
    # Extra safety: .venv is a common large directory that may not be in .gitignore
    args.extend(["--glob", "!.venv/**"])
    return args


def _ignore_globs_for_fd() -> list[str]:
    """Build --exclude arguments for fd covering skipped dirs/files."""
    args: list[str] = []
    for d in sorted(_SKIPPED_DIRS):
        args.extend(["--exclude", d])
    for g in _SKIPPED_FILE_GLOBS:
        args.extend(["--exclude", g])
    return args


def _is_skipped_file(path: str | Path) -> bool:
    """True if the file looks like a binary/build artefact we want to filter out."""
    p = Path(path)
    name = p.name
    suffix = p.suffix.lower()
    if suffix in {".pyc", ".pyo", ".pyd", ".o", ".a", ".so", ".dylib",
                  ".dll", ".exe", ".class", ".jar", ".wasm"}:
        return True
    if name.endswith(".min.js") or name.endswith(".min.css"):
        return True
    return False


def _is_source_text_file(path: str | Path) -> bool:
    """True when the path looks like source code or source-adjacent text."""
    p = Path(path)
    if _in_skipped_dir(p) or _is_skipped_file(p):
        return False
    return p.suffix.lower() in _SOURCE_FILE_EXTENSIONS or p.name in _SOURCE_FILENAMES


def code_find_files(project_root: str | Path, pattern: str, file_type: str = "") -> list[dict[str, Any]]:
    """Find files by name pattern using fd or fallback glob.

    Args:
        project_root: Absolute path to the project.
        pattern: File name pattern (glob or substring).
        file_type: Optional file extension filter, e.g. ".py".

    Returns:
        List of {path, size, language}.
    """
    root = Path(project_root)
    results: list[dict] = []

    # Try fd
    try:
        cmd = ["fd", "--max-results", "50"]
        # Apply standard exclusions so dependency / build directories don't
        # blow up the result set or cause timeouts in the absence of a
        # .gitignore (Phase 1 fix).
        cmd.extend(_ignore_globs_for_fd())
        if file_type:
            cmd.extend(["-e", file_type.lstrip(".")])
        cmd.append(pattern)
        proc = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(root), timeout=5
        )
        if proc.returncode == 0:
            for line in proc.stdout.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                p = root / line
                if not p.is_file():
                    continue
                if _in_skipped_dir(p) or _is_skipped_file(p):
                    continue
                size = p.stat().st_size
                lang = _language_from_ext(p.suffix)
                results.append({"path": str(p), "size": size, "language": lang})
            return results[:50]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: glob, skipping well-known non-source directories
    glob_pattern = f"**/*{pattern}*"
    if file_type:
        glob_pattern += f".{file_type.lstrip('.')}"
    for p in root.rglob(glob_pattern):
        if not p.is_file():
            continue
        if _in_skipped_dir(p) or _is_skipped_file(p):
            continue
        size = p.stat().st_size
        lang = _language_from_ext(p.suffix)
        results.append({"path": str(p), "size": size, "language": lang})
        if len(results) >= 50:
            break
    return results


def code_search(
    project_root: str | Path,
    query: str,
    regex: bool = False,
    context: int = 3,
) -> list[dict[str, Any]]:
    """Search code using ripgrep, returning matches with context.

    Args:
        project_root: Absolute path to the project.
        query: Search term or regex.
        regex: Whether to treat query as regex.
        context: Lines of context to include.

    Returns:
        List of {file, line, column, match, context_lines}.
    """
    root = Path(project_root)
    results: list[dict] = []

    cmd = [
        "rg",
        "--line-number",
        "--no-heading",
        "--color=never",
        f"--context={context}",
        "--max-count=15",
        "--max-filesize=1M",  # skip large files that slow down rg
    ]
    # Apply default exclusions so a missing .gitignore doesn't push rg into
    # .venv/node_modules/.git and time out (Phase 1 fix).
    cmd.extend(_ignore_globs_for_rg())
    if not regex:
        cmd.append("--fixed-strings")
    cmd.append("--")
    cmd.append(query)
    # Explicit search path prevents ripgrep from blocking on stdin when
    # run as a child process (MCP server) instead of searching the filesystem.
    cmd.append(".")

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(root), timeout=15
        )
        output = proc.stdout.strip()
        if not output:
            # ripgrep returns exit code 1 for "no matches" — that is not an error.
            # Exit code 2+ means an actual failure (bad regex, permission, etc.)
            if proc.returncode >= 2:
                return [{
                    "file": "",
                    "line": 0,
                    "text": f"ripgrep error (exit {proc.returncode}): {proc.stderr.strip()[:200]}",
                    "is_match": False,
                    "_error": True,
                }]
            # ── Fuzzy fallback: when exact search returns no matches, try
            # with --smart-case and partial word matching so minor typos
            # and nomenclature mismatches (e.g. "CodeForgeServer" vs "FastMCP")
            # still surface useful context (Phase 5 fix).
            if not regex:
                fuzzy_cmd = [
                    "rg",
                    "--line-number",
                    "--no-heading",
                    "--color=never",
                    f"--context={context}",
                    "--max-count=15",
                    "--max-filesize=1M",
                    "--smart-case",
                ]
                fuzzy_cmd.extend(_ignore_globs_for_rg())
                # Build a pattern that matches any word of the query as a
                # substring (e.g. "codeforge" matches "codeforge_mcp").
                # Surround each escaped word with .*? for non-greedy partial match.
                words = [w.strip() for w in query.split() if w.strip()]
                if words:
                    fuzzy_pattern = "|".join(f".*?{re.escape(w)}.*?" for w in words)
                    fuzzy_cmd.append("--")
                    fuzzy_cmd.append(fuzzy_pattern)
                    fuzzy_cmd.append(".")
                    try:
                        proc2 = subprocess.run(
                            fuzzy_cmd, capture_output=True, text=True, cwd=str(root), timeout=15
                        )
                        if proc2.stdout.strip():
                            output = proc2.stdout.strip()
                        elif proc2.returncode >= 2:
                            return [{
                                "file": "",
                                "line": 0,
                                "text": f"ripgrep error (exit {proc2.returncode}): {proc2.stderr.strip()[:200]}",
                                "is_match": False,
                                "_error": True,
                            }]
                    except (FileNotFoundError, subprocess.TimeoutExpired):
                        pass
            if not output:
                return results

        # Parse ripgrep output
        current_file = ""
        current_lines: list[dict] = []
        prefix_re = re.compile(r'^([^:]+):(\d+):(.*)')
        prefix_ctx_re = re.compile(r'^([^-]+)-(\d+)-(.*)')
        for line in output.split("\n"):
            if line == "--":
                if current_lines:
                    results.extend(current_lines)
                    current_lines = []
                continue

            # Match pattern: file:line:text (match) or file-line-text (context)
            # Ripgrep uses ':' for matches and '-' for context.
            m = prefix_re.match(line)
            if m:
                fname, lnum, text = m.group(1), m.group(2), m.group(3)
                is_match = True
            else:
                m = prefix_ctx_re.match(line)
                if m:
                    fname, lnum, text = m.group(1), m.group(2), m.group(3)
                    is_match = False
                else:
                    continue

            if current_file and fname != current_file:
                results.extend(current_lines)
                current_lines = []

            current_file = fname
            current_lines.append({
                "file": fname,
                "line": int(lnum),
                "text": text,
                "is_match": is_match,
                "match_type": "match" if is_match else "context",
            })
        results.extend(current_lines)

    except FileNotFoundError:
        return [{
            "file": "",
            "line": 0,
            "text": "ripgrep (rg) not found on PATH. Install ripgrep for code search.",
            "is_match": False,
            "_error": True,
        }]
    except subprocess.TimeoutExpired:
        return [{
            "file": "",
            "line": 0,
            "text": f"ripgrep timed out after 15s for query: {query[:100]}",
            "is_match": False,
            "_error": True,
        }]

    # Drop non-source text and skipped artefacts defensively so transcript
    # files such as results.md do not pollute code-centric search results.
    results = [r for r in results if _is_source_text_file(r["file"])]

    # BM25-like re-rank: prioritize exact matches and shorter paths
    query_lower = query.lower()
    # When regex=True, query may contain regex metacharacters that won't
    # match text literally.  Extract plain words (alphanumeric + underscore)
    # from the query for scoring so regex searches also get relevance scores.
    if regex:
        score_words = [w for w in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', query) if len(w) >= 2]
        if not score_words:
            score_words = [query_lower]
    else:
        score_words = query_lower.split()
    for r in results:
        text_lower = r["text"].lower()
        score = 0
        # Exact match bonus
        if query_lower in text_lower:
            score += 5
        # Partial match
        for word in score_words:
            word_lower = word.lower()
            if word_lower in text_lower:
                score += 1
        r["score"] = score

    results.sort(key=lambda r: r.get("score", 0), reverse=True)
    return results[:50]


async def symbol_lookup(lsp_multiplexer: Any, graph: Any, name: str) -> dict[str, Any] | None:
    """Look up a symbol via LSP workspace/symbol, falling back to the graph.

    Returns dict with name, kind, file, line, or None.

    If LSP returns a non-exact-name match (e.g. ``ast_indexer`` for the
    query ``ASTIndexer``), we additionally consult the knowledge graph for
    an exact match and prefer it when found — the graph stores indexed
    definitions verbatim and is the authoritative source for "did this
    exact symbol exist?".
    """
    # Try LSP first
    lsp_result = await lsp_multiplexer.symbol_lookup(name)
    if lsp_result and lsp_result.get("name") == name:
        return lsp_result

    # Either LSP returned nothing, or returned an inexact match — consult
    # the graph for an exact-name hit.
    sym = graph.get_symbol(name)
    if sym:
        return {
            "name": sym["name"],
            "kind": sym["kind"],
            "file": sym["file"],
            "line": sym["line"],
        }

    # Fall back to whatever LSP gave us (case-insensitive match) if no exact
    # match exists in the graph.
    return lsp_result


def _in_skipped_dir(path: Path) -> bool:
    """True if any path component is a well-known non-source directory."""
    return not _SKIPPED_DIRS.isdisjoint(path.parts)


def _language_from_ext(ext: str) -> str:
    """Map file extension to a language label."""
    mapping = {
        ".py": "Python",
        ".ts": "TypeScript",
        ".tsx": "TypeScript React",
        ".js": "JavaScript",
        ".jsx": "JavaScript React",
        ".mjs": "JavaScript",
        ".rs": "Rust",
        ".go": "Go",
        ".c": "C",
        ".h": "C Header",
        ".cpp": "C++",
        ".cc": "C++",
        ".hpp": "C++ Header",
        ".sh": "Bash",
        ".bash": "Bash",
        ".lua": "Lua",
    }
    return mapping.get(ext, "Unknown")
