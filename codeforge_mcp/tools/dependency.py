"""AST dependency graph — parse import/require/use statements across modules.

Builds a module-level dependency graph from tree-sitter import parsing.
Supports: Python, JavaScript/TypeScript, Rust, Go, C/C++.

Known limitations (v0.1):
- Python relative imports beyond single-dot (e.g., '..foo.bar') use
  best-effort path resolution; deeply nested relative imports may fail.
- JavaScript/TypeScript path aliases (tsconfig paths, webpack resolve)
  are not resolved — only relative and bare specifiers are attempted.
- Package.json 'main'/'exports' field resolution is not supported.
- Namespace packages (PEP 420) and editable installs are not detected.
- C/C++ include paths (gcc -I, clang -isystem) are not resolved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Cache for tsconfig paths: project_root -> {alias: [paths]}
_TSCONFIG_CACHE: dict[str, dict[str, list[str]]] = {}


def clear_tsconfig_cache() -> None:
    """Clear the global tsconfig path cache."""
    _TSCONFIG_CACHE.clear()


def _load_tsconfig_paths(root: Path) -> dict[str, list[str]]:
    """Parse tsconfig.json to extract compilerOptions.paths."""
    root_str = str(root.resolve())
    if root_str in _TSCONFIG_CACHE:
        return _TSCONFIG_CACHE[root_str]

    tsconfig = root / "tsconfig.json"
    jsconfig = root / "jsconfig.json"
    config_file = tsconfig if tsconfig.exists() else jsconfig if jsconfig.exists() else None

    paths: dict[str, list[str]] = {}
    if config_file:
        try:
            # Note: json.load fails on comments (common in tsconfig), 
            # but we'll try best-effort or simple strip.
            text = config_file.read_text()
            # Simple regex to remove single-line comments
            import re
            text = re.sub(r'//.*', '', text)
            data = json.loads(text)
            opts = data.get("compilerOptions", {})
            raw_paths = opts.get("paths", {})
            base_url = opts.get("baseUrl", ".")
            
            for alias, targets in raw_paths.items():
                # Store as relative paths from root
                clean_alias = alias.replace("/*", "")
                paths[clean_alias] = [
                    str(Path(base_url) / t.replace("/*", "")) for t in targets
                ]
        except Exception:
            pass

    _TSCONFIG_CACHE[root_str] = paths
    return paths


def ast_dependency_graph(
    ast_indexer: Any,
    project_root: str,
    focus_file: str | None = None,
    max_files: int = 200,
) -> dict[str, Any]:
    """Build a module-level dependency graph from import statements.

    Parses import/require/use statements using tree-sitter and returns:
    - Nodes: {file: {imports: [name], language}}
    - Edges: [{from_file, to_file, kind}] (imports relationship)

    Args:
        ast_indexer: ASTIndexer instance (for parsing).
        project_root: Project root directory.
        focus_file: Optional file to focus the graph on (only show its direct deps).
        max_files: Maximum files to scan (default 200).

    Returns:
        {nodes: {file: {imports: [...], language: str}}, edges: [{from_file, to_file, kind}]}
    """
    root = Path(project_root)
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []

    # Resolve which files to scan
    if focus_file:
        files_to_scan = _resolve_dep_files(root, focus_file, ast_indexer, max_files)
    else:
        files_to_scan = _discover_importable_files(root, max_files)

    # Phase 1: Parse each file and extract imports → build nodes dict
    # Include ALL files, even those with no imports (they may be import targets)
    for fpath in files_to_scan[:max_files]:
        try:
            imports = _extract_imports(root, fpath)
            rel_path = str(Path(fpath).relative_to(root))
            lang = _file_language(fpath)
            nodes[rel_path] = {"imports": imports, "language": lang}
        except Exception:
            continue

    # Phase 2: Build edges using pre-computed reverse index (rel_path → absolute path)
    # _resolve_import returns absolute paths, so build a lookup from abs → rel
    abs_to_rel: dict[str, str] = {}
    for rel_path in nodes:
        abs_path = str((root / rel_path).resolve())
        abs_to_rel[abs_path] = rel_path

    for rel_path, node_data in nodes.items():
        abs_from = str((root / rel_path).resolve())
        for imp in node_data.get("imports", []):
            resolved = _resolve_import(root, abs_from, imp)
            if resolved is not None:
                resolved_str = str(resolved.resolve())
                if resolved_str in abs_to_rel:
                    edges.append({
                        "from_file": rel_path,
                        "to_file": abs_to_rel[resolved_str],
                        "kind": "imports",
                    })

    return {
        "nodes": nodes,
        "edges": edges,
        "total_files": len(nodes),
        "total_edges": len(edges),
    }


def _discover_importable_files(root: Path, max_files: int) -> list[str]:
    """Find all importable source files in the project.

    Delegates to the canonical discover_files() to ensure consistent
    gitignore handling and avoid duplicate traversal logic (MIN-6).
    """
    from codeforge_mcp.indexer import discover_files

    rel_paths = discover_files(root)
    # discover_files returns relative paths; convert to absolute
    files = [str((root / rp).resolve()) for rp in rel_paths[:max_files]]
    return files


def _resolve_dep_files(root: Path, focus_file: str, ast_indexer: Any, max_files: int) -> list[str]:
    """Resolve all files that the focus_file imports, recursively."""
    from collections import deque

    visited: set[str] = set()
    result: list[str] = []

    focus_path = root / focus_file
    if focus_path.exists():
        visited.add(str(focus_path.resolve()))
        result.append(str(focus_path))

    # BFS on imports
    queue: deque[Path] = deque([focus_path] if focus_path.exists() else [])
    while queue and len(result) < max_files:
        current = queue.popleft()
        imports = _extract_imports(root, current)
        for imp in imports:
            resolved = _resolve_import(root, current, imp)
            if resolved and str(resolved) not in visited:
                visited.add(str(resolved))
                result.append(str(resolved))
                queue.append(resolved)

    return result


def _file_language(file_path: str | Path) -> str:
    """Detect language from file extension."""
    path = Path(file_path)
    ext = path.suffix
    lang_map = {
        ".py": "python", ".js": "javascript", ".mjs": "javascript",
        ".ts": "typescript", ".tsx": "typescript", ".jsx": "javascript",
        ".rs": "rust", ".go": "go",
        ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp",
    }
    return lang_map.get(ext, "unknown")


def _extract_imports(root: Path, file_path: str | Path) -> list[str]:
    """Extract import/module names from a source file.

    Uses tree-sitter for Python and JavaScript/TypeScript for accurate
    AST-based extraction (ignores imports in strings/comments, handles
    multi-line imports).  Falls back to regex for Rust, Go, C/C++.
    """
    path = Path(file_path)
    lang = _file_language(path)

    if lang == "unknown":
        return []

    try:
        source = path.read_bytes()
    except (OSError, PermissionError):
        return []

    imports: list[str] = []

    if lang == "python":
        imports = _extract_imports_ts_python(source, path)
    elif lang in ("javascript", "typescript"):
        imports = _extract_imports_ts_js(source, lang)
    elif lang == "rust":
        import re
        text = source.decode("utf-8", errors="replace")
        for m in re.finditer(r"use\s+([a-zA-Z_:]+)", text):
            mod_path = m.group(1)
            if "::" in mod_path:
                first = mod_path.split("::")[0]
                if first not in ("crate", "self", "super"):
                    imports.append(first)
            else:
                imports.append(mod_path)
    elif lang == "go":
        import re
        text = source.decode("utf-8", errors="replace")
        for m in re.finditer(r'import\s+(?:[\w\s]*\()?\s*"([^"]+)"', text):
            imports.append(m.group(1))
    elif lang in ("c", "cpp"):
        import re
        text = source.decode("utf-8", errors="replace")
        for m in re.finditer(r'#include\s+[<"]([^>"]+)[>"]', text):
            imports.append(m.group(1))

    # Deduplicate
    seen: set[str] = set()
    unique: list[str] = []
    for imp in imports:
        if imp not in seen:
            seen.add(imp)
            unique.append(imp)

    return unique


def _extract_imports_ts_python(source: bytes, file_path: Path) -> list[str]:
    """Extract Python imports using tree-sitter AST."""
    try:
        import tree_sitter
        lang_mod = __import__("tree_sitter_python")
        lang = tree_sitter.Language(lang_mod.language())
        parser = tree_sitter.Parser()
        parser.language = lang
        tree = parser.parse(source)
    except (ImportError, AttributeError):
        # Fall back to regex if tree-sitter-python isn't installed
        return _extract_imports_regex_python(source, file_path)

    imports: list[str] = []
    for child in tree.root_node.children:
        if child.type == "import_statement":
            # import X, import X.Y, import X as Z
            for name_node in child.children:
                if name_node.type == "dotted_name" and name_node.text:
                    imports.append(name_node.text.decode())
                elif name_node.type == "aliased_import":
                    for n in name_node.children:
                        if n.type == "dotted_name" and n.text:
                            imports.append(n.text.decode())
                            break
        elif child.type == "import_from_statement":
            # from X import Y
            for name_node in child.children:
                if name_node.type == "dotted_name" and name_node.text:
                    imports.append(name_node.text.decode())
                    break
                elif name_node.type == "relative_import" and name_node.text:
                    # from .foo import bar
                    dot_text = name_node.text.decode()
                    imports.append(_resolve_relative_import(file_path, dot_text))
                    break
    return imports


def _extract_imports_regex_python(source: bytes, file_path: Path) -> list[str]:
    """Fallback regex-based Python import extraction."""
    text = source.decode("utf-8", errors="replace")
    imports: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("import "):
            mod = line[7:].split("#")[0].split(" as ")[0].strip()
            for part in mod.split(","):
                part = part.strip()
                if part:
                    imports.append(part)
        elif line.startswith("from ") and " import " in line:
            mod = line[5:].split(" import ")[0].strip()
            if mod.startswith("."):
                imports.append(_resolve_relative_import(file_path, mod))
            else:
                imports.append(mod)
    return imports


def _extract_imports_ts_js(source: bytes, lang_name: str) -> list[str]:
    """Extract JS/TS imports using tree-sitter AST."""
    try:
        import tree_sitter
        ts_lang_name = lang_name if lang_name != "typescript" else "typescript"
        lang_mod = __import__(f"tree_sitter_{ts_lang_name}")
        lang = tree_sitter.Language(lang_mod.language())
        parser = tree_sitter.Parser()
        parser.language = lang
        tree = parser.parse(source)
    except (ImportError, AttributeError):
        return _extract_imports_regex_js(source)

    imports: list[str] = []

    def _walk(node: Any) -> None:
        if node.type == "import_statement":
            # import ... from 'module' or import 'module'
            for child in node.children:
                if child.type == "string":
                    val = child.text.decode().strip("'\"")
                    if val:
                        imports.append(val)
        elif node.type == "call_expression":
            # require('module')
            if node.children and node.children[0].type == "identifier":
                if node.children[0].text == b"require":
                    args = node.child_by_field_name("arguments")
                    if args:
                        for arg in args.children:
                            if arg.type == "string":
                                val = arg.text.decode().strip("'\"")
                                if val:
                                    imports.append(val)
        for child in node.children:
            _walk(child)

    _walk(tree.root_node)
    return imports


def _extract_imports_regex_js(source: bytes) -> list[str]:
    """Fallback regex-based JS/TS import extraction."""
    import re
    text = source.decode("utf-8", errors="replace")
    imports: list[str] = []
    for m in re.finditer(r"""from\s+['"]([^'"]+)['\"]""", text):
        imports.append(m.group(1))
    for m in re.finditer(r"""import\s+['"]([^'"]+)['\"]""", text):
        imports.append(m.group(1))
    for m in re.finditer(r"""require\s*\(\s*['"]([^'"]+)['\"]""", text):
        imports.append(m.group(1))
    return imports


def _resolve_import(root: Path, from_file: str | Path, import_name: str) -> Path | None:
    """Resolve an import name to a file path.

    import_name is either a dotted module name (e.g. 'os', 'src.foo.bar')
    or an absolute filesystem path produced by _resolve_relative_import.
    """
    # ── TSConfig Alias Resolution ───────────────────────────────────
    tsconfig_paths = _load_tsconfig_paths(root)
    for alias, targets in tsconfig_paths.items():
        if import_name.startswith(alias):
            remainder = import_name[len(alias):].lstrip("/")
            for target in targets:
                # Resolve target relative to root
                cand_base = root / target / remainder
                for ext in (".ts", ".tsx", ".js", ".jsx", ".mjs"):
                    probe = cand_base.with_suffix(ext)
                    if probe.is_file():
                        return probe
                    probe = cand_base / f"index{ext}"
                    if probe.is_file():
                        return probe

    # ── Path Resolution ──────────────────────────────────────────────
    # _resolve_relative_import returns an extensionless absolute path string.
    # Resolve it directly as a filesystem path (Python only — relative imports
    # only arise from Python code).
    if import_name.startswith('/'):
        cand = Path(import_name)
        for probe in (cand.with_suffix('.py'), cand / '__init__.py', cand):
            try:
                if probe.is_file():
                    return probe
            except (OSError, PermissionError):
                continue
        return None

    from_file_path = Path(from_file)

    # Try typical resolution strategies for dotted names (order: local first)
    candidates: list[Path] = []
    rel_dir = from_file_path.parent
    # Preserve relative import specifiers (./foo, ../foo); only replace dots for absolute imports
    if import_name.startswith('.'):
        dotted_path = import_name
    else:
        dotted_path = import_name.replace('.', '/')

    # 1. Relative to importing file's directory (Python local imports)
    candidates.append(rel_dir / f"{dotted_path}.py")
    candidates.append(rel_dir / f"{dotted_path}/__init__.py")

    # 2. Relative to project root (absolute imports)
    candidates.append(root / f"{dotted_path}.py")
    candidates.append(root / f"{dotted_path}/__init__.py")

    # 3. For JS/TS: .js, .ts, .tsx, .jsx extensions (local first, then project root)
    for ext in (".ts", ".tsx", ".js", ".jsx", ".mjs"):
        candidates.append(rel_dir / f"{dotted_path}{ext}")
        candidates.append(root / f"{dotted_path}{ext}")
        candidates.append(rel_dir / f"{dotted_path}/index{ext}")
        candidates.append(root / f"{dotted_path}/index{ext}")

    for cand in candidates:
        try:
            if cand.is_file():
                return cand
        except (OSError, PermissionError):
            continue

    return None





def _resolve_relative_import(from_file: Path, import_path: str) -> str:
    """Resolve a relative Python import like '.foo' or '..bar.baz' to an absolute dotted name."""
    # Count dots at start
    dots = 0
    for c in import_path:
        if c == ".":
            dots += 1
        else:
            break
    remainder = import_path[dots:]
    # Walk up 'dots - 1' directories from the file's parent
    current = from_file.resolve().parent
    for _ in range(dots - 1):
        current = current.parent
    # Convert to a path from project root
    # This is best-effort
    return str(current / remainder.replace(".", "/"))
