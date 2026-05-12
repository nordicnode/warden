"""AST layer — tree-sitter incremental indexing with XXH3 checksums.

Uses individual tree-sitter language packages (tree-sitter-python,
tree-sitter-javascript, etc.) for reliable grammar loading.
Re-parses only changed files on re-index.

Supported languages: Python, JavaScript, TypeScript, Rust, Go, C, C++,
Bash, Lua.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import xxhash

# Lazy imports to avoid hard dependency failures at import time
_tree_sitter: Any = None


def _ensure_tree_sitter() -> Any:
    global _tree_sitter
    if _tree_sitter is None:
        import tree_sitter as ts

        _tree_sitter = ts
    return _tree_sitter


# Individual language packages — each exposes a .language() → PyCapsule.
# Exceptions: tree-sitter-typescript uses .language_typescript()/.language_tsx()
_LANG_PACKAGE: dict[str, str] = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "tsx": "tree_sitter_typescript",
    "rust": "tree_sitter_rust",
    "go": "tree_sitter_go",
    "c": "tree_sitter_c",
    "cpp": "tree_sitter_cpp",
    "bash": "tree_sitter_bash",
    "lua": "tree_sitter_lua",
}

# tree-sitter-typescript uses special method names (not .language())
_LANG_GETTER: dict[str, str] = {
    "typescript": "language_typescript",
    "tsx": "language_tsx",
}

# Cache: lang_name → ts.Language object
_LANGUAGE_CACHE: dict[str, Any] = {}


def _get_language(lang_name: str) -> Any | None:
    """Load a tree-sitter Language for the given language name.

    Uses individual language packages (tree-sitter-python, etc.).
    Returns None if the language cannot be loaded.
    """
    if lang_name in _LANGUAGE_CACHE:
        return _LANGUAGE_CACHE[lang_name]

    ts = _ensure_tree_sitter()
    pkg_name = _LANG_PACKAGE.get(lang_name)

    lang: Any = None

    if pkg_name:
        try:
            pkg = importlib.import_module(pkg_name)
            # Some packages use non-standard getter names (e.g. tree-sitter-typescript)
            getter_name = _LANG_GETTER.get(lang_name, "language")
            getter = getattr(pkg, getter_name, None)
            if getter is not None:
                capsule = getter()
                lang = ts.Language(capsule)
        except (ImportError, AttributeError, TypeError, ValueError):
            lang = None

    if lang is not None:
        _LANGUAGE_CACHE[lang_name] = lang
    return lang


# Map file extensions to tree-sitter language names
EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".sh": "bash",
    ".bash": "bash",
    ".lua": "lua",
}

KEYWORD_KINDS: dict[str, str] = {
    "function_definition": "function",
    "function_declaration": "function",
    "function_item": "function",  # Rust
    "method_definition": "method",
    "method_declaration": "method",  # Go
    "class_definition": "class",
    "class_declaration": "class",
    "struct_item": "class",  # Rust
    "enum_item": "enum",  # Rust
    "trait_item": "interface",  # Rust
    "impl_item": "implementation",  # Rust
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
    "type_item": "type",  # Rust
    "type_declaration": "type",  # Go
    "type_spec": "type",  # Go
    "enum_declaration": "enum",
    "variable_declaration": "variable",
    "variable_declarator": "variable",  # JS/TS
    "lexical_declaration": "variable",  # TS/JS: const/let/var
    "var_spec": "variable",  # Go
    "const_spec": "constant",  # Go
    "module": "module",
    "mod_item": "module",  # Rust
    "namespace_definition": "module",  # C++
    "struct_specifier": "class",  # C/C++
    "const_declaration": "constant",
}

ParsedSymbol = dict[str, str | int]

CallEdge = dict[str, str]


def hash_source(source: str) -> str:
    """Compute XXH3 hash of source for incremental indexing."""
    return xxhash.xxh3_64(source.encode()).hexdigest()


def _supported_ext(path: Path) -> str | None:
    """Return the language name if the file extension is supported, else None."""
    suffixes = path.suffixes
    for suf in reversed(suffixes):
        if suf in EXT_TO_LANG:
            return EXT_TO_LANG[suf]
    return None


def _get_text(node: Any, source: bytes) -> str:
    """Extract source text for a node."""
    return source[node.start_byte : node.end_byte].decode(errors="replace")


def _node_kind(node: Any) -> str:
    """Map a tree-sitter node type to our SymbolKind."""
    return KEYWORD_KINDS.get(node.type, "unknown")


def _extract_symbols(
    source: bytes, node: Any, file_path: str, depth: int = 0
) -> list[ParsedSymbol]:
    """Recursively walk the AST and extract symbol definitions."""
    symbols: list[ParsedSymbol] = []
    kind = _node_kind(node)

    if kind != "unknown":
        name = ""
        for child in node.children:
            if child.type in (
                "identifier",
                "name",
                "property_identifier",
                "type_identifier",
                "word",  # Bash
                "dot_index_expression",  # Lua
                "method_index_expression",  # Lua
                "field_identifier",  # C/C++
            ):
                name = _get_text(child, source)
                break

        if name:
            text = _get_text(node, source)
            first_line = text.split("\n")[0][:200]
            signature = first_line
            doc = _extract_doc(source, node)

            symbols.append({
                "name": name,
                "kind": kind,
                "file": file_path,
                "line": node.start_point[0] + 1,
                "signature": signature,
                "doc": doc,
                "hash": hash_source(text),
            })

    # Recurse into ALL named children (not just known symbols) to find
    # nested definitions like methods inside classes, decorated functions, etc.
    for child in node.named_children:
        symbols.extend(_extract_symbols(source, child, file_path, depth + 1))

    return symbols


def _clean_quotes(text: str) -> str:
    """Remove surrounding Python quotes from a string literal.

    Handles triple-quoted strings, single-quoted, and double-quoted
    strings.
    """
    text = text.strip()
    for quote in ('"""', "'''", '"', "'"):
        if text.startswith(quote) and text.endswith(quote):
            return text[len(quote):-len(quote)].strip()
    return text


def _extract_doc(source: bytes, node: Any) -> str:
    """Extract docstring or JSDoc comment for a symbol node.

    Supports:
    - Python: string expression_statement (docstrings), walks past decorators
    - JavaScript/TypeScript: JSDoc comment nodes
    """
    try:
        actual_node = node
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type in ("function_definition", "class_definition"):
                    actual_node = child
                    break

        body = actual_node.child_by_field_name("body")
        if body and body.children:
            first = body.children[0]
            if first.type == "expression_statement":
                for gc in first.children:
                    if gc.type == "string":
                        text = _get_text(gc, source)
                        text = _clean_quotes(text)
                        first_line = text.split("\n")[0].strip()
                        return first_line[:200]
    except Exception:
        pass

    # JavaScript/TypeScript: JSDoc comments
    try:
        parent = node.parent
        if parent is not None:
            prev_sibling = None
            for i, child in enumerate(parent.children):
                if child == node and i > 0:
                    prev_sibling = parent.children[i - 1]
                    break
            if prev_sibling is not None and prev_sibling.type == "comment":
                text = _get_text(prev_sibling, source)
                if text.startswith("/**"):
                    cleaned = text.replace("/**", "").replace("*/", "").strip()
                    for line in cleaned.split("\n"):
                        stripped = line.strip().lstrip("*").strip()
                        if stripped and not stripped.startswith("@"):
                            return stripped[:200]
    except Exception:
        pass

    return ""


def parse_file(file_path: str | Path) -> list[ParsedSymbol]:
    """Parse a single file with tree-sitter and return extracted symbols.

    Returns empty list if the language is not supported or parsing fails.
    """
    path = Path(file_path)
    lang_name = _supported_ext(path)
    if lang_name is None:
        return []

    language = _get_language(lang_name)
    if language is None:
        return []

    try:
        source_bytes = path.read_bytes()
    except (OSError, PermissionError):
        return []

    try:
        ts = _ensure_tree_sitter()
        parser = ts.Parser()
        parser.language = language
        tree = parser.parse(source_bytes)
        symbols = _extract_symbols(source_bytes, tree.root_node, str(path))
        return symbols
    except Exception as e:
        from codeforge_mcp import logging as log
        log.warn("parse error", file=str(path), error=str(e))
        return []


class ASTIndexer:
    """Manages parsing files and feeding results into the KnowledgeGraph."""

    def __init__(self, graph: Any) -> None:
        self.graph = graph

    def _create_edges_from_tree(
        self,
        source_bytes: bytes,
        root_node: Any,
        file_path: str,
        name_to_id: dict[str, int],
    ) -> int:
        """Extract call edges from a parsed tree and persist them.

        Uses *name_to_id* for within-file callee resolution; falls back
        to graph.get_symbol() for cross-file callees.

        Returns the number of edges created.
        """
        calls = _extract_call_edges(source_bytes, root_node)
        count = 0
        for call in calls:
            caller_name = call["caller_name"]
            callee_name = call["callee_name"]

            # Resolve caller (prefer local name→id, fall back to graph)
            caller_id = name_to_id.get(caller_name)
            if caller_id is None:
                caller_sym = self.graph.get_symbol(caller_name, file=file_path)
                if caller_sym:
                    caller_id = caller_sym["id"]
            if caller_id is None:
                continue

            # Resolve callee (local first, then cross-file graph lookup)
            callee_id = name_to_id.get(callee_name)
            if callee_id is None:
                callee_sym = self.graph.get_symbol(callee_name)
                if callee_sym:
                    callee_id = callee_sym["id"]
            if callee_id is None:
                continue

            # Skip self-calls (recursion edges aren't useful for navigation)
            if caller_id == callee_id:
                continue

            self.graph.add_edge(caller_id, callee_id, "calls")
            count += 1

        return count

    def _parse_and_index(
        self,
        file_path: str | Path,
        clear_first: bool = False,
        create_edges_only: bool = False,
    ) -> int:
        """Parse one file: extract symbols, upsert them, then create call edges.

        Args:
            file_path: Absolute or relative path to the source file.
            clear_first: If True, delete existing symbols (and cascade edges)
                before re-indexing. Used by incremental re-indexing.
            create_edges_only: If True, skip the UPSERT phase and only create
                edges.  The caller guarantees all symbols already exist in
                the graph.  Used by index_project's second pass to avoid
                wasteful re-UPSERTs.

        Returns:
            Number of symbols parsed from the file.
            Returns 0 when the language is not supported, the parser
            is unavailable, or the file cannot be read.
            Raises on parse errors (tree-sitter crashes, broken syntax
            that the parser can't handle) so callers can distinguish
            "no symbols" from "parser crashed".
            When `create_edges_only` is True no symbols are upserted
            but the return value is still the symbol count for
            consistency.
        """
        path = str(file_path)
        path_obj = Path(file_path)

        lang_name = _supported_ext(path_obj)
        if lang_name is None:
            return 0

        language = _get_language(lang_name)
        if language is None:
            return 0

        try:
            source_bytes = path_obj.read_bytes()
        except (OSError, PermissionError):
            return 0

        # Canonicalize to a resolved absolute path so symbol file paths
        # are consistent across index passes.  Without this, the first
        # pass might store "/abs/path/server.py" while the second pass
        # (edge-only) stores "server.py", causing get_symbol() to miss
        # the callee during cross-file edge resolution and producing a
        # graph with no upstream callers for entry-point functions.
        path = str(path_obj.resolve())

        # If parsing fails the exception propagates so callers can
        # distinguish parse crashes from "file had zero symbols"
        # (unsupported lang / unreadable file return 0 silently).
        ts = _ensure_tree_sitter()
        parser = ts.Parser()
        parser.language = language
        tree = parser.parse(source_bytes)

        symbols = _extract_symbols(source_bytes, tree.root_node, path)

        # ── Phase 1: upsert symbols (skipped when create_edges_only) ─
        self.graph.begin_batch()
        try:
            if not create_edges_only:
                if clear_first:
                    self.graph.delete_symbols_in_file(path)

                name_to_id: dict[str, int] = {}
                for sym in symbols:
                    sid = self.graph.upsert_symbol(
                        name=str(sym["name"]),
                        kind=str(sym["kind"]),
                        file=path,
                        line=int(sym["line"]),
                        signature=str(sym["signature"]),
                        doc=str(sym["doc"]),
                        content_hash=str(sym["hash"]),
                    )
                    name_to_id[str(sym["name"])] = sid
            else:
                # Edges-only: build name→id from existing graph rows
                name_to_id = {}
                for sym in symbols:
                    name = str(sym["name"])
                    row = self.graph.get_symbol(name, file=path)
                    if row:
                        name_to_id[name] = row["id"]

            # ── Phase 2: create call edges ──────────────────────────
            self._create_edges_from_tree(source_bytes, tree.root_node, path, name_to_id)
        finally:
            self.graph.end_batch()

        return len(symbols)

    def index_file(self, file_path: str | Path) -> int:
        """Parse one file, upsert symbols, and create call edges.

        Does NOT clear existing symbols first — use for initial indexing
        or when you know the file is new. For re-indexing, use
        index_file_incremental which only rewrites if content changed.

        Returns count of symbols added (0 if unsupported / unreadable).
        Raises on parse errors so callers can distinguish "no symbols"
        from "parser crashed".
        """
        return self._parse_and_index(file_path, clear_first=False)

    def index_file_incremental(self, file_path: str | Path) -> int:
        """Re-index a file only if any symbol hash changed.

        Parses the file once, checks hashes, and if changed:
        deletes old symbols (cascade-deletes edges), re-upserts, and
        recreates call edges — all from the same parse tree (no double-parse).

        Returns count of symbols added (0 = no change, or unsupported/
        unreadable file).  Raises on parse errors so callers can
        distinguish "no symbols" from "parser crashed".
        """
        path = str(file_path)
        path_obj = Path(file_path)

        lang_name = _supported_ext(path_obj)
        if lang_name is None:
            return 0
        language = _get_language(lang_name)
        if language is None:
            return 0

        try:
            source_bytes = path_obj.read_bytes()
        except (OSError, PermissionError):
            return 0

        # Canonicalize path (same as _parse_and_index)
        path = str(path_obj.resolve())

        # If parsing fails the exception propagates so callers can
        # distinguish parse crashes from "file had zero symbols"
        # (unsupported lang / unreadable file return 0 silently).
        ts = _ensure_tree_sitter()
        parser = ts.Parser()
        parser.language = language
        tree = parser.parse(source_bytes)

        # Extract symbols from the single parse
        new_symbols = _extract_symbols(source_bytes, tree.root_node, path)
        new_hashes = {str(s["name"]): str(s["hash"]) for s in new_symbols}
        existing_hashes = self.graph.hashes_for_file(path)

        # Check if anything changed
        changed = False
        for name, h in new_hashes.items():
            if existing_hashes.get(name) != h:
                changed = True
                break
        for name in existing_hashes:
            if name not in new_hashes:
                changed = True
                break

        if not changed:
            return 0

        # Delete old symbols (edges cascade-delete via FK)
        self.graph.begin_batch()
        try:
            self.graph.delete_symbols_in_file(path)

            # Upsert symbols from the already-parsed tree
            name_to_id: dict[str, int] = {}
            for sym in new_symbols:
                sid = self.graph.upsert_symbol(
                    name=str(sym["name"]),
                    kind=str(sym["kind"]),
                    file=path,
                    line=int(sym["line"]),
                    signature=str(sym["signature"]),
                    doc=str(sym["doc"]),
                    content_hash=str(sym["hash"]),
                )
                name_to_id[str(sym["name"])] = sid

            # Create edges from the same parse tree
            self._create_edges_from_tree(source_bytes, tree.root_node, path, name_to_id)
        finally:
            self.graph.end_batch()

        return len(new_symbols)

    def run_ast_query(
        self, file_path: str | Path, xpath: str
    ) -> list[dict[str, Any]]:
        """Run a tree-sitter query against a file and return matching nodes.

        Supports two modes:
        1. Keyword mode: 'function', 'class', 'variable', 'import', 'all'
        2. Tree-sitter query mode: S-expression patterns like
           '(function_definition) @func'

        Returns a list of match dicts.  If ``xpath`` is not a recognized
        keyword, node-type, or S-expression, returns a single error entry
        so the caller can tell "bad query" from "no matches".
        """
        path = Path(file_path)
        try:
            source_bytes = path.read_bytes()
        except (OSError, PermissionError):
            return [{"type": "error", "message": f"Cannot read file: {file_path}"}]

        lang_name = _supported_ext(path)
        if lang_name is None:
            return [{"type": "error", "message": f"Unsupported file type: {path.suffix}"}]

        language = _get_language(lang_name)
        if language is None:
            return [{"type": "error", "message": f"tree-sitter grammar not available for: {lang_name}"}]

        ts = _ensure_tree_sitter()
        parser = ts.Parser()
        parser.language = language
        tree = parser.parse(source_bytes)

        if xpath.startswith("("):
            return self._run_tree_sitter_query(tree, source_bytes, xpath)

        # ── Keyword / node-type validation ─────────────────────────────
        # Valid keywords are the unique *values* in KEYWORD_KINDS plus "all".
        # Valid node types are the *keys* in KEYWORD_KINDS.
        valid_keywords = {"all"} | set(KEYWORD_KINDS.values())
        valid_node_types = set(KEYWORD_KINDS.keys())
        if xpath not in valid_keywords and xpath not in valid_node_types:
            return [{
                "type": "error",
                "message": (
                    f"Unknown query keyword: '{xpath}'. "
                    f"Valid keywords: {sorted(valid_keywords)}. "
                    f"Or use an S-expression like '(function_definition) @func'."
                ),
            }]

        results: list[dict[str, Any]] = []
        # Cap to prevent OOM on large files — a 5K-line file can produce
        # tens of thousands of named nodes in "all" mode.
        _MAX_RESULTS = 1000

        def _walk(node: Any) -> None:
            if len(results) >= _MAX_RESULTS:
                return
            if xpath == "all":
                if node.type != "source_file":
                    results.append({
                        "type": node.type,
                        "line": node.start_point[0] + 1,
                        "text": _get_text(node, source_bytes)[:200],
                    })
            elif node.type in KEYWORD_KINDS and (
                xpath in ("all", KEYWORD_KINDS[node.type])
                or xpath == node.type
            ):
                results.append({
                    "type": node.type,
                    "name": _find_name(node, source_bytes),
                    "line": node.start_point[0] + 1,
                    "kind": KEYWORD_KINDS[node.type],
                })
            for child in node.children:
                _walk(child)

        _walk(tree.root_node)
        return results

    def _run_tree_sitter_query(
        self,
        tree: Any,
        source_bytes: bytes,
        query_str: str,
    ) -> list[dict[str, Any]]:
        """Execute a tree-sitter query (S-expression) against the AST."""
        ts = _ensure_tree_sitter()
        results: list[dict[str, Any]] = []

        try:
            query = ts.Query(tree.language, query_str)
            # tree-sitter 0.25.x: .captures() returns dict[str, list[Node]]
            # (the old .matches() attribute was removed in 0.25.x)
            captures_dict = query.captures(tree.root_node)
            for cap_name, cap_nodes in captures_dict.items():
                for node in cap_nodes:
                    results.append({
                        "type": "query_match",
                        "captures": {cap_name: _get_text(node, source_bytes)},
                    })
        except Exception as e:
            results.append({
                "type": "error",
                "message": f"Query failed: {e}",
            })

        return results


def _find_name(node: Any, source: bytes) -> str:
    for child in node.children:
        if child.type in (
            "identifier", "name", "property_identifier", "type_identifier"
        ):
            return _get_text(child, source)
    return ""


# ── Call-graph extraction ───────────────────────────────────────────
# Walks function/method bodies looking for call expressions and returns
# (caller_name, caller_kind, callee_name) tuples that ASTIndexer then
# resolves to symbol IDs and persists as "calls" edges in the graph.


def _extract_callee_name(call_node: Any, source: bytes) -> str | None:
    """Extract the callee name from a call / call_expression node.

    Handles:
    - Simple calls:   foo()        → "foo"
    - Method calls:   obj.method() → "method"
    - Module calls:   os.path.join(...) → "join"

    Returns None if the callee cannot be statically resolved (e.g.
    computed property access like obj[expr]()).
    """
    # tree-sitter exposes the function being called via the 'function'
    # field on call / call_expression nodes.
    func_node = call_node.child_by_field_name("function")
    if func_node is None:
        return None

    # Simple name: foo()
    if func_node.type == "identifier":
        return _get_text(func_node, source)

    # Attribute access: obj.method() or module.func()
    # The rightmost identifier is the callee name.
    if func_node.type == "attribute":
        for child in reversed(func_node.children):
            if child.type in ("property_identifier", "identifier"):
                return _get_text(child, source)
        return None

    # Member expression: a.b.c() (some grammars use member_expression)
    if func_node.type == "member_expression":
        for child in reversed(func_node.children):
            if child.type in ("property_identifier", "identifier"):
                return _get_text(child, source)
        return None

    return None


def _find_calls_in_subtree(
    node: Any,
    source: bytes,
    results: list[CallEdge],
    caller_name: str,
    caller_kind: str,
) -> None:
    """Recursively search a subtree for call expressions.

    Appends {caller_name, caller_kind, callee_name} dicts to *results*
    for every call site found.
    """
    if node.type in ("call", "call_expression"):
        callee = _extract_callee_name(node, source)
        if callee:
            results.append({
                "caller_name": caller_name,
                "caller_kind": caller_kind,
                "callee_name": callee,
            })

    for child in node.named_children:
        _find_calls_in_subtree(child, source, results, caller_name, caller_kind)


def _extract_call_edges(source: bytes, root_node: Any) -> list[CallEdge]:
    """Walk the full AST and extract (caller, callee) pairs.

    For every function / method definition, walks its body subtree
    looking for call expressions.  Attributes calls inside anonymous
    closures (arrow functions, lambdas) to the nearest named parent.
    """
    edges: list[CallEdge] = []
    # Stack of (name, kind) tuples
    context_stack: list[tuple[str, str]] = []

    def walk(node: Any) -> None:
        kind = _node_kind(node)
        is_named = kind in ("function", "method")
        # Anonymous callable types (grammar-specific)
        is_anonymous = node.type in (
            "arrow_function", "lambda", "closure_expression", "function_expression"
        )
        
        name = ""
        if is_named:
            name = _find_name(node, source)
            if name:
                context_stack.append((name, kind))
        elif is_anonymous:
            # For anonymous functions, we stay in the current named context
            # but still need to walk their children.
            pass

        # If we are inside a named context, look for calls
        if context_stack:
            curr_name, curr_kind = context_stack[-1]
            # Only search for calls in the immediate children of callable nodes
            # to avoid duplicate counting as we walk the full tree.
            if is_named or is_anonymous:
                body = node.child_by_field_name("body")
                if body is not None:
                    _find_calls_in_subtree(body, source, edges, curr_name, curr_kind)

        for child in node.named_children:
            walk(child)

        if is_named and name:
            context_stack.pop()

    walk(root_node)
    return edges
