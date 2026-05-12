"""Understanding tools — ast_query, call_graph, impact_analysis."""

from __future__ import annotations

from typing import Any, Literal


def ast_query(ast_indexer: Any, file_path: str, xpath: str) -> list[dict[str, Any]]:
    """Run a tree-sitter query against a file.

    Args:
        ast_indexer: ASTIndexer instance.
        file_path: File to query.
        xpath: Query type: 'function', 'class', 'variable', 'import', 'all'.

    Returns:
        List of {type, name, line, kind} for each matching node.
    """
    return ast_indexer.run_ast_query(file_path, xpath)


def call_graph(
    graph: Any,
    function: str,
    file_path: str | None = None,
    direction: str = "both",
    depth: int = 2,
) -> dict[str, Any]:
    """Traverse the call graph from a symbol.

    Note: This reports **static** call edges extracted from the AST. Calls
    made via dynamic dispatch (e.g. MCP framework tool registration,
    decorator-based routing, event handlers) are not captured.  For
    entry-point functions with 0 callers, this is expected.

    Args:
        graph: KnowledgeGraph instance.
        function: Symbol name to start from.
        file_path: Optional file to disambiguate.
        direction: 'upstream' (callers), 'downstream' (callees), or 'both'.
        depth: Max traversal depth.

    Returns:
        {upstream: [...], downstream: [...], cycles: [...]}
    """
    sym = graph.get_symbol(function, file_path)
    if sym is None:
        # Try fuzzy search
        results = graph.search_symbols(function, limit=1)
        if not results:
            return {"upstream": [], "downstream": [], "cycles": []}
        sym = results[0]

    symbol_id = sym["id"]
    result = graph.call_graph(symbol_id, direction, depth)
    cycles = _detect_cycles(graph, symbol_id)
    return {**result, "cycles": cycles}


def _detect_cycles(graph: Any, start_id: int) -> list[list[str]]:
    """Detect cycles in the call graph starting from start_id."""
    # Simple DFS cycle detection with depth cap
    _MAX_DEPTH = 50
    cycles: list[list[str]] = []
    visited: set[int] = set()
    path: list[int] = []

    def dfs(node_id: int):
        if len(path) >= _MAX_DEPTH:
            return
        if node_id in path:
            cycle_start = path.index(node_id)
            cycles.append([
                graph.conn.execute(
                    "SELECT name FROM symbols WHERE id = ?", (nid,)
                ).fetchone()[0]
                for nid in path[cycle_start:]
            ])
            return
        if node_id in visited:
            return
        visited.add(node_id)
        path.append(node_id)
        rows = graph.conn.execute(
            "SELECT to_id FROM edges WHERE from_id = ? AND kind = 'calls'",
            (node_id,),
        ).fetchall()
        for (to_id,) in rows:
            dfs(to_id)
        path.pop()

    dfs(start_id)
    return cycles


def impact_analysis(
    graph: Any,
    ast_indexer: Any,
    target: str,
    file_path: str | None = None,
) -> dict[str, Any]:
    """Analyze the blast radius of changing a symbol.

    Args:
        graph: KnowledgeGraph instance.
        ast_indexer: ASTIndexer instance.
        target: Symbol name.
        file_path: Optional file to disambiguate.

    Returns:
        {risk: 'HIGH'|'MED'|'LOW', callers: int, tests: [...], modules: [...], module_count: int, note?: str}
    """
    sym = graph.get_symbol(target, file_path)
    if sym is None:
        results = graph.search_symbols(target, limit=1)
        if not results:
            return {
                "risk": "LOW",
                "callers": 0,
                "tests": [],
                "modules": [],
                "module_count": 0,
            }
        sym = results[0]

    symbol_id = sym["id"]
    upstream = graph.upstream(symbol_id, depth=3)
    downstream = graph.downstream(symbol_id, depth=3)

    callers = len(upstream)
    callees = len(downstream)

    # Find tests among callers (heuristic: file contains 'test')
    tests = []
    for caller in upstream:
        fname = caller.get("file", "")
        name = caller.get("name", "")
        if "test" in fname.lower() or "test" in name.lower() or "spec" in fname.lower():
            tests.append({
                "name": name,
                "file": fname,
                "line": caller.get("line", 0),
            })

    # Affected modules
    modules = graph.affected_modules(symbol_id, depth=3)

    # Risk scoring
    total_affected = callers + callees
    if total_affected > 10:
        risk = "HIGH"
    elif total_affected > 3:
        risk = "MED"
    else:
        risk = "LOW"

    result: dict[str, Any] = {
        "risk": risk,
        "callers": callers,
        "callees": callees,
        "tests": tests,
        "modules": modules,
        "module_count": len(modules),
    }

    # Diagnostic note for entry-point functions that have 0 callers
    # but many callees — likely called via dynamic dispatch (MCP tool
    # registration, decorators, event handlers, etc.)
    if callers == 0 and callees > 0:
        sig = sym.get("signature", "")
        name = sym.get("name", target)
        # Heuristics: decorated functions, handler-like names, public API
        is_likely_entry_point = (
            not name.startswith("_")
            or name.startswith("_ensure")
            or "handler" in name.lower()
            or "main" in name.lower()
        )
        if is_likely_entry_point:
            result["note"] = (
                f"'{name}' has 0 static callers but {callees} callees. "
                f"It may be an entry point invoked via dynamic dispatch "
                f"(decorator, framework registration, event handler). "
                f"The static call graph cannot capture these edges."
            )

    return result
