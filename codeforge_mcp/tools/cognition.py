"""Codebase Cognition Map — dense, model-friendly 10,000-foot view of a repository.

Produces a 2,000–4,000 token Markdown document covering architecture layers,
data flow, key abstractions, and cross-cutting concerns. Distilled from the
AST dependency graph, knowledge graph, and LSP hover/docstrings.

Uses networkx community detection to identify domain clusters from the
module-level import graph.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _build_networkx_graph(dep_graph: dict[str, Any]) -> Any:
    """Convert the dependency graph into a networkx DiGraph.

    Returns the graph object, or None if networkx is unavailable.
    """
    try:
        import networkx as nx
    except ImportError:
        return None

    g = nx.DiGraph()
    for rel_path in dep_graph.get("nodes", {}):
        g.add_node(rel_path)

    for edge in dep_graph.get("edges", []):
        g.add_edge(edge["from_file"], edge["to_file"])

    return g


def _detect_communities(g: Any) -> list[set[str]]:
    """Run greedy modularity community detection on the dependency graph.

    Returns a list of community sets (each set contains relative file paths).
    Falls back to connected-components if community detection fails.
    Merges communities smaller than MIN_COMMUNITY_SIZE files into the
    nearest larger community to prevent excessive granularity.
    """
    import networkx as nx

    MIN_COMMUNITY_SIZE = 3

    try:
        # Convert to undirected for community detection (imports flow both ways)
        ug = g.to_undirected()
        communities: list[set[str]] = [
            set(c) for c in nx.community.greedy_modularity_communities(ug)
        ]
    except Exception:
        # Fallback: weakly connected components
        communities = [set(c) for c in nx.weakly_connected_components(g)]

    # Merge tiny communities (fewer than MIN_COMMUNITY_SIZE files) into
    # the nearest larger community to keep the cognition map useful at
    # a high level.
    large = [c for c in communities if len(c) >= MIN_COMMUNITY_SIZE]
    small = [c for c in communities if len(c) < MIN_COMMUNITY_SIZE]

    if large and small:
        for sc in small:
            # Find the large community with the most cross-edges to sc
            best_community = large[0]
            best_edge_count = 0
            for lc in large:
                edges = sum(
                    1 for u in sc for v in lc
                    if g.has_edge(u, v) or g.has_edge(v, u)
                )
                if edges > best_edge_count:
                    best_edge_count = edges
                    best_community = lc
            best_community.update(sc)
        communities = large
    elif not large and small:
        # All communities are small — merge them all into one
        merged: set[str] = set()
        for c in small:
            merged.update(c)
        communities = [merged]

    return communities


def _comm_name_at_depth(comm: set[str], depth: int) -> str:
    """Most common directory prefix for a community at the given depth.

    Files with fewer path segments than the requested depth are counted
    as root-level.  When root-level files dominate, returns "core".
    """
    if not comm:
        return "core"
    prefix_counts: dict[str, int] = {}
    root_count = 0
    for f in comm:
        parts = f.split("/")
        # Files with no directory separator (len 1) are always root-level;
        # files with fewer parts than the requested depth are too shallow
        if len(parts) == 1 or len(parts) < depth:
            root_count += 1
        else:
            prefix = "/".join(parts[:depth])
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1

    # If root-level files are at least as common as any directory prefix,
    # the community is effectively at the project root — call it "core"
    if root_count >= max(prefix_counts.values(), default=0):
        return "core"

    best = max(prefix_counts.items(), key=lambda x: x[1])
    return best[0] if best[0] else "core"


def _disambiguate_cluster_names(communities: list[set[str]]) -> dict[int, str]:
    """Assign unique human-readable names to each community.

    Uses the most common directory prefix at increasing depths.
    When N communities share the same prefix at depth D (e.g. three
    clusters all rooted under "src"), ALL of them deepen to "src/auth",
    "src/db", "src/api" — not just the later ones.
    Falls back to numeric suffixes when communities are truly
    indistinguishable (same files at every depth).
    """
    names: dict[int, str] = {}
    used_names: set[str] = set()  # O(1) collision checks
    # Communities still needing a name: maps index → current depth
    pending: dict[int, int] = {i: 1 for i in range(len(communities))}

    while pending:
        # Compute candidate name for each pending community at current depth
        round_candidates: dict[int, str] = {
            i: _comm_name_at_depth(communities[i], depth)
            for i, depth in pending.items()
        }

        # Group by candidate name to detect collisions
        by_candidate: dict[str, list[int]] = {}
        for i, cand in round_candidates.items():
            by_candidate.setdefault(cand, []).append(i)

        for cand, idxs in list(by_candidate.items()):
            if len(idxs) == 1:
                # No collision — assign this name immediately
                names[idxs[0]] = cand
                used_names.add(cand)
                del pending[idxs[0]]
            else:
                # Collision — all communities sharing this candidate deepen
                for i in idxs:
                    comm_max_parts = max(
                        (len(f.split("/")) for f in communities[i]),
                        default=1,
                    )
                    max_depth = min(comm_max_parts, 10)
                    if pending[i] < max_depth:
                        pending[i] += 1
                    else:
                        # Indistinguishable at max depth — numeric suffix
                        suffix = 2
                        base = cand
                        while f"{base} ({suffix})" in used_names:
                            suffix += 1
                        names[i] = f"{base} ({suffix})"
                        used_names.add(f"{base} ({suffix})")
                        del pending[i]

    return names


def _top_symbols_in_files(
    graph: Any,
    files: set[str],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Get the top symbols (by call count) within a set of files.

    Uses edge count as a proxy for importance — symbols with more
    edges (callers/callees) are more central.
    """
    if not files:
        return []

    # Build file filter for SQL
    placeholders = ",".join("?" for _ in files)
    file_list = list(files)

    # Get symbols in these files, ordered by total edge count
    # SUM/GROUP BY wraps the UNION ALL so both from_id and to_id
    # contributions are added together (COALESCE would only pick one)
    try:
        rows = graph.conn.execute(
            f"""SELECT s.id, s.name, s.kind, s.file, s.line, s.signature, s.doc,
                COALESCE(e_total.total_cnt, 0) AS edge_count
            FROM symbols s
            LEFT JOIN (
                SELECT sid, SUM(cnt) AS total_cnt
                FROM (
                    SELECT from_id AS sid, COUNT(*) AS cnt FROM edges GROUP BY from_id
                    UNION ALL
                    SELECT to_id AS sid, COUNT(*) AS cnt FROM edges GROUP BY to_id
                )
                GROUP BY sid
            ) e_total ON s.id = e_total.sid
            WHERE s.file IN ({placeholders})
            GROUP BY s.id
            ORDER BY edge_count DESC, s.name
            LIMIT ?""",
            file_list + [limit],
        ).fetchall()
    except Exception:
        return []

    return [
        {
            "name": r[1],
            "kind": r[2],
            "file": r[3],
            "line": r[4],
            "signature": r[5] or "",
            "doc": r[6] or "",
            "edge_count": r[7],
        }
        for r in rows
    ]


def _generate_markdown(
    communities: list[set[str]],
    dep_graph: dict[str, Any],
    graph: Any,
    include_data_flow: bool,
) -> str:
    """Generate the cognition map Markdown document."""
    lines: list[str] = []
    total_files = len(dep_graph.get("nodes", {}))
    total_edges = len(dep_graph.get("edges", []))
    total_symbols = graph.symbol_count() if graph else 0

    # ── Header ──────────────────────────────────────────────────────
    lines.append("# Codebase Cognition Map\n")
    lines.append(
        f"**{total_files}** files · **{total_symbols}** symbols · "
        f"**{total_edges}** import edges · "
        f"**{len(communities)}** architecture layers\n"
    )

    # ── Pre-compute cluster mappings (used by multiple sections) ────
    module_cluster: dict[str, int] = {}
    for i, comm in enumerate(communities):
        for f in comm:
            module_cluster[f] = i

    cluster_edges: dict[tuple[int, int], int] = {}
    for edge in dep_graph.get("edges", []):
        src = edge["from_file"]
        dst = edge["to_file"]
        if src in module_cluster and dst in module_cluster:
            sc = module_cluster[src]
            dc = module_cluster[dst]
            if sc != dc:
                key = (sc, dc)
                cluster_edges[key] = cluster_edges.get(key, 0) + 1

    # ── Architecture Layers ──────────────────────────────────────────
    lines.append("## Architecture Layers\n")
    # Pre-compute unique cluster names (disambiguate collisions)
    cluster_name: dict[int, str] = _disambiguate_cluster_names(communities) if communities else {}
    if not communities:
        lines.append("*No module-level structure detected.*\n")
    else:

        for i, comm in enumerate(communities):
            name = cluster_name[i]
            files_list = sorted(comm)[:10]
            file_preview = ", ".join(files_list)
            if len(comm) > 10:
                file_preview += f" (+{len(comm) - 10} more)"

            lines.append(f"### {i + 1}. {name}\n")
            lines.append(f"**{len(comm)} files**: `{file_preview}`\n")

            # Find outgoing edges from this cluster
            outgoing = [
                (dc, cnt)
                for (sc, dc), cnt in cluster_edges.items()
                if sc == i
            ]
            if outgoing:
                deps = sorted(outgoing, key=lambda x: x[1], reverse=True)[:3]
                dep_strs = [
                    f"→ {cluster_name[dc]} ({cnt} imports)"
                    for dc, cnt in deps
                ]
                lines.append(f"Imports from: {', '.join(dep_strs)}\n")

            # Top symbols in this cluster
            top = _top_symbols_in_files(graph, comm, limit=10)
            if top:
                lines.append("\n**Key symbols:**\n")
                for sym in top[:10]:
                    kind_icon = _kind_icon(sym["kind"])
                    doc_snippet = ""
                    if sym["doc"]:
                        doc_snippet = f" — {sym['doc'][:80]}"
                    lines.append(
                        f"- {kind_icon} `{sym['name']}` "
                        f"({sym['kind']}, {sym['file']}:{sym['line']})"
                        f"{doc_snippet}\n"
                    )
            lines.append("")

    # ── Data Flow ────────────────────────────────────────────────────
    if include_data_flow and dep_graph.get("edges"):
        lines.append("## Data Flow\n")

        # Find top 10 most-imported files (sinks) and top importers (sources)
        imported_count: dict[str, int] = {}
        importers_count: dict[str, int] = {}
        for edge in dep_graph["edges"]:
            dst = edge["to_file"]
            src = edge["from_file"]
            imported_count[dst] = imported_count.get(dst, 0) + 1
            importers_count[src] = importers_count.get(src, 0) + 1

        top_imported = sorted(imported_count.items(), key=lambda x: x[1], reverse=True)[:5]
        top_importers = sorted(importers_count.items(), key=lambda x: x[1], reverse=True)[:5]

        if top_imported:
            lines.append("### Most imported modules (hubs)\n")
            for fpath, count in top_imported:
                lines.append(f"- **{fpath}** — imported by {count} modules\n")
            lines.append("")

        if top_importers:
            lines.append("### Top importers (orchestrators)\n")
            for fpath, count in top_importers:
                lines.append(f"- **{fpath}** — imports {count} modules\n")
            lines.append("")

        # Show top cross-cluster dependencies
        if len(communities) > 1 and cluster_edges:
            lines.append("### Cross-layer dependencies\n")
            sorted_edges = sorted(cluster_edges.items(), key=lambda x: x[1], reverse=True)
            for (sc, dc), cnt in sorted_edges[:5]:
                src_name = cluster_name[sc]
                dst_name = cluster_name[dc]
                lines.append(
                    f"- **{src_name}** → **{dst_name}** ({cnt} imports)\n"
                )
            lines.append("")

    # ── Key Abstractions ─────────────────────────────────────────────
    lines.append("## Key Abstractions\n")

    # Query top symbols globally by total edge count
    # NOTE: LSP hover would add richer type info; currently uses AST docstrings only (v0.1)
    try:
        rows = graph.conn.execute(
            """SELECT s.name, s.kind, s.file, s.line, s.signature, s.doc,
                COALESCE(e_total.total_cnt, 0) AS edge_count
            FROM symbols s
            LEFT JOIN (
                SELECT sid, SUM(cnt) AS total_cnt
                FROM (
                    SELECT from_id AS sid, COUNT(*) AS cnt FROM edges GROUP BY from_id
                    UNION ALL
                    SELECT to_id AS sid, COUNT(*) AS cnt FROM edges GROUP BY to_id
                )
                GROUP BY sid
            ) e_total ON s.id = e_total.sid
            GROUP BY s.id
            ORDER BY edge_count DESC
            LIMIT 15"""
        ).fetchall()

        if rows:
            rows_by_kind: dict[str, list[tuple]] = {}
            for r in rows:
                kind = r[1]
                if kind not in rows_by_kind:
                    rows_by_kind[kind] = []
                rows_by_kind[kind].append(r)

            # Pluralization map for symbol kind headings — avoids
            # naive ".capitalize() + 's'" which produces "Classs".
            _KIND_PLURAL: dict[str, str] = {
                "class": "Classes",
                "function": "Functions",
                "method": "Methods",
                "interface": "Interfaces",
                "type": "Types",
                "module": "Modules",
            }
            for kind in ("class", "function", "method", "interface", "type", "module"):
                kind_rows = rows_by_kind.get(kind, [])
                if not kind_rows:
                    continue
                kind_label = _KIND_PLURAL.get(kind, kind.capitalize() + "s")
                lines.append(f"### {kind_label}\n")
                for r in kind_rows[:5]:
                    name, _, file, line, sig, doc, ec = r
                    icon = _kind_icon(kind)
                    desc = doc[:100] if doc else sig[:100] if sig else ""
                    lines.append(
                        f"- {icon} **`{name}`** "
                        f"({file}:{line}) — {desc}\n"
                    )
                lines.append("")
    except Exception:
        lines.append("*No symbol data available.*\n")

    # ── Cross-cutting Concerns ───────────────────────────────────────
    lines.append("## Cross-cutting Concerns\n")

    # Find cross-cutting symbols used across multiple clusters
    if len(communities) > 1 and graph:
        try:
            # Single GROUP BY query to find symbols appearing in 3+ distinct files
            # — replaces the previous O(N²) per-symbol COUNT(*) loop
            rows = graph.conn.execute(
                """SELECT name, kind, MIN(file) AS file, MIN(signature) AS sig,
                       MIN(doc) AS doc, COUNT(DISTINCT file) AS file_count
                FROM symbols
                GROUP BY name
                HAVING file_count >= 3
                ORDER BY file_count DESC
                LIMIT 20"""
            ).fetchall()

            cross_cutting: list[dict[str, Any]] = []
            for r in rows:
                name, kind, file, sig, doc, file_count = r
                if file in module_cluster:
                    cross_cutting.append({
                        "name": name,
                        "kind": kind,
                        "file": file,
                        "sig": sig or "",
                        "doc": doc or "",
                        "refs_across": file_count,
                    })

            if cross_cutting:
                for cc in cross_cutting[:10]:
                    icon = _kind_icon(cc["kind"])
                    lines.append(
                        f"- {icon} **`{cc['name']}`** "
                        f"({cc['kind']}) — referenced across "
                        f"{cc['refs_across']} files: {cc['doc'][:80]}\n"
                    )
            else:
                lines.append("*No cross-cutting symbols detected.*\n")
        except Exception:
            lines.append("*Unable to compute cross-cutting concerns.*\n")
    else:
        lines.append("*Not enough architectural layers to detect cross-cutting concerns.*\n")

    return "".join(lines)


_KIND_ICONS: dict[str, str] = {
    "function": "⚡",
    "method": "🔧",
    "class": "🏗️",
    "module": "📦",
    "interface": "🔌",
    "type": "📐",
    "enum": "🔢",
    "variable": "📌",
    "constant": "🔒",
}


def _kind_icon(kind: str) -> str:
    """Return an emoji icon for a symbol kind."""
    return _KIND_ICONS.get(kind, "•")


def cognition_map(
    graph: Any,
    ast_indexer: Any,
    project_root: str,
    include_data_flow: bool = True,
    max_files: int = 300,
) -> dict[str, Any]:
    """Generate a Codebase Cognition Map — a dense, model-friendly overview.

    Uses community detection on the module import graph to identify
    architecture layers, then extracts key symbols and data flow patterns.

    The output is a Markdown document (2,000–4,000 tokens) suitable for
    pasting into a weak model's context window.

    Args:
        graph: KnowledgeGraph instance.
        ast_indexer: ASTIndexer instance.
        project_root: Project root directory.
        include_data_flow: Whether to include data flow analysis.
        max_files: Maximum files to scan for the dependency graph.

    Returns:
        {markdown: str, stats: {files, symbols, communities, tokens_estimate}}
    """
    from codeforge_mcp.tools.dependency import ast_dependency_graph

    # Build the module dependency graph
    dep_graph = ast_dependency_graph(
        ast_indexer, project_root, focus_file=None, max_files=max_files
    )

    # Build networkx graph and detect communities
    g = _build_networkx_graph(dep_graph)
    if g is None:
        return {
            "markdown": "",
            "stats": {
                "files": 0,
                "symbols": 0,
                "communities": 0,
                "tokens_estimate": 0,
            },
            "error": "networkx not installed. Run: pip install networkx",
        }

    communities = _detect_communities(g)

    # Generate the Markdown document
    markdown = _generate_markdown(communities, dep_graph, graph, include_data_flow)

    # Estimate token count (rough: ~1.3 tokens per word)
    word_count = len(markdown.split())
    tokens_estimate = int(word_count * 1.3)

    return {
        "markdown": markdown,
        "stats": {
            "files": len(dep_graph.get("nodes", {})),
            "symbols": graph.symbol_count() if graph else 0,
            "communities": len(communities),
            "tokens_estimate": tokens_estimate,
        },
    }
