"""Tests for understanding tools — call_graph, impact_analysis, ast_query, cycle detection.

Verifies:
- call_graph resolves symbols and returns upstream/downstream
- impact_analysis computes risk scores correctly
- call_graph falls back to search_symbols when exact match fails
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codeforge_mcp.tools.understanding import (
    ast_query,
    call_graph,
    impact_analysis,
    _detect_cycles,
)
from codeforge_mcp.graph import KnowledgeGraph
from codeforge_mcp.ast.indexer import ASTIndexer


class TestCallGraph:
    """Verify call_graph traverses the graph correctly."""

    @pytest.fixture
    def graph(self, tmp_path: Path) -> KnowledgeGraph:
        db = tmp_path / "test.db"
        g = KnowledgeGraph(str(db))
        # Create: main() calls helper(), helper() calls util()
        self.main_id = g.upsert_symbol("main", "function", "main.py", line=1)
        self.helper_id = g.upsert_symbol("helper", "function", "helper.py", line=1)
        self.util_id = g.upsert_symbol("util", "function", "util.py", line=1)
        g.add_edge(self.main_id, self.helper_id, "calls")
        g.add_edge(self.helper_id, self.util_id, "calls")
        return g

    def test_exact_symbol_match(self, graph: KnowledgeGraph) -> None:
        result = call_graph(graph, "main", direction="both", depth=2)
        assert "upstream" in result
        assert "downstream" in result
        assert "cycles" in result
        # main calls helper
        downstream_names = [d["name"] for d in result["downstream"]]
        assert "helper" in downstream_names

    def test_fuzzy_fallback(self, graph: KnowledgeGraph) -> None:
        """When exact match fails, fall back to search_symbols."""
        result = call_graph(graph, "helpe", direction="downstream", depth=1)
        # Should find "helper" via fuzzy search
        downstream_names = [d["name"] for d in result["downstream"]]
        assert "util" in downstream_names  # helper calls util

    def test_no_match_returns_empty(self, graph: KnowledgeGraph) -> None:
        result = call_graph(graph, "nonexistent", direction="both", depth=2)
        assert result["upstream"] == []
        assert result["downstream"] == []
        assert result["cycles"] == []

    def test_upstream_only(self, graph: KnowledgeGraph) -> None:
        result = call_graph(graph, "helper", direction="upstream", depth=1)
        caller_names = [c["name"] for c in result["upstream"]]
        assert "main" in caller_names
        assert "downstream" not in result  # Only upstream requested

    def test_downstream_only(self, graph: KnowledgeGraph) -> None:
        result = call_graph(graph, "main", direction="downstream", depth=2)
        callee_names = [c["name"] for c in result["downstream"]]
        assert "helper" in callee_names
        assert "util" in callee_names
        assert "upstream" not in result


class TestImpactAnalysis:
    """Verify impact_analysis computes risk and blast radius."""

    @pytest.fixture
    def graph(self, tmp_path: Path) -> KnowledgeGraph:
        db = tmp_path / "test.db"
        g = KnowledgeGraph(str(db))
        # Create a heavily-used symbol with many callers
        self.util_id = g.upsert_symbol("util_func", "function", "util.py", line=1)
        for i in range(5):
            caller_name = f"caller_{i}"
            cid = g.upsert_symbol(caller_name, "function", f"caller{i}.py", line=1)
            g.add_edge(cid, self.util_id, "calls")
        # Test file caller
        test_id = g.upsert_symbol("test_util", "function", "tests/test_util.py", line=1)
        g.add_edge(test_id, self.util_id, "calls")
        return g

    def test_risk_high_with_many_callers(self, graph: KnowledgeGraph) -> None:
        result = impact_analysis(graph, None, "util_func")
        assert result["callers"] >= 5
        # 5 callers + util itself (which also makes it call itself perhaps)
        # but risk scoring: callers > 10 → HIGH, > 3 → MED
        assert result["risk"] in ("MED", "HIGH")

    def test_risk_low_with_few_callers(self, graph: KnowledgeGraph) -> None:
        # Add a new symbol with only 1 caller
        sid = graph.upsert_symbol("low_impact", "function", "low.py", line=1)
        cid = graph.upsert_symbol("single_caller", "function", "single.py", line=1)
        graph.add_edge(cid, sid, "calls")

        result = impact_analysis(graph, None, "low_impact")
        assert result["risk"] == "LOW"
        assert result["callers"] >= 1

    def test_detects_tests_among_callers(self, graph: KnowledgeGraph) -> None:
        result = impact_analysis(graph, None, "util_func")
        assert len(result["tests"]) >= 1
        test_names = [t["name"] for t in result["tests"]]
        assert any("test" in name.lower() for name in test_names)

    def test_no_match_returns_low_risk(self, graph: KnowledgeGraph) -> None:
        result = impact_analysis(graph, None, "nonexistent_symbol_xyz")
        assert result["risk"] == "LOW"
        assert result["callers"] == 0
        assert result["module_count"] == 0

    def test_includes_modules(self, graph: KnowledgeGraph) -> None:
        result = impact_analysis(graph, None, "util_func")
        assert "modules" in result
        assert result["module_count"] >= 1


class TestAstQuery:
    """Verify ast_query runs tree-sitter queries and returns correct results."""

    @pytest.fixture
    def sample_file(self, tmp_path: Path) -> Path:
        f = tmp_path / "sample.py"
        f.write_text(
            "def greet(name):\n"
            "    return f'hello, {name}'\n"
            "\n"
            "class Greeter:\n"
            "    def __init__(self, greeting):\n"
            "        self.greeting = greeting\n"
            "\n"
            "    def greet(self, name):\n"
            "        return f'{self.greeting}, {name}'\n"
        )
        return f

    def test_ast_query_keyword_function(self, sample_file: Path) -> None:
        pytest.importorskip("tree_sitter")
        db = sample_file.parent / "graph.db"
        graph = KnowledgeGraph(str(db))
        indexer = ASTIndexer(graph)

        result = ast_query(indexer, str(sample_file), "function")
        assert "error" not in result
        names = [r["name"] for r in result]
        assert "greet" in names

    def test_ast_query_keyword_class(self, sample_file: Path) -> None:
        pytest.importorskip("tree_sitter")
        db = sample_file.parent / "graph.db"
        graph = KnowledgeGraph(str(db))
        indexer = ASTIndexer(graph)

        result = ast_query(indexer, str(sample_file), "class")
        assert "error" not in result
        names = [r["name"] for r in result]
        assert "Greeter" in names

    def test_ast_query_keyword_all(self, sample_file: Path) -> None:
        pytest.importorskip("tree_sitter")
        db = sample_file.parent / "graph.db"
        graph = KnowledgeGraph(str(db))
        indexer = ASTIndexer(graph)

        result = ast_query(indexer, str(sample_file), "all")
        assert "error" not in result
        assert len(result) > 0
        # Every result must have type, name, line, kind
        for r in result:
            assert "type" in r
            assert "line" in r

    def test_ast_query_sexp_query(self, sample_file: Path) -> None:
        pytest.importorskip("tree_sitter")
        db = sample_file.parent / "graph.db"
        graph = KnowledgeGraph(str(db))
        indexer = ASTIndexer(graph)

        result = ast_query(indexer, str(sample_file), "(function_definition) @func")
        assert "error" not in result
        # S-expression results should now have {type, name, line, kind}
        for r in result:
            assert "type" in r
            assert "line" in r
            assert "kind" in r

    def test_ast_query_nonexistent_file(self, sample_file: Path) -> None:
        pytest.importorskip("tree_sitter")
        db = sample_file.parent / "graph.db"
        graph = KnowledgeGraph(str(db))
        indexer = ASTIndexer(graph)

        result = ast_query(indexer, "/nonexistent/file.py", "function")
        assert len(result) == 1
        assert result[0]["type"] == "error"

    def test_ast_query_unknown_keyword(self, sample_file: Path) -> None:
        pytest.importorskip("tree_sitter")
        db = sample_file.parent / "graph.db"
        graph = KnowledgeGraph(str(db))
        indexer = ASTIndexer(graph)

        result = ast_query(indexer, str(sample_file), "not_a_keyword")
        assert len(result) == 1
        assert result[0]["type"] == "error"
        assert "Unknown query keyword" in result[0]["message"]


class TestCycleDetection:
    """Verify _detect_cycles finds cycles in the call graph."""

    @pytest.fixture
    def graph(self, tmp_path: Path) -> KnowledgeGraph:
        db = tmp_path / "test.db"
        g = KnowledgeGraph(str(db))
        self.a_id = g.upsert_symbol("cycle_a", "function", "cycle.py", line=1)
        self.b_id = g.upsert_symbol("cycle_b", "function", "cycle.py", line=2)
        self.c_id = g.upsert_symbol("cycle_c", "function", "cycle.py", line=3)
        g.add_edge(self.a_id, self.b_id, "calls")
        g.add_edge(self.b_id, self.c_id, "calls")
        g.add_edge(self.c_id, self.a_id, "calls")
        return g

    def test_three_node_cycle(self, graph: KnowledgeGraph) -> None:
        cycles = _detect_cycles(graph, self.a_id)
        assert len(cycles) >= 1
        cycle_names = cycles[0]
        assert "cycle_a" in cycle_names
        assert "cycle_b" in cycle_names
        assert "cycle_c" in cycle_names

    def test_no_cycle_returns_empty(self, graph: KnowledgeGraph) -> None:
        # Remove one edge to break the cycle
        graph.conn.execute("DELETE FROM edges WHERE from_id = ? AND to_id = ?",
                           (self.c_id, self.a_id))
        graph.conn.commit()
        cycles = _detect_cycles(graph, self.a_id)
        assert cycles == []
