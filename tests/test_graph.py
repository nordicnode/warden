"""Integration tests for the knowledge graph (SQLite store)."""

from pathlib import Path

import pytest

from codeforge_mcp.graph import KnowledgeGraph


class TestKnowledgeGraph:
    """Verify CRUD operations on the knowledge graph."""

    @pytest.fixture
    def graph(self, tmp_path: Path) -> KnowledgeGraph:
        db = tmp_path / "test.db"
        return KnowledgeGraph(str(db))

    def test_upsert_symbol(self, graph: KnowledgeGraph) -> None:
        sid = graph.upsert_symbol("test_func", "function", "test.py", line=10)
        assert sid > 0

        sym = graph.get_symbol("test_func")
        assert sym is not None
        assert sym["name"] == "test_func"
        assert sym["kind"] == "function"
        assert sym["line"] == 10

    def test_upsert_is_idempotent(self, graph: KnowledgeGraph) -> None:
        sid1 = graph.upsert_symbol("func", "function", "test.py", line=5)
        sid2 = graph.upsert_symbol("func", "function", "test.py", line=10)
        assert sid1 == sid2

        # Line should be updated
        sym = graph.get_symbol("func")
        assert sym is not None
        assert sym["line"] == 10

    def test_get_symbol_by_file_scope(self, graph: KnowledgeGraph) -> None:
        graph.upsert_symbol("helper", "function", "a.py", line=1)
        graph.upsert_symbol("helper", "function", "b.py", line=5)

        sym = graph.get_symbol("helper", file="a.py")
        assert sym is not None
        assert sym["file"] == "a.py"

    def test_delete_symbols_in_file(self, graph: KnowledgeGraph) -> None:
        graph.upsert_symbol("func1", "function", "mod.py", line=1)
        graph.upsert_symbol("func2", "function", "mod.py", line=5)
        graph.upsert_symbol("func3", "function", "other.py", line=1)

        count = graph.delete_symbols_in_file("mod.py")
        assert count == 2

        assert graph.get_symbol("func1") is None
        assert graph.get_symbol("func2") is None
        assert graph.get_symbol("func3") is not None

    def test_symbol_count(self, graph: KnowledgeGraph) -> None:
        assert graph.symbol_count() == 0
        graph.upsert_symbol("a", "function", "x.py")
        graph.upsert_symbol("b", "class", "y.py")
        assert graph.symbol_count() == 2

    def test_file_count(self, graph: KnowledgeGraph) -> None:
        graph.upsert_symbol("a", "function", "x.py")
        graph.upsert_symbol("b", "function", "x.py")
        graph.upsert_symbol("c", "function", "y.py")
        assert graph.file_count() == 2

    def test_symbols_in_file(self, graph: KnowledgeGraph) -> None:
        graph.upsert_symbol("first", "function", "mod.py", line=1)
        graph.upsert_symbol("second", "function", "mod.py", line=10)

        syms = graph.symbols_in_file("mod.py")
        assert len(syms) == 2
        assert syms[0]["line"] <= syms[1]["line"]


class TestSearch:
    """Verify BM25 search functionality."""

    @pytest.fixture
    def graph(self, tmp_path: Path) -> KnowledgeGraph:
        db = tmp_path / "test.db"
        g = KnowledgeGraph(str(db))
        g.upsert_symbol("authenticate_user", "function", "auth.py", line=10)
        g.upsert_symbol("authenticate_admin", "function", "auth.py", line=50)
        g.upsert_symbol("logout", "function", "auth.py", line=80)
        g.upsert_symbol("login_form", "function", "login.py", line=5)
        return g

    def test_exact_match_ranks_highest(self, graph: KnowledgeGraph) -> None:
        results = graph.search_symbols("authenticate_user")
        assert len(results) > 0
        assert results[0]["name"] == "authenticate_user"

    def test_partial_match(self, graph: KnowledgeGraph) -> None:
        results = graph.search_symbols("authenticate")
        assert len(results) >= 2
        names = [r["name"] for r in results]
        assert "authenticate_user" in names
        assert "authenticate_admin" in names

    def test_no_match(self, graph: KnowledgeGraph) -> None:
        results = graph.search_symbols("nonexistent_symbol_xyz")
        assert results == []

    def test_limit(self, graph: KnowledgeGraph) -> None:
        results = graph.search_symbols("auth", limit=1)
        assert len(results) <= 1


class TestEdges:
    """Verify edge (call graph) operations."""

    @pytest.fixture
    def graph(self, tmp_path: Path) -> KnowledgeGraph:
        db = tmp_path / "test.db"
        g = KnowledgeGraph(str(db))
        # Create: main() calls helper(), helper() calls util()
        self.a_id = g.upsert_symbol("main", "function", "main.py", line=1)
        self.b_id = g.upsert_symbol("helper", "function", "helper.py", line=1)
        self.c_id = g.upsert_symbol("util", "function", "util.py", line=1)
        g.add_edge(self.a_id, self.b_id, "calls")
        g.add_edge(self.b_id, self.c_id, "calls")
        return g

    def test_upstream(self, graph: KnowledgeGraph) -> None:
        # Who calls helper? → main
        callers = graph.upstream(self.b_id, depth=1)
        caller_names = [c["name"] for c in callers]
        assert "main" in caller_names

    def test_downstream(self, graph: KnowledgeGraph) -> None:
        # What does main call? → helper
        callees = graph.downstream(self.a_id, depth=1)
        callee_names = [c["name"] for c in callees]
        assert "helper" in callee_names

    def test_call_graph_both(self, graph: KnowledgeGraph) -> None:
        cg = graph.call_graph(self.b_id, direction="both", depth=2)
        assert "upstream" in cg
        assert "downstream" in cg
        assert len(cg["upstream"]) >= 1  # main
        assert len(cg["downstream"]) >= 1  # util

    def test_depth_limit(self, graph: KnowledgeGraph) -> None:
        # With depth=1 from main: only helper, not util
        callees = graph.downstream(self.a_id, depth=1)
        callee_names = [c["name"] for c in callees]
        assert "helper" in callee_names
        assert "util" not in callee_names  # One hop deeper

    def test_affected_modules(self, graph: KnowledgeGraph) -> None:
        modules = graph.affected_modules(self.b_id, depth=2)
        assert "main.py" in modules
        assert "helper.py" in modules
        assert "util.py" in modules


class TestDecisions:
    """Verify decision recording."""

    @pytest.fixture
    def graph(self, tmp_path: Path) -> KnowledgeGraph:
        db = tmp_path / "test.db"
        return KnowledgeGraph(str(db))

    def test_add_decision(self, graph: KnowledgeGraph) -> None:
        did = graph.add_decision("Use SQLite", "Lightweight and persistent", ["db.py"])
        assert did > 0

        decisions = graph.recent_decisions()
        assert len(decisions) == 1
        assert decisions[0]["title"] == "Use SQLite"
        assert "SQLite" in decisions[0]["title"]

    def test_recent_decisions_limit(self, graph: KnowledgeGraph) -> None:
        for i in range(5):
            graph.add_decision(f"Decision {i}", "Test", [])

        results = graph.recent_decisions(limit=3)
        assert len(results) == 3


class TestKnowledgeScore:
    """Verify knowledge score heuristic."""

    @pytest.fixture
    def graph(self, tmp_path: Path) -> KnowledgeGraph:
        return KnowledgeGraph(str(tmp_path / "test.db"))

    def test_empty_graph_zero_score(self, graph: KnowledgeGraph) -> None:
        assert graph.knowledge_score() == 0.0

    def test_score_with_edges(self, graph: KnowledgeGraph) -> None:
        a = graph.upsert_symbol("a", "function", "x.py")
        b = graph.upsert_symbol("b", "function", "x.py")
        graph.add_edge(a, b, "calls")
        score = graph.knowledge_score()
        # New formula: sym_score=log10(2)*20≈6.02 + edge_score=0.5*33=16.5 → 22.52
        assert score > 0.0
        assert score < 100.0

    def test_brief(self, graph: KnowledgeGraph) -> None:
        graph.upsert_symbol("a", "function", "x.py")
        brief = graph.brief()
        assert brief["file_count"] == 1
        assert brief["symbol_count"] == 1
        assert brief["knowledge_score"] == 0.0
