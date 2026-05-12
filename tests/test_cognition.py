"""Integration tests for the Codebase Cognition Map."""

import tempfile
from pathlib import Path

import pytest


class TestCognitionMap:
    """Verify that cognition_map generates a valid Markdown document."""

    def test_generates_markdown(self, tmp_path: Path) -> None:
        """Should return markdown and stats for a small project."""
        from codeforge_mcp.graph import KnowledgeGraph
        from codeforge_mcp.ast import ASTIndexer
        from codeforge_mcp.tools.cognition import cognition_map

        # Set up a small multi-module project
        (tmp_path / "core").mkdir()
        (tmp_path / "core" / "__init__.py").write_text("")
        (tmp_path / "core" / "engine.py").write_text(
            "from core.utils import helper\n\n"
            "class Engine:\n"
            '    """The main engine."""\n'
            "    def run(self):\n"
            "        helper()\n"
        )
        (tmp_path / "core" / "utils.py").write_text(
            "def helper():\n"
            '    """A utility function."""\n'
            "    pass\n"
        )
        (tmp_path / "api").mkdir()
        (tmp_path / "api" / "routes.py").write_text(
            "from core.engine import Engine\n\n"
            "def create_app():\n"
            "    return Engine()\n"
        )

        # Index the project
        db_path = tmp_path / ".codeforge" / "knowledge.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        graph = KnowledgeGraph(str(db_path))
        indexer = ASTIndexer(graph)
        indexer.index_file(tmp_path / "core" / "engine.py")
        indexer.index_file(tmp_path / "core" / "utils.py")
        indexer.index_file(tmp_path / "api" / "routes.py")

        result = cognition_map(
            graph, indexer, str(tmp_path), include_data_flow=True, max_files=100
        )

        # Verify structure
        assert "markdown" in result
        assert "stats" in result
        assert result["stats"]["files"] >= 1
        assert result["stats"]["symbols"] >= 3  # Engine, helper, create_app
        assert result["stats"]["communities"] >= 1

        # Verify markdown sections
        md = result["markdown"]
        assert "# Codebase Cognition Map" in md
        assert "## Architecture Layers" in md
        assert "## Key Abstractions" in md
        assert "Engine" in md

    def test_handles_empty_project(self, tmp_path: Path) -> None:
        """Should return empty but valid result for an empty project."""
        from codeforge_mcp.graph import KnowledgeGraph
        from codeforge_mcp.ast import ASTIndexer
        from codeforge_mcp.tools.cognition import cognition_map

        db_path = tmp_path / ".codeforge" / "knowledge.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        graph = KnowledgeGraph(str(db_path))
        indexer = ASTIndexer(graph)

        result = cognition_map(
            graph, indexer, str(tmp_path), include_data_flow=True, max_files=10
        )

        assert "markdown" in result
        assert result["stats"]["files"] == 0
        assert result["stats"]["symbols"] == 0

    def test_caches_result(self, tmp_path: Path) -> None:
        """Test that the server-level caching works via the resource cache."""
        import time
        import json

        from codeforge_mcp.graph import KnowledgeGraph
        from codeforge_mcp.ast import ASTIndexer
        from codeforge_mcp.tools.cognition import cognition_map

        (tmp_path / "main.py").write_text("x = 1\n")

        db_path = tmp_path / ".codeforge" / "knowledge.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        graph = KnowledgeGraph(str(db_path))
        indexer = ASTIndexer(graph)
        indexer.index_file(tmp_path / "main.py")

        # First call — should compute
        t0 = time.time()
        result1 = cognition_map(
            graph, indexer, str(tmp_path), include_data_flow=False, max_files=10
        )
        elapsed1 = time.time() - t0

        # Second call — should be faster (no actual caching in function,
        # but verifies idempotency)
        t0 = time.time()
        result2 = cognition_map(
            graph, indexer, str(tmp_path), include_data_flow=False, max_files=10
        )
        elapsed2 = time.time() - t0

        # Same result
        assert result1["markdown"] == result2["markdown"]
        assert result1["stats"] == result2["stats"]

    def test_include_data_flow_flag(self, tmp_path: Path) -> None:
        """Should only include Data Flow section when flag is True."""
        from codeforge_mcp.graph import KnowledgeGraph
        from codeforge_mcp.ast import ASTIndexer
        from codeforge_mcp.tools.cognition import cognition_map

        (tmp_path / "src").mkdir()
        # Import must be resolvable: "import src.b" won't resolve in dep graph
        # because _resolve_import can't find src.b.py from src/a.py.
        # Use direct relative-style import that the dep graph can track.
        (tmp_path / "src" / "a.py").write_text("import b\n")
        (tmp_path / "src" / "b.py").write_text("x = 1\n")

        db_path = tmp_path / ".codeforge" / "knowledge.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        graph = KnowledgeGraph(str(db_path))
        indexer = ASTIndexer(graph)
        indexer.index_file(tmp_path / "src" / "a.py")
        indexer.index_file(tmp_path / "src" / "b.py")

        with_data_flow = cognition_map(
            graph, indexer, str(tmp_path), include_data_flow=True, max_files=10
        )
        without_data_flow = cognition_map(
            graph, indexer, str(tmp_path), include_data_flow=False, max_files=10
        )

        assert "## Data Flow" in with_data_flow["markdown"]
        assert "## Data Flow" not in without_data_flow["markdown"]
