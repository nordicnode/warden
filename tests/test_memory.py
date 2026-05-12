"""Tests for memory tools — decision_record, brief.

Verifies:
- decision_record writes to graph and creates .codeforge/decisions.md
- decision_record appends to existing decisions.md
- brief delegates to graph.brief()
"""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from codeforge_mcp.tools.memory import decision_record, brief


class TestDecisionRecord:
    """Verify decision_record writes to graph and disk."""

    @pytest.fixture
    def mock_graph(self) -> MagicMock:
        g = MagicMock()
        g.add_decision.return_value = "dec_001"
        return g

    def test_records_decision_to_graph(self, tmp_path: Path, mock_graph: MagicMock) -> None:
        result = decision_record(
            tmp_path,
            mock_graph,
            title="Use PostgreSQL",
            why="Better for relational data",
            files=["db/schema.sql"],
        )

        mock_graph.add_decision.assert_called_once_with(
            "Use PostgreSQL",
            "Better for relational data",
            ["db/schema.sql"],
        )
        assert result["id"] == "dec_001"
        assert result["title"] == "Use PostgreSQL"
        assert "date" in result

    def test_writes_decisions_file(self, tmp_path: Path, mock_graph: MagicMock) -> None:
        decision_record(
            tmp_path,
            mock_graph,
            title="Use Redis for caching",
            why="Low latency needed",
            files=[],
        )

        decisions_file = tmp_path / ".codeforge" / "decisions.md"
        assert decisions_file.exists()
        content = decisions_file.read_text()
        assert "Use Redis for caching" in content
        assert "Low latency needed" in content
        assert "# Codeforge Decisions" in content
        assert "dec_001" in content

    def test_appends_to_existing_decisions(self, tmp_path: Path, mock_graph: MagicMock) -> None:
        decisions_dir = tmp_path / ".codeforge"
        decisions_dir.mkdir(parents=True)
        decisions_file = decisions_dir / "decisions.md"
        decisions_file.write_text("# Codeforge Decisions\n\n## Earlier decision\n\n")

        mock_graph.add_decision.return_value = "dec_002"

        decision_record(
            tmp_path,
            mock_graph,
            title="Second decision",
            why="Another reason",
        )

        content = decisions_file.read_text()
        assert "Earlier decision" in content
        assert "Second decision" in content
        # Two decision entries (header is "# Codeforge Decisions", not "## ")
        assert content.count("## ") == 2

    def test_writes_file_list(self, tmp_path: Path, mock_graph: MagicMock) -> None:
        decision_record(
            tmp_path,
            mock_graph,
            title="Refactor",
            why="Clean up",
            files=["src/main.py", "tests/test_main.py"],
        )

        content = (tmp_path / ".codeforge" / "decisions.md").read_text()
        assert "src/main.py" in content
        assert "tests/test_main.py" in content


class TestBrief:
    """Verify brief delegates to graph.brief()."""

    def test_brief_calls_graph(self) -> None:
        mock_graph = MagicMock()
        mock_graph.brief.return_value = {"symbol_count": 42, "file_count": 10}

        result = brief(mock_graph)
        assert result == {"symbol_count": 42, "file_count": 10}
        mock_graph.brief.assert_called_once()
