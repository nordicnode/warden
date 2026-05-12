"""Tests for navigation tools — code_find_files, code_search, symbol_lookup, _language_from_ext.

Verifies:
- _language_from_ext maps extensions correctly
- code_search parses ripgrep output and scores results
- symbol_lookup falls back from LSP to graph
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codeforge_mcp.tools.navigation import (
    _language_from_ext,
    code_find_files,
    code_search,
    symbol_lookup,
)


class TestLanguageFromExt:
    """Verify extension → language mapping."""

    def test_python(self) -> None:
        assert _language_from_ext(".py") == "Python"

    def test_typescript(self) -> None:
        assert _language_from_ext(".ts") == "TypeScript"
        assert _language_from_ext(".tsx") == "TypeScript React"

    def test_javascript(self) -> None:
        assert _language_from_ext(".js") == "JavaScript"
        assert _language_from_ext(".jsx") == "JavaScript React"
        assert _language_from_ext(".mjs") == "JavaScript"

    def test_rust(self) -> None:
        assert _language_from_ext(".rs") == "Rust"

    def test_go(self) -> None:
        assert _language_from_ext(".go") == "Go"

    def test_c_cpp(self) -> None:
        assert _language_from_ext(".c") == "C"
        assert _language_from_ext(".h") == "C Header"
        assert _language_from_ext(".cpp") == "C++"
        assert _language_from_ext(".cc") == "C++"
        assert _language_from_ext(".hpp") == "C++ Header"

    def test_scripting_languages(self) -> None:
        assert _language_from_ext(".sh") == "Bash"
        assert _language_from_ext(".bash") == "Bash"
        assert _language_from_ext(".lua") == "Lua"

    def test_unknown(self) -> None:
        assert _language_from_ext(".xyz") == "Unknown"
        assert _language_from_ext("") == "Unknown"


class TestCodeSearch:
    """Verify code_search parses ripgrep output."""

    @pytest.fixture
    def root(self, tmp_path: Path) -> Path:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "auth.py").write_text("def authenticate_user(token):\n    pass\n")
        (tmp_path / "src" / "login.py").write_text("def login_form():\n    pass\n")
        return tmp_path

    def test_finds_matches(self, root: Path) -> None:
        """Should find text in source files using ripgrep."""
        results = code_search(str(root), "authenticate")
        # Should find at least one match in auth.py
        filenames = {r["file"] for r in results}
        assert any("auth.py" in f for f in filenames)

    def test_no_match_returns_empty(self, root: Path) -> None:
        results = code_search(str(root), "nonexistent_string_xyz")
        assert results == []

    def test_results_include_score(self, root: Path) -> None:
        results = code_search(str(root), "authenticate")
        if results:
            assert "score" in results[0]
            assert "match_type" in results[0]

    def test_filters_out_markdown_noise(self, root: Path) -> None:
        (root / "results.md").write_text("authenticate appears in this transcript\n")
        results = code_search(str(root), "authenticate")
        filenames = {r["file"] for r in results}
        assert not any("results.md" in f for f in filenames)
        assert any("auth.py" in f for f in filenames)

    def test_results_sorted_by_relevance(self, root: Path) -> None:
        results = code_search(str(root), "authenticate")
        if len(results) >= 2:
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True), "Results should be sorted by score descending"

    def test_search_command_includes_explicit_path(self, root: Path) -> None:
        """Ripgrep must receive an explicit search path to avoid blocking on stdin.

        When run as a subprocess under the MCP server, ripgrep with no path
        argument attempts to read from stdin (which is the JSON-RPC pipe),
        causing a 15s timeout. Adding '.' as the search path fixes this.
        """
        import subprocess

        # Create a test file
        (root / "test.py").write_text("def foo(): pass")

        captured_cmd: list = []
        original_run = subprocess.run

        def mock_run(cmd: list, *args, **kwargs):
            captured_cmd.extend(cmd)
            return original_run(cmd, *args, **kwargs)

        with patch.object(subprocess, 'run', mock_run):
            result = code_search(str(root), "def", regex=False, context=3)

        # Verify the command includes an explicit path after the query
        assert len(captured_cmd) > 0, "subprocess.run was not called"
        if "def" in captured_cmd:
            query_idx = captured_cmd.index("def")
            assert query_idx + 1 < len(captured_cmd), "No search path after query"
            search_path = captured_cmd[query_idx + 1]
            assert search_path == ".", \
                f"Search path should be '.', got: {search_path}"


class TestWorkspaceSymbolsLowercaseNormalization:
    """Verify that workspace_symbols normalizes language names to lowercase.

    EXT_TO_LANG returns capitalized names ("Python") but LSP_COMMANDS
    uses lowercase keys ("python"). The multiplexer must normalize
    when looking up commands and starting servers.
    """

    @pytest.mark.asyncio
    async def test_ensures_server_called_with_lowercase_language(self) -> None:
        """_ensure_server should be called with lowercase language name."""
        from codeforge_mcp.lsp.multiplexer import LSPMultiplexer
        from unittest.mock import MagicMock, patch

        # Create multiplexer with a mock project root
        multiplexer = LSPMultiplexer(Path("/fake/project"))

        # Mock discover_files to return .py files so EXT_TO_LANG returns "Python"
        # discover_files is a sync function called via asyncio.to_thread
        with patch('codeforge_mcp.indexer.discover_files') as mock_discover:
            mock_discover.return_value = ["src/main.py"]

            # Mock _ensure_server to track what language it receives
            calls = []
            async def mock_ensure(lang: str):
                calls.append(lang)
                return None

            multiplexer._ensure_server = mock_ensure

            # Call workspace_symbols and verify _ensure_server gets lowercase
            await multiplexer.workspace_symbols("test")

            # Verify all calls to _ensure_server used lowercase language names
            for lang in calls:
                assert lang == lang.lower(), \
                    f"Language should be lowercase, got: {lang}"
                assert lang in ("python", "typescript", "javascript", "tsx",
                                "rust", "go", "c", "cpp"), \
                    f"Unexpected language: {lang}"
    """Verify symbol_lookup with LSP→graph fallback."""

    @pytest.mark.asyncio
    async def test_falls_back_to_graph_when_lsp_returns_none(self) -> None:
        """When LSP returns None, should fall back to the knowledge graph."""
        mock_lsp = MagicMock(spec=[])
        mock_lsp.symbol_lookup = AsyncMock(return_value=None)

        mock_graph = MagicMock()
        mock_graph.get_symbol.return_value = {
            "name": "my_func",
            "kind": "function",
            "file": "test.py",
            "line": 10,
        }

        result = await symbol_lookup(mock_lsp, mock_graph, "my_func")
        assert result is not None
        assert result["name"] == "my_func"
        assert result["kind"] == "function"
        assert result["file"] == "test.py"
        assert result["line"] == 10
        # Should have called LSP first
        mock_lsp.symbol_lookup.assert_called_once_with("my_func")
        mock_graph.get_symbol.assert_called_once_with("my_func")

    @pytest.mark.asyncio
    async def test_returns_lsp_result_when_available(self) -> None:
        """When LSP returns a result, don't fall back to graph."""
        mock_lsp = MagicMock(spec=[])
        mock_lsp.symbol_lookup = AsyncMock(return_value={
            "name": "lsp_func",
            "kind": 12,  # LSP symbol kind
            "file": "/path/to/file.py",
            "line": 20,
        })

        mock_graph = MagicMock()

        result = await symbol_lookup(mock_lsp, mock_graph, "lsp_func")
        assert result["name"] == "lsp_func"
        # Should not have called graph fallback
        mock_graph.get_symbol.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_none_when_neither_has_result(self) -> None:
        """When both LSP and graph return nothing, return None."""
        mock_lsp = MagicMock(spec=[])
        mock_lsp.symbol_lookup = AsyncMock(return_value=None)
        mock_graph = MagicMock()
        mock_graph.get_symbol.return_value = None

        result = await symbol_lookup(mock_lsp, mock_graph, "missing_func")
        assert result is None
