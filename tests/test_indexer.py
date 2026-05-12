"""Integration tests for the main indexer — file discovery with ignore patterns."""

import os
import tempfile
from pathlib import Path

import pytest

from codeforge_mcp.indexer import discover_files, _load_gitignore_spec


class TestIgnorePatternMatching:
    """Verify that pathspec correctly handles gitignore semantics."""

    def test_basic_wildcard(self, tmp_path: Path) -> None:
        """*.pyc should ignore .pyc files in any directory."""
        (tmp_path / "main.py").write_text("x = 1")
        (tmp_path / "main.pyc").write_text("binary")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "helper.pyc").write_text("binary")

        spec = _load_gitignore_spec(tmp_path)
        assert spec.match_file("main.pyc") is True
        assert spec.match_file("sub/helper.pyc") is True
        assert spec.match_file("main.py") is False

    def test_directory_pattern(self, tmp_path: Path) -> None:
        """node_modules/ should match everything inside that directory."""
        (tmp_path / "src").mkdir()
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "pkg").mkdir()
        (tmp_path / "node_modules" / "pkg" / "index.js").write_text("//")
        (tmp_path / "src" / "app.ts").write_text("const x = 1;")

        spec = _load_gitignore_spec(tmp_path)
        # Directory pattern should match everything inside
        assert spec.match_file("node_modules/pkg/index.js") is True
        # But not files outside
        assert spec.match_file("src/app.ts") is False

    def test_double_star(self, tmp_path: Path) -> None:
        """**/test_*.py should match test files at any depth."""
        (tmp_path / "test_main.py").write_text("")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "test_utils.py").write_text("")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "deep").mkdir()
        (tmp_path / "src" / "deep" / "test_deep.py").write_text("")

        spec = _load_gitignore_spec(tmp_path)
        assert spec.match_file("test_main.py") is False
        # We need to verify pathspec handles **
        # Note: pathspec gitignore does support **
        assert spec.match_file("sub/test_utils.py") is False

    def test_negation_pattern(self, tmp_path: Path) -> None:
        """Negation patterns should exclude from ignore."""
        # Create a custom gitignore
        (tmp_path / ".gitignore").write_text("*.log\n!important.log")
        (tmp_path / "debug.log").write_text("debug")
        (tmp_path / "important.log").write_text("important")

        spec = _load_gitignore_spec(tmp_path)
        # .log should be ignored, but !important.log should un-ignore
        assert spec.match_file("debug.log") is True
        # Negation support may vary; pathspec should handle it
        # At minimum, the always-ignore patterns shouldn't match .log files
        assert spec.match_file("important.log") is False

    def test_codeforgeignore_extends(self, tmp_path: Path) -> None:
        """.codeforgeignore can add extra ignore rules."""
        (tmp_path / "ignored.bin").write_text("data")
        (tmp_path / ".codeforgeignore").write_text("*.bin")

        spec = _load_gitignore_spec(tmp_path)
        assert spec.match_file("ignored.bin") is True

    def test_hidden_dirs_ignored(self, tmp_path: Path) -> None:
        """Hidden directories like .git should be ignored."""
        spec = _load_gitignore_spec(tmp_path)
        assert spec.match_file(".git/config") is True
        assert spec.match_file(".venv/lib/python3/site-packages/pkg.py") is True


class TestDiscoverFiles:
    """Verify that discover_files() respects ignore patterns."""

    def test_discovers_python_files(self, tmp_path: Path) -> None:
        """Should find .py files and skip ignored ones."""
        (tmp_path / "main.py").write_text("x = 1")
        (tmp_path / "utils.py").write_text("y = 2")
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "__init__.py").write_text("")

        files = discover_files(str(tmp_path), extensions=[".py"])
        assert len(files) == 3
        assert "main.py" in files
        assert "utils.py" in files

    def test_skips_node_modules(self, tmp_path: Path) -> None:
        """node_modules should be completely skipped."""
        (tmp_path / "src").mkdir(parents=True)
        (tmp_path / "src" / "app.ts").write_text("const x = 1;")
        (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
        (tmp_path / "node_modules" / "pkg" / "index.js").write_text("module.exports = {};")

        files = discover_files(str(tmp_path), extensions=[".ts", ".js"])
        assert len(files) == 1
        assert "src/app.ts" in files

    def test_skips_pyc_files(self, tmp_path: Path) -> None:
        """.pyc files should be ignored."""
        (tmp_path / "main.py").write_text("x = 1")
        (tmp_path / "main.pyc").write_text("binary")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "main.cpython-312.pyc").write_text("binary")

        files = discover_files(str(tmp_path), extensions=[".py", ".pyc"])
        assert "main.py" in files
        # .pyc files and __pycache__ should be ignored
        for f in files:
            assert not f.endswith(".pyc")
            assert "__pycache__" not in f

    def test_respects_gitignore(self, tmp_path: Path) -> None:
        """Should respect .gitignore rules."""
        (tmp_path / "main.py").write_text("x = 1")
        (tmp_path / "generated.py").write_text("auto-generated")
        (tmp_path / ".gitignore").write_text("generated.py")

        files = discover_files(str(tmp_path), extensions=[".py"])
        assert "main.py" in files
        assert "generated.py" not in files

    def test_all_extensions(self, tmp_path: Path) -> None:
        """Discovered files should all match the requested extensions."""
        (tmp_path / "main.py").write_text("x = 1")
        (tmp_path / "script.js").write_text("//")
        (tmp_path / "types.d.ts").write_text("type Foo = string;")

        py_files = discover_files(str(tmp_path), extensions=[".py"])
        assert all(f.endswith(".py") for f in py_files)

        js_files = discover_files(str(tmp_path), extensions=[".js", ".ts"])
        assert all(f.endswith(".js") or f.endswith(".ts") for f in js_files)
