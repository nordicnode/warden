"""Tests for file operation tools — read_file, write_file, list_directory, git_diff.

Verifies:
- Path traversal prevention (Bug #5 fix: is_relative_to checks)
- write_file rejects .git/.codeforge protected paths
- read_file line ranges, encoding fallback
- list_directory recursion depth, hidden files, deduplication
"""

import tempfile
from pathlib import Path

import pytest

from codeforge_mcp.tools.file_ops import read_file, write_file, list_directory


class TestPathTraversalPrevention:
    """Bug #5: path traversal via similarly-named sibling directories."""

    @pytest.fixture
    def root(self, tmp_path: Path) -> Path:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("x = 1")
        return tmp_path

    def test_read_file_rejects_traversal(self, root: Path) -> None:
        """Path outside project root must be rejected."""
        # Create a sibling directory with a similar name prefix
        sibling = root.parent / (root.name + "_secret")
        sibling.mkdir(exist_ok=True)
        (sibling / "passwords.txt").write_text("secret")

        # Try to read ../proj_secret/passwords.txt
        result = read_file(str(root), "../" + root.name + "_secret/passwords.txt")
        assert "error" in result
        assert "traversal" in result["error"].lower()
        assert result["content"] == ""

    def test_read_file_rejects_absolute_outside(self, root: Path) -> None:
        """Absolute path outside project root must be rejected."""
        outside = root.parent / "outside.txt"
        outside.write_text("outside")

        result = read_file(str(root), str(outside))
        # Not relative, so it gets joined with root then resolved
        assert "error" in result
        assert "traversal" in result["error"].lower()

    def test_write_file_rejects_traversal(self, root: Path) -> None:
        """Write outside project root must be rejected."""
        sibling = root.parent / (root.name + "_secret")
        sibling.mkdir(exist_ok=True)

        result = write_file(str(root), "../" + root.name + "_secret/passwords.txt", "secret")
        assert "error" in result
        assert not result["written"]

    def test_list_directory_rejects_traversal(self, root: Path) -> None:
        """List outside project root must be rejected."""
        sibling = root.parent / (root.name + "_secret")
        sibling.mkdir(exist_ok=True)

        result = list_directory(str(root), "../" + root.name + "_secret")
        assert "error" in result
        assert "traversal" in result["error"].lower()

    def test_read_file_allows_normal_path(self, root: Path) -> None:
        """Normal path inside project root should succeed."""
        result = read_file(str(root), "src/main.py")
        assert "error" not in result
        assert result["content"] == "x = 1"
        assert result["total_lines"] == 1


class TestReadFileLineRange:
    """Verify read_file line range slicing."""

    @pytest.fixture
    def root(self, tmp_path: Path) -> Path:
        f = tmp_path / "lines.py"
        f.write_text("line1\nline2\nline3\nline4\nline5\nline6")
        return tmp_path

    def test_read_all_lines(self, root: Path) -> None:
        result = read_file(str(root), "lines.py")
        assert result["total_lines"] == 6
        assert "line1" in result["content"]
        assert "line6" in result["content"]

    def test_read_line_range(self, root: Path) -> None:
        result = read_file(str(root), "lines.py", start_line=2, end_line=4)
        lines = result["content"].split("\n")
        assert len(lines) == 3  # lines 2,3,4
        assert lines[0] == "line2"
        assert lines[2] == "line4"

    def test_read_from_start_line(self, root: Path) -> None:
        result = read_file(str(root), "lines.py", start_line=3)
        lines = result["content"].split("\n")
        assert lines[0] == "line3"

    def test_read_to_end_line(self, root: Path) -> None:
        result = read_file(str(root), "lines.py", end_line=3)
        lines = result["content"].split("\n")
        assert len(lines) == 3
        assert lines[-1] == "line3"

    def test_nonexistent_file(self, root: Path) -> None:
        result = read_file(str(root), "nonexistent.py")
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_includes_hash(self, root: Path) -> None:
        result = read_file(str(root), "lines.py")
        assert "hash" in result
        assert len(result["hash"]) == 16  # XXH3_64 hex


class TestWriteFile:
    """Verify write_file behavior."""

    @pytest.fixture
    def root(self, tmp_path: Path) -> Path:
        return tmp_path

    def test_write_new_file(self, root: Path) -> None:
        result = write_file(str(root), "new.txt", "hello world")
        assert result["written"] is True
        assert (root / "new.txt").read_text() == "hello world"

    def test_write_creates_parent_dirs(self, root: Path) -> None:
        result = write_file(str(root), "deep/nested/file.txt", "content")
        assert result["written"] is True
        assert (root / "deep" / "nested" / "file.txt").read_text() == "content"

    def test_write_creates_parent_dirs_disabled(self, root: Path) -> None:
        result = write_file(str(root), "deep/file.txt", "content", create_dirs=False)
        assert result["written"] is False
        assert "error" in result

    def test_write_rejects_dot_git(self, root: Path) -> None:
        (root / ".git").mkdir()
        result = write_file(str(root), ".git/config", "malicious")
        assert result["written"] is False
        assert "protected" in result["error"].lower()

    def test_write_rejects_dot_codeforge(self, root: Path) -> None:
        (root / ".codeforge").mkdir()
        result = write_file(str(root), ".codeforge/config.json", "{}")
        assert result["written"] is False
        assert "protected" in result["error"].lower()

    def test_write_overwrites_existing(self, root: Path) -> None:
        (root / "file.txt").write_text("old")
        result = write_file(str(root), "file.txt", "new")
        assert result["written"] is True
        assert (root / "file.txt").read_text() == "new"

    def test_write_includes_hash(self, root: Path) -> None:
        result = write_file(str(root), "hashed.txt", "test")
        assert "hash" in result
        assert len(result["hash"]) == 16


class TestWriteFileEscapeWarning:
    """Test literal \\n detection in write_file (prevents JSON encoding mistakes)."""

    @pytest.fixture
    def root(self, tmp_path: Path) -> Path:
        return tmp_path

    def test_escape_warning_triggered_for_literal_backslash_n_in_code(
        self, root: Path
    ) -> None:
        """Content with >2 literal \\n sequences and code chars should trigger warning."""
        # This simulates a client sending JSON with double backslash-n
        # e.g. "def test():\\n pass\\n" instead of actual newlines
        # Need 3+ occurrences since threshold is > 2
        content = "def test():\\n    pass\\n\\n"
        result = write_file(str(root), "bad.py", content)
        assert result["written"] is True  # Write still succeeds
        assert "escape_warning" in result
        assert "literal '\\n'" in result["escape_warning"]
        assert "Did you forget to use actual newlines" in result["escape_warning"]

    def test_escape_warning_not_triggered_for_few_occurrences(self, root: Path) -> None:
        """Content with <=2 literal \\n sequences should not trigger warning."""
        content = "x = 1\\ny = 2"  # Only one literal \\n
        result = write_file(str(root), "ok.py", content)
        assert result["written"] is True
        assert "escape_warning" not in result

    def test_escape_warning_not_triggered_without_code_chars(self, root: Path) -> None:
        """Content with >2 \\n but no code-like chars (colon, braces, etc.) should NOT warn."""
        content = "first\\nsecond\\nthird\\nfourth"  # No code chars

        result = write_file(str(root), "plain.txt", content)
        assert result["written"] is True
        assert "escape_warning" not in result

    def test_escape_warning_detects_high_density_literal_backslash_n(
        self, root: Path
    ) -> None:
        """Multiple code-like lines with literal \\n should trigger warning."""
        content = "class Foo:\\n    def bar(self):\\n        pass\\n"

        result = write_file(str(root), "code.py", content)
        assert result["written"] is True
        assert "escape_warning" in result
        # Should mention literal backslash-n sequences
        assert "literal" in result["escape_warning"].lower()

    def test_actual_newlines_do_not_trigger_warning(self, root: Path) -> None:
        """Content with actual newlines (not literal \\n) should not trigger warning."""
        content = "def test():\n    pass\n"

        result = write_file(str(root), "good.py", content)
        assert result["written"] is True
        assert "escape_warning" not in result
        # Verify actual newlines are in the file
        assert "\n" in (root / "good.py").read_text()


class TestListDirectory:
    """Verify list_directory behavior."""

    @pytest.fixture
    def root(self, tmp_path: Path) -> Path:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("x = 1")
        (tmp_path / "src" / "utils.py").write_text("y = 2")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_main.py").write_text("test")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "main.cpython-312.pyc").write_bytes(b"pyc")
        (tmp_path / ".hidden_dir").mkdir()
        (tmp_path / ".hidden_file").write_text("hidden")
        (tmp_path / "README.md").write_text("# Project")
        return tmp_path

    def test_lists_root_files(self, root: Path) -> None:
        result = list_directory(str(root), "", depth=1)
        assert result["path"] == "."
        names = [f["name"] for f in result["files"]]
        # Should list directories and files, but not hidden ones
        assert "src" in names
        assert "tests" in names
        assert "README.md" in names
        assert ".hidden_dir" not in names
        assert ".hidden_file" not in names
        assert "__pycache__" not in names

    def test_skips_cache_artifacts_by_default(self, root: Path) -> None:
        result = list_directory(str(root), "", depth=2)
        names = [f["name"] for f in result["files"]]
        assert "__pycache__" not in names
        assert not any(name.endswith(".pyc") for name in names)

    def test_show_hidden(self, root: Path) -> None:
        result = list_directory(str(root), "", depth=1, show_hidden=True)
        names = [f["name"] for f in result["files"]]
        assert ".hidden_dir" in names
        assert ".hidden_file" in names

    def test_recursive_depth(self, root: Path) -> None:
        result = list_directory(str(root), "src", depth=1)
        names = [f["name"] for f in result["files"]]
        assert "main.py" in names

    def test_entries_include_relative_paths(self, root: Path) -> None:
        result = list_directory(str(root), "", depth=2)
        entries = {item["name"]: item for item in result["files"]}
        assert entries["src"]["path"] == "src"
        assert entries["main.py"]["path"] == "src/main.py"
        assert entries["utils.py"]["path"] == "src/utils.py"

    def test_depth_limit(self, root: Path) -> None:
        (root / "deep").mkdir()
        (root / "deep" / "deeper").mkdir()
        (root / "deep" / "deeper" / "file.txt").write_text("deep")
        result = list_directory(str(root), "", depth=1)
        names = [f["name"] for f in result["files"]]
        assert "deep" in names  # Directory at depth 1
        # deeper should not be listed (depth 2+)
        assert not any(f["name"] == "deeper" for f in result["files"])

    def test_nonexistent_directory(self, root: Path) -> None:
        result = list_directory(str(root), "nonexistent")
        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_subdirectory_path(self, root: Path) -> None:
        result = list_directory(str(root), "src", depth=1)
        assert result["path"] == "src"

    def test_counts_files_and_dirs(self, root: Path) -> None:
        result = list_directory(str(root), "", depth=1, show_hidden=True)
        assert result["total_dirs"] >= 3  # src, tests, .hidden_dir
        assert result["total_files"] >= 2  # README.md, .hidden_file (files in subdirs not counted at depth=1)
