"""Tests for the file watcher — lifecycle and file extension support detection.

Verifies:
- _is_supported correctly identifies supported file extensions
- _reindex_file handles deleted files
- start/stop lifecycle manages the background task
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codeforge_mcp.watcher import FileWatcher


class TestFileExtensionSupport:
    """Verify _is_supported checks file extensions."""

    @pytest.fixture
    def watcher(self, tmp_path: Path) -> FileWatcher:
        return FileWatcher(
            project_root=str(tmp_path),
            graph=MagicMock(spec=[]),
            ast_indexer=MagicMock(spec=[]),
        )

    def test_python_is_supported(self, watcher: FileWatcher) -> None:
        assert watcher._is_supported(Path("test.py"))

    def test_typescript_is_supported(self, watcher: FileWatcher) -> None:
        assert watcher._is_supported(Path("component.ts"))
        assert watcher._is_supported(Path("component.tsx"))

    def test_javascript_is_supported(self, watcher: FileWatcher) -> None:
        assert watcher._is_supported(Path("app.js"))
        assert watcher._is_supported(Path("app.jsx"))

    def test_rust_is_supported(self, watcher: FileWatcher) -> None:
        assert watcher._is_supported(Path("main.rs"))

    def test_go_is_supported(self, watcher: FileWatcher) -> None:
        assert watcher._is_supported(Path("main.go"))

    def test_c_cpp_is_supported(self, watcher: FileWatcher) -> None:
        assert watcher._is_supported(Path("main.c"))
        assert watcher._is_supported(Path("header.h"))
        assert watcher._is_supported(Path("main.cpp"))
        assert watcher._is_supported(Path("main.cc"))

    def test_bash_is_supported(self, watcher: FileWatcher) -> None:
        assert watcher._is_supported(Path("script.sh"))

    def test_lua_is_supported(self, watcher: FileWatcher) -> None:
        assert watcher._is_supported(Path("init.lua"))

    def test_unsupported_extension(self, watcher: FileWatcher) -> None:
        assert not watcher._is_supported(Path("readme.md"))
        assert not watcher._is_supported(Path("config.toml"))
        assert not watcher._is_supported(Path("image.png"))
        assert not watcher._is_supported(Path("dockerfile"))


class TestReindexFile:
    """Verify _reindex_file handles deleted and modified files."""

    @pytest.fixture
    def watcher(self, tmp_path: Path) -> FileWatcher:
        mock_graph = MagicMock()
        mock_graph.delete_symbols_in_file.return_value = 3
        mock_ast = MagicMock()
        mock_ast.index_file_incremental.return_value = 2
        return FileWatcher(
            project_root=str(tmp_path),
            graph=mock_graph,
            ast_indexer=mock_ast,
        )

    def test_deleted_file_removes_symbols(self, watcher: FileWatcher, tmp_path: Path) -> None:
        watcher.project_root = tmp_path
        path_str = str(tmp_path / "deleted.py")

        # File doesn't exist → should call delete_symbols_in_file
        watcher._reindex_file(path_str)
        watcher.graph.delete_symbols_in_file.assert_called_once()

    def test_existing_file_reindexes(self, watcher: FileWatcher, tmp_path: Path) -> None:
        f = tmp_path / "existing.py"
        f.write_text("def foo(): pass")

        watcher._reindex_file(str(f))
        watcher.ast_indexer.index_file_incremental.assert_called_once_with(str(f))

    def test_existing_file_invalidates_caches(self, watcher: FileWatcher, tmp_path: Path) -> None:
        call_count = [0]

        def callback() -> None:
            call_count[0] += 1

        watcher._clear_caches = callback
        f = tmp_path / "modified.py"
        f.write_text("x = 1")

        watcher._reindex_file(str(f))
        assert call_count[0] == 1


class TestWatcherLifecycle:
    """Verify start/stop lifecycle."""

    @pytest.fixture
    def watcher(self, tmp_path: Path) -> FileWatcher:
        return FileWatcher(
            project_root=str(tmp_path),
            graph=MagicMock(spec=[]),
            ast_indexer=MagicMock(spec=[]),
        )

    @pytest.fixture(autouse=True)
    def _require_watchfiles(self) -> None:
        pytest.importorskip("watchfiles", reason="watchfiles not installed")

    @pytest.mark.asyncio
    async def test_start_sets_running(self, watcher: FileWatcher) -> None:
        # awatch is imported locally inside _watch_loop: from watchfiles import awatch
        with patch('watchfiles.awatch', side_effect=asyncio.CancelledError):
            await watcher.start(debounce_ms=10)
            assert watcher.running is True
            await watcher.stop()

    @pytest.mark.asyncio
    async def test_stop_sets_not_running(self, watcher: FileWatcher) -> None:
        with patch('watchfiles.awatch', side_effect=asyncio.CancelledError):
            await watcher.start(debounce_ms=10)

        await watcher.stop()
        assert watcher.running is False

    @pytest.mark.asyncio
    async def test_double_start_is_idempotent(self, watcher: FileWatcher) -> None:
        with patch('watchfiles.awatch', side_effect=asyncio.CancelledError):
            await watcher.start(debounce_ms=10)
            await watcher.start(debounce_ms=10)  # Should be no-op
            assert watcher.running is True
            await watcher.stop()

    @pytest.mark.asyncio
    async def test_stop_cleans_up_task(self, watcher: FileWatcher) -> None:
        with patch('watchfiles.awatch', side_effect=asyncio.CancelledError):
            await watcher.start(debounce_ms=10)

        await watcher.stop()
        assert watcher._task is None

    def test_stats_initial_values(self, watcher: FileWatcher) -> None:
        stats = watcher.stats
        assert stats["events_received"] == 0
        assert stats["files_reindexed"] == 0
        assert stats["last_event"] == 0

    def test_running_initial_value(self, watcher: FileWatcher) -> None:
        assert watcher.running is False
