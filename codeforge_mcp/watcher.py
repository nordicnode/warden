"""File watcher — invalidates AST and LSP caches on file changes.

Uses watchfiles (inotify-based) to detect file saves and trigger
incremental re-indexing. Runs as a background asyncio task.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from codeforge_mcp import logging as log


class FileWatcher:
    """Watches the project directory for file changes and triggers re-indexing.

    When a supported source file is created, modified, or deleted:
    - The AST indexer re-parses the file (incremental)
    - LSP diagnostics are invalidated for that file
    - Server-level resource caches are invalidated (cognition map, etc.)
    """

    def __init__(
        self,
        project_root: str | Path,
        graph: Any,
        ast_indexer: Any,
        lsp_multiplexer: Any | None = None,
        cache_clear_callback: Callable[[], None] | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.graph = graph
        self.ast_indexer = ast_indexer
        self.lsp = lsp_multiplexer
        self._clear_caches = cache_clear_callback
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._stats: dict[str, int] = {
            "events_received": 0,
            "files_reindexed": 0,
            "last_event": 0,
        }

    @property
    def running(self) -> bool:
        return self._running

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    async def start(self, debounce_ms: int = 150) -> None:
        """Start the file watcher background task.

        Args:
            debounce_ms: Debounce delay in milliseconds (default 300ms).
        """
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._watch_loop(debounce_ms))
        log.info("File watcher started", project=str(self.project_root))

    async def stop(self) -> None:
        """Stop the file watcher."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        log.info("File watcher stopped")

    async def _watch_loop(self, debounce_ms: int) -> None:
        """Main watch loop: detect changes and re-index modified files.

        Uses a periodic drain task so that a single isolated change is
        reindexed within 2× debounce_ms even when no follow-up event
        arrives (previously it waited indefinitely for the next event).
        """
        try:
            from watchfiles import awatch
        except ImportError:
            log.warn("watchfiles not installed, file watching disabled")
            self._running = False
            return

        watched_dir = str(self.project_root)
        debounce_s = debounce_ms / 1000.0

        # Coalesce events within debounce window
        pending: dict[str, float] = {}  # file_path → earliest event time

        # ── Phase 5 fix: immediate status update on first event ────────
        # Previously the watcher_status tool could report stale stats
        # because _stats was only updated inside _accept_event, which ran
        # once per inotify batch.  Now we bump the event counter eagerly
        # so watcher_status reflects recent activity immediately.

        def _accept_event(path_str: str) -> bool:
            # Compute relative path to avoid incorrectly rejecting events when
            # the project lives under a hidden ancestor directory
            try:
                rel_path = Path(path_str).resolve().relative_to(self.project_root.resolve())
                parts = rel_path.parts
            except ValueError:
                parts = Path(path_str).parts

            # Skip hidden files, .git, .codeforge, node_modules, etc.
            # Exclude the leading "." part (current dir from relative paths)
            # so "./codeforge_mcp/tools/foo.py" isn't filtered out.
            if any(
                part.startswith(".") and part not in (".", "..")
                for part in parts
            ):
                return False
            if "node_modules" in parts or "target" in parts:
                return False
            if "__pycache__" in parts:
                return False

            # Special case for JS/TS project configs
            path = Path(path_str)
            if path.name in ("tsconfig.json", "jsconfig.json"):
                return True

            if not self._is_supported(path):
                return False
            return True

        def _drain() -> None:
            """Process files that have been stable for debounce_ms."""
            now = time.time()
            to_process: list[str] = []
            for path_str, earliest in list(pending.items()):
                if now - earliest >= debounce_s:
                    to_process.append(path_str)
                    del pending[path_str]
            for path_str in to_process:
                try:
                    self._reindex_file(path_str)
                    self._stats["files_reindexed"] += 1
                except Exception as e:
                    log.error("Re-index failed", file=path_str, error=str(e))

        async def _periodic_drain() -> None:
            """Periodically drain pending items even without new events."""
            while self._running:
                await asyncio.sleep(debounce_s / 2)  # check twice per debounce window
                _drain()

        drain_task = asyncio.create_task(_periodic_drain())

        try:
            async for changes in awatch(watched_dir, ignore_permission_denied=True):
                if not self._running:
                    break

                now = time.time()
                # Eagerly update stats so watcher_status reflects recent activity.
                # Count accepted files (not just batches) for accurate event tracking.
                accepted = sum(1 for _, path_str in changes if _accept_event(path_str))
                if accepted:
                    self._stats["events_received"] += accepted
                    self._stats["last_event"] = int(now)
                for change_type, path_str in changes:
                    if not _accept_event(path_str):
                        continue
                    if path_str not in pending:
                        pending[path_str] = now

                _drain()
        finally:
            drain_task.cancel()
            try:
                await drain_task
            except asyncio.CancelledError:
                pass

    def _is_supported(self, path: Path) -> bool:
        """Check if the file extension is supported by tree-sitter."""
        from codeforge_mcp.ast.indexer import _supported_ext
        return _supported_ext(path) is not None

    def _reindex_file(self, path_str: str) -> None:
        """Re-index a single file and invalidate related caches."""
        path = Path(path_str)

        # Special case for JS/TS project configs: they are not source files
        # but changing them affects how other files are indexed/resolved.
        if path.name in ("tsconfig.json", "jsconfig.json"):
            self._invalidate_caches()
            return

        if not path.is_file():
            # File was deleted — use the resolved absolute path to match
            # what ast/indexer.py:_parse_and_index stores (it canonicalizes
            # to str(path.resolve()) so the graph lookup succeeds).
            abs_path = str(path.resolve())
            self.graph.delete_symbols_in_file(abs_path)
            log.info("File deleted, symbols removed", file=abs_path)
            self._invalidate_caches()
            return

        # Re-index the file incrementally.  Parse errors are non-fatal —
        # the file may be mid-edit; we still invalidate caches below.
        rel = str(path.relative_to(self.project_root))
        try:
            count = self.ast_indexer.index_file_incremental(str(path))
            if count > 0:
                log.info("File re-indexed", file=rel, symbols_updated=count)
        except Exception as e:
            log.warn("Parse error during re-index", file=rel, error=str(e))

        # Always invalidate caches on any structural change (adds, deletes,
        # imports, etc.) — not just when symbols change — since the cognition
        # map, dependency graph, and workspace structure could all be affected.
        self._invalidate_caches()

        # Invalidate LSP caches for this file (if LSP is available)
        if self.lsp is not None:
            uri = path.resolve().as_uri()
            # Clear the 10-minute LSP result cache (hover, references, definition, etc.)
            if hasattr(self.lsp, "clear_cache"):
                self.lsp.clear_cache(file_uri=uri)
            # Clear per-server diagnostic state
            for lang_state in getattr(self.lsp, "_states", {}).values():
                lang_state.diag_events.pop(uri, None)
                lang_state.diagnostics.pop(uri, None)

    def _invalidate_caches(self) -> None:
        """Invalidate server-level resource caches via callback.

        Called on every structural change (file added/deleted/modified).
        The callback is wired up by the server to clear _resource_cache
        entries (cognition_map, workspace/structure, dependencies).
        Uses a callback to avoid circular imports (watcher ↔ server).
        """
        if self._clear_caches is not None:
            self._clear_caches()
