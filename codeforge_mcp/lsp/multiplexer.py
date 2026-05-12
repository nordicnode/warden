"""LSP Multiplexer — on-demand LSP server per language.

Starts one LSP server per language lazily, communicates via JSON-RPC,
caches results in-memory for 10 minutes. Uses a single-reader architecture:
one background task reads all messages from stdout and dispatches them:
  - Responses (have 'id') → enqueued per-request
  - Notifications (have 'method') → routed to handlers (e.g. publishDiagnostics)

This eliminates the race condition between the notification listener and
request-response handler.
"""

from __future__ import annotations

import asyncio
import os
import json
import re
import time
from pathlib import Path
from typing import Any


# Regex matching identifier tokens on a source line.  We use finditer()
# to skip language keywords and land on the first *real* symbol name.
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")

# Language keywords that LSP servers cannot resolve with goto-definition,
# hover, or find-references.  When _first_identifier_col lands on one of
# these, it advances to the next identifier on the line.
_SKIP_KEYWORDS: frozenset[str] = frozenset({
    # Python
    "def", "class", "from", "import", "as", "if", "elif", "else",
    "for", "while", "with", "try", "except", "finally", "raise",
    "return", "yield", "pass", "break", "continue", "global",
    "nonlocal", "assert", "del", "lambda", "async", "await",
    "and", "or", "not", "in", "is", "True", "False", "None",
    # JavaScript / TypeScript
    "function", "const", "let", "var", "export", "default",
    "new", "typeof", "instanceof", "void", "delete", "throw",
    "switch", "case", "this", "super", "extends", "implements",
    "interface", "type", "enum", "abstract", "static", "readonly",
    "declare", "namespace", "module",
    # Rust
    "fn", "pub", "mod", "use", "struct", "impl", "trait", "where",
    "crate", "self", "Self", "mut", "ref", "move", "match", "loop",
    # Go
    "func", "package", "range", "defer", "go", "select", "chan",
    # C / C++
    "auto", "register", "extern", "inline", "virtual", "override",
    "template", "typename", "typedef", "sizeof", "alignof",
    "using", "public", "private", "protected",
})


def _first_identifier_col(file_path: str, line: int) -> int | None:
    """Return the 0-based column of the first *meaningful* identifier on ``line``.

    Skips language keywords (``def``, ``from``, ``class``, etc.) so the
    LSP cursor lands on the actual symbol name rather than a keyword
    that the server cannot resolve.

    Returns ``None`` if the file cannot be read or the line has no
    resolvable identifier.
    """
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            for i, text in enumerate(f, start=1):
                if i == line:
                    for m in _IDENT_RE.finditer(text):
                        if m.group() not in _SKIP_KEYWORDS:
                            return m.start()
                    return None
                if i > line:
                    break
    except OSError:
        return None
    return None


# Language → LSP server command
LSP_COMMANDS: dict[str, list[str]] = {
    "python": ["pyright-langserver", "--stdio"],
    "typescript": ["typescript-language-server", "--stdio"],
    "javascript": ["typescript-language-server", "--stdio"],
    "tsx": ["typescript-language-server", "--stdio"],
    "rust": ["rust-analyzer"],
    "go": ["gopls"],
    "c": ["clangd"],
    "cpp": ["clangd"],
}

EXT_TO_LSP: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
}


class LSPResult:
    """Cached LSP query result."""

    def __init__(self, data: Any, timestamp: float | None = None) -> None:
        self.data = data
        self.timestamp = timestamp or time.time()

    @property
    def expired(self) -> bool:
        return time.time() - self.timestamp > 600


class _ServerState:
    """Per-LSP-server state: process, reader task, response queue, diagnostics."""

    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self.proc = proc
        self.reader_task: asyncio.Task[None] | None = None
        # Map request id → asyncio.Future that resolves to the response result
        self.pending: dict[int, asyncio.Future[dict[str, Any] | None]] = {}
        # Per-URI diagnostics events (set when publishDiagnostics arrives)
        self.diag_events: dict[str, asyncio.Event] = {}
        # Per-URI diagnostic results
        self.diagnostics: dict[str, list[dict[str, Any]]] = {}
        # URIs that have been opened via textDocument/didOpen (avoids
        # duplicate opens which some LSP servers reject).
        self._opened_uris: set[str] = set()
        # Wall-clock timestamp at which the server finished `initialize`.
        # Used by query helpers to wait briefly while the server completes
        # its first workspace crawl (Phase 3 fix — addresses cold-start
        # empty results from pyright et al.).
        self.initialized_at: float = 0.0


class LSPMultiplexer:
    """Manages LSP servers per language, lazy-start, with 10-min caching."""

    # Minimum warmup time (seconds) to wait after initialization before
    # accepting empty results as genuine.  LSP servers like pyright need
    # several seconds to finish their first workspace scan; querying too
    # early produces blank responses that look like tool failures.
    # Raised from 3.0 → 5.0 (Phase 3 fix): pyright on medium-sized
    # Python projects (~100 files) can take 4-5s to complete its initial
    # indexing pass.
    _WARMUP_SECS: float = 5.0

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root)
        self._states: dict[str, _ServerState] = {}
        self._cache: dict[str, LSPResult] = {}
        self._next_id = 1

    def _cache_key(self, namespace: str, params: Any) -> str:
        return f"{namespace}:{json.dumps(params, sort_keys=True)}"

    def _build_process_env(self) -> dict[str, str]:
        """Build a PATH that includes common local language-server install dirs."""
        env = os.environ.copy()
        path_parts = [
            str(self.project_root / ".venv" / "bin"),
            str(self.project_root / "node_modules" / ".bin"),
            str(Path.home() / "go" / "bin"),
            str(Path.home() / ".local" / "bin"),
        ]
        existing = env.get("PATH", "")
        if existing:
            path_parts.append(existing)
        deduped: list[str] = []
        seen: set[str] = set()
        for part in path_parts:
            if not part or part in seen:
                continue
            seen.add(part)
            deduped.append(part)
        env["PATH"] = ":".join(deduped)
        return env

    def _get_lsp_lang(self, file_path: str) -> str | None:
        path = Path(file_path)
        for ext, lang in sorted(EXT_TO_LSP.items(), key=lambda x: -len(x[0])):
            if path.suffixes and path.suffixes[-1] == ext:
                return lang
            if path.name.endswith(ext):
                return lang
        return None

    def _next_id_val(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _build_initialize_params(self) -> dict[str, Any]:
        """Build LSP initialize params with explicit workspace metadata.

        Pyright's workspace-wide features depend on more than ``rootUri`` in
        practice. Supplying ``rootPath`` and ``workspaceFolders`` allows
        workspace/symbol and cross-file reference indexing to function
        consistently across clients.
        """
        root = self.project_root.resolve()
        root_uri = root.as_uri()
        return {
            "processId": None,
            "rootUri": root_uri,
            "rootPath": str(root),
            "workspaceFolders": [{"uri": root_uri, "name": root.name}],
            "capabilities": {},
        }

    async def _ensure_server(self, lang: str) -> _ServerState | None:
        if lang in self._states:
            state = self._states[lang]
            if state.proc.returncode is not None:
                # Clean up dead server state: cancel reader, cancel pending futures
                if state.reader_task is not None:
                    state.reader_task.cancel()
                for future in state.pending.values():
                    if not future.done():
                        future.cancel()
                state.pending.clear()
                del self._states[lang]
            else:
                return state

        cmd = LSP_COMMANDS.get(lang)
        if cmd is None:
            raise RuntimeError(f"No LSP command configured for language: {lang}")

        env = self._build_process_env()

        try:
            from codeforge_mcp import logging as log
            log.info("lsp", status="starting", lang=lang, cmd=cmd)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_root),
                env=env,
            )
            state = _ServerState(proc)

            # Send initialize (uses a temporary pending slot since reader isn't started yet)
            init_id = self._next_id_val()
            init_msg = json.dumps({
                "jsonrpc": "2.0",
                "id": init_id,
                "method": "initialize",
                "params": self._build_initialize_params(),
            })

            # Start the single reader task BEFORE sending init
            state.reader_task = asyncio.create_task(self._reader_loop(state))
            # Allow the reader to start
            await asyncio.sleep(0)

            # Create a future for the init response
            loop = asyncio.get_running_loop()
            init_future: asyncio.Future[dict[str, Any] | None] = loop.create_future()
            state.pending[init_id] = init_future

            await self._send_raw(proc, init_msg)
            await asyncio.wait_for(init_future, timeout=10)

            # Send initialized notification
            await self._send_raw(
                proc, json.dumps({"jsonrpc": "2.0", "method": "initialized", "params": {}})
            )

            state.initialized_at = time.time()
            self._states[lang] = state
            log.info("lsp", status="started", lang=lang)
        except Exception as e:
            from codeforge_mcp import logging as log
            log.error("LSP server failed to start", lang=lang, cmd=cmd, error=str(e))
            raise RuntimeError(f"LSP server for {lang} failed to start. Is '{cmd[0]}' installed? Error: {e}") from e

        return state

    async def _await_server_ready(self, state: _ServerState) -> None:
        """Wait for the LSP server to finish its initial workspace scan.

        Servers like pyright take 1–3 seconds after `initialize` to crawl the
        workspace.  Querying before that finishes returns empty results that
        surface as tool failures.  This helper sleeps the remaining warmup
        time after ``state.initialized_at`` so subsequent queries don't see
        false-empty responses.
        """
        if state.initialized_at == 0.0:
            return  # not yet initialized — shouldn't happen, guard anyway
        elapsed = time.time() - state.initialized_at
        remaining = self._WARMUP_SECS - elapsed
        if remaining > 0:
            await asyncio.sleep(remaining)

    async def _reader_loop(self, state: _ServerState) -> None:
        """Single reader task: read all messages from stdout and dispatch.

        - Messages with 'id' but no 'method' → response → resolve pending future
        - Messages with 'method' → notification → route to handler
        - Messages with both 'id' and 'method' → server→client request (ignored)
        """
        proc = state.proc
        if proc.stdout is None:
            return
        try:
            while True:
                msg = await self._read_message_raw(proc.stdout)
                if msg is None:
                    break

                msg_id = msg.get("id")
                method = msg.get("method", "")

                if method:
                    # Notification or server→client request: route by method
                    self._handle_notification(state, method, msg.get("params", {}))
                elif msg_id is not None:
                    # Response: resolve the pending future
                    future = state.pending.pop(msg_id, None)
                    if future is not None and not future.done():
                        future.set_result(msg)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            from codeforge_mcp import logging as log
            log.error("LSP reader loop crashed", error=str(exc))
        finally:
            # Cancel all pending futures so callers don't hang
            for future in list(state.pending.values()):
                if not future.done():
                    future.cancel()
            state.pending.clear()

    async def _read_message_raw(self, stdout: asyncio.StreamReader) -> dict[str, Any] | None:
        """Read a single LSP message: header + content.

        Uses readuntil for efficient header parsing (avoids 1-byte reads).
        """
        try:
            header = await stdout.readuntil(b"\r\n\r\n")
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            return None

        header_str = header.decode("utf-8")
        length = 0
        for line in header_str.split("\r\n"):
            if line.lower().startswith("content-length:"):
                try:
                    length = int(line.split(":")[1].strip())
                except ValueError:
                    pass
                break
        if length == 0:
            return None
        if length > 10_000_000:  # Safety cap: 10MB
            return None
        try:
            content = await stdout.readexactly(length)
        except asyncio.IncompleteReadError:
            return None
        return json.loads(content.decode("utf-8"))

    def _handle_notification(self, state: _ServerState, method: str, params: dict[str, Any]) -> None:
        """Route a server→client notification to the appropriate handler."""
        if method == "textDocument/publishDiagnostics":
            uri = params.get("uri", "")
            diags_raw = params.get("diagnostics", [])
            parsed: list[dict[str, Any]] = []
            for item in diags_raw:
                rng = item.get("range", {})
                start = rng.get("start", {})
                parsed.append({
                    "line": start.get("line", 0) + 1,
                    "severity": item.get("severity", 2),
                    "message": item.get("message", ""),
                    "source": item.get("source", ""),
                })
            state.diagnostics[uri] = parsed
            # Signal any waiter
            event = state.diag_events.get(uri)
            if event is not None:
                event.set()

    async def _send_raw(self, proc: asyncio.subprocess.Process, message: str) -> None:
        if proc.stdin is None:
            return
        content = message.encode("utf-8")
        header = f"Content-Length: {len(content)}\r\n\r\n".encode("utf-8")
        proc.stdin.write(header + content)
        await proc.stdin.drain()

    async def _lsp_request(self, state: _ServerState, method: str, params: Any) -> Any:
        """Send a request and wait for the matching response via the reader loop."""
        req_id = self._next_id_val()
        msg = json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        })

        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any] | None] = loop.create_future()
        state.pending[req_id] = future

        await self._send_raw(state.proc, msg)

        try:
            response = await asyncio.wait_for(future, timeout=15)
        except asyncio.TimeoutError:
            state.pending.pop(req_id, None)
            return None

        if response is None:
            return None
        return response.get("result")

    async def _lsp_notify(self, state: _ServerState, method: str, params: Any) -> None:
        """Send a notification (no id, no response expected)."""
        msg = json.dumps({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        })
        await self._send_raw(state.proc, msg)

    # ── Diagnostics (push-based via publishDiagnostics) ───────────────

    async def diagnostics(self, file_path: str) -> list[dict[str, Any]]:
        """Get diagnostics for a file by opening it and waiting for publishDiagnostics.

        Uses an asyncio.Event per URI to wait for the notification.
        Increased timeout (30s) because some servers (pyright) can be slow
        on first analysis of a large project.
        """
        lang = self._get_lsp_lang(file_path)
        if lang is None:
            return []

        state = await self._ensure_server(lang)
        if state is None:
            return []

        await self._await_server_ready(state)

        uri = Path(file_path).resolve().as_uri()
        try:
            content = Path(file_path).read_text()
        except (OSError, PermissionError):
            return []

        # Check if we already have diagnostics for this URI (from a prior
        # publishDiagnostics that arrived while the file was open for another
        # query like hover / goto-definition). Pop so they aren't returned
        # indefinitely.
        existing = state.diagnostics.pop(uri, None)
        if existing is not None:
            return existing

        # Create the event BEFORE sending didOpen to avoid missing fast notifications
        event = asyncio.Event()
        state.diag_events[uri] = event

        # Open the file (notification)
        await self._lsp_notify(state, "textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": lang,
                "version": 1,
                "text": content,
            },
        })

        # Wait for the publishDiagnostics notification.
        # Servers like pyright can take several seconds on the first
        # diagnostic pass for a large project.
        try:
            await asyncio.wait_for(event.wait(), timeout=15)
        except asyncio.TimeoutError:
            pass

        # Close the file (notification)
        await self._lsp_notify(state, "textDocument/didClose", {
            "textDocument": {"uri": uri},
        })

        # Cleanup
        state.diag_events.pop(uri, None)
        return state.diagnostics.pop(uri, [])

    # ── Document lifecycle helpers ──────────────────────────────────

    async def _ensure_document_open(
        self, state: _ServerState, file_path: str, lang: str
    ) -> str | None:
        """Send textDocument/didOpen for a file if needed.  Returns the URI.

        Keeps a set of opened URIs per server to avoid duplicate opens.
        Returns None if the file cannot be read.
        """
        uri = Path(file_path).resolve().as_uri()
        if uri in state._opened_uris:
            return uri
        try:
            content = Path(file_path).read_text()
        except (OSError, PermissionError) as e:
            raise RuntimeError(f"Cannot read file {file_path}: {e}") from e
        await self._lsp_notify(state, "textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": lang,
                "version": 1,
                "text": content,
            },
        })
        state._opened_uris.add(uri)
        return uri

    # ── References ────────────────────────────────────────────────────

    async def references(self, name: str, file_path: str, line: int = 1, col: int = 0) -> list[dict[str, Any]]:
        """Find all references to a symbol via LSP.

        Includes a retry loop for cold-start servers that return empty
        results while still indexing the workspace (Phase 3 fix).
        """
        lang = self._get_lsp_lang(file_path)
        if lang is None:
            return []

        state = await self._ensure_server(lang)
        if state is None:
            return []

        await self._await_server_ready(state)

        uri = await self._ensure_document_open(state, file_path, lang)
        if uri is None:
            return []

        # If col=0 was passed (typical when callers don't know the symbol's
        # exact column), bias toward the first identifier on that line so
        # imports / class/function defs / call sites resolve cleanly.
        if col == 0:
            col = _first_identifier_col(file_path, line) or 0

        # Retry loop: LSP servers like pyright may return empty results
        # while still scanning the workspace.  Retry up to 3 times with
        # a 1s backoff before accepting empty as genuine.
        # Use truthiness check (not `is not None`) so empty lists also
        # trigger a retry — pyright returns [] when cold, not null.
        result = None
        for attempt in range(3):
            result = await self._lsp_request(state, "textDocument/references", {
                "textDocument": {"uri": uri},
                "position": {"line": line - 1, "character": col},
                "context": {"includeDeclaration": True},
            })
            if result:
                break
            if attempt < 2:
                await asyncio.sleep(1.0)

        if not result:
            return []

        refs: list[dict[str, Any]] = []
        for ref in result:
            uri_ref = ref.get("uri", "")
            rng = ref.get("range", {})
            start = rng.get("start", {})
            refs.append({
                "file": uri_ref.replace("file://", ""),
                "line": start.get("line", 0) + 1,
                "character": start.get("character", 0),
            })
        return refs

    # ── Definition ────────────────────────────────────────────────────

    async def goto_definition(self, file_path: str, line: int, col: int = 0) -> dict[str, Any] | None:
        """Go to the definition of a symbol at the given position."""
        lang = self._get_lsp_lang(file_path)
        if lang is None:
            return None

        state = await self._ensure_server(lang)
        if state is None:
            return None

        await self._await_server_ready(state)

        uri = await self._ensure_document_open(state, file_path, lang)
        if uri is None:
            return None

        # When col=0 the cursor sits on column 0 (often whitespace or the
        # start of a keyword like "from"/"import"/"def") which the LSP
        # rejects as "no symbol here". Bias to the first identifier on the
        # line so callers can simply pass a line number.
        if col == 0:
            col = _first_identifier_col(file_path, line) or 0

        result = await self._lsp_request(state, "textDocument/definition", {
            "textDocument": {"uri": uri},
            "position": {"line": line - 1, "character": col},
        })

        if result is None:
            return None

        if isinstance(result, list):
            result = result[0] if result else None
        if result is None:
            return None

        rng = result.get("range", {})
        start = rng.get("start", {})
        return {
            "file": result.get("uri", "").replace("file://", ""),
            "line": start.get("line", 0) + 1,
            "character": start.get("character", 0),
        }

    # ── Hover ─────────────────────────────────────────────────────────

    async def hover(self, file_path: str, line: int, col: int = 0) -> dict[str, Any] | None:
        """Get hover information for a position.

        Includes a retry loop for cold-start servers (Phase 3 fix).
        """
        lang = self._get_lsp_lang(file_path)
        if lang is None:
            return None

        state = await self._ensure_server(lang)
        if state is None:
            return None

        await self._await_server_ready(state)

        uri = await self._ensure_document_open(state, file_path, lang)
        if uri is None:
            return None

        # See goto_definition: bias col=0 toward the first identifier on
        # the line.
        if col == 0:
            col = _first_identifier_col(file_path, line) or 0

        # Retry loop: LSP servers may return None while still warming up
        # Use truthiness check so empty dicts also trigger retry.
        result = None
        for attempt in range(3):
            result = await self._lsp_request(state, "textDocument/hover", {
                "textDocument": {"uri": uri},
                "position": {"line": line - 1, "character": col},
            })
            if result:
                break
            if attempt < 2:
                await asyncio.sleep(1.0)

        if not result:
            # Phase 3 fix: differentiate "no hover at this position" from
            # "LSP server still warming up".
            from codeforge_mcp import logging as log
            warming = (time.time() - state.initialized_at) < self._WARMUP_SECS
            log.info(
                "lsp",
                status="hover_empty",
                file=file_path,
                line=line,
                col=col,
                lsp_likely_not_ready=warming,
            )
            return None

        contents = result.get("contents", {})
        value = ""
        if isinstance(contents, str):
            value = contents
        elif isinstance(contents, dict):
            value = contents.get("value", str(contents))
        elif isinstance(contents, list):
            parts = []
            for c in contents:
                if isinstance(c, str):
                    parts.append(c)
                elif isinstance(c, dict):
                    parts.append(c.get("value", ""))
            value = "\n".join(parts)

        return {
            "value": value,
            "range": result.get("range"),
        }

    # ── Workspace Symbols ─────────────────────────────────────────────

    async def workspace_symbols(self, query: str) -> list[dict[str, Any]]:
        """Search for symbols across the entire workspace."""
        cache_key = self._cache_key("workspace_symbols", {"query": query})
        if cache_key in self._cache and not self._cache[cache_key].expired:
            return self._cache[cache_key].data or []

        from codeforge_mcp import logging as log

        # Optimisation: discover active languages in the project
        from codeforge_mcp.indexer import discover_files
        from codeforge_mcp.ast.indexer import EXT_TO_LANG

        rel_paths = await asyncio.to_thread(discover_files, self.project_root)
        active_langs: set[str] = set()
        for p in rel_paths:
            ext = Path(p).suffix
            if ext in EXT_TO_LANG:
                active_langs.add(EXT_TO_LANG[ext])

        log.info("lsp", status="workspace_symbols", query=query, active_langs=sorted(active_langs))

        # Start unique servers for active languages in parallel
        seen_cmds: set[tuple[str, ...]] = set()
        starters: list[asyncio.Task[_ServerState | None]] = []
        for lang in active_langs:
            # EXT_TO_LANG returns capitalized names ("Python") but
            # LSP_COMMANDS uses lowercase keys ("python") — normalize here.
            lang_lower = lang.lower()
            cmd_key = tuple(LSP_COMMANDS.get(lang_lower, []))
            if cmd_key and cmd_key not in seen_cmds:
                seen_cmds.add(cmd_key)
                starters.append(asyncio.create_task(self._ensure_server(lang_lower)))
        if starters:
            await asyncio.gather(*starters)

        # Wait-for-ready: when a server has only just been initialised it
        # may not have finished its first workspace scan yet, so an
        # immediate workspace/symbol request returns 0 results. We retry up
        # to ~7s with a short backoff before giving up (Phase 3 fix).
        all_symbols: list[dict[str, Any]] = []
        # Retry loop: _await_server_ready handles the initial warmup, but
        # transient LSP issues (slow workspace scans, server hiccups) can
        # still produce empty results.  We retry up to 4 times with a
        # short backoff as a safety net.
        max_attempts = 4
        retry_delay = 0.5  # seconds, doubles each attempt
        for attempt in range(max_attempts):
            all_symbols = []

            for lang, state in list(self._states.items()):
                if state.proc.returncode is not None:
                    continue
                # Wait for the server to finish its initial workspace scan
                # before querying — avoids false-empty results from pyright et al.
                await self._await_server_ready(state)
                result = await self._lsp_request(state, "workspace/symbol", {"query": query})
                if isinstance(result, list):
                    for sym in result:
                        loc = sym.get("location", {})
                        uri = loc.get("uri", "")
                        rng = loc.get("range", {})
                        start = rng.get("start", {})
                        all_symbols.append({
                            "name": sym.get("name", ""),
                            "kind": sym.get("kind", 0),
                            "file": uri.replace("file://", ""),
                            "line": start.get("line", 0) + 1,
                        })
                if len(all_symbols) >= 50:
                    break

            if all_symbols or attempt == max_attempts - 1:
                break

            log.info(
                "lsp",
                status="workspace_symbols_retry",
                query=query,
                attempt=attempt + 1,
                delay_s=retry_delay,
            )
            await asyncio.sleep(retry_delay)
            retry_delay *= 2

        if not all_symbols:
            # Distinguish "no symbol matches" from "LSP server not ready".
            still_warming = [
                lang for lang, st in self._states.items()
                if st.proc.returncode is None and (time.time() - st.initialized_at) < 2.0
            ]
            log.warn(
                "workspace_symbols returned 0 results",
                query=query,
                active_langs=sorted(active_langs),
                lsp_likely_not_ready=bool(still_warming),
                warming_langs=sorted(still_warming),
            )

        self._cache[cache_key] = LSPResult(all_symbols)
        return all_symbols

    # ── Symbol Lookup ─────────────────────────────────────────────────

    async def symbol_lookup(self, name: str, file_path: str | None = None) -> dict[str, Any] | None:
        """Look up a symbol via LSP workspace/symbol.

        When ``file_path`` is None, queries ALL active LSP servers in the
        project rather than defaulting to Python — this fixes the common
        case where a TypeScript symbol couldn't be found because only
        pyright was consulted.
        """
        lang = None
        if file_path:
            lang = self._get_lsp_lang(file_path)

        cache_key = self._cache_key("symbol_lookup", {"query": name})
        if cache_key in self._cache and not self._cache[cache_key].expired:
            return self._cache[cache_key].data

        # ── Multi-language lookup (Phase 3 fix) ──────────────────────
        # When no file_path hint is provided, discover active languages
        # from the project and query each LSP server.  Previously we
        # defaulted to "python", which made TypeScript / Rust / Go symbols
        # permanently unresolvable via LSP.
        # Cache the active-language set at class level so repeated
        # symbol_lookup calls for different names don't each walk the
        # filesystem.
        if lang is None:
            cached_langs: set[str] | None = getattr(self, "_active_langs", None)
            if cached_langs is None:
                from codeforge_mcp.indexer import discover_files
                from codeforge_mcp.ast.indexer import EXT_TO_LANG
                rel_paths = await asyncio.to_thread(discover_files, self.project_root)
                cached_langs = set()
                for p in rel_paths:
                    ext = Path(p).suffix
                    if ext in EXT_TO_LANG:
                        lang_name = EXT_TO_LANG[ext].lower()
                        # Only include languages that have an LSP server configured
                        if lang_name in LSP_COMMANDS:
                            cached_langs.add(lang_name)
                self._active_langs = cached_langs  # type: ignore[attr-defined]
            langs_to_try = sorted(cached_langs) if cached_langs else ["python"]
        else:
            langs_to_try = [lang]

        # LSP SymbolKind values we treat as "definitions":
        #   Class=5, Method=6, Constructor=9, Interface=11,
        #   Function=12, Struct=23.
        DEFINITION_KINDS = {5, 6, 9, 11, 12, 23}

        def _rank(sym: dict[str, Any]) -> tuple[int, int, int]:
            sym_name = sym.get("name", "")
            kind = sym.get("kind", 0)
            # Lower tuple sorts first.
            exact = 0 if sym_name == name else 1
            case_insensitive = 0 if sym_name.lower() == name.lower() else 1
            kind_rank = 0 if kind in DEFINITION_KINDS else 1
            return (exact, case_insensitive, kind_rank)

        data = None
        for lang_candidate in langs_to_try:
            state = await self._ensure_server(lang_candidate)
            if state is None:
                continue
            await self._await_server_ready(state)
            result = await self._lsp_request(state, "workspace/symbol", {"query": name})
            if result is None or not isinstance(result, list) or not result:
                continue

            best = min(result, key=_rank)
            loc = best.get("location", {})
            uri = loc.get("uri", "")
            range_info = loc.get("range", {})
            start = range_info.get("start", {})
            data = {
                "name": best.get("name", name),
                "kind": best.get("kind", 0),
                "file": uri.replace("file://", ""),
                "line": start.get("line", 0) + 1,
            }
            break  # first language that returns results wins

        self._cache[cache_key] = LSPResult(data)
        return data

    def clear_cache(self, file_uri: str | None = None) -> None:
        """Clear the LSP result cache.

        Args:
            file_uri: If provided, only clear entries containing this URI.
                      If None (default), clear the entire cache.
        """
        if file_uri is None:
            self._cache.clear()
            # Also reset the cached active-languages set so the next
            # symbol_lookup call rediscovers the project layout.
            if hasattr(self, "_active_langs"):
                delattr(self, "_active_langs")
        else:
            # Remove all cache entries that reference this URI
            to_remove = [key for key, val in self._cache.items() if file_uri in key]
            for key in to_remove:
                del self._cache[key]

    async def shutdown(self) -> None:
        for state in self._states.values():
            try:
                # Try graceful shutdown
                await asyncio.wait_for(self._lsp_request(state, "shutdown", {}), timeout=2)
                await self._send_raw(state.proc, json.dumps({"jsonrpc": "2.0", "method": "exit", "params": {}}))
                state.proc.terminate()
                await asyncio.sleep(0.1)
            except Exception:
                state.proc.kill()
            finally:
                if state.reader_task is not None:
                    state.reader_task.cancel()
        self._states.clear()
        self._cache.clear()
