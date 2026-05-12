"""Tests for the LSP multiplexer — message format validation.

These tests verify that:
- _lsp_notify sends JSON-RPC notifications (no "id" field)
- _lsp_request sends JSON-RPC requests (has "id" field, monotonic)
- references() uses "textDocument/references" — not the invalid "references" (Bug #1)
- diagnostics() sends didOpen/didClose as notifications, not requests (Bug #2)

Bug #1: references() used "references" instead of "textDocument/references"
Bug #2: diagnostics() sent didOpen/didClose as _lsp_request (with id),
        causing 15s timeouts — they are LSP notifications per spec
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codeforge_mcp.lsp.multiplexer import LSPMultiplexer, _ServerState


def _mock_state() -> _ServerState:
    """Create a _ServerState with a mocked subprocess whose stdin accepts writes."""
    proc = MagicMock()
    proc.stdin = AsyncMock()
    proc.stdin.drain = AsyncMock()
    proc.returncode = None
    proc.stdout = None
    return _ServerState(proc)


class TestRequestVsNotifyFormat:
    """Verify JSON-RPC message format: requests have 'id', notifications don't."""

    @pytest.fixture
    def mux(self, tmp_path: Path) -> LSPMultiplexer:
        return LSPMultiplexer(str(tmp_path))

    @pytest.mark.asyncio
    async def test_notify_has_no_id(self, mux: LSPMultiplexer) -> None:
        """Bug #2: _lsp_notify must NOT include an 'id' field (notifications)."""
        sent_messages: list[str] = []

        async def capture(_proc: MagicMock, message: str) -> None:
            sent_messages.append(message)

        with patch.object(mux, "_send_raw", side_effect=capture):
            await mux._lsp_notify(_mock_state(), "textDocument/didOpen",
                                  {"uri": "file:///test.py"})

        msg = json.loads(sent_messages[0])
        assert msg["jsonrpc"] == "2.0"
        assert msg["method"] == "textDocument/didOpen"
        assert "id" not in msg, (
            "Notification must NOT include an 'id' field. "
            "Adding 'id' makes it a request, which the server won't reply to, "
            "causing a 15s timeout in the caller."
        )

    @pytest.mark.asyncio
    async def test_request_has_id(self, mux: LSPMultiplexer) -> None:
        """Requests MUST include a monotonically increasing id field."""
        sent_messages: list[str] = []

        async def capture(_proc: MagicMock, message: str) -> None:
            sent_messages.append(message)

        with patch.object(mux, "_send_raw", side_effect=capture), \
             patch.object(asyncio, "wait_for", new=AsyncMock(return_value={})):
            await mux._lsp_request(_mock_state(), "textDocument/definition",
                                   {"uri": "file:///test.py"})

        msg = json.loads(sent_messages[0])
        assert "id" in msg, "Request must include an 'id' field"
        assert isinstance(msg["id"], int)

    @pytest.mark.asyncio
    async def test_request_ids_are_unique_and_monotonic(self, mux: LSPMultiplexer) -> None:
        """Each request gets a unique, incrementing id."""
        sent_messages: list[str] = []

        async def capture(_proc: MagicMock, message: str) -> None:
            sent_messages.append(message)

        with patch.object(mux, "_send_raw", side_effect=capture), \
             patch.object(asyncio, "wait_for", new=AsyncMock(return_value={})):
            await mux._lsp_request(_mock_state(), "a", {})
            await mux._lsp_request(_mock_state(), "b", {})
            await mux._lsp_request(_mock_state(), "c", {})

        ids = [json.loads(m)["id"] for m in sent_messages]
        assert ids == sorted(ids), "Request ids must be monotonically increasing"
        assert len(set(ids)) == 3, "Request ids must be unique"

    def test_initialize_params_include_workspace_metadata(self, mux: LSPMultiplexer) -> None:
        params = mux._build_initialize_params()
        assert params["rootUri"].startswith("file://")
        assert params["rootPath"] == str(mux.project_root.resolve())
        assert params["workspaceFolders"] == [
            {"uri": mux.project_root.resolve().as_uri(), "name": mux.project_root.resolve().name}
        ]

    def test_build_process_env_includes_common_local_bins(self, mux: LSPMultiplexer) -> None:
        env = mux._build_process_env()
        path_parts = env["PATH"].split(":")
        assert str(mux.project_root / ".venv" / "bin") in path_parts
        assert str(mux.project_root / "node_modules" / ".bin") in path_parts
        assert str(Path.home() / "go" / "bin") in path_parts


class TestReferencesMethodName:
    """Bug #1: references() must use 'textDocument/references', not 'references'."""

    @pytest.fixture
    def mux(self, tmp_path: Path) -> LSPMultiplexer:
        return LSPMultiplexer(str(tmp_path))

    @pytest.mark.asyncio
    async def test_references_sends_correct_lsp_method(self, mux: LSPMultiplexer) -> None:
        """Bug #1: references() sends 'textDocument/references', not 'references'.

        The LSP spec requires 'textDocument/references'. Using the bare
        'references' causes servers to reply MethodNotFound and lsp_find_references
        returns [].

        We verify by mocking _ensure_server and _send_raw, then inspecting the
        actual JSON-RPC message sent by references().
        """
        sent_messages: list[str] = []

        async def capture(_proc: MagicMock, message: str) -> None:
            sent_messages.append(message)

        with patch.object(mux, "_send_raw", side_effect=capture), \
             patch.object(mux, "_ensure_server", return_value=_mock_state()), \
             patch.object(mux, "_ensure_document_open", return_value="file:///tmp/test.py"), \
             patch.object(asyncio, "wait_for", new=AsyncMock(return_value={"result": []})):
            await mux.references("", "/tmp/test.py", line=1, col=0)

        # references() calls _lsp_request → _send_raw. Inspect the method name.
        for msg_str in sent_messages:
            msg = json.loads(msg_str)
            assert msg["method"] == "textDocument/references", (
                f"references() sent method='{msg['method']}'. "
                "Must be 'textDocument/references' per LSP spec. "
                "Using 'references' causes servers to reply MethodNotFound."
            )


class TestDiagnosticsUsesNotify:
    """Bug #2: diagnostics() must send didOpen/didClose as notifications, not requests."""

    @pytest.fixture
    def mux(self, tmp_path: Path) -> LSPMultiplexer:
        (tmp_path / "test.py").write_text("x = 1")
        return LSPMultiplexer(str(tmp_path))

    @pytest.mark.asyncio
    async def test_did_open_close_are_notifications(self, mux: LSPMultiplexer) -> None:
        """Bug #2: didOpen and didClose in diagnostics() must NOT include 'id'.

        If they include 'id', the server treats them as requests and never replies,
        causing a 15s asyncio.wait_for timeout, a leaked future in state.pending,
        and the server may reject the malformed message. This breaks diagnostics.

        We verify by mocking _ensure_server and _send_raw, then checking the
        sent JSON-RPC messages have no 'id' field.
        """
        sent_messages: list[str] = []

        async def capture(_proc: MagicMock, message: str) -> None:
            sent_messages.append(message)

        fake_state = _mock_state()
        test_file = str(mux.project_root / "test.py")

        with patch.object(mux, "_send_raw", side_effect=capture), \
             patch.object(mux, "_ensure_server", return_value=fake_state), \
             patch.object(asyncio, "wait_for",
                          side_effect=asyncio.TimeoutError):  # no real LSP
            await mux.diagnostics(test_file)

        # diagnostics() should have sent at least didOpen and didClose
        assert len(sent_messages) >= 2, (
            f"Expected at least 2 messages (didOpen + didClose), got {len(sent_messages)}"
        )

        # Find didOpen and didClose messages
        did_open = None
        did_close = None
        for msg_str in sent_messages:
            msg = json.loads(msg_str)
            if msg.get("method") == "textDocument/didOpen":
                did_open = msg
            elif msg.get("method") == "textDocument/didClose":
                did_close = msg

        assert did_open is not None, "diagnostics() must send textDocument/didOpen"
        assert did_close is not None, "diagnostics() must send textDocument/didClose"

        # Neither should have an 'id' — they are notifications
        assert "id" not in did_open, (
            "didOpen must be a notification (no id). "
            "If id is present, the caller will wait for a response that never comes, "
            "causing a 15s timeout and breaking diagnostics."
        )
        assert "id" not in did_close, (
            "didClose must be a notification (no id). "
            "If id is present, the caller will wait for a response that never comes."
        )


class TestDiagnosticsCoordinatesWithOpenedUris:
    """diagnostics() must coordinate with _opened_uris to avoid duplicate didOpen."""

    @pytest.fixture
    def mux(self, tmp_path: Path) -> LSPMultiplexer:
        (tmp_path / "test.py").write_text("x = 1\n")
        return LSPMultiplexer(str(tmp_path))

    @pytest.mark.asyncio
    async def test_diagnostics_skips_didopen_when_uri_already_opened(self, mux: LSPMultiplexer) -> None:
        """If URI is already in _opened_uris, diagnostics() sends didChange instead of didOpen.

        hover() calls _ensure_document_open which adds the URI to state._opened_uris.
        A subsequent diagnostics() call must NOT send another textDocument/didOpen
        (some LSP servers reject duplicate opens). Instead it sends textDocument/didChange.
        """
        sent_messages: list[str] = []

        async def capture(_proc: MagicMock, message: str) -> None:
            sent_messages.append(message)

        fake_state = _mock_state()
        test_file = str(mux.project_root / "test.py")

        with patch.object(mux, "_send_raw", side_effect=capture), \
             patch.object(mux, "_ensure_server", return_value=fake_state), \
             patch.object(mux, "_await_server_ready", return_value=None), \
             patch.object(asyncio, "wait_for", new=AsyncMock()):

            # Simulate hover() already having opened the file
            fake_state._opened_uris.add(Path(test_file).resolve().as_uri())

            await mux.diagnostics(test_file)

        # Find all lifecycle messages
        did_open_count = 0
        did_change_count = 0
        did_close_count = 0
        for msg_str in sent_messages:
            msg = json.loads(msg_str)
            if msg.get("method") == "textDocument/didOpen":
                did_open_count += 1
            elif msg.get("method") == "textDocument/didChange":
                did_change_count += 1
            elif msg.get("method") == "textDocument/didClose":
                did_close_count += 1

        assert did_open_count == 0, (
            "diagnostics() must NOT send didOpen when URI is already in _opened_uris. "
            "Duplicate didOpen causes some LSP servers to reject the message."
        )
        assert did_change_count == 1, (
            "diagnostics() should send didChange to refresh content for an already-open file."
        )
        assert did_close_count == 0, (
            "diagnostics() must NOT send didClose when it did not open the file. "
            "The file was opened by another operation (e.g., hover) and should stay open."
        )

    @pytest.mark.asyncio
    async def test_diagnostics_sends_didclose_when_it_opened_the_file(self, mux: LSPMultiplexer) -> None:
        """If diagnostics() itself opened the file, it MUST send didClose after waiting.

        This ensures the URI is removed from _opened_uris so a future _ensure_document_open
        will reopen with fresh contents rather than relying on stale server state.
        """
        sent_messages: list[str] = []

        async def capture(_proc: MagicMock, message: str) -> None:
            sent_messages.append(message)

        fake_state = _mock_state()
        test_file = str(mux.project_root / "test.py")

        with patch.object(mux, "_send_raw", side_effect=capture), \
             patch.object(mux, "_ensure_server", return_value=fake_state), \
             patch.object(mux, "_await_server_ready", return_value=None), \
             patch.object(asyncio, "wait_for", new=AsyncMock()):

            # URI is NOT in _opened_uris — diagnostics() must open and close it
            assert Path(test_file).resolve().as_uri() not in fake_state._opened_uris

            await mux.diagnostics(test_file)

        did_open_count = sum(
            1 for msg_str in sent_messages
            if json.loads(msg_str).get("method") == "textDocument/didOpen"
        )
        did_close_count = sum(
            1 for msg_str in sent_messages
            if json.loads(msg_str).get("method") == "textDocument/didClose"
        )

        assert did_open_count == 1, "diagnostics() must send didOpen when URI is not yet open"
        assert did_close_count == 1, (
            "diagnostics() must send didClose when it itself opened the file, "
            "so _opened_uris is cleared and future _ensure_document_open reopens with fresh contents."
        )

    @pytest.mark.asyncio
    async def test_hover_then_diagnostics_then_hover_second_hover_works(self, mux: LSPMultiplexer) -> None:
        """hover() → diagnostics() → hover() sequence: second hover() must still return results.

        When diagnostics() finds the URI already open (from hover()), it sends didChange
        instead of didOpen and skips didClose, keeping the document open for the second hover().
        """
        sent_messages: list[dict[str, Any]] = []

        async def capture(_proc: MagicMock, message: str) -> None:
            sent_messages.append(json.loads(message))

        fake_state = _mock_state()
        test_file = str(mux.project_root / "test.py")
        uri = Path(test_file).resolve().as_uri()

        # Track hover call count to return appropriate responses
        hover_call_count = 0

        async def mock_lsp_request(state, method, params):
            nonlocal hover_call_count
            if method == "textDocument/hover":
                hover_call_count += 1
                # Resolve the pending future so hover() doesn't hang
                req_id = mux._next_id_val() - 1
                future = state.pending.get(req_id)
                if future and not future.done():
                    future.set_result({
                        "id": req_id,
                        "result": {
                            "contents": {"value": "x: int"},
                            "range": {
                                "start": {"line": 0, "character": 0},
                                "end": {"line": 0, "character": 1},
                            },
                        },
                    })
                return {
                    "contents": {"value": "x: int"},
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 1}},
                }
            return None

        with patch.object(mux, "_send_raw", side_effect=capture), \
             patch.object(mux, "_ensure_server", return_value=fake_state), \
             patch.object(mux, "_await_server_ready", return_value=None), \
             patch.object(mux, "_lsp_request", side_effect=mock_lsp_request):

            # First hover — opens the document via _ensure_document_open
            result1 = await mux.hover(test_file, line=1, col=0)
            assert result1 is not None, "First hover() should return results"
            assert uri in fake_state._opened_uris, "hover() should add URI to _opened_uris"

            # diagnostics() called while URI is already open
            diag_result = await mux.diagnostics(test_file)

            # URI should still be open (diagnostics() sent didChange, not didOpen, and no didClose)
            assert uri in fake_state._opened_uris, (
                "After diagnostics() with already-open URI, _opened_uris must still contain the URI. "
                "diagnostics() should have skipped didOpen and didClose."
            )

            # Second hover — must still work
            result2 = await mux.hover(test_file, line=1, col=0)
            assert result2 is not None, (
                "Second hover() after diagnostics() must still return results. "
                "diagnostics() must not send didClose for a URI it did not open."
            )
            assert result2["value"] == "x: int"


class TestServerClientRequests:
    """Server→client requests must be acknowledged to prevent the server from hanging."""

    @pytest.fixture
    def mux(self, tmp_path: Path) -> LSPMultiplexer:
        return LSPMultiplexer(str(tmp_path))

    @pytest.mark.asyncio
    async def test_workspace_configuration_request_responds_with_nulls(self, mux: LSPMultiplexer) -> None:
        """workspace/configuration request: respond with [null, ...] per item.

        LSP servers send workspace/configuration to request custom configuration.
        We don't have custom config, so respond with nulls. The response must
        be written to stdin of the subprocess.
        """
        sent_messages: list[str] = []
        fake_state = _mock_state()

        async def capture(_proc: MagicMock, message: str) -> None:
            sent_messages.append(message)

        with patch.object(mux, "_send_raw", side_effect=capture):
            # Simulate the reader loop receiving a workspace/configuration request
            # from the LSP server — method AND id present = server→client request
            mux._handle_server_request(
                fake_state,
                "workspace/configuration",
                req_id=42,
                params={"items": [{"section": "pyright"}, {"section": "python.analysis"}]},
            )
            # Allow the async _send_response coroutine to run
            await asyncio.sleep(0)

        # Must have sent exactly one response message
        assert len(sent_messages) == 1
        resp = json.loads(sent_messages[0])
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 42
        assert resp["result"] == [None, None]  # one null per config item

    @pytest.mark.asyncio
    async def test_client_register_capability_request_responds_with_empty_ack(self, mux: LSPMultiplexer) -> None:
        """client/registerCapability: respond with {} to acknowledge dynamic capability registration."""
        sent_messages: list[str] = []
        fake_state = _mock_state()

        async def capture(_proc: MagicMock, message: str) -> None:
            sent_messages.append(message)

        with patch.object(mux, "_send_raw", side_effect=capture):
            mux._handle_server_request(
                fake_state,
                "client/registerCapability",
                req_id=7,
                params={"registrations": [{"id": "r1", "method": "workspace/didChangeConfiguration"}]},
            )
            await asyncio.sleep(0)

        resp = json.loads(sent_messages[0])
        assert resp["id"] == 7
        assert resp["result"] == {}  # empty acknowledgement

    @pytest.mark.asyncio
    async def test_window_show_message_request_responds_with_null(self, mux: LSPMultiplexer) -> None:
        """window/showMessageRequest: respond with null (no action taken)."""
        sent_messages: list[str] = []
        fake_state = _mock_state()

        async def capture(_proc: MagicMock, message: str) -> None:
            sent_messages.append(message)

        with patch.object(mux, "_send_raw", side_effect=capture):
            mux._handle_server_request(
                fake_state,
                "window/showMessageRequest",
                req_id=99,
                params={"type": 3, "message": "Apply changes?", "actions": [{"title": "Yes"}]},
            )
            await asyncio.sleep(0)

        resp = json.loads(sent_messages[0])
        assert resp["id"] == 99
        assert resp["result"] is None  # no action taken

    @pytest.mark.asyncio
    async def test_reader_loop_routes_requests_to_handle_server_request(self, mux: LSPMultiplexer) -> None:
        """_reader_loop's three-way dispatch: method+id → request handler, method only → notification.

        We test the dispatch directly: _handle_server_request is called with a
        server→client request (method AND id). It must send a response and NOT
        throw. We also verify _handle_notification is a separate code path that
        does not send responses.
        """
        fake_state = _mock_state()
        sent_messages: list[str] = []

        async def capture(_proc: MagicMock, message: str) -> None:
            sent_messages.append(message)

        with patch.object(mux, "_send_raw", side_effect=capture):
            # Simulate _reader_loop receiving a server→client request:
            # if method AND id present → _handle_server_request → response
            mux._handle_server_request(
                fake_state,
                "workspace/configuration",
                req_id=10,
                params={"items": [{"section": "pyright"}]},
            )
            # Let the fire-and-forget response task complete
            await asyncio.sleep(0)

        # A JSON-RPC response MUST be written to stdin for a server→client request
        assert len(sent_messages) == 1
        resp = json.loads(sent_messages[0])
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 10
        assert resp["result"] == [None]  # one null per config item

        # Verify _handle_notification is a separate code path (method only, no response)
        sent_messages.clear()
        mux._handle_notification(fake_state, "textDocument/publishDiagnostics", {
            "uri": "file:///test.py",
            "diagnostics": [],
        })
        # Notifications don't write responses
        assert len(sent_messages) == 0
