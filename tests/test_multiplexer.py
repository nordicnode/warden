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
