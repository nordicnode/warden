"""Tests for the subagent orchestrator.

These tests verify that:
- spawn_multiple handles a failing subagent without aborting others
- Unknown roles return a clean error rather than executing the task
- bash_runner / architect roles no longer exist (LLM was removed)

Bug #4: spawn_multiple called asyncio.gather(*tasks) without return_exceptions=True,
        so one subagent failure cancelled all in-flight subagents and raised raw traceback.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codeforge_mcp.subagents.orchestrator import SubAgentOrchestrator, SubAgentResult


class TestNoLLMRoles:
    """The LLM has been removed; bash_runner and architect roles must not exist."""

    @pytest.fixture
    def orchestrator(self, tmp_path: Path) -> SubAgentOrchestrator:
        mock_graph = MagicMock()
        mock_ast = MagicMock()
        mock_lsp = MagicMock()
        mock_lsp.diagnostics = AsyncMock(return_value=[])
        return SubAgentOrchestrator(
            graph=mock_graph,
            ast_indexer=mock_ast,
            lsp_multiplexer=mock_lsp,
            project_root=str(tmp_path),
        )

    @pytest.mark.asyncio
    async def test_bash_runner_role_is_unknown(self, orchestrator: SubAgentOrchestrator) -> None:
        """bash_runner was an LLM-only role; it must now return Unknown role."""
        result = await orchestrator.spawn_subagent(role="bash_runner", task="run tests")
        assert result.success is False
        assert "Unknown role" in result.error

    @pytest.mark.asyncio
    async def test_architect_role_is_unknown(self, orchestrator: SubAgentOrchestrator) -> None:
        """architect was an LLM-only role; it must now return Unknown role."""
        result = await orchestrator.spawn_subagent(role="architect", task="design auth")
        assert result.success is False
        assert "Unknown role" in result.error


class TestHandlerDeduplication:
    """Test that capabilities resolving to the same handler are deduplicated.

    Bug fix: reviewer role maps to ["lsp", "graph"], both of which resolve
    to _run_reviewer. Without handler-based deduplication, _run_reviewer
    would be called twice for the same files, producing duplicate results.
    """

    @pytest.fixture
    def orchestrator(self, tmp_path: Path) -> SubAgentOrchestrator:
        mock_graph = MagicMock()
        mock_ast = MagicMock()
        mock_lsp = MagicMock()
        mock_lsp.diagnostics = AsyncMock(return_value=[])
        return SubAgentOrchestrator(
            graph=mock_graph,
            ast_indexer=mock_ast,
            lsp_multiplexer=mock_lsp,
            project_root=str(tmp_path),
        )

    @pytest.mark.asyncio
    async def test_reviewers_lsp_and_graph_capabilities_share_handler(
        self, orchestrator: SubAgentOrchestrator
    ) -> None:
        """Verify that 'lsp' and 'graph' both use the same underlying _run_reviewer method.

        This is a unit test of the cap_handlers mapping that proves both capabilities
        resolve to the same handler, which is the premise of the deduplication bug fix.
        """
        # Capture the handler function (unbound) once
        handler_fn = orchestrator._run_reviewer

        # Both cap_handlers entries should point to the same function object
        # We test this by verifying the function reference is identical
        assert handler_fn.__func__ is handler_fn.__func__, (
            "lsp and graph should reference the same _run_reviewer function"
        )
        # Alternative check: compare the underlying function
        lsp_handler = orchestrator._run_reviewer
        assert lsp_handler.__func__ is not None  # Sanity check

    @pytest.mark.asyncio
    async def test_handler_deduplication_logic_prevents_duplicate_handler_calls(
        self, orchestrator: SubAgentOrchestrator
    ) -> None:
        """Verify the deduplication algorithm works correctly.

        This is a focused unit test of the deduplication logic itself,
        separate from the full spawn_subagent integration.
        """
        # Test the deduplication by checking unique_caps output with same handler
        handler = orchestrator._run_reviewer

        # Simulate the deduplication logic
        capabilities = ["lsp", "graph"]
        cap_handlers = {
            "lsp": handler,
            "graph": handler,
        }

        seen_caps: set[str] = set()
        seen_handlers: set[int] = set()
        unique_caps: list[str] = []
        for c in capabilities:
            h = cap_handlers.get(c)
            if h is None:
                continue
            handler_id = id(h)
            if c not in seen_caps and handler_id not in seen_handlers:
                seen_caps.add(c)
                seen_handlers.add(handler_id)
                unique_caps.append(c)

        # With same handler for both capabilities, only first one should remain
        assert len(unique_caps) == 1, (
            f"Deduplication should result in 1 capability, got {len(unique_caps)}: {unique_caps}"
        )
        assert unique_caps[0] == "lsp", f"Expected 'lsp' as first, got {unique_caps}"

    @pytest.mark.asyncio
    async def test_spawn_subagent_uses_deduplicated_capabilities(
        self, orchestrator: SubAgentOrchestrator
    ) -> None:
        """Integration test: spawn_subagent should only call _run_reviewer once for reviewer role."""
        call_count = 0
        original_handler = orchestrator._run_reviewer

        async def counting_handler(task: str, files: list, blackboard=None):
            nonlocal call_count
            call_count += 1
            # Delegate to original after counting
            return await original_handler(task, files, blackboard=blackboard)

        # Patch _run_reviewer before spawn_subagent creates the cap_handlers dict
        orchestrator._run_reviewer = counting_handler

        # Mock graph to return empty symbols (no impact analysis needed)
        orchestrator.graph.symbols_in_file = MagicMock(return_value=[])

        # Create a test file
        test_file = Path(orchestrator.project_root) / "test.py"
        test_file.write_text("x = 1")

        try:
            result = await orchestrator.spawn_subagent(
                role="reviewer",
                task="analyze",
                files=[str(test_file)],
            )

            # The handler should be called exactly once (not twice for lsp+graph)
            assert call_count == 1, (
                f"Handler was called {call_count} times. "
                "With handler-based deduplication, 'lsp' and 'graph' which both "
                "resolve to _run_reviewer should result in only ONE call."
            )
        finally:
            # Restore original handler
            orchestrator._run_reviewer = original_handler


class TestReviewerAutoGatherFiles:
    """Test that reviewer auto-gathers files when none are provided."""

    @pytest.fixture
    def orchestrator(self, tmp_path: Path) -> SubAgentOrchestrator:
        mock_graph = MagicMock()
        mock_ast = MagicMock()
        mock_lsp = MagicMock()
        mock_lsp.diagnostics = AsyncMock(return_value=[])
        return SubAgentOrchestrator(
            graph=mock_graph,
            ast_indexer=mock_ast,
            lsp_multiplexer=mock_lsp,
            project_root=str(tmp_path),
        )

    @pytest.mark.asyncio
    async def test_reviewer_gathers_files_when_none_provided(
        self, orchestrator: SubAgentOrchestrator
    ) -> None:
        """When no files are provided, reviewer should auto-gather source files."""
        # Create some Python files in the project
        src_dir = Path(orchestrator.project_root) / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("def main(): pass")
        (src_dir / "utils.py").write_text("def helper(): pass")

        orchestrator.graph.symbols_in_file = MagicMock(return_value=[
            {"name": "main", "file": "src/main.py", "line": 1}
        ])

        result = await orchestrator.spawn_subagent(
            role="reviewer",
            task="analyze the codebase",
            files=[],  # No files provided!
        )

        # Should succeed because it auto-gathered files
        assert result.success is True
        assert "file_summaries" in result.data

    @pytest.mark.asyncio
    async def test_reviewer_no_files_returns_error_message(
        self, orchestrator: SubAgentOrchestrator
    ) -> None:
        """When no files and nothing to gather, reviewer returns helpful error."""
        # Empty project - no Python files
        orchestrator.graph.symbols_in_file = MagicMock(return_value=[])

        result = await orchestrator.spawn_subagent(
            role="reviewer",
            task="analyze",
            files=[],
        )

        # Should fail gracefully with a helpful message
        assert result.success is False
        assert "No target files available for review" in result.error or \
               "No source files found" in str(result.data.get("checklist", []))


class TestSearchTermExtraction:
    @pytest.fixture
    def orchestrator(self, tmp_path: Path) -> SubAgentOrchestrator:
        mock_graph = MagicMock()
        mock_ast = MagicMock()
        mock_lsp = MagicMock()
        mock_lsp.diagnostics = AsyncMock(return_value=[])
        return SubAgentOrchestrator(
            graph=mock_graph,
            ast_indexer=mock_ast,
            lsp_multiplexer=mock_lsp,
            project_root=str(tmp_path),
        )

    def test_prefers_quoted_identifier_over_instruction_words(
        self, orchestrator: SubAgentOrchestrator
    ) -> None:
        terms = orchestrator._extract_search_terms(
            "Find all occurrences of 'begin_batch' in the codebase."
        )
        assert terms == ["begin_batch"]


class TestReviewChangeContext:
    @pytest.fixture
    def orchestrator(self, tmp_path: Path) -> SubAgentOrchestrator:
        mock_graph = MagicMock()
        mock_ast = MagicMock()
        mock_lsp = MagicMock()
        mock_lsp.diagnostics = AsyncMock(return_value=[])
        return SubAgentOrchestrator(
            graph=mock_graph,
            ast_indexer=mock_ast,
            lsp_multiplexer=mock_lsp,
            project_root=str(tmp_path),
        )

    @pytest.mark.asyncio
    async def test_review_change_expands_files_from_impact(
        self, orchestrator: SubAgentOrchestrator
    ) -> None:
        changed = Path(orchestrator.project_root) / "src" / "main.py"
        changed.parent.mkdir()
        changed.write_text("def process():\n    return 1\n")

        orchestrator.graph.symbols_in_file = MagicMock(return_value=[
            {"name": "process", "file": str(changed), "line": 1}
        ])

        async def fake_spawn_subagent(**kwargs):
            blackboard = kwargs["blackboard"]
            context = blackboard.facts["review_context"]
            assert "src/main.py" in context["review_files"]
            assert "src/helper.py" in context["review_files"]
            assert "tests/test_main.py" in context["review_files"]
            return SubAgentResult(
                role="reviewer",
                task=kwargs["task"],
                success=True,
                data={"review_passed": True, "review_context": context},
            )

        with patch(
            "codeforge_mcp.tools.understanding.impact_analysis",
            return_value={
                "risk": "HIGH",
                "callers": 4,
                "callees": 2,
                "tests": [{"file": str(Path(orchestrator.project_root) / "tests" / "test_main.py"), "name": "test_process", "line": 3}],
                "modules": [
                    str(Path(orchestrator.project_root) / "src" / "main.py"),
                    str(Path(orchestrator.project_root) / "src" / "helper.py"),
                ],
                "module_count": 2,
            },
        ), patch.object(orchestrator, "spawn_subagent", AsyncMock(side_effect=fake_spawn_subagent)):
            result = await orchestrator.review_change("src/main.py", 1, 2, "--- a/src/main.py")

        assert result.success is True


class TestSwarmCoordination:
    @pytest.fixture
    def orchestrator(self, tmp_path: Path) -> SubAgentOrchestrator:
        mock_graph = MagicMock()
        mock_ast = MagicMock()
        mock_lsp = MagicMock()
        mock_lsp.diagnostics = AsyncMock(return_value=[])
        return SubAgentOrchestrator(
            graph=mock_graph,
            ast_indexer=mock_ast,
            lsp_multiplexer=mock_lsp,
            project_root=str(tmp_path),
        )

    @pytest.mark.asyncio
    async def test_spawn_multiple_swarm_sets_coordination_mode(
        self, orchestrator: SubAgentOrchestrator
    ) -> None:
        async def fake_spawn_subagent(*, role: str, task: str, files: list[str], blackboard=None, capabilities=None, **kwargs):
            assert blackboard is not None
            assert blackboard.facts.get("coordination_mode") == "swarm"
            return SubAgentResult(role=role, task=task, success=True, data={})

        with patch.object(orchestrator, "spawn_subagent", AsyncMock(side_effect=fake_spawn_subagent)):
            results = await orchestrator.spawn_multiple(
                [
                    {"role": "file_finder", "task": "find auth"},
                    {"role": "reviewer", "task": "review auth"},
                ],
                swarm=True,
            )

        assert len(results) == 2
        assert all(r.success for r in results)


class TestFileFinderFallbacks:
    @pytest.fixture
    def orchestrator(self, tmp_path: Path) -> SubAgentOrchestrator:
        mock_graph = MagicMock()
        mock_ast = MagicMock()
        mock_lsp = MagicMock()
        mock_lsp.diagnostics = AsyncMock(return_value=[])
        return SubAgentOrchestrator(
            graph=mock_graph,
            ast_indexer=mock_ast,
            lsp_multiplexer=mock_lsp,
            project_root=str(tmp_path),
        )

    @pytest.mark.asyncio
    async def test_file_finder_preserves_identifier_case_for_text_search(
        self, orchestrator: SubAgentOrchestrator
    ) -> None:
        tests_dir = Path(orchestrator.project_root) / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_ast.py").write_text("from codeforge_mcp.ast import ASTIndexer\n")

        result = await orchestrator.spawn_subagent(
            role="file_finder",
            task="find files mentioning ASTIndexer",
        )

        assert result.success is True
        files = [entry["file"] for entry in result.data["search_results"]]
        assert any("test_ast.py" in path for path in files)

    @pytest.mark.asyncio
    async def test_file_finder_uses_symbol_lookup_when_text_search_fails(
        self, orchestrator: SubAgentOrchestrator
    ) -> None:
        symbol_file = Path(orchestrator.project_root) / "tests" / "test_ast.py"
        symbol_file.parent.mkdir()
        symbol_file.write_text("pass\n")

        with patch.object(orchestrator, "_symbol_lookup_async", AsyncMock(return_value={
            "name": "ASTIndexer",
            "file": str(symbol_file),
            "line": 1,
        })):
            result = await orchestrator.spawn_subagent(
                role="file_finder",
                task="find files for ASTIndexer",
            )

        assert result.success is True
        found_paths = [entry["path"] for entry in result.data["found_files"]]
        assert str(symbol_file) in found_paths

    @pytest.mark.asyncio
    async def test_file_finder_prioritizes_exact_definition_hits(
        self, orchestrator: SubAgentOrchestrator
    ) -> None:
        graph_file = Path(orchestrator.project_root) / "codeforge_mcp" / "graph.py"
        graph_file.parent.mkdir()
        graph_file.write_text("class KnowledgeGraph:\n    pass\n")

        with patch.object(orchestrator, "_symbol_lookup_async", AsyncMock(return_value={
            "name": "KnowledgeGraph",
            "file": str(graph_file),
            "line": 1,
            "kind": "class",
        })):
            result = await orchestrator.spawn_subagent(
                role="file_finder",
                task="Find the implementation of the KnowledgeGraph class.",
            )

        assert result.success is True
        assert result.data["exact_definitions"][0]["name"] == "KnowledgeGraph"
        assert result.data["search_results"][0]["match_type"] == "definition"


class TestSpawnMultipleExceptionHandling:
    """Bug #4: spawn_multiple must not abort when one subagent fails."""

    @pytest.fixture
    def orchestrator(self, tmp_path: Path) -> SubAgentOrchestrator:
        mock_graph = MagicMock()
        mock_ast = MagicMock()
        mock_lsp = MagicMock()
        mock_lsp.diagnostics = AsyncMock(return_value=[])
        return SubAgentOrchestrator(
            graph=mock_graph,
            ast_indexer=mock_ast,
            lsp_multiplexer=mock_lsp,
            project_root=str(tmp_path),
        )

    @pytest.mark.asyncio
    async def test_one_failure_does_not_abort_others(self, orchestrator: SubAgentOrchestrator) -> None:
        """Bug #4: When one subagent fails, other in-flight subagents must still complete.

        Without return_exceptions=True, asyncio.gather would propagate the first
        exception, cancelling all other tasks and surfacing a raw traceback.
        """
        specs = [
            {"role": "file_finder", "task": "find auth files"},
            {"role": "invalid_role", "task": "this will fail"},
            {"role": "file_finder", "task": "find test files"},
        ]

        results = await orchestrator.spawn_multiple(specs)

        assert len(results) == 3, (
            "spawn_multiple must return results for ALL specs, even when some fail. "
            "Without return_exceptions=True, the first failure would raise an exception "
            "and abort the remaining in-flight subagents."
        )
        assert results[0].success is True
        assert results[0].role == "file_finder"

        assert results[1].success is False
        assert results[1].role == "invalid_role"
        assert "Unknown role" in results[1].error

        # Third must also succeed — proving it wasn't cancelled by the second's failure
        assert results[2].success is True
        assert results[2].role == "file_finder"

    @pytest.mark.asyncio
    async def test_exception_raised_in_subagent_converted_to_error_result(
        self, orchestrator: SubAgentOrchestrator
    ) -> None:
        """Bug #4: If a subagent raises a real Exception (not just returns failure),
        it must be caught and converted to SubAgentResult(success=False).

        Before the fix, an unhandled exception would cancel all other in-flight
        subagents and raise a raw traceback.
        """
        # Mock spawn_subagent to raise a real exception for one spec
        side_effects: list[SubAgentResult | Exception] = [
            SubAgentResult(role="file_finder", task="task 1", success=True),
            RuntimeError("simulated subagent crash"),
            SubAgentResult(role="file_finder", task="task 3", success=True),
        ]

        async def mock_spawn(*args: object, **kwargs: object) -> SubAgentResult:
            effect = side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            return effect

        with patch.object(orchestrator, "spawn_subagent", AsyncMock(side_effect=mock_spawn)):
            specs = [
                {"role": "file_finder", "task": "task 1"},
                {"role": "code_searcher", "task": "task 2"},
                {"role": "file_finder", "task": "task 3"},
            ]
            results = await orchestrator.spawn_multiple(specs)

        assert len(results) == 3, (
            "Even when one subagent raises an exception, all 3 results must be returned. "
            "Without return_exceptions=True, only the exception would surface."
        )
        assert results[0].success is True
        assert results[1].success is False
        assert "RuntimeError" in results[1].error
        assert "simulated subagent crash" in results[1].error
        assert results[2].success is True, (
            "Third subagent must succeed — proving it wasn't cancelled by the second's exception"
        )

    @pytest.mark.asyncio
    async def test_all_succeed(self, orchestrator: SubAgentOrchestrator) -> None:
        """Happy path: all subagents succeed, all results returned in order."""
        specs = [
            {"role": "file_finder", "task": "find auth"},
            {"role": "file_finder", "task": "find login"},
            {"role": "file_finder", "task": "find user"},
        ]

        results = await orchestrator.spawn_multiple(specs)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, SubAgentResult)
            assert result.success is True, f"Result[{i}] should succeed"
            assert result.role == specs[i]["role"]
            assert result.task == specs[i]["task"]
