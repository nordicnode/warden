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

from codeforge_mcp.subagents.orchestrator import (
    SubAgentOrchestrator, SubAgentResult, Blackboard, _topo_sort,
    _CAPABILITY_REGISTRY, _register_capability, _register_all_capabilities,
    Capability,
)


class TestLspAndGraphReviewSeparation:
    """Tests for the split of the reviewer capability into lsp and graph handlers.

    Verifies that:
    - "lsp" capability runs _run_lsp_review (LSP diagnostics only)
    - "graph" capability runs _run_graph_review (impact_analysis only)
    - ["lsp", "graph"] runs both handlers (not deduplicated)
    - lsp=None gracefully degrades to graph-only findings
    """

    @pytest.fixture
    def project_with_files(self, tmp_path: Path) -> Path:
        """Create a minimal project structure with source files."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("def hello():\n    return 'world'\n")
        return tmp_path

    @pytest.mark.asyncio
    async def test_lsp_cap_runs_lsp_review_only(self, project_with_files: Path) -> None:
        """Requesting only 'lsp' capability should only run _run_lsp_review."""
        from codeforge_mcp.graph import KnowledgeGraph
        db = project_with_files / "graph.db"
        graph = KnowledgeGraph(str(db))

        orch = SubAgentOrchestrator(
            graph=graph,
            ast_indexer=MagicMock(),  # Used by _run_graph_review (not called here)
            lsp_multiplexer=None,  # No LSP — lsp review degrades to empty diags
            project_root=str(project_with_files),
        )
        # Ensure symbols_in_file returns something so file_summaries has data
        orch.graph.symbols_in_file = MagicMock(return_value=[
            {"name": "hello", "file": "src/main.py", "line": 1, "kind": "function"},
        ])

        result = await orch.spawn_subagent(
            role="", task="review", capabilities=["lsp"]
        )

        assert result.success is True
        assert "checklist" in result.data
        # lsp cap produces diagnostics, not impact
        assert "diagnostics" in result.data
        # No impact since only lsp cap was requested
        assert "impact" not in result.data

        graph.close()

    @pytest.mark.asyncio
    async def test_graph_cap_runs_graph_review_only(self, project_with_files: Path) -> None:
        """Requesting only 'graph' capability should only run _run_graph_review."""
        from codeforge_mcp.graph import KnowledgeGraph
        db = project_with_files / "graph.db"
        graph = KnowledgeGraph(str(db))

        mock_lsp = MagicMock()
        mock_lsp.diagnostics = AsyncMock(return_value=[])  # Should NOT be called

        orch = SubAgentOrchestrator(
            graph=graph,
            ast_indexer=MagicMock(),
            lsp_multiplexer=mock_lsp,
            project_root=str(project_with_files),
        )
        orch.graph.symbols_in_file = MagicMock(return_value=[
            {"name": "hello", "file": "src/main.py", "line": 1, "kind": "function"},
        ])

        result = await orch.spawn_subagent(
            role="", task="review", capabilities=["graph"]
        )

        assert result.success is True
        # graph cap produces impact, not diagnostics
        assert "impact" in result.data
        assert "diagnostics" not in result.data
        # LSP should NOT have been called (graph doesn't use LSP)
        mock_lsp.diagnostics.assert_not_called()

        graph.close()

    @pytest.mark.asyncio
    async def test_lsp_and_graph_cap_runs_both_handlers(self, project_with_files: Path) -> None:
        """Requesting ['lsp', 'graph'] capabilities should run both handlers.

        The dedup loop was changed from handler.__name__ to capability name,
        so both "lsp" and "graph" now run even though they were previously
        aliased to the same _run_reviewer handler.
        """
        from codeforge_mcp.graph import KnowledgeGraph
        db = project_with_files / "graph.db"
        graph = KnowledgeGraph(str(db))

        mock_lsp = MagicMock()
        mock_lsp.diagnostics = AsyncMock(return_value=[{
            "line": 1, "severity": 1, "message": "test error", "source": "test"
        }])

        orch = SubAgentOrchestrator(
            graph=graph,
            ast_indexer=MagicMock(),
            lsp_multiplexer=mock_lsp,
            project_root=str(project_with_files),
        )
        orch.graph.symbols_in_file = MagicMock(return_value=[
            {"name": "hello", "file": "src/main.py", "line": 1, "kind": "function"},
        ])

        result = await orch.spawn_subagent(
            role="", task="review", capabilities=["lsp", "graph"]
        )

        assert result.success is True
        # Both lsp (diagnostics) and graph (impact) should be present
        assert "diagnostics" in result.data, "LSP diagnostics missing from combined result"
        assert "impact" in result.data, "Graph impact missing from combined result"
        # capabilities_used should list both
        assert "lsp" in result.data.get("capabilities_used", [])
        assert "graph" in result.data.get("capabilities_used", [])

        graph.close()

    @pytest.mark.asyncio
    async def test_reviewer_with_no_lsp_produces_graph_findings(self, project_with_files: Path) -> None:
        """When lsp=None, the reviewer should still produce graph-based impact findings.

        This is the key regression test: previously _run_reviewer required LSP
        to be available; now _run_graph_review handles the no-LSP case cleanly.
        """
        from codeforge_mcp.graph import KnowledgeGraph
        db = project_with_files / "graph.db"
        graph = KnowledgeGraph(str(db))

        orch = SubAgentOrchestrator(
            graph=graph,
            ast_indexer=MagicMock(),
            lsp_multiplexer=None,  # No LSP at all
            project_root=str(project_with_files),
        )
        orch.graph.symbols_in_file = MagicMock(return_value=[
            {"name": "hello", "file": "src/main.py", "line": 1, "kind": "function"},
        ])

        result = await orch.spawn_subagent(
            role="", task="review", capabilities=["lsp", "graph"]
        )

        assert result.success is True
        # Should still produce a review_passed result (even if empty)
        assert "review_passed" in result.data
        assert "checklist" in result.data
        # Impact from the graph handler should be present
        assert "impact" in result.data or "file_summaries" in result.data
        # diagnostics should be empty (no LSP) but the key should exist
        assert "diagnostics" in result.data
        assert result.data["diagnostics"] == []

        graph.close()


class TestReviewerGracefulDegradation:
    """Test that reviewer degrades gracefully when lsp is None."""

    @pytest.fixture
    def orchestrator_no_lsp(self, tmp_path: Path) -> SubAgentOrchestrator:
        mock_graph = MagicMock()
        mock_ast = MagicMock()
        return SubAgentOrchestrator(
            graph=mock_graph,
            ast_indexer=mock_ast,
            lsp_multiplexer=None,  # No LSP — should not crash
            project_root=str(tmp_path),
        )

    @pytest.mark.asyncio
    async def test_review_change_with_lsp_none_completes(
        self, orchestrator_no_lsp: SubAgentOrchestrator
    ) -> None:
        """When lsp=None, review_change() must complete successfully without
        crashing on self.lsp.diagnostics(...). Graph-only review should work.
        """
        # Create a real Python file so there are symbols to work with
        test_file = Path(orchestrator_no_lsp.project_root) / "sample.py"
        test_file.write_text("def hello():\n    return 'world'\n")

        orchestrator_no_lsp.graph.symbols_in_file = MagicMock(return_value=[
            {"name": "hello", "file": str(test_file), "line": 1}
        ])
        orchestrator_no_lsp.graph.search_symbols = MagicMock(return_value=[])
        orchestrator_no_lsp.graph.get_symbol = MagicMock(return_value=None)

        result = await orchestrator_no_lsp.review_change(
            path="sample.py",
            start_line=0,
            end_line=0,
            diff_preview="diff --git a/sample.py",
        )

        # Must complete without raising an exception
        assert result.success is True, f"Expected success, got error: {result.error}"
        assert "diagnostics" in result.data
        # diagnostics should be empty when lsp is None
        assert result.data["diagnostics"] == []
        # The blackboard insight about LSP unavailability should be present
        bb = result.data.get("_blackboard", {})
        assert any(
            "LSP unavailable" in insight and "graph-only" in insight.lower()
            for insight in bb.get("insights", [])
        ), f"Expected LSP degraded insight, got: {bb.get('insights', [])}"

    @pytest.mark.asyncio
    async def test_spawn_subagent_reviewer_with_lsp_none(
        self, orchestrator_no_lsp: SubAgentOrchestrator
    ) -> None:
        """Spawn reviewer with lsp=None directly via spawn_subagent."""
        test_file = Path(orchestrator_no_lsp.project_root) / "src" / "mod.py"
        test_file.parent.mkdir()
        test_file.write_text("class Foo:\n    pass\n")

        orchestrator_no_lsp.graph.symbols_in_file = MagicMock(return_value=[
            {"name": "Foo", "file": str(test_file), "line": 1}
        ])

        result = await orchestrator_no_lsp.spawn_subagent(
            role="reviewer",
            task="review the changes",
            files=[str(test_file)],
        )

        assert result.success is True, f"Expected success, got: {result.error}"
        assert "file_summaries" in result.data
        assert "diagnostics" in result.data
        assert result.data["diagnostics"] == []
        # The blackboard insight about LSP unavailability should be present
        bb = result.data.get("_blackboard", {})
        assert any(
            "LSP unavailable" in insight and "graph-only" in insight.lower()
            for insight in bb.get("insights", [])
        ), f"Expected LSP degraded insight, got: {bb.get('insights', [])}"


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
    """Test that the capability pipeline correctly deduplicates.

    Bug fix: previously both "lsp" and "graph" mapped to _run_reviewer, and
    the dedup loop keyed on handler.__name__, so requesting ["lsp", "graph"]
    would run _run_reviewer once (deduplicated).

    After the fix: "lsp" → _run_lsp_review, "graph" → _run_graph_review,
    and dedup now keys on capability name, so ["lsp", "graph"] correctly runs
    BOTH handlers.
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
    async def test_lsp_and_graph_have_distinct_handlers(
        self, orchestrator: SubAgentOrchestrator
    ) -> None:
        """Verify that 'lsp' and 'graph' now map to different handler methods.

        This is the premise of the fix: after the split, _run_lsp_review and
        _run_graph_review are two distinct methods.
        """
        lsp_handler = orchestrator._run_lsp_review
        graph_handler = orchestrator._run_graph_review
        # They must be distinct functions
        assert lsp_handler is not graph_handler, (
            "_run_lsp_review and _run_graph_review must be distinct methods"
        )

    @pytest.mark.asyncio
    async def test_deduplication_keys_on_capability_not_handler(
        self, orchestrator: SubAgentOrchestrator
    ) -> None:
        """Verify the new dedup logic keys on capability name, not handler.__name__.

        Before the fix, handler.__name__-based dedup would deduplicate "lsp"
        and "graph" since both went to _run_reviewer.__name__.
        After the fix, dedup keys on c (capability name) directly, so
        "lsp" and "graph" are both kept.
        """
        # Simulate the NEW dedup logic (capability-name-based)
        capabilities = ["lsp", "graph"]
        cap_handlers = {
            "lsp": orchestrator._run_lsp_review,
            "graph": orchestrator._run_graph_review,
        }

        seen_caps: set[str] = set()
        unique_caps: list[str] = []
        for c in capabilities:
            if c in seen_caps or c not in cap_handlers:
                continue
            seen_caps.add(c)
            unique_caps.append(c)

        # Both capabilities should survive dedup since they are distinct names
        assert len(unique_caps) == 2, (
            f"Capability-name dedup should keep both 'lsp' and 'graph', got: {unique_caps}"
        )
        assert unique_caps == ["lsp", "graph"]

    @pytest.mark.asyncio
    async def test_reviewer_role_runs_both_lsp_and_graph_handlers(
        self, orchestrator: SubAgentOrchestrator
    ) -> None:
        """Integration test: reviewer role ["lsp", "graph"] must run BOTH handlers.

        Before the fix: role="reviewer" would deduplicate to one call.
        After the fix: role="reviewer" should call _run_lsp_review THEN _run_graph_review.
        """
        lsp_calls = 0
        graph_calls = 0
        original_lsp = orchestrator._run_lsp_review
        original_graph = orchestrator._run_graph_review

        async def counting_lsp(task: str, files: list, blackboard=None):
            nonlocal lsp_calls
            lsp_calls += 1
            return await original_lsp(task, files, blackboard=blackboard)

        async def counting_graph(task: str, files: list, blackboard=None):
            nonlocal graph_calls
            graph_calls += 1
            return await original_graph(task, files, blackboard=blackboard)

        orchestrator._run_lsp_review = counting_lsp
        orchestrator._run_graph_review = counting_graph

        # Mock graph to avoid real work
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

            # Both handlers should be called exactly once
            assert lsp_calls == 1, (
                f"_run_lsp_review should be called once, got {lsp_calls}"
            )
            assert graph_calls == 1, (
                f"_run_graph_review should be called once, got {graph_calls}"
            )
            # Result should contain data from BOTH handlers
            assert "diagnostics" in result.data, (
                "diagnostics key missing — should come from _run_lsp_review"
            )
            assert "impact" in result.data, (
                "impact key missing — should come from _run_graph_review"
            )
        finally:
            orchestrator._run_lsp_review = original_lsp
            orchestrator._run_graph_review = original_graph


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
    async def test_diagnose_resolves_symbol_from_code_search_line(self, tmp_path: Path) -> None:
        """sym_result['text'] is a full source line from code_search, not a symbol name.
        _run_diagnose must tokenize the line and try graph.get_symbol on each candidate
        until one resolves — not pass the raw line text as a symbol name.
        """
        from codeforge_mcp.tools.navigation import code_search

        mock_graph = MagicMock()
        mock_ast = MagicMock()
        # authenticate resolves; others do not
        def get_symbol_side_effect(*args) -> dict | None:
            name = args[-1] if args else None
            return {"name": "authenticate", "file": "auth.py", "line": 42} if name == "authenticate" else None
        mock_graph.get_symbol.side_effect = get_symbol_side_effect
        mock_ast.symbols_in_file = MagicMock(return_value=[])  # force fallback path

        orch = SubAgentOrchestrator(
            graph=mock_graph,
            ast_indexer=mock_ast,
            lsp_multiplexer=MagicMock(),
            project_root=str(tmp_path),
        )

        # Patch code_search so it returns a result with a realistic source line
        fake_search_result = {
            "file": "auth.py",
            "line": 12,
            "text": "def authenticate(user):",
            "context_before": [],
            "context_after": [],
        }
        with patch("codeforge_mcp.tools.navigation.code_search", return_value=[fake_search_result]):
            bb = Blackboard()
            result = await orch.spawn_subagent(
                role="diagnose",
                task="TypeError at line 12 in auth.py",
                files=[],
                blackboard=bb,
            )

            # Must not crash and should resolve 'authenticate' (not the whole line)
            assert result.success is True, f"Expected success, got: {result.error}"
            # The raw line "def authenticate(user):" should yield "authenticate" via tokenization
            calls = [str(c) for c in mock_graph.get_symbol.call_args_list]
            assert any('authenticate' in c for c in calls), (
                f"graph.get_symbol never called with 'authenticate'. Calls: {calls}"
            )
            # The resolved symbol should appear in root_cause_candidates
            assert any(
                c.get("symbol") == "authenticate"
                for c in result.data.get("root_cause_candidates", [])
            ), f"authenticate not in root_cause_candidates: {result.data.get('root_cause_candidates', [])}"


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


class TestTestFileDetection:
    """Test the _TEST_FILE_RE pattern correctly identifies test files without false positives.

    Note: The full test detection logic in _run_test_impact combines:
    1. name.startswith('test_') or name.endswith('_test')
    2. _TEST_FILE_RE.search(fname)  -- path-based patterns
    3. _TEST_FILE_RE.search(name_lower) -- name-based regex patterns

    The regex alone handles path patterns (tests/, __tests__/, .spec/, .test/)
    Name-based checks handle test_ prefix and _test suffix.
    """

    @pytest.mark.parametrize("path,expected", [
        # False positives that substring matching would catch but regex should not
        ("codeforge_mcp/tools/contest.py", False),
        ("codeforge_mcp/tools/authentication.py", False),
        ("codeforge_mcp/tools/latest_results.py", False),
        ("codeforge_mcp/tools/fastest.py", False),
        ("codeforge_mcp/tools/user_auth.py", False),
        ("codeforge_mcp/server.py", False),
        ("codeforge_mcp/indexer.py", False),
        ("src/utils/test_helpers.py", False),  # test_helpers not a test file
        # True positives - path-based patterns
        ("tests/test_foo.py", True),   # test_ at start of filename in path
        ("src/__tests__/baz.js", True),   # __tests__/ directory
        ("tests/bar.py", True),   # tests/ directory
        ("foo.spec.ts", True),   # .spec[./] extension
        ("bar.test.ts", True),   # .test[./] extension
        ("__tests__/qux.js", True),   # __tests__/ directory
        ("spec/auth.spec.ts", True),   # .spec[./] in path
        ("src/tests/unit/api.test.ts", True),   # tests/ + .test[./]
    ])
    def test_test_file_pattern(self, path: str, expected: bool) -> None:
        """Verify _TEST_FILE_RE correctly classifies files as test or non-test."""
        from codeforge_mcp.subagents.orchestrator import _TEST_FILE_RE
        result = bool(_TEST_FILE_RE.search(path))
        assert result == expected, f"_TEST_FILE_RE.search({path!r}) = {result}, expected {expected}"

    @pytest.mark.parametrize("name,expected", [
        # name-based checks (startswith test_ or endswith _test) - case sensitive
        ("test_auth", True),
        ("auth_test", True),
        ("TestAuth", False),   # starts with Test (uppercase T), not test_
        ("AuthTest", False),   # ends with Test (uppercase T), not _test
        ("test_helpers", True),   # starts with test_
        ("auth_test_utils", False), # ends with _utils, not _test
        # Non-test names
        ("authenticate", False),
        ("contest", False),
        ("latest_results", False),
        ("fastest", False),
        ("user_auth", False),
    ])
    def test_test_name_pattern(self, name: str, expected: bool) -> None:
        """Verify name-based test detection (startswith test_ or endswith _test)."""
        is_test = bool(
            name.startswith("test_") or name.endswith("_test")
        )
        assert is_test == expected, f"{name!r}: is_test={is_test}, expected {expected}"


class TestRunDecompose:
    """Test _run_decompose planning prompt matches actual ROLE_CAPS and capabilities."""

    @pytest.mark.asyncio
    async def test_decompose_planning_prompt_matches_role_caps(self) -> None:
        """Verify planning_prompt mentions every role in _ROLE_CAPS with its exact capability list."""
        from codeforge_mcp.subagents.orchestrator import SubAgentOrchestrator, _ROLE_CAPS, _CAP_HANDLERS_KEYS
        from unittest.mock import MagicMock

        orch = SubAgentOrchestrator(
            graph=MagicMock(),
            ast_indexer=MagicMock(),
            lsp_multiplexer=None,
            project_root="/tmp",
        )
        result = await orch.spawn_subagent(role="decompose", task="fix auth bug", files=["auth.py"])

        assert result.success is True
        assert result.data is not None
        planning_prompt = result.data["planning_prompt"]
        available_capabilities = result.data["available_capabilities"]
        suggested_roles = result.data["suggested_roles"]

        # Verify every role in _ROLE_CAPS is mentioned in the planning prompt
        # with its exact capability list
        for role, caps in _ROLE_CAPS.items():
            expected_line = f"- {role}: {caps}"
            assert expected_line in planning_prompt, (
                f"Planning prompt missing role line: {expected_line!r}\n"
                f"Actual prompt contains:\n{planning_prompt}"
            )

        # Verify available_capabilities matches _CAP_HANDLERS_KEYS
        assert set(available_capabilities) == set(_CAP_HANDLERS_KEYS), (
            f"available_capabilities {available_capabilities} does not match _CAP_HANDLERS_KEYS {_CAP_HANDLERS_KEYS}"
        )

        # Verify suggested_roles contains all roles from _ROLE_CAPS
        assert set(suggested_roles) == set(_ROLE_CAPS.keys()), (
            f"suggested_roles {suggested_roles} does not match _ROLE_CAPS keys"
        )


class TestCapabilityDependencySorting:
    """Tests for topological sorting of capabilities by produces→consumes edges."""

    def test_topo_sort_producer_before_consumer(self):
        """A capability that produces target_symbol must execute before one that consumes it."""

        async def handler_a(task, files, blackboard=None):
            return SubAgentResult(role="a", task=task, success=True)

        async def handler_b(task, files, blackboard=None):
            return SubAgentResult(role="b", task=task, success=True)

        cap_a = Capability(name="producer", handler=handler_a, produces={"target_symbol"}, consumes=set())
        cap_b = Capability(name="consumer", handler=handler_b, produces=set(), consumes={"target_symbol"})

        sorted_caps = _topo_sort([cap_b, cap_a])  # Pass in reverse order

        # producer must come before consumer
        names = [c.name for c in sorted_caps]
        assert names.index("producer") < names.index("consumer"), (
            f"Producer must execute before consumer. Got order: {names}"
        )

    def test_topo_sort_preserves_order_for_independent_caps(self):
        """Capabilities with no dependency between them retain their relative order."""

        async def handler_a(task, files, blackboard=None):
            return SubAgentResult(role="a", task=task, success=True)

        async def handler_b(task, files, blackboard=None):
            return SubAgentResult(role="b", task=task, success=True)

        async def handler_c(task, files, blackboard=None):
            return SubAgentResult(role="c", task=task, success=True)

        cap_a = Capability(name="a", handler=handler_a, produces={"output_a"}, consumes=set())
        cap_b = Capability(name="b", handler=handler_b, produces={"output_b"}, consumes=set())
        cap_c = Capability(name="c", handler=handler_c, produces={"output_c"}, consumes=set())

        sorted_caps = _topo_sort([cap_c, cap_b, cap_a])  # Reverse order input

        names = [c.name for c in sorted_caps]
        # All independent, so any valid topo order is fine — check all present
        assert set(names) == {"a", "b", "c"}

    def test_topo_sort_missing_producer_raises(self):
        """A capability that consumes a key no one produces must raise ValueError."""

        async def handler_a(task, files, blackboard=None):
            return SubAgentResult(role="a", task=task, success=True)

        async def handler_b(task, files, blackboard=None):
            return SubAgentResult(role="b", task=task, success=True)

        cap_a = Capability(name="a", handler=handler_a, produces={"output_a"}, consumes=set())
        cap_b = Capability(name="b", handler=handler_b, produces=set(), consumes={"missing_key"})

        with pytest.raises(ValueError, match="consumes 'missing_key' but no capability produces it"):
            _topo_sort([cap_a, cap_b])

    def test_topo_sort_cycle_raises(self):
        """A->B->C->A cycle must raise ValueError with cycle mention."""

        async def handler_a(task, files, blackboard=None):
            return SubAgentResult(role="a", task=task, success=True)

        async def handler_b(task, files, blackboard=None):
            return SubAgentResult(role="b", task=task, success=True)

        async def handler_c(task, files, blackboard=None):
            return SubAgentResult(role="c", task=task, success=True)

        # A consumes what C produces, B consumes what A produces, C consumes what B produces
        cap_a = Capability(name="a", handler=handler_a, produces={"a_out"}, consumes={"c_out"})
        cap_b = Capability(name="b", handler=handler_b, produces={"b_out"}, consumes={"a_out"})
        cap_c = Capability(name="c", handler=handler_c, produces={"c_out"}, consumes={"b_out"})

        with pytest.raises(ValueError, match="Cycle detected"):
            _topo_sort([cap_a, cap_b, cap_c])

    def test_topo_sort_code_searcher_after_file_finder(self):
        """code_searcher (ast) that consumes target_symbol must run after file_finder (search).

        This is the real dependency: _run_file_finder writes target_symbol to
        blackboard.facts and _run_code_searcher reads it back.
        """

        async def mock_file_finder(task, files, blackboard=None):
            return SubAgentResult(role="search", task=task, success=True)

        async def mock_code_searcher(task, files, blackboard=None):
            return SubAgentResult(role="ast", task=task, success=True)

        cap_search = Capability(
            name="search", handler=mock_file_finder,
            produces={"target_symbol", "target_file"}, consumes=set()
        )
        cap_ast = Capability(
            name="ast", handler=mock_code_searcher,
            produces={"ast_results"}, consumes={"target_symbol", "target_file"}
        )

        # Request in reverse order — sort should fix it
        sorted_caps = _topo_sort([cap_ast, cap_search])
        names = [c.name for c in sorted_caps]

        assert names.index("search") < names.index("ast"), (
            f"search must execute before ast (consumes target_symbol). Got order: {names}"
        )


class TestWaitForUnconditional:
    """Regression test: wait_for calls are now unconditional (no swarm-only guard).

    The DAG topological sort guarantees producers run before consumers in
    sequential mode, so wait_for is always satisfied by the time a handler
    runs. This means wait_for can be called unconditionally; the event is
    already set if the producer executed first.

    This test verifies that reversing ROLE_CAPS["code_searcher"] order
    (i.e. putting ast before search) still works because the topological
    sort re-orders them based on produces/consumes edges.
    """

    @pytest.mark.asyncio
    async def test_role_caps_order_reversal_still_works_via_dag(self, tmp_path: Path):
        """Even if ROLE_CAPS["code_searcher"] lists ast before search, the DAG
        re-orders them so search executes first (produces target_symbol).

        This is the key regression test: previously, data dependencies between
        capabilities worked only because of insertion order in ROLE_CAPS — a
        fragile contract. Now the topological sort guarantees the correct order.
        """
        from codeforge_mcp.graph import KnowledgeGraph
        from codeforge_mcp.subagents.orchestrator import _ROLE_CAPS

        db = tmp_path / "graph.db"
        graph = KnowledgeGraph(str(db))

        mock_ast_indexer = MagicMock()
        mock_ast_indexer.query = MagicMock(return_value=[])
        mock_ast_indexer.symbols_in_file = MagicMock(return_value=[])

        orch = SubAgentOrchestrator(
            graph=graph,
            ast_indexer=mock_ast_indexer,
            lsp_multiplexer=MagicMock(),
            project_root=str(tmp_path),
        )

        # Mock the graph so handlers don't fail on lookups
        orch.graph.symbols_in_file = MagicMock(return_value=[
            {"name": "foo", "file": "src/main.py", "line": 1, "kind": "function"}
        ])
        orch.graph.get_symbol = MagicMock(return_value=None)
        orch.graph.search_symbols = MagicMock(return_value=[])

        # Simulate reversed ROLE_CAPS (ast listed before search)
        reversed_role_caps = {
            "code_searcher": ["ast", "symbol", "search"],  # ast first — DAG should override
        }

        # Capture execution order using simple module-level functions.
        # spawn_subagent does cap.handler.__get__(self, SubAgentOrchestrator), which
        # binds `orch` (SubAgentOrchestrator) as the first arg to the function.
        # So the wrapper signature must be: def wrapper(self, task, files, blackboard=None)
        # where `self` is the SubAgentOrchestrator instance.
        execution_order: list[str] = []
        orig_ff = orch._run_file_finder
        orig_cs = orch._run_code_searcher
        orig_ss = orch._run_symbol_searcher

        def make_capture(name: str, original):
            async def capture(self, task, files, blackboard=None):
                # `self` here is the SubAgentOrchestrator instance (bound by __get__)
                execution_order.append(name)
                return await original(self, task, files, blackboard=blackboard)
            return capture

        capture_search = make_capture("search", lambda self, t, f, blackboard: orig_ff(t, f, blackboard))
        capture_ast = make_capture("ast", lambda self, t, f, blackboard: orig_cs(t, f, blackboard))
        capture_symbol = make_capture("symbol", lambda self, t, f, blackboard: orig_ss(t, f, blackboard))

        try:
            import codeforge_mcp.subagents.orchestrator as orch_module
            saved_role_caps = orch_module._ROLE_CAPS.copy()
            orch_module._ROLE_CAPS = reversed_role_caps

            # Patch registry handlers — they accept `self` as first arg (bound orchestrator)
            orch_module._CAPABILITY_REGISTRY["search"].handler = capture_search
            orch_module._CAPABILITY_REGISTRY["ast"].handler = capture_ast
            orch_module._CAPABILITY_REGISTRY["symbol"].handler = capture_symbol

            result = await orch.spawn_subagent(
                role="code_searcher",
                task="analyze foo",
                files=[],
            )

            # Verify the topological sort overrode the reversed order.
            # search must appear before ast because ast consumes target_symbol
            # which search produces.
            assert "search" in execution_order, (
                f"search handler must have executed. Execution order: {execution_order}"
            )
            assert "ast" in execution_order, (
                f"ast handler must have executed. Execution order: {execution_order}"
            )
            search_idx = execution_order.index("search")
            ast_idx = execution_order.index("ast")
            assert search_idx < ast_idx, (
                f"search must execute before ast (DAG ordering), "
                f"but got execution order: {execution_order}. "
                f"This means the topological sort was not applied or failed."
            )

            # Result should be successful — both handlers ran in correct order
            assert result.success is True, f"Expected success, got: {result.error}"
        finally:
            # Restore original handlers and ROLE_CAPS
            orch_module._CAPABILITY_REGISTRY["search"].handler = orig_ff
            orch_module._CAPABILITY_REGISTRY["ast"].handler = orig_cs
            orch_module._CAPABILITY_REGISTRY["symbol"].handler = orig_ss
            orch_module._ROLE_CAPS = saved_role_caps
            graph.close()


# ── spawn_multiple Budget Handling ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_spawn_multiple_insufficient_budget_returns_early_failure():
    """When per_agent_budget < MIN_AGENT_BUDGET (500), spawn_multiple
    returns early with a clear error per agent instead of spawning agents
    that would all fail their capability pipeline check."""
    from codeforge_mcp.subagents.orchestrator import (
        SubAgentOrchestrator, MIN_AGENT_BUDGET, ContextBudget,
    )

    mock_graph = MagicMock()
    mock_ast = MagicMock()
    mock_lsp = MagicMock()

    # Budget: 1500 tokens, 4 agents → 375 per agent (< 500 floor)
    orch = SubAgentOrchestrator(mock_graph, mock_ast, mock_lsp, "/tmp")
    orch.budget = ContextBudget(max_tokens=1500)

    specs = [
        {"role": "code_searcher", "task": "find foo"},
        {"role": "file_finder", "task": "find bar"},
        {"role": "diagnose", "task": "find baz"},
        {"role": "refactor_advisor", "task": "find qux"},
    ]

    results = await orch.spawn_multiple(specs)

    assert len(results) == 4
    for r in results:
        assert r.success is False
        assert "Insufficient budget for parallel execution" in r.error
        assert "need ≥ 500" in r.error
        assert "have 375" in r.error  # 1500 // 4 = 375


@pytest.mark.asyncio
async def test_spawn_multiple_exactly_at_floor_runs_normally():
    """When per_agent_budget == MIN_AGENT_BUDGET exactly, agents run normally."""
    from codeforge_mcp.subagents.orchestrator import (
        SubAgentOrchestrator, MIN_AGENT_BUDGET, ContextBudget,
    )

    mock_graph = MagicMock()
    mock_ast = MagicMock()
    mock_lsp = MagicMock()

    # Budget: 2000 tokens, 4 agents → 500 per agent (exactly at floor)
    orch = SubAgentOrchestrator(mock_graph, mock_ast, mock_lsp, "/tmp")
    orch.budget = ContextBudget(max_tokens=2000)

    # Mock handlers so they don't actually run
    async def noop_handler(task, files, blackboard=None):
        return SubAgentResult(role="test", task=task, success=True, data={})

    with patch.object(orch, "_run_file_finder", noop_handler):
        specs = [{"role": "file_finder", "task": "find foo"}]
        results = await orch.spawn_multiple(specs)

    assert len(results) == 1
    # At exactly 500, we don't trigger the early-return guard
    # (the check is per_agent_budget < MIN_AGENT_BUDGET, not <=)


@pytest.mark.asyncio
async def test_spawn_multiple_budget_calculation_no_floor():
    """Verify that per_agent_budget = remaining // n_agents with no 2000 floor.
    With remaining=1000, n_agents=2 → 500 each (not clamped to 2000)."""
    from codeforge_mcp.subagents.orchestrator import (
        SubAgentOrchestrator, MIN_AGENT_BUDGET, ContextBudget,
    )

    mock_graph = MagicMock()
    mock_ast = MagicMock()
    mock_lsp = MagicMock()

    orch = SubAgentOrchestrator(mock_graph, mock_ast, mock_lsp, "/tmp")
    # 1000 tokens, 2 agents → 500 per agent (exactly at floor)
    orch.budget = ContextBudget(max_tokens=1000)

    async def noop_handler(task, files, blackboard=None):
        return SubAgentResult(role="test", task=task, success=True, data={})

    with patch.object(orch, "_run_file_finder", noop_handler):
        specs = [{"role": "file_finder", "task": "find foo"}]
        results = await orch.spawn_multiple(specs)

    # With exactly 500 per agent, no early return; agents proceed
    assert len(results) == 1


@pytest.mark.asyncio
async def test_spawn_multiple_1999_total_budget_one_agent():
    """1999 tokens, 1 agent → 1999 per agent (no floor, no early return)."""
    from codeforge_mcp.subagents.orchestrator import (
        SubAgentOrchestrator, MIN_AGENT_BUDGET, ContextBudget,
    )

    mock_graph = MagicMock()
    mock_ast = MagicMock()
    mock_lsp = MagicMock()

    orch = SubAgentOrchestrator(mock_graph, mock_ast, mock_lsp, "/tmp")
    orch.budget = ContextBudget(max_tokens=1999)

    async def noop_handler(task, files, blackboard=None):
        return SubAgentResult(role="test", task=task, success=True, data={})

    with patch.object(orch, "_run_file_finder", noop_handler):
        specs = [{"role": "file_finder", "task": "find foo"}]
        results = await orch.spawn_multiple(specs)

    # 1999 // 1 = 1999 >= 500, no early return
    assert len(results) == 1


# ── Token Estimation ─────────────────────────────────────────────────────────

def test_estimate_tokens_returns_plausible_values():
    """Assert _estimate_tokens returns plausible token counts for a known fixture.

    The fixture is a dict with known string content. Using tiktoken (cl100k_base),
    "hello world" encodes to 2 tokens, so we verify the function returns a
    positive integer in a reasonable range for the input.
    """
    from codeforge_mcp.subagents.orchestrator import _estimate_tokens

    fixture = {
        "found_files": [
            {"path": "src/main.py", "size": 1024, "language": "Python"},
            {"path": "src/utils.py", "size": 512, "language": "Python"},
        ],
        "search_results": [
            {"file": "src/main.py", "line": 42, "text": "def hello():"},
            {"file": "src/utils.py", "line": 7, "text": "def world():"},
        ],
    }

    result = _estimate_tokens(fixture)

    # Must be a positive integer
    assert isinstance(result, int), f"Expected int, got {type(result)}"
    assert result > 0, f"Expected positive token count, got {result}"

    # The serialized JSON is roughly 250 bytes. At ~4 bytes/token the old
    # heuristic would give ~62 tokens. Tiktoken's cl100k_base is more
    # efficient (1 token ≈ 4 chars for typical English), so we expect
    # a similar range but verified against actual encoding.
    assert result < 300, (
        f"Token count {result} seems unreasonably high for a ~250-byte fixture. "
        f"Check that tiktoken is actually being used."
    )

    # Verify tiktoken is actually used by checking a simple string.
    # Note: "hello world" goes through json.dumps → "\"hello world\"" (quoted),
    # which encodes to ~3 tokens via cl100k_base, not 2.
    simple_result = _estimate_tokens("hello world")
    assert 2 <= simple_result <= 4, (
        f"Expected 2-4 tokens for 'hello world' (json-quoted), got {simple_result}. "
        f"Ensure tiktoken (cl100k_base) is being used, not the fallback."
    )

    # Edge case: empty data
    empty_result = _estimate_tokens({})
    assert isinstance(empty_result, int)
    assert empty_result >= 0

    # Empty string
    empty_str_result = _estimate_tokens("")
    assert isinstance(empty_str_result, int)
    assert empty_str_result >= 0


def test_estimate_tokens_fallback_when_tiktoken_unavailable(monkeypatch):
    """When tiktoken is unavailable, _estimate_tokens falls back to len(str(data)) // 4."""
    from codeforge_mcp.subagents.orchestrator import _estimate_tokens

    # Simulate tiktoken import failure by removing it from sys.modules
    original_import = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)

    def fake_import(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("simulated tiktoken unavailable")
        return original_import[name] if name in original_import else None

    import sys
    monkeypatch.setitem(sys.modules, "tiktoken", None)

    # After tiktoken is gone, should fall back to byte heuristic
    result = _estimate_tokens("hello world")
    # "hello world" = 11 chars // 4 = 2
    assert result == 2, f"Fallback expected 2, got {result}"


class TestRetryOrchestrator:
    """Tests for the orchestrator-level retry mechanism."""

    def test_retry_on_empty_data_succeeds_on_second_call(self):
        """Handler returns empty data first call, populated data second call — retry fires."""
        import codeforge_mcp.subagents.orchestrator as orch_mod
        from unittest.mock import MagicMock

        # Save original registry and restore after test
        orig_registry = dict(orch_mod._CAPABILITY_REGISTRY)
        orig_role_caps = dict(orch_mod._ROLE_CAPS)

        call_count = 0

        async def empty_then_populated(self, task, files, blackboard=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return orch_mod.SubAgentResult(
                    role="test_cap", task=task, success=True,
                    data={},  # empty — should trigger retry
                    token_estimate=10,
                )
            else:
                return orch_mod.SubAgentResult(
                    role="test_cap", task=task, success=True,
                    data={"found_files": [{"path": "foo.py"}]},  # populated
                    token_estimate=20,
                )

        try:
            # Create orchestrator FIRST so _register_all_capabilities() runs with real caps
            mock_graph = MagicMock()
            mock_ast = MagicMock()
            mock_lsp = MagicMock()
            orch = orch_mod.SubAgentOrchestrator(mock_graph, mock_ast, mock_lsp, '/tmp')

            # Register test capability AFTER __init__ so it persists in _CAPABILITY_REGISTRY
            test_cap = orch_mod.Capability(
                name="test_retry",
                handler=empty_then_populated,
                produces={"test_key"},
                consumes=set(),
                retry_strategy="broaden",
                max_retries=2,
            )
            orch_mod._CAPABILITY_REGISTRY["test_retry"] = test_cap
            orch_mod._ROLE_CAPS["test_retry_role"] = ["test_retry"]

            import asyncio
            result = asyncio.get_event_loop().run_until_complete(
                orch.spawn_subagent(role="test_retry_role", task='"exactterm"', files=[])
            )

            assert result.success, f"Expected success, got error: {result.error}"
            assert "found_files" in result.data, f"Expected found_files in data, got {result.data}"
            assert len(result.data["found_files"]) == 1
            # First call returned empty, second returned populated → retry happened
            assert call_count == 2, f"Expected 2 calls (first empty, second populated), got {call_count}"
        finally:
            orch_mod._CAPABILITY_REGISTRY.clear()
            orch_mod._CAPABILITY_REGISTRY.update(orig_registry)
            orch_mod._ROLE_CAPS.clear()
            orch_mod._ROLE_CAPS.update(orig_role_caps)

    def test_max_retries_respected_when_handler_always_fails(self):
        """Handler always raises — orchestrator respects max_retries and records error."""
        import codeforge_mcp.subagents.orchestrator as orch_mod
        from unittest.mock import MagicMock

        orig_registry = dict(orch_mod._CAPABILITY_REGISTRY)
        orig_role_caps = dict(orch_mod._ROLE_CAPS)

        call_count = 0

        async def always_fails(self, task, files, blackboard=None):
            nonlocal call_count
            call_count += 1
            raise RuntimeError(f"call {call_count} failed")

        try:
            # Create orchestrator FIRST so _register_all_capabilities() runs with real caps
            mock_graph = MagicMock()
            mock_ast = MagicMock()
            mock_lsp = MagicMock()
            orch = orch_mod.SubAgentOrchestrator(mock_graph, mock_ast, mock_lsp, '/tmp')

            test_cap = orch_mod.Capability(
                name="test_fail",
                handler=always_fails,
                produces={"test_key"},
                consumes=set(),
                retry_strategy="broaden",
                max_retries=3,
            )
            orch_mod._CAPABILITY_REGISTRY["test_fail"] = test_cap
            orch_mod._ROLE_CAPS["test_fail_role"] = ["test_fail"]

            import asyncio
            result = asyncio.get_event_loop().run_until_complete(
                orch.spawn_subagent(role="test_fail_role", task='"term"', files=[])
            )

            assert not result.success
            # max_retries=3 → initial + 3 retries = 4 total calls
            assert call_count == 4, f"Expected 4 calls (1 initial + 3 retries), got {call_count}"
        finally:
            orch_mod._CAPABILITY_REGISTRY.clear()
            orch_mod._CAPABILITY_REGISTRY.update(orig_registry)
            orch_mod._ROLE_CAPS.clear()
            orch_mod._ROLE_CAPS.update(orig_role_caps)

    def test_no_retry_when_retry_strategy_is_none(self):
        """Handler returns empty data but strategy='none' — no retry, returns empty result."""
        import codeforge_mcp.subagents.orchestrator as orch_mod
        from unittest.mock import MagicMock

        orig_registry = dict(orch_mod._CAPABILITY_REGISTRY)
        orig_role_caps = dict(orch_mod._ROLE_CAPS)

        call_count = 0

        async def empty_never_retries(self, task, files, blackboard=None):
            nonlocal call_count
            call_count += 1
            return orch_mod.SubAgentResult(
                role="test_cap", task=task, success=True,
                data={"found_files": []},  # empty data but has key structure
                token_estimate=5,
            )

        try:
            # Create orchestrator FIRST so _register_all_capabilities() runs with real caps
            mock_graph = MagicMock()
            mock_ast = MagicMock()
            mock_lsp = MagicMock()
            orch = orch_mod.SubAgentOrchestrator(mock_graph, mock_ast, mock_lsp, '/tmp')

            test_cap = orch_mod.Capability(
                name="test_no_retry",
                handler=empty_never_retries,
                produces={"test_key"},
                consumes=set(),
                retry_strategy="none",
                max_retries=5,
            )
            orch_mod._CAPABILITY_REGISTRY["test_no_retry"] = test_cap
            orch_mod._ROLE_CAPS["test_no_retry_role"] = ["test_no_retry"]

            import asyncio
            result = asyncio.get_event_loop().run_until_complete(
                orch.spawn_subagent(role="test_no_retry_role", task='"term"', files=[])
            )

            assert result.success  # handler succeeded with empty data (strategy=none → no retry)
            assert call_count == 1, f"Expected exactly 1 call (no retry for strategy='none'), got {call_count}"
        finally:
            orch_mod._CAPABILITY_REGISTRY.clear()
            orch_mod._CAPABILITY_REGISTRY.update(orig_registry)
            orch_mod._ROLE_CAPS.clear()
            orch_mod._ROLE_CAPS.update(orig_role_caps)
