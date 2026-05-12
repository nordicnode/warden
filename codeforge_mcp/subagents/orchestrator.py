"""Subagent orchestrator — spawn_subagent with role-based hardcoded handlers.

The MCP server itself does NOT call any external LLM. Reasoning is delegated
to the calling MCP client's model. This module provides deterministic,
fact-returning hardcoded handlers that the client's LLM can compose.

Supports:
- Hardcoded fast handlers for predefined roles (file_finder, code_searcher,
  reviewer, test_impact)
- Parallel subagent execution (spawn_multiple / spawn_subagents)
- Task decomposition (decompose → spawn multiple automatically)
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Awaitable


@dataclass
class SubAgentResult:
    role: str
    task: str
    success: bool
    data: Any = None
    error: str = ""
    token_estimate: int = 0


class Blackboard:
    """Shared semantic state for a group of subagents (GAP-1).

    Allows subagents to 'post' findings (files, symbols, facts) that
    subsequent or parallel subagents can leverage.
    """

    def __init__(self) -> None:
        self.files: set[str] = set()
        self.symbols: set[str] = set()
        self.facts: dict[str, Any] = {}
        self.insights: list[str] = []
        self._events: dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()

    async def add_files(self, paths: list[str]) -> None:
        async with self._lock:
            self.files.update(paths)

    async def add_symbols(self, names: list[str]) -> None:
        async with self._lock:
            self.symbols.update(names)

    async def add_insight(self, text: str) -> None:
        async with self._lock:
            self.insights.append(text)

    async def add_fact(self, key: str, value: Any) -> None:
        async with self._lock:
            self.facts[key] = value

    async def append_fact(self, key: str, value: Any) -> None:
        async with self._lock:
            existing = self.facts.get(key)
            if isinstance(existing, list):
                existing.append(value)
            elif existing is None:
                self.facts[key] = [value]
            else:
                self.facts[key] = [existing, value]

    async def mark(self, event_name: str) -> None:
        async with self._lock:
            event = self._events.setdefault(event_name, asyncio.Event())
            event.set()

    async def wait_for(self, event_name: str, timeout: float = 1.0) -> bool:
        async with self._lock:
            event = self._events.setdefault(event_name, asyncio.Event())
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def summary(self) -> dict[str, Any]:
        return {
            "files": list(self.files),
            "symbols": list(self.symbols),
            "insights": self.insights,
            "facts": self.facts,
        }


class ContextBudget:
    """Tracks cumulative token usage across subagent calls (GAP-5)."""

    def __init__(self, max_tokens: int = 16_000) -> None:
        self.max_tokens = max_tokens
        self.used_tokens = 0
        self.call_count = 0
        self.tool_call_count = 0
        self.subagent_call_count = 0

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)

    @property
    def utilization(self) -> float:
        return self.used_tokens / self.max_tokens if self.max_tokens else 0.0

    def consume(self, tokens: int, *, calls: int = 1, source: str = "subagent") -> None:
        if self.would_exceed(tokens):
            # We still record the attempt but the caller should have checked would_exceed
            pass
        self.used_tokens += max(0, tokens)
        self.call_count += max(0, calls)
        if source == "tool":
            self.tool_call_count += max(0, calls)
        else:
            self.subagent_call_count += max(0, calls)

    def would_exceed(self, tokens: int) -> bool:
        """Check if adding 'tokens' would exceed the max budget."""
        return self.used_tokens + tokens > self.max_tokens

    def summary(self) -> dict[str, Any]:
        return {
            "used_tokens": self.used_tokens,
            "max_tokens": self.max_tokens,
            "remaining": self.remaining,
            "utilization_pct": round(self.utilization * 100, 1),
            "call_count": self.call_count,
            "tool_call_count": self.tool_call_count,
            "subagent_call_count": self.subagent_call_count,
            "budget_warning": self.utilization > 0.8,
        }


class SubAgentOrchestrator:
    """Manages spawning and merging subagent results.

    All handlers are hardcoded (no LLM). The orchestrator returns raw,
    fact-shaped data so the calling MCP client's model can reason over it.
    """

    def __init__(
        self,
        graph: Any,
        ast_indexer: Any,
        lsp_multiplexer: Any,
        project_root: str,
        context_budget: ContextBudget | None = None,
    ) -> None:
        self.graph = graph
        self.ast_indexer = ast_indexer
        self.lsp = lsp_multiplexer
        self.project_root = project_root
        self.budget = context_budget or ContextBudget()

    def _normalize_file_path(self, path: str | Path) -> str:
        """Return a stable, project-relative path when possible."""
        p = Path(path)
        if not p.is_absolute():
            p = (Path(self.project_root).resolve() / p).resolve()
        try:
            return str(p.relative_to(Path(self.project_root).resolve()))
        except ValueError:
            return str(p)

    def _symbol_candidates_for_file(self, file_path: str) -> list[dict[str, Any]]:
        """Look up symbols for a file using both relative and absolute forms."""
        normalized = self._normalize_file_path(file_path)
        abs_path = str((Path(self.project_root).resolve() / normalized).resolve())
        symbols = self.graph.symbols_in_file(abs_path)
        if not symbols and normalized != abs_path:
            symbols = self.graph.symbols_in_file(normalized)
        return symbols

    def _read_changed_excerpt(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        padding: int = 2,
    ) -> str:
        """Read a compact window around the changed lines for reviewer heuristics."""
        normalized = self._normalize_file_path(file_path)
        abs_path = (Path(self.project_root).resolve() / normalized).resolve()
        if not abs_path.is_file():
            return ""
        lines = abs_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if not lines:
            return ""
        start = max(1, start_line or 1)
        end = max(start, end_line or start)
        window_start = max(1, start - padding)
        window_end = min(len(lines), end + padding)
        excerpt: list[str] = []
        for idx in range(window_start, window_end + 1):
            excerpt.append(f"{idx}: {lines[idx - 1]}")
        return "\n".join(excerpt)

    def _build_review_context(
        self,
        path: str,
        start_line: int = 0,
        end_line: int = 0,
        diff_preview: str = "",
        max_files: int = 12,
    ) -> dict[str, Any]:
        """Build a blast-radius-aware context bundle for post-edit review."""
        normalized = self._normalize_file_path(path)
        symbols = self._symbol_candidates_for_file(normalized)
        changed_symbols = [
            sym for sym in symbols
            if start_line <= 0 or end_line <= 0 or (start_line <= int(sym.get("line", 0)) <= end_line)
        ]
        if not changed_symbols:
            changed_symbols = symbols[:8]

        review_files: list[str] = [normalized]
        impacted_symbols: list[dict[str, Any]] = []
        tests: list[str] = []
        seen_files = {normalized}
        root = Path(self.project_root).resolve()

        from codeforge_mcp.tools.understanding import impact_analysis

        for sym in changed_symbols[:8]:
            impact = impact_analysis(self.graph, self.ast_indexer, sym["name"], sym["file"])
            impacted_symbols.append({
                "name": sym["name"],
                "file": self._normalize_file_path(sym.get("file", normalized)),
                "line": sym.get("line", 0),
                "impact": impact,
            })
            for module in impact.get("modules", []):
                candidate = self._normalize_file_path(module)
                if candidate not in seen_files and len(review_files) < max_files:
                    seen_files.add(candidate)
                    review_files.append(candidate)
            for test in impact.get("tests", []):
                candidate = self._normalize_file_path(test.get("file", ""))
                if candidate and candidate not in tests:
                    tests.append(candidate)
                if candidate and candidate not in seen_files and len(review_files) < max_files:
                    seen_files.add(candidate)
                    review_files.append(candidate)

        changed_end = end_line
        if changed_end <= 0:
            abs_path = (root / normalized).resolve()
            if abs_path.is_file():
                changed_end = len(abs_path.read_text(encoding="utf-8", errors="replace").splitlines())

        return {
            "changed_files": [normalized],
            "changed_ranges": [{
                "file": normalized,
                "start_line": max(1, start_line or 1),
                "end_line": max(1, changed_end or max(1, start_line or 1)),
            }],
            "changed_excerpt": self._read_changed_excerpt(
                normalized,
                max(1, start_line or 1),
                max(1, changed_end or max(1, start_line or 1)),
            ),
            "diff_preview": diff_preview,
            "impacted_symbols": impacted_symbols,
            "impacted_symbol_names": [sym["name"] for sym in impacted_symbols],
            "review_files": review_files,
            "test_files": tests[:6],
        }

    async def review_change(
        self,
        path: str,
        start_line: int = 0,
        end_line: int = 0,
        diff_preview: str = "",
    ) -> SubAgentResult:
        """Run the reviewer with blast-radius context for a recent edit."""
        context = self._build_review_context(
            path=path,
            start_line=start_line,
            end_line=end_line,
            diff_preview=diff_preview,
        )
        bb = Blackboard()
        await bb.add_files(context["review_files"])
        await bb.add_symbols(context["impacted_symbol_names"])
        await bb.add_fact("review_context", context)
        await bb.add_insight(
            f"Post-edit review seeded with {len(context['review_files'])} files and "
            f"{len(context['impacted_symbol_names'])} impacted symbols."
        )

        task = (
            f"Review the recent edit to '{context['changed_files'][0]}'. "
            f"Changed lines {context['changed_ranges'][0]['start_line']}-"
            f"{context['changed_ranges'][0]['end_line']}. "
            f"Check the changed file, affected modules, and nearby tests for "
            f"blocking diagnostics or suspicious edit artifacts."
        )
        return await self.spawn_subagent(
            role="reviewer",
            task=task,
            files=context["review_files"],
            blackboard=bb,
        )

    def _extract_search_terms(self, task: str, limit: int = 5) -> list[str]:
        """Extract the most relevant search terms from a natural-language task.

        Quoted/backticked identifiers take precedence, which prevents generic
        directive words like "find" or "occurrences" from polluting search
        queries.
        """
        quoted = [
            match.strip()
            for match in re.findall(r"[\"'`](.+?)[\"'`]", task)
            if match.strip()
        ]
        if quoted:
            seen: set[str] = set()
            out: list[str] = []
            for term in quoted:
                if term not in seen:
                    seen.add(term)
                    out.append(term)
                if len(out) >= limit:
                    break
            return out

        stop_words = {
            "the", "find", "file", "files", "that", "handles", "and", "list",
            "its", "main", "class", "where", "is", "a", "of", "to", "in",
            "all", "occurrence", "occurrences", "codebase", "workspace",
            "show", "search", "for", "with", "from",
        }
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_./:-]*", task)
        seen = set()
        out = []
        for token in tokens:
            lowered = token.lower()
            if lowered in stop_words or len(token) < 3:
                continue
            if token not in seen:
                seen.add(token)
                out.append(token)
            if len(out) >= limit:
                break
        return out

    def _extract_definition_targets(self, task: str, limit: int = 3) -> list[str]:
        """Extract likely symbol names when the task asks for a definition/implementation."""
        quoted = self._extract_search_terms(task, limit=limit)
        if quoted:
            return quoted[:limit]

        candidates = re.findall(r"\b[A-Z][A-Za-z0-9_]+\b|\b[a-z_][A-Za-z0-9_]*\b", task)
        stop = {
            "find", "implementation", "implementations", "definition", "definitions",
            "class", "function", "method", "where", "file", "files", "the", "of", "for",
        }
        out: list[str] = []
        for cand in candidates:
            if cand.lower() in stop or len(cand) < 3:
                continue
            if cand not in out:
                out.append(cand)
            if len(out) >= limit:
                break
        return out

    # ── Public API ────────────────────────────────────────────────────────

    # ── Output size limits (Fix 7: Truncation) ──────────────────────
    _MAX_FILES_IN_RESULT = 20
    _MAX_SEARCH_RESULTS = 50
    _MAX_DIAGNOSTICS = 20
    _MAX_IMPACT_RESULTS = 10
    _MAX_CHECKLIST_ITEMS = 30
    _MAX_FILE_SUMMARIES = 15
    _MAX_FINDINGS = 25
    _MAX_BLOCKING_ISSUES = 20

    def _trim_result_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Trim large result dicts to prevent interleaved/truncated output."""
        if not isinstance(data, dict):
            return data
        limits = {
            "found_files": self._MAX_FILES_IN_RESULT,
            "search_results": self._MAX_SEARCH_RESULTS,
            "diagnostics": self._MAX_DIAGNOSTICS,
            "impact": self._MAX_IMPACT_RESULTS,
            "checklist": self._MAX_CHECKLIST_ITEMS,
            "file_summaries": self._MAX_FILE_SUMMARIES,
            "findings": self._MAX_FINDINGS,
            "blocking_issues": self._MAX_BLOCKING_ISSUES,
        }
        for key, limit in limits.items():
            if key in data and isinstance(data[key], list) and len(data[key]) > limit:
                data[f"{key}_truncated"] = True
                data[f"{key}_total"] = len(data[key])
                data[key] = data[key][:limit]
        return data

    async def spawn_subagent(
        self,
        role: str = "",
        task: str = "",
        files: list[str] | None = None,
        capabilities: list[str] | None = None,
        blackboard: Blackboard | None = None,
        budget_override: ContextBudget | None = None,
    ) -> SubAgentResult:
        """Spawn a subagent by role or dynamic capabilities.

        Args:
            role: Preset role (convenience alias for capabilities).
            task: Natural-language task for the subagent.
            files: Optional list of file paths for context scope.
            capabilities: Explicit list of capability strings (dynamic mode).
            blackboard: Shared state session (optional).
            budget_override: Isolated budget for this subagent (optional).
                When provided, all budget checks inside this call use this
                budget instead of ``self.budget``.  Used by ``spawn_multiple``
                to give each parallel subagent its own independent budget
                slice without race conditions on ``self.budget``.

        Returns:
            SubAgentResult with merged data.
        """
        budget = budget_override or self.budget
        if files is None:
            files = []
        if capabilities is None:
            capabilities = []
        
        bb = blackboard or Blackboard()

        # ── Role-to-Capability Mapping ──────────────────────────────────
        # Roles are now aliases for capability pipelines. Each capability
        # maps to a specific hardcoded handler that returns facts.
        ROLE_CAPS: dict[str, list[str]] = {
            "file_finder": ["search"],
            "code_searcher": ["search", "ast", "symbol"],
            "reviewer": ["lsp", "graph"],
            "test_impact": ["graph_upstream"],
            "diagnose": ["diagnose"],
            "refactor_advisor": ["graph", "ast"],
            "security_auditor": ["ast", "search"],
            "doc_generator": ["ast", "lsp"],
        }
        # Common aliases — promote intuitive role names to the canonical set.
        ROLE_ALIASES: dict[str, str] = {
            "researcher": "code_searcher",
            "investigator": "code_searcher",
            "search": "code_searcher",
            "auditor": "security_auditor",
            "security": "security_auditor",
            "review": "reviewer",
            "doc": "doc_generator",
            "docs": "doc_generator",
            "refactor": "refactor_advisor",
            "tests": "test_impact",
            "diag": "diagnose",
        }
        if role in ROLE_ALIASES:
            role = ROLE_ALIASES[role]

        if role == "decompose":
            # decompose is a special role that returns a planning prompt
            # instead of executing a pipeline of handlers.
            if budget.would_exceed(100): # Minimal estimate for decompose
                return SubAgentResult(
                    role="decompose", task=task, success=False,
                    error=f"Context budget exceeded ({budget.used_tokens}/{budget.max_tokens})",
                )
            return await self._run_decompose(task, files)

        if role and not capabilities:
            capabilities = ROLE_CAPS.get(role, [])
            if not capabilities:
                 return SubAgentResult(
                    role=role, task=task, success=False,
                    error=f"Unknown role: {role}. Available roles: {', '.join(ROLE_CAPS.keys())}",
                )

        if not capabilities and not role:
             return SubAgentResult(
                role="error", task=task, success=False,
                error="Either 'role' or 'capabilities' must be provided.",
            )

        # ── Execute Capability Pipeline ──────────────────────────────────
        # We execute each capability in sequence, merging their results
        # into a single shared results_data dictionary.
        results_data: dict[str, Any] = {}
        any_success = False
        total_tokens = 0
        errors = []

        # Available primitive handlers (hardcoded fact-returning logic)
        cap_handlers: dict[str, Callable[..., Awaitable[SubAgentResult]]] = {
            "search": self._run_file_finder,
            "ast": self._run_code_searcher,
            "symbol": self._run_symbol_searcher,
            "lsp": self._run_reviewer, # aliased for now
            "graph": self._run_reviewer, # aliased for now
            "graph_upstream": self._run_test_impact,
            "diagnose": self._run_diagnose,
        }

        # Deduplicate capabilities that resolve to the same handler, preserving order.
        # This prevents _run_reviewer from being called twice when capabilities
        # ["lsp", "graph"] are both present (both map to the same handler).
        # Use handler.__name__ instead of id(handler) — Python bound-method
        # ids are not stable across accesses (CPython creates a new bound
        # method object each time handler is retrieved from the dict).
        seen_caps: set[str] = set()
        seen_handlers: set[str] = set()
        unique_caps: list[str] = []
        for c in capabilities:
            handler = cap_handlers.get(c)
            if handler is None:
                continue
            handler_key = handler.__name__
            if c not in seen_caps and handler_key not in seen_handlers:
                seen_caps.add(c)
                seen_handlers.add(handler_key)
                unique_caps.append(c)

        for cap in unique_caps:
            # Check budget before each capability to prevent runaway usage
            if budget.would_exceed(500): # Conservative estimate per capability
                 errors.append(f"{cap}: Budget exceeded")
                 continue

            handler = cap_handlers.get(cap)
            if handler is None:
                errors.append(f"Unknown capability: {cap}")
                continue
            
            try:
                # Handlers update the blackboard and return a result object
                res = await handler(task, files, blackboard=bb)
                total_tokens += res.token_estimate
                if res.success:
                    any_success = True
                    # Merge result data: append to lists, overwrite other types
                    if isinstance(res.data, dict):
                        for k, v in res.data.items():
                            if k in results_data and isinstance(v, list) and isinstance(results_data[k], list):
                                results_data[k].extend(v)
                            else:
                                results_data[k] = v
                else:
                    errors.append(f"{cap}: {res.error}")
            except Exception as e:
                errors.append(f"{cap}: {str(e)}")

        # Track cumulative budget usage across all subagent calls.
        # When a budget_override is supplied we consume against THAT budget
        # so callers (spawn_multiple) can isolate parallel agents.  When no
        # override is given, consume against the shared budget as usual.
        budget.consume(total_tokens)
        
        # Final combined result includes the final blackboard state
        results_data["_blackboard"] = bb.summary()
        results_data["capabilities_used"] = unique_caps
        results_data = self._trim_result_data(results_data)

        # A subagent is considered successful if at least one capability
        # produced useful data — partial results are better than nothing.
        # Only report failure when ALL capabilities failed.
        return SubAgentResult(
            role=role or "dynamic",
            task=task,
            success=any_success,
            data=results_data,
            error="; ".join(errors),
            token_estimate=total_tokens,
        )

    async def spawn_multiple(
        self,
        specs: list[dict[str, Any]],
        swarm: bool = False,
    ) -> list[SubAgentResult]:
        """Run multiple subagents in parallel and return all results.

        Args:
            specs: List of {role, task, files?} dicts.

        Returns:
            List of SubAgentResult in the same order.
        """
        bb = Blackboard()
        if swarm:
            await bb.add_fact("coordination_mode", "swarm")

        # ── Budget isolation: allocate per-agent sub-budgets ───────────
        # When subagents run in parallel (asyncio.gather), they all compete
        # for the shared self.budget pool, causing premature "Budget
        # exceeded" errors.  We split the remaining budget evenly among
        # the agents and pass each one its own isolated budget via
        # budget_override so that concurrent coroutine execution doesn't
        # race on self.budget.
        n_agents = len(specs)
        remaining = self.budget.remaining
        base = max(2000, remaining // max(n_agents, 1))
        if base * n_agents > remaining:
            per_agent_budget = remaining // max(n_agents, 1)
        else:
            per_agent_budget = base

        tasks = []
        for spec in specs:
            # Each parallel subagent gets its own isolated budget slice —
            # passed via budget_override to avoid the race condition that
            # would occur if we mutated self.budget before coroutine
            # execution (all concurrent coroutines would then share the
            # last-written budget object).
            agent_budget = ContextBudget(max_tokens=per_agent_budget)
            tasks.append(
                self.spawn_subagent(
                    role=spec["role"],
                    task=spec["task"],
                    files=spec.get("files", []),
                    blackboard=bb,
                    budget_override=agent_budget,
                )
            )
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge actual token usage back into the shared budget
        total_tokens = 0
        for result in raw_results:
            if isinstance(result, SubAgentResult):
                total_tokens += result.token_estimate
        self.budget.consume(total_tokens)

        # Convert any unhandled exceptions to SubAgentResult errors
        safe_results: list[SubAgentResult] = []
        for i, result in enumerate(raw_results):
            if isinstance(result, BaseException):
                spec = specs[i]
                safe_results.append(SubAgentResult(
                    role=spec["role"],
                    task=spec["task"],
                    success=False,
                    error=f"Unhandled exception: {type(result).__name__}: {result}",
                ))
            elif isinstance(result, SubAgentResult):
                # Inject blackboard summary into the result data for context-aware merging
                if result.data is None:
                    result.data = {}
                if isinstance(result.data, dict):
                    result.data["_blackboard"] = bb.summary()
                safe_results.append(result)
            else:
                # Should never happen, but guard against unexpected types
                spec = specs[i]
                safe_results.append(SubAgentResult(
                    role=spec["role"],
                    task=spec["task"],
                    success=False,
                    error=f"Unexpected result type: {type(result).__name__}",
                ))
        
        # Merge all results into a session summary
        merged_insight = self._merge_results(safe_results, bb)
        for r in safe_results:
            if isinstance(r.data, dict):
                r.data["session_summary"] = merged_insight
                r.data = self._trim_result_data(r.data)

        return safe_results

    # ── Decompose ─────────────────────────────────────────────────────────

    async def _run_decompose(self, task: str, files: list[str]) -> SubAgentResult:
        """Decompose a complex task into sub-tasks by providing a planning prompt.

        Instead of hardcoded regex, this returns a schema that the client LLM
        can use to formulate its own plan using spawn_subagents.
        """
        planning_prompt = f"""
You are a planning agent. Decompose the following task into a series of subagent calls.
Task: {task}
Available Files: {files}

Capabilities:
- search: Find files and search code for patterns.
- ast: Extract symbol definitions and structure from files.
- symbol: Look up specific symbol definitions (cross-file).
- lsp: Run diagnostics and impact analysis.
- graph: Analyze dependencies and call graphs.
- graph_upstream: Identify test impact and upstream callers.

Preset Roles (Aliased to Capabilities):
- file_finder: ['search']
- code_searcher: ['ast', 'symbol']
- reviewer: ['lsp', 'graph']
- test_impact: ['graph_upstream']
- diagnose: ['search', 'lsp', 'graph']

Return a list of JSON objects for 'spawn_subagents', each containing:
{{ "role": "...", "task": "...", "files": [...], "capabilities": [...] }}
"""
        data = {
            "planning_prompt": planning_prompt,
            "available_capabilities": ["search", "ast", "symbol", "lsp", "graph", "graph_upstream"],
            "suggested_roles": ["file_finder", "code_searcher", "reviewer", "test_impact", "diagnose"]
        }

        return SubAgentResult(
            role="decompose", task=task, success=True,
            data=data, token_estimate=len(planning_prompt) // 4,
        )

    def _merge_results(self, results: list[SubAgentResult], blackboard: Blackboard) -> dict[str, Any]:
        """Synthesize overlapping insights from multiple subagents (GAP-4)."""
        summary = blackboard.summary()
        
        # Cross-reference: find files mentioned by multiple capabilities
        file_frequency: dict[str, int] = {}
        for res in results:
            if isinstance(res.data, dict) and "found_files" in res.data:
                for f in res.data["found_files"]:
                    path = f["path"] if isinstance(f, dict) else f
                    file_frequency[path] = file_frequency.get(path, 0) + 1
            if isinstance(res.data, dict) and "diagnostics" in res.data:
                for d in res.data["diagnostics"]:
                    path = d["file"]
                    file_frequency[path] = file_frequency.get(path, 0) + 1

        hot_files = [f for f, count in file_frequency.items() if count > 1]
        
        merged = {
            "blackboard": summary,
            "hot_files": hot_files,
            "subagent_count": len(results),
            "total_tokens": sum(r.token_estimate for r in results),
            "coordination_mode": summary.get("facts", {}).get("coordination_mode", "parallel"),
        }
        
        return merged

    # ── Hardcoded handlers (deterministic, no LLM) ────────────────────────

    async def _run_file_finder(self, task: str, files: list[str], blackboard: Blackboard) -> SubAgentResult:
        from codeforge_mcp.tools.navigation import code_find_files, code_search, _language_from_ext

        raw_terms = [t.strip() for t in self._extract_search_terms(task, limit=5) if t.strip()]
        definition_targets = self._extract_definition_targets(task)
        wants_exact_definition = any(
            word in task.lower()
            for word in ("implementation", "definition", "defined", "class", "function", "method")
        )

        # ── Phase 5 fix: respect explicit file directives ─────────────
        # When the caller provides explicit files, scope ALL searches to
        # those files first before falling back to codebase-wide queries.
        # This prevents the diagnose subagent from ignoring file path
        # arguments and returning irrelevant results.
        has_explicit_files = bool(files)

        # Stemming: basic removal of common suffixes to increase recall
        terms: list[str] = []
        seen_terms: set[str] = set()

        def _add_term(term: str) -> None:
            if term and term not in seen_terms:
                seen_terms.add(term)
                terms.append(term)

        for t in raw_terms:
            _add_term(t)
            lowered = t.lower()
            _add_term(lowered)
            if lowered.endswith("ing") and len(lowered) > 6:
                _add_term(lowered[:-3])
            elif lowered.endswith("es") and len(lowered) > 5:
                _add_term(lowered[:-2])
            elif lowered.endswith("s") and len(lowered) > 4:
                _add_term(lowered[:-1])

        # If all terms were filtered, fallback to original logic
        if not terms:
            terms = [t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_./:-]*", task) if len(t) >= 3]

        results: dict[str, list[Any]] = {
            "found_files": [],
            "search_results": [],
            "exact_definitions": [],
        }

        if definition_targets:
            for target in definition_targets:
                try:
                    symbol_result = await self._symbol_lookup_async(target)
                except Exception:
                    symbol_result = None
                if not symbol_result:
                    continue
                sym_file = str(symbol_result.get("file", ""))
                if sym_file:
                    results["found_files"].append({
                        "path": sym_file,
                        "size": 0,
                        "language": _language_from_ext(Path(sym_file).suffix),
                    })
                    definition_hit = {
                        "name": symbol_result.get("name", target),
                        "file": sym_file,
                        "line": symbol_result.get("line", 0),
                        "kind": symbol_result.get("kind", ""),
                    }
                    results["exact_definitions"].append(definition_hit)
                    results["search_results"].append({
                        "file": sym_file,
                        "line": symbol_result.get("line", 0),
                        "text": f"exact definition: {symbol_result.get('name', target)}",
                        "is_match": True,
                        "match_type": "definition",
                        "score": 20,
                    })
                    await blackboard.append_fact("primary_definitions", definition_hit)
                await blackboard.add_symbols([symbol_result.get("name", target)])
                await blackboard.mark("symbols_seeded")

        # First pass: meaningful terms
        # When explicit files are provided, scope searches to them first.
        # Only add files that match at least one search term so diagnose
        # subagents don't return all provided files as "found".
        if has_explicit_files:
            # Normalize paths: strip leading ./ so "tests/foo.py" matches
            # both "tests/foo.py" and "./tests/foo.py" from rg output.
            explicit_set = {f.lstrip("./") for f in files}
            # Do text search scoped to those files
            for term in terms[:3]:
                search = code_search(self.project_root, term, context=2)
                # Filter search results to only include explicit files
                matched = [r for r in search if r.get("file", "").lstrip("./") in explicit_set]
                results["search_results"].extend(matched)
            # Only add files that appeared in search results to found_files
            matched_files = {r["file"] for r in results["search_results"]}
            for f in matched_files:
                abs_path = str((Path(self.project_root).resolve() / f).resolve())
                if not Path(abs_path).is_file():
                    continue
                results["found_files"].append({
                    "path": f,
                    "size": Path(abs_path).stat().st_size,
                    "language": _language_from_ext(Path(f).suffix),
                })
        else:
            for term in terms[:5]:
                found = code_find_files(self.project_root, term)
                results["found_files"].extend(found)
                search = code_search(self.project_root, term, context=2)
                results["search_results"].extend(search)

        # Feedback loop: if no results, try broader terms (substrings)
        # Skip this fallback when explicit files were provided — the caller
        # is being specific about scope; codebase-wide fallback defeats that.
        if not has_explicit_files and not results["found_files"] and not results["search_results"]:
            await blackboard.add_insight(f"Search for '{terms}' yielded no results. Retrying with broader substring match.")
            for term in terms[:3]:
                 # Substring search fallback
                 search = code_search(self.project_root, term, context=1, regex=True)
                 results["search_results"].extend(search)

        # Symbol-aware fallback: if text search still fails, resolve the
        # likely identifier and surface its defining file directly.
        # Skip when explicit files were provided (same reasoning as above).
        if not has_explicit_files and not results["found_files"] and not results["search_results"]:
            for term in raw_terms[:3]:
                try:
                    symbol_result = await self._symbol_lookup_async(term)
                except Exception:
                    symbol_result = None
                if not symbol_result:
                    continue
                sym_file = str(symbol_result.get("file", ""))
                if sym_file:
                    results["found_files"].append({
                        "path": sym_file,
                        "size": 0,
                        "language": _language_from_ext(Path(sym_file).suffix),
                    })
                    results["search_results"].append({
                        "file": sym_file,
                        "line": symbol_result.get("line", 0),
                        "text": f"symbol lookup: {symbol_result.get('name', term)}",
                        "is_match": True,
                        "score": 10,
                    })
                await blackboard.add_symbols([symbol_result.get("name", term)])
                await blackboard.add_insight(
                    f"Resolved '{term}' via symbol lookup after text search returned no matches."
                )

        # Defensive filter: drop binary / cache artefacts that may have
        # slipped past the search tools (Phase 2 fix — keeps the blackboard
        # focused on real source files).
        from codeforge_mcp.tools.navigation import _is_skipped_file, _in_skipped_dir

        def _is_source(path_str: str) -> bool:
            p = Path(path_str)
            return not (_is_skipped_file(p) or _in_skipped_dir(p))

        seen_paths: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for f in results["found_files"]:
            if f["path"] in seen_paths:
                continue
            if not _is_source(f["path"]):
                continue
            seen_paths.add(f["path"])
            deduped.append(f)
        results["found_files"] = deduped[:50]

        # Post found files to blackboard
        await blackboard.add_files([f["path"] for f in results["found_files"]])
        if results["found_files"]:
            await blackboard.mark("files_seeded")

        seen: set[str] = set()
        deduped_search: list[dict[str, Any]] = []
        for s in results["search_results"]:
            if not _is_source(s["file"]):
                continue
            key = f"{s['file']}:{s['line']}"
            if key not in seen:
                seen.add(key)
                deduped_search.append(s)
        results["search_results"] = deduped_search[:50]
        if wants_exact_definition and results["exact_definitions"]:
            results["search_results"].sort(
                key=lambda r: (r.get("match_type") != "definition", -int(r.get("score", 0)))
            )
            primary = results["exact_definitions"][0]
            await blackboard.add_fact("target_symbol", primary["name"])
            await blackboard.add_fact("target_file", primary["file"])
        if results["search_results"]:
            await blackboard.mark("search_seeded")

        return SubAgentResult(
            role="file_finder", task=task, success=True,
            data=results, token_estimate=len(str(results)) // 4,
        )

    async def _run_code_searcher(self, task: str, files: list[str], blackboard: Blackboard) -> SubAgentResult:
        from codeforge_mcp.tools.understanding import ast_query

        # Use files from blackboard if none provided — process up to 20 files
        # (previously limited to 3, which was too restrictive for "find all
        # usages" tasks)
        target_files = files or list(blackboard.files)[:20]
        if not target_files and blackboard.facts.get("coordination_mode") == "swarm":
            await blackboard.wait_for("files_seeded")
            target_files = files or list(blackboard.files)[:20]
        ast_results: list[dict[str, Any]] = []
        exact_definitions: list[dict[str, Any]] = []

        target_symbol = blackboard.facts.get("target_symbol")
        target_file = blackboard.facts.get("target_file")

        # Run AST queries on target files
        if target_files:
            for f in target_files:
                query_results = ast_query(self.ast_indexer, f, "all")
                ast_results.extend(query_results)
                if target_symbol:
                    for node in query_results:
                        if str(node.get("name", "")) == str(target_symbol):
                            exact_definitions.append(node)

        if target_symbol and not exact_definitions and target_file:
            exact_definitions.extend([
                node for node in ast_results
                if str(node.get("name", "")) == str(target_symbol)
                and target_file in str(node.get("file", target_file))
            ])

        data = {"ast_results": ast_results[:50], "exact_definitions": exact_definitions[:10]}
        return SubAgentResult(
            role="code_searcher", task=task, success=True,
            data=data, token_estimate=len(str(data)) // 4,
        )

    async def _run_symbol_searcher(self, task: str, files: list[str], blackboard: Blackboard) -> SubAgentResult:
        """Search for symbol usages across the codebase using ripgrep and LSP.

        Runs broad text search then narrows with symbol lookup.
        """
        from codeforge_mcp.tools.navigation import code_search

        symbol_results: dict[str, Any] | None = None
        search_results: list[dict[str, Any]] = []

        # Extract meaningful terms from the task and run ripgrep searches
        terms = self._extract_search_terms(task, limit=5)
        for term in terms[:5]:
            sr = code_search(self.project_root, term, context=1)
            search_results.extend(sr)

        # Symbol lookup for the most significant term
        if terms:
            top_term = terms[0]
            symbol_results = await self._symbol_lookup_async(top_term)
            if symbol_results:
                await blackboard.add_symbols([symbol_results["name"]])

        # Post search results to the blackboard so downstream capabilities
        # can leverage them
        search_files = list({r["file"] for r in search_results[:50]})
        if search_files:
            await blackboard.add_files(search_files)
            await blackboard.mark("files_seeded")
        if symbol_results:
            await blackboard.mark("symbols_seeded")

        data = {
            "symbol_results": symbol_results,
            "search_results": search_results[:50],
        }
        return SubAgentResult(
            role="symbol_searcher", task=task, success=True,
            data=data, token_estimate=len(str(data)) // 4,
        )

    async def _symbol_lookup_async(self, name: str) -> dict[str, Any] | None:
        from codeforge_mcp.tools.navigation import symbol_lookup
        return await symbol_lookup(self.lsp, self.graph, name)

    async def _run_reviewer(self, task: str, files: list[str], blackboard: Blackboard) -> SubAgentResult:
        from codeforge_mcp.tools.understanding import impact_analysis

        diagnostics_results: list[dict[str, Any]] = []
        impact_results: list[dict[str, Any]] = []
        findings: list[dict[str, Any]] = []
        blocking_issues: list[dict[str, Any]] = []
        review_context = blackboard.facts.get("review_context", {}) if blackboard else {}

        if not files and not blackboard.files and blackboard.facts.get("coordination_mode") == "swarm":
            await blackboard.wait_for("files_seeded")

        # Combine provided files with blackboard files
        target_files = sorted(list(set(files) | blackboard.files))

        # ── Guard: no files to review ─────────────────────────────────
        # If no target files were supplied and the blackboard is empty,
        # auto-gather a list of source files from the project (up to 20)
        # rather than returning an error — the reviewer should be able to
        # do something useful without explicit file arguments.
        if not target_files:
            from codeforge_mcp.indexer import discover_files
            from codeforge_mcp.tools.navigation import _in_skipped_dir
            all_files = discover_files(self.project_root)
            # Deduplicate and limit to 20
            seen: set[str] = set()
            gathered: list[str] = []
            for f in all_files:
                if f not in seen and not _in_skipped_dir(Path(f)):
                    seen.add(f)
                    gathered.append(f)
                    if len(gathered) >= 20:
                        break
            target_files = gathered
            if not target_files:
                return SubAgentResult(
                    role="reviewer",
                    task=task,
                    success=False,
                    data={
                        "diagnostics": [],
                        "impact": [],
                        "checklist": [
                            "⚠ No source files found in the project. "
                            "Ensure the project has been indexed or provide "
                            "explicit file paths.",
                        ],
                    },
                    error="No target files available for review",
                    token_estimate=50,
                )

        file_summaries: list[dict[str, Any]] = []
        for f in target_files:
            # Handle both relative and absolute paths for the graph
            abs_path = str((Path(self.project_root).resolve() / f).resolve())
            
            diags = await self.lsp.diagnostics(abs_path)
            diagnostics_results.extend([{**d, "file": f} for d in diags])

            # Try absolute then relative
            syms = self.graph.symbols_in_file(abs_path)
            if not syms:
                syms = self.graph.symbols_in_file(f)

            file_summary: dict[str, Any] = {
                "file": f,
                "symbols": len(syms),
                "diagnostics": len(diags),
            }
                
            # ── Phase 5 fix: limit impact analysis to top 5 symbols ───
            # Previously we analyzed up to 20 symbols per file, which
            # produced excessive output and token consumption.  Limit to
            # 5 symbols per file, prioritizing definition-kinds.
            defin_kinds = {"class", "function", "method", "struct", "interface", "constructor"}
            ranked_syms = sorted(
                syms,
                key=lambda s: (0 if str(s.get("kind", "")).lower() in defin_kinds else 1,
                              -int(s.get("line", 0))),
            )
            for sym in ranked_syms[:5]:
                impact = impact_analysis(
                    self.graph, self.ast_indexer, sym["name"], sym["file"]
                )
                # Truncate impact data to reduce output size
                imp_summary = {
                    "risk": impact.get("risk", "LOW"),
                    "module_count": impact.get("module_count", 0),
                    "callers": impact.get("callers", 0),
                    "tests": len(impact.get("tests", [])),
                }
                if impact.get("note"):
                    imp_summary["note"] = impact["note"]
                impact_results.append({"symbol": sym["name"], "impact": imp_summary})

                # Surface entry-point notes from impact_analysis
                if impact.get("note"):
                    file_summary.setdefault("notes", []).append(impact["note"])

            file_summaries.append(file_summary)

        checklist: list[str] = []
        if diagnostics_results:
            checklist.append(f"⚠ Found {len(diagnostics_results)} diagnostic issues")
            for diag in diagnostics_results:
                severity = diag.get("severity", 0)
                severity_label = "error" if severity == 1 else "warning"
                line = diag.get("line")
                if line is None:
                    line = diag.get("range", {}).get("start", {}).get("line", 0) + 1
                finding = {
                    "severity": severity_label,
                    "code": "lsp_diagnostic",
                    "file": diag.get("file", ""),
                    "line": line,
                    "message": diag.get("message", ""),
                }
                findings.append(finding)
                if severity == 1:
                    blocking_issues.append(finding)
        for r in impact_results[:15]:
            imp = r["impact"]
            if imp.get("risk") == "HIGH":
                checklist.append(f"🔴 HIGH risk: {r['symbol']} affects {imp.get('module_count', 0)} modules")
            elif imp.get("risk") == "MED":
                checklist.append(f"🟡 MED risk: {r['symbol']} has {imp.get('callers', 0)} callers")

        changed_ranges = review_context.get("changed_ranges", [])
        for changed in changed_ranges[:5]:
            suspicious = self._inspect_changed_range(
                changed.get("file", ""),
                int(changed.get("start_line", 0) or 0),
                int(changed.get("end_line", 0) or 0),
            )
            findings.extend(suspicious)
            for finding in suspicious:
                if finding["severity"] == "error":
                    blocking_issues.append(finding)
                    checklist.append(
                        f"🔴 Suspicious edit artifact in {finding['file']}:{finding['line']} - "
                        f"{finding['message']}"
                    )

        if checklist:
            for item in checklist:
                await blackboard.add_insight(item)
        else:
            # Provide a meaningful "clean" summary with per-file detail
            # instead of a generic "✅ No issues found"
            total_symbols = sum(fs["symbols"] for fs in file_summaries)
            checklist.append(
                f"✅ {len(target_files)} files reviewed, "
                f"{total_symbols} symbols analyzed, "
                f"0 diagnostic issues, 0 high-risk symbols."
            )
            for fs in file_summaries:
                checklist.append(f"  • {fs['file']}: {fs['symbols']} symbols, {fs['diagnostics']} diagnostics")

        data = {
            "diagnostics": diagnostics_results[:20],
            "impact": impact_results[:10],
            "checklist": checklist,
            "file_summaries": file_summaries,
            "review_context": review_context,
            "findings": findings[:25],
            "blocking_issues": blocking_issues[:20],
            "review_passed": len(blocking_issues) == 0,
        }
        data = self._trim_result_data(data)
        return SubAgentResult(
            role="reviewer", task=task, success=True,
            data=data, token_estimate=len(str(data)) // 4,
        )

    def _inspect_changed_range(self, file_path: str, start_line: int, end_line: int) -> list[dict[str, Any]]:
        """Detect suspicious artifacts in or around a changed range."""
        normalized = self._normalize_file_path(file_path)
        abs_path = (Path(self.project_root).resolve() / normalized).resolve()
        if not abs_path.is_file():
            return []
        lines = abs_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if not lines:
            return []

        start = max(1, start_line or 1)
        end = max(start, end_line or start)
        window_start = max(1, start - 1)
        window_end = min(len(lines), end + 1)
        findings: list[dict[str, Any]] = []
        for idx in range(window_start, window_end):
            current = lines[idx - 1].strip()
            nxt = lines[idx].strip()
            if current and current == nxt:
                findings.append({
                    "severity": "error",
                    "code": "duplicate_adjacent_line",
                    "file": normalized,
                    "line": idx,
                    "message": "Adjacent duplicate non-empty lines detected near the edited region.",
                })
        return findings

    async def _run_test_impact(self, task: str, files: list[str], blackboard: Blackboard) -> SubAgentResult:
        from codeforge_mcp.tools.understanding import call_graph

        # Extract candidate symbol names from the task or blackboard
        candidates: list[str] = list(blackboard.symbols)
        if not candidates and blackboard.facts.get("coordination_mode") == "swarm":
            await blackboard.wait_for("symbols_seeded")
            candidates = list(blackboard.symbols)
        for word in task.split():
            cleaned = word.strip(".?!:;()[]{}\"'`,")
            if (
                len(cleaned) >= 3
                and not cleaned.startswith("/")
                and not cleaned.startswith("./")
                and "." not in cleaned  # no file extensions
            ):
                candidates.append(cleaned)

        # Try exact lookup first (longest candidates first — more specific)
        candidates.sort(key=len, reverse=True)
        symbol_name = ""
        for cand in candidates[:5]:
            sym = self.graph.get_symbol(cand)
            if sym is not None:
                symbol_name = cand
                break

        # Fall back to BM25 search with the cleaned candidates
        if not symbol_name and candidates:
            query = " ".join(candidates[:5])
            results = self.graph.search_symbols(query, limit=1)
            if results:
                symbol_name = results[0]["name"]

        # Absolute last resort: pass cleaned candidates as query
        # (call_graph does its own internal fuzzy search fallback)
        if not symbol_name:
            symbol_name = " ".join(candidates) if candidates else task

        tests_found: list[dict[str, Any]] = []
        all_callers: list[dict[str, Any]] = []
        cg = call_graph(self.graph, symbol_name, direction="upstream", depth=3)

        for caller in cg.get("upstream", []):
            all_callers.append(caller)
            name = caller.get("name", "")
            fname = caller.get("file", "")
            if "test" in name.lower() or "test" in fname.lower() or "spec" in fname.lower():
                tests_found.append({"name": name, "file": fname, "line": caller.get("line", 0)})

        if tests_found:
            await blackboard.add_files([t["file"] for t in tests_found])
            await blackboard.add_insight(f"Identified {len(tests_found)} affected test files for {symbol_name}")
            await blackboard.mark("files_seeded")

        return SubAgentResult(
            role="test_impact", task=task, success=True,
            data={"tests_found": tests_found, "total_callers": len(all_callers), "call_graph": cg},
            token_estimate=len(str(all_callers)) // 4,
        )

    async def _run_diagnose(self, task: str, files: list[str], blackboard: Blackboard) -> SubAgentResult:
        """Diagnose errors by mapping stack traces/messages to root cause (GAP-4).

        Analyzes the task text for error patterns, symbol names, and file paths,
        then uses the knowledge graph + LSP to locate the root cause.
        """
        import re
        from codeforge_mcp.tools.navigation import code_search
        from codeforge_mcp.tools.understanding import impact_analysis

        diagnosis: dict[str, Any] = {
            "error_patterns": [],
            "relevant_symbols": [],
            "diagnostics": [],
            "root_cause_candidates": [],
            "suggested_files": [],
        }

        # Extract error-like patterns from the task text
        error_patterns = re.findall(
            r'(?:Error|Exception|Traceback|TypeError|ValueError|KeyError|'
            r'AttributeError|ImportError|NameError|RuntimeError|'
            r'IndexError|FileNotFoundError)\b[^.]*',
            task, re.IGNORECASE
        )
        diagnosis["error_patterns"] = error_patterns[:5]

        # Extract file:line references from stack traces
        file_refs = re.findall(r'["\']?([\w/.-]+\.\w{1,5})["\']?(?::?(\d+))?', task)
        for fpath, line_str in file_refs[:10]:
            if any(fpath.endswith(ext) for ext in ('.py', '.ts', '.js', '.rs', '.go')):
                diagnosis["suggested_files"].append({"file": fpath, "line": int(line_str) if line_str else 0})
        
        await blackboard.add_files([f["file"] for f in diagnosis["suggested_files"]])
        if diagnosis["suggested_files"]:
            await blackboard.mark("files_seeded")

        # Search for error keywords in the codebase
        search_terms = []
        for word in task.split():
            cleaned = word.strip('"\':;,.()')  
            if len(cleaned) > 3 and not cleaned.startswith('/'):
                search_terms.append(cleaned)

        for term in search_terms[:3]:
            results = code_search(self.project_root, term, context=2)
            if results:
                for r in results[:3]:
                    diagnosis["relevant_symbols"].append(r)

        # Run LSP diagnostics on affected files
        target_files = sorted(list(set(files) | blackboard.files))
        if not target_files and blackboard.facts.get("coordination_mode") == "swarm":
            await blackboard.wait_for("files_seeded")
            target_files = sorted(list(set(files) | blackboard.files))
        for f in target_files[:5]:
            try:
                abs_path = str((Path(self.project_root).resolve() / f).resolve())
                diags = await self.lsp.diagnostics(abs_path)
                diagnosis["diagnostics"].extend([{**d, "file": f} for d in diags[:5]])
            except Exception:
                continue

        # Try to identify root cause via symbol lookup and impact analysis
        for sym_result in diagnosis["relevant_symbols"][:3]:
            sym_name = sym_result.get("text", "").strip()
            if sym_name:
                sym = self.graph.get_symbol(sym_name)
                if sym:
                    impact = impact_analysis(self.graph, self.ast_indexer, sym["name"], sym["file"])
                    diagnosis["root_cause_candidates"].append({
                        "symbol": sym["name"],
                        "file": sym["file"],
                        "line": sym["line"],
                        "risk": impact.get("risk", "LOW"),
                        "callers": impact.get("callers", 0),
                    })
                    await blackboard.add_symbols([sym["name"]])

        return SubAgentResult(
            role="diagnose", task=task, success=True,
            data=diagnosis, token_estimate=len(str(diagnosis)) // 4,
        )
