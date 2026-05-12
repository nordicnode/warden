"""Memory tools — decision_record, brief.

decision_record: writes institutional memory to .codeforge/decisions.md
brief: returns a summary of the codebase state.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Sequence


def decision_record(
    project_root: str | Path,
    graph: Any,
    title: str,
    why: str,
    files: Sequence[str] = (),
) -> dict[str, Any]:
    """Record a design decision in the knowledge graph and on-disk markdown.

    Args:
        project_root: Project directory.
        graph: KnowledgeGraph instance.
        title: Decision title.
        why: Reason for the decision.
        files: Files affected by the decision.

    Returns:
        {id, title, date}
    """
    root = Path(project_root)
    decision_id = graph.add_decision(title, why, files)

    # Also write to .codeforge/decisions.md
    decisions_dir = root / ".codeforge"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    decisions_file = decisions_dir / "decisions.md"

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry = f"\n## {now} — {title}\n\n**Why:** {why}\n\n"
    if files:
        entry += "**Files:**\n"
        for f in files:
            entry += f"- `{f}`\n"
    entry += f"\n**ID:** {decision_id}\n"

    _append_to_file(decisions_file, entry)

    return {
        "id": decision_id,
        "title": title,
        "date": now,
    }


def _append_to_file(path: Path, content: str) -> None:
    """Append text to a file, creating it if needed."""
    mode = "a" if path.exists() else "w"
    if mode == "w":
        content = "# Codeforge Decisions\n" + content
    with open(path, mode) as f:
        f.write(content)


def brief(graph: Any) -> dict[str, Any]:
    """Return a summary of the codebase: symbol count, file count, knowledge score."""
    # This calls the graph.brief() method to get the stats
    return graph.brief()
