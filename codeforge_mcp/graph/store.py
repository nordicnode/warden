"""Knowledge Graph — SQLite-backed symbol, edge, constraint, and decision storage.

Schema:
  symbols(id, name, kind, file, line, signature, doc, hash)
  edges(from_id, to_id, kind)
  constraints(symbol_id, text)
  decisions(id, date, title, why, files_json)
"""

from __future__ import annotations

import sqlite3
import json
import math
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Sequence

SymbolKind = Literal[
    "function", "method", "class", "module", "variable",
    "interface", "type", "enum", "constant", "unknown",
]

EdgeKind = Literal["calls", "imports", "implements", "references", "inherits"]


class KnowledgeGraph:
    """Persistent SQLite knowledge graph for code symbols and their relationships."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._batch_depth = 0  # nested begin_batch support
        self._write_lock = threading.RLock()  # reentrant: supports batch nesting
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'unknown',
                file TEXT NOT NULL,
                line INTEGER NOT NULL DEFAULT 0,
                signature TEXT DEFAULT '',
                doc TEXT DEFAULT '',
                hash TEXT DEFAULT '',
                UNIQUE(file, name, kind)
            );
            CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
            CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file);
            CREATE INDEX IF NOT EXISTS idx_symbols_kind ON symbols(kind);

            -- FTS5 Virtual Table for fast symbol search
            CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
                name,
                content='symbols',
                content_rowid='id',
                tokenize='trigram'
            );

            -- Triggers to keep FTS index in sync
            CREATE TRIGGER IF NOT EXISTS symbols_ai AFTER INSERT ON symbols BEGIN
                INSERT INTO symbols_fts(rowid, name) VALUES (new.id, new.name);
            END;
            CREATE TRIGGER IF NOT EXISTS symbols_ad AFTER DELETE ON symbols BEGIN
                INSERT INTO symbols_fts(symbols_fts, rowid, name) VALUES('delete', old.id, old.name);
            END;
            CREATE TRIGGER IF NOT EXISTS symbols_au AFTER UPDATE ON symbols BEGIN
                INSERT INTO symbols_fts(symbols_fts, rowid, name) VALUES('delete', old.id, old.name);
                INSERT INTO symbols_fts(rowid, name) VALUES (new.id, new.name);
            END;

            CREATE TABLE IF NOT EXISTS edges (
                from_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
                to_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
                kind TEXT NOT NULL DEFAULT 'references',
                PRIMARY KEY(from_id, to_id, kind)
            );
            CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_id);
            CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_id);
            CREATE INDEX IF NOT EXISTS idx_edges_kind ON edges(kind);

            CREATE TABLE IF NOT EXISTS constraints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol_id INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
                text TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_constraints_symbol ON constraints(symbol_id);

            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL DEFAULT (datetime('now')),
                title TEXT NOT NULL,
                why TEXT NOT NULL,
                files_json TEXT DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_decisions_date ON decisions(date);

            -- Diff baseline: snapshot of symbol hashes at the last manual reindex.
            -- _synthetic_diff compares current disk state against this snapshot
            -- rather than the live graph (which is auto-updated after every edit
            -- via _reindex_after_edit).
            CREATE TABLE IF NOT EXISTS diff_baseline (
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                file TEXT NOT NULL,
                hash TEXT DEFAULT '',
                PRIMARY KEY (file, name, kind)
            );
        """)

    # ── Batch commit support ─────────────────────────────────────────────

    def commit(self) -> None:
        """Commit the current transaction.  No-op inside a begin_batch() block."""
        if self._batch_depth == 0:
            self.conn.commit()

    def begin_batch(self) -> None:
        """Start a batch block — commits are suppressed until end_batch().

        Also acquires the write lock so the batch is atomic w.r.t.
        concurrent writers.
        """
        if self._batch_depth == 0:
            self._write_lock.acquire()
        self._batch_depth += 1

    def end_batch(self) -> None:
        """End a batch block and commit if this is the outermost block."""
        self._batch_depth = max(0, self._batch_depth - 1)
        if self._batch_depth == 0:
            try:
                self.conn.commit()
            finally:
                self._write_lock.release()

    # ── Symbols ──────────────────────────────────────────────────────────

    def upsert_symbol(
        self,
        name: str,
        kind: SymbolKind,
        file: str,
        line: int = 0,
        signature: str = "",
        doc: str = "",
        content_hash: str = "",
    ) -> int:
        """Insert or update a symbol; return its id."""
        with self._write_lock:
            # Check for existing symbol first if we're in a batch and need an ID
            # lastrowid is unreliable for ON CONFLICT DO UPDATE in some SQLite versions
            existing = self.conn.execute(
                "SELECT id FROM symbols WHERE file = ? AND name = ? AND kind = ?",
                (file, name, kind)
            ).fetchone()

            cur = self.conn.execute(
                """INSERT INTO symbols (name, kind, file, line, signature, doc, hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(file, name, kind) DO UPDATE SET
                   line=excluded.line, signature=excluded.signature,
                   doc=excluded.doc, hash=excluded.hash""",
                (name, kind, file, line, signature, doc, content_hash),
            )
            
            sid = existing[0] if existing else cur.lastrowid
            self.commit()
            return sid or 0


    def delete_symbols_in_file(self, file: str) -> int:
        """Remove all symbols for a given file; return count removed."""
        with self._write_lock:
            cur = self.conn.execute("DELETE FROM symbols WHERE file = ?", (file,))
            self.commit()
            return cur.rowcount

    def get_symbol(self, name: str, file: str | None = None) -> dict[str, Any] | None:
        """Look up a symbol by name, optionally scoped to a file."""
        if file:
            row = self.conn.execute(
                "SELECT * FROM symbols WHERE name = ? AND file = ? LIMIT 1",
                (name, file),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM symbols WHERE name = ? LIMIT 1", (name,)
            ).fetchone()
        if row is None:
            return None
        return dict(zip([c[0] for c in self.conn.execute("SELECT * FROM symbols LIMIT 0").description], row))

    def search_symbols(self, query: str, limit: int = 50) -> list[dict[str, Any]]:
        """High-performance symbol search using SQLite FTS5."""
        if not query.strip():
            return []

        # Sanitize query for FTS5 (escape special characters)
        sanitized = query.replace('"', '""')
        
        # We use a 'trigram' tokenizer which allows for substring matches
        # e.g. "search" matches "search_symbols"
        sql = """
            SELECT s.*, rank
            FROM symbols s
            JOIN symbols_fts f ON s.id = f.rowid
            WHERE symbols_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """
        
        try:
            rows = self.conn.execute(sql, (sanitized, limit)).fetchall()
        except sqlite3.OperationalError:
            # Fallback for complex queries that FTS5 might reject
            sql = "SELECT *, 0 as rank FROM symbols WHERE name LIKE ? LIMIT ?"
            rows = self.conn.execute(sql, (f"%{query}%", limit)).fetchall()

        if not rows:
            return []

        cols = [c[0] for c in self.conn.execute("SELECT * FROM symbols LIMIT 0").description]
        return [dict(zip(cols, r)) for r in rows]

    def _avg_name_length(self) -> float:
        row = self.conn.execute(
            "SELECT AVG(LENGTH(name)) FROM symbols"
        ).fetchone()
        return float(row[0]) if row and row[0] else 10.0

    def symbols_in_file(self, file: str) -> list[dict[str, Any]]:
        """Return all symbols declared in a file."""
        rows = self.conn.execute(
            "SELECT * FROM symbols WHERE file = ? ORDER BY line", (file,)
        ).fetchall()
        cols = [c[0] for c in self.conn.execute("SELECT * FROM symbols LIMIT 0").description]
        return [dict(zip(cols, r)) for r in rows]

    def symbol_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]

    def file_count(self) -> int:
        return self.conn.execute("SELECT COUNT(DISTINCT file) FROM symbols").fetchone()[0]

    def hashes_for_file(self, file: str) -> dict[str, str]:
        """Return {name: hash} for all symbols in a file (for incremental indexing)."""
        rows = self.conn.execute(
            "SELECT name, hash FROM symbols WHERE file = ?", (file,)
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    # ── Edges ────────────────────────────────────────────────────────────

    def add_edge(self, from_id: int, to_id: int, kind: EdgeKind = "references") -> None:
        with self._write_lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO edges (from_id, to_id, kind) VALUES (?, ?, ?)",
                (from_id, to_id, kind),
            )
            self.commit()

    def clear_edges_from(self, from_id: int) -> None:
        with self._write_lock:
            self.conn.execute("DELETE FROM edges WHERE from_id = ?", (from_id,))
            self.commit()

    def upstream(self, symbol_id: int, depth: int = 3) -> list[dict[str, Any]]:
        """Return callers (who calls this symbol) up to *depth* hops."""
        visited: set[int] = set()
        frontier = [symbol_id]
        results: list[dict] = []

        for _ in range(depth):
            if not frontier:
                break
            placeholders = ",".join("?" * len(frontier))
            rows = self.conn.execute(
                f"SELECT s.*, e.kind FROM symbols s JOIN edges e ON s.id = e.from_id "
                f"WHERE e.to_id IN ({placeholders})",
                frontier,
            ).fetchall()
            cols = [c[0] for c in self.conn.execute("SELECT * FROM symbols LIMIT 0").description] + ["edge_kind"]
            new_frontier: list[int] = []
            for r in rows:
                d = dict(zip(cols, r))
                sid = d["id"]
                if sid not in visited:
                    visited.add(sid)
                    results.append(d)
                    new_frontier.append(sid)
            frontier = new_frontier

        return results

    def downstream(self, symbol_id: int, depth: int = 3) -> list[dict[str, Any]]:
        """Return callees (what this symbol calls) up to *depth* hops."""
        visited: set[int] = set()
        frontier = [symbol_id]
        results: list[dict] = []

        for _ in range(depth):
            if not frontier:
                break
            placeholders = ",".join("?" * len(frontier))
            rows = self.conn.execute(
                f"SELECT s.*, e.kind FROM symbols s JOIN edges e ON s.id = e.to_id "
                f"WHERE e.from_id IN ({placeholders})",
                frontier,
            ).fetchall()
            cols = [c[0] for c in self.conn.execute("SELECT * FROM symbols LIMIT 0").description] + ["edge_kind"]
            new_frontier: list[int] = []
            for r in rows:
                d = dict(zip(cols, r))
                sid = d["id"]
                if sid not in visited:
                    visited.add(sid)
                    results.append(d)
                    new_frontier.append(sid)
            frontier = new_frontier

        return results

    def call_graph(self, symbol_id: int, direction: str = "both", depth: int = 2) -> dict[str, Any]:
        """Return {upstream: [...], downstream: [...]}."""
        result: dict = {}
        if direction in ("upstream", "both"):
            result["upstream"] = self.upstream(symbol_id, depth)
        if direction in ("downstream", "both"):
            result["downstream"] = self.downstream(symbol_id, depth)
        return result

    def affected_modules(self, symbol_id: int, depth: int = 3) -> list[str]:
        """Return list of unique files in the blast radius."""
        upstream = self.upstream(symbol_id, depth)
        downstream = self.downstream(symbol_id, depth)
        files = set()
        for d in upstream + downstream:
            files.add(d["file"])
        # Also include the symbol's own file
        row = self.conn.execute(
            "SELECT file FROM symbols WHERE id = ?", (symbol_id,)
        ).fetchone()
        if row:
            files.add(row[0])
        return sorted(files)

    # ── Constraints ──────────────────────────────────────────────────────

    def add_constraint(self, symbol_id: int, text: str) -> int:
        with self._write_lock:
            cur = self.conn.execute(
                "INSERT INTO constraints (symbol_id, text) VALUES (?, ?)",
                (symbol_id, text),
            )
            self.commit()
            return cur.lastrowid or 0

    def constraints_for(self, symbol_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM constraints WHERE symbol_id = ?", (symbol_id,)
        ).fetchall()
        cols = [c[0] for c in self.conn.execute("SELECT * FROM constraints LIMIT 0").description]
        return [dict(zip(cols, r)) for r in rows]

    # ── Decisions ────────────────────────────────────────────────────────

    def add_decision(self, title: str, why: str, files: Sequence[str] = ()) -> int:
        with self._write_lock:
            cur = self.conn.execute(
                "INSERT INTO decisions (date, title, why, files_json) VALUES (datetime('now'), ?, ?, ?)",
                (title, why, json.dumps(list(files))),
            )
            self.commit()
            return cur.lastrowid or 0

    def recent_decisions(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM decisions ORDER BY date DESC LIMIT ?", (limit,)
        ).fetchall()
        cols = [c[0] for c in self.conn.execute("SELECT * FROM decisions LIMIT 0").description]
        return [dict(zip(cols, r)) for r in rows]

    # ── Stats ────────────────────────────────────────────────────────────

    def knowledge_score(self) -> float:
        """Heuristic: combo of symbol coverage, edge density, and decisions.

        Score ranges from 0.0 (empty) to ~100.0 (well-indexed with many
        call edges and architecture decisions recorded).

        Components:
          - Symbol count (log-scaled, max ~66)
          - Edge density (edges/symbols, max ~33)
          - Decision count bonus (up to ~1)
        """
        syms = self.symbol_count()
        if syms == 0:
            return 0.0
        edges = self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        decisions = self.conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]

        # Log-scaled symbol score (up to ~66 for 100,000 symbols)
        sym_score = min(math.log(max(syms, 1), 10) * 20, 66)
        # Edge density score (edges/symbols, capped at 1.0)
        density = min(edges / syms, 1.0)
        edge_score = density * 33
        # Decision bonus (tiny)
        dec_score = min(decisions * 0.1, 1)

        return round(sym_score + edge_score + dec_score, 2)

    def brief(self) -> dict[str, Any]:
        """Return a summary: file count, symbol count, knowledge score."""
        return {
            "file_count": self.file_count(),
            "symbol_count": self.symbol_count(),
            "knowledge_score": self.knowledge_score(),
        }

    def clear_symbols(self) -> int:
        """Delete all symbols from the graph. Returns count of symbols removed.

        Edges and constraints are cascade-deleted via foreign keys.
        Decisions are preserved (they are independent records).
        """
        with self._write_lock:
            count = self.symbol_count()
            self.conn.execute("DELETE FROM symbols")
            self.commit()
            return count

    # ── Diff Baseline ────────────────────────────────────────────────────

    def snapshot_diff_baseline(self) -> int:
        """Copy current symbol hashes into the diff_baseline table.

        This is called after a manual full reindex (reindex() tool) to
        establish a stable comparison point.  Incremental updates from
        _reindex_after_edit() do NOT call this, so _synthetic_diff can
        detect changes that happened after the last manual index.

        Returns the number of rows written.
        """
        with self._write_lock:
            self.conn.execute("DELETE FROM diff_baseline")
            self.conn.execute(
                """INSERT INTO diff_baseline (name, kind, file, hash)
                   SELECT name, kind, file, hash FROM symbols"""
            )
            self.commit()
            return self.conn.execute("SELECT COUNT(*) FROM diff_baseline").fetchone()[0]

    def baseline_hashes_for_file(self, file: str) -> dict[str, str]:
        """Return {name: hash} from the diff baseline for a file.

        Returns an empty dict if no baseline has been captured yet
        (the first reindex hasn't happened or the baseline was cleared).
        """
        rows = self.conn.execute(
            "SELECT name, hash FROM diff_baseline WHERE file = ?", (file,)
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def close(self) -> None:
        self.conn.close()
