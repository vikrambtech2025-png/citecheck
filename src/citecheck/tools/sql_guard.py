"""SQLite read-only enforcement for the `query_sqlite` tool.

Defences, in order of application:
  1. Statement-type allowlist -- only SELECT / WITH may execute.
  2. Single-statement check -- semicolon-separated batches are rejected.
  3. Read-only connection (`mode=ro`), so even a bypassed check cannot write.
  4. Injected LIMIT, so a model cannot pull the whole table.
  5. `set_progress_handler` wall-clock cap, so a pathological join cannot hang
     the agent loop.

Layer 3 is the important one: it is a capability guarantee from SQLite itself,
not a string check we might get wrong.
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Any

# Keywords that mutate state. Rejected even if a validator were bypassed.
_FORBIDDEN = (
    "insert", "update", "delete", "drop", "alter", "create",
    "replace", "truncate", "attach", "detach", "vacuum", "reindex",
)

# sqlite_* functions that expose the filesystem. Not dangerous here because the
# connection is read-only and sandboxed, but they are noise in a research agent.
_FORBIDDEN_FUNCS = ("load_extension", "readfile", "writefile", "edit")


class SqlRejected(Exception):
    """The query failed a safety check."""


def validate_sql(sql: str) -> str:
    """Return the query if safe, else raise SqlRejected."""
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        raise SqlRejected("empty query")

    # -- (b) single statement ---------------------------------------------
    body = _strip_literals_and_comments(stripped)
    if ";" in body:
        raise SqlRejected("multiple statements are not allowed")

    lowered = body.lower()
    first = lowered.lstrip().split(None, 1)[0] if lowered.split() else ""
    if first not in ("select", "with"):
        raise SqlRejected(f"only SELECT/WITH allowed, got {first.upper() or 'nothing'!r}")

    # -- (a) forbidden keywords -------------------------------------------
    for kw in _FORBIDDEN:
        if re.search(rf"\b{kw}\b", lowered):
            raise SqlRejected(f"statement contains forbidden keyword: {kw.upper()}")

    for fn in _FORBIDDEN_FUNCS:
        if re.search(rf"\b{fn}\s*\(", lowered):
            raise SqlRejected(f"function not permitted: {fn}()")

    return stripped


def _strip_literals_and_comments(sql: str) -> str:
    """Remove string literals and comments so keyword scanning cannot be fooled.

    Without this, `SELECT 'drop table'` would trip the keyword filter, and
    `-- drop table` in a trailing comment would too.
    """
    out: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch == "'":
            i += 1
            while i < n:
                if sql[i] == "'":
                    if i + 1 < n and sql[i + 1] == "'":
                        i += 2
                        continue
                    break
                i += 1
        elif ch == '"':
            i += 1
            while i < n and sql[i] != '"':
                i += 1
        elif ch == "-" and i + 1 < n and sql[i + 1] == "-":
            while i < n and sql[i] != "\n":
                i += 1
            continue
        elif ch == "/" and i + 1 < n and sql[i + 1] == "*":
            i += 2
            closed = False
            while i + 1 < n:
                if sql[i] == "*" and sql[i + 1] == "/":
                    closed = True
                    break
                i += 1
            if not closed:
                # An unterminated comment would otherwise swallow everything after
                # it, hiding a trailing write statement from the keyword scan.
                raise SqlRejected("unterminated block comment")
            i += 2
            continue
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def apply_limit(sql: str, max_rows: int) -> str:
    """Wrap the query so it cannot return more than `max_rows` rows."""
    return f"SELECT * FROM ({sql}) LIMIT {int(max_rows)}"


def open_readonly(db_path: Path) -> sqlite3.Connection:
    """Open `db_path` in SQLite's own read-only mode."""
    if not db_path.exists():
        raise SqlRejected(f"database not found: {db_path.name}")
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=5.0)


def run_query(
    db_path: Path, sql: str, *, max_rows: int = 200, timeout_s: float = 5.0
) -> dict[str, Any]:
    """Validate, execute read-only, and return rows. Raises SqlRejected on refusal."""
    safe = validate_sql(sql)
    limited = apply_limit(safe, max_rows)

    conn = open_readonly(db_path)
    try:
        _install_timeout(conn, timeout_s)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(limited)
        rows = cur.fetchmany(max_rows)
        columns = [d[0] for d in cur.description] if cur.description else []

        # Drop trailing LIMIT-0 padding rows produced by the wrapper.
        trimmed = [dict(r) for r in rows][:max_rows]
        return {
            "columns": columns,
            "rows": trimmed,
            "row_count": len(trimmed),
            "truncated": len(rows) >= max_rows,
        }
    except sqlite3.Error as e:
        raise SqlRejected(f"sqlite error: {e}") from e
    finally:
        conn.close()


def _install_timeout(conn: sqlite3.Connection, timeout_s: float) -> None:
    """Abort the query if it exceeds `timeout_s` (checked every 1000 VM steps)."""
    deadline = time.perf_counter() + timeout_s

    def handler() -> int:
        # Returning a non-zero value tells SQLite to interrupt the statement.
        return 1 if time.perf_counter() > deadline else 0

    conn.set_progress_handler(handler, 1000)


def init_demo_db(db_path: Path) -> Path:
    """Create a small seeded knowledge base so the agent has something to query.

    Realistic enough to make demos meaningful, small enough to reason about.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL UNIQUE,
                authors TEXT NOT NULL,
                year INTEGER NOT NULL,
                venue TEXT,
                abstract TEXT
            );
            CREATE TABLE IF NOT EXISTS datasets (
                name TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                size_mb INTEGER,
                license TEXT
            );
            """
        )
        papers = [
            ("Attention Is All You Need",
             "Vaswani et al.", 2017, "NeurIPS",
             "Introduces the Transformer, dispensing with recurrence and convolutions."),
            ("BERT: Pre-training of Deep Bidirectional Transformers",
             "Devlin et al.", 2019, "NAACL",
             "Bidirectional pre-training for language representation understanding."),
            ("Retrieval-Augmented Generation for Knowledge-Intensive NLP",
             "Lewis et al.", 2020, "NeurIPS",
             "Combines parametric and non-parametric memory; improves factuality on open QA."),
            ("ReAct: Synergizing Reasoning and Acting in Language Models",
             "Yao et al.", 2023, "ICLR",
             "Interleaves reasoning traces and task actions so agents use external tools."),
            ("Toolformer: Language Models Can Teach Themselves to Use Tools",
             "Schick et al.", 2023, "NeurIPS",
             "Self-supervised method for teaching an LM when to call APIs."),
            ("Model Context Protocol",
             "Anthropic", 2024, "Spec",
             "Open protocol exposing tools and resources to LLM applications."),
        ]

        # `INSERT OR IGNORE` is only as idempotent as the constraints behind it.
        # `papers.title` had no UNIQUE index, so `OR IGNORE` had nothing to ignore
        # against and every `citecheck init` silently appended a second copy of
        # every paper. `COUNT(*)` then reported 12 for a 6-paper database, and the
        # agent was graded -- correctly -- against a knowledge base that lied.
        # Guarding on existence works regardless of the table's constraints, so
        # re-initialising is now safe even for a database already corrupted this
        # way.
        conn.executemany(
            "INSERT INTO papers (title, authors, year, venue, abstract) "
            "SELECT ?,?,?,?,? WHERE NOT EXISTS (SELECT 1 FROM papers WHERE title = ?)",
            [(*p, p[0]) for p in papers],
        )

        # Heal databases that were already doubled, keeping the lowest id per
        # title so `citecheck init` is a repair as well as a seed.
        conn.execute(
            "DELETE FROM papers WHERE id NOT IN (SELECT MIN(id) FROM papers GROUP BY title)"
        )
        conn.executemany(
            "INSERT OR IGNORE INTO datasets (name, domain, size_mb, license) VALUES (?,?,?,?)",
            [
                ("MS MARCO", "search", 320, "research"),
                ("Natural Questions", "qa", 120, "cc-by-sa-3.0"),
                ("HotpotQA", "multi-hop qa", 47, "cc-by-sa-4.0"),
                ("MMLU", "knowledge", 165, "MIT"),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return db_path