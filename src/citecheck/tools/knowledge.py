"""SQLite tool: `query_sqlite`, plus schema discovery.

Every query goes through `sql_guard.validate_sql` and executes on a connection
opened with `mode=ro`. See that module for why the allowlist is defence-in-depth
rather than the primary control.
"""

from __future__ import annotations

from pathlib import Path

from citecheck.tools import sql_guard
from citecheck.tools.base import Source, ToolContext, ToolResult


def _db_path(ctx: ToolContext) -> Path:
    s = ctx.settings
    return s.sqlite_path or (s.workspace_dir / "knowledge.sqlite3")


def database_source_id(db: Path) -> str:
    """Stable source id for the knowledge base.

    Exposed so callers (and tests) can predict the id the agent will be asked to
    cite, rather than hard-coding a string that can silently drift.
    """
    return f"D{db.name.replace('.', '')[:6]}"


def query_sqlite(sql: str, ctx: ToolContext) -> ToolResult:
    """Run a read-only SELECT against the local knowledge base."""
    s = ctx.settings
    if not (sql or "").strip():
        return ToolResult.failure("empty query")

    db = _db_path(ctx)
    try:
        result = sql_guard.run_query(
            db, sql,
            max_rows=s.sqlite_max_rows,
            timeout_s=s.sqlite_query_timeout_s,
        )
    except sql_guard.SqlRejected as e:
        return ToolResult.failure(str(e), rejected=True)
    except Exception as e:
        return ToolResult.failure(f"unexpected: {type(e).__name__}: {e}")

    db_name = db.name
    text = _rows_to_text(result, sql)
    source = Source(
        id=database_source_id(db),
        uri=f"sqlite://{db_name}",
        title=f"Knowledge base ({db_name})",
        text=text,
        kind="db",
        content_hash=__import__("hashlib").sha256(text.encode()).hexdigest()[:16],
    )
    return ToolResult(
        data={**result, "source_id": source.id, "database": db_name},
        source=source,
        meta={"row_count": result["row_count"], "truncated": result["truncated"]},
    )


def list_tables(ctx: ToolContext) -> ToolResult:
    """Return the tables and columns available in the knowledge base."""
    db = _db_path(ctx)
    try:
        result = sql_guard.run_query(
            db, "SELECT name, type FROM sqlite_master WHERE type IN ('table','view')",
            max_rows=100, timeout_s=ctx.settings.sqlite_query_timeout_s,
        )
    except sql_guard.SqlRejected as e:
        return ToolResult.failure(str(e), rejected=True)
    return ToolResult(data={"objects": result["rows"]}, meta={"count": result["row_count"]})


def _rows_to_text(result: dict, sql: str = "") -> str:
    """Render the retrieved artifact as TSV, prefixed by the statement that ran.

    The statement is part of the provenance, not decoration: it names the table
    a figure came from, so `SELECT COUNT(*) FROM papers` -> `12` can support
    "there are 12 papers". Without it the recorded text was a bare cell and
    nothing in it could ground the claim it came from.
    """
    cols = result["columns"]
    lines = [sql.strip()] if sql.strip() else []
    lines.append("\t".join(cols))
    for row in result["rows"]:
        lines.append("\t".join("" if row.get(c) is None else str(row.get(c)) for c in cols))
    return "\n".join(lines)