"""Sandbox and SQL guard: the capability-level controls."""

from __future__ import annotations

from pathlib import Path

import pytest

from citecheck.tools import sandbox, sql_guard


# --------------------------------------------------------------------------
# filesystem sandbox
# --------------------------------------------------------------------------
def test_relative_path_inside_root_is_allowed(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    assert sandbox.safe_read("a.txt", tmp_path, max_bytes=1000) == "hello"


def test_parent_traversal_is_blocked(tmp_path: Path) -> None:
    secret = tmp_path.parent / "secret.txt"
    secret.write_text("API_KEY=hunter2", encoding="utf-8")
    try:
        with pytest.raises(sandbox.SandboxViolation):
            sandbox.safe_read("../secret.txt", tmp_path, max_bytes=10_000)
    finally:
        secret.unlink(missing_ok=True)


def test_deep_traversal_is_blocked(tmp_path: Path) -> None:
    with pytest.raises(sandbox.SandboxViolation):
        sandbox.resolve_in(tmp_path, "../../../../Windows/System32/config")


def test_absolute_path_outside_root_is_blocked(tmp_path: Path) -> None:
    with pytest.raises(sandbox.SandboxViolation):
        sandbox.resolve_in(tmp_path, "C:/Windows/System32")


def test_absolute_path_inside_root_is_allowed(tmp_path: Path) -> None:
    target = tmp_path / "inside.txt"
    target.write_text("ok", encoding="utf-8")
    assert sandbox.safe_read(str(target), tmp_path, max_bytes=100) == "ok"


def test_write_cannot_escape_root(tmp_path: Path) -> None:
    with pytest.raises(sandbox.SandboxViolation):
        sandbox.safe_write("../escaped.md", tmp_path, "nope", max_bytes=1000)
    assert not (tmp_path.parent / "escaped.md").exists()


def test_write_creates_nested_dirs(tmp_path: Path) -> None:
    written = sandbox.safe_write("a/b/c.md", tmp_path, "# hi", max_bytes=1000)
    assert written.read_text(encoding="utf-8") == "# hi"


def test_size_cap_is_enforced(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("x" * 500, encoding="utf-8")
    with pytest.raises(ValueError, match="too large"):
        sandbox.safe_read("big.txt", tmp_path, max_bytes=100)


def test_list_dir_reports_entries(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_text("1", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    entries = sandbox.safe_list(".", tmp_path)
    names = {e["name"]: e["type"] for e in entries}
    assert names == {"one.txt": "file", "sub": "dir"}


def test_guarded_converts_violation_to_result(tmp_path: Path) -> None:
    from citecheck.tools.base import ToolResult

    result = sandbox.guarded(lambda: sandbox.safe_read("../x", tmp_path, max_bytes=10))
    assert isinstance(result, ToolResult)
    assert not result.ok
    assert result.meta.get("blocked") is True


# --------------------------------------------------------------------------
# SQL guard
# --------------------------------------------------------------------------
def _db(tmp_path: Path) -> Path:
    from citecheck.tools.sql_guard import init_demo_db

    return init_demo_db(tmp_path / "kb.sqlite3")


def test_query_sqlite_records_the_row_count_on_the_source(seeded) -> None:
    """An empty result is the absence of evidence, and must be markable as such."""
    from citecheck.tools.base import ToolContext
    from citecheck.tools.knowledge import query_sqlite

    ctx = ToolContext(settings=seeded, run_id="test")
    hit = query_sqlite("SELECT title FROM papers LIMIT 1", ctx)
    assert hit.source is not None and hit.source.row_count == 1

    miss = query_sqlite("SELECT title FROM papers WHERE title = 'nope'", ctx)
    assert miss.source is not None and miss.source.row_count == 0


def test_select_is_allowed(tmp_path: Path) -> None:
    result = sql_guard.run_query(_db(tmp_path), "SELECT title FROM papers", max_rows=10)
    assert result["row_count"] == 6
    assert result["columns"] == ["title"]


def _count(db: Path, table: str) -> int:
    return int(sql_guard.run_query(db, f"SELECT COUNT(*) FROM {table}", max_rows=1)["rows"][0]["COUNT(*)"])


def test_init_demo_db_is_idempotent(tmp_path: Path) -> None:
    """Re-initialising must not append a second copy of every row.

    Regression found by the eval: `papers.title` carried no UNIQUE constraint,
    so `INSERT OR IGNORE` had nothing to ignore against and every `citecheck
    init` appended six more rows. `COUNT(*)` then reported 12 for a 6-paper
    database, and the agent was graded -- correctly, against the evidence it
    was shown -- for answering 12. The citation guarantee was working perfectly
    on a knowledge base that lied.
    """
    from citecheck.tools.sql_guard import init_demo_db

    db = tmp_path / "kb.sqlite3"
    for _ in range(4):
        init_demo_db(db)
    assert _count(db, "papers") == 6
    assert _count(db, "datasets") == 4


def test_init_demo_db_heals_a_database_the_old_seeder_duplicated(
    tmp_path: Path,
) -> None:
    """A legacy database with no UNIQUE constraint is repaired, not compounded."""
    import sqlite3

    from citecheck.tools.sql_guard import init_demo_db

    db = tmp_path / "kb.sqlite3"
    conn = sqlite3.connect(db)
    # Exactly the pre-fix schema, so the new seeder meets a legacy database.
    conn.executescript(
        """
        CREATE TABLE papers (
            id INTEGER PRIMARY KEY, title TEXT NOT NULL, authors TEXT NOT NULL,
            year INTEGER NOT NULL, venue TEXT, abstract TEXT
        );
        CREATE TABLE datasets (
            name TEXT PRIMARY KEY, domain TEXT NOT NULL,
            size_mb INTEGER, license TEXT
        );
        """
    )
    conn.commit()
    conn.close()

    init_demo_db(db)
    assert _count(db, "papers") == 6
    # Reproduce the corruption the old seeder caused: six more rows, no UNIQUE.
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO papers (title, authors, year, venue, abstract) "
                 "SELECT title, authors, year, venue, abstract FROM papers")
    conn.commit()
    conn.close()
    assert _count(db, "papers") == 12

    init_demo_db(db)  # must repair, not merely stop growing
    assert _count(db, "papers") == 6
    titles = sql_guard.run_query(db, "SELECT title FROM papers", max_rows=10)
    assert len({r["title"] for r in titles["rows"]}) == 6


def test_write_statements_are_rejected(tmp_path: Path) -> None:
    for sql in (
        "DELETE FROM papers",
        "UPDATE papers SET title = 'x'",
        "INSERT INTO papers VALUES (99,'a','b',2020,NULL,NULL)",
        "DROP TABLE papers",
        "ALTER TABLE papers ADD COLUMN x INT",
    ):
        with pytest.raises(sql_guard.SqlRejected):
            sql_guard.run_query(_db(tmp_path), sql)


def test_multiple_statements_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(sql_guard.SqlRejected, match="multiple statements"):
        sql_guard.run_query(
            _db(tmp_path), "SELECT 1; DROP TABLE papers", max_rows=5
        )


def test_read_only_connection_blocks_writes_even_if_validation_is_bypassed(tmp_path: Path) -> None:
    # Bypass validate_sql entirely and hand a DELETE straight to the executor.
    # SQLite itself refuses -- this is the capability guarantee, independent of
    # the string allowlist.
    import sqlite3

    db = _db(tmp_path)
    conn = sql_guard.open_readonly(db)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM papers")
    finally:
        conn.close()
    # And the data is intact.
    assert sql_guard.run_query(db, "SELECT COUNT(*) c FROM papers")["rows"][0]["c"] == 6


def test_keyword_inside_string_literal_does_not_trip_the_filter() -> None:
    # A legitimate SELECT mentioning 'drop' must not be rejected.
    assert sql_guard.validate_sql("SELECT 'please drop everything' AS note")


def test_keyword_in_comment_does_not_trip_the_filter() -> None:
    assert sql_guard.validate_sql("SELECT 1 AS a -- drop table papers")


def test_block_comment_hiding_a_keyword_does_not_false_positive() -> None:
    # Comments are stripped before the keyword scan, so a harmless comment
    # mentioning DROP does not cause a rejection.
    assert sql_guard.validate_sql("SELECT 1 /* now DROP TABLE papers */")


def test_unterminated_block_comment_is_not_an_allowlist_bypass() -> None:
    # An unterminated comment must not swallow a real write statement.
    with pytest.raises(sql_guard.SqlRejected):
        sql_guard.validate_sql("SELECT 1 /* DROP TABLE papers")


def test_row_cap_is_applied(tmp_path: Path) -> None:
    result = sql_guard.run_query(
        _db(tmp_path), "SELECT * FROM papers", max_rows=2
    )
    assert result["row_count"] == 2
    assert result["truncated"] is True


def test_cte_is_allowed(tmp_path: Path) -> None:
    result = sql_guard.run_query(
        _db(tmp_path),
        "WITH recent AS (SELECT * FROM papers WHERE year >= 2023) SELECT title FROM recent",
    )
    assert result["row_count"] >= 2


def test_non_select_is_rejected_by_type() -> None:
    with pytest.raises(sql_guard.SqlRejected, match="only SELECT"):
        sql_guard.validate_sql("PRAGMA table_info(papers)")


def test_filesystem_functions_are_rejected() -> None:
    with pytest.raises(sql_guard.SqlRejected, match="not permitted"):
        sql_guard.validate_sql("SELECT load_extension('evil.dll')")