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


def test_select_is_allowed(tmp_path: Path) -> None:
    result = sql_guard.run_query(_db(tmp_path), "SELECT title FROM papers", max_rows=10)
    assert result["row_count"] == 6
    assert result["columns"] == ["title"]


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