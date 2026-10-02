"""Tool behaviour: registry dispatch, risk gating, and the finalize tool."""

from __future__ import annotations

import pytest
from tests.conftest import make_source

from citecheck.config import Settings
from citecheck.models.base import RISK_DESTRUCTIVE, RISK_READ, RISK_WRITE, ToolCall
from citecheck.tools.base import ToolContext
from citecheck.tools.registry import APPROVAL_TIERS, build_registry


@pytest.fixture
def ctx(seeded: Settings) -> ToolContext:
    return ToolContext(settings=seeded, run_id="test")


def _sources(*ids: str) -> dict:
    return {sid: make_source(sid, f"Content about transformers and attention for {sid}.")
            for sid in ids}


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
def test_registry_exposes_all_nine_tools(seeded: Settings) -> None:
    registry = build_registry()
    assert len(registry.tools) == 9
    assert {"finalize", "delete_file", "query_sqlite"} <= set(registry.names)


def test_schemas_are_well_formed(seeded: Settings) -> None:
    for schema in build_registry().schemas():
        fn = schema["function"]
        assert fn["name"] and fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        # Every property must itself be typed, or strict providers reject the call.
        for prop in (params.get("properties") or {}).values():
            assert "type" in prop, f"{fn['name']} has an untyped property"


def test_risk_tiers_are_assigned(seeded: Settings) -> None:
    risks = {t.name: t.risk for t in build_registry().tools}
    assert risks["query_sqlite"] == RISK_READ
    assert risks["fetch_url"] == RISK_READ
    assert risks["finalize"] == RISK_WRITE
    assert risks["delete_file"] == RISK_DESTRUCTIVE
    # Only the destructive tier should ever require a human decision.
    assert RISK_DESTRUCTIVE in APPROVAL_TIERS
    assert RISK_WRITE not in APPROVAL_TIERS


def test_finalize_is_write_tier(seeded: Settings) -> None:
    assert build_registry().get("finalize").risk == RISK_WRITE


def test_include_filter_restricts_the_surface(seeded: Settings) -> None:
    registry = build_registry(include=["query_sqlite", "finalize"])
    assert registry.names == ["finalize", "query_sqlite"]


def test_unknown_tool_reports_available_options(seeded: Settings) -> None:
    registry = build_registry()
    result = registry.dispatch(ToolCall(id="1", name="nope", arguments={}), ToolContext(settings=seeded))
    assert not result.ok
    assert "unknown_tool" in result.error
    assert "query_sqlite" in result.error  # tells the model what it could call


def test_tool_exception_is_caught_not_raised(seeded: Settings) -> None:
    registry = build_registry()
    # Bad arguments that would make the underlying implementation raise.
    result = registry.dispatch(
        ToolCall(id="1", name="read_file", arguments={"path": None}),
        ToolContext(settings=seeded),
    )
    assert not result.ok
    assert result.error


# --------------------------------------------------------------------------
# risk gating
# --------------------------------------------------------------------------
def test_destructive_tool_requires_approval(seeded: Settings) -> None:
    (seeded.workspace_dir / "victim.md").write_text("x", encoding="utf-8")
    registry = build_registry()
    call = ToolCall(id="1", name="delete_file",
                    arguments={"path": "victim.md", "reason": "cleanup"})

    denied = registry.dispatch(call, ToolContext(settings=seeded, approval_required=lambda *_: False))
    assert not denied.ok and denied.meta.get("denied") is True
    assert (seeded.workspace_dir / "victim.md").exists(), "file must survive a denial"

    approved = registry.dispatch(call, ToolContext(settings=seeded, approval_required=lambda *_: True))
    assert approved.ok
    assert not (seeded.workspace_dir / "victim.md").exists()


def test_read_tools_do_not_require_approval(seeded: Settings) -> None:
    registry = build_registry()
    calls: list[str] = []
    result = registry.dispatch(
        ToolCall(id="1", name="query_sqlite", arguments={"sql": "SELECT 1 AS a"}),
        ToolContext(settings=seeded, approval_required=lambda t, a: calls.append(t) or True),
    )
    assert result.ok
    assert calls == [], "read tools must not trigger an approval prompt"


def test_delete_refuses_directories(seeded: Settings) -> None:
    (seeded.workspace_dir / "sub").mkdir()
    registry = build_registry()
    result = registry.dispatch(
        ToolCall(id="1", name="delete_file", arguments={"path": "sub", "reason": "x"}),
        ToolContext(settings=seeded, approval_required=lambda *_: True),
    )
    assert not result.ok
    assert (seeded.workspace_dir / "sub").exists()


# --------------------------------------------------------------------------
# file tools
# --------------------------------------------------------------------------
def test_read_file_returns_a_citable_source(seeded: Settings, ctx: ToolContext) -> None:
    registry = build_registry()
    result = registry.dispatch(
        ToolCall(id="1", name="read_file", arguments={"path": "notes/project_notes.md"}), ctx
    )
    assert result.ok
    assert result.source is not None
    assert result.source.kind == "file"
    assert result.source.id in result.data["source_id"] or result.data["source_id"]


def test_read_file_cannot_escape_the_workspace(seeded: Settings, ctx: ToolContext) -> None:
    registry = build_registry()
    result = registry.dispatch(
        ToolCall(id="1", name="read_file", arguments={"path": "../../.env"}), ctx
    )
    assert not result.ok
    assert result.meta.get("blocked") is True
    assert "sandbox_violation" in result.error


def test_save_report_is_confined_to_outbox(seeded: Settings, ctx: ToolContext) -> None:
    registry = build_registry()
    result = registry.dispatch(
        ToolCall(id="1", name="save_report",
                 arguments={"path": "../../escaped.md", "markdown": "nope"}),
        ctx,
    )
    assert not result.ok
    assert result.meta.get("blocked") is True


def test_save_report_writes_inside_outbox(seeded: Settings, ctx: ToolContext) -> None:
    registry = build_registry()
    result = registry.dispatch(
        ToolCall(id="1", name="save_report",
                 arguments={"path": "report.md", "markdown": "# hello"}),
        ctx,
    )
    assert result.ok
    assert (seeded.outbox_dir / "report.md").read_text(encoding="utf-8") == "# hello"


def test_workspace_tree_lists_notes(seeded: Settings, ctx: ToolContext) -> None:
    registry = build_registry()
    result = registry.dispatch(ToolCall(id="1", name="workspace_tree", arguments={}), ctx)
    assert result.ok
    paths = {e["path"] for e in result.data["entries"]}
    assert "notes/project_notes.md" in paths
    assert "notes/injected_notes.md" in paths


# --------------------------------------------------------------------------
# sqlite tool
# --------------------------------------------------------------------------
def test_query_sqlite_returns_rows_and_a_source(seeded: Settings, ctx: ToolContext) -> None:
    registry = build_registry()
    result = registry.dispatch(
        ToolCall(id="1", name="query_sqlite", arguments={"sql": "SELECT COUNT(*) AS n FROM papers"}),
        ctx,
    )
    assert result.ok
    assert result.data["rows"][0]["n"] == 6
    assert result.source is not None and result.source.kind == "db"


def test_query_sqlite_rejects_writes_from_the_model(seeded: Settings, ctx: ToolContext) -> None:
    registry = build_registry()
    result = registry.dispatch(
        ToolCall(id="1", name="query_sqlite", arguments={"sql": "DELETE FROM papers"}), ctx
    )
    assert not result.ok
    assert result.meta.get("rejected") is True


# --------------------------------------------------------------------------
# finalize
# --------------------------------------------------------------------------
def test_finalize_accepts_a_valid_citation(seeded: Settings, ctx: ToolContext) -> None:
    ctx.sources = _sources("Sabc1234")
    registry = build_registry()
    result = registry.dispatch(
        ToolCall(id="1", name="finalize", arguments={
            "answer": "Attention is a core mechanism in transformers [Sabc1234]."}), ctx
    )
    assert result.ok
    assert result.meta["citation_passed"] is True
    assert result.data["validation"]["ok"] is True
    assert "## Sources" in result.data["answer"]


def test_finalize_rejects_a_fabricated_source_id(seeded: Settings, ctx: ToolContext) -> None:
    ctx.sources = _sources("Sabc1234")
    registry = build_registry()
    result = registry.dispatch(
        ToolCall(id="1", name="finalize", arguments={
            "answer": "Transformers use attention mechanisms [Shall9999]."}), ctx
    )
    assert not result.ok
    assert result.meta.get("citation_failed") is True
    assert "unknown_source" in result.error


def test_finalize_rejects_an_empty_answer(seeded: Settings, ctx: ToolContext) -> None:
    registry = build_registry()
    result = registry.dispatch(
        ToolCall(id="1", name="finalize", arguments={"answer": "   "}), ctx
    )
    assert not result.ok


def test_finalize_can_save_the_validated_report(seeded: Settings, ctx: ToolContext) -> None:
    ctx.sources = _sources("Sabc1234")
    registry = build_registry()
    result = registry.dispatch(
        ToolCall(id="1", name="finalize", arguments={
            "answer": "Attention is a core mechanism in transformers [Sabc1234].",
            "save_as": "saved.md"}),
        ctx,
    )
    assert result.ok
    assert (seeded.outbox_dir / "saved.md").exists()
    assert result.data["saved_to"].endswith("saved.md")


def test_finalize_rejects_an_uncited_answer(seeded: Settings, ctx: ToolContext) -> None:
    ctx.sources = _sources("Sabc1234")
    registry = build_registry()
    result = registry.dispatch(
        ToolCall(id="1", name="finalize", arguments={
            "answer": "Transformers use attention mechanisms because they scale well."}),
        ctx,
    )
    assert not result.ok
    assert "no_citations" in result.error