"""Run state, the budget ceiling, and the SQLite state store.

The round-trip tests matter most: state is only useful for debugging if it
survives a round trip intact. Losing a field silently makes a persisted run
look like it did less work than it actually did.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import make_source

from citecheck.agent.state import (
    AgentState,
    Budget,
    StateStore,
    StepRecord,
    fingerprint,
    new_run_id,
)
from citecheck.config import Settings
from citecheck.models.base import Message, ToolCall, Usage


# --------------------------------------------------------------------------
# budget
# --------------------------------------------------------------------------
def test_budget_starts_under_every_limit() -> None:
    assert Budget().check() is None


def test_budget_reports_step_exhaustion() -> None:
    b = Budget(max_steps=2)
    b.steps = 2
    assert "step_budget_exhausted" in b.check()


def test_budget_reports_tool_call_exhaustion() -> None:
    b = Budget(max_tool_calls=3)
    b.tool_calls = 3
    assert "tool_call_budget_exhausted" in b.check()


def test_budget_reports_token_exhaustion() -> None:
    b = Budget(max_tokens=1000)
    b.tokens_used = 1000
    assert "token_budget_exhausted" in b.check()


def test_budget_reports_wall_clock_exhaustion() -> None:
    b = Budget(max_wall_clock_s=0.0)
    assert "wall_clock_budget_exhausted" in b.check()


def test_budget_snapshot_reports_counters_and_limits() -> None:
    b = Budget(max_steps=5)
    b.steps = 1
    b.tool_calls = 2
    b.tokens_used = 30
    snap = b.snapshot()
    assert snap["steps"] == 1 and snap["tool_calls"] == 2 and snap["tokens_used"] == 30
    assert snap["limits"]["max_steps"] == 5


# --------------------------------------------------------------------------
# state round trip
# --------------------------------------------------------------------------
def _rich_state() -> AgentState:
    st = AgentState(run_id="run_abc", task="What is attention?")
    st.messages = [
        Message.system("you are a researcher"),
        Message.user("what is attention?"),
        Message.assistant(tool_calls=[ToolCall(id="c1", name="query_sqlite",
                                               arguments={"sql": "SELECT 1"})]),
        Message.tool_result(
            ToolCall(id="c1", name="query_sqlite", arguments={}), {"rows": [{"n": 1}]}
        ),
    ]
    st.plan = ["look it up", "cite it"]
    st.register_source(make_source("Sabc1234"))
    st.steps = [StepRecord(
        index=1, reasoning="I should query the database",
        tool_calls=[("query_sqlite", '{"sql": "SELECT 1"}')],
        results=[{"ok": True}], tokens=120, latency_s=1.5,
    )]
    st.usage = Usage(input_tokens=100, output_tokens=20)
    st.final_answer = "Attention is the core mechanism [Sabc1234]."
    st.validation = {"ok": True, "precision": 1.0, "coverage": 0.8}
    st.stop_reason = "finalized"
    st.injection_flags = [{"source_id": "Sbad", "matches": ["ignore all previous"]}]
    st.denied_tool_calls = [{"tool": "delete_file", "reason": "destructive"}]
    return st


def test_state_round_trips_through_json() -> None:
    original = _rich_state()
    restored = AgentState.from_json(original.to_json())

    assert restored.run_id == original.run_id
    assert restored.task == original.task
    assert restored.final_answer == original.final_answer
    assert restored.stop_reason == original.stop_reason
    assert restored.validation == original.validation
    assert restored.plan == original.plan


def test_state_round_trip_preserves_messages_and_tool_calls() -> None:
    original = _rich_state()
    restored = AgentState.from_json(original.to_json())

    assert [m.role for m in restored.messages] == ["system", "user", "assistant", "tool"]
    assert restored.messages[2].tool_calls[0].name == "query_sqlite"
    # Arguments were JSON-encoded on the way out and must come back as a dict.
    assert restored.messages[2].tool_calls[0].arguments == {"sql": "SELECT 1"}
    assert restored.messages[3].tool_call_id == "c1"


def test_state_round_trip_preserves_sources() -> None:
    restored = AgentState.from_json(_rich_state().to_json())
    src = restored.fetched_sources["Sabc1234"]
    assert src.uri == "https://example.com/transformer"
    assert src.kind == "web"
    assert src.text  # the body must survive, it is what validation re-checks


def test_state_round_trip_preserves_steps() -> None:
    restored = AgentState.from_json(_rich_state().to_json())
    assert len(restored.steps) == 1
    step = restored.steps[0]
    assert step.reasoning == "I should query the database"
    # tool_calls were tuples in memory and must come back as pairs, not lists.
    assert step.tool_calls == [("query_sqlite", '{"sql": "SELECT 1"}')]
    assert step.tokens == 120


def test_state_round_trip_preserves_guardrail_events() -> None:
    restored = AgentState.from_json(_rich_state().to_json())
    assert restored.injection_flags[0]["source_id"] == "Sbad"
    assert restored.denied_tool_calls[0]["tool"] == "delete_file"


def test_state_round_trip_restores_budget_limits() -> None:
    st = _rich_state()
    st.budget = Budget(max_steps=7, max_tool_calls=9, max_tokens=11, max_wall_clock_s=13.0)
    st.budget.steps = 3
    restored = AgentState.from_json(st.to_json())
    assert restored.budget.max_steps == 7
    assert restored.budget.max_tool_calls == 9
    assert restored.budget.steps == 3


def test_empty_state_round_trips() -> None:
    restored = AgentState.from_json(AgentState(run_id="r", task="t").to_json())
    assert restored.steps == [] and restored.messages == []


# --------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------
def test_store_round_trips_a_run(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    store.save(_rich_state(), status="done")

    loaded = store.load("run_abc")
    assert loaded is not None
    assert loaded.final_answer == "Attention is the core mechanism [Sabc1234]."
    assert loaded.stop_reason == "finalized"
    assert len(loaded.steps) == 1


def test_store_load_returns_none_for_unknown_run(tmp_path: Path) -> None:
    assert StateStore(tmp_path / "state.sqlite3").load("nope") is None


def test_store_save_upserts_rather_than_duplicating(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    st = _rich_state()
    store.save(st, status="running")
    st.final_answer = "updated"
    store.save(st, status="done")

    assert len(store.list_runs()) == 1
    assert store.load("run_abc").final_answer == "updated"


def test_store_lists_newest_first(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    for i in range(3):
        store.save(AgentState(run_id=f"r{i}", task=f"t{i}"), status="done")
    assert [r["run_id"] for r in store.list_runs()]
    assert len(store.list_runs()) == 3


def test_store_respects_limit(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state.sqlite3")
    for i in range(5):
        store.save(AgentState(run_id=f"r{i}", task="t"), status="done")
    assert len(store.list_runs(limit=2)) == 2


def test_store_creates_parent_directories(tmp_path: Path) -> None:
    db = tmp_path / "deeply" / "nested" / "state.sqlite3"
    StateStore(db)
    assert db.parent.is_dir()


# --------------------------------------------------------------------------
# ids
# --------------------------------------------------------------------------
def test_run_ids_are_unique() -> None:
    assert new_run_id() != new_run_id()


def test_run_id_carries_the_prefix() -> None:
    assert new_run_id("eval").startswith("eval_")


def test_fingerprint_is_stable_and_short() -> None:
    a = fingerprint("some content")
    assert a == fingerprint("some content")
    assert len(a) == 12
    assert a != fingerprint("other content")


# --------------------------------------------------------------------------
# source registration
# --------------------------------------------------------------------------
def test_quarantine_is_derived_from_injection_flags() -> None:
    """Derived, not stored, so it can never disagree with what was flagged."""
    st = AgentState(run_id="r", task="t")
    st.register_source(make_source("Sclean1234"))
    assert st.quarantined == frozenset()

    st.injection_flags.append({"source_id": "Sbad12345", "risk_score": 9})
    assert st.quarantined == frozenset({"Sbad12345"})


def test_register_source_records_a_finding() -> None:
    st = AgentState(run_id="r", task="t")
    st.register_source(make_source("Sabc1234"))
    assert st.findings[0]["source_id"] == "Sabc1234"
    assert st.findings[0]["kind"] == "web"


def test_registering_the_same_id_twice_does_not_duplicate_the_source() -> None:
    st = AgentState(run_id="r", task="t")
    st.register_source(make_source("Sabc1234"))
    st.register_source(make_source("Sabc1234"))
    assert len(st.fetched_sources) == 1


def test_state_starts_empty(settings: Settings) -> None:
    st = AgentState(run_id="r", task="t")
    assert st.fetched_sources == {} and st.usage.total == 0
