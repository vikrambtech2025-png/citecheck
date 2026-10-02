"""Run state.

Persisted to SQLite so a run survives a process restart and can be replayed for
debugging. Keeping `fetched_sources` in state (not just in memory) is what makes
citation validation reproducible after the fact.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from citecheck.models.base import Message, ToolCall, Usage
from citecheck.tools.base import Source


def new_run_id(prefix: str = "run") -> str:
    """Time-ordered id, unique per call.

    The random suffix is not decoration: `runs.run_id` is a PRIMARY KEY and
    `StateStore.save` upserts, so two ids minted in the same millisecond would
    silently overwrite each other's persisted state. Millisecond resolution
    alone does not guarantee uniqueness.
    """
    return f"{prefix}_{int(time.time() * 1000):x}{secrets.token_hex(3)}"


@dataclass(slots=True)
class Budget:
    """Hard ceilings. The loop refuses to continue once one is exhausted."""

    max_steps: int = 12
    max_tool_calls: int = 40
    max_tokens: int = 120_000
    max_wall_clock_s: float = 300.0

    steps: int = 0
    tool_calls: int = 0
    tokens_used: int = 0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def check(self) -> str | None:
        """Return a reason string if the budget is exhausted, else None."""
        if self.steps >= self.max_steps:
            return f"step_budget_exhausted ({self.steps}/{self.max_steps})"
        if self.tool_calls >= self.max_tool_calls:
            return f"tool_call_budget_exhausted ({self.tool_calls}/{self.max_tool_calls})"
        if self.tokens_used >= self.max_tokens:
            return f"token_budget_exhausted ({self.tokens_used}/{self.max_tokens})"
        if self.elapsed >= self.max_wall_clock_s:
            return f"wall_clock_budget_exhausted ({self.elapsed:.0f}s/{self.max_wall_clock_s:.0f}s)"
        return None

    def snapshot(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "tokens_used": self.tokens_used,
            "elapsed_s": round(self.elapsed, 2),
            "limits": {
                "max_steps": self.max_steps,
                "max_tool_calls": self.max_tool_calls,
                "max_tokens": self.max_tokens,
                "max_wall_clock_s": self.max_wall_clock_s,
            },
        }


@dataclass(slots=True)
class StepRecord:
    """One think-act-observe cycle, for tracing and for eval metrics."""

    index: int
    reasoning: str
    tool_calls: list[tuple[str, str]] = field(default_factory=list)  # (name, args)
    results: list[dict[str, Any]] = field(default_factory=list)
    tokens: int = 0
    latency_s: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StepRecord:
        # tool_calls round-trips as lists: JSON has no tuples.
        return cls(
            index=d.get("index", 0),
            reasoning=d.get("reasoning", ""),
            tool_calls=[(c[0], c[1]) for c in d.get("tool_calls") or []],
            results=d.get("results") or [],
            tokens=d.get("tokens", 0),
            latency_s=d.get("latency_s", 0.0),
            error=d.get("error"),
        )


@dataclass(slots=True)
class AgentState:
    run_id: str
    task: str
    messages: list[Message] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    fetched_sources: dict[str, Source] = field(default_factory=dict)
    steps: list[StepRecord] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    budget: Budget = field(default_factory=Budget)
    final_answer: str | None = None
    validation: dict[str, Any] | None = None
    stop_reason: str | None = None
    #: Sources whose text tripped the injection scanner, with the matched spans.
    injection_flags: list[dict[str, Any]] = field(default_factory=list)
    denied_tool_calls: list[dict[str, Any]] = field(default_factory=list)

    # -- source registration ----------------------------------------------
    def register_source(self, source: Source) -> None:
        self.fetched_sources[source.id] = source
        self.findings.append({
            "source_id": source.id,
            "uri": source.uri,
            "title": source.title,
            "kind": source.kind,
        })

    # -- persistence -------------------------------------------------------
    @property
    def quarantined(self) -> frozenset[str]:
        """Source ids whose text tripped the prompt-injection scanner.

        Derived rather than stored, so the two can never disagree: the
        validator is handed exactly the ids the loop actually flagged.
        """
        return frozenset(f["source_id"] for f in self.injection_flags)

    def to_json(self) -> str:
        return json.dumps(
            {
                "run_id": self.run_id,
                "task": self.task,
                "messages": [m.to_dict() for m in self.messages],
                "plan": self.plan,
                "findings": self.findings,
                "fetched_sources": {
                    k: {
                        "id": v.id, "uri": v.uri, "title": v.title,
                        "text": v.text, "kind": v.kind, "content_hash": v.content_hash,
                    }
                    for k, v in self.fetched_sources.items()
                },
                "steps": [s.to_dict() for s in self.steps],
                "usage": asdict(self.usage),
                "budget": self.budget.snapshot(),
                "final_answer": self.final_answer,
                "validation": self.validation,
                "stop_reason": self.stop_reason,
                "injection_flags": self.injection_flags,
                "denied_tool_calls": self.denied_tool_calls,
            },
            default=str,
        )

    @classmethod
    def from_json(cls, blob: str) -> AgentState:
        d = json.loads(blob)
        st = cls(run_id=d["run_id"], task=d["task"])
        st.messages = [_message_from_dict(m) for m in d.get("messages", [])]
        st.plan = d.get("plan", [])
        st.findings = d.get("findings", [])
        # Steps must be restored: replaying a run after the fact is the whole
        # point of persisting state, and a run with no step history is
        # undebuggable.
        st.steps = [StepRecord.from_dict(s) for s in d.get("steps") or []]
        st.fetched_sources = {
            k: Source(**v) for k, v in (d.get("fetched_sources") or {}).items()
        }
        st.usage = Usage(**d.get("usage", {}))
        b = d.get("budget", {})
        st.budget = Budget(
            max_steps=b.get("limits", {}).get("max_steps", 12),
            max_tool_calls=b.get("limits", {}).get("max_tool_calls", 40),
            max_tokens=b.get("limits", {}).get("max_tokens", 120_000),
            max_wall_clock_s=b.get("limits", {}).get("max_wall_clock_s", 300.0),
            steps=b.get("steps", 0),
            tool_calls=b.get("tool_calls", 0),
            tokens_used=b.get("tokens_used", 0),
        )
        st.final_answer = d.get("final_answer")
        st.validation = d.get("validation")
        st.stop_reason = d.get("stop_reason")
        st.injection_flags = d.get("injection_flags", [])
        st.denied_tool_calls = d.get("denied_tool_calls", [])
        return st


def _message_from_dict(d: dict[str, Any]) -> Message:
    calls = [
        ToolCall(id=tc["id"], name=tc["function"]["name"],
                 arguments=json.loads(tc["function"].get("arguments") or "{}"))
        for tc in d.get("tool_calls") or []
    ]
    return Message(
        role=d["role"], content=d.get("content", ""),
        tool_calls=calls, tool_call_id=d.get("tool_call_id"), name=d.get("name"),
    )


class StateStore:
    """Tiny SQLite-backed state store, one row per run."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(
                """CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    status TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    state TEXT NOT NULL
                )"""
            )

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, state: AgentState, status: str = "running") -> None:
        with self._conn() as c:
            c.execute(
                """INSERT INTO runs (run_id, task, status, updated_at, state)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(run_id) DO UPDATE SET
                     status=excluded.status,
                     updated_at=excluded.updated_at,
                     state=excluded.state""",
                (state.run_id, state.task, status, time.time(), state.to_json()),
            )

    def load(self, run_id: str) -> AgentState | None:
        with self._conn() as c:
            row = c.execute("SELECT state FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return AgentState.from_json(row["state"]) if row else None

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT run_id, task, status, updated_at FROM runs ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


def fingerprint(text: str, length: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:length]