"""FastAPI surface: run the agent, inspect traces, answer approval prompts.

The approval queue is the reason this exists rather than a plain CLI. Destructive
tools deny by default, so a run that reaches one produces a pending request the
operator must resolve over HTTP. That is the human-in-the-loop path, and it is
what a production deployment needs in order to run unattended.

    citecheck serve      # http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from citecheck.agent.loop import Agent
from citecheck.config import Settings, get_settings
from citecheck.models.clients import build_client
from citecheck.tools.registry import build_registry
from citecheck.tracing import build_tracer

log = logging.getLogger(__name__)

#: A pending approval is auto-denied after this long. Failing closed matters: an
#: operator who walks away must not leave a destructive call running.
APPROVAL_TIMEOUT_S = 120.0


@dataclass(slots=True)
class PendingApproval:
    id: str
    run_id: str
    tool: str
    arguments: dict[str, Any]
    created_at: float
    resolved: bool = False
    approved: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "tool": self.tool,
            "arguments": self.arguments,
            "created_at": self.created_at,
            "resolved": self.resolved,
            "approved": self.approved,
            "reason": self.reason,
        }


class ApprovalBroker:
    """Bridges the agent's synchronous approval callback to async HTTP resolution.

    The agent loop runs in a worker thread and calls `request` synchronously; the
    HTTP handler resolves the future from the event loop. That mismatch is handled
    with `asyncio.run_coroutine_threadsafe` on a single shared loop.
    """

    def __init__(self) -> None:
        self._pending: dict[str, PendingApproval] = {}
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def request(self, run_id: str, tool: str, arguments: dict[str, Any]) -> bool:
        """Called from the agent thread. Blocks until approved or timed out."""
        with self._lock:
            existing = next(
                (p for p in self._pending.values()
                 if not p.resolved and p.run_id == run_id and p.tool == tool),
                None,
            )
        # The same tool reached twice in one run (e.g. a retry) reuses the
        # original decision rather than prompting the operator again.
        if existing is not None and existing.resolved:
            return existing.approved

        item = PendingApproval(
            id=uuid.uuid4().hex[:12], run_id=run_id, tool=tool,
            arguments=arguments, created_at=time.time(),
        )
        with self._lock:
            self._pending[item.id] = item

        decision = threading.Event()

        def resolve(approved: bool, reason: str) -> None:
            item.resolved = True
            item.approved = approved
            item.reason = reason
            decision.set()

        _LOOP_CALLBACKS[run_id] = resolve
        log.info("approval requested: %s %s", item.id, tool)

        if not decision.wait(timeout=APPROVAL_TIMEOUT_S):
            with self._lock:
                item.resolved = True
                item.approved = False
                item.reason = "auto-denied: no operator response within timeout"
            log.warning("approval %s timed out; denied", item.id)
            return False
        return item.approved

    def resolve(self, approval_id: str, approved: bool, reason: str = "") -> dict[str, Any]:
        with self._lock:
            item = self._pending.get(approval_id)
        if item is None:
            raise KeyError(approval_id)
        if item.resolved:
            return item.to_dict()
        callback = _LOOP_CALLBACKS.pop(item.run_id, None)
        if callback is None:
            raise KeyError(f"no waiter for approval {approval_id}")
        callback(approved, reason)
        return item.to_dict()

    def pending(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                p.to_dict() for p in self._pending.values()
                if not p.resolved and p.created_at > time.time() - APPROVAL_TIMEOUT_S
            ]


# Per-run resolver, kept module-level so the agent thread can reach it without
# threading the broker through `ToolContext`.
_LOOP_CALLBACKS: dict[str, Callable[[bool, str], None]] = {}


# --------------------------------------------------------------------------
# request/response models
# --------------------------------------------------------------------------
class RunRequest(BaseModel):
    task: str = Field(..., min_length=3, description="The research question.")
    max_steps: int | None = Field(None, ge=1, le=40)
    architecture: str = "react"
    save_as: str | None = None


class RunSummary(BaseModel):
    run_id: str
    task: str
    final_answer: str | None = None
    validation: dict[str, Any] | None = None
    stop_reason: str
    usage: dict[str, Any] = Field(default_factory=dict)
    budget: dict[str, Any] = Field(default_factory=dict)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    injection_flags: list[dict[str, Any]] = Field(default_factory=list)
    denied_tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class ApprovalDecision(BaseModel):
    approved: bool
    reason: str = ""


# --------------------------------------------------------------------------
def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or get_settings()
    s.ensure_dirs()
    broker = ApprovalBroker()
    tracer = build_tracer(s)
    app = FastAPI(
        title="CiteCheck",
        version="0.1.0",
        description="Research agent with programmatically verified citations.",
    )
    app.state.settings = s
    app.state.broker = broker
    app.state.tracer = tracer

    @app.on_event("startup")
    async def _startup() -> None:
        broker.bind_loop(asyncio.get_running_loop())

    # -- runs --------------------------------------------------------------
    @app.post("/runs", response_model=RunSummary, tags=["runs"])
    async def create_run(req: RunRequest) -> RunSummary:
        """Execute the agent.

        Run in a worker thread: the loop is synchronous and blocks on the model,
        which would otherwise stall the event loop.
        """
        client = build_client(s)
        ok, detail = client.health_check() if hasattr(client, "health_check") else (True, "")
        if not ok:
            raise HTTPException(status_code=503, detail=f"model unavailable: {detail}")

        registry = build_registry(source_provider=lambda: {})
        agent = Agent(
            client, settings=s, registry=registry,
            architecture=req.architecture, tracer=tracer,
            on_event=lambda e: _record(app, e),
        )

        def _run() -> RunSummary:
            result = agent.run(
                req.task,
                approval_required=broker.request,
                max_steps=req.max_steps or s.max_steps,
            )
            if req.save_as and result.final_answer:
                _write_outbox(s, req.save_as, result.final_answer)
            return RunSummary(**_summary_fields(result))

        try:
            return await asyncio.to_thread(_run)
        except Exception as e:
            log.exception("run failed")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/runs", tags=["runs"])
    async def list_runs(limit: int = 25) -> list[dict[str, Any]]:
        from citecheck.agent.state import StateStore

        return StateStore(s.state_db_path).list_runs(limit)

    @app.get("/runs/{run_id}", tags=["runs"])
    async def get_run(run_id: str) -> dict[str, Any]:
        from citecheck.agent.state import StateStore

        state = StateStore(s.state_db_path).load(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"no such run: {run_id}")
        import json

        return json.loads(state.to_json())

    # -- approvals ---------------------------------------------------------
    @app.get("/approvals", tags=["approvals"])
    async def list_approvals() -> list[dict[str, Any]]:
        """Destructive-tool requests awaiting a human decision."""
        return broker.pending()

    @app.post("/approvals/{approval_id}", tags=["approvals"])
    async def decide(approval_id: str, decision: ApprovalDecision) -> dict[str, Any]:
        try:
            return broker.resolve(approval_id, decision.approved, decision.reason)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    # -- tools & health ----------------------------------------------------
    @app.get("/tools", tags=["introspection"])
    async def list_tools() -> list[dict[str, Any]]:
        registry = build_registry()
        return [
            {"name": t.name, "risk": t.risk, "description": t.description,
             "parameters": t.parameters}
            for t in registry.tools
        ]

    @app.get("/health", tags=["introspection"])
    async def health() -> dict[str, Any]:
        client = build_client(s)
        ok, detail = client.health_check() if hasattr(client, "health_check") else (True, "")
        return {
            "status": "ok" if ok else "degraded",
            "model_backend": s.model_backend,
            "model": client.model_name,
            "model_reachable": ok,
            "model_detail": detail,
            "tracing": getattr(tracer, "name", "none"),
            "workspace": str(s.workspace_dir),
            "outbox": str(s.outbox_dir),
        }

    @app.get("/traces", tags=["introspection"])
    async def traces() -> dict[str, Any]:
        spans = getattr(tracer, "spans", [])
        runs = getattr(tracer, "runs", [])
        return {"spans": spans[-200:], "runs": runs[-50:]}

    # -- artifacts ---------------------------------------------------------
    @app.get("/outbox", tags=["artifacts"])
    async def outbox() -> list[dict[str, Any]]:
        if not s.outbox_dir.exists():
            return []
        return [
            {"name": p.name, "size": p.stat().st_size,
             "modified": p.stat().st_mtime}
            for p in sorted(s.outbox_dir.glob("*.md"))
        ]

    @app.get("/outbox/{name}", tags=["artifacts"])
    async def read_outbox(name: str) -> dict[str, str]:
        target = (s.outbox_dir / name).resolve()
        # Same containment rule as the file tool: the API must not be a bypass.
        try:
            target.relative_to(s.outbox_dir.resolve())
        except ValueError as e:
            raise HTTPException(status_code=400, detail="path escapes outbox") from e
        if not target.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return {"name": target.name, "content": target.read_text(encoding="utf-8")}

    return app


# --------------------------------------------------------------------------
def _summary_fields(result: Any) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "task": result.task,
        "final_answer": result.final_answer,
        "validation": result.validation,
        "stop_reason": result.stop_reason,
        "usage": result.usage,
        "budget": result.budget,
        "sources": result.sources,
        "injection_flags": result.injection_flags,
        "denied_tool_calls": result.denied_tool_calls,
    }


def _write_outbox(s: Settings, name: str, content: str) -> None:
    from citecheck.tools.sandbox import safe_write

    safe_write(name, s.outbox_dir, content, max_bytes=s.max_file_bytes)


def _record(app: FastAPI, event: dict[str, Any]) -> None:
    events = getattr(app.state, "events", None)
    if events is None:
        events = []
        app.state.events = events
    events.append({**event, "t": time.time()})
    del events[:-1000]


#: Module-level ASGI app, so `uvicorn citecheck.api:app` works directly.
#: The broker binds its event loop in the startup hook.
app = create_app()


def serve_main(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


__all__ = ["ApprovalBroker", "PendingApproval", "app", "create_app", "serve_main"]