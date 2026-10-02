"""Tracing backends.

`NoopTracer` is the default so the package works with no observability stack
installed. `LangfuseTracer` and `OtelTracer` both satisfy the same tiny
interface, so `agent/loop.py` never knows which is active.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Literal

from citecheck.config import Settings, get_settings

log = logging.getLogger(__name__)


class Span:
    """Minimal span handle: a context manager that accumulates key/value data."""

    def __init__(self, tracer: Tracer, name: str) -> None:
        self._tracer = tracer
        self.name = name
        self.data: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []

    def set_data(self, **kwargs: Any) -> None:
        self.data.update(kwargs)

    def event(self, name: str, **kwargs: Any) -> None:
        self.events.append({"name": name, **kwargs})

    def __enter__(self) -> Span:
        return self

    def __exit__(self, *exc: Any) -> Literal[False]:
        self._tracer.finish(self)
        # Literal[False], not bool: a bool return type would let a reader
        # believe this context manager can swallow exceptions. It cannot.
        return False


class Tracer:
    """Interface: a span factory plus a run-level lifecycle."""

    name = "base"

    @contextmanager
    def span(self, name: str):
        s = Span(self, name)
        try:
            yield s
        finally:
            self.finish(s)

    def finish(self, span: Span) -> None:
        """Called when a span exits."""

    def start_run(self, run_id: str, task: str, **meta: Any) -> None:
        """Called once per agent run."""

    def end_run(self, result: Any) -> None:
        """Called once when a run completes."""


class NoopTracer(Tracer):
    """Collects spans in memory. Useful in tests and for the `/traces` endpoint."""

    name = "noop"

    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []
        self.runs: list[dict[str, Any]] = []

    def finish(self, span: Span) -> None:
        self.spans.append({"name": span.name, "data": span.data, "events": span.events})

    def start_run(self, run_id: str, task: str, **meta: Any) -> None:
        self.runs.append({"run_id": run_id, "task": task, **meta})

    def end_run(self, result: Any) -> None:
        if self.runs:
            self.runs[-1]["result"] = result.to_dict() if hasattr(result, "to_dict") else str(result)

    def clear(self) -> None:
        self.spans.clear()
        self.runs.clear()


class LangfuseTracer(Tracer):
    """Langfuse v2 adapter. Self-hosted via `docker compose -f docker/langfuse.yml`."""

    name = "langfuse"

    def __init__(self, settings: Settings | None = None) -> None:
        s = settings or get_settings()
        try:
            from langfuse import Langfuse
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "Install tracing with:  uv sync --extra tracing"
            ) from e
        if not s.langfuse_public_key:
            raise RuntimeError("CITECHECK_LANGFUSE_PUBLIC_KEY is not set")
        self.client = Langfuse(
            public_key=s.langfuse_public_key,
            secret_key=s.langfuse_secret_key,
            host=s.langfuse_host,
        )
        # Annotated: inferred as None from the assignment, which made every
        # `self._trace.update(...)` below an attribute error for a type checker.
        self._trace: Any | None = None

    def start_run(self, run_id: str, task: str, **meta: Any) -> None:

        self._trace = self.client.start_span(name="agent.run", metadata={"run_id": run_id, **meta})
        self._trace.update(input=task)

    def end_run(self, result: Any) -> None:
        if self._trace is not None:
            payload = result.to_dict() if hasattr(result, "to_dict") else {"result": str(result)}
            self._trace.update(output=payload.get("final_answer"))
            for key in ("usage", "stop_reason", "validation"):
                if payload.get(key) is not None:
                    self._trace.update_metadata({key: payload[key]})
            self._trace.end()
            self._trace = None

    @contextmanager
    def span(self, name: str):

        parent = self._trace
        handle = parent.span(name=name) if parent else self.client.start_span(name=name)
        wrapper = Span(self, name)
        try:
            yield wrapper
        finally:
            if wrapper.data:
                handle.update(metadata=wrapper.data)
            handle.end()


class OtelTracer(Tracer):
    """OpenTelemetry with GenAI semantic conventions."""

    name = "otel"

    def __init__(self, settings: Settings | None = None) -> None:
        s = settings or get_settings()
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        self._trace = trace
        provider = TracerProvider(
            resource=Resource.create({"service.name": "citecheck"})
        )
        endpoint = s.otel_otlp_endpoint
        if endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        self._provider = provider
        self.tracer = provider.get_tracer("citecheck.agent")

    @contextmanager
    def span(self, name: str):
        with self.tracer.start_as_current_span(name) as otel_span:
            wrapper = Span(self, name)
            try:
                yield wrapper
            finally:
                for k, v in wrapper.data.items():
                    try:
                        otel_span.set_attribute(f"gen_ai.{k}", v)
                    except (TypeError, ValueError):
                        otel_span.set_attribute(f"gen_ai.{k}", str(v))


def build_tracer(settings: Settings | None = None) -> Tracer:
    """Resolve `settings.tracing_backend`. Falls back to Noop on any failure."""
    s = settings or get_settings()
    backend = (s.tracing_backend or "none").lower()
    if backend == "none":
        return NoopTracer()
    try:
        if backend == "langfuse":
            return LangfuseTracer(s)
        if backend == "otel":
            return OtelTracer(s)
        log.warning("unknown tracing backend %r; using noop", backend)
    except Exception as e:
        log.warning("tracing unavailable (%s); continuing without it", e)
    return NoopTracer()