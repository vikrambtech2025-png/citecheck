"""Eval harness: runs the task suite and produces report tables.

Deliberately separates *running* tasks from *reporting* them, so `--bench`
(model sweep) and `--arch` (architecture ablation) reuse the same execution and
grading path and differ only in what they vary.

Cost control, since a 7B local model on a laptop is not free in wall clock:
  - `--limit N` truncates the suite
  - `--categories` narrows it
  - `--offline-only` drops network-dependent tasks
  - `--max-steps` tightens the per-run budget
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from citecheck.agent.loop import Agent
from citecheck.config import REPO_ROOT, Settings, get_settings
from citecheck.eval.graders import Grade, aggregate, grade
from citecheck.eval.tasks import Task
from citecheck.models.clients import ModelClient, build_client
from citecheck.tools.registry import build_registry
from citecheck.tracing import build_tracer

log = logging.getLogger(__name__)

RESULTS_DIR = REPO_ROOT / "eval" / "results"


@dataclass(slots=True)
class TaskOutcome:
    task_id: str
    category: str
    grader: str
    prompt: str
    grade: Grade
    stop_reason: str
    steps: int
    tokens: int
    latency_s: float
    wall_s: float
    sources: list[str] = field(default_factory=list)
    validation: dict[str, Any] | None = None
    injection_flags: int = 0
    denied_calls: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "grader": self.grader,
            "prompt": self.prompt,
            "grade": self.grade.to_dict(),
            "stop_reason": self.stop_reason,
            "steps": self.steps,
            "tokens": self.tokens,
            "latency_s": round(self.latency_s, 2),
            "wall_s": round(self.wall_s, 2),
            "sources": self.sources,
            "validation": self.validation,
            "injection_flags": self.injection_flags,
            "denied_calls": self.denied_calls,
            "error": self.error,
        }


@dataclass(slots=True)
class SuiteReport:
    label: str
    model: str
    architecture: str
    outcomes: list[TaskOutcome]
    summary: dict[str, Any]
    started_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "model": self.model,
            "architecture": self.architecture,
            "started_at": self.started_at,
            "summary": self.summary,
            "by_category": self.by_category(),
            "outcomes": [o.to_dict() for o in self.outcomes],
        }

    def by_category(self) -> dict[str, dict[str, Any]]:
        buckets: dict[str, list[TaskOutcome]] = {}
        for o in self.outcomes:
            buckets.setdefault(o.category, []).append(o)
        return {
            cat: {
                "n": len(items),
                "passed": sum(1 for i in items if i.grade.passed),
                "pass_rate": sum(1 for i in items if i.grade.passed) / len(items),
                "mean_score": statistics.fmean(i.grade.score for i in items),
            }
            for cat, items in sorted(buckets.items())
        }


# --------------------------------------------------------------------------
class Harness:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        tracer: Any = None,
        approval_required: Any = None,
    ) -> None:
        self.s = settings or get_settings()
        self.tracer = tracer if tracer is not None else build_tracer(self.s)
        self.approval_required = approval_required or _deny_all

    def run_task(
        self,
        task: Task,
        client: ModelClient,
        *,
        architecture: str = "react",
        max_steps: int | None = None,
    ) -> TaskOutcome:
        start = time.perf_counter()
        # A fresh registry per task: source state must not leak between tasks,
        # or citation validation would accept another task's documents.
        registry = build_registry(source_provider=lambda: {})

        agent = Agent(
            client,
            settings=self.s,
            registry=registry,
            architecture=architecture,
            tracer=self.tracer,
        )
        try:
            result = agent.run(
                task.prompt,
                approval_required=self.approval_required,
                max_steps=max_steps or self.s.max_steps,
            )
        except Exception as e:
            log.warning("task %s raised: %s", task.id, e)
            return TaskOutcome(
                task_id=task.id, category=task.category, grader=task.grader,
                prompt=task.prompt,
                grade=Grade(False, 0.0, task.grader, failure_reason=f"error: {e}"),
                stop_reason="exception", steps=0, tokens=0,
                latency_s=0.0, wall_s=time.perf_counter() - start,
                error=f"{type(e).__name__}: {e}",
            )

        wall = time.perf_counter() - start
        g = grade(task, result)
        return TaskOutcome(
            task_id=task.id,
            category=task.category,
            grader=task.grader,
            prompt=task.prompt,
            grade=g,
            stop_reason=result.stop_reason,
            steps=len(result.steps),
            tokens=result.usage.get("total_tokens", 0),
            latency_s=sum(s.latency_s for s in result.steps),
            wall_s=wall,
            sources=[s["id"] for s in result.sources],
            validation=result.validation,
            injection_flags=len(result.injection_flags),
            denied_calls=len(result.denied_tool_calls),
        )

    def run_suite(
        self,
        tasks: list[Task],
        client: ModelClient,
        *,
        label: str = "core",
        architecture: str = "react",
        max_steps: int | None = None,
        on_task: Any = None,
    ) -> SuiteReport:
        started = time.time()
        outcomes: list[TaskOutcome] = []
        log.info("running %d tasks with %s (%s)", len(tasks), client.model_name, architecture)

        for i, task in enumerate(tasks, 1):
            log.info("[%d/%d] %s %s", i, len(tasks), task.id, task.prompt[:70])
            outcome = self.run_task(task, client, architecture=architecture, max_steps=max_steps)
            outcomes.append(outcome)
            if on_task is not None:
                on_task(i, len(tasks), outcome)
            log.info(
                "    -> %s (score %.2f, %d steps, %.1fs) %s",
                "PASS" if outcome.grade.passed else "FAIL",
                outcome.grade.score, outcome.steps, outcome.wall_s,
                outcome.grade.failure_reason or "",
            )

        results = [o.to_dict() for o in outcomes]
        summary = aggregate([o.grade for o in outcomes], results)
        return SuiteReport(
            label=label, model=client.model_name, architecture=architecture,
            outcomes=outcomes, summary=summary, started_at=started,
        )


# --------------------------------------------------------------------------
# sweeps
# --------------------------------------------------------------------------
def run_model_sweep(
    models: list[str],
    tasks: list[Task],
    settings: Settings | None = None,
    *,
    architecture: str = "react",
    max_steps: int | None = None,
    on_report: Any = None,
) -> list[SuiteReport]:
    """Table A: same tasks, different models."""
    s = settings or get_settings()
    harness = Harness(s)
    reports: list[SuiteReport] = []

    for model_name in models:
        client = build_client(s.model_copy(update={"ollama_model": model_name,
                                                  "model_backend": "ollama"}))
        ok, detail = client.health_check()
        if not ok:
            log.error("skipping %s: %s", model_name, detail)
            reports.append(SuiteReport(
                label=model_name, model=model_name, architecture=architecture,
                outcomes=[], summary={"error": detail, "n": 0}, started_at=time.time(),
            ))
            continue
        reports.append(harness.run_suite(
            tasks, client, label=model_name, architecture=architecture,
            max_steps=max_steps, on_task=on_report,
        ))
    return reports


def run_arch_sweep(
    architectures: list[str],
    tasks: list[Task],
    settings: Settings | None = None,
    *,
    on_report: Any = None,
) -> list[SuiteReport]:
    """Table B: same tasks, different agent architectures."""
    s = settings or get_settings()
    harness = Harness(s)
    client = build_client(s)
    return [
        harness.run_suite(tasks, client, label=arch, architecture=arch, on_task=on_report)
        for arch in architectures
    ]


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def save_report(report: SuiteReport, out_dir: Path | None = None) -> Path:
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(f"{report.label}_{report.architecture}")
    path = out_dir / f"{slug}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
    return path


def render_table(reports: list[SuiteReport]) -> str:
    """Markdown comparison table."""
    header = (
        "| Run | Model | Architecture | n | Pass rate | Score | Steps | Tokens | "
        "Cite precision | Cite coverage | Latency (s) |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = []
    for r in reports:
        s = r.summary
        if s.get("error"):
            rows.append(
                f"| {r.label} | {r.model} | {r.architecture} | - | ERROR: {s['error']} | | | | | | |"
            )
            continue
        rows.append(
            f"| {r.label} | {r.model} | {r.architecture} "
            f"| {s.get('n', 0)} "
            f"| {s.get('pass_rate', 0):.0%} "
            f"| {s.get('mean_score', 0):.2f} "
            f"| {s.get('mean_steps', 0):.1f} "
            f"| {s.get('mean_tokens', 0):,.0f} "
            f"| {s.get('citation_precision', 0):.2f} "
            f"| {s.get('citation_coverage', 0):.2f} "
            f"| {s.get('mean_latency_s', 0):.1f} |"
        )
    return header + "\n".join(rows)


def render_failures(report: SuiteReport, limit: int = 12) -> str:
    failed = [o for o in report.outcomes if not o.grade.passed]
    if not failed:
        return "No failures."
    lines = [f"## Failures ({len(failed)}/{len(report.outcomes)})", ""]
    for o in failed[:limit]:
        lines.append(f"- **{o.task_id}** [{o.category}] {o.grade.failure_reason or 'unspecified'}")
        lines.append(f"  - prompt: {o.prompt[:110]}")
        lines.append(f"  - stop: `{o.stop_reason}` steps: {o.steps}")
    if len(failed) > limit:
        lines.append(f"\n_...and {len(failed) - limit} more._")
    return "\n".join(lines)


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text).lower()


def _deny_all(_tool: str, _args: dict[str, Any]) -> bool:
    """Eval policy: destructive tools are never auto-approved.

    This is deliberate. An eval that auto-approves cannot measure whether the
    approval gate works, so the suite always records these as denied.
    """
    return False