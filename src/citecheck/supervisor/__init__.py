"""Supervisor pattern: multi-stage research with a measurable delta."""

from citecheck.supervisor.runner import (
    Finding,
    Handoff,
    Supervisor,
    SupervisorResult,
    Verdict,
    compare_architectures,
    parse_findings,
    parse_verdicts,
    render_comparison,
)

__all__ = [
    "Finding",
    "Handoff",
    "Supervisor",
    "SupervisorResult",
    "Verdict",
    "compare_architectures",
    "parse_findings",
    "parse_verdicts",
    "render_comparison",
]