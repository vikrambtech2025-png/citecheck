"""Supervisor pattern: route a research task across specialised sub-agents.

Compare against the monolithic agent in [`agent/loop.py`]. The point of this
module is not that multi-agent is better -- it frequently is not -- but that the
difference is *measurable*. `compare_architectures` runs both against the same
tasks and reports the delta, which is a more useful claim than either.

Roles:
  - **researcher**  gathers sources, emits findings with citations. No writing.
  - **verifier**     reviews findings against the sources, drops unsupported ones.
  - **writer**       turns verified findings into the final report.

Handoffs are structured `Handoff` records rather than free text, so a sub-agent's
output is inspectable and the verifier cannot be fed a different claim than the
one that was gathered.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from citecheck.agent.loop import Agent, RunResult
from citecheck.config import Settings, get_settings
from citecheck.models.base import Message
from citecheck.models.clients import ModelClient, build_client
from citecheck.tools.registry import build_registry
from citecheck.tracing import build_tracer

log = logging.getLogger(__name__)

Role = Literal["researcher", "verifier", "writer"]

RESEARCHER_PROMPT = """\
You are the research stage. Gather evidence for the task using your tools.

Rules:
- Use tools. Do not answer from memory.
- For each finding, emit a line of the form:
  FINDING: <claim> || SOURCE: <source_id> || QUOTE: "<verbatim span from the source>"
- Only use source_ids you actually retrieved.
- Emit 1 to 6 FINDING lines. Then emit exactly: DONE
"""

VERIFIER_PROMPT = """\
You are the verification stage. You are given findings and the sources they claim.

For each finding, decide whether the quoted text actually supports the claim.
Emit one line per finding:
  VERDICT: <source_id> || SUPPORTED || <reason>
or
  VERDICT: <source_id> || REJECTED || <reason>
Keep reasons under 15 words. Then emit exactly: DONE
Reject anything whose QUOTE does not appear in the cited source.
"""

WRITER_PROMPT = """\
You are the writing stage. Produce the final report from verified findings only.

Rules:
- Use ONLY findings marked SUPPORTED. Never introduce a fact not in a finding.
- Every factual sentence ends with a citation marker: [SourceId]
- If a finding was rejected, do not mention its claim.
- Call finalize with the report.
"""


@dataclass(slots=True)
class Finding:
    claim: str
    source_id: str
    quote: str

    def to_dict(self) -> dict[str, str]:
        return {"claim": self.claim, "source_id": self.source_id, "quote": self.quote}


@dataclass(slots=True)
class Verdict:
    source_id: str
    supported: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"source_id": self.source_id, "supported": self.supported,
                "reason": self.reason}


@dataclass(slots=True)
class Handoff:
    """Structured transfer of work between stages."""

    frm: Role
    #: Stages hand to the next stage, or to "user" for the final return.
    to: Role | Literal["user"]
    payload: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"from": self.frm, "to": self.to, "note": self.note,
                "payload_keys": sorted(self.payload)}


@dataclass(slots=True)
class SupervisorResult:
    """Outcome of a supervised run, plus the intermediate artefacts."""

    run_id: str
    task: str
    final_answer: str | None
    validation: dict[str, Any] | None
    stop_reason: str
    findings: list[Finding] = field(default_factory=list)
    verdicts: list[Verdict] = field(default_factory=list)
    handoffs: list[Handoff] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    steps: dict[str, int] = field(default_factory=dict)

    @property
    def rejection_rate(self) -> float:
        if not self.verdicts:
            return 0.0
        return sum(1 for v in self.verdicts if not v.supported) / len(self.verdicts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "final_answer": self.final_answer,
            "validation": self.validation,
            "stop_reason": self.stop_reason,
            "findings": [f.to_dict() for f in self.findings],
            "verdicts": [v.to_dict() for v in self.verdicts],
            "handoffs": [h.to_dict() for h in self.handoffs],
            "usage": self.usage,
            "steps_per_stage": self.steps,
            "rejection_rate": round(self.rejection_rate, 3),
        }


# --------------------------------------------------------------------------
# parsing -- tolerant of small-model formatting
# --------------------------------------------------------------------------
_FINDING_RE = re.compile(
    r"FINDING\s*:?\s*(?P<claim>.+?)\s*\|\|\s*SOURCE\s*:?\s*(?P<sid>[A-Za-z0-9_]+)\s*"
    r"\|\|\s*QUOTE\s*:?\s*(?P<quote>.+?)(?=\n\s*FINDING|\n\s*DONE|$)",
    re.IGNORECASE | re.DOTALL,
)
_VERDICT_RE = re.compile(
    r"VERDICT\s*:?\s*(?P<sid>[A-Za-z0-9_]+)\s*\|\|\s*(?P<status>SUPPORTED|REJECTED)"
    r"\s*\|\|\s*(?P<reason>.+?)(?=\n\s*VERDICT|\n\s*DONE|$)",
    re.IGNORECASE | re.DOTALL,
)


def parse_findings(text: str) -> list[Finding]:
    out: list[Finding] = []
    for m in _FINDING_RE.finditer(text or ""):
        claim = m.group("claim").strip(" -*\t")
        quote = m.group("quote").strip().strip('"').strip()
        sid = m.group("sid").strip()
        if claim and sid:
            out.append(Finding(claim=claim, source_id=sid, quote=quote))
    return out[:12]


def parse_verdicts(text: str) -> list[Verdict]:
    out: list[Verdict] = []
    for m in _VERDICT_RE.finditer(text or ""):
        status = m.group("status").upper()
        out.append(Verdict(
            source_id=m.group("sid").strip(),
            supported=status == "SUPPORTED",
            reason=m.group("reason").strip()[:120],
        ))
    return out[:24]


# --------------------------------------------------------------------------
# supervisor
# --------------------------------------------------------------------------
class Supervisor:
    """Runs research → verify → write, each as a separate `Agent`."""

    def __init__(
        self,
        client: ModelClient,
        *,
        settings: Settings | None = None,
        tracer: Any = None,
        skip_verification: bool = False,
    ) -> None:
        self.s = settings or get_settings()
        self.client = client
        self.tracer = tracer if tracer is not None else build_tracer(self.s)
        self.skip_verification = skip_verification

    def run(self, task: str, *, max_steps: int = 8) -> SupervisorResult:
        usage = {"input_tokens": 0, "output_tokens": 0}
        handoffs: list[Handoff] = []
        steps_per_stage: dict[str, int] = {}

        # -- stage 1: research --------------------------------------------
        research = self._stage(
            RESEARCHER_PROMPT, task, tools=True, max_steps=max_steps,
            registry_fresh=True, usage=usage, stage="researcher",
        )
        steps_per_stage["researcher"] = len(research.steps)
        findings = parse_findings(research.final_answer or "")
        handoffs.append(Handoff("researcher", "verifier",
                                {"findings": len(findings)},
                                note="collected candidate findings with citations"))

        # -- stage 2: verify ----------------------------------------------
        verdicts: list[Verdict] = []
        if findings and not self.skip_verification:
            verify = self._stage(
                VERIFIER_PROMPT,
                _render_verification_input(findings),
                tools=False, max_steps=1, usage=usage, stage="verifier",
            )
            steps_per_stage["verifier"] = len(verify.steps)
            verdicts = parse_verdicts(verify.final_answer or "")
            handoffs.append(Handoff("verifier", "writer",
                                    {"verdicts": len(verdicts)},
                                    note="checked each finding against its quoted source"))
        else:
            steps_per_stage["verifier"] = 0
            verdicts = [Verdict(f.source_id, True, "verification skipped") for f in findings]
            handoffs.append(Handoff("verifier", "writer", {"verdicts": len(verdicts)},
                                    note="verification skipped"))

        # -- stage 3: write -----------------------------------------------
        write = self._stage(
            WRITER_PROMPT,
            _render_writing_input(task, findings, verdicts),
            tools=True, max_steps=4, registry_fresh=True, usage=usage, stage="writer",
        )
        steps_per_stage["writer"] = len(write.steps)
        handoffs.append(Handoff("writer", "user", {"answer": bool(write.final_answer)},
                                note="produced the final cited report"))

        return SupervisorResult(
            run_id=research.run_id,
            task=task,
            final_answer=write.final_answer,
            validation=write.validation,
            stop_reason=write.stop_reason,
            findings=findings,
            verdicts=verdicts,
            handoffs=handoffs,
            usage=usage,
            steps=steps_per_stage,
        )

    def _stage(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        tools: bool,
        max_steps: int,
        usage: dict[str, int],
        stage: str,
        registry_fresh: bool = False,
    ) -> RunResult:
        registry = (
            build_registry(source_provider=lambda: {})
            if (tools and registry_fresh)
            else (build_registry(source_provider=lambda: {}, include=["finalize"])
                  if tools else None)
        )
        agent = Agent(
            self.client,
            settings=self.s,
            registry=registry,
            tracer=self.tracer,
            system_prompt=system_prompt,
        )
        result = agent.run(user_prompt, approval_required=lambda *_: False, max_steps=max_steps)
        usage["input_tokens"] += result.usage.get("input_tokens", 0)
        usage["output_tokens"] += result.usage.get("output_tokens", 0)
        log.info("stage %s -> %s (%d steps)", stage, result.stop_reason, len(result.steps))
        return result


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def _render_verification_input(findings: list[Finding]) -> str:
    lines = ["Here are candidate findings. Verify each against its quoted source.", ""]
    for i, f in enumerate(findings, 1):
        lines.append(f"{i}. CLAIM: {f.claim}")
        lines.append(f"   SOURCE_ID: {f.source_id}")
        lines.append(f'   QUOTE: "{f.quote}"')
    lines.append("")
    lines.append("For each finding emit a VERDICT line, then DONE.")
    return "\n".join(lines)


def _render_writing_input(task: str, findings: list[Finding], verdicts: list[Verdict]) -> str:
    by_sid = {v.source_id: v for v in verdicts}
    lines = [f"Task: {task}", "", "Verified findings:"]
    kept = 0
    for f in findings:
        verdict = by_sid.get(f.source_id)
        if verdict is not None and not verdict.supported:
            lines.append(f"- REJECTED ({verdict.reason}): {f.claim}")
            continue
        kept += 1
        lines.append(f"- {f.claim} [{f.source_id}]")
    if not kept:
        lines.append(
            "- No finding survived verification. Say so explicitly and do not invent "
            "facts. Explain what you were unable to verify and why."
        )
    lines += ["", "Write the report citing each fact with its [SourceId]. Then call finalize."]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------
def compare_architectures(
    task: str,
    *,
    settings: Settings | None = None,
    max_steps: int = 8,
) -> dict[str, Any]:
    """Run the monolithic agent and the supervisor on the same task.

    This is the number the P4 milestone is for: whether splitting the work into
    stages improves citation quality enough to justify the extra model calls.
    """
    s = settings or get_settings()
    client = build_client(s)

    mono_agent = Agent(client, settings=s,
                       registry=build_registry(source_provider=lambda: {}))
    mono = mono_agent.run(task, approval_required=lambda *_: False, max_steps=max_steps)

    sup = Supervisor(client, settings=s, tracer=build_tracer(s))
    sup_result = sup.run(task, max_steps=max_steps)

    mono_val = mono.validation or {}
    sup_val = sup_result.validation or {}
    return {
        "task": task,
        "monolithic": {
            "stop_reason": mono.stop_reason,
            "steps": len(mono.steps),
            "tokens": mono.usage.get("total_tokens", 0),
            "citation_precision": mono_val.get("precision"),
            "citation_coverage": mono_val.get("coverage"),
            "validated": mono_val.get("ok", False),
        },
        "supervisor": {
            "stop_reason": sup_result.stop_reason,
            "steps": sum(sup_result.steps.values()),
            "steps_per_stage": sup_result.steps,
            "tokens": sup_result.usage.get("input_tokens", 0)
            + sup_result.usage.get("output_tokens", 0),
            "findings": len(sup_result.findings),
            "verdicts": len(sup_result.verdicts),
            "rejection_rate": round(sup_result.rejection_rate, 3),
            "citation_precision": sup_val.get("precision"),
            "citation_coverage": sup_val.get("coverage"),
            "validated": sup_val.get("ok", False),
        },
    }


def render_comparison(data: dict[str, Any]) -> str:
    m, s = data["monolithic"], data["supervisor"]
    rows = [
        ("Metric", "Monolithic", "Supervisor"),
        ("-" * 34, "-" * 22, "-" * 22),
        ("Total model steps", str(m["steps"]), str(s["steps"])),
        ("Total tokens", f"{m['tokens']:,}", f"{s['tokens']:,}"),
        ("Citations validated", str(m["validated"]), str(s["validated"])),
        ("Citation precision",
         _fmt(m["citation_precision"]), _fmt(s["citation_precision"])),
        ("Citation coverage",
         _fmt(m["citation_coverage"]), _fmt(s["citation_coverage"])),
        ("Findings rejected", "n/a", f"{s['rejection_rate']:.0%}"),
    ]
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def _fmt(v: Any) -> str:
    return "n/a" if v is None else f"{float(v):.2f}"


def _json_dump(data: Any) -> str:  # small helper for CLI output
    return json.dumps(data, indent=2, default=str)


def _msg(text: str) -> Message:  # re-exported for callers building ad-hoc stages
    return Message.user(text)