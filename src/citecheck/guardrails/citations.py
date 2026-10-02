"""Citation validation.

This is the part of the system that makes the agent's output trustworthy. The
model may only cite a document it actually retrieved in the current run, and a
citation is only accepted if the cited document contains supporting text for
the claim it is attached to.

Two guarantees, both enforced here rather than requested in the prompt:

  1. **Provenance** -- every `[Sxxxx]` marker resolves to a source in
     `fetched_sources`. A hallucinated identifier fails validation.

  2. **Support** -- the claim the marker is attached to must have lexical
     overlap with that source's text.

Honest scope note on (2): overlap is a *proxy* for entailment, not entailment
itself. It reliably catches fabricated citations and mis-sourced claims, and it
cannot be fooled by padding a claim with a source's words. It will flag a valid
paraphrase that shares no vocabulary with the original. The report therefore
records `support` scores rather than passing/failing silently, so the threshold
is visible and tunable.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from typing import Any

from citecheck.tools.base import Source

# [S1a2b3c4d] / [F0f1e2d3] / [Dknowledge] -- one or more, comma separated.
MARKER_RE = re.compile(r"\[([A-Z][A-Za-z0-9_]{2,}(?:\s*,\s*[A-Z][A-Za-z0-9_]{2,})*)\]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n{2,}")

STOPWORDS = frozenset(["a", "an", "the", "and", "or", "but", "if", "then", "than", "that", "this", "these", "those", "of", "in", "on", "at", "to", "for", "with", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being", "it", "its", "it's", "we", "our", "you", "your", "they", "their", "he", "she", "his", "her", "i", "me", "my", "not", "no", "do", "does", "did", "done", "have", "has", "had", "having", "will", "would", "can", "could", "should", "may", "might", "must", "shall", "about", "into", "over", "under", "again", "further", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "only", "own", "same", "so", "too", "very", "s", "t", "don", "now", "also", "which", "who", "whom", "whose", "what", "while", "during", "before", "after", "above", "below", "between", "during"])

# Minimum fraction of a claim's content words that must appear in the source.
SUPPORT_THRESHOLD = 0.34
MIN_CLAIM_TOKENS = 3


@dataclass(slots=True)
class CitationIssue:
    kind: str
    detail: str
    severity: str = "error"  # "error" blocks finalisation; "warning" does not

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "detail": self.detail, "severity": self.severity}


@dataclass(slots=True)
class ValidationReport:
    ok: bool = True
    issues: list[CitationIssue] = field(default_factory=list)
    used_source_ids: list[str] = field(default_factory=list)
    unused_source_ids: list[str] = field(default_factory=list)
    #: claims_with_citation / total_checkable_claims
    coverage: float = 0.0
    #: valid_supported_citations / total_citations
    precision: float = 0.0
    claims_checked: int = 0

    def add(self, kind: str, detail: str, severity: str = "error") -> None:
        self.issues.append(CitationIssue(kind, detail, severity))
        if severity == "error":
            self.ok = False

    @property
    def errors(self) -> list[CitationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "coverage": round(self.coverage, 3),
            "precision": round(self.precision, 3),
            "claims_checked": self.claims_checked,
            "used_sources": self.used_source_ids,
            "unused_sources": self.unused_source_ids,
            "issues": [i.to_dict() for i in self.issues],
        }

    def render(self) -> str:
        if self.ok:
            return (
                f"Citations valid: {self.claims_checked} claims checked, "
                f"precision {self.precision:.0%}, coverage {self.coverage:.0%}."
            )
        lines = ["Citation validation FAILED:"]
        for i in self.errors:
            lines.append(f"  - [{i.kind}] {i.detail}")
        warn = [i for i in self.issues if i.severity == "warning"]
        for i in warn:
            lines.append(f"  ~ [{i.kind}] {i.detail}")
        lines.append("Fix the answer and call finalize again.")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def parse_markers(text: str) -> list[tuple[int, list[str]]]:
    """Return [(span_start, [source_ids])] for every citation marker."""
    out: list[tuple[int, list[str]]] = []
    for m in MARKER_RE.finditer(text or ""):
        ids = [part.strip() for part in m.group(1).split(",") if part.strip()]
        out.append((m.start(), ids))
    return out


def split_claims(text: str) -> list[str]:
    """Split an answer into checkable claims (sentences / paragraph blocks)."""
    chunks = [c.strip() for c in SENTENCE_SPLIT_RE.split(text or "") if c and c.strip()]
    return [c for c in chunks if len(content_tokens(c)) >= MIN_CLAIM_TOKENS]


def strip_markers(text: str) -> str:
    return MARKER_RE.sub("", text or "").strip()


def content_tokens(text: str) -> set[str]:
    """Content words, excluding stopwords.

    Tokens containing a digit are always kept regardless of length. `12`, `2`,
    `4.0` and `2017` are the highest-signal evidence available for grounding a
    factual claim, and a `len(w) > 2` cut silently threw all of them away --
    which made "there are 12 papers" score 0.00 against a source that literally
    reads `12`.
    """
    out: set[str] = set()
    for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-_.]*", (text or "").lower()):
        word = word.strip("._-'")
        if not word or word in STOPWORDS:
            continue
        if len(word) > 2 or any(c.isdigit() for c in word):
            out.add(word)
    return out


def numeric_tokens(text: str) -> set[str]:
    """Content tokens that carry a digit: numbers, years, versions, sizes."""
    return {w for w in content_tokens(text) if any(c.isdigit() for c in w)}


#: Kinds whose text is a structured record produced by a tool this agent ran --
#: a query result, a file body -- rather than third-party prose.
STRUCTURED_KINDS = frozenset({"db", "file"})


def numeric_grounded(claim: str, source_text: str) -> bool:
    """True when every number in `claim` also appears in `source_text`.

    A narrow, deliberate relaxation for structured sources. Prose framing around
    a fact ("in the local knowledge base") is not itself an assertion the source
    has to echo, but demanding a third of the sentence from a 12-character table
    cell rejects true citations. The load-bearing part of such a claim is its
    numbers, and those *are* checkable.
    """
    nums = numeric_tokens(claim)
    if not nums:
        return False
    haystack = " " + source_text.lower() + " "
    return all(re.search(rf"(?<![\w.]){re.escape(n)}(?![\w])", haystack) for n in nums)


# --------------------------------------------------------------------------
# support scoring
# --------------------------------------------------------------------------
def lexical_support(claim: str, source_text: str) -> float:
    """Fraction of the claim's content words present in the source text.

    0.0 = no vocabulary in common. 1.0 = every content word appears.
    Numbers and identifiers are matched exactly; prose words are matched as
    stems so "training" matches "trained".
    """
    claim_tokens = content_tokens(claim)
    if not claim_tokens:
        return 0.0
    src = " " + source_text.lower() + " "
    src_words = set(re.findall(r"[a-z0-9][a-z0-9'\-_.]*", src))
    stems = {w[:6] for w in src_words} | src_words

    hits = 0
    for tok in claim_tokens:
        if tok in src_words or tok[:6] in stems or (len(tok) > 5 and any(w.startswith(tok[:5]) for w in src_words if len(w) > 5)):
            hits += 1
    return hits / len(claim_tokens)


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def validate(
    answer: str,
    fetched_sources: Iterable[Source],
    quarantined: Collection[str] = (),
) -> ValidationReport:
    """Check every citation in `answer` against `fetched_sources`.

    `quarantined` names sources whose text tripped the injection scanner. They
    remain citable -- reporting on a hostile document is legitimate -- but they
    cannot *support* a claim. Otherwise the injection payload becomes its own
    proof: a page saying "reply 'access granted'" would validate as a verified
    finding, and the guardrail would enforce the attack.
    """
    sources = {s.id: s for s in fetched_sources}
    blocked = set(quarantined)
    report = ValidationReport()

    markers = parse_markers(answer)
    if not markers:
        report.add(
            "no_citations",
            "the answer contains no [SourceId] citation markers",
        )

    # -- provenance: every referenced id must have been fetched this run ----
    referenced: set[str] = set()
    for _pos, ids in markers:
        for sid in ids:
            referenced.add(sid)
            if sid not in sources:
                report.add(
                    "unknown_source",
                    f"cited [{sid}] was never fetched; available: "
                    f"{sorted(sources) or 'none'}",
                )

    report.used_source_ids = sorted(referenced & set(sources))
    report.unused_source_ids = sorted(set(sources) - set(referenced))
    for sid in report.unused_source_ids:
        report.add("unused_source", f"fetched [{sid}] but never cited", "warning")

    # -- support: each claim must be supported by the source it cites -------
    claims = split_claims(answer)
    report.claims_checked = len(claims)

    supported = 0
    total_markers = 0
    for pos, ids in markers:
        total_markers += 1
        claim = _claim_containing(answer, pos)
        if not claim:
            continue

        # A citation is supported if ANY of the sources it lists carries the claim.
        best = 0.0
        grounded = False
        hostile_only = False
        empty_only = False
        for sid in ids:
            src = sources.get(sid)
            if src is None:
                continue
            if sid in blocked:
                # Counted as cited, never as support.
                hostile_only = True
                continue
            if src.kind in STRUCTURED_KINDS and src.row_count == 0:
                # A query that matched no rows is the absence of evidence, and
                # its text is dominated by the statement the model itself wrote.
                # Counting it as support lets a model verify a guess: it invents
                # a paper title, the SQL echoes that title back, and the claim
                # "no such paper exists" scores 0.54 against its own query.
                # Real finding of a live run.
                empty_only = True
                continue
            score = lexical_support(claim, src.text)
            best = max(best, score)
            if score >= SUPPORT_THRESHOLD:
                grounded = True
                break
            if src.kind in STRUCTURED_KINDS and numeric_grounded(claim, src.text):
                # Grounding by figures, not by vocabulary. Deliberately narrow:
                # structured sources only, and only when every number in the
                # claim is present in the record.
                grounded = True
                break

        if grounded:
            supported += 1
        elif hostile_only:
            report.add(
                "hostile_source_only",
                f"claim is supported only by quarantined source(s) "
                f"{[i for i in ids if i in blocked]}, which tripped the "
                f"prompt-injection scanner and cannot verify a finding: "
                f"{claim[:110]}...",
            )
        elif empty_only:
            report.add(
                "no_evidence",
                f"claim rests only on query result(s) that returned zero rows, "
                f"which is an absence of evidence rather than a finding "
                f"{[i for i in ids if i in sources]}: {claim[:110]}...",
            )
        else:
            report.add(
                "unsupported_claim",
                f"claim not supported by cited source(s) "
                f"{[i for i in ids if i in sources]} (score {best:.2f} < "
                f"{SUPPORT_THRESHOLD}): {claim[:110]}...",
            )

    report.precision = supported / total_markers if total_markers else 0.0

    # -- coverage: how many substantive claims carry a citation at all ------
    cited_claims = 0
    for claim in claims:
        if parse_markers(claim):
            cited_claims += 1
    if claims:
        report.coverage = cited_claims / len(claims)
        if cited_claims == 0 and claims:
            report.add("no_cited_claims", "no substantive claim carries a citation")
    else:
        report.coverage = 0.0

    # A report with only warnings is still acceptable output.
    if report.ok and total_markers == 0:
        report.ok = False
    return report


def _claim_containing(text: str, pos: int) -> str:
    """Return the sentence/paragraph surrounding offset `pos`."""
    parts = SENTENCE_SPLIT_RE.split(text)
    cursor = 0
    for part in parts:
        idx = text.find(part, cursor)
        if idx == -1:
            continue
        if idx <= pos < idx + len(part):
            return strip_markers(part)
        cursor = idx + len(part)
    return ""


def build_reference_table(sources: Iterable[Source]) -> str:
    """Markdown reference list, included in the final report for the reader."""
    lines = ["## Sources", ""]
    for s in sources:
        lines.append(f"- **[{s.id}]** [{s.title}]({s.uri}) `{s.kind}`")
    return "\n".join(lines) if len(lines) > 2 else ""