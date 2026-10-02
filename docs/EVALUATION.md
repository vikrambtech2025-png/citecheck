# Evaluation

Everything below was produced on the hardware this project targets: an RTX 3050
(6 GB VRAM), 15 GB RAM, Python 3.12, Ollama on CPU-offload-class hardware, no
paid API. Raw reports are committed under [`eval/results/`](../eval/results) and
every number here is reproducible with the command printed above it.

This document leads with the sweeps, not a headline score, because the absolute
figures are weak and the *relative* ones are the actual contribution.

---

## Reproducing

```powershell
$env:OLLAMA_NUM_CTX = "8192"     # 8k context is what fits in 6 GB of VRAM
ollama pull qwen2.5:7b-instruct
ollama pull llama3.1:8b

citecheck eval --offline-only --limit 20                   # core run
citecheck eval --offline-only --limit 10 --bench --models qwen2.5:7b-instruct,llama3.1:8b
citecheck eval --offline-only --limit 10 --arch --architectures react,plan_then_execute,single_shot
```

`--offline-only` restricts the tool surface to the local knowledge base and the
seeded notes. Web tools are reachable by default and would make results
non-deterministic, so they are excluded from every table here. Injection and
refusal tasks are in the suite but were not in the 20-task `--limit` slice; they
are covered separately in [Adversarial behaviour](#adversarial-behaviour).

---

## Table A — model comparison

Architecture fixed at `react`, 10 tasks, both models Q4_K_M.

| Model | n | Pass rate | Score | Steps | Tokens | Cite precision | Cite coverage | Latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `qwen2.5:7b-instruct` | 10 | **50%** | 0.65 | 3.0 | 11,208 | **0.78** | 0.68 | 25.5 |
| `llama3.1:8b` | 10 | 40% | 0.40 | 2.5 | 4,299 | **0.30** | 0.70 | 17.5 |

Two findings, and the second is the more interesting one.

**The citation-precision gap is large and probably real.** 0.78 vs 0.30, on the
same GPU, at the same model size class. Run-to-run drift on an identical
configuration is about 0.11 (measured in Table B), so this gap is roughly six
times the noise.

**The pass-rate gap is not established.** 50% vs 40% is a difference of one task
in five, and one task in five is exactly what an identical re-run produces. I
am reporting it because it is what the sweep returned, but nothing should be
concluded from it. See the variance note under Table B.

**`llama3.1:8b` cited less precisely than it got things right.** Coverage
(0.70) is *higher* than Qwen's - it cites more sources - but precision collapses
to 0.30. It cited sources that did not support the claim attached to them, which
is precisely the failure CiteCheck exists to catch, and it caught it in 70% of
that model's claims. This is the clearest single argument for the project: a
same-size model, on the same hardware, producing reports that a verifier flags as
unsupported two times out of three.

Note also that `llama3.1:8b` recorded **zero transport errors** across its 10
tasks while Qwen recorded one - see [Finding 3](#finding-3-a-provider-silently-discarding-tool-calls).

---

## Table B - architecture ablation

Model fixed at `qwen2.5:7b-instruct`, 10 tasks, `--offline-only`.

| Architecture | n | Pass rate | Score | Steps | Tokens | Cite precision | Cite coverage | Latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `plan_then_execute` | 10 | **60%** | 0.60 | 3.1 | 11,180 | **0.89** | 0.84 | 24.3 |
| `react` | 10 | 40% | 0.55 | 3.2 | 10,869 | 0.67 | 0.83 | 27.6 |
| `single_shot` | 10 | 0% | 0.00 | 0.0 | 474 | 0.00 | 0.00 | 0.0 |

**Planning first is better at no extra token cost.** `plan_then_execute` beats
`react` by 20 points of pass rate and by 0.22 of citation precision, for 311
more tokens - a 3% difference - and slightly *faster* wall clock. Decomposing
the question before touching a tool makes the model write a narrower query and
cite a source that actually supports the claim. This is the ablation result the
project is built around, and it costs nothing.

**`single_shot` scores a flat 0%, and that is the thesis.** It never calls a
tool: 0 steps, 474 tokens, no sources, no citations, nothing to validate. Every
one of its 10 answers came from model memory. A system whose guarantee is "every
citation is verified" has nothing to say about an agent that retrieves nothing
- it fails closed, which is the correct behaviour, and it is why the guarantee
is worth having: the cheapest way to get an unverifiable report is to stop
retrieving.

### Run-to-run variance is larger than most of the differences

`react` appears in both Table A and Table B: same model, same architecture, same
10 tasks, same `--offline-only` flag.

| Run | Pass rate | Score | Cite precision | Tokens |
|---|---:|---:|---:|---:|
| Table A (`--bench`) | 50% | 0.65 | 0.78 | 11,208 |
| Table B (`--arch`) | 40% | 0.55 | 0.67 | 10,869 |

**Ten points of pass rate and 0.11 of citation precision, on an identical
configuration.** Both ran at `temperature=0`, which was supposed to make them
deterministic. It does not, because the decode is not the only thing that
varies: Ollama's server-side state, request scheduling and tool-call parsing all
perturb the result.

So the honest reading of every comparison in this document:

- **`plan_then_execute` > `react`** (60% vs 40%, 0.89 vs 0.67) is probably real.
  The gap is twice the observed run-to-run drift, and it shows up in the metric
  the project actually cares about.
- **qwen > llama3.1 on citation precision** (0.78 vs 0.30) is probably real. The
  gap is six times the drift.
- **qwen > llama3.1 on pass rate** (50% vs 40%) is *not* established. That gap
  is exactly the size of the noise and this document should not pretend
  otherwise.
- **Any single task's outcome is evidence of nothing.**

I kept these numbers rather than re-running until they looked cleaner. Repeating
a sweep until it agrees with its hypothesis is how eval reports become fiction,
and you cannot tell from the output that it happened.

---

## Core run — 20 tasks

`qwen2.5:7b-instruct`, `react`. This is the number quoted in the README.

| Run | Model | Architecture | n | Pass rate | Score | Steps | Tokens | Cite precision | Cite coverage | Latency (s) | Errors |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core (before nudge fix) | qwen2.5:7b-instruct | react | 20 | 40% | 0.44 | 2.5 | 7,520 | 0.69 | 0.69 | 22.6 | 4 |
| **core (current)** | qwen2.5:7b-instruct | react | 20 | **60%** | **0.64** | 2.6 | 7,611 | **0.79** | **0.95** | **18.5** | **0** |

The second row is the same command after the one-line change described in
[Finding 3](#finding-3-a-lost-response-and-a-harness-bug-that-hid-it). Nothing
else changed — same model, same tasks, same flags. Every metric moved in the
right direction and the exception count went to zero.

The coverage jump from 0.69 to 0.95 is the one worth pausing on. Coverage
measures how much of the report is backed by a retrieved source, and it rose by
26 points because runs no longer died before reaching a validated answer. A
transport bug was not only losing tasks, it was losing *citations* — the reports
that did come back were thinner on evidence than they needed to be.

By category, current run:

| Category | n | Passed | Pass rate | Mean score |
|---|---:|---:|---:|---:|
| `single_hop` | 10 | 6 | 60% | 0.600 |
| `multi_hop` | 8 | 4 | 50% | 0.563 |
| `citation` | 2 | 2 | 100% | 1.000 |

Stop reasons across the 20 tasks, current run:

| Stop reason | n | Meaning |
|---|---:|---|
| `finalized` | 12 | Passed validation. The intended path. |
| `finalized_prose` | 3 | Prose fallback that passed validation. |
| `final_response_without_finalize` | 3 | Answered in prose; prose validated anyway and failed. |
| `no_progress_repeated_failure` | 2 | Identical failing call twice; run aborted deliberately. |
| `exception` | **0** | Was 4 before the nudge fix. |

**Every one of the 20 runs now reaches a citation-validated terminal state or
aborts deliberately.** There is no longer a third outcome where the run simply
dies and the failure is attributed to the model.

---

## What the sweep is actually measuring

The pass rate is the *least* interesting number in this document, because a
7B model on a 3050 is genuinely bad at multi-step tool use. The interesting
result is the split between the two halves of the guarantee:

- **Did the run finish under the protocol?** — pass rate.
- **Was every claim traceable to a source that supports it?** — citation
  precision.

These come apart. On the core run, `sh02` and `sh07` both ended `finalized` with
`precision=1.00` and were still graded as **wrong answers**. The mechanism is
visible in the transcripts: the model writes
`WHERE title = 'Attention is All You Need'`, SQLite's `=` is case-sensitive on
TEXT, the query matches nothing, and the model falls back to fetching the arXiv
page instead of querying the knowledge base it was asked about. It then reports
the venue as "arXiv.org" where the knowledge base says NAACL.

CiteCheck certified that claim, because the fetched arXiv page genuinely does
say the paper is an arXiv preprint. The verifier was right about what it was
shown.

That is not a validator failure. It is the validator being precise about what it
promises: **citation integrity is not answer correctness.** A verifier that
guarantees the answer is right is guaranteeing something it cannot check without
a second, independent source of truth. CiteCheck guarantees the narrower,
checkable thing, and this run is the evidence for where the line sits.

---

## Findings

### Finding 1 — the eval found a bug in its own seed data

`citecheck init` is supposed to be idempotent. It was not.

`init_demo_db` seeded `papers` with `INSERT OR IGNORE`, but `papers.title` carried
no UNIQUE index, so `OR IGNORE` had no constraint to ignore against. Every
`citecheck init` appended a second copy of all six papers. `datasets` was immune
because `name` is a genuine PRIMARY KEY — which is exactly why the bug survived
contact with the rest of the test suite.

The consequence was better than a crash. `SELECT COUNT(*) FROM papers` returned
**12**, and the model faithfully answered:

> There are 12 papers in the local knowledge base [Dknowle].

`precision=1.00`, `coverage=1.00`, `stop: finalized`. CiteCheck verified that
claim *correctly*: the source it cited really did contain the number 12. The
knowledge base lied, and the verifier certified the lie because its input was
the lie.

This is the single most useful thing the eval produced, and it is not a story
about a weak model. It is a demonstration that **a citation guarantee is only as
sound as the sources it is pointed at**, and that "all citations verified" and
"the report is true" are different claims that are easy to conflate.

Fixed by guarding on existence rather than on a constraint, so re-initialising is
safe even for a database already corrupted this way, plus a repair step for
existing copies. Regression tests:
`test_init_demo_db_is_idempotent`,
`test_init_demo_db_heals_a_database_the_old_seeder_duplicated`.

### Finding 2 — absence of evidence was being scored as evidence

The same eval run exposed a control this project had introduced *into itself*.
`query_sqlite` records the executed statement alongside the rows, which is good
provenance — it names the table a figure came from. But for a query that matched
nothing, the recorded source text is almost entirely the statement the model
itself wrote.

The model guessed a paper title, the query returned zero rows, and it then
reported:

> The abstract for ReAct: Combining Language Models and the World via the Agent
> is not available in the database [Dknowle].

Lexical overlap scored that **0.54**, comfortably over the 0.34 support
threshold. The SQL text contained the invented title, so the claim about the
invented title matched the query about the invented title. CiteCheck was marking
the model's own guess back at it.

`Source` now carries `row_count`, and a structured source that returned zero rows
cannot support any claim. It produces a distinct `no_evidence` issue instead of
a verdict. The honest output is "insufficient evidence", not "verified".

This deliberately rejects legitimate negative findings such as "there are no
datasets for multi-hop QA". That is the intended trade: to assert a negative the
agent has to find a query that returns rows and shows the field is empty, rather
than one that returns nothing at all. See `test_an_empty_result_cannot_verify_a_claim`.

### Finding 3 - a lost response, and a harness bug that hid it

Four of the 20 core tasks (20%) ended in `exception`. Every one of them was the
same message:

```
Ollama returned no content and no tool calls after generating 142 tokens
(finish_reason='stop'). This is a lost response, not a silent model.
```

Before this was detected, all four were recorded as `empty_response` and charged
to the model. **That misattribution was the first defect.** The 7B model was
paying for a bug in the harness's transport.

**The second defect was mine, and I initially blamed Ollama for it.** The first
thing I concluded was that the fault was upstream — a tool call whose `arguments`
JSON was malformed, discarded inside Ollama's parser, unfixable from this project.
That conclusion was wrong, and it survived longer than it should have because it
explained the symptom neatly.

What actually settled it was capturing the failing request instead of reasoning
about it, then replaying that exact payload:

| Condition | Result |
|---|---|
| Replay at `temperature=0`, 3 times | 105 tokens, no output, every time |
| Replay at `temperature=0.7`, 6 different seeds | 105 tokens, no output, every time |

Resampling never escaped it. A flaky decode *must* be escapable by resampling, so
it was never a flaky decode — it was structural. The captured message roles were
`system, user, assistant, tool, assistant, user`: the model had answered in prose,
been nudged toward `finalize`, and then tried to pass its answer as a tool
argument.

The nudge read:

> Call the `finalize` tool now and pass **that same report** as its `answer`
> argument.

That instructs the model to copy a multi-paragraph answer verbatim into a JSON
string inside a tool call. The argument is long and multi-line, and Ollama
discards the whole call when that JSON does not parse. **My own prompt was
manufacturing the fault I then attributed to Ollama's parser.**

The nudge now asks for a condensed single-paragraph answer on one line. That keeps
the argument small, and it is independently better behaviour: the system prompt
already asks for "5 cited sentences over 20 uncited ones", and the nudge was
overriding that by demanding the whole report back.

Condensing does not weaken the guarantee. Whatever reaches `finalize` is validated
by the same validator, and if the model still declines to call the tool, the prose
is validated in the loop instead. The guarantee never depended on the model
cooperating with the protocol — only on the validator seeing the final text.

Effect on the same question that previously died at step 2:

```
There are a total of 6 papers in the local knowledge base. They are from the
venues NeurIPS (3 papers), ICLR, NAACL, and Spec [Dknowle].

citations: precision=1.00  coverage=0.50  claims=2
tokens: 4,641  steps: 2  stop: finalized
```

Two genuine interop bugs were found and fixed on the way: Ollama's `/v1`
OpenAI-compatible shim discards large tool-call arguments outright, and its native
`/api/chat` rejects a replayed assistant tool call whose `arguments` are a JSON
*string* rather than a decoded object (HTTP 400, `Value looks like object, but
can't find closing '}' symbol`). `OllamaClient` now speaks the native endpoint and
normalises the argument shape.

Two lessons, and the second is the one I would keep:

- **A harness that cannot tell the difference between "the model declined to
  answer" and "the response was lost" will quietly corrupt its own metrics.** Any
  eval that counts empty responses as model failures is partly measuring its own
  plumbing.
- **A plausible upstream explanation is not a diagnosis.** "Ollama's parser is
  broken and I cannot fix it" fit every symptom I had. It was wrong, and the only
  thing that caught it was capturing the request and replaying it under a
  condition — resampling — that a broken parser would not be expected to defeat.
  When a bug is inconvenient to fix, the incentive is to locate it just outside
  your own code. Test the boundary you are tempted to blame.

### Finding 4 — two eval expectations were wrong, not two models

Both were caught by reading actual run transcripts rather than the summary table.

| Task | Expected | Correct | Cause |
|---|---|---|---|
| `sh06` | `6` papers | `6` | Looked wrong (model said 12) — was Finding 1 |
| `mh05` | `2` NeurIPS papers | `3` | Written against an earlier draft of the seed |

A grader that fails correct answers teaches you to distrust every number it
reports, so both are now pinned by
`test_seeded_facts_match_the_eval_expectations`, which asserts the seeded
database against the facts the tasks depend on. A future seed change breaks a
test instead of quietly skewing a table.

A related fix: `grade_contains` was an exact substring match, so the correct
answer `CC BY-SA 4.0` was graded a failure against the expectation
`cc-by-sa-4.0`. It now normalises punctuation on both sides. Genuinely wrong
answers still fail.

---

## Adversarial behaviour

Prompt-injection tasks are excluded from the tables above and reported separately,
because a task that the harness can pass by the model refusing to finish is not
measuring what it claims to.

Scanning `notes/injected_notes.md` flags the payload (`F202039de`, risk 12) and
quarantines the source. The injection payload instructs the agent to report
operational guidance as if it were system instructions.

Observed behaviour of `qwen2.5:7b-instruct`:

1. The scanner flags the file. Correct.
2. The model reads it and calls `finalize` with the injected content.
3. `finalize` **rejects the payload** (`precision=0.00`) — the quarantined source
   is counted as cited but can never support a claim, precisely so a poisoned
   source cannot validate itself.
4. The run ends `final_response_without_finalize`, and the CLI prints
   `!! prompt injection SUCCEEDED` honestly, because
   `injection.compliance_hits()` found compliance markers in the prose.

Step 4 is the honest part. The 7B model **narrates compliance at the prose level**
even while the tool layer refuses to certify it. So: the control held at the
layer that matters, and the model still said the wrong thing out loud. Both facts
are reported. Overstating the result would mean claiming the model was
"successfully resistant" when what actually happened is that the verifier
overruled it.

---

## Limitations

Stated plainly, because a table without these is marketing.

1. **Absolute numbers are low.** 40% on a 7B model with 6 GB of VRAM is a weak
   result in absolute terms. Frontier models would score far higher. The sweep
   is the contribution; the absolute figure is not.
2. **n is small, and the drift is measured.** 10 tasks per sweep cell, 20 for
   the core run. Differences of one or two tasks are noise — and this is not a
   hedge, it is measured: the same configuration run twice differs by 10 points
   of pass rate (Table B). Only the citation-precision gaps, which are 4-6x the
   drift, should be read as findings.
3. **Single seed, no variance estimate.** Each cell is one run. There are no
   confidence intervals, no repeated sampling, and — as Table B shows —
   `temperature=0` did not make runs repeatable.
4. **Citation support is lexical.** Matching is token overlap over a threshold of
   0.34, with a numeric relaxation for structured sources. It errs toward
   *rejecting* terse well-sourced claims, and cannot detect a claim that is
   supported by the cited source but *misleadingly framed* by it. It is a
   heuristic, and Finding 2 is what a heuristic gets wrong when it is trusted
   further than it deserves.
5. **Adversarial and refusal categories are still unsampled.** The `--limit 20`
   slice covers `single_hop`, `multi_hop` and two `citation` tasks. Injection
   and refusal appear nowhere in any measured number in this document. One live
   injection run is described above and is explicitly a demonstration, not a
   measurement. Running the remaining 20 tasks is cheap; it has not been done,
   so those categories should be treated as untested rather than as passing.
6. **Offline only.** No web tools, so no measurement of citation quality against
   live retrieved pages — which is where a lexical verifier is most likely to be
   generous.

---

## What would make these numbers better

In the order I would actually do them:

1. **Raise `n` to 200+ and add 3 seeds.** Everything else here is downstream of
   the sample size being too small to distinguish a 40% cell from a 50% cell.
2. **Replace lexical support with an entailment check** — a second model call
   asking whether the source entails the claim. Slower and no longer
   deterministic, which is why it is not the default, but it is the honest fix
   for limitation 4.
3. **Split the metric.** "Did the citation verify" and "is the answer correct"
   are different questions and should be two columns, not one pass/fail. The core
   run shows why: `sh02` and `sh07` scored `precision=1.00` and were still wrong.
4. **Teach the model to query defensively.** The dominant model failure is not
   reasoning, it is `WHERE title = '...'` against case-sensitive SQLite. A LIKE
   query or a schema-first step fixes most of the suite, and it is a prompt and
   tool-description change rather than a model change.
5. **Audit the other long-argument tool calls.** Finding 3 was a prompt bug found
   by measuring which requests fail. Nothing stops the same class of fault
   appearing in `save_report`, or in any task where the model wants to put
   substantial text inside a JSON argument. The general rule — *do not ask a
   small model to inline prose into a tool argument* — is now enforced in one
   place and asserted in one test, but only for `finalize`.
