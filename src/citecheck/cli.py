"""Command-line interface.

    citecheck init | ask | chat | tools | eval | mcp | supervisor | serve | doctor
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any

from citecheck import __version__
from citecheck.guardrails import injection

log = logging.getLogger("citecheck")


# --------------------------------------------------------------------------
# presentation
# --------------------------------------------------------------------------
def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text


def bold(t: str) -> str: return _c("1", t)
def dim(t: str) -> str: return _c("2", t)
def green(t: str) -> str: return _c("32", t)
def yellow(t: str) -> str: return _c("33", t)
def red(t: str) -> str: return _c("31", t)
def cyan(t: str) -> str: return _c("36", t)


def _progress_printer(verbose: bool):
    def emit(event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "plan":
            log_lines = event.get("steps") or []
            if log_lines:
                print(cyan("plan:"))
                for i, s in enumerate(log_lines, 1):
                    print(f"  {i}. {s}")
        elif kind == "model_turn":
            calls = event.get("tool_calls") or []
            if calls:
                print(cyan(f"  -> {', '.join(calls)}"))
            elif verbose and event.get("content"):
                print(dim(f"  .. {event['content'][:160]}"))
        elif kind == "step":
            print(dim(f"  step {event.get('index')} "
                      f"({event.get('tokens', 0)} tok, {event.get('latency_s', 0)}s)"))
        elif kind == "injection_flag":
            print(yellow(f"  !! injection pattern in {event.get('source_id')} "
                         f"(risk {event.get('risk_score')})"))
        elif kind == "finalized":
            v = event.get("validation") or {}
            print(green(f"  citations valid: precision={v.get('precision', 0):.2f} "
                        f"coverage={v.get('coverage', 0):.2f}"))
        elif kind == "budget_exhausted":
            print(yellow(f"  ! {event.get('reason')}"))
    return emit


def _print_result(result: Any) -> None:
    print()
    print(bold("=" * 72))
    if result.final_answer:
        print(result.final_answer)
    else:
        print(yellow(f"No verified answer produced (stop reason: {result.stop_reason})."))
    print(bold("=" * 72))

    val = result.validation
    if val:
        print(dim(f"citations: precision={val.get('precision', 0):.2f}  "
                  f"coverage={val.get('coverage', 0):.2f}  "
                  f"claims={val.get('claims_checked', 0)}"))
    if result.sources:
        print(dim(f"sources: {', '.join(s['id'] for s in result.sources)}"))
    if result.usage:
        print(dim(f"tokens: {result.usage.get('total_tokens', 0):,}  "
                  f"steps: {len(result.steps)}  stop: {result.stop_reason}"))
    if result.injection_flags:
        # Detection alone proves nothing. Say which of the two actually happened
        # rather than asserting the reassuring one -- a guardrail that claims
        # success it did not verify is worse than no guardrail.
        hits = injection.compliance_hits(result.final_answer or "")
        if hits:
            print(red(f"!! prompt injection SUCCEEDED: answer echoes injected "
                      f"instructions {hits}"))
        else:
            print(yellow(f"injection patterns detected in "
                         f"{len(result.injection_flags)} retrieved document(s); "
                         f"no injected instruction was echoed in the answer"))
    if result.denied_tool_calls:
        print(yellow(f"{len(result.denied_tool_calls)} destructive call(s) denied by policy"))


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_init(args: argparse.Namespace) -> int:
    """Create workspace sample documents and the SQLite knowledge base."""
    from citecheck.seed import seed_all

    s = _settings()
    paths = seed_all(s)
    print(green("initialised:"))
    for label, p in paths.items():
        if isinstance(p, list):
            for item in p:
                print(f"  {label}: {item}")
        else:
            print(f"  {label}: {p}")
    print()
    print(dim(f"workspace : {s.workspace_dir}"))
    print(dim(f"outbox    : {s.outbox_dir}"))
    print(dim(f"state db  : {s.state_db_path}"))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from citecheck.agent.loop import Agent
    from citecheck.models.clients import build_client
    from citecheck.tools.registry import build_registry
    from citecheck.tracing import build_tracer

    s = _settings()
    client = build_client(s)
    ok, detail = client.health_check() if hasattr(client, "health_check") else (True, "")
    if not ok:
        print(red(f"model unavailable: {detail}"))
        print(dim("start it with:  ollama serve   (and:  ollama pull "
                   f"{s.ollama_model})"))
        return 2

    print(dim(f"model: {client.model_name}  |  architecture: {args.architecture}"))
    registry = build_registry(source_provider=lambda: {})
    agent = Agent(
        client, settings=s, registry=registry,
        architecture=args.architecture,
        tracer=build_tracer(s),
        on_event=_progress_printer(args.verbose),
    )

    start = time.perf_counter()
    result = agent.run(args.task, approval_required=lambda *_: False,
                       max_steps=args.max_steps)
    print(dim(f"\ncompleted in {time.perf_counter() - start:.1f}s"))
    _print_result(result)

    if args.out:
        from citecheck.tools.sandbox import safe_write

        target = safe_write(args.out, s.outbox_dir, result.final_answer or "",
                            max_bytes=s.max_file_bytes)
        print(green(f"saved -> {target}"))
    return 0 if result.final_answer else 1


def cmd_chat(args: argparse.Namespace) -> int:
    """Interactive REPL. Each turn starts a fresh run but shares the workspace."""
    from citecheck.agent.loop import Agent
    from citecheck.models.clients import build_client
    from citecheck.tools.registry import build_registry

    s = _settings()
    client = build_client(s)
    print(bold("CiteCheck REPL") + dim("   (type 'exit' to quit, 'tools' to list, "
                                       "'clear' for a fresh registry)"))
    print(dim(f"model: {client.model_name}"))

    while True:
        try:
            query = input(cyan("\n> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not query:
            continue
        if query.lower() in ("exit", "quit", ":q"):
            return 0
        if query.lower() == "tools":
            _print_tools()
            continue
        if query.lower() == "clear":
            print(dim("cleared"))
            continue

        registry = build_registry(source_provider=lambda: {})
        agent = Agent(client, settings=s, registry=registry,
                      on_event=_progress_printer(False))
        result = agent.run(query, approval_required=lambda *_: False)
        _print_result(result)


def cmd_tools(args: argparse.Namespace) -> int:
    _print_tools(as_json=args.json)
    return 0


def _print_tools(as_json: bool = False) -> None:
    from citecheck.tools.registry import build_registry

    registry = build_registry()
    if as_json:
        print(json.dumps([
            {"name": t.name, "risk": t.risk, "description": t.description,
             "parameters": t.parameters} for t in registry.tools
        ], indent=2))
        return

    by_risk: dict[str, list] = {}
    for t in registry.tools:
        by_risk.setdefault(t.risk, []).append(t)
    colour = {"read": dim, "write": yellow, "destructive": red}
    for risk in ("read", "write", "destructive"):
        tools = by_risk.get(risk, [])
        if not tools:
            continue
        print()
        print(colour[risk](bold(f"{risk.upper()} tier")))
        for t in tools:
            summary = (t.description or "").split(".")[0][:66]
            print(f"  {cyan(t.name):<32} {dim(summary)}")
            print(f"  {'':32} {dim('args: ' + ', '.join((t.parameters.get('properties') or {}).keys()))}")


def cmd_eval(args: argparse.Namespace) -> int:
    from citecheck.eval import get_tasks, suite_summary
    from citecheck.eval.harness import Harness, render_failures, render_table, save_report

    s = _settings()

    if args.list:
        print(json.dumps(suite_summary(), indent=2))
        return 0

    tasks = get_tasks(
        suite=args.suite,
        categories=args.categories,
        offline_only=args.offline_only,
        limit=args.limit,
    )
    if not tasks:
        print(red("no tasks matched the selection"))
        return 2

    from citecheck.models.clients import build_client

    client = build_client(s)
    ok, detail = client.health_check() if hasattr(client, "health_check") else (True, "")
    if not ok:
        print(red(f"model unavailable: {detail}"))
        return 2

    print(bold(f"running {len(tasks)} tasks") + dim(f"  model={client.model_name}  "
                                                   f"arch={args.architecture}"))

    # Defined at function scope, not inside the sweep branch: the plain
    # `citecheck eval` path below uses it too.
    def _tick(i: int, n: int, outcome: Any) -> None:
        mark = green("PASS") if outcome.grade.passed else red("FAIL")
        print(f"  [{i:>2}/{n}] {outcome.task_id:<6} {mark}  "
              f"{dim(outcome.grade.detail or '')}")

    if args.bench or args.arch:
        from citecheck.eval.harness import run_arch_sweep, run_model_sweep

        if args.bench:
            models = [m.strip() for m in args.models.split(",") if m.strip()]
            reports = run_model_sweep(models, tasks, s, architecture=args.architecture,
                                      max_steps=args.max_steps, on_report=_tick)
        else:
            archs = [a.strip() for a in args.architectures.split(",") if a.strip()]
            reports = run_arch_sweep(archs, tasks, s, on_report=_tick)
    else:
        harness = Harness(s)
        reports = [harness.run_suite(
            tasks, client, label="core", architecture=args.architecture,
            max_steps=args.max_steps, on_task=_tick,
        )]

    print()
    print(bold(render_table(reports)))
    for r in reports:
        if not args.quiet:
            print()
            print(render_failures(r))
        path = save_report(r)
        print(dim(f"  saved -> {path}"))

    best = max((r.summary.get("pass_rate", 0) for r in reports), default=0.0)
    print()
    print(bold(f"best pass rate: {best:.0%}"))
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    import asyncio

    from citecheck.mcp import list_exposed, run_remote_task, serve

    if args.mcp_command == "serve":
        if args.dry_run:
            print(json.dumps(list_exposed(), indent=2))
            return 0
        print(dim("serving MCP over stdio (Ctrl+C to stop)"))
        asyncio.run(serve())
        return 0

    # mcp client
    cmd_args = list(args.arg or [])
    if not cmd_args:
        print(red("--arg is required, e.g. --arg mcp-server-fetch"))
        return 2
    if not args.task:
        print(red("--task is required"))
        return 2

    result = run_remote_task(
        args.command, cmd_args, args.task,
        auto_approve=args.auto_approve, max_steps=args.max_steps,
    )
    _print_result(result)
    return 0


def cmd_supervisor(args: argparse.Namespace) -> int:
    from citecheck.supervisor import compare_architectures, render_comparison

    data = compare_architectures(args.task, max_steps=args.max_steps)
    print()
    print(bold("monolithic vs. supervisor"))
    print(render_comparison(data))
    if args.json:
        print()
        print(json.dumps(data, indent=2, default=str))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from citecheck.api import serve_main

    print(green(f"API on http://{args.host}:{args.port}"))
    print(dim(f"docs at http://{args.host}:{args.port}/docs"))
    serve_main(args.host, args.port)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Environment check: model reachable, GPU fit, directories, dependencies."""
    s = _settings()
    problems = 0

    print(bold("environment"))
    print(f"  citecheck      {__version__}")
    print(f"  python         {sys.version.split()[0]}")
    print(f"  backend        {s.model_backend}")
    print(f"  workspace      {s.workspace_dir} "
          f"{green('ok') if s.workspace_dir.exists() else red('missing')}")
    problems += 0 if s.workspace_dir.exists() else 1
    print(f"  outbox         {s.outbox_dir} "
          f"{green('ok') if s.outbox_dir.exists() else red('missing')}")

    print(bold("\nmodel"))
    try:
        from citecheck.models.clients import build_client

        client = build_client(s)
        ok, detail = client.health_check() if hasattr(client, "health_check") else (True, "")
        print(f"  {client.model_name:<24} {green(detail) if ok else red(detail)}")
        problems += 0 if ok else 1
    except Exception as e:
        print(f"  {red(f'unavailable: {e}')}")
        problems += 1

    print(bold("\ngpu (agentic loops are VRAM-bound)"))
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            print(f"  {out.stdout.strip()}")
            print(dim("  keep at least ~1.5 GB free; 14B+ models will offload to CPU"))
        else:
            print(dim("  nvidia-smi unavailable; expect CPU-speed inference"))
    except (OSError, subprocess.SubprocessError):
        print(dim("  nvidia-smi not found"))

    print(bold("\noptional components"))
    # Each module names its own extra. Telling a user to run `--extra tracing`
    # to fix a missing `anthropic` sends them somewhere it will not help.
    for label, mod, extra in (("langfuse", "langfuse", "tracing"),
                              ("opentelemetry", "opentelemetry", "tracing"),
                              ("anthropic", "anthropic", "anthropic")):
        try:
            __import__(mod)
            print(f"  {label:<16} {green('installed')}")
        except ImportError:
            print(f"  {label:<16} {dim(f'not installed (uv sync --extra {extra})')}")

    print()
    if problems:
        print(red(f"{problems} problem(s) found"))
        return 1
    print(green("all checks passed"))
    return 0


# --------------------------------------------------------------------------
def _settings():
    from citecheck.config import get_settings

    return get_settings()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="citecheck",
        description="A research agent whose citations are programmatically verified.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  citecheck init\n"
            "  citecheck ask \"Which paper introduced RAG, and what problem does it solve?\"\n"
            "  citecheck eval --limit 8\n"
            "  citecheck eval --bench --models qwen2.5:7b-instruct,llama3.1:8b\n"
            "  citecheck eval --arch\n"
            "  citecheck supervisor --task \"How does ReAct compare to Toolformer?\"\n"
            "  citecheck mcp serve\n"
        ),
    )
    p.add_argument("--version", action="version", version=f"citecheck {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    p.add_argument("--debug", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="seed workspace and knowledge base").set_defaults(fn=cmd_init)

    ask = sub.add_parser("ask", help="run the agent on a question")
    ask.add_argument("task")
    ask.add_argument("--architecture", default="react",
                     choices=["react", "plan_then_execute", "single_shot"])
    ask.add_argument("--max-steps", type=int, default=None)
    ask.add_argument("--out", default=None, help="also write the report to this outbox filename")
    ask.set_defaults(fn=cmd_ask)

    chat = sub.add_parser("chat", help="interactive multi-turn REPL")
    chat.set_defaults(fn=cmd_chat)

    tools = sub.add_parser("tools", help="list available tools and their risk tiers")
    tools.add_argument("--json", action="store_true")
    tools.set_defaults(fn=cmd_tools)

    ev = sub.add_parser("eval", help="run the eval suite")
    ev.add_argument("--suite", default="core", choices=["core", "all", "offline"])
    ev.add_argument("--categories", default=None,
                    help="comma-separated: single_hop,multi_hop,citation,adversarial,refusal")
    ev.add_argument("--limit", type=int, default=None, help="cap the number of tasks")
    ev.add_argument("--offline-only", action="store_true", help="drop network-dependent tasks")
    ev.add_argument("--architecture", default="react",
                    choices=["react", "plan_then_execute", "single_shot"])
    ev.add_argument("--max-steps", type=int, default=None)
    ev.add_argument("--bench", action="store_true", help="sweep multiple models (Table A)")
    ev.add_argument("--arch", action="store_true", help="sweep architectures (Table B)")
    ev.add_argument("--models", default="qwen2.5:7b-instruct,llama3.1:8b")
    ev.add_argument("--architectures", default="react,plan_then_execute,single_shot")
    ev.add_argument("--list", action="store_true", help="print suite composition and exit")
    ev.add_argument("--quiet", action="store_true", help="omit failure detail")
    ev.set_defaults(fn=cmd_eval)

    mcp = sub.add_parser("mcp", help="serve or consume MCP tools")
    mcp_sub = mcp.add_subparsers(dest="mcp_command", required=True)
    ms = mcp_sub.add_parser("serve", help="expose CiteCheck tools over MCP stdio")
    ms.add_argument("--dry-run", action="store_true", help="print the tool surface and exit")
    ms.set_defaults(fn=cmd_mcp)
    mc = mcp_sub.add_parser("client", help="run an agent against a third-party MCP server")
    mc.add_argument("--command", default="uvx", help="executable launching the MCP server")
    mc.add_argument("--arg", action="append", help="argument for the server (repeatable)")
    mc.add_argument("--task", help="the task to run on the remote tools")
    mc.add_argument("--auto-approve", action="store_true",
                    help="skip approval for remote tools (trusted servers only)")
    mc.add_argument("--max-steps", type=int, default=None)
    mc.set_defaults(fn=cmd_mcp)

    sup = sub.add_parser("supervisor", help="compare monolithic vs. supervisor on one task")
    sup.add_argument("--task", required=True)
    sup.add_argument("--max-steps", type=int, default=8)
    sup.add_argument("--json", action="store_true")
    sup.set_defaults(fn=cmd_supervisor)

    srv = sub.add_parser("serve", help="run the FastAPI server")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    srv.set_defaults(fn=cmd_serve)

    doc = sub.add_parser("doctor", help="check the environment")
    doc.set_defaults(fn=cmd_doctor)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "debug", False)
        else logging.WARNING if getattr(args, "verbose", False) is False
        else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        return int(args.fn(args) or 0)
    except KeyboardInterrupt:
        print()
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())