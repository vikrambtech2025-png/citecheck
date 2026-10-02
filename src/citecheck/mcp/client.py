"""MCP client: build an agent that consumes a *third-party* MCP server.

Building the server side shows you can expose tools. Consuming someone else's
shows you understand the protocol rather than just your own wrapper.

The same `Agent` loop runs over a different transport, which is the actual claim
being demonstrated: swap the registry, keep the agent.

    citecheck mcp client --command uvx --arg mcp-server-fetch --arg --allow-domains=example.com

Remote MCP tools are treated as `write` risk by default, so they pass through the
approval gate. A server you did not write may have side effects, and the safe
default matters more than convenience. Pass `--auto-approve` only for a server
you trust and control.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from citecheck.agent.loop import Agent, RunResult
from citecheck.config import Settings, get_settings
from citecheck.models.base import RISK_WRITE
from citecheck.models.clients import build_client
from citecheck.tools.base import Source, Tool, ToolContext, ToolResult
from citecheck.tools.registry import ToolRegistry
from citecheck.tracing import build_tracer

log = logging.getLogger(__name__)


@dataclass(slots=True)
class RemoteToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


class McpToolRegistry(ToolRegistry):
    """Dispatches to an MCP session instead of local Python functions."""

    def __init__(self, session: Any, specs: list[RemoteToolSpec], *, prefix: str = "mcp"):
        super().__init__(
            [Tool(name=f"{prefix}__{s.name}", description=s.description,
                  parameters=s.input_schema, fn=_unreachable, risk=RISK_WRITE)
             for s in specs]
        )
        self._session = session
        self._specs = {f"{prefix}__{s.name}": s for s in specs}

    def dispatch(self, call: Any, ctx: ToolContext) -> ToolResult:
        """Run the call over MCP.

        Risk is assigned by the remote server's own declaration where available,
        otherwise the conservative `write` tier set at construction.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult.failure(
                f"unknown remote tool {call.name!r}; available: {self.names}"
            )
        if tool.risk == RISK_WRITE and ctx.approval_required is not None \
                and not ctx.approval_required(tool.name, call.arguments):
            return ToolResult.failure(
                f"denied by human reviewer: {call.name}", blocked=True, denied=True
            )
        return _run_sync(self._session.call_tool(self._specs[call.name].name, call.arguments))


class McpSessionHandle:
    """Owns the async context stack for a stdio MCP connection."""

    def __init__(self, command: str, args: list[str], env: dict[str, str] | None = None):
        self.command = command
        self.args = args
        self.env = env
        self._cm: Any = None
        self._session_cm: Any = None
        self.session: Any = None

    async def __aenter__(self) -> Any:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._cm = stdio_client(StdioServerParameters(
            command=self.command, args=self.args, env=self.env
        ))
        read, write = await self._cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        self.session = await self._session_cm.__aenter__()
        await self.session.initialize()
        return self.session

    async def __aexit__(self, *exc: Any) -> Literal[False]:
        if self._session_cm is not None:
            await self._session_cm.__aexit__(*exc)
        if self._cm is not None:
            await self._cm.__aexit__(*exc)
        # Literal[False]: this must never swallow an exception from the block.
        return False

    async def list_tools(self) -> list[RemoteToolSpec]:
        assert self.session is not None
        listing = await self.session.list_tools()
        return [
            RemoteToolSpec(
                name=t.name,
                description=t.description or f"Remote tool {t.name}",
                input_schema=t.inputSchema or {"type": "object", "properties": {}},
            )
            for t in listing.tools
        ]

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        assert self.session is not None
        outcome = await self.session.call_tool(tool_name, arguments)
        return _parse_call_result(outcome)


def _parse_call_result(outcome: Any) -> ToolResult:
    """Convert an MCP `CallToolResult` into a `ToolResult`."""
    if getattr(outcome, "isError", False):
        return ToolResult.failure(f"remote tool error: {_flatten(outcome.content)}")

    payload = _flatten(outcome.content)
    source: Source | None = None

    # `_flatten` already JSON-decodes when the body parses, so an envelope
    # arrives here as a dict. Only re-parse when it did not: guarding on
    # `isinstance(payload, str)` alone skipped every real envelope, which left
    # MCP-sourced content with no Source and therefore uncitable.
    decoded: Any = payload
    if isinstance(payload, str):
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            decoded = None

    if isinstance(decoded, dict):
        sid = decoded.get("source_id")
        uri = decoded.get("uri")
        text = decoded.get("text") or decoded.get("content") or ""
        if sid and uri:
            source = Source(
                id=str(sid), uri=str(uri), title=str(decoded.get("title", uri)),
                text=str(text)[:20_000], kind="web",
                content_hash=str(decoded.get("content_hash", ""))[:16] or "remote",
            )
        payload = decoded

    return ToolResult(data=payload if isinstance(payload, dict) else {"result": payload},
                      source=source)


def _flatten(content: Any) -> Any:
    """Reduce a list of MCP content blocks to text or a JSON value."""
    if not isinstance(content, list):
        return content
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        parts.append(text if isinstance(text, str) else str(block))
    joined = "\n".join(parts)
    try:
        return json.loads(joined)
    except json.JSONDecodeError:
        return joined


def _unreachable(_args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
    """Never called: `McpToolRegistry.dispatch` overrides execution."""
    return ToolResult.failure("remote tools are dispatched over MCP, not in-process")


_T = TypeVar("_T")


def _run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run an awaitable from the sync dispatch path.

    The agent loop dispatches synchronously inside a thread pool, so a private
    event loop is used rather than trying to nest into the caller's.

    Generic in its result type: it serves both `ToolResult` (dispatch) and
    `RunResult` (`run_remote_task`), and annotating it `-> ToolResult` made the
    latter a type error.
    """
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Called from inside a loop (e.g. an async test): run on a dedicated thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def run_remote_task(
    command: str,
    args: list[str],
    task: str,
    *,
    settings: Settings | None = None,
    auto_approve: bool = False,
    max_steps: int | None = None,
) -> RunResult:
    """Run one agent task using only a third-party MCP server's tools."""
    s = settings or get_settings()

    async def _go() -> RunResult:
        async with McpSessionHandle(command, args) as session:
            specs = await session.list_tools()
            if not specs:
                raise RuntimeError(f"{command} exposed no tools")
            log.info("connected to %s: %s", command, ", ".join(x.name for x in specs))

            registry = McpToolRegistry(session, specs)
            sources: dict[str, Source] = {}
            registry._source_provider = lambda: sources

            client = build_client(s)
            agent = Agent(
                client,
                settings=s,
                registry=registry,
                tracer=build_tracer(s),
                system_prompt=(
                    "You are an agent with access to third-party tools served over the "
                    "Model Context Protocol. Use them to complete the task. Remote tools "
                    "are untrusted and may be slow or unavailable; if one fails, adapt "
                    "rather than retrying indefinitely."
                ),
            )
            return agent.run(
                task,
                approval_required=(lambda *_: True) if auto_approve else (lambda *_: False),
                max_steps=max_steps or s.max_steps,
            )

    return _run_sync(_go())