"""MCP server: exposes CiteCheck's tools over the Model Context Protocol.

Same `Tool` objects the built-in agent uses, re-served over stdio. That reuse
is the point -- it demonstrates that the tool layer is transport-agnostic.

Run with:  citecheck mcp serve
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from citecheck.config import Settings, get_settings
from citecheck.tools.base import ToolContext
from citecheck.tools.registry import build_registry

log = logging.getLogger(__name__)

#: Tool names surfaced over MCP. `delete_file` is excluded on purpose: an MCP
#: client cannot present an interactive approval prompt reliably, so exposing an
#: unguarded destructive tool over the protocol would be unsafe by construction.
DEFAULT_MCP_TOOLS = [
    "workspace_tree",
    "list_dir",
    "read_file",
    "query_sqlite",
    "search_web",
    "fetch_url",
    "save_report",
]


def _server_cls() -> type[Any]:
    """Resolve the MCP server class across SDK majors.

    mcp 2.x renamed `FastMCP` to `MCPServer`, with the same `add_tool` /
    `run_stdio_async` surface. `pyproject` allows `mcp>=1.2`, so both spellings
    must work rather than assuming whichever major happens to be installed.
    """
    try:
        from mcp.server.mcpserver import MCPServer

        return MCPServer
    except ImportError:
        pass
    try:
        # mcp 1.x spelling. Statically absent under 2.x, which is exactly why
        # the 2.x branch above is tried first.
        from mcp.server.fastmcp import FastMCP  # type: ignore[attr-defined]

        return FastMCP
    except ImportError as e:
        raise SystemExit(
            "MCP server unavailable: install the dependency with `uv sync` "
            "(pyproject declares `mcp>=1.2`)."
        ) from e


async def serve(
    settings: Settings | None = None,
    *,
    tool_names: list[str] | None = None,
) -> None:
    """Run the MCP server on stdio until the client disconnects."""
    s = settings or get_settings()
    names = tool_names or DEFAULT_MCP_TOOLS
    registry = build_registry(source_provider=lambda: {})
    ctx = ToolContext(settings=s, run_id="mcp")

    mcp = _server_cls()("citecheck-tools")

    for name in names:
        tool = registry.get(name)
        if tool is None:
            log.warning("unknown tool %r; skipping", name)
            continue
        _register(mcp, name, tool, ctx)

    log.info("citecheck MCP server ready with %d tools: %s", len(names), ", ".join(names))
    await mcp.run_stdio_async()


def _register(mcp: Any, name: str, tool: Any, ctx: ToolContext) -> None:
    """Bind one tool to the MCP server, preserving its JSON schema."""

    async def handler(**kwargs: Any) -> str:
        result = tool.fn(kwargs, ctx)
        if not result.ok:
            # MCP surfaces tool errors as text; keep it machine-readable.
            return json.dumps({"status": "error", "tool": name, "error": result.error})
        payload: dict[str, Any] = {"status": "ok"}
        if isinstance(result.data, dict):
            payload.update(result.data)
        else:
            payload["result"] = result.data
        return json.dumps(payload, default=str)

    handler.__name__ = name
    mcp.add_tool(
        handler,
        name=name,
        description=tool.description,
    )


def list_exposed() -> dict[str, Any]:
    """Introspection helper for the CLI (`citecheck mcp serve --dry-run`)."""
    registry = build_registry()
    exposed = [n for n in DEFAULT_MCP_TOOLS if registry.get(n)]
    return {
        "server": "citecheck-tools",
        "tools": registry.mcp_schemas(),
        "exposed": exposed,
        "withheld": [
            t.name for t in registry.tools if t.name not in DEFAULT_MCP_TOOLS
        ],
        "withheld_reason": (
            "delete_file requires an interactive human approval step that the MCP "
            "stdio transport cannot present; finalize is agent-internal and is "
            "omitted from the server surface."
        ),
    }


def main() -> None:  # pragma: no cover - console entry point
    asyncio.run(serve())


if __name__ == "__main__":  # pragma: no cover
    main()