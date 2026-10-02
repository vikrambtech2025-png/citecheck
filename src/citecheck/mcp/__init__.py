"""Model Context Protocol: serve our tools, and consume someone else's."""

from citecheck.mcp.client import McpSessionHandle, McpToolRegistry, run_remote_task
from citecheck.mcp.server import DEFAULT_MCP_TOOLS, list_exposed, serve

__all__ = [
    "DEFAULT_MCP_TOOLS",
    "McpSessionHandle",
    "McpToolRegistry",
    "list_exposed",
    "run_remote_task",
    "serve",
]