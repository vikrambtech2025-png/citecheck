"""CiteCheck -- a research agent whose citations are programmatically verified."""

from __future__ import annotations

__version__ = "0.1.0"

from citecheck.agent.loop import Agent, RunResult
from citecheck.config import Settings, get_settings
from citecheck.models.clients import ModelClient, build_client
from citecheck.tools.base import Source, Tool, ToolContext, ToolResult
from citecheck.tools.registry import ToolRegistry, build_registry

__all__ = [
    "Agent",
    "ModelClient",
    "RunResult",
    "Settings",
    "Source",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "__version__",
    "build_client",
    "build_registry",
    "get_settings",
]