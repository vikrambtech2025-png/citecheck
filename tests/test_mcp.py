"""MCP: remote tool registry, result conversion, and the sync/async bridge.

No server is launched here. These tests cover the parts that are pure Python --
schema conversion, result unwrapping, risk gating -- which is where MCP bugs
actually live. A transport bug shows up as a hang, not an assertion.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from citecheck.config import Settings
from citecheck.mcp.client import (
    McpToolRegistry,
    RemoteToolSpec,
    _flatten,
    _parse_call_result,
    _run_sync,
)
from citecheck.models.base import RISK_WRITE, ToolCall
from citecheck.tools.base import ToolContext, ToolResult


class _Block:
    def __init__(self, text: str) -> None:
        self.text = text


class _Outcome:
    def __init__(self, content: object, is_error: bool = False) -> None:
        self.content = content
        self.isError = is_error


class _FakeSession:
    """Stands in for an `mcp.ClientSession`.

    `call_tool` must be a coroutine function: the real client's method is, and
    `McpToolRegistry.dispatch` pipes its return value through `_run_sync`.
    """

    def __init__(self, result: ToolResult | None = None) -> None:
        self.result = result or ToolResult(data={"ok": True})
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict) -> ToolResult:
        self.calls.append((name, arguments))
        return self.result


def _specs() -> list[RemoteToolSpec]:
    return [
        RemoteToolSpec(
            name="fetch",
            description="Fetch a URL",
            input_schema={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        ),
    ]


# --------------------------------------------------------------------------
# _flatten
# --------------------------------------------------------------------------
def test_flatten_joins_text_blocks() -> None:
    assert _flatten([_Block("a"), _Block("b")]) == "a\nb"


def test_flatten_parses_json_blocks_into_a_value() -> None:
    assert _flatten([_Block('{"n": 1}')]) == {"n": 1}


def test_flatten_passes_through_non_lists() -> None:
    assert _flatten("plain") == "plain"


# --------------------------------------------------------------------------
# _parse_call_result
# --------------------------------------------------------------------------
def test_call_result_unwraps_our_own_envelope_into_a_source() -> None:
    payload = json.dumps({
        "source_id": "Sabc1234",
        "uri": "https://example.com/a",
        "title": "A page",
        "text": "The Transformer was introduced in 2017.",
        "content_hash": "abcdef1234567890",
    })
    result = _parse_call_result(_Outcome([_Block(payload)]))

    assert result.ok
    assert result.source is not None
    assert result.source.id == "Sabc1234"
    assert result.source.uri == "https://example.com/a"


def test_call_result_without_an_envelope_has_no_source() -> None:
    result = _parse_call_result(_Outcome([_Block("just some prose")]))
    assert result.ok
    assert result.source is None


def test_call_result_error_becomes_a_failure() -> None:
    result = _parse_call_result(_Outcome([_Block("upstream exploded")], is_error=True))
    assert not result.ok
    assert "upstream exploded" in result.error


def test_call_result_truncates_enormous_remote_text() -> None:
    payload = json.dumps({"source_id": "S1", "uri": "u", "text": "x" * 50_000})
    result = _parse_call_result(_Outcome([_Block(payload)]))
    assert len(result.source.text) <= 20_000


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
def test_remote_tools_are_namespaced() -> None:
    reg = McpToolRegistry(_FakeSession(), _specs())
    assert reg.names == ["mcp__fetch"]


def test_remote_tools_default_to_write_risk() -> None:
    # A server we did not write may have side effects; read-tier would be unsafe.
    reg = McpToolRegistry(_FakeSession(), _specs())
    assert reg.get("mcp__fetch").risk == RISK_WRITE


def test_remote_schemas_are_valid() -> None:
    for schema in McpToolRegistry(_FakeSession(), _specs()).schemas():
        fn = schema["function"]
        assert fn["name"] == "mcp__fetch"
        assert fn["parameters"]["type"] == "object"
        assert "url" in fn["parameters"]["properties"]


def test_unknown_remote_tool_lists_the_available_ones(seeded: Settings) -> None:
    reg = McpToolRegistry(_FakeSession(), _specs())
    result = reg.dispatch(
        ToolCall(id="1", name="mcp__nope", arguments={}),
        ToolContext(settings=seeded),
    )
    assert not result.ok
    assert "mcp__fetch" in result.error


def test_remote_call_is_denied_without_approval(seeded: Settings) -> None:
    reg = McpToolRegistry(_FakeSession(), _specs())
    result = reg.dispatch(
        ToolCall(id="1", name="mcp__fetch", arguments={"url": "https://example.com"}),
        ToolContext(settings=seeded, approval_required=lambda *_: False),
    )
    assert not result.ok
    assert result.meta.get("denied") is True


def test_remote_call_proceeds_when_approved(seeded: Settings) -> None:
    session = _FakeSession()
    reg = McpToolRegistry(session, _specs())
    result = reg.dispatch(
        ToolCall(id="1", name="mcp__fetch", arguments={"url": "https://example.com"}),
        ToolContext(settings=seeded, approval_required=lambda *_: True),
    )
    assert result.ok
    # The un-prefixed name must be what goes over the wire.
    assert session.calls == [("fetch", {"url": "https://example.com"})]


# --------------------------------------------------------------------------
# _run_sync
# --------------------------------------------------------------------------
def test_run_sync_works_outside_a_loop() -> None:
    async def coro() -> int:
        return 7

    assert _run_sync(coro()) == 7


def test_run_sync_works_inside_a_running_loop() -> None:
    """The dispatch path is called from worker threads that may hold a loop."""

    async def inner() -> int:
        async def coro() -> int:
            return 11

        return _run_sync(coro())

    assert asyncio.run(inner()) == 11


# --------------------------------------------------------------------------
# server side
# --------------------------------------------------------------------------
def test_server_exposes_the_safe_subset(seeded: Settings) -> None:
    from citecheck.mcp.server import DEFAULT_MCP_TOOLS, list_exposed

    body = list_exposed()
    assert body["server"] == "citecheck-tools"
    # delete_file needs an interactive decision; stdio cannot present one.
    assert "delete_file" not in body["exposed"]
    assert set(body["exposed"]) <= set(DEFAULT_MCP_TOOLS)
    assert "delete_file" in body["withheld"]


def test_server_schemas_are_well_formed(seeded: Settings) -> None:
    from citecheck.mcp.server import list_exposed

    for schema in list_exposed()["tools"]:
        # MCP uses the flat {name, description, inputSchema} shape, not the
        # OpenAI {"type": "function"} envelope.
        assert schema["name"] and schema["description"]
        assert schema["inputSchema"]["type"] == "object"


def test_server_class_resolves_on_the_installed_sdk() -> None:
    """mcp 2.x renamed FastMCP -> MCPServer; the shim must handle both.

    Without this, `citecheck mcp serve` breaks on whichever major is not the
    one hard-coded, making pyproject's `mcp>=1.2` bound a lie.
    """
    from citecheck.mcp.server import _server_cls

    cls = _server_cls()
    assert cls.__name__ in {"MCPServer", "FastMCP"}
    assert hasattr(cls, "add_tool")
    assert hasattr(cls, "run_stdio_async")


def test_register_binds_one_tool(seeded: Settings) -> None:
    from citecheck.mcp.server import _register

    registered: list[str] = []

    class _FakeServer:
        def add_tool(self, fn, name=None, description=None):
            registered.append(name)

    class _FakeTool:
        description = "d"

        def fn(self, args, ctx):
            return ToolResult(data={})

    _register(_FakeServer(), "read_file", _FakeTool(), ToolContext(settings=seeded))
    assert registered == ["read_file"]


# --------------------------------------------------------------------------
# fixtures sanity
# --------------------------------------------------------------------------
def test_remote_spec_defaults_are_supplied_by_the_caller() -> None:
    # list_tools() is responsible for defaulting; this pins the contract.
    spec = RemoteToolSpec(name="x", description="d", input_schema={"type": "object"})
    assert spec.input_schema["type"] == "object"


@pytest.mark.parametrize("payload", ["{}", "not json", "[1, 2, 3]"])
def test_flatten_survives_odd_payloads(payload: str) -> None:
    assert _flatten([_Block(payload)]) is not None
