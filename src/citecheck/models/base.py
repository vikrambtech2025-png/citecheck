"""Provider-agnostic message/tool types and the `ModelClient` interface.

Everything above this layer (agent loop, tools, evals) depends only on these
types. That is what lets `eval/bench.py` sweep models without touching agent code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]

# Tools whose arguments could destroy state if the model is wrong.
RISK_READ = "read"
RISK_WRITE = "write"
RISK_DESTRUCTIVE = "destructive"


@dataclass(slots=True)
class ToolCall:
    """A model's request to invoke a tool."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Set only on role="tool" -- echoes ToolCall.id back to the model.
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {
                    "name": tc.name, "arguments": json.dumps(tc.arguments)}}
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d

    @classmethod
    def system(cls, text: str) -> Message:
        return cls(role="system", content=text)

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role="user", content=text)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> Message:
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool_result(cls, call: ToolCall, result: Any, *, name: str | None = None) -> Message:
        """Tool output is always a JSON string; the model cannot parse objects."""
        body = result if isinstance(result, str) else json.dumps(result, default=str)
        return cls(
            role="tool",
            content=body,
            tool_call_id=call.id,
            name=name or call.name,
        )


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens)


@dataclass(slots=True)
class Turn:
    """One model response."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    latency_s: float = 0.0
    model: str = ""

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class ModelError(RuntimeError):
    """Raised on unrecoverable model failure (auth, quota, malformed response)."""