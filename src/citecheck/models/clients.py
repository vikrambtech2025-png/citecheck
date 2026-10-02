"""`ModelClient` interface plus the factory that resolves a backend from config."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from citecheck.config import Settings, get_settings
from citecheck.models.base import Message, ModelError, ToolCall, Turn, Usage

log = logging.getLogger(__name__)


class ModelClient(ABC):
    """The only surface the agent loop depends on."""

    #: Human-readable identifier used in traces and eval result tables.
    model_name: str = "unknown"

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Turn:
        """Return one assistant turn. May request tool calls."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Cheap token estimate. Used for budget accounting, not billing."""

    def health_check(self) -> tuple[bool, str]:
        """Return (ok, detail). Overridden by clients that can probe their backend."""
        return True, "no health check implemented"


def _estimate_tokens(text: str) -> int:
    # ~4 chars/token is close enough for budget ceilings on English text.
    return max(1, len(text) // 4)


class OllamaClient(ModelClient):
    """Talks to Ollama's OpenAI-compatible endpoint.

    Using the OpenAI wire format (rather than Ollama's native API) is deliberate:
    one parser covers Ollama, vLLM, LM Studio, and OpenAI itself.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self.model_name = self.s.ollama_model
        self._client = self._build()

    def _build(self):
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise ModelError(
                "The `openai` package is required for the Ollama/OpenAI-compatible "
                "backend. Install it with:  uv sync --extra openai"
            ) from e
        return openai.OpenAI(
            base_url=self.s.ollama_base_url,
            api_key="ollama",  # required by the SDK, ignored by Ollama
            timeout=self.s.request_timeout_s,
            max_retries=self.s.max_retries,
        )

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Turn:
        import time

        payload: dict = {
            "model": self.s.ollama_model,
            "messages": [m.to_dict() for m in messages],
            "temperature": self.s.temperature if temperature is None else temperature,
            "max_tokens": self.s.max_output_tokens if max_tokens is None else max_tokens,
        }
        if tools:
            payload["tools"] = tools

        start = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(**payload)
        except Exception as e:
            raise ModelError(f"Ollama call failed: {e}") from e
        latency = time.perf_counter() - start

        msg = resp.choices[0].message
        calls: list[ToolCall] = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            fn = tc.function
            calls.append(ToolCall(id=tc.id, name=fn.name, arguments=_safe_json(fn.arguments)))

        u = getattr(resp, "usage", None)
        usage = Usage(
            input_tokens=getattr(u, "prompt_tokens", 0) or 0,
            output_tokens=getattr(u, "completion_tokens", 0) or 0,
        )
        if usage.total == 0:  # Ollama omits usage on some builds
            usage = Usage(_estimate_tokens(payload["messages"].__str__()), 0)

        return Turn(
            content=(msg.content or "").strip(),
            tool_calls=calls,
            usage=usage,
            finish_reason=resp.choices[0].finish_reason,
            latency_s=latency,
            model=self.s.ollama_model,
        )

    def count_tokens(self, text: str) -> int:
        return _estimate_tokens(text)

    def health_check(self) -> tuple[bool, str]:
        try:
            models = self._client.models.list()
            names = [m.id for m in models.data]
            if self.s.ollama_model not in names:
                return False, f"{self.s.ollama_model} not pulled. Available: {names or 'none'}"
            return True, f"{self.s.ollama_model} ready"
        except Exception as e:
            return False, f"cannot reach {self.s.ollama_base_url}: {e}"


class OpenAICompatClient(OllamaClient):
    """Any OpenAI-compatible server. Identical wire format, different URL/model."""

    def __init__(self, settings: Settings | None = None) -> None:
        s = settings or get_settings()
        self.s = s
        self.model_name = s.openai_compat_model
        self._client = self._build()

    def _build(self):
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise ModelError("Install with:  uv sync --extra openai") from e
        return openai.OpenAI(
            base_url=self.s.openai_compat_base_url,
            api_key=self.s.openai_compat_api_key,
            timeout=self.s.request_timeout_s,
            max_retries=self.s.max_retries,
        )

    def chat(self, messages, *, tools=None, temperature=None, max_tokens=None) -> Turn:
        # Reuse the wire handling but point at the configured model/endpoint.
        original = self.s.ollama_model
        self.s.ollama_model = self.s.openai_compat_model
        try:
            return super().chat(
                messages, tools=tools, temperature=temperature, max_tokens=max_tokens
            )
        finally:
            self.s.ollama_model = original


class AnthropicClient(ModelClient):
    """Optional hosted backend, used for a polished demo run.

    Anthropic's native API differs in two ways we normalise here: a separate
    `system` parameter, and a `tool_use` block type instead of `tool_calls`.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self.model_name = self.s.anthropic_model
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ModelError("Install with:  uv sync --extra anthropic") from e
        if not self.s.anthropic_api_key:
            raise ModelError("CITECHECK_ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic(
            api_key=self.s.anthropic_api_key, timeout=self.s.request_timeout_s, max_retries=0
        )

    def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Turn:
        import time

        system = "\n\n".join(m.content for m in messages if m.role == "system")
        # Anthropic takes `system` separately and expects tool results as user-role
        # blocks, so normalise before sending rather than after.
        convo = _anthropic_tool_results([m.to_dict() for m in messages if m.role != "system"])
        payload: dict = {
            "model": self.s.anthropic_model,
            "messages": convo,
            "max_tokens": self.s.max_output_tokens if max_tokens is None else max_tokens,
        }
        if system:
            payload["system"] = system
        if temperature is not None or self.s.temperature is not None:
            payload["temperature"] = self.s.temperature if temperature is None else temperature
        if tools:
            payload["tools"] = [
                {"name": t["function"]["name"], "description": t["function"].get("description", ""),
                 "input_schema": t["function"].get("parameters", {"type": "object"})}
                for t in tools
            ]

        start = time.perf_counter()
        try:
            resp = self._client.messages.create(**payload)
        except Exception as e:
            raise ModelError(f"Anthropic call failed: {e}") from e
        latency = time.perf_counter() - start

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                args = dict(block.input or {})
                calls.append(ToolCall(id=block.id, name=block.name, arguments=args))

        # Tool results must come back as user-role blocks in the native API.
        return Turn(
            content="\n".join(text_parts).strip(),
            tool_calls=calls,
            usage=Usage(input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens),
            finish_reason=resp.stop_reason,
            latency_s=latency,
            model=self.s.anthropic_model,
        )

    def count_tokens(self, text: str) -> int:
        return _estimate_tokens(text)


class MockClient(ModelClient):
    """Deterministic scripted backend for tests and CI.

    Consumes a queue of turns. Never touches the network, so the agent loop,
    guardrails, and graders are all testable offline.
    """

    def __init__(self, script: list[Turn] | None = None, model_name: str = "mock") -> None:
        self.model_name = model_name
        self._script = list(script or [])
        self.calls: list[list[Message]] = []

    def queue(self, *turns: Turn) -> MockClient:
        self._script.extend(turns)
        return self

    def chat(self, messages, *, tools=None, temperature=None, max_tokens=None) -> Turn:
        self.calls.append(list(messages))
        if not self._script:
            return Turn(content="No further steps.", model=self.model_name)
        return self._script.pop(0)

    def count_tokens(self, text: str) -> int:
        return _estimate_tokens(text)


def _safe_json(raw: str | None) -> dict:
    """Models sometimes emit prose around JSON. Recover what we can."""
    import json
    import re

    # Strip before the emptiness check: a model that returns only whitespace
    # should look empty, not like unparseable content.
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {"value": val}
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            val = json.loads(m.group(0))
            return val if isinstance(val, dict) else {"value": val}
        except json.JSONDecodeError:
            pass
    return {"_raw": raw}


def _anthropic_tool_results(convo: list[dict]) -> list[dict]:
    """Convert `role=tool` messages into Anthropic's `tool_result` user blocks."""
    out: list[dict] = []
    for m in convo:
        if m.get("role") == "tool":
            out.append(
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m["tool_call_id"],
                        "content": m.get("content", ""),
                    }],
                }
            )
        else:
            out.append(m)
    return out


def build_client(settings: Settings | None = None) -> ModelClient:
    """Resolve `settings.model_backend` into a concrete client."""
    s = settings or get_settings()
    backend = s.model_backend.lower()
    if backend == "ollama":
        return OllamaClient(s)
    if backend == "openai_compat":
        return OpenAICompatClient(s)
    if backend == "openai":
        return OllamaClient(s.model_copy(update={"ollama_base_url": "https://api.openai.com/v1"}))
    if backend == "anthropic":
        return AnthropicClient(s)
    if backend == "mock":
        return MockClient()
    raise ModelError(f"Unknown model_backend: {backend!r}")