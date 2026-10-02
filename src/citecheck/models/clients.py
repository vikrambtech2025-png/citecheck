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


def _ollama_native_messages(messages: list[Message]) -> list[dict]:
    """Serialise messages for Ollama's native `/api/chat`.

    One field differs from the OpenAI wire format: a replayed assistant
    tool call must carry `function.arguments` as a decoded **object**. Ollama's
    chat template renders that value straight into the prompt and re-parses it,
    so a JSON *string* arrives as `"{\"sql\": \"...\"}"` and its JSON scanner
    rejects the whole request with `Value looks like object, but can't find
    closing '}' symbol` -- an HTTP 400 on a conversation that is perfectly
    valid. The `/v1` shim wants the string form, so this cannot be shared with
    `OpenAICompatClient`.
    """
    out: list[dict] = []
    for m in messages:
        d = m.to_dict()
        if d.get("tool_calls"):
            for tc in d["tool_calls"]:
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    tc["function"]["arguments"] = _safe_json(args)
        out.append(d)
    return out


class OllamaClient(ModelClient):
    """Talks to Ollama's native `/api/chat` endpoint.

    Ollama also exposes an OpenAI-compatible `/v1` surface, and using that was
    the original choice here so one parser would cover Ollama, vLLM, LM Studio
    and OpenAI alike. It had to be abandoned: on this build the `/v1` shim
    silently discards a tool call whose arguments are large, returning
    `{"content": "", "tool_calls": []}` after generating the tokens. Reproduced
    with a ~49-token `finalize` argument; a ~36-token one survived. The agent
    cannot tell that apart from a model that chose to say nothing, so the loss
    showed up as `empty_response` and was charged to the model in the eval --
    25% of the suite.

    The native endpoint returns the same tool calls correctly, so it is what this
    client speaks. `OpenAICompatClient` still covers vLLM and LM Studio, which
    do not have this defect.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.s = settings or get_settings()
        self.model_name = self.s.ollama_model

    @property
    def _endpoint(self) -> str:
        base = self.s.ollama_base_url.rstrip("/")
        # Config may point at either `/v1` or the server root; both appear in
        # the wild, so normalise rather than assume.
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        return f"{base}/api/chat"

    def _post(self, payload: dict) -> dict:
        import json
        import urllib.error
        import urllib.request

        req = urllib.request.Request(
            self._endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.s.request_timeout_s) as fh:
                return json.loads(fh.read())
        except urllib.error.HTTPError as e:  # pragma: no cover - network path
            raise ModelError(f"Ollama returned HTTP {e.code}: {e.read()[:400]!r}") from e
        except Exception as e:  # pragma: no cover - network path
            raise ModelError(f"Ollama call failed: {e}") from e

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
            "messages": _ollama_native_messages(messages),
            "stream": False,
            "options": {
                "temperature": self.s.temperature if temperature is None else temperature,
                "num_predict": self.s.max_output_tokens if max_tokens is None else max_tokens,
            },
        }
        if tools:
            payload["tools"] = tools

        start = time.perf_counter()
        data = self._post(payload)
        latency = time.perf_counter() - start

        msg = data.get("message") or {}
        content = (msg.get("content") or "").strip()

        calls: list[ToolCall] = []
        for tc in (msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            # Native Ollama hands back a decoded object; the OpenAI shape uses a
            # JSON string. Accept both so this parser is not the fragile part.
            calls.append(
                ToolCall(
                    id=tc.get("id") or "",
                    name=fn.get("name") or "",
                    arguments=_safe_json(fn.get("arguments")),
                )
            )

        input_tokens = int(data.get("prompt_eval_count") or 0)
        output_tokens = int(data.get("eval_count") or 0)
        usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)

        if not content and not calls and output_tokens > 0:
            # A model that stops cleanly without saying anything costs ~0 tokens.
            # Generating real tokens and returning neither content nor a tool call
            # means the response was lost in transport, and reporting that as an
            # empty model turn would hide a client bug behind a model metric.
            raise ModelError(
                f"Ollama returned no content and no tool calls after generating "
                f"{output_tokens} tokens (finish_reason={data.get('done_reason')!r}). "
                f"This is a lost response, not a silent model -- do not count it "
                f"as an empty turn."
            )

        return Turn(
            content=content,
            tool_calls=calls,
            usage=usage,
            finish_reason=data.get("done_reason") or "stop",
            latency_s=latency,
            model=self.s.ollama_model,
        )

    def count_tokens(self, text: str) -> int:
        return _estimate_tokens(text)

    def health_check(self) -> tuple[bool, str]:
        try:
            self._post({"model": self.s.ollama_model, "messages": [], "stream": False})
            return True, f"{self.s.ollama_model} reachable at {self._endpoint}"
        except ModelError as e:
            return False, f"cannot reach Ollama at {self.s.ollama_base_url}: {e}"


class OpenAICompatClient(ModelClient):
    """Any OpenAI-compatible server (vLLM, LM Studio, Together, ...).

    Uses the OpenAI SDK directly rather than inheriting `OllamaClient`, because
    those servers speak `/v1/chat/completions`, not Ollama's `/api/chat`. It
    carries the same guard against a silently-dropped tool call: a completion
    that generated tokens but returned neither content nor a call is a
    transport loss, and reporting it as an empty model turn would hide a client
    bug behind a model metric.
    """

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
            "model": self.s.openai_compat_model,
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
            raise ModelError(f"OpenAI-compatible call failed: {e}") from e
        latency = time.perf_counter() - start

        msg = resp.choices[0].message
        calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=_safe_json(tc.function.arguments))
            for tc in (getattr(msg, "tool_calls", None) or [])
        ]
        content = (msg.content or "").strip()

        u = getattr(resp, "usage", None)
        usage = Usage(
            input_tokens=getattr(u, "prompt_tokens", 0) or 0,
            output_tokens=getattr(u, "completion_tokens", 0) or 0,
        )

        if not content and not calls and usage.output_tokens > 0:
            raise ModelError(
                f"{self.s.openai_compat_base_url} returned no content and no tool "
                f"calls after generating {usage.output_tokens} tokens. The "
                f"response was lost in transport, not withheld by the model."
            )

        return Turn(
            content=content,
            tool_calls=calls,
            usage=usage,
            finish_reason=resp.choices[0].finish_reason,
            latency_s=latency,
            model=self.s.openai_compat_model,
        )

    def count_tokens(self, text: str) -> int:
        return _estimate_tokens(text)

    def health_check(self) -> tuple[bool, str]:
        try:
            names = [m.id for m in self._client.models.list().data]
            if self.s.openai_compat_model not in names:
                return False, f"{self.s.openai_compat_model} not served. Available: {names or 'none'}"
            return True, f"{self.s.openai_compat_model} ready"
        except Exception as e:
            return False, f"cannot reach {self.s.openai_compat_base_url}: {e}"


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


def _safe_json(raw: str | dict | None) -> dict:
    """Models sometimes emit prose around JSON. Recover what we can.

    Accepts a dict as well as a JSON string, because providers disagree: the
    OpenAI wire format sends `arguments` as a string, Ollama's native API sends
    it already decoded. Rejecting the decoded form would drop every tool call
    from the native endpoint with an `AttributeError` rather than a diagnosis.
    """
    import json
    import re

    if isinstance(raw, dict):
        return raw

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