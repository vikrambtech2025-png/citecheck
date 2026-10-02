"""Model client resolution and the offline MockClient.

The point of `ModelClient` is that swapping models is a config change, not a code
change. These tests pin that: the backend string must be the only difference
between a real model and the mock the whole test suite runs on.
"""

from __future__ import annotations

import pytest
from tests.conftest import make_turn

from citecheck.config import Settings
from citecheck.models.base import Message, ModelError, ToolCall
from citecheck.models.clients import (
    AnthropicClient,
    MockClient,
    OllamaClient,
    OpenAICompatClient,
    _safe_json,
    build_client,
)


# --------------------------------------------------------------------------
# build_client
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("ollama", OllamaClient),
        ("openai_compat", OpenAICompatClient),
        ("openai", OllamaClient),          # OpenAI speaks the compat dialect
        ("mock", MockClient),
    ],
)
def test_build_client_maps_every_backend(
    settings: Settings, backend: str, expected: type
) -> None:
    client = build_client(settings.model_copy(update={"model_backend": backend}))
    assert isinstance(client, expected)


def test_build_client_maps_anthropic(settings: Settings) -> None:
    # The anthropic SDK is an optional extra; either it is importable and we
    # get a real client, or we get the install instruction and nothing else.
    pytest.importorskip("anthropic", reason="optional `anthropic` extra not installed")
    client = build_client(settings.model_copy(update={"model_backend": "anthropic"}))
    assert isinstance(client, AnthropicClient)


def test_anthropic_backend_reports_a_missing_dependency(settings: Settings) -> None:
    pytest.importorskip  # noqa: B018 - keep import list tidy
    try:
        import anthropic  # noqa: F401
    except ImportError:
        with pytest.raises(ModelError, match="extra anthropic"):
            build_client(settings.model_copy(update={"model_backend": "anthropic"}))
    else:
        pytest.skip("anthropic extra is installed")


def test_build_client_is_case_insensitive(settings: Settings) -> None:
    s = settings.model_copy(update={"model_backend": "OLLAMA"})
    assert isinstance(build_client(s), OllamaClient)


def test_build_client_rejects_an_unknown_backend(settings: Settings) -> None:
    s = settings.model_copy(update={"model_backend": "gpt-9000"})
    with pytest.raises(ModelError, match="Unknown model_backend"):
        build_client(s)


def test_openai_backend_retargets_the_base_url(settings: Settings) -> None:
    s = settings.model_copy(update={"model_backend": "openai"})
    assert "api.openai.com" in build_client(s).s.ollama_base_url
    # The caller's own value must not be mutated by the retarget.
    assert settings.ollama_base_url != build_client(s).s.ollama_base_url


def test_client_exposes_a_model_name(settings: Settings) -> None:
    assert build_client(settings).model_name


def test_every_client_is_a_model_client(settings: Settings) -> None:
    for backend in ("ollama", "openai_compat", "mock"):
        client = build_client(settings.model_copy(update={"model_backend": backend}))
        assert hasattr(client, "chat") and hasattr(client, "count_tokens")


# --------------------------------------------------------------------------
# MockClient
# --------------------------------------------------------------------------
def test_mock_client_consumes_its_script_in_order() -> None:
    c = MockClient([make_turn("first"), make_turn("second")])
    assert c.chat([]).content == "first"
    assert c.chat([]).content == "second"


def test_mock_client_repeats_a_final_message_when_spent() -> None:
    """Exhausting the script must not raise; the loop has to be able to stop."""
    c = MockClient([make_turn("only")])
    assert c.chat([]).content == "only"
    assert c.chat([]).content == "No further steps."


def test_mock_client_records_the_conversation_it_was_given() -> None:
    c = MockClient([make_turn("x")])
    convo = [Message.system("s"), Message.user("u")]
    c.chat(convo)
    assert len(c.calls[0]) == 2


def test_mock_client_queue_extends_the_script() -> None:
    c = MockClient([make_turn("a")]).queue(make_turn("b"))
    assert c.chat([]).content == "a"
    assert c.chat([]).content == "b"


def test_mock_client_reports_usage() -> None:
    c = MockClient([make_turn("x", tokens=80)])
    assert c.chat([]).usage.total == 80


def test_mock_client_count_tokens_is_deterministic() -> None:
    c = MockClient()
    assert c.count_tokens("hello world") == c.count_tokens("hello world")
    # Never reports 0: a budget counter that stalls at zero never trips.
    assert c.count_tokens("") >= 1
    assert c.count_tokens("x" * 400) > c.count_tokens("x" * 40)


# --------------------------------------------------------------------------
# Anthropic normalisation (must happen before the request is sent)
# --------------------------------------------------------------------------
def test_tool_messages_become_tool_result_user_blocks() -> None:
    from citecheck.models.clients import _anthropic_tool_results

    blocks = _anthropic_tool_results([
        {"role": "tool", "tool_call_id": "c1", "content": "row data"},
    ])
    assert blocks[0]["role"] == "user"
    assert blocks[0]["content"][0]["type"] == "tool_result"
    assert blocks[0]["content"][0]["tool_use_id"] == "c1"


def test_assistant_messages_pass_through_untouched() -> None:
    from citecheck.models.clients import _anthropic_tool_results

    original = [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}]
    assert _anthropic_tool_results(original) == original


# --------------------------------------------------------------------------
# tolerant JSON recovery
# --------------------------------------------------------------------------
def test_safe_json_parses_clean_json() -> None:
    assert _safe_json('{"a": 1}') == {"a": 1}


def test_safe_json_recovers_json_embedded_in_prose() -> None:
    """Local models wrap tool calls in commentary constantly."""
    assert _safe_json('Sure! {"name": "finalize", "arguments": {}} done') == {
        "name": "finalize", "arguments": {}
    }


def test_safe_json_handles_empty_and_none() -> None:
    assert _safe_json(None) == {}
    assert _safe_json("") == {}
    assert _safe_json("   ") == {}


def test_safe_json_keeps_unparseable_text_under_raw() -> None:
    out = _safe_json("no json at all")
    assert out["_raw"] == "no json at all"


def test_safe_json_wraps_a_non_object_payload() -> None:
    assert _safe_json("[1, 2]") == {"value": [1, 2]}


# --------------------------------------------------------------------------
# tool-call round trip through the base types
# --------------------------------------------------------------------------
def test_message_tool_call_serialises_to_the_openai_shape() -> None:
    m = Message.assistant(tool_calls=[ToolCall(id="c1", name="finalize",
                                               arguments={"answer": "x"})])
    d = m.to_dict()
    assert d["tool_calls"][0]["type"] == "function"
    assert d["tool_calls"][0]["function"]["name"] == "finalize"


def test_tool_result_message_echoes_the_call_id() -> None:
    call = ToolCall(id="c9", name="read_file", arguments={})
    m = Message.tool_result(call, {"ok": True})
    assert m.role == "tool"
    assert m.tool_call_id == "c9"
    # Content must be a string: the model cannot parse a JSON object.
    assert isinstance(m.content, str)


def test_usage_adds() -> None:
    from citecheck.models.base import Usage

    assert (Usage(10, 5) + Usage(1, 2)).total == 18
