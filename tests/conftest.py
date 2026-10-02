"""Shared fixtures.

Every test runs against isolated temp directories and a `MockClient`, so the
suite needs no model, no network, and no API key.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from citecheck.config import Settings, get_settings
from citecheck.models.base import Message, ToolCall, Turn, Usage
from citecheck.models.clients import MockClient
from citecheck.tools.base import Source


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point every path in Settings at tmp_path and reset the settings cache."""
    env = {
        "CITECHECK_WORKSPACE_DIR": str(tmp_path / "workspace"),
        "CITECHECK_OUTBOX_DIR": str(tmp_path / "outbox"),
        "CITECHECK_STATE_DB_PATH": str(tmp_path / "runs" / "state.sqlite3"),
        "CITECHECK_MODEL_BACKEND": "mock",
        "CITECHECK_TRACING_BACKEND": "none",
        "CITECHECK_SEARCH_BACKEND": "off",
        "CITECHECK_MAX_STEPS": "8",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    # .env in the repo root would otherwise leak real values into tests.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


@pytest.fixture
def settings(isolated_env: Path) -> Settings:
    return get_settings()


@pytest.fixture
def seeded(settings: Settings) -> Settings:
    """A workspace containing the sample notes and the knowledge base."""
    from citecheck.seed import seed_all

    seed_all(settings)
    return settings


@pytest.fixture
def db_cite(seeded: Settings) -> str:
    """The source_id the `query_sqlite` tool will return.

    Depends on `seeded` so the knowledge base exists, and derives the id from the
    same helper the tool uses, so a change to the id scheme cannot leave tests
    asserting a citation the validator would reject.
    """
    from citecheck.tools.base import ToolContext
    from citecheck.tools.knowledge import _db_path, database_source_id

    return database_source_id(_db_path(ToolContext(settings=seeded)))


# --------------------------------------------------------------------------
# helper builders
# --------------------------------------------------------------------------
def make_turn(
    content: str = "",
    calls: list[tuple[str, dict]] | None = None,
    *,
    tokens: int = 100,
) -> Turn:
    """Build a scripted model turn. `calls` is [(tool_name, arguments)]."""
    tool_calls = [
        ToolCall(id=f"call_{i}", name=name, arguments=args)
        for i, (name, args) in enumerate(calls or [])
    ]
    return Turn(
        content=content,
        tool_calls=tool_calls,
        usage=Usage(input_tokens=tokens // 2, output_tokens=tokens // 2),
        model="mock",
    )


def make_source(
    source_id: str = "Sabc1234",
    text: str = "The Transformer was introduced by Vaswani et al. in 2017 at NeurIPS.",
    uri: str = "https://example.com/transformer",
    kind: str = "web",
) -> Source:
    import hashlib

    return Source(
        id=source_id,
        uri=uri,
        title="Example",
        text=text,
        kind=kind,
        content_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
    )


@pytest.fixture
def turn_builder():
    return make_turn


@pytest.fixture
def source_builder():
    return make_source


@pytest.fixture
def mock_client() -> MockClient:
    return MockClient()


@pytest.fixture
def tool_result():
    """A successful tool call the agent can be scripted to make."""
    from citecheck.models.base import ToolCall

    def build(name: str, args: dict, call_id: str = "c1") -> ToolCall:
        return ToolCall(id=call_id, name=name, arguments=args)

    return build


__all__ = ["Message", "make_source", "make_turn"]