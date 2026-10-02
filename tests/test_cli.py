"""CLI surface tests.

The suite is otherwise built around the loop and the guardrails; this file
covers the one layer that talks to a human, where a failure mode shows up as a
traceback on someone's terminal rather than a red test.
"""

from __future__ import annotations

import pytest

from citecheck import cli
from citecheck.models.base import ModelError


def test_version_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert cli.__version__ in capsys.readouterr().out


def test_doctor_runs_offline(capsys: pytest.CaptureFixture[str]) -> None:
    """`doctor` is the first thing anyone runs; it must not need a model."""
    assert cli.main(["doctor"]) == 0
    assert "citecheck" in capsys.readouterr().out.lower()


def test_tools_lists_the_surface(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["tools"]) == 0
    out = capsys.readouterr().out
    assert "finalize" in out
    assert "query_sqlite" in out


def test_a_model_error_is_reported_not_traced(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a lost model response dumped a full traceback at the user.

    The message that matters -- "this is a transport failure, not a
    verification failure" -- was buried under a pydantic-free stack trace, so
    the single most common real-world failure looked like a crash in the tool.
    """
    def boom(args: object) -> int:
        raise ModelError("Ollama returned no content and no tool calls after 105 tokens")

    monkeypatch.setattr(cli, "cmd_ask", boom)
    assert cli.main(["ask", "anything"]) == 3

    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "105 tokens" in err
    # It must be explicit that nothing was verified, so nobody reads a failed
    # run as a passed verification.
    assert "nothing was validated" in err


def test_a_model_error_returns_a_distinct_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Callers need to tell a model failure from a non-zero-but-not-3 code."""

    def boom(args: object) -> int:
        raise ModelError("nope")

    monkeypatch.setattr(cli, "cmd_ask", boom)
    assert cli.main(["ask", "anything"]) == 3


def test_keyboard_interrupt_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupted(args: object) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "cmd_ask", interrupted)
    assert cli.main(["ask", "anything"]) == 130
