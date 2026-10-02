"""FastAPI surface and the human-in-the-loop approval broker.

The approval path is the reason this module exists, so it gets the most
attention: a destructive tool must block until a human decides, and it must
deny if nobody ever does.
"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi.testclient import TestClient

from citecheck.api import ApprovalBroker, PendingApproval, create_app
from citecheck.config import Settings


@pytest.fixture
def client(seeded: Settings) -> TestClient:
    with TestClient(create_app(seeded)) as c:
        yield c


# --------------------------------------------------------------------------
# introspection endpoints
# --------------------------------------------------------------------------
def test_health_reports_the_mock_backend(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_backend"] == "mock"
    assert body["model_reachable"] is True


def test_tools_endpoint_lists_risk_tiers(client: TestClient) -> None:
    tools = {t["name"]: t for t in client.get("/tools").json()}
    assert "finalize" in tools and "delete_file" in tools
    assert tools["delete_file"]["risk"] == "destructive"
    assert tools["query_sqlite"]["risk"] == "read"
    for t in tools.values():
        assert t["description"], f"{t['name']} has no description"


def test_outbox_is_empty_initially(client: TestClient) -> None:
    assert client.get("/outbox").json() == []


def test_unknown_run_returns_404(client: TestClient) -> None:
    r = client.get("/runs/does-not-exist")
    assert r.status_code == 404


def test_run_listing_starts_empty(client: TestClient) -> None:
    assert client.get("/runs").json() == []


# --------------------------------------------------------------------------
# validation of request bodies
# --------------------------------------------------------------------------
def test_run_request_rejects_a_too_short_task(client: TestClient) -> None:
    # min_length=3 is a real guard: an empty question would burn a model call.
    assert client.post("/runs", json={"task": ""}).status_code == 422


def test_run_request_rejects_out_of_range_max_steps(client: TestClient) -> None:
    assert client.post("/runs", json={"task": "hello", "max_steps": 0}).status_code == 422
    assert client.post("/runs", json={"task": "hello", "max_steps": 999}).status_code == 422


def test_run_request_requires_task(client: TestClient) -> None:
    assert client.post("/runs", json={}).status_code == 422


# --------------------------------------------------------------------------
# approval broker -- the part that must fail closed
# --------------------------------------------------------------------------
def test_broker_resolves_an_approval() -> None:
    broker = ApprovalBroker()
    result: dict[str, bool] = {}

    def worker() -> None:
        result["approved"] = broker.request("run1", "delete_file", {"path": "a.md"})

    t = threading.Thread(target=worker)
    t.start()

    # Wait for the request to register before looking it up.
    deadline = time.time() + 5
    pending = broker.pending()
    while not pending and time.time() < deadline:
        time.sleep(0.01)
        pending = broker.pending()
    assert len(pending) == 1

    broker.resolve(pending[0]["id"], True, reason="looks fine")
    t.join(timeout=5)
    assert result["approved"] is True
    assert broker.pending() == []


def test_broker_denies_when_the_operator_says_no() -> None:
    broker = ApprovalBroker()
    out: dict[str, bool] = {}
    t = threading.Thread(
        target=lambda: out.update(ok=broker.request("r", "delete_file", {}))
    )
    t.start()

    deadline = time.time() + 5
    while not broker.pending() and time.time() < deadline:
        time.sleep(0.01)
    broker.resolve(broker.pending()[0]["id"], False, reason="nope")
    t.join(timeout=5)
    assert out["ok"] is False


def test_broker_times_out_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator who walks away must not leave a destructive call armed."""
    import citecheck.api as api_mod

    monkeypatch.setattr(api_mod, "APPROVAL_TIMEOUT_S", 0.15)
    broker = api_mod.ApprovalBroker()
    out: dict[str, bool] = {}

    t = threading.Thread(
        target=lambda: out.update(ok=broker.request("r", "delete_file", {}))
    )
    t.start()
    t.join(timeout=5)

    assert out["ok"] is False, "timed-out approval must deny"
    # And the record must survive as evidence of the decision.
    assert broker.pending() == []


def test_resolving_an_unknown_approval_raises() -> None:
    with pytest.raises(KeyError):
        ApprovalBroker().resolve("nope", True)


def test_pending_excludes_resolved_items() -> None:
    broker = ApprovalBroker()
    item = PendingApproval(id="a1", run_id="r", tool="delete_file",
                           arguments={}, created_at=time.time())
    broker._pending["a1"] = item
    assert len(broker.pending()) == 1
    item.resolved = True
    assert broker.pending() == []


# --------------------------------------------------------------------------
# outbox confinement -- the API must not be a sandbox bypass
# --------------------------------------------------------------------------
def test_outbox_read_confines_to_the_outbox(client: TestClient, seeded: Settings) -> None:
    (seeded.outbox_dir / "ok.md").write_text("# fine", encoding="utf-8")
    assert client.get("/outbox/ok.md").json()["content"] == "# fine"

    r = client.get("/outbox/../../etc/passwd")
    assert r.status_code in (400, 404)


def test_outbox_read_of_a_missing_file_404s(client: TestClient) -> None:
    assert client.get("/outbox/nope.md").status_code == 404


def test_outbox_listing_shows_saved_reports(client: TestClient, seeded: Settings) -> None:
    (seeded.outbox_dir / "report.md").write_text("body", encoding="utf-8")
    names = [e["name"] for e in client.get("/outbox").json()]
    assert "report.md" in names
