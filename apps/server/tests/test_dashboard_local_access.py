"""A launcher capability can open local dashboard sessions without revealing keys publicly."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from molto_server.auth import AuthContext


@pytest.fixture
def client(monkeypatch):
    from molto_server.api.dashboard_access_routes import router

    monkeypatch.setenv("MOLTO_SUPERVISED", "application")
    monkeypatch.setenv("MOLTO_LOCAL_ACCESS_TOKEN", "launcher-capability")
    app = FastAPI()
    app.include_router(router)
    app.state.management_auth_provider = lambda: AuthContext(
        "main-key", [], "127.0.0.1"
    )
    return TestClient(app, client=("127.0.0.1", 5000))


def test_local_dashboard_uses_current_effective_key(client):
    headers = {"X-Molto-Local-Access": "launcher-capability"}
    response = client.post("/_internal/dashboard-key", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"key": "main-key"}
    assert response.headers["cache-control"] == "no-store"
    client.app.state.management_auth_provider = lambda: AuthContext(
        "rotated-key", [], "127.0.0.1"
    )
    assert client.post("/_internal/dashboard-key", headers=headers).json() == {
        "key": "rotated-key"
    }


@pytest.mark.parametrize(
    "key", [None, "", "main-key", "inference-key", "wrong-capability"]
)
def test_key_handoff_requires_launcher_capability(client, key):
    headers = {"X-Molto-Local-Access": key} if key is not None else {}
    response = client.post("/_internal/dashboard-key", headers=headers)
    assert response.status_code == 403
    assert "main-key" not in response.text


@pytest.mark.parametrize(
    "setting,value", [("MOLTO_SUPERVISED", ""), ("MOLTO_LOCAL_ACCESS_TOKEN", "")]
)
def test_standalone_server_cannot_hand_off_keys(client, monkeypatch, setting, value):
    monkeypatch.setenv(setting, value)
    assert (
        client.post(
            "/_internal/dashboard-key",
            headers={"X-Molto-Local-Access": "launcher-capability"},
        ).status_code
        == 403
    )


def test_key_handoff_rejects_public_bind_and_forwarding(client):
    headers = {"X-Molto-Local-Access": "launcher-capability"}
    client.app.state.management_auth_provider = lambda: AuthContext(
        "main-key", [], "0.0.0.0"
    )
    assert client.post("/_internal/dashboard-key", headers=headers).status_code == 403
    client.app.state.management_auth_provider = lambda: AuthContext(
        "main-key", [], "127.0.0.1"
    )
    for name in ("Forwarded", "X-Forwarded-For", "X-Real-IP"):
        assert (
            client.post(
                "/_internal/dashboard-key", headers={**headers, name: "127.0.0.1"}
            ).status_code
            == 403
        )
    with TestClient(client.app, client=("203.0.113.5", 5000)) as remote:
        assert (
            remote.post("/_internal/dashboard-key", headers=headers).status_code == 403
        )


def test_first_run_does_not_create_or_replace_key(client):
    client.app.state.management_auth_provider = lambda: AuthContext(
        None, [], "127.0.0.1"
    )
    response = client.post(
        "/_internal/dashboard-key",
        headers={"X-Molto-Local-Access": "launcher-capability"},
    )
    assert response.status_code == 409
    assert "key" not in response.json()
