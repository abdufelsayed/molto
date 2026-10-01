# SPDX-License-Identifier: Apache-2.0
"""Bearer authority is distinct for inference and engine management."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from molto_server.auth import (
    compare_keys,
    fingerprint_key,
    validate_api_key,
    verify_any_api_key,
    verify_api_key,
)
from molto_server.server import create_app
from molto_server.state import ServerState


def test_key_utilities_accept_unicode_without_leaking_key():
    assert compare_keys("caf\u00e9", "caf\u00e9")
    assert not compare_keys("caf\u00e9", "cafe")
    assert verify_api_key("main", "main")
    assert not verify_api_key("", "main")
    assert verify_any_api_key("sub", "main", [SimpleNamespace(key="sub")])
    assert fingerprint_key("secret") != "secret"
    assert len(fingerprint_key("secret")) == 8


@pytest.mark.parametrize(
    ("key", "valid"),
    [("abcd", True), ("abc", False), ("ab cd", False), ("caf\u00e9", False)],
)
def test_key_configuration_validation(key, valid):
    assert validate_api_key(key)[0] is valid


@pytest.fixture
def client(monkeypatch):
    settings = SimpleNamespace(
        auth=SimpleNamespace(
            sub_keys=[SimpleNamespace(key="sub-key")],
            skip_api_key_verification=False,
        ),
        server=SimpleNamespace(host="127.0.0.1"),
    )
    state = ServerState()
    monkeypatch.setattr(state, "api_key", "main-key")
    monkeypatch.setattr(state, "global_settings", settings)
    monkeypatch.setattr(state, "bind_host", "127.0.0.1")
    monkeypatch.setattr(state, "engine_pool", None)
    monkeypatch.setattr(state, "settings_manager", None)
    return TestClient(create_app(state))


def test_management_requires_main_bearer(client):
    assert client.get("/management/v1/state").status_code == 401
    sub = client.get(
        "/management/v1/state", headers={"Authorization": "Bearer sub-key"}
    )
    assert sub.status_code == 401
    assert sub.headers["www-authenticate"] == "Bearer"
    main = client.get(
        "/management/v1/state", headers={"Authorization": "Bearer main-key"}
    )
    assert main.status_code == 503  # authorized; the test pool is uninitialized


def test_management_requires_key_even_when_bound_to_loopback(client, monkeypatch):
    monkeypatch.setattr(client.app.state.server_state, "api_key", None)
    assert client.get("/management/v1/state").status_code == 401


def test_explicit_loopback_skip_allows_management(client):
    client.app.state.server_state.global_settings.auth.skip_api_key_verification = True
    assert client.get("/management/v1/state").status_code == 503
    client.app.state.server_state.bind_host = "0.0.0.0"
    assert client.get("/management/v1/state").status_code == 401


def test_explicit_management_credentials_are_checked_with_local_bypass(client):
    client.app.state.server_state.global_settings.auth.skip_api_key_verification = True
    assert client.get("/management/v1/state").status_code == 503
    for key in ("sub-key", "invalid-key"):
        response = client.get(
            "/management/v1/state", headers={"Authorization": f"Bearer {key}"}
        )
        assert response.status_code == 401
    assert (
        client.get(
            "/management/v1/state", headers={"Authorization": "Bearer main-key"}
        ).status_code
        == 503
    )


def test_legacy_load_accepts_subkey_but_unload_requires_main(client, monkeypatch):
    entry = SimpleNamespace(loaded=True, is_loading=False)
    pool = SimpleNamespace(
        get_model_view=lambda model_id: entry if model_id == "sample" else None,
        unload_engine=AsyncMock(),
    )
    monkeypatch.setattr(client.app.state.server_state, "engine_pool", pool)
    header = {"Authorization": "Bearer sub-key"}

    load = client.post("/v1/models/sample/load", headers=header)
    assert load.status_code == 200
    assert load.json()["message"].startswith("Already loaded")
    assert client.post("/v1/models/sample/unload", headers=header).status_code == 401
    pool.unload_engine.assert_not_awaited()

    main = client.post(
        "/v1/models/sample/unload",
        headers={"Authorization": "Bearer main-key"},
    )
    assert main.status_code == 200
    pool.unload_engine.assert_awaited_once_with("sample")
