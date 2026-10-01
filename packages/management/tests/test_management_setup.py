# SPDX-License-Identifier: Apache-2.0
"""Disposable initial setup, without real configuration, sockets, or engines."""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from molto_config.settings import GlobalSettings, SubKeyEntry
from molto_management.management import ManagementContext, ManagementError
from molto_management.management_setup import ManagementSetupService
from molto_server.api.management_dependencies import get_runtime
from molto_server.api.management_setup_routes import router

PATH = "/management/v1/setup"


@pytest.fixture
def setup(tmp_path):
    settings = GlobalSettings(base_path=tmp_path)
    active = {"key": None, "host": "127.0.0.1"}
    calls = []

    def activate(key):
        calls.append(key)
        active["key"] = key

    context = ManagementContext(
        engine_pool=Mock(),
        settings_manager=Mock(),
        global_settings=settings,
        get_default_model=lambda: None,
        set_default_model=lambda _: None,
        apply_sampling=Mock(),
        get_api_key=lambda: active["key"],
        set_api_key=activate,
        get_bind_host=lambda: active["host"],
    )
    runtime = SimpleNamespace(mutation_lock=asyncio.Lock())
    app = FastAPI()
    app.state.management_context_provider = lambda: context
    app.dependency_overrides[get_runtime] = lambda: runtime
    app.include_router(router)
    client = TestClient(app, client=("127.0.0.1", 51000))
    return SimpleNamespace(
        settings=settings,
        active=active,
        context=context,
        calls=calls,
        runtime=runtime,
        app=app,
        client=client,
        service=ManagementSetupService(context),
    )


def submit(client, key="initial-main", confirmation=None):
    return client.post(
        PATH,
        json={
            "key": key,
            "confirmation": key if confirmation is None else confirmation,
        },
    )


def test_fresh_loopback_setup_is_discoverable_persistent_and_activated_after_save(
    setup,
):
    assert setup.client.get(PATH).json() == {
        "setup_required": True,
        "allowed": True,
        "reason": None,
    }
    assert not (setup.settings.base_path / "settings.json").exists()
    response = submit(setup.client)
    assert response.status_code == 201 and response.json() == {"configured": True}
    assert setup.settings.auth.api_key == setup.active["key"] == "initial-main"
    assert setup.calls == ["initial-main"]
    path = setup.settings.base_path / "settings.json"
    assert json.loads(path.read_text())["auth"]["api_key"] == "initial-main"
    assert path.stat().st_mode & 0o777 == 0o600
    assert setup.client.get(PATH).json() == {
        "setup_required": False,
        "allowed": False,
        "reason": "already_configured",
    }
    assert submit(setup.client, "replacement").status_code == 409
    setup.context.engine_pool.assert_not_called()


@pytest.mark.parametrize(
    "host", ["0.0.0.0", "192.168.1.8", "::", "127.0.0.1,192.168.1.8"]
)
def test_active_public_bind_forbids_setup_even_for_loopback_peer(setup, host):
    setup.active["host"] = host
    assert setup.client.get(PATH).json()["allowed"] is False
    assert submit(setup.client).status_code == 403
    assert not setup.calls


@pytest.mark.parametrize(
    "peer", ["192.168.1.8", "203.0.113.5", "::ffff:192.168.1.8", "testclient"]
)
def test_remote_peer_forbidden_and_forwarded_headers_cannot_grant_loopback(setup, peer):
    client = TestClient(setup.app, client=(peer, 51000))
    client.headers.update(
        {"x-forwarded-for": "127.0.0.1", "x-real-ip": "127.0.0.1", "host": "localhost"}
    )
    assert client.get(PATH).json()["reason"] == "loopback_required"
    assert submit(client).status_code == 403
    assert not setup.calls


@pytest.mark.parametrize("peer", ["127.0.0.1", "::1", "::ffff:127.0.0.1"])
def test_real_loopback_and_mapped_ipv6_peers_can_setup(setup, peer):
    client = TestClient(setup.app, client=(peer, 51000))
    assert submit(client).status_code == 201


def test_missing_actual_peer_is_forbidden_even_for_loopback_bind(setup):
    assert setup.service.status(None)["allowed"] is False
    with pytest.raises(ManagementError) as result:
        asyncio.run(
            setup.service.create("initial-main", "initial-main", None, setup.runtime)
        )
    assert result.value.code == "forbidden"
    assert not setup.calls


def test_configured_lan_without_active_bind_callback_is_forbidden(setup):
    setup.settings.server.host = "192.168.1.8"
    service = ManagementSetupService(replace(setup.context, get_bind_host=None))
    assert service.status("127.0.0.1")["allowed"] is False
    with pytest.raises(ManagementError) as result:
        asyncio.run(
            service.create("initial-main", "initial-main", "127.0.0.1", setup.runtime)
        )
    assert result.value.code == "forbidden"


@pytest.mark.parametrize("source", ["effective", "settings", "persisted"])
def test_any_existing_key_blocks_initial_setup_despite_other_missing_views(
    setup, source
):
    if source == "effective":
        setup.active["key"] = "runtime-override"
    elif source == "settings":
        setup.settings.auth.api_key = "effective-settings"
    else:
        setup.settings._save_data({"auth": {"api_key": "saved-key"}})
    data = setup.client.get(PATH).json()
    assert data == {
        "setup_required": False,
        "allowed": False,
        "reason": "already_configured",
    }
    assert "key" not in data
    assert submit(setup.client).status_code == 409
    assert not setup.calls


@pytest.mark.parametrize(
    "key,confirmation",
    [
        ("key-one", "key-two"),
        ("abc", "abc"),
        ("has space", "has space"),
        ("nonascii-ä", "nonascii-ä"),
        ("control\x00", "control\x00"),
    ],
)
def test_invalid_keys_and_confirmation_never_save_or_activate(setup, key, confirmation):
    assert submit(setup.client, key, confirmation).status_code == 422
    assert not setup.calls
    assert not (setup.settings.base_path / "settings.json").exists()


def test_parallel_first_attempts_have_exactly_one_winner(setup):
    async def requests():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=setup.app, client=("127.0.0.1", 51000)),
            base_url="http://local.test",
        ) as client:
            return await asyncio.gather(
                client.post(
                    PATH, json={"key": "first-main", "confirmation": "first-main"}
                ),
                client.post(
                    PATH, json={"key": "second-main", "confirmation": "second-main"}
                ),
            )

    results = asyncio.run(requests())
    assert sorted(response.status_code for response in results) == [201, 409]
    assert len(setup.calls) == 1
    assert setup.active["key"] == setup.calls[0]
    assert (
        json.loads((setup.settings.base_path / "settings.json").read_text())["auth"][
            "api_key"
        ]
        == setup.calls[0]
    )


def test_failed_save_cannot_activate_or_change_runtime(setup, monkeypatch):
    monkeypatch.setattr(
        GlobalSettings, "_save_data", Mock(side_effect=OSError("disk full"))
    )
    response = submit(setup.client)
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "persistence_failed"
    assert setup.active["key"] is None and setup.settings.auth.api_key is None
    assert not setup.calls


def test_setup_updates_only_main_key_preserving_persisted_override_views(setup):
    original = {
        "server": {"port": 8000},
        "auth": {"api_key": None, "secret_key": "existing-secret"},
        "extra": {"preserve": True},
    }
    setup.settings._save_data(original)
    setup.settings.server.port = 9123
    assert submit(setup.client).status_code == 201
    saved = json.loads((setup.settings.base_path / "settings.json").read_text())
    assert saved["server"]["port"] == 8000
    assert saved["extra"] == original["extra"]
    assert saved["auth"]["secret_key"] == "existing-secret"
    assert setup.settings.server.port == 9123


@pytest.mark.parametrize("persisted", [False, True])
def test_activation_failure_restores_exact_runtime_and_persistence(setup, persisted):
    from molto_server.api.management_setup_routes import service

    path = setup.settings.base_path / "settings.json"
    before = {"server": {"port": 8123}, "auth": {"api_key": None}, "extra": True}
    if persisted:
        setup.settings._save_data(before)

    def activate(key):
        setup.calls.append(key)
        if key is not None:
            # Persistence must already have succeeded before activation.
            assert json.loads(path.read_text())["auth"]["api_key"] == key
        setup.active["key"] = key
        if key is not None:
            raise RuntimeError("activation failed")

    svc = ManagementSetupService(replace(setup.context, set_api_key=activate))
    setup.app.dependency_overrides[service] = lambda: svc
    response = submit(setup.client)
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "runtime_failed"
    assert setup.active["key"] is None and setup.settings.auth.api_key is None
    assert setup.calls == ["initial-main", None]
    if persisted:
        assert json.loads(path.read_text()) == before
    else:
        assert not path.exists()


def test_repeated_activation_failure_still_restores_persistence(setup):
    from molto_server.api.management_setup_routes import service

    setup.settings._save_data({"auth": {"api_key": None}, "extra": True})
    before = json.loads((setup.settings.base_path / "settings.json").read_text())

    def activate(key):
        setup.calls.append(key)
        if key is not None:
            setup.active["key"] = key
        raise RuntimeError("always fails")

    setup.app.dependency_overrides[service] = lambda: ManagementSetupService(
        replace(setup.context, set_api_key=activate)
    )
    response = submit(setup.client)
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "rollback_failed"
    assert setup.settings.auth.api_key is None
    assert (
        json.loads((setup.settings.base_path / "settings.json").read_text()) == before
    )


def test_subkey_cannot_be_reused_as_initial_main_key(setup):
    setup.settings.auth.sub_keys = [SubKeyEntry(key="existing-sub")]
    assert submit(setup.client, "existing-sub").status_code == 409
    assert not setup.calls


def test_invalid_persisted_settings_fails_closed(setup):
    (setup.settings.base_path / "settings.json").write_text("not-json")
    assert setup.client.get(PATH).status_code == 503
    assert submit(setup.client).status_code == 503
    assert not setup.calls


def test_parallel_setup_attempts_wait_for_the_shared_mutation_lock(setup):
    async def attempts():
        await setup.runtime.mutation_lock.acquire()
        first = asyncio.create_task(
            setup.service.create("first-main", "first-main", "127.0.0.1", setup.runtime)
        )
        second = asyncio.create_task(
            setup.service.create(
                "second-main", "second-main", "127.0.0.1", setup.runtime
            )
        )
        await asyncio.sleep(0)
        assert not first.done() and not second.done()
        assert not setup.calls
        assert not (setup.settings.base_path / "settings.json").exists()
        setup.runtime.mutation_lock.release()
        return await asyncio.gather(first, second, return_exceptions=True)

    results = asyncio.run(attempts())
    assert results[0] == {"configured": True}
    assert isinstance(results[1], ManagementError)
    assert results[1].code == "conflict"
    assert setup.calls == ["first-main"]
