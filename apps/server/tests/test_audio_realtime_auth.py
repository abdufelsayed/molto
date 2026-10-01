# SPDX-License-Identifier: Apache-2.0
"""Start-frame auth follows the logical public listener, without audio work."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from omlx_config.settings import GlobalSettings, SubKeyEntry
from omlx_server.api import audio_routes
from starlette.websockets import WebSocketDisconnect


@pytest.fixture
def realtime_auth(tmp_path, monkeypatch):
    settings = GlobalSettings(base_path=tmp_path)
    settings.server.host = "0.0.0.0"
    settings.auth.api_key = "public-main"
    settings.auth.sub_keys = [SubKeyEntry(key="public-sub")]
    active = {"host": "0.0.0.0", "key": "public-main"}
    context = SimpleNamespace(
        global_settings=settings,
        get_api_key=lambda: active["key"],
        get_bind_host=lambda: active["host"],
        runtime_state=SimpleNamespace(uds=str(tmp_path / "backend.sock")),
    )
    app = FastAPI()
    app.state.management_context_provider = lambda: context
    app.include_router(audio_routes.realtime_router)
    engine_pool = Mock(
        side_effect=AssertionError("Auth test must never load an audio model")
    )
    monkeypatch.setattr(audio_routes, "_get_engine_pool", engine_pool)
    return SimpleNamespace(
        settings=settings,
        active=active,
        context=context,
        client=TestClient(app),
        engine_pool=engine_pool,
    )


def start_result(state, key=None):
    with state.client.websocket_connect("/v1/audio/transcriptions/realtime") as socket:
        frame = {"type": "start"}
        if key is not None:
            frame["api_key"] = key
        socket.send_json(frame)
        result = socket.receive_json()
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_json()
        assert closed.value.code == 1008
    state.engine_pool.assert_not_called()
    return result["detail"]


@pytest.mark.parametrize(
    "public_host", ["0.0.0.0", "192.168.1.20", "127.0.0.1,192.168.1.20", "::"]
)
def test_private_uds_never_allows_missing_main_key_on_public_bind(
    realtime_auth, public_host
):
    realtime_auth.active.update(host=public_host, key=None)
    assert realtime_auth.context.runtime_state.uds.endswith("backend.sock")
    assert start_result(realtime_auth) == "Invalid API key"


@pytest.mark.parametrize(
    "key", [None, "invalid", 123, ["public-main"], {"key": "public-main"}]
)
def test_public_bind_skip_toggle_cannot_bypass_auth(realtime_auth, key):
    realtime_auth.settings.auth.skip_api_key_verification = True
    assert start_result(realtime_auth, key) == "Invalid API key"


@pytest.mark.parametrize("key", ["public-main", "public-sub"])
def test_public_bind_accepts_valid_main_and_subkey(realtime_auth, key):
    assert start_result(realtime_auth, key) == "Missing 'model' in start message"


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.1,::1"])
def test_loopback_without_main_key_preserves_local_allowance(realtime_auth, host):
    realtime_auth.active.update(host=host, key=None)
    assert start_result(realtime_auth) == "Missing 'model' in start message"


def test_loopback_with_key_requires_it_until_explicit_skip(realtime_auth):
    realtime_auth.active["host"] = "127.0.0.1"
    assert start_result(realtime_auth) == "Invalid API key"
    realtime_auth.settings.auth.skip_api_key_verification = True
    assert start_result(realtime_auth) == "Missing 'model' in start message"


def test_explicit_unauthenticated_inference_opt_in_applies_to_public_bind(
    realtime_auth,
):
    realtime_auth.settings.auth.allow_unauthenticated_inference = True
    assert start_result(realtime_auth) == "Missing 'model' in start message"
    realtime_auth.settings.auth.allow_unauthenticated_inference = False
    assert start_result(realtime_auth) == "Invalid API key"


def test_auth_uses_active_public_bind_despite_pending_loopback_settings(realtime_auth):
    realtime_auth.settings.server.host = "127.0.0.1"
    realtime_auth.settings.auth.skip_api_key_verification = True
    assert start_result(realtime_auth) == "Invalid API key"


def test_auth_falls_back_to_configured_public_bind_if_callback_missing(realtime_auth):
    realtime_auth.context.get_bind_host = None
    realtime_auth.active["key"] = None
    assert start_result(realtime_auth) == "Invalid API key"


def test_live_key_rotation_and_subkey_revocation_take_effect_next_start(realtime_auth):
    realtime_auth.active["key"] = "rotated-main"
    assert start_result(realtime_auth, "public-main") == "Invalid API key"
    assert (
        start_result(realtime_auth, "rotated-main")
        == "Missing 'model' in start message"
    )
    realtime_auth.settings.auth.sub_keys.clear()
    assert start_result(realtime_auth, "public-sub") == "Invalid API key"
