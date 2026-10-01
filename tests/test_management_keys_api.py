"""Credential durability and live authorization without real user settings."""

import json
from unittest.mock import Mock

import pytest

from omlx.services.management_server import ServerManagementService
from omlx.settings import GlobalSettings

pytest_plugins = ("test_management_server_api",)


def test_create_edit_reveal_revoke_stable_opaque_id(setup):
    created = setup.client.post("/management/v1/auth/subkeys", json={"name": "CLI"})
    assert created.status_code == 201
    entry = created.json()["sub_key"]
    assert entry["id"] != entry["key"] and len(entry["key"]) >= 32
    edited = setup.client.patch(
        f"/management/v1/auth/subkeys/{entry['id']}",
        json={"name": "Changed", "key": "changed-key"},
    )
    assert edited.status_code == 200 and edited.json()["sub_key"]["id"] == entry["id"]
    assert (
        ServerManagementService(setup.context).keys()["sub_keys"][0]["id"]
        == entry["id"]
    )
    reloaded = GlobalSettings.load(base_path=setup.settings.base_path)
    from dataclasses import replace

    assert (
        ServerManagementService(
            replace(setup.context, global_settings=reloaded)
        ).keys()["sub_keys"][0]["id"]
        == entry["id"]
    )
    assert (
        setup.client.delete(f"/management/v1/auth/subkeys/{entry['id']}").status_code
        == 200
    )
    assert setup.settings.auth.sub_keys == []
    assert (
        setup.client.delete(f"/management/v1/auth/subkeys/{entry['id']}").status_code
        == 404
    )


def test_duplicates_and_invalid_keys(setup):
    assert (
        setup.client.post(
            "/management/v1/auth/subkeys", json={"key": "main-original"}
        ).status_code
        == 409
    )
    assert (
        setup.client.post(
            "/management/v1/auth/subkeys", json={"key": "duplicate"}
        ).status_code
        == 201
    )
    assert (
        setup.client.post(
            "/management/v1/auth/subkeys", json={"key": "duplicate"}
        ).status_code
        == 409
    )
    assert (
        setup.client.patch(
            "/management/v1/auth/main-key", json={"key": "duplicate"}
        ).status_code
        == 409
    )
    for key in ["abc", "space key", "äbcde", "line\nbreak"]:
        assert (
            setup.client.patch(
                "/management/v1/auth/main-key", json={"key": key}
            ).status_code
            == 422
        )


def test_main_rotation_applies_to_next_request(setup):
    assert (
        setup.client.patch(
            "/management/v1/auth/main-key", json={"key": "new-main"}
        ).status_code
        == 200
    )
    assert setup.active["key"] == "new-main"
    assert setup.client.get("/management/v1/auth/keys").status_code == 401
    setup.client.headers["Authorization"] = "Bearer new-main"
    assert setup.client.get("/management/v1/auth/keys").json()["main_key"] == "new-main"
    assert (
        json.loads((setup.settings.base_path / "settings.json").read_text())["auth"][
            "api_key"
        ]
        == "new-main"
    )


def test_subkey_cannot_read_or_mutate_management(setup):
    setup.client.post("/management/v1/auth/subkeys", json={"key": "inference-only"})
    setup.client.headers["Authorization"] = "Bearer inference-only"
    for method, path in [
        ("GET", "auth/keys"),
        ("GET", "server/settings"),
        ("PATCH", "auth/main-key"),
    ]:
        assert (
            setup.client.request(
                method, "/management/v1/" + path, json={"key": "takeover"}
            ).status_code
            == 401
        )


def test_failed_main_key_save_does_not_rotate(setup, monkeypatch):
    monkeypatch.setattr(
        GlobalSettings, "_save_data", Mock(side_effect=OSError("disk full"))
    )
    assert (
        setup.client.patch(
            "/management/v1/auth/main-key", json={"key": "new-main"}
        ).status_code
        == 500
    )
    assert setup.active["key"] == setup.settings.auth.api_key == "main-original"


def test_failed_subkey_save_restores_registry_and_credentials(setup, monkeypatch):
    entry = setup.client.post(
        "/management/v1/auth/subkeys", json={"key": "original-sub"}
    ).json()["sub_key"]
    registry = (setup.settings.base_path / "management-key-ids.json").read_text()
    monkeypatch.setattr(
        GlobalSettings, "_save_data", Mock(side_effect=OSError("disk full"))
    )
    assert (
        setup.client.patch(
            f"/management/v1/auth/subkeys/{entry['id']}", json={"key": "changed-sub"}
        ).status_code
        == 500
    )
    assert setup.settings.auth.sub_keys[0].key == "original-sub"
    assert (
        setup.settings.base_path / "management-key-ids.json"
    ).read_text() == registry


def test_auth_policy_checks_active_bind_despite_pending_loopback(setup):
    setup.active["host"] = "0.0.0.0"
    response = setup.client.patch(
        "/management/v1/auth/policy", json={"skip_api_key_verification": True}
    )
    assert response.status_code == 422 and "Active bind" in response.text
    assert setup.settings.auth.skip_api_key_verification is False
    assert (
        setup.client.patch(
            "/management/v1/auth/policy", json={"allow_unauthenticated_inference": True}
        ).status_code
        == 200
    )


def test_auth_policy_checks_pending_bind_too(setup):
    assert (
        setup.client.patch(
            "/management/v1/auth/policy", json={"skip_api_key_verification": True}
        ).status_code
        == 200
    )
    assert (
        setup.client.patch(
            "/management/v1/server/settings", json={"server": {"host": "0.0.0.0"}}
        ).status_code
        == 422
    )


def test_policy_null_and_unknown_fields_rejected(setup):
    for body in [
        {},
        {"skip_api_key_verification": None},
        {"skip_api_key_verification": "false"},
        {"admin": True},
    ]:
        assert (
            setup.client.patch("/management/v1/auth/policy", json=body).status_code
            == 422
        )


def test_live_rotation_callback_failure_restores_memory_and_disk(setup):
    setup.settings.save()
    from dataclasses import replace

    calls = []

    def rotate(key):
        calls.append(key)
        setup.active["key"] = key
        if key == "failing-main":
            raise RuntimeError("callback failed")

    svc = ServerManagementService(replace(setup.context, set_api_key=rotate))
    from omlx.services.management import ManagementError

    with pytest.raises(ManagementError, match="settings restored"):
        svc.main_key("failing-main")
    assert calls == ["failing-main", "main-original"]
    assert setup.settings.auth.api_key == setup.active["key"] == "main-original"
    assert (
        json.loads((setup.settings.base_path / "settings.json").read_text())["auth"][
            "api_key"
        ]
        == "main-original"
    )


def test_legacy_subkey_id_survives_edit_and_registry_reload(setup):
    from omlx.settings import SubKeyEntry

    setup.settings.auth.sub_keys.append(SubKeyEntry(key="legacy-key"))
    old_id = setup.svc.keys()["sub_keys"][0]["id"]
    edited = setup.svc.subkey({"key": "legacy-new"}, old_id)
    assert edited["sub_key"]["id"] == old_id
    assert ServerManagementService(setup.context).keys()["sub_keys"][0]["id"] == old_id


def test_revoke_removes_inference_authorization(setup):
    from omlx.auth import verify_any_api_key

    entry = setup.svc.subkey({"key": "inference-sub"})["sub_key"]
    assert verify_any_api_key(
        "inference-sub", setup.active["key"], setup.settings.auth.sub_keys
    )
    setup.svc.subkey({}, entry["id"], delete=True)
    assert not verify_any_api_key(
        "inference-sub", setup.active["key"], setup.settings.auth.sub_keys
    )


def test_repeated_rotation_callback_failure_still_restores_disk(setup):
    from dataclasses import replace

    from omlx.services.management import ManagementError

    setup.settings.save()
    path = setup.settings.base_path / "settings.json"
    before = json.loads(path.read_text())
    calls = []

    def failing_rotate(key):
        calls.append(key)
        if key == "unrestored-main":
            setup.active["key"] = key
        raise RuntimeError("callback failed on application and rollback")

    svc = ServerManagementService(replace(setup.context, set_api_key=failing_rotate))
    with pytest.raises(ManagementError) as result:
        svc.main_key("unrestored-main")
    assert result.value.code == "rollback_failed"
    assert "main API key callback" in result.value.detail
    assert "persisted settings restored" in result.value.detail
    assert calls == ["unrestored-main", "main-original"]
    assert setup.settings.auth.api_key == "main-original"
    assert setup.active["key"] == "unrestored-main"
    assert json.loads(path.read_text()) == before
