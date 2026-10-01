# SPDX-License-Identifier: Apache-2.0
"""Stored model profiles through the backend management service."""

import pytest
from molto_config.model_settings import ModelSettings, ModelSettingsManager
from molto_contracts.management import (
    ModelSettingsPatch,
    ProfileUpdate,
    ProfileWrite,
)
from molto_management.management import (
    ManagementContext,
    ManagementError,
    ManagementService,
)
from molto_runtime.engine_pool import EngineEntry, EnginePool
from pydantic import ValidationError


@pytest.fixture
def profiles(tmp_path):
    pool = EnginePool()
    pool._entries["model-a"] = EngineEntry(
        model_id="model-a",
        model_path="/unused",
        model_type="llm",
        engine_type="batched",
        estimated_size=1,
    )
    manager = ModelSettingsManager(tmp_path)
    service = ManagementService(
        ManagementContext(
            engine_pool=pool,
            settings_manager=manager,
            global_settings=None,
            get_default_model=lambda: None,
            set_default_model=lambda _: None,
            apply_sampling=lambda: None,
        )
    )
    return service, pool, manager


def _create(service, name="coding", **settings):
    return service.create_profile(
        "model-a",
        ProfileWrite(
            name=name,
            display_name="Coding",
            settings=ModelSettingsPatch(**settings),
        ),
    )["profile"]


def test_profile_create_list_update_delete_are_durable(profiles):
    service, _, manager = profiles
    assert service.list_profiles("model-a") == {"profiles": []}

    created = _create(service, temperature=0.2, is_pinned=True)
    assert created["name"] == "coding"
    assert created["settings"]["temperature"] == 0.2
    assert "is_pinned" not in created["settings"]

    updated = service.update_profile(
        "model-a",
        "coding",
        ProfileUpdate(
            display_name="Code", settings=ModelSettingsPatch(temperature=0.1)
        ),
    )["profile"]
    assert updated["display_name"] == "Code"
    assert updated["settings"]["temperature"] == 0.1
    assert (
        ModelSettingsManager(manager.base_path).get_profile("model-a", "coding")[
            "settings"
        ]["temperature"]
        == 0.1
    )

    assert service.delete_profile("model-a", "coding") == {
        "deleted": True,
        "name": "coding",
    }
    assert ModelSettingsManager(manager.base_path).list_profiles("model-a") == []


def test_duplicate_profile_conflicts_without_mutating_storage(profiles):
    service, _, manager = profiles
    _create(service)
    before = manager.profiles_file.read_bytes()

    with pytest.raises(ManagementError) as exc_info:
        _create(service)

    assert exc_info.value.code == "conflict"
    assert manager.profiles_file.read_bytes() == before


@pytest.mark.asyncio
async def test_profile_apply_resets_universal_fields_and_preserves_engine_fields(
    profiles,
):
    service, _, manager = profiles
    manager.set_settings(
        "model-a",
        ModelSettings(
            temperature=0.9,
            guided_grammar_enabled=True,
            index_cache_freq=4,
        ),
    )
    _create(service, temperature=0.2)

    result = await service.apply_profile("model-a", "coding")

    assert result["settings"]["temperature"] == 0.2
    assert result["settings"]["guided_grammar_enabled"] is False
    assert result["settings"]["index_cache_freq"] == 4
    assert manager.get_settings("model-a").active_profile_name == "coding"
    persisted = ModelSettingsManager(manager.base_path).get_settings("model-a")
    assert persisted.temperature == 0.2
    assert persisted.index_cache_freq == 4


def test_profile_exposure_cannot_collide_with_real_model_id(profiles):
    service, pool, manager = profiles
    pool._entries["model-a:model-b"] = EngineEntry(
        model_id="model-a:model-b",
        model_path="/unused",
        model_type="llm",
        engine_type="batched",
        estimated_size=1,
    )

    with pytest.raises(ManagementError) as exc_info:
        service.create_profile(
            "model-a",
            ProfileWrite(
                name="collision",
                display_name="Collision",
                settings=ModelSettingsPatch(),
                expose_as_model=True,
                api_name="model-b",
            ),
        )

    assert exc_info.value.code == "conflict"
    assert manager.list_profiles("model-a") == []


def test_invalid_profile_settings_do_not_change_existing_profile(profiles):
    service, _, manager = profiles
    _create(service)
    before = manager.profiles_file.read_bytes()

    with pytest.raises(ValidationError):
        ProfileUpdate(settings=ModelSettingsPatch(turboquant_kv_bits=5, typo=True))

    assert manager.profiles_file.read_bytes() == before


def test_incompatible_profile_settings_do_not_change_storage(profiles):
    service, _, manager = profiles

    with pytest.raises(ManagementError) as exc_info:
        service.create_profile(
            "model-a",
            ProfileWrite(
                name="invalid",
                settings=ModelSettingsPatch(
                    moe_expert_offload_enabled=True,
                    dflash_enabled=True,
                ),
            ),
        )

    assert exc_info.value.code == "invalid_configuration"
    assert manager.list_profiles("model-a") == []


@pytest.mark.asyncio
async def test_invalid_merged_profile_is_not_partially_applied(profiles):
    service, _, manager = profiles
    manager.set_settings("model-a", ModelSettings(temperature=0.5))
    manager.save_profile(
        "model-a",
        "legacy",
        "Legacy",
        None,
        settings={"moe_expert_offload_resident_fraction": 0},
    )
    before = manager.settings_file.read_bytes()

    with pytest.raises(ManagementError) as exc_info:
        await service.apply_profile("model-a", "legacy")

    assert exc_info.value.code == "invalid_configuration"
    assert manager.settings_file.read_bytes() == before
    assert manager.get_settings("model-a").temperature == 0.5
    assert manager.get_settings("model-a").active_profile_name is None


@pytest.mark.asyncio
async def test_missing_profile_and_model_have_semantic_errors(profiles):
    service, _, _ = profiles
    with pytest.raises(ManagementError) as exc_info:
        await service.apply_profile("model-a", "missing")
    assert exc_info.value.code == "not_found"

    with pytest.raises(ManagementError) as exc_info:
        service.list_profiles("missing")
    assert exc_info.value.code == "not_found"


def test_rename_and_exposure_survive_metadata_update(profiles):
    service, _, manager = profiles
    service.create_profile(
        "model-a",
        ProfileWrite(
            name="coding",
            display_name="Coding",
            settings=ModelSettingsPatch(temperature=0.2),
            expose_as_model=True,
            api_name="coding-api",
        ),
    )

    updated = service.update_profile(
        "model-a", "coding", ProfileUpdate(new_name="review", display_name="Review")
    )["profile"]

    assert updated["name"] == "review"
    assert updated["display_name"] == "Review"
    assert updated["expose_as_model"] is True
    assert updated["api_name"] == "coding-api"
    assert manager.get_profile("model-a", "coding") is None
    assert manager.get_profile("model-a", "review") is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "deferred,pinned", [(False, False), (True, False), (False, True)]
)
async def test_profile_apply_uses_settings_reload_lifecycle(
    profiles, monkeypatch, deferred, pinned
):
    service, pool, manager = profiles
    manager.set_settings("model-a", ModelSettings(index_cache_freq=2, is_pinned=pinned))
    entry = pool.get_entry("model-a")
    entry.engine = object()
    entry.is_pinned = pinned
    _create(service, index_cache_freq=5)
    calls = []

    async def unload(model_id, *, reason, abort_active):
        calls.append(("unload", model_id, abort_active))
        return not deferred

    async def load(model_id):
        calls.append(("load", model_id))

    monkeypatch.setattr(pool, "request_unload", unload)
    monkeypatch.setattr(pool, "get_engine", load)
    result = await service.apply_profile("model-a", "coding")
    assert result["requires_reload"] is True
    assert result["reload_deferred"] is deferred
    assert result["auto_unloaded"] is not deferred
    assert result["auto_reloaded"] is (pinned and not deferred)
    assert calls[0] == ("unload", "model-a", False)
    assert (len(calls) == 2) is pinned
    assert manager.get_settings("model-a").active_profile_name == "coding"
