# SPDX-License-Identifier: Apache-2.0
"""Persistence and runtime safety for the engine management settings API."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from omlx.api.management_routes import get_management_service, router
from omlx.auth import require_management_key
from omlx.engine_pool import EngineEntry, EnginePool
from omlx.model_settings import ModelSettings, ModelSettingsManager
from omlx.services.management import (
    ManagementContext,
    ManagementError,
    ManagementService,
)
from omlx.services.management_models import GlobalSettingsPatch, ModelSettingsPatch
from omlx.settings import SamplingSettings, SchedulerSettings


@pytest.fixture
def management(tmp_path):
    pool = EnginePool()
    entry = EngineEntry(
        model_id="model-a",
        model_path=str(tmp_path / "model-a"),
        model_type="llm",
        engine_type="batched",
        estimated_size=1,
        config_model_type="llama",
    )
    pool._entries["model-a"] = entry
    manager = ModelSettingsManager(tmp_path)
    pointer = {"default": None}
    settings = MagicMock()
    settings.sampling = SamplingSettings()
    settings.scheduler = SchedulerSettings()
    service = ManagementService(
        ManagementContext(
            engine_pool=pool,
            settings_manager=manager,
            global_settings=settings,
            get_default_model=lambda: pointer["default"],
            set_default_model=lambda model_id: pointer.__setitem__("default", model_id),
            apply_sampling=lambda: None,
        )
    )
    return service, pool, entry, manager, settings, pointer


@pytest.mark.asyncio
async def test_model_settings_patch_is_durable_and_null_clears_to_default(management):
    service, _, _, manager, _, _ = management
    await service.update_model_settings(
        "model-a", ModelSettingsPatch(temperature=0.25, is_pinned=True)
    )
    assert ModelSettingsManager(manager.base_path).get_settings("model-a").temperature == 0.25

    result = await service.update_model_settings(
        "model-a", ModelSettingsPatch(temperature=None)
    )
    assert result["settings"]["temperature"] is None
    assert result["settings"]["is_pinned"] is True
    persisted = ModelSettingsManager(manager.base_path).get_settings("model-a")
    assert persisted.temperature is None
    assert persisted.is_pinned is True


@pytest.mark.asyncio
async def test_invalid_model_settings_leave_memory_and_file_unchanged(management):
    service, _, _, manager, _, _ = management
    manager.set_settings("model-a", ModelSettings(temperature=0.5))
    before = manager.settings_file.read_bytes()

    with pytest.raises(ValidationError):
        ModelSettingsPatch(temperature=-0.1)
    with pytest.raises(ValidationError):
        ModelSettingsPatch(unrecognized_option=True)

    assert manager.settings_file.read_bytes() == before
    assert manager.get_settings("model-a").temperature == 0.5
    assert service.get_model_settings("model-a")["settings"]["temperature"] == 0.5


@pytest.mark.asyncio
async def test_model_settings_save_failure_rolls_back_memory_and_runtime(management, monkeypatch):
    service, _, entry, manager, _, pointer = management
    manager.set_settings("model-a", ModelSettings(temperature=0.5))
    before = manager.settings_file.read_bytes()

    def fail_save():
        raise OSError("disk full")

    monkeypatch.setattr(manager, "_save", fail_save)
    with pytest.raises(ManagementError) as exc_info:
        await service.update_model_settings(
            "model-a", ModelSettingsPatch(temperature=0.7, is_pinned=True, is_default=True)
        )

    assert exc_info.value.code == "unavailable"
    assert manager.get_settings("model-a").temperature == 0.5
    assert manager.settings_file.read_bytes() == before
    assert entry.is_pinned is False
    assert pointer["default"] is None


def test_settings_manager_save_failure_restores_all_model_records(tmp_path, monkeypatch):
    manager = ModelSettingsManager(tmp_path)
    manager.set_settings("model-a", ModelSettings(temperature=0.5, is_default=True))
    manager.set_settings("model-b", ModelSettings(temperature=0.8))
    before = manager.settings_file.read_bytes()

    def fail_save():
        raise OSError("disk full")

    monkeypatch.setattr(manager, "_save", fail_save)
    with pytest.raises(OSError, match="disk full"):
        manager.set_settings(
            "model-b", ModelSettings(temperature=0.2, is_default=True)
        )

    assert manager.get_settings("model-a").is_default is True
    assert manager.get_settings("model-b").is_default is False
    assert manager.get_settings("model-b").temperature == 0.8
    assert manager.settings_file.read_bytes() == before


@pytest.mark.asyncio
async def test_loaded_model_change_defers_unload_without_aborting_active_work(management):
    service, pool, entry, _, _, _ = management
    entry.engine = MagicMock()
    pool.request_unload = AsyncMock(return_value=False)

    result = await service.update_model_settings(
        "model-a", ModelSettingsPatch(index_cache_freq=2)
    )

    assert result["requires_reload"] is True
    assert result["reload_deferred"] is True
    assert result["auto_unloaded"] is False
    pool.request_unload.assert_awaited_once_with(
        "model-a", reason="settings changed", abort_active=False
    )


@pytest.mark.asyncio
async def test_sampling_change_does_not_unload_loaded_model(management):
    service, pool, entry, _, _, _ = management
    entry.engine = MagicMock()
    pool.request_unload = AsyncMock()

    result = await service.update_model_settings(
        "model-a", ModelSettingsPatch(temperature=0.2)
    )

    assert result["requires_reload"] is False
    pool.request_unload.assert_not_awaited()


@pytest.mark.asyncio
async def test_alias_collision_does_not_write_settings(management):
    service, pool, _, manager, _, _ = management
    pool._entries["model-b"] = EngineEntry(
        model_id="model-b",
        model_path="/unused",
        model_type="llm",
        engine_type="batched",
        estimated_size=1,
    )

    with pytest.raises(ManagementError) as exc_info:
        await service.update_model_settings(
            "model-a", ModelSettingsPatch(model_alias="model-b")
        )

    assert exc_info.value.code == "conflict"
    assert manager.get_settings("model-a").model_alias is None


@pytest.mark.asyncio
async def test_incompatible_engine_settings_fail_before_persistence(management):
    service, _, _, manager, _, _ = management
    manager.set_settings("model-a", ModelSettings(temperature=0.5))
    before = manager.settings_file.read_bytes()

    with pytest.raises(ManagementError) as exc_info:
        await service.update_model_settings(
            "model-a",
            ModelSettingsPatch(
                moe_expert_offload_enabled=True,
                dflash_enabled=True,
            ),
        )

    assert exc_info.value.code == "invalid_configuration"
    assert manager.settings_file.read_bytes() == before
    assert manager.get_settings("model-a").temperature == 0.5


@pytest.mark.asyncio
async def test_global_settings_save_failure_rolls_back_values(management, monkeypatch):
    service, pool, _, _, settings, _ = management
    pool.apply_embedding_batch_size = AsyncMock()
    old_temperature = settings.sampling.temperature
    old_batch_size = settings.scheduler.embedding_batch_size
    settings.save.side_effect = OSError("disk full")

    with pytest.raises(ManagementError) as exc_info:
        await service.update_global_settings(
            GlobalSettingsPatch(temperature=0.2, embedding_batch_size=8)
        )

    assert exc_info.value.code == "unavailable"
    assert settings.sampling.temperature == old_temperature
    assert settings.scheduler.embedding_batch_size == old_batch_size
    pool.apply_embedding_batch_size.assert_not_awaited()


@pytest.mark.asyncio
async def test_global_settings_update_applies_sampling_and_batch_size(management):
    service, pool, _, _, settings, _ = management
    applied = MagicMock()
    service.context = ManagementContext(
        engine_pool=service.pool,
        settings_manager=service.manager,
        global_settings=settings,
        get_default_model=service.context.get_default_model,
        set_default_model=service.context.set_default_model,
        apply_sampling=applied,
    )
    pool.apply_embedding_batch_size = AsyncMock()

    result = await service.update_global_settings(
        GlobalSettingsPatch(temperature=0.3, embedding_batch_size=12)
    )

    assert result["sampling"]["temperature"] == 0.3
    assert result["scheduler"]["embedding_batch_size"] == 12
    assert result["requires_restart"] is False
    settings.save.assert_called_once_with()
    applied.assert_called_once_with()
    pool.apply_embedding_batch_size.assert_awaited_once_with(12)


@pytest.mark.asyncio
async def test_concurrent_disjoint_model_patches_preserve_both_fields(management):
    service, _, _, manager, _, _ = management
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_management_key] = lambda: True
    app.dependency_overrides[get_management_service] = lambda: service

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        first, second = await asyncio.gather(
            client.patch(
                "/management/v1/models/model-a/settings",
                json={"temperature": 0.2},
            ),
            client.patch(
                "/management/v1/models/model-a/settings",
                json={"is_pinned": True},
            ),
        )

    assert first.status_code == second.status_code == 200
    persisted = ModelSettingsManager(manager.base_path).get_settings("model-a")
    assert persisted.temperature == 0.2
    assert persisted.is_pinned is True
