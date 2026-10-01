"""Model helper boundaries, using temporary settings and synthetic engines."""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from omlx.engine_pool import EngineEntry, EnginePool
from omlx.model_settings import ModelSettings, ModelSettingsManager
from omlx.services.management import (
    ManagementContext,
    ManagementError,
    ManagementService,
)
from omlx.services.management_model_options import ModelHelpers, options
from omlx.services.management_models import (
    ModelSettingsPatch,
    ProfileWrite,
    TemplateWrite,
)
from omlx.services.recipe import encode_recipe


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "checkpoint"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"model_type": "llama", "max_position_embeddings": 8192})
    )
    pool = EnginePool()
    entry = EngineEntry(
        model_id="org/model",
        model_path=str(path),
        model_type="llm",
        engine_type="batched",
        estimated_size=1,
        config_model_type="llama",
    )
    pool._entries[entry.model_id] = entry
    manager = ModelSettingsManager(tmp_path / "state")
    service = ManagementService(
        ManagementContext(
            pool, manager, None, lambda: None, lambda _: None, lambda: None
        )
    )
    return service, ModelHelpers(service), entry, manager, pool


@pytest.mark.asyncio
async def test_explicit_profile_reset_survives_reload_and_inference(model):
    service, _, _, manager, _ = model
    manager.set_settings("org/model", ModelSettings(mtp_fixed_depth=5, temperature=1))
    service.create_profile(
        "org/model",
        ProfileWrite(
            name="reset",
            expose_as_model=True,
            settings=ModelSettingsPatch(mtp_fixed_depth=None, temperature=None),
        ),
    )
    reloaded = ModelSettingsManager(manager.base_path)
    profile = reloaded.get_profile("org/model", "reset")
    assert profile["settings"] == {"temperature": None, "mtp_fixed_depth": None}
    runtime = manager.get_exposed_profile_runtime_settings_for_request(
        next(iter(manager.get_exposed_profile_model_ids()))
    )
    assert runtime is not None and runtime[1].mtp_fixed_depth is None
    result = await service.apply_profile("org/model", "reset")
    assert result["settings"].get("mtp_fixed_depth") is None
    assert manager.get_settings("org/model").temperature is None


@pytest.mark.asyncio
async def test_trust_explicit_reload_only_uses_mock_engine(model):
    service, _, entry, _, pool = model
    entry.engine = object()
    pool.request_unload = AsyncMock(return_value=True)
    pool.get_engine = AsyncMock()
    result = await service.update_model_settings(
        entry.model_id, ModelSettingsPatch(trust_remote_code=True)
    )
    assert result["settings"]["trust_remote_code"] is True
    assert result["requires_reload"] and result["auto_unloaded"]
    pool.get_engine.assert_not_called()
    assert (
        "execute"
        in next(
            f for f in options(service)["fields"] if f["key"] == "trust_remote_code"
        )["description"]
    )


@pytest.mark.asyncio
async def test_family_validation_and_busy_leave_persistence_unchanged(model):
    service, _, entry, manager, _ = model
    manager.set_settings(entry.model_id, ModelSettings())
    before = manager.settings_file.read_bytes()
    with pytest.raises(ManagementError):
        await service.update_model_settings(
            entry.model_id, ModelSettingsPatch(qwen4_ple_ssd_offload=True)
        )
    entry.is_loading = True
    with pytest.raises(ManagementError) as error:
        await service.update_model_settings(
            entry.model_id, ModelSettingsPatch(is_favorite=True)
        )
    assert error.value.code == "busy"
    assert manager.settings_file.read_bytes() == before


@pytest.mark.asyncio
async def test_template_atomic_failure_and_application(model, monkeypatch):
    service, helper, entry, manager, _ = model
    helper.write_template(
        TemplateWrite(name="fast", settings=ModelSettingsPatch(top_k=20))
    )
    before = manager.templates_file.read_bytes()

    def fail():
        raise OSError("disk full")

    monkeypatch.setattr(manager, "_save_templates", fail)
    with pytest.raises(ManagementError):
        helper.write_template(
            TemplateWrite(name="failed", settings=ModelSettingsPatch(top_k=10))
        )
    assert manager.get_template("failed") is None
    assert manager.templates_file.read_bytes() == before
    result = await helper.apply_template(entry.model_id, "fast")
    assert result["settings"]["top_k"] == 20
    assert manager.get_settings(entry.model_id).active_profile_name == "fast"


@pytest.mark.asyncio
async def test_generation_recipe_and_full_reset(model):
    service, helper, entry, manager, _ = model
    (Path(entry.model_path) / "generation_config.json").write_text(
        json.dumps({"temperature": 0.7, "do_sample": False, "top_p": 0.9})
    )
    assert helper.generation_config(entry.model_id)["settings"] == {
        "temperature": 0.0,
        "top_p": 0.9,
        "max_context_window": 8192,
    }
    await helper.recipe(
        entry.model_id, encode_recipe({"top_k": 12, "trust_remote_code": True})
    )
    assert manager.get_settings(entry.model_id).top_k == 12
    assert not manager.get_settings(entry.model_id).trust_remote_code
    await service.update_model_settings(
        entry.model_id, ModelSettingsPatch(is_favorite=True, display_name="Demo")
    )
    await helper.reset(entry.model_id)
    assert manager.get_settings(entry.model_id).to_dict() == ModelSettings().to_dict()


@pytest.mark.parametrize(
    "values",
    [
        {"temperature": float("nan")},
        {"dflash_draft_quant_weight_bits": 3},
        {"mtp_fixed_depth": 9},
        {"is_favorite": "true"},
    ],
)
def test_strict_wire_validation(values):
    with pytest.raises(ValidationError):
        ModelSettingsPatch(**values)


@pytest.mark.asyncio
async def test_authenticated_router_supports_canonical_slash_ids(model):
    from fastapi import Depends, FastAPI
    from httpx import ASGITransport, AsyncClient

    from omlx.api.management_model_routes import router
    from omlx.auth import AuthContext, require_management_key

    service, _, _, _, _ = model
    app = FastAPI()
    app.state.management_context_provider = lambda: service.context
    app.state.management_auth_provider = lambda: AuthContext(
        "main-key", [], "127.0.0.1"
    )
    app.include_router(
        router, prefix="/management/v1", dependencies=[Depends(require_management_key)]
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get("/management/v1/model-options")).status_code == 401
        headers = {"Authorization": "Bearer main-key"}
        response = await client.get(
            "/management/v1/models/org/model/options", headers=headers
        )
        assert response.status_code == 200
        body = response.json()
        assert body["model_id"] == "org/model"
        assert all("description" in field for field in body["fields"])
        created = await client.post(
            "/management/v1/templates",
            headers=headers,
            json={"name": "quiet", "settings": {"temperature": None}},
        )
        assert created.status_code == 200
        assert created.json()["template"]["settings"] == {"temperature": None}
        applied = await client.post(
            "/management/v1/models/org/model/templates/quiet/apply", headers=headers
        )
        assert applied.status_code == 200
        assert applied.json()["settings"]["active_profile_name"] == "quiet"
        presets = await client.get("/management/v1/presets", headers=headers)
        assert presets.status_code == 200 and isinstance(
            presets.json()["presets"], list
        )


@pytest.mark.asyncio
async def test_profile_save_failure_and_drift(model, monkeypatch):
    service, _, entry, manager, _ = model

    def fail():
        raise OSError("disk full")

    with monkeypatch.context() as patcher:
        patcher.setattr(manager, "_save_profiles", fail)
        with pytest.raises(ManagementError) as error:
            service.create_profile(
                entry.model_id,
                ProfileWrite(
                    name="failed", settings=ModelSettingsPatch(temperature=0.2)
                ),
            )
        assert error.value.code == "unavailable"
        assert manager.list_profiles(entry.model_id) == []
    service.create_profile(
        entry.model_id,
        ProfileWrite(name="stable", settings=ModelSettingsPatch(temperature=0.2)),
    )
    await service.apply_profile(entry.model_id, "stable")
    assert options(service, entry.model_id)["profile_drift"] is False
    await service.update_model_settings(
        entry.model_id, ModelSettingsPatch(temperature=0.4)
    )
    assert options(service, entry.model_id)["profile_drift"] is True


@pytest.mark.asyncio
async def test_optimal_and_refresh_use_explicit_mock_network(model, monkeypatch):
    _, helper, entry, manager, _ = model
    helper.fetch = AsyncMock(return_value={"model_settings": {"top_p": 0.7}})
    result = await helper.optimal(entry.model_id, "abc123")
    assert result["settings"]["top_p"] == 0.7
    helper.fetch.assert_awaited_once_with("https://omlx.ai/api/benchmarks/abc123")
    helper.fetch = AsyncMock(
        return_value={"results": [{"id": "abc123", "pp_tps": 200}]}
    )
    result = await helper.optimal_candidates(entry.model_id)
    assert result["found"] and result["by_pp"][0]["benchmark_id"] == "abc123"
    helper.fetch = AsyncMock(
        return_value={"model_settings": {"trust_remote_code": True}}
    )
    with pytest.raises(ManagementError):
        await helper.optimal(entry.model_id, "abc123")
    assert not manager.get_settings(entry.model_id).trust_remote_code


@pytest.mark.asyncio
async def test_invalid_grammar_and_nested_nonfinite_are_atomic(model):
    service, _, entry, manager, _ = model
    manager.set_settings(entry.model_id, ModelSettings())
    before = manager.settings_file.read_bytes()
    for values in (
        {"guided_grammar_enabled": True, "guided_grammar": "invalid"},
        {"chat_template_kwargs": {"bad": float("nan")}},
    ):
        with pytest.raises(ManagementError) as error:
            await service.update_model_settings(
                entry.model_id, ModelSettingsPatch(**values)
            )
        assert error.value.code == "invalid_configuration"
        assert manager.settings_file.read_bytes() == before


@pytest.mark.asyncio
async def test_recipe_preserves_local_grammar(model):
    _, helper, entry, manager, _ = model
    manager.set_settings(
        entry.model_id,
        ModelSettings(guided_grammar='root ::= "hello"', guided_grammar_enabled=True),
    )
    await helper.recipe(entry.model_id, encode_recipe({"top_k": 10}))
    assert manager.get_settings(entry.model_id).guided_grammar == 'root ::= "hello"'
    assert manager.get_settings(entry.model_id).guided_grammar_enabled is True


@pytest.mark.asyncio
async def test_remote_preset_shape_validation(model):
    _, helper, _, _, _ = model
    helper.fetch = AsyncMock(
        return_value={
            "presets": [{"name": "unsafe", "settings": {"trust_remote_code": True}}]
        }
    )
    with pytest.raises(ManagementError):
        await helper.refresh_presets()
    helper.fetch = AsyncMock(
        return_value={
            "version": 1,
            "presets": [{"name": "safe", "settings": {"top_k": 20}}],
        }
    )
    assert (await helper.refresh_presets())["presets"][0]["name"] == "safe"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "legacy", [{"max_tokens": -1}, {"top_p": 2}, {"dflash_verify_mode": "unknown"}]
)
async def test_unrelated_patch_rejects_invalid_full_candidate_atomically(model, legacy):
    service, _, entry, manager, _ = model
    manager.set_settings(entry.model_id, ModelSettings(**legacy))
    before = manager.settings_file.read_bytes()
    assert service.get_model_settings(entry.model_id)["settings"][
        next(iter(legacy))
    ] == next(iter(legacy.values()))
    with pytest.raises(ManagementError) as error:
        await service.update_model_settings(
            entry.model_id, ModelSettingsPatch(model_alias="safe-alias")
        )
    assert error.value.code == "invalid_configuration"
    assert manager.settings_file.read_bytes() == before
    assert manager.get_settings(entry.model_id).model_alias is None


@pytest.mark.asyncio
async def test_profile_and_template_reject_invalid_inherited_engine_settings(model):
    service, helper, entry, manager, _ = model
    manager.set_settings(entry.model_id, ModelSettings(dflash_verify_mode="unknown"))
    before = manager.settings_file.read_bytes()
    with pytest.raises(ManagementError) as error:
        service.create_profile(
            entry.model_id,
            ProfileWrite(name="bad", settings=ModelSettingsPatch(temperature=0.1)),
        )
    assert error.value.code == "invalid_configuration"
    assert manager.list_profiles(entry.model_id) == []
    manager.save_profile(entry.model_id, "stored", "Stored", None, {"temperature": 0.1})
    profiles_before = manager.profiles_file.read_bytes()
    with pytest.raises(ManagementError):
        await service.apply_profile(entry.model_id, "stored")
    helper.write_template(
        TemplateWrite(name="stored", settings=ModelSettingsPatch(temperature=0.1))
    )
    with pytest.raises(ManagementError):
        await helper.apply_template(entry.model_id, "stored")
    assert manager.settings_file.read_bytes() == before
    assert manager.profiles_file.read_bytes() == profiles_before
    assert manager.get_settings(entry.model_id).active_profile_name is None


def test_metadata_only_profile_and_template_updates_validate_saved_candidate(model):
    from omlx.services.management_models import ProfileUpdate, TemplateUpdate

    service, helper, entry, manager, _ = model
    manager.save_profile(entry.model_id, "bad", "Bad", None, {"top_p": 2})
    manager.save_template("bad", "Bad", None, {"top_p": 2})
    profile_before = manager.profiles_file.read_bytes()
    template_before = manager.templates_file.read_bytes()
    with pytest.raises(ManagementError) as profile_error:
        service.update_profile(
            entry.model_id, "bad", ProfileUpdate(display_name="Changed")
        )
    with pytest.raises(ManagementError) as template_error:
        helper.write_template(TemplateUpdate(display_name="Changed"), "bad")
    assert (
        profile_error.value.code == template_error.value.code == "invalid_configuration"
    )
    assert manager.profiles_file.read_bytes() == profile_before
    assert manager.templates_file.read_bytes() == template_before


@pytest.mark.asyncio
async def test_exclusive_pool_blocks_runtime_mutations_but_keeps_reads_and_metadata(
    model, monkeypatch
):
    from omlx.services.management_models import GlobalSettingsPatch

    service, helper, entry, manager, pool = model
    manager.set_settings(entry.model_id, ModelSettings(temperature=0.2))
    service.create_profile(
        entry.model_id,
        ProfileWrite(name="stored", settings=ModelSettingsPatch(temperature=0.5)),
    )
    helper.write_template(
        TemplateWrite(name="stored", settings=ModelSettingsPatch(temperature=0.5))
    )
    before = manager.settings_file.read_bytes()
    monkeypatch.setattr(pool, "management_operation_allowed", lambda: False)
    calls = [
        service.load(entry.model_id),
        service.unload(entry.model_id),
        service.refresh(),
        service.update_model_settings(
            entry.model_id, ModelSettingsPatch(temperature=0.4)
        ),
        service.apply_profile(entry.model_id, "stored"),
        helper.apply_template(entry.model_id, "stored"),
        service.update_global_settings(GlobalSettingsPatch(temperature=0.4)),
        service.clear_cache("hot"),
    ]
    for call in calls:
        with pytest.raises(ManagementError) as error:
            await call
        assert error.value.code == "busy"
    assert manager.settings_file.read_bytes() == before
    assert service.get_model_settings(entry.model_id)["settings"]["temperature"] == 0.2
    assert service.list_profiles(entry.model_id)["profiles"]
    helper.write_template(
        TemplateWrite(name="metadata", settings=ModelSettingsPatch(top_k=10))
    )
    assert helper.templates()["templates"]


@pytest.mark.asyncio
async def test_exclusive_owner_can_mutate_while_outsider_is_blocked(model):
    import asyncio

    service, _, entry, manager, pool = model
    async with pool.exclusive_management():
        assert pool._preparation_active and pool.management_operation_allowed()
        await service.update_model_settings(
            entry.model_id, ModelSettingsPatch(temperature=0.8)
        )
        # A new empty context represents a separate incoming request.
        import contextvars

        outsider = asyncio.create_task(
            service.update_model_settings(
                entry.model_id, ModelSettingsPatch(temperature=0.1)
            ),
            context=contextvars.Context(),
        )
        with pytest.raises(ManagementError) as error:
            await outsider
        assert error.value.code == "busy"
    assert manager.get_settings(entry.model_id).temperature == 0.8


def test_legacy_pool_preparation_fallback():
    from types import SimpleNamespace

    service = ManagementService(
        SimpleNamespace(engine_pool=SimpleNamespace(_preparation_active=True))
    )
    with pytest.raises(ManagementError) as error:
        service._require_mutation_admission()
    assert error.value.code == "busy"


@pytest.fixture
def mtplx_model(model, monkeypatch):
    from types import SimpleNamespace

    from omlx.services.management_runtime import ManagementRuntime

    service, helper, entry, manager, pool = model
    path = Path(entry.model_path)
    (path / "mtp.safetensors").write_bytes(b"synthetic checkpoint, never opened")
    (path / "mtplx_runtime.json").write_text(json.dumps({"arch_id": "qwen3-next-mtp"}))
    settings = SimpleNamespace(
        base_path=manager.base_path,
        model=SimpleNamespace(get_model_dirs=lambda _: [str(path.parent)]),
    )
    context = ManagementContext(
        pool, manager, settings, lambda: None, lambda _: None, lambda: None
    )
    runtime = ManagementRuntime(context)
    worker = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock(
        return_value={"merge_mode": "rename", "mtp_tensors": 2}
    )
    monkeypatch.setattr("omlx.oq.import_mtplx_sidecar", worker)
    return helper, entry, runtime, worker


@pytest.mark.asyncio
async def test_mtplx_import_reserves_native_gate_and_refreshes_options(mtplx_model):
    helper, entry, runtime, worker = mtplx_model
    result = await helper.import_mtplx(entry.model_id, runtime)
    worker.assert_called_once_with(str(Path(entry.model_path).resolve()))
    assert result["status"] == "ok" and result["mtp_tensors"] == 2
    assert result["options"]["model_id"] == entry.model_id
    assert not runtime.operation_lock.locked() and not runtime.mutation_lock.locked()
    assert runtime.path_activities == {}
    assert not helper.service.pool._preparation_active


@pytest.mark.asyncio
async def test_mtplx_busy_loaded_reserved_and_escaping_paths_refuse_before_worker(
    mtplx_model, tmp_path
):
    helper, entry, runtime, worker = mtplx_model
    entry.engine = object()
    with pytest.raises(ManagementError) as error:
        await helper.import_mtplx(entry.model_id, runtime)
    assert error.value.code == "busy"
    entry.engine = None
    runtime.reserve_paths([entry.model_path], "other")
    with pytest.raises(ManagementError) as error:
        await helper.import_mtplx(entry.model_id, runtime)
    assert error.value.code == "busy"
    runtime.release_paths("other")
    external = tmp_path / "external.safetensors"
    external.write_bytes(b"outside")
    config_path = Path(entry.model_path) / "config.json"
    config_path.write_text(
        json.dumps({"mlx_lm_extra_tensors": {"mtp_file": str(external)}})
    )
    with pytest.raises(ManagementError) as error:
        await helper.import_mtplx(entry.model_id, runtime)
    assert error.value.code == "invalid_configuration"
    worker.assert_not_called()
    assert runtime.path_activities == {}


@pytest.mark.asyncio
async def test_mtplx_worker_failure_releases_admission(mtplx_model):
    helper, entry, runtime, worker = mtplx_model
    worker.side_effect = ValueError("unsupported contract")
    with pytest.raises(ManagementError) as error:
        await helper.import_mtplx(entry.model_id, runtime)
    assert error.value.code == "invalid_configuration"
    assert runtime.path_activities == {}
    assert not runtime.operation_lock.locked()
    assert not helper.service.pool._preparation_active


@pytest.mark.asyncio
async def test_mtplx_cancellation_retains_gate_and_reservation_until_worker_stops(
    mtplx_model,
):
    import asyncio
    import threading

    helper, entry, runtime, worker = mtplx_model
    started, release = threading.Event(), threading.Event()

    def slow_import(_):
        started.set()
        assert release.wait(timeout=5)
        return {"merge_mode": "rename", "mtp_tensors": 2}

    worker.side_effect = slow_import
    task = asyncio.create_task(helper.import_mtplx(entry.model_id, runtime))
    assert await asyncio.to_thread(started.wait, 2)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert runtime.operation_lock.locked() and runtime.mutation_lock.locked()
    assert runtime.path_activities
    assert not helper.service.pool.management_operation_allowed()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not runtime.operation_lock.locked() and not runtime.mutation_lock.locked()
    assert runtime.path_activities == {}
    assert helper.service.pool.management_operation_allowed()


def test_mtplx_detection_hides_ordinary_models_and_preserves_invalid_export_reason(
    model,
):
    service, _, entry, _, _ = model
    ordinary = options(service, entry.model_id)
    assert ordinary["mtplx_sidecar_detected"] is False
    assert ordinary["capabilities"]["mtplx_import"] is False
    path = Path(entry.model_path)
    (path / "mtp.safetensors").write_bytes(b"synthetic; never opened")
    invalid = options(service, entry.model_id)
    assert invalid["mtplx_sidecar_detected"] is True
    assert invalid["capabilities"]["mtplx_import"] is False
    assert invalid["mtplx_import_reason"]
    (path / "mtp.safetensors").unlink()
    (path / "config.json").write_text(
        json.dumps({"mlx_lm_extra_tensors": {"mtp_file": "missing.safetensors"}})
    )
    declared = options(service, entry.model_id)
    assert declared["mtplx_sidecar_detected"] is True
    assert declared["capabilities"]["mtplx_import"] is False
    assert declared["mtplx_import_reason"] == "No MTPLX sidecar found"
