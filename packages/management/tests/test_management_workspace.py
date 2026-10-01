"""Workspace mutations use disposable models and mock lifecycle engines."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from omlx_config.model_settings import ModelSettings, ModelSettingsManager
from omlx_management.management import ManagementContext, ManagementError
from omlx_management.management_runtime import ManagementRuntime
from omlx_management.management_workspace import WorkspaceService, root_id


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    model = root / "model-a"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (model / "model.safetensors").write_bytes(b"fixture")
    manager = ModelSettingsManager(tmp_path / "state")
    pool = MagicMock()
    pool.preparation_active = False
    pool.preparation_pending = False
    pool.is_model_unloading.return_value = False
    pool.is_model_busy.side_effect = lambda mid: (
        bool(entry.in_use or entry.is_loading or entry.pending_unload_reason)
        if mid == "model-a"
        else False
    )
    pool._unloading_models = set()
    pool.model_count = 1
    entry = SimpleNamespace(
        model_path=str(model),
        engine=None,
        is_loading=False,
        in_use=0,
        pending_unload_reason=None,
        is_pinned=False,
        config_model_type="llama",
        model_type="llm",
        engine_type="batched",
        virtual=False,
    )
    pool.get_entry.side_effect = lambda mid: entry if mid == "model-a" else None

    def model_view(mid):
        current = pool.get_entry(mid)
        return (
            SimpleNamespace(**vars(current), loaded=current.engine is not None)
            if current
            else None
        )

    pool.get_model_view.side_effect = model_view

    async def load_model(mid):
        await pool.get_engine(mid)

    pool.load_model = AsyncMock(side_effect=load_model)

    async def smoke_model(mid, kind):
        from omlx_runtime.model_probe import smoke

        return await smoke(await pool.get_engine(mid), kind)

    pool.smoke_model = AsyncMock(side_effect=smoke_model)

    pool.get_model_ids.return_value = ["model-a"]
    pool.get_status.side_effect = lambda: {
        "models": [
            {
                "id": "model-a",
                "model_path": str(model),
                "loaded": entry.engine is not None,
                "is_loading": entry.is_loading,
                "load_failed": False,
                "model_type": "llm",
                "engine_type": "batched",
                "estimated_size": 10,
                "pinned": entry.is_pinned,
            }
        ],
        "model_count": 1,
        "final_ceiling": 100,
        "current_model_memory": 10 if entry.engine else 0,
    }

    async def unload(*args, **kwargs):
        entry.engine = None
        return True

    pool.request_unload = AsyncMock(side_effect=unload)

    async def load(*args):
        if entry.engine is not None:
            return entry.engine
        entry.engine = SimpleNamespace(
            generate=AsyncMock(return_value=SimpleNamespace(text="OK"))
        )
        return entry.engine

    pool.get_engine = AsyncMock(side_effect=load)
    settings = SimpleNamespace(
        base_path=tmp_path / "state", get_effective_model_dirs=lambda: [root]
    )
    default = {"id": None}
    context = ManagementContext(
        pool,
        manager,
        settings,
        lambda: default["id"],
        lambda value: default.__setitem__("id", value),
        lambda: None,
    )
    runtime = ManagementRuntime(context)
    # No exclusive pool preparation is needed by workspace file mutations.
    service = WorkspaceService(runtime)
    return service, model, entry, default


@pytest.mark.asyncio
async def test_registry_contains_unmanaged_incomplete_and_unique_root_ids(
    workspace, tmp_path
):
    service, model, _, _ = workspace
    second = tmp_path / "second"
    second.mkdir()
    for root in [model.parent, second]:
        incomplete = root / "unfinished"
        incomplete.mkdir()
        (incomplete / "config.json").write_text('{"model_type":"llama"}')
    service.context.global_settings.get_effective_model_dirs = lambda: [
        model.parent,
        second,
    ]
    records = (await service.registry())["models"]
    missing = [r for r in records if r["kind"] == "incomplete"]
    assert len(missing) == 2
    assert len({r["id"] for r in records}) == 3
    assert (await service.storage())["roots"][0]["id"] == root_id(model.parent)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "guard", ["pinned", "default", "busy", "loaded", "dependency", "locked"]
)
async def test_delete_enforces_live_guards(workspace, guard):
    service, model, entry, default = workspace
    plan = await service.delete_plan("model-a")
    if guard == "pinned":
        entry.is_pinned = True
    elif guard == "default":
        default["id"] = "model-a"
    elif guard == "busy":
        entry.in_use = 1
    elif guard == "loaded":
        entry.engine = object()
    elif guard == "dependency":
        service.control.store.put(
            "collections", "saved", {"id": "saved", "model_ids": ["model-a"]}
        )
    elif guard == "locked":
        await service.runtime.operation_lock.acquire()
    try:
        with pytest.raises(ManagementError):
            await service.delete("model-a", plan_token=plan["plan_token"])
        assert model.exists()
    finally:
        if guard == "locked":
            service.runtime.operation_lock.release()


@pytest.mark.asyncio
async def test_failed_unload_cannot_delete(workspace):
    service, model, entry, _ = workspace
    entry.engine = object()
    service.pool.request_unload = AsyncMock(return_value=False)
    plan = await service.delete_plan("model-a")
    with pytest.raises(ManagementError):
        await service.delete("model-a", plan_token=plan["plan_token"], drain=True)
    assert model.exists()
    assert service.pool.request_unload.call_args.kwargs["abort_active"] is False


@pytest.mark.asyncio
async def test_delete_success_and_stale_plan(workspace):
    service, model, _, _ = workspace
    plan = await service.delete_plan("model-a")
    (model / "new-file").write_text("new")
    with pytest.raises(ManagementError):
        await service.delete("model-a", plan_token=plan["plan_token"])
    plan = await service.delete_plan("model-a")
    result = await service.delete("model-a", plan_token=plan["plan_token"])
    assert result["deleted"] and not model.exists()


@pytest.mark.asyncio
async def test_delete_refresh_failure_restores_model(workspace):
    service, model, _, _ = workspace
    plan = await service.delete_plan("model-a")
    service.refresh_files = AsyncMock(
        side_effect=[RuntimeError("discover failure"), {}]
    )
    with pytest.raises(RuntimeError):
        await service.delete("model-a", plan_token=plan["plan_token"])
    assert model.exists()


@pytest.mark.asyncio
async def test_confined_symlink_blocks_delete(workspace, tmp_path):
    service, model, _, _ = workspace
    outside = tmp_path / "user-data"
    outside.write_text("keep")
    (model / "escape").symlink_to(outside)
    with pytest.raises(ManagementError):
        await service.delete_plan("model-a")
    assert outside.read_text() == "keep"


@pytest.mark.asyncio
async def test_collection_crud_validation_and_store_rollback(workspace, monkeypatch):
    service, _, _, _ = workspace
    body = {
        "name": "Pair",
        "model_ids": ["model-a"],
        "description": "",
        "preload": False,
    }
    with pytest.raises(ManagementError):
        await service.save_collection("../escape", body)
    result = await service.save_collection("pair", body)
    assert result["collection"]["model_ids"] == ["model-a"]
    before = service.collections()
    monkeypatch.setattr(
        service.control.store,
        "_save_locked",
        MagicMock(side_effect=OSError("disk full")),
    )
    with pytest.raises(OSError):
        await service.save_collection("pair", {**body, "name": "changed"})
    assert service.collections() == before
    monkeypatch.undo()
    assert (await service.delete_collection("pair"))["success"]


@pytest.mark.asyncio
async def test_collection_load_success_and_rollback(workspace):
    service, _, entry, _ = workspace
    await service.save_collection(
        "pair", {"name": "Pair", "model_ids": ["model-a"], "preload": False}
    )
    result = await service.load_collection("pair")
    assert result["loaded"] == ["model-a"] and entry.engine is not None
    entry.engine = None
    service.control.store.put(
        "collections", "pair", {"id": "pair", "model_ids": ["model-a", "model-b"]}
    )
    other = SimpleNamespace(
        model_path=entry.model_path,
        engine=None,
        is_loading=False,
        in_use=0,
        pending_unload_reason=None,
    )
    service.pool.get_entry.side_effect = lambda mid: (
        entry if mid == "model-a" else other
    )
    service.plan = lambda _: {"fits": True, "evictions": []}

    async def load(mid):
        if mid == "model-b":
            raise RuntimeError("load failed")
        entry.engine = object()

    service.pool.get_engine = AsyncMock(side_effect=load)
    with pytest.raises(RuntimeError):
        await service.load_collection("pair")
    assert entry.engine is None


@pytest.mark.asyncio
async def test_import_preview_apply_and_failed_persistence_rollback(
    workspace, monkeypatch
):
    service, _, _, _ = workspace
    bundle = {
        "schema_version": 1,
        "models": [{"id": "model-a", "settings": {"temperature": 0.3}}],
        "collections": [],
    }
    preview = await service.import_bundle(bundle)
    assert (
        preview["dry_run"]
        and service.manager.get_settings("model-a").temperature is None
    )
    assert not (await service.import_bundle(bundle, dry_run=False))["dry_run"]
    assert service.manager.get_settings("model-a").temperature == 0.3
    save = service.manager._save
    calls = {"count": 0}

    def fail_once():
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("disk full")
        save()

    monkeypatch.setattr(service.manager, "_save", fail_once)
    bundle["models"][0]["settings"]["temperature"] = 0.9
    with pytest.raises(OSError):
        await service.import_bundle(bundle, dry_run=False)
    assert service.manager.get_settings("model-a").temperature == 0.3
    assert (
        ModelSettingsManager(service.manager.base_path)
        .get_settings("model-a")
        .temperature
        == 0.3
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "settings", [{"trust_remote_code": True}, {"temperature": -1}, {"unknown": 1}]
)
async def test_import_rejects_unsafe_and_invalid_settings(workspace, settings):
    service, _, _, _ = workspace
    with pytest.raises(ManagementError):
        await service.import_bundle(
            {"schema_version": 1, "models": [{"id": "model-a", "settings": settings}]}
        )


@pytest.mark.asyncio
async def test_import_rejects_busy_loaded_and_unknown(workspace):
    service, _, entry, _ = workspace
    bundle = {
        "schema_version": 1,
        "models": [{"id": "model-a", "settings": {"temperature": 0.3}}],
    }
    entry.engine = object()
    with pytest.raises(ManagementError):
        await service.import_bundle(bundle, dry_run=False)
    bundle["models"][0]["id"] = "unknown"
    with pytest.raises(ManagementError):
        await service.import_bundle(bundle)


@pytest.mark.asyncio
async def test_move_derived_destination_and_rollback(workspace, tmp_path):
    service, model, _, _ = workspace
    target = tmp_path / "target"
    service.context.global_settings.get_effective_model_dirs = lambda: [
        model.parent,
        target,
    ]
    with pytest.raises(ManagementError):
        await service.move("model-a", destination_root_id="/tmp/arbitrary")
    service.refresh_files = AsyncMock(side_effect=[RuntimeError("refresh"), {}])
    with pytest.raises(RuntimeError):
        await service.move("model-a", destination_root_id=root_id(target))
    assert model.exists() and not (target / model.name).exists()
    service.refresh_files = AsyncMock(return_value={})
    # This case isolates byte rollback; real destination identity is proved
    # separately through EnginePool and ModelDiscovery below.
    service.verify_move_destination = AsyncMock(return_value=["model-a"])
    result = await service.move("model-a", destination_root_id=root_id(target))
    assert result["moved"] and not model.exists() and (target / model.name).exists()


@pytest.mark.asyncio
async def test_verify_structural_honest_operation(workspace):
    service, _, _, _ = workspace
    result = await service.verify("model-a")
    operation = result["operation"]
    assert operation["cancellable"] is False and operation["retryable"] is False
    await service.control._tasks[operation["id"]]
    record = service.control.store.get("operations", operation["id"])
    assert record["status"] == "succeeded"
    assert record["result"]["mode"] == "structural"
    assert record["result"]["smoke_verified"] is False
    assert service.control.store.get("health", "model-a")


@pytest.mark.asyncio
async def test_revision_rejects_traversal_and_noncache(workspace):
    service, _, _, _ = workspace
    for revision in ["../escape", "abc"]:
        with pytest.raises(ManagementError):
            await service.activate_revision("model-a", revision)


@pytest.mark.asyncio
async def test_smoke_loads_probes_and_unloads_with_shared_gate(workspace, monkeypatch):
    service, _, entry, _ = workspace
    monkeypatch.setattr(
        "omlx_management.management_workspace.verify_model_files",
        lambda _: {"status": "ready", "summary": "files", "checks": []},
    )
    result = await service.verify("model-a", mode="smoke")
    await service.control._tasks[result["operation"]["id"]]
    record = service.control.store.get("operations", result["operation"]["id"])
    assert record["status"] == "succeeded" and record["result"]["smoke_verified"]
    assert entry.engine is None
    service.pool.exclusive_management.assert_called_once()
    service.pool.request_unload.assert_awaited_once()


@pytest.mark.asyncio
async def test_smoke_failures_persist_failed_health(workspace, monkeypatch):
    service, _, entry, _ = workspace
    monkeypatch.setattr(
        "omlx_management.management_workspace.verify_model_files",
        lambda _: {"status": "ready", "summary": "files", "checks": []},
    )
    entry.engine = SimpleNamespace(
        generate=AsyncMock(return_value=SimpleNamespace(text=None))
    )
    operation = (await service.verify("model-a", mode="smoke"))["operation"]
    await service.control._tasks[operation["id"]]
    assert (
        service.control.store.get("operations", operation["id"])["status"] == "failed"
    )
    assert service.control.store.get("health", "model-a")["status"] == "smoke_failed"


@pytest.mark.asyncio
async def test_collection_preload_persists_pinning_without_loading(workspace):
    service, _, _, _ = workspace
    await service.save_collection(
        "preload", {"name": "Startup", "model_ids": ["model-a"], "preload": True}
    )
    assert service.manager.get_settings("model-a").is_pinned
    assert (
        ModelSettingsManager(service.manager.base_path)
        .get_settings("model-a")
        .is_pinned
    )
    service.pool.get_engine.assert_not_awaited()


@pytest.mark.asyncio
async def test_import_preview_reports_resets_without_losing_trust(workspace):
    service, _, _, _ = workspace
    service.manager.set_settings(
        "model-a", ModelSettings(temperature=0.9, trust_remote_code=True)
    )
    bundle = {"schema_version": 1, "models": [{"id": "model-a", "settings": {}}]}
    preview = await service.import_bundle(bundle)
    assert {
        "model_id": "model-a",
        "field": "settings.temperature",
        "before": 0.9,
        "after": None,
    } in preview["plan"]["changes"]
    await service.import_bundle(bundle, dry_run=False)
    assert service.manager.get_settings("model-a").trust_remote_code is True


@pytest.mark.asyncio
async def test_revision_activation_and_failed_refresh_rollback(workspace, monkeypatch):
    service, model, _, _ = workspace
    cache = model.parent / "models--org--model"
    snapshots = cache / "snapshots"
    first, second = snapshots / "aaaaaaa", snapshots / "bbbbbbb"
    # Change the synthetic inventory to a snapshot cache without touching weights.
    snapshots.mkdir(parents=True)
    first.mkdir()
    second.mkdir()
    (cache / "refs").mkdir()
    ref = cache / "refs" / "main"
    ref.write_text("aaaaaaa")
    row = {
        "id": "model-a",
        "model_path": str(first),
        "model_type": "llm",
        "loaded": False,
    }
    service.model = AsyncMock(return_value=row)
    service.guard = AsyncMock()
    service.refresh_files = AsyncMock()
    monkeypatch.setattr(
        "omlx_management.management_workspace.verify_model_files",
        lambda _: {"status": "ready"},
    )
    assert (await service.activate_revision("model-a", "bbbbbbb"))["activated"]
    assert ref.read_text() == "bbbbbbb"
    service.refresh_files = AsyncMock(side_effect=[RuntimeError("discover"), None])
    with pytest.raises(RuntimeError):
        await service.activate_revision("model-a", "aaaaaaa")
    assert ref.read_text() == "bbbbbbb"


@pytest.mark.asyncio
async def test_check_and_stage_updates_use_pinned_commit_without_activation(
    workspace, monkeypatch
):
    import huggingface_hub

    service, model, _, _ = workspace
    cache = model.parent / "models--org--model"
    snapshots = cache / "snapshots"
    current, staged = snapshots / ("a" * 40), snapshots / ("b" * 40)
    current.mkdir(parents=True)
    staged.mkdir()
    (cache / "refs").mkdir()
    ref = cache / "refs" / "main"
    ref.write_text("a" * 40)
    row = {
        "id": "model-a",
        "model_path": str(current),
        "source_repo_id": "org/model",
        "model_type": "llm",
    }
    service.model = AsyncMock(return_value=row)
    api = SimpleNamespace(
        model_info=MagicMock(return_value=SimpleNamespace(sha="b" * 40))
    )
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    download = MagicMock(return_value=str(staged))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(
        "omlx_management.management_workspace.verify_model_files",
        lambda _: {"status": "ready"},
    )
    check = (await service.check_update("model-a"))["operation"]
    await service.control._tasks[check["id"]]
    assert (
        service.control.store.get("updates", "model-a")["status"] == "update_available"
    )
    stage = (await service.stage_update("model-a"))["operation"]
    await service.control._tasks[stage["id"]]
    assert service.control.store.get("operations", stage["id"])["status"] == "succeeded"
    assert download.call_args.kwargs["revision"] == "b" * 40
    assert ref.read_text() == "a" * 40


@pytest.mark.asyncio
async def test_cancelled_smoke_drains_worker_before_releasing_gate(
    workspace, monkeypatch
):
    import asyncio

    service, _, entry, _ = workspace
    started, finish = asyncio.Event(), asyncio.Event()

    async def generate(*args, **kwargs):
        started.set()
        await finish.wait()
        return SimpleNamespace(text="OK")

    entry.engine = SimpleNamespace(generate=generate)
    monkeypatch.setattr(
        "omlx_management.management_workspace.verify_model_files",
        lambda _: {"status": "ready", "checks": []},
    )
    operation = (await service.verify("model-a", mode="smoke"))["operation"]
    task = service.control._tasks[operation["id"]]
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert service.runtime.operation_lock.locked()
    assert not task.done()
    with pytest.raises(ManagementError):
        service.runtime.assert_paths_idle([entry.model_path])
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not service.runtime.operation_lock.locked()
    assert (
        service.control.store.get("operations", operation["id"])["status"]
        == "cancelled"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["delete", "move", "revision"])
async def test_acquisition_path_reservations_block_file_mutations(workspace, operation):
    service, model, _, _ = workspace
    plan = await service.delete_plan("model-a")
    service.runtime.reserve_paths([model / "model.safetensors"], "download-worker")
    try:
        with pytest.raises(ManagementError) as error:
            if operation == "delete":
                await service.delete("model-a", plan_token=plan["plan_token"])
            elif operation == "move":
                await service.move("model-a", destination_root_id=root_id(model.parent))
            else:
                await service.guard(await service.model("model-a"), drain=True)
        assert error.value.code == "busy"
        assert model.exists()
    finally:
        service.runtime.release_paths("download-worker")


@pytest.mark.asyncio
async def test_registry_exposed_profile_ids_use_physical_canonical_base(workspace):
    service, _, _, _ = workspace
    service.manager.set_settings("model-a", ModelSettings(model_alias="pretty-alias"))
    service.manager.save_profile(
        "model-a",
        "draft",
        "Draft",
        None,
        {},
        expose_as_model=True,
        api_name="draft-api",
    )
    records = (await service.registry())["models"]
    virtual = next(r for r in records if r["kind"] == "virtual")
    assert virtual["id"] == "model-a:draft-api"
    assert virtual["lineage"]["parents"] == [{"id": "model-a", "relation": "profile"}]
    plan = await service.delete_plan("model-a")
    assert not plan["safe"] and any(
        d["relation"] == "profile" for d in plan["dependents"]
    )


@pytest.mark.asyncio
async def test_import_profiles_validates_and_preserves_api_name(workspace):
    service, _, _, _ = workspace
    bundle = {
        "schema_version": 1,
        "models": [
            {
                "id": "model-a",
                "settings": {},
                "profiles": [
                    {
                        "name": "draft",
                        "settings": {},
                        "api_name": "draft-api",
                        "expose_as_model": True,
                    }
                ],
            }
        ],
    }
    await service.import_bundle(bundle, dry_run=False)
    assert service.manager.get_profile("model-a", "draft")["api_name"] == "draft-api"
    bundle["models"][0]["profiles"][0]["settings"]["trust_remote_code"] = True
    with pytest.raises(ManagementError):
        await service.import_bundle(bundle)


@pytest.mark.asyncio
async def test_nested_directory_link_escape_rejects_workspace_reads_and_moves(
    workspace, tmp_path
):
    service, model, _, _ = workspace
    linked = model.parent / "linked"
    linked.mkdir()
    private = tmp_path / "private"
    private.mkdir()
    secret = private / "private.txt"
    secret.write_text("do not copy")
    (model / "nested").symlink_to(linked, target_is_directory=True)
    (linked / "external").symlink_to(private, target_is_directory=True)
    destination = tmp_path / "destination"
    destination.mkdir()
    service.context.global_settings.get_effective_model_dirs = lambda: [
        model.parent,
        destination,
    ]
    with pytest.raises(ManagementError, match="link outside"):
        service.confined(model, inspect_links=True)
    with pytest.raises(ManagementError, match="link outside"):
        await service.move("model-a", destination_root_id=root_id(destination))
    with pytest.raises(ManagementError, match="link outside"):
        await service.delete_plan("model-a")
    with pytest.raises(ManagementError, match="link outside"):
        await service.verify("model-a")
    assert not list(destination.iterdir())
    assert secret.read_text() == "do not copy"
    assert not service.control.operations()


def test_hf_snapshot_links_within_storage_remain_valid(workspace):
    service, model, _, _ = workspace
    cache = model.parent / "models--org--model"
    blob = cache / "blobs" / "weights"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"fixture")
    snapshot = cache / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").symlink_to("../../blobs/weights")
    assert service.confined(snapshot, inspect_links=True) == snapshot
    assert blob.resolve() in service.runtime.normalized_paths([snapshot])


@pytest.mark.asyncio
async def test_linked_weight_read_reservation_blocks_target_model_move(
    workspace, tmp_path
):
    service, model, _, _ = workspace
    reader = model.parent / "reader"
    reader.mkdir()
    (reader / "weights.safetensors").symlink_to(model / "model.safetensors")
    destination = tmp_path / "destination"
    destination.mkdir()
    service.context.global_settings.get_effective_model_dirs = lambda: [
        model.parent,
        destination,
    ]
    service.runtime.reserve_paths([reader], "publish-reader")
    try:
        with pytest.raises(ManagementError, match="files are in use"):
            await service.move("model-a", destination_root_id=root_id(destination))
        assert model.exists()
        assert not list(destination.iterdir())
    finally:
        service.runtime.release_paths("publish-reader")


@pytest.mark.asyncio
@pytest.mark.parametrize("guard", ["loaded", "busy", "files_in_use", "management_busy"])
async def test_import_preview_and_apply_share_admission_blockers(workspace, guard):
    service, model, entry, _ = workspace
    bundle = {
        "schema_version": 1,
        "models": [{"id": "model-a", "settings": {"temperature": 0.3}}],
    }
    if guard == "loaded":
        entry.engine = object()
    elif guard == "busy":
        entry.in_use = 1
    elif guard == "files_in_use":
        service.runtime.reserve_paths([model], "download")
    else:
        await service.runtime.operation_lock.acquire()
    try:
        plan = (await service.import_bundle(bundle))["plan"]
        assert plan["affected_model_ids"] == ["model-a"]
        assert not plan["can_apply"]
        assert any(
            blocker["code"] == ("busy" if guard == "management_busy" else guard)
            for blocker in plan["blockers"]
        )
        with pytest.raises(ManagementError) as error:
            await service.import_bundle(bundle, dry_run=False)
        assert all(
            blocker["reason"] in error.value.detail for blocker in plan["blockers"]
        )
        assert service.manager.get_settings("model-a").temperature is None
        service.pool.request_unload.assert_not_awaited()
    finally:
        service.runtime.release_paths("download")
        if guard == "management_busy":
            service.runtime.operation_lock.release()


@pytest.mark.asyncio
async def test_unchanged_full_export_skips_loaded_model_profiles_and_writes(
    workspace, monkeypatch
):
    service, _, entry, _ = workspace
    service.manager.set_settings(
        "model-a", ModelSettings(temperature=0.4, trust_remote_code=True)
    )
    service.manager.save_profile(
        "model-a",
        "draft",
        "Draft",
        "Description",
        {"temperature": 0.2},
        expose_as_model=True,
        api_name="draft-api",
    )
    entry.engine = object()
    entry.in_use = 1
    bundle = await service.export()
    # A historical export may contain a materialized policy plus new timestamps.
    bundle["models"][0]["policy"] = {
        "mode": "on_demand",
        "ttl_seconds": None,
        "updated_at": "other-export-time",
    }
    bundle["models"][0]["profiles"][0]["updated_at"] = "other-export-time"
    preview = (await service.import_bundle(bundle))["plan"]
    assert preview["affected_model_ids"] == []
    assert (
        preview["changes"] == [] and preview["blockers"] == [] and preview["can_apply"]
    )
    assert preview["settings_updates"] == preview["profile_updates"] == 0
    setter = MagicMock(
        side_effect=AssertionError("unchanged model must not be rewritten")
    )
    monkeypatch.setattr(service.manager, "set_settings", setter)
    result = await service.import_bundle(bundle, dry_run=False)
    assert not result["dry_run"] and not result["plan"]["changes"]
    setter.assert_not_called()
    service.pool.exclusive_management.assert_not_called()
    service.pool.request_unload.assert_not_awaited()


@pytest.mark.asyncio
async def test_import_changes_only_target_while_other_model_remains_loaded_busy(
    workspace, monkeypatch
):
    import copy

    service, model, entry, _ = workspace
    other = copy.copy(entry)
    other.model_path = str(model.parent / "model-b")
    other.engine = object()
    other.in_use = 2
    service.pool.get_entry.side_effect = lambda mid: (
        entry if mid == "model-a" else other if mid == "model-b" else None
    )
    service.pool.get_model_ids.return_value = ["model-a", "model-b"]
    bundle = {
        "schema_version": 1,
        "models": [
            {"id": "model-a", "settings": {"temperature": 0.3}},
            {"id": "model-b", "settings": {}},
        ],
    }
    plan = (await service.import_bundle(bundle))["plan"]
    assert plan["affected_model_ids"] == ["model-a"]
    assert plan["can_apply"] and not plan["blockers"]
    setter = MagicMock(wraps=service.manager.set_settings)
    monkeypatch.setattr(service.manager, "set_settings", setter)
    applied = await service.import_bundle(bundle, dry_run=False)
    assert applied["plan"]["affected_model_ids"] == ["model-a"]
    assert setter.call_count == 1 and setter.call_args.args[0] == "model-a"
    assert other.engine is not None and other.in_use == 2
    service.pool.request_unload.assert_not_awaited()


@pytest.mark.asyncio
async def test_import_rechecks_loaded_state_after_clean_preview(workspace):
    service, _, entry, _ = workspace
    bundle = {
        "schema_version": 1,
        "models": [{"id": "model-a", "settings": {"temperature": 0.3}}],
    }
    assert (await service.import_bundle(bundle))["plan"]["can_apply"]
    entry.engine = object()
    with pytest.raises(ManagementError, match="Unload model model-a"):
        await service.import_bundle(bundle, dry_run=False)
    assert service.manager.get_settings("model-a").temperature is None
    service.pool.request_unload.assert_not_awaited()


@pytest.mark.asyncio
async def test_import_default_change_previews_previous_default_side_effect(workspace):
    import copy

    service, model, entry, _ = workspace
    other = copy.copy(entry)
    other.model_path = str(model.parent / "model-b")
    other.engine = object()
    service.pool.get_entry.side_effect = lambda mid: (
        entry if mid == "model-a" else other if mid == "model-b" else None
    )
    service.pool.get_model_ids.return_value = ["model-a", "model-b"]
    service.manager.set_settings("model-b", ModelSettings(is_default=True))
    bundle = {
        "schema_version": 1,
        "models": [{"id": "model-a", "settings": {"is_default": True}}],
    }
    plan = (await service.import_bundle(bundle))["plan"]
    assert plan["affected_model_ids"] == ["model-a", "model-b"]
    assert {
        "model_id": "model-b",
        "field": "settings.is_default",
        "before": True,
        "after": False,
    } in plan["changes"]
    assert any(
        blocker["model_id"] == "model-b" and blocker["code"] == "loaded"
        for blocker in plan["blockers"]
    )
    with pytest.raises(ManagementError, match="Unload model model-b"):
        await service.import_bundle(bundle, dry_run=False)
    assert service.manager.get_default_model_id() == "model-b"


def discovered_workspace(tmp_path, *, cached=False):
    """Real pool/discovery over synthetic bytes; never load an engine."""
    from omlx_runtime.engine_pool import EnginePool
    from omlx_runtime.model_discovery import discover_models

    roots = [tmp_path / "source", tmp_path / "destination"]
    for root in roots:
        root.mkdir()
    if cached:
        container = roots[0] / "models--mlx-community--fixture"
        blob = container / "blobs" / "weights"
        blob.parent.mkdir(parents=True)
        blob.write_bytes(b"0" * 1000)
        for revision in ["a" * 40, "b" * 40]:
            checkpoint = container / "snapshots" / revision
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text('{"model_type":"llama"}')
            (checkpoint / "model.safetensors").symlink_to("../../blobs/weights")
        (container / "refs").mkdir()
        (container / "refs" / "main").write_text("a" * 40)
        (container / "refs" / "test").write_text("b" * 40)
        checkpoint = container / "snapshots" / ("a" * 40)
    else:
        container = checkpoint = roots[0] / "owner" / "fixture"
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text('{"model_type":"llama"}')
        (checkpoint / "model.safetensors").write_bytes(b"0" * 1000)
    models = discover_models(roots[0])
    assert len(models) == 1
    model_id = next(iter(models))
    manager = ModelSettingsManager(tmp_path / "state")
    manager.set_settings(
        model_id,
        ModelSettings(
            model_alias="public-name", temperature=0.4, is_pinned=True, is_default=True
        ),
    )
    manager.save_profile(
        model_id,
        "draft",
        "Draft",
        None,
        {"temperature": 0.2},
        expose_as_model=True,
        api_name="draft-api",
    )
    pool = EnginePool()
    pool.discover_models([str(root) for root in roots], manager.get_pinned_model_ids())
    pool.apply_settings_overrides(manager)
    settings = SimpleNamespace(
        base_path=tmp_path / "state", get_effective_model_dirs=lambda: roots
    )
    default = {"id": model_id}
    context = ManagementContext(
        pool,
        manager,
        settings,
        lambda: default["id"],
        lambda value: default.__setitem__("id", value),
        lambda: None,
    )
    service = WorkspaceService(ManagementRuntime(context))
    service.control.store.put(
        "policies", model_id, {"mode": "always_resident", "ttl_seconds": None}
    )
    service.control.store.put(
        "collections",
        "saved",
        {"id": "saved", "name": "Saved", "model_ids": [model_id]},
    )
    return service, roots, container, checkpoint, model_id, default


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", [False, True])
async def test_real_discovery_move_preserves_identity_metadata_and_cache_links(
    tmp_path, cached
):
    from omlx_runtime.model_discovery import discover_models, model_display_name

    service, roots, container, checkpoint, model_id, default = discovered_workspace(
        tmp_path, cached=cached
    )
    metadata = service.manager.get_settings(model_id).to_dict()
    profiles = service.manager.list_profiles(model_id)
    policy = service.control.store.get("policies", model_id)
    with pytest.raises(ManagementError, match="explicitly drain"):
        await service.move(model_id, destination_root_id=root_id(roots[1]))
    result = await service.move(
        model_id, destination_root_id=root_id(roots[1]), drain=True
    )
    destination = roots[1] / container.relative_to(roots[0])
    moved_checkpoint = destination / checkpoint.relative_to(container)
    assert result["path"] == str(moved_checkpoint)
    assert result["container_path"] == str(destination)
    assert result["moved_model_ids"] == [model_id]
    assert not container.exists() and destination.is_dir()
    assert service.pool.get_entry(model_id).model_path == str(moved_checkpoint)
    assert set(discover_models(roots[1])) == {model_id}
    assert service.manager.get_settings(model_id).to_dict() == metadata
    assert service.manager.list_profiles(model_id) == profiles
    assert service.control.store.get("policies", model_id) == policy
    assert service.control.store.get("collections", "saved")["model_ids"] == [model_id]
    assert default["id"] == model_id and service.pool.get_entry(model_id).is_pinned
    assert (
        service.manager.get_exposed_profile_source_model_id("public-name:draft-api")
        == model_id
    )
    if cached:
        assert (destination / "refs" / "main").read_text() == "a" * 40
        assert (destination / "refs" / "test").read_text() == "b" * 40
        for revision in ["a" * 40, "b" * 40]:
            link = destination / "snapshots" / revision / "model.safetensors"
            assert link.is_symlink() and str(link.readlink()) == "../../blobs/weights"
            assert link.resolve() == destination / "blobs" / "weights"
            assert link.read_bytes() == b"0" * 1000
    else:
        assert moved_checkpoint.relative_to(roots[1]).as_posix() == "owner/fixture"
        assert model_display_name(model_id, moved_checkpoint, roots) == "owner/fixture"
    assert not service.runtime.path_activities


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["copy", "discovery", "identity"])
async def test_real_move_rolls_back_before_original_deletion(
    tmp_path, monkeypatch, failure
):
    service, roots, container, checkpoint, model_id, default = discovered_workspace(
        tmp_path, cached=True
    )
    original_refresh = service.refresh_files
    if failure == "copy":

        def failing_copy(source, staging, destination):
            staging.mkdir()
            (staging / "partial").write_text("partial")
            raise OSError("copy failure")

        monkeypatch.setattr(
            "omlx_management.management_workspace.copy_model_container", failing_copy
        )
    elif failure == "discovery":
        calls = 0

        async def fail_once():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("discovery failure")
            await original_refresh()

        service.refresh_files = fail_once
    else:

        async def wrong_destination():
            await original_refresh()
            if not container.exists():
                service.pool._entries.pop(model_id, None)

        service.refresh_files = wrong_destination
    with pytest.raises((OSError, RuntimeError, ManagementError)):
        await service.move(model_id, destination_root_id=root_id(roots[1]), drain=True)
    assert container.exists()
    assert (checkpoint / "model.safetensors").read_bytes() == b"0" * 1000
    assert not (roots[1] / container.name).exists()
    assert service.pool.get_entry(model_id).model_path == str(checkpoint)
    assert default["id"] == model_id
    assert service.manager.get_profile(model_id, "draft")
    assert not list(roots[0].glob(".workspace-*"))
    assert not list(roots[1].glob(".workspace-*"))
    assert not service.runtime.path_activities


@pytest.mark.asyncio
@pytest.mark.parametrize("dependency", ["path", "symlink"])
async def test_real_move_rejects_external_path_and_link_dependencies(
    tmp_path, dependency
):
    service, roots, container, checkpoint, model_id, _ = discovered_workspace(tmp_path)
    consumer = roots[0] / "consumer"
    consumer.mkdir()
    (consumer / "config.json").write_text('{"model_type":"llama"}')
    if dependency == "symlink":
        (consumer / "model.safetensors").symlink_to(checkpoint / "model.safetensors")
    else:
        (consumer / "model.safetensors").write_bytes(b"0" * 1000)
        service.manager.set_settings(
            "consumer", ModelSettings(dflash_draft_model=str(checkpoint))
        )
    await service.refresh_files()
    with pytest.raises(
        ManagementError, match="depends on an explicit path|links to files"
    ):
        await service.move(model_id, destination_root_id=root_id(roots[1]), drain=True)
    assert container.exists()
    assert not (roots[1] / container.relative_to(roots[0])).exists()


@pytest.mark.asyncio
async def test_real_move_rejects_active_requests_without_waiting_for_gate(tmp_path):
    service, roots, container, _, model_id, _ = discovered_workspace(
        tmp_path, cached=True
    )
    service.pool.get_entry(model_id).in_use = 1
    with pytest.raises(ManagementError, match="active requests"):
        await service.move(model_id, destination_root_id=root_id(roots[1]), drain=True)
    assert not service.runtime.operation_lock.locked()
    assert not service.pool._preparation_active
    assert container.exists()


@pytest.mark.asyncio
async def test_move_unmanaged_metadata_persistence_failure_rolls_back(
    workspace, tmp_path, monkeypatch
):
    service, model, _, _ = workspace
    artifact = model.parent / "adapter"
    artifact.mkdir()
    (artifact / "adapter_config.json").write_text('{"peft_type":"LORA"}')
    (artifact / "adapter.safetensors").write_bytes(b"fixture")
    destination = tmp_path / "destination"
    destination.mkdir()
    service.context.global_settings.get_effective_model_dirs = lambda: [
        model.parent,
        destination,
    ]
    row = next(row for row in await service.rows() if row["model_type"] == "adapter")
    service.control.store.put("health", row["id"], {"status": "unverified"})
    old_control = service.control.store.export_state()
    old_bytes = service.control.store.path.read_bytes()
    monkeypatch.setattr(
        service.control.store,
        "_save_locked",
        MagicMock(side_effect=OSError("persist failure")),
    )
    with pytest.raises(OSError, match="persist failure"):
        await service.move(row["id"], destination_root_id=root_id(destination))
    assert artifact.exists() and not (destination / "adapter").exists()
    assert service.control.store.export_state() == old_control
    assert service.control.store.path.read_bytes() == old_bytes
    assert not service.runtime.path_activities


@pytest.mark.asyncio
async def test_move_rebases_absolute_internal_links_without_copying_blob_twice(
    tmp_path,
):
    service, roots, container, checkpoint, model_id, _ = discovered_workspace(
        tmp_path, cached=True
    )
    absolute_link = checkpoint / "absolute-weights.safetensors"
    absolute_link.symlink_to(container / "blobs" / "weights")
    result = await service.move(
        model_id, destination_root_id=root_id(roots[1]), drain=True
    )
    destination = roots[1] / container.name
    copied = Path(result["path"]) / absolute_link.name
    assert copied.is_symlink()
    assert not copied.readlink().is_absolute()
    assert copied.resolve() == destination / "blobs" / "weights"
    assert len(list((destination / "blobs").iterdir())) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("unloaded", [False, True])
async def test_real_move_explicit_drain_uses_no_abort_and_requires_completed_unload(
    tmp_path, monkeypatch, unloaded
):
    service, roots, container, checkpoint, model_id, _ = discovered_workspace(tmp_path)
    entry = service.pool.get_entry(model_id)
    entry.engine = SimpleNamespace()

    async def unload(*args, **kwargs):
        # Native readers/writers must be excluded before lifecycle awaits.
        for path in (container, roots[1] / container.relative_to(roots[0])):
            with pytest.raises(ManagementError, match="in use"):
                service.runtime.reserve_paths([path], "competing-native-job")
        if unloaded:
            entry.engine = None
        return unloaded

    worker = AsyncMock(side_effect=unload)
    monkeypatch.setattr(service.pool, "request_unload", worker)
    if unloaded:
        result = await service.move(
            model_id, destination_root_id=root_id(roots[1]), drain=True
        )
        assert result["moved"] and not container.exists()
    else:
        with pytest.raises(ManagementError, match="did not finish unloading"):
            await service.move(
                model_id, destination_root_id=root_id(roots[1]), drain=True
            )
        assert container.exists() and checkpoint.exists()
        assert not (roots[1] / container.relative_to(roots[0])).exists()
    worker.assert_awaited_once_with(
        model_id, reason="workspace storage move", abort_active=False
    )


@pytest.mark.asyncio
async def test_cache_move_preflight_checks_requests_on_every_container_model(tmp_path):
    import copy

    service, roots, container, _, model_id, _ = discovered_workspace(
        tmp_path, cached=True
    )
    extra = copy.copy(service.pool.get_entry(model_id))
    extra.model_id = "other-revision"
    extra.model_path = str(container / "snapshots" / ("b" * 40))
    extra.in_use = 1
    service.pool._entries[extra.model_id] = extra
    with pytest.raises(ManagementError, match="other-revision has active requests"):
        await service.move(model_id, destination_root_id=root_id(roots[1]), drain=True)
    assert container.exists() and not (roots[1] / container.name).exists()
    assert not service.pool._preparation_active


@pytest.mark.asyncio
async def test_move_reserves_staging_and_original_backup_through_native_copy(
    tmp_path, monkeypatch
):
    service, roots, _, _, model_id, _ = discovered_workspace(tmp_path, cached=True)
    from omlx_management.management_workspace import copy_model_container

    observed = []

    def inspect_copy(source, staging, destination):
        reservations = next(
            paths
            for owner, paths in service.runtime.path_activities.items()
            if owner.startswith("workspace:")
        )
        assert (
            staging in reservations
            and source in reservations
            and destination in reservations
        )
        assert any(
            path.name.startswith(".workspace-original-") for path in reservations
        )
        observed.append(True)
        copy_model_container(source, staging, destination)

    monkeypatch.setattr(
        "omlx_management.management_workspace.copy_model_container", inspect_copy
    )
    await service.move(model_id, destination_root_id=root_id(roots[1]), drain=True)
    assert observed == [True] and not service.runtime.path_activities


@pytest.mark.asyncio
async def test_move_copy_failure_preserves_unrelated_new_operation_records(
    tmp_path, monkeypatch
):
    service, roots, container, _, model_id, _ = discovered_workspace(tmp_path)

    def fail_copy(source, staging, destination):
        service.control.store.put(
            "operations",
            "other-operation",
            {"id": "other-operation", "status": "running"},
        )
        raise OSError("copy failed")

    monkeypatch.setattr(
        "omlx_management.management_workspace.copy_model_container", fail_copy
    )
    with pytest.raises(OSError, match="copy failed"):
        await service.move(model_id, destination_root_id=root_id(roots[1]), drain=True)
    assert container.exists()
    assert (
        service.control.store.get("operations", "other-operation")["status"]
        == "running"
    )


@pytest.mark.asyncio
async def test_move_rejects_profile_with_explicit_old_container_path(tmp_path):
    service, roots, container, checkpoint, model_id, _ = discovered_workspace(tmp_path)
    service.manager.update_profile(
        model_id, "draft", settings={"dflash_draft_model": str(checkpoint)}
    )
    with pytest.raises(ManagementError, match="depends on an explicit path"):
        await service.move(model_id, destination_root_id=root_id(roots[1]), drain=True)
    assert container.exists()
