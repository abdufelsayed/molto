"""Synthetic management jobs never download weights or publish models."""

import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter, Depends, FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from omlx.api.management_acquisition_routes import router
from omlx.auth import AuthContext, require_management_key
from omlx.services.management import ManagementError
from omlx.services.management_acquisition import AcquisitionService, convert_local
from omlx.services.management_runtime import ManagementRuntime


@pytest.fixture
def runtime(tmp_path):
    root = tmp_path / "models"
    root.mkdir()

    class Pool:
        _preparation_active = False

        @asynccontextmanager
        async def exclusive_preparation(self, check_cancel):
            assert not self._preparation_active
            self._preparation_active = True
            try:
                yield
            finally:
                self._preparation_active = False

        def get_status(self):
            return {"models": [], "model_count": 0}

    settings = SimpleNamespace(
        base_path=tmp_path, model=SimpleNamespace(get_model_dirs=lambda _: [root])
    )
    context = SimpleNamespace(
        global_settings=settings,
        settings_manager=SimpleNamespace(base_path=tmp_path),
        engine_pool=Pool(),
    )
    runtime = ManagementRuntime(context)
    runtime.refresh = AsyncMock()
    return runtime


class FakeManager:
    def __init__(self):
        self._active_tasks = {}
        self._tasks = {}
        self._cancelled = set()
        self.finish = asyncio.Event()
        self.counter = 0
        self.tokens = []

    def update_model_dir(self, root):
        pass

    def update_model_dirs(self, roots):
        pass

    async def start_download(self, repo, token):
        self.counter += 1
        raw = str(self.counter)
        self.tokens.append(token)
        data = {
            "task_id": raw,
            "repo_id": repo,
            "status": "downloading",
            "progress": 25,
        }
        self._tasks[raw] = data

        async def run():
            await self.finish.wait()
            data["status"] = "cancelled" if raw in self._cancelled else "completed"
            data["progress"] = 100

        self._active_tasks[raw] = asyncio.create_task(run())
        return SimpleNamespace(task_id=raw)

    def get_tasks(self):
        return list(self._tasks.values())

    async def shutdown(self):
        pass


@pytest.mark.asyncio
async def test_download_progress_cancel_drain_history_and_retry(runtime):
    manager = FakeManager()
    runtime._managers["hf"] = manager
    svc = AcquisitionService(runtime)
    operation = await svc.download("hf", "owner/model", "super-secret")
    await asyncio.sleep(0.01)
    assert svc.get(operation["id"])["progress"] == 25
    assert svc.get(operation["id"])["actions"]["cancel"]
    await svc.cancel(operation["id"])
    await asyncio.sleep(0.01)
    assert not runtime.control._tasks[operation["id"]].done()
    assert svc.get(operation["id"])["stage"] == "cancelling"
    manager.finish.set()
    await asyncio.gather(
        runtime.control._tasks[operation["id"]], return_exceptions=True
    )
    assert svc.get(operation["id"])["status"] == "cancelled"
    assert "super-secret" not in runtime.control.store.path.read_text()
    retried = await svc.retry(operation["id"], "new-secret")
    await asyncio.sleep(0.3)
    assert retried["retry_of"] == operation["id"]
    assert manager.tokens == ["super-secret", "new-secret"]
    assert svc.get(retried["id"])["status"] == "succeeded"
    assert svc.delete(operation["id"])["deleted"]
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_exclusive_native_cancel_retains_pool_and_admission(runtime):
    started = threading.Event()
    finish = threading.Event()

    def native():
        started.set()
        finish.wait(3)
        return {"finished": True}

    async def run(op):
        return await runtime.run_native(native)

    operation = await runtime.start_exclusive("test", run)
    while not started.is_set():
        await asyncio.sleep(0.001)
    assert runtime.context.engine_pool._preparation_active
    runtime.control.cancel_operation(operation["id"])
    await asyncio.sleep(0.01)
    assert runtime.operation_lock.locked()
    with pytest.raises(ManagementError, match="preparation") as busy:
        await runtime.start_exclusive("other", run)
    assert busy.value.code == "busy"
    finish.set()
    await asyncio.gather(
        runtime.control._tasks[operation["id"]], return_exceptions=True
    )
    assert not runtime.operation_lock.locked()
    assert not runtime.context.engine_pool._preparation_active
    assert (
        runtime.control.store.get("operations", operation["id"])["status"]
        == "cancelled"
    )


@pytest.mark.asyncio
async def test_restart_interrupted_actions_and_safe_paths(runtime, tmp_path):
    runtime.control.store.put(
        "operations",
        "old",
        {
            "id": "old",
            "kind": "publish",
            "status": "running",
            "payload": {},
            "created_at": None,
        },
    )
    replacement = ManagementRuntime(runtime.context)
    svc = AcquisitionService(replacement)
    assert svc.get("old")["status"] == "interrupted"
    assert svc.get("old")["actions"] == {
        "cancel": False,
        "retry": False,
        "delete": True,
    }
    with pytest.raises(ManagementError):
        svc.local_path(str(tmp_path))
    model = runtime.roots[0] / "model"
    model.mkdir()
    (model / "escape").symlink_to(tmp_path / "outside")
    with pytest.raises(ManagementError, match="link outside"):
        svc.local_path(str(model))


@pytest.mark.asyncio
async def test_persistence_failure_does_not_launch_or_leak_lock(runtime, monkeypatch):
    monkeypatch.setattr(
        runtime.control.store,
        "put",
        lambda *args: (_ for _ in ()).throw(OSError("disk full")),
    )
    ran = AsyncMock()
    with pytest.raises(OSError):
        await runtime.start_exclusive("convert", ran)
    assert not runtime.operation_lock.locked()
    ran.assert_not_called()


@pytest.mark.asyncio
async def test_conversion_preserves_dtype_and_refuses_output_escape(
    runtime, monkeypatch
):
    import sys

    def convert(**options):
        captured.update(options)

    captured = {}
    monkeypatch.setitem(sys.modules, "mlx_lm.convert", SimpleNamespace(convert=convert))
    convert_local("mlx-lm", Path("source"), Path("output"), lambda _: None)
    assert captured == {
        "hf_path": "source",
        "mlx_path": "output",
        "quantize": False,
        "dtype": None,
    }
    svc = AcquisitionService(runtime)
    svc.selected = AsyncMock(
        return_value={
            "id": "model",
            "path": "source",
            "conversion": {"adapter": "mlx-lm"},
        }
    )
    with pytest.raises(ManagementError):
        await svc.convert("source", "../outside")


@pytest.fixture
def client(runtime, monkeypatch):
    app = FastAPI()
    app.state.management_auth_provider = lambda: AuthContext(
        "main-key", [SimpleNamespace(key="sub-key")], "127.0.0.1"
    )
    app.state.management_context_provider = lambda: runtime.context
    app.state.management_runtime = runtime
    parent = APIRouter(
        prefix="/management/v1", dependencies=[Depends(require_management_key)]
    )
    parent.include_router(router)
    app.include_router(parent)

    @app.exception_handler(ManagementError)
    async def error(request, exc):
        return JSONResponse(
            status_code={
                "not_found": 404,
                "busy": 409,
                "conflict": 409,
                "invalid_configuration": 400,
            }.get(exc.code, 503),
            content={"detail": exc.detail},
        )

    return TestClient(app)


def test_routes_main_auth_unknown_fields_missing_history(client):
    endpoint = "/management/v1/operations"
    assert client.get(endpoint).status_code == 401
    assert (
        client.get(endpoint, headers={"Authorization": "Bearer sub-key"}).status_code
        == 401
    )
    headers = {"Authorization": "Bearer main-key"}
    assert client.get(endpoint, headers=headers).json() == {"operations": []}
    assert client.get(endpoint + "/missing", headers=headers).status_code == 404
    assert (
        client.post(
            "/management/v1/acquisition/hf/downloads",
            headers=headers,
            json={"repo_id": "owner/model", "unknown": 1},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/management/v1/acquisition/hf/downloads",
            headers=headers,
            json={"repo_id": "../outside"},
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/management/v1/acquisition/prepare/quantize",
            headers=headers,
            json={"model_path": "/outside", "oq_level": 4, "dtype": "float32"},
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/management/v1/acquisition/prepare/options", headers=headers
        ).status_code
        == 200
    )


@pytest.mark.asyncio
async def test_queued_cancellation_finalizes_and_releases_reservation(runtime):
    runner = AsyncMock()
    operation = await runtime.start_exclusive("queued", runner)
    task = runtime.control._tasks[operation["id"]]
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    runner.assert_not_called()
    assert not runtime.operation_lock.locked()
    assert AcquisitionService(runtime).get(operation["id"])["status"] == "cancelled"


@pytest.mark.asyncio
async def test_shutdown_drains_worker_before_releasing_pool(runtime):
    started = threading.Event()
    finish = threading.Event()

    async def runner(op):
        def native():
            started.set()
            finish.wait(3)
            return {}

        return await runtime.run_native(native)

    await runtime.start_exclusive("shutdown", runner)
    while not started.is_set():
        await asyncio.sleep(0.001)
    shutdown = asyncio.create_task(runtime.shutdown())
    await asyncio.sleep(0.01)
    assert not shutdown.done()
    assert runtime.operation_lock.locked()
    finish.set()
    await shutdown
    assert not runtime.operation_lock.locked()


@pytest.mark.parametrize("action", ["put", "merge", "remove"])
def test_store_rolls_back_failed_save(runtime, monkeypatch, action):
    store = runtime.control.store
    store.put("operations", "before", {"id": "before"})
    original = store.export_state()
    monkeypatch.setattr(
        store, "_save_locked", lambda: (_ for _ in ()).throw(OSError("disk full"))
    )
    with pytest.raises(OSError):
        if action == "put":
            store.put("operations", "before", {"id": "changed"})
        elif action == "merge":
            store.merge("operations", {"after": {"id": "after"}})
        else:
            store.remove("operations", "before")
    assert store.export_state() == original


@pytest.mark.asyncio
async def test_publish_validation_namespaces_and_secret_error(runtime):
    uploader = SimpleNamespace(
        update_model_dirs=lambda _: None,
        validate_token=AsyncMock(
            return_value={
                "username": "owner",
                "orgs": [{"name": "org"}],
                "token": "secret",
            }
        ),
    )
    runtime._managers["uploader"] = uploader
    svc = AcquisitionService(runtime)
    assert await svc.validate_publish("secret") == {
        "valid": True,
        "can_write": True,
        "username": "owner",
        "orgs": [{"name": "org"}],
    }
    uploader.validate_token.side_effect = ValueError("invalid secret")
    with pytest.raises(ManagementError) as error:
        await svc.validate_publish("secret")
    assert "secret" not in error.value.detail


@pytest.mark.asyncio
async def test_publish_private_readme_options_and_durable_no_token(runtime):
    model = runtime.roots[0] / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    manager = FakeManager()
    captured = {}

    async def start_upload(path, repo, token, **options):
        captured.update(path=path, repo=repo, **options)
        return await manager.start_download(repo, token)

    manager.start_upload = start_upload
    runtime._managers["uploader"] = manager
    svc = AcquisitionService(runtime)
    operation = await svc.publish(
        str(model),
        "owner/private",
        "write-secret",
        private=True,
        auto_readme=False,
        readme_source_path=str(model),
        redownload_notice=True,
    )
    await asyncio.sleep(0.01)
    assert not svc.get(operation["id"])["actions"]["cancel"]
    assert captured["private"] is True and captured["auto_readme"] is False
    assert captured["readme_source_path"] == str(model)
    manager.finish.set()
    await asyncio.gather(runtime.control._tasks[operation["id"]])
    assert svc.get(operation["id"])["status"] == "succeeded"
    assert "write-secret" not in runtime.control.store.path.read_text()


def test_routes_busy_state_and_operation_delete(client, runtime):
    headers = {"Authorization": "Bearer main-key"}
    runtime.control.store.put(
        "operations",
        "busy",
        {
            "id": "busy",
            "kind": "test",
            "status": "running",
            "payload": {},
            "created_at": None,
        },
    )
    assert (
        client.delete("/management/v1/operations/busy", headers=headers).status_code
        == 409
    )
    assert (
        client.post(
            "/management/v1/operations/busy/cancel", headers=headers
        ).status_code
        == 409
    )
    assert (
        client.post(
            "/management/v1/operations/busy/retry", headers=headers, json={}
        ).status_code
        == 409
    )
    runtime.control.update_operation("busy", status="failed")
    assert client.delete("/management/v1/operations/busy", headers=headers).json()[
        "deleted"
    ]


@pytest.mark.asyncio
async def test_download_rejects_external_symlink_and_preexisting_ms_model(
    runtime, tmp_path
):
    svc = AcquisitionService(runtime)
    external = tmp_path / "external"
    external.mkdir()
    (runtime.roots[0] / "owner").symlink_to(external, target_is_directory=True)
    with pytest.raises(ManagementError, match="escapes"):
        await svc.download("hf", "owner/model")
    (runtime.roots[0] / "owner").unlink()
    target = runtime.roots[0] / "owner" / "model"
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}")
    with pytest.raises(ManagementError, match="already exists"):
        await svc.download("ms", "owner/model")
    assert (target / "config.json").is_file()
    assert not runtime.path_activities


@pytest.mark.asyncio
async def test_file_reservation_retained_during_cancel_drain(runtime):
    manager = FakeManager()
    runtime._managers["hf"] = manager
    svc = AcquisitionService(runtime)
    operation = await svc.download("hf", "owner/model")
    await asyncio.sleep(0.01)
    with pytest.raises(ManagementError, match="files are in use"):
        runtime.assert_path_available(runtime.roots[0] / "owner")
    await svc.cancel(operation["id"])
    await asyncio.sleep(0.01)
    assert runtime.path_activities
    manager.finish.set()
    await asyncio.gather(
        runtime.control._tasks[operation["id"]], return_exceptions=True
    )
    assert not runtime.path_activities


def test_embedding_conversion_preserves_actual_dtype_and_rejects_mixed(
    runtime, monkeypatch
):
    import sys

    import numpy as np
    from safetensors.numpy import save_file

    source = runtime.roots[0] / "embedding"
    source.mkdir()
    shard = source / "model.safetensors"
    save_file({"weights": np.ones((2, 2), dtype=np.float32)}, str(shard))
    captured = {}

    def convert(**options):
        assert isinstance(options["dtype"], str)
        captured.update(options)

    monkeypatch.setitem(
        sys.modules, "mlx_embeddings.convert", SimpleNamespace(convert=convert)
    )
    convert_local("mlx-embeddings", source, runtime.roots[0] / "output", lambda _: None)
    assert captured["dtype"] == "float32"
    save_file(
        {
            "first": np.ones((2, 2), dtype=np.float32),
            "second": np.ones((2, 2), dtype=np.float16),
        },
        str(shard),
    )
    with pytest.raises(ValueError, match="mixed precision"):
        convert_local(
            "mlx-embeddings", source, runtime.roots[0] / "other", lambda _: None
        )


@pytest.mark.asyncio
async def test_quantization_cancel_drains_retained_worker_and_paths(runtime):
    from omlx.services.oq_manager import QuantStatus

    model = runtime.roots[0] / "source"
    model.mkdir()
    started = threading.Event()
    finish = threading.Event()
    manager = SimpleNamespace(
        _tasks={}, _active_tasks={}, _cancelled=set(), update_model_dirs=lambda _: None
    )

    async def start_quantization(**options):
        task = SimpleNamespace(
            task_id="quant",
            output_path=str(runtime.roots[0] / "source-oQ4"),
            imatrix_cache_path="",
            status=QuantStatus.QUANTIZING,
        )
        manager._tasks["quant"] = task

        def native():
            started.set()
            finish.wait(3)

        async def worker():
            await asyncio.to_thread(native)
            task.status = (
                QuantStatus.CANCELLED
                if "quant" in manager._cancelled
                else QuantStatus.COMPLETED
            )

        manager._active_tasks["quant"] = asyncio.create_task(worker())
        return task

    manager.start_quantization = start_quantization
    manager.get_tasks = lambda: [
        {
            "task_id": "quant",
            "status": manager._tasks["quant"].status.value,
            "progress": 40,
        }
    ]
    runtime._managers["oq"] = manager
    svc = AcquisitionService(runtime)
    svc.selected = AsyncMock(return_value={"id": "source", "path": str(model)})
    operation = await svc.quantize(model_path=str(model), oq_level=4)
    while not started.is_set():
        await asyncio.sleep(0.001)
    await svc.cancel(operation["id"])
    await asyncio.sleep(0.01)
    assert runtime.operation_lock.locked()
    assert runtime.path_activities
    with pytest.raises(ManagementError):
        runtime.assert_path_available(runtime.roots[0] / "source-oQ4")
    finish.set()
    await asyncio.gather(
        runtime.control._tasks[operation["id"]], return_exceptions=True
    )
    assert not runtime.operation_lock.locked()
    assert not runtime.path_activities
    assert svc.get(operation["id"])["status"] == "cancelled"


@pytest.mark.asyncio
async def test_resumed_download_rejects_external_cache_link(runtime, tmp_path):
    destination = runtime.roots[0] / "owner" / "model"
    destination.mkdir(parents=True)
    external = tmp_path / "external-cache"
    external.mkdir()
    (destination / ".cache").symlink_to(external, target_is_directory=True)
    runtime.control.store.put(
        "operations",
        "interrupted",
        {
            "id": "interrupted",
            "kind": "download_hf",
            "status": "interrupted",
            "payload": {"destination": str(destination)},
            "created_at": None,
        },
    )
    with pytest.raises(ManagementError, match="link outside"):
        await AcquisitionService(runtime).download("hf", "owner/model")
    assert not list(external.iterdir())
    assert not runtime.path_activities


def test_reservation_covers_descendant_links_and_preserves_hf_blob_links(runtime):
    root = runtime.roots[0]
    model_a = root / "model-a"
    model_b = root / "model-b"
    model_a.mkdir()
    model_b.mkdir()
    weights = model_b / "weights.safetensors"
    weights.write_bytes(b"synthetic")
    (model_a / "weights.safetensors").symlink_to(weights)
    AcquisitionService(runtime).local_path(str(model_a))
    runtime.reserve_paths([model_a], "reader")
    with pytest.raises(ManagementError, match="files are in use"):
        runtime.assert_path_available(model_b)
    runtime.release_paths("reader")
    cache = root / ".huggingface" / "hub" / "models--owner--model"
    snapshot = cache / "snapshots" / "revision"
    blob = cache / "blobs" / "blob"
    snapshot.mkdir(parents=True)
    blob.parent.mkdir()
    blob.write_bytes(b"synthetic")
    (snapshot / "model.safetensors").symlink_to(blob)
    assert AcquisitionService(runtime).local_path(str(snapshot)) == snapshot
    runtime.reserve_paths([snapshot], "hf-reader")
    with pytest.raises(ManagementError):
        runtime.assert_path_available(cache)
    runtime.release_paths("hf-reader")


def test_nested_directory_link_external_child_rejected_without_external_walk(
    runtime, tmp_path
):
    root = runtime.roots[0]
    source = root / "source-links"
    linked = root / "linked-files"
    source.mkdir()
    linked.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (source / "nested").symlink_to(linked, target_is_directory=True)
    (linked / "external").symlink_to(external, target_is_directory=True)
    with pytest.raises(ManagementError, match="link outside"):
        AcquisitionService(runtime).local_path(str(source))
