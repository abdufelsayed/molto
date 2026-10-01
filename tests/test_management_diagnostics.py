import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.api.management_diagnostics_routes import router, service
from omlx.auth import AuthContext, require_management_key
from omlx.services.diagnostics.benchmark import BenchmarkRequest, BenchmarkRun
from omlx.services.management import ManagementError
from omlx.services.management_diagnostics import (
    RUNNERS,
    DiagnosticRequest,
    DiagnosticsService,
)


def runtime(tmp_path):
    entry = SimpleNamespace(
        engine_type="batched", config_model_type="qwen3_5", in_use=0
    )
    pool = SimpleNamespace(
        get_entry=lambda model: entry if model == "model" else None,
        get_status=lambda: {"models": [{"id": "model"}]},
        _entries={"model": entry},
    )
    return SimpleNamespace(
        settings=SimpleNamespace(base_path=tmp_path),
        context=SimpleNamespace(engine_pool=pool),
        operation_lock=asyncio.Lock(),
    )


def request(**options):
    return DiagnosticRequest(
        kind="throughput",
        options={"model_id": "model", "prompt_lengths": [1024], **options},
    )


@pytest.mark.asyncio
async def test_success_progress_history_and_recovery(tmp_path):
    rt = runtime(tmp_path)
    from omlx.services.model_control import ModelControl

    rt.control = ModelControl(tmp_path)

    async def runner(run, pool):
        run.events.append({"type": "progress", "current": 1, "total": 1})
        run.results.append({"tps": 123})
        run.status = "completed"

    svc = DiagnosticsService(
        rt, runners={"throughput": (BenchmarkRequest, BenchmarkRun, runner)}
    )
    record = await svc.start(request())
    await svc.tasks[record["id"]]
    assert svc.get(record["id"])["status"] == "completed"
    assert svc.results(record["id"])["results"] == [{"tps": 123}]
    assert not rt.operation_lock.locked()
    restored = DiagnosticsService(rt)
    assert restored.get(record["id"])["progress"]["current"] == 1
    raw = json.loads(svc.path.read_text())
    raw[record["id"]]["status"] = "running"
    svc.path.write_text(json.dumps(raw))
    assert DiagnosticsService(rt).get(record["id"])["status"] == "error"


@pytest.mark.asyncio
async def test_cancel_retains_gate_until_worker_drains(tmp_path):
    rt = runtime(tmp_path)
    started, drain = asyncio.Event(), asyncio.Event()

    async def runner(run, pool):
        started.set()
        await drain.wait()
        run.status = "completed"

    svc = DiagnosticsService(
        rt, runners={"throughput": (BenchmarkRequest, BenchmarkRun, runner)}
    )
    record = await svc.start(request())
    await started.wait()
    assert svc.cancel(record["id"])["status"] == "cancelling"
    assert rt.operation_lock.locked()
    with pytest.raises(ManagementError, match="Another inference"):
        await svc.start(request())
    drain.set()
    await svc.tasks[record["id"]]
    assert svc.get(record["id"])["status"] == "cancelled"
    assert not rt.operation_lock.locked()


@pytest.mark.asyncio
async def test_task_cancel_drains_native_worker(tmp_path):
    rt = runtime(tmp_path)
    started, drain = asyncio.Event(), asyncio.Event()

    async def runner(run, pool):
        started.set()
        await drain.wait()

    svc = DiagnosticsService(
        rt, runners={"throughput": (BenchmarkRequest, BenchmarkRun, runner)}
    )
    record = await svc.start(request())
    await started.wait()
    task = svc.tasks[record["id"]]
    task.cancel()
    await asyncio.sleep(0)
    assert rt.operation_lock.locked()
    assert not task.done()
    drain.set()
    await task
    assert not rt.operation_lock.locked()


@pytest.mark.asyncio
async def test_validation_and_failed_persistence(tmp_path, monkeypatch):
    rt = runtime(tmp_path)
    svc = DiagnosticsService(rt)
    for options in (
        {"model_id": "missing"},
        {"generation_length": -1},
        {"unexpected": True},
    ):
        with pytest.raises(ManagementError):
            await svc.start(request(**options))
    monkeypatch.setattr(
        svc,
        "_save",
        lambda: (_ for _ in ()).throw(ManagementError("persistence", "write failed")),
    )
    with pytest.raises(ManagementError, match="write failed"):
        await svc.start(request())
    assert not rt.operation_lock.locked()
    assert svc.records == {}


@pytest.mark.asyncio
async def test_error_and_secret_redaction(tmp_path):
    rt = runtime(tmp_path)

    async def runner(run, pool):
        raise RuntimeError("sensitive upstream detail")

    svc = DiagnosticsService(
        rt, runners={"throughput": (BenchmarkRequest, BenchmarkRun, runner)}
    )
    record = await svc.start(
        request(
            external={
                "base_url": "https://example.invalid/v1",
                "model": "remote",
                "api_key": "secret-key",
            }
        )
    )
    await svc.tasks[record["id"]]
    assert svc.get(record["id"])["status"] == "error"
    assert "secret-key" not in svc.path.read_text()
    assert "sensitive upstream detail" not in svc.path.read_text()


def test_routes_auth_capabilities_and_missing(tmp_path):
    app = FastAPI()
    app.include_router(router)
    app.state.management_auth_provider = lambda: AuthContext(
        "main-key", [SimpleNamespace(key="sub-key")], "127.0.0.1"
    )
    svc = DiagnosticsService(runtime(tmp_path))
    app.dependency_overrides[service] = lambda: svc
    client = TestClient(app)
    assert client.get("/diagnostics/capabilities").status_code in {401, 403}
    assert (
        client.get(
            "/diagnostics/capabilities", headers={"Authorization": "Bearer sub-key"}
        ).status_code
        == 401
    )
    app.dependency_overrides[require_management_key] = lambda: None
    caps = client.get("/diagnostics/capabilities").json()
    assert len(caps["kinds"]) == 4
    assert caps["publication"] is False
    assert caps["models"][0]["model_id"] == "model"
    assert client.get("/diagnostics/runs/missing").status_code == 404
    assert client.post("/diagnostics/runs", json={"kind": "invalid"}).status_code == 422


def test_real_runners_have_no_admin_or_upload_imports():
    import inspect

    for spec in RUNNERS.values():
        module = inspect.getmodule(spec[2])
        source = inspect.getsource(module)
        assert "from .accuracy_upload" not in source
        assert "from ..server" not in source
        assert "await _upload_to_omlx_ai" not in source


@pytest.mark.asyncio
async def test_shared_operation_history_and_unstarted_cancel(tmp_path):
    from omlx.services.model_control import ModelControl

    rt = runtime(tmp_path)
    rt.control = ModelControl(tmp_path)

    async def runner(run, pool):
        run.status = "completed"

    svc = DiagnosticsService(
        rt, runners={"throughput": (BenchmarkRequest, BenchmarkRun, runner)}
    )
    record = await svc.start(request())
    task = svc.tasks[record["id"]]
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    assert not rt.operation_lock.locked()
    assert svc.get(record["id"])["status"] == "cancelled"


@pytest.mark.asyncio
async def test_external_url_and_local_busy_validation(tmp_path):
    rt = runtime(tmp_path)
    svc = DiagnosticsService(rt)
    with pytest.raises(ManagementError, match="embedded credentials"):
        await svc.start(
            request(
                external={
                    "base_url": "https://user:secret@example.invalid/v1",
                    "model": "remote",
                }
            )
        )
    rt.context.engine_pool._entries["model"].in_use = 1
    with pytest.raises(ManagementError, match="active inference"):
        await svc.start(request())
    assert not rt.operation_lock.locked()


@pytest.mark.asyncio
async def test_pool_admission_spans_runner_and_cleanup(tmp_path):
    from contextlib import asynccontextmanager

    rt = runtime(tmp_path)
    entered, exited = [], []

    @asynccontextmanager
    async def admission(check):
        check()
        entered.append(True)
        try:
            yield
        finally:
            exited.append(True)

    rt.context.engine_pool.exclusive_management = admission

    async def runner(run, pool):
        assert entered and not exited
        run.status = "completed"

    svc = DiagnosticsService(
        rt, runners={"throughput": (BenchmarkRequest, BenchmarkRun, runner)}
    )
    result = await svc.start(request())
    await svc.tasks[result["id"]]
    assert entered == exited == [True]
    assert not rt.operation_lock.locked()


def test_context_default_does_not_apply_settings():
    from omlx.services.diagnostics.context_benchmark import ContextBenchmarkRequest

    assert ContextBenchmarkRequest(model_id="model").apply_result is False


@pytest.mark.asyncio
async def test_failed_cancel_persistence_does_not_cancel_worker(tmp_path, monkeypatch):
    rt = runtime(tmp_path)
    started, finish = asyncio.Event(), asyncio.Event()

    async def runner(run, pool):
        started.set()
        await finish.wait()
        run.status = "completed"

    svc = DiagnosticsService(
        rt, runners={"throughput": (BenchmarkRequest, BenchmarkRun, runner)}
    )
    result = await svc.start(request())
    await started.wait()
    save = svc._save
    monkeypatch.setattr(
        svc,
        "_save",
        lambda: (_ for _ in ()).throw(ManagementError("persistence", "write failed")),
    )
    with pytest.raises(ManagementError):
        svc.cancel(result["id"])
    assert svc.get(result["id"])["status"] == "running"
    assert not getattr(svc.active[result["id"]], "_cancel_requested", False)
    assert rt.operation_lock.locked()
    monkeypatch.setattr(svc, "_save", save)
    finish.set()
    await svc.tasks[result["id"]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "engine_type",
    [
        "audio_stt",
        "audio_tts",
        "audio_sts",
        "embedding",
        "reranker",
        "image_generation",
        "unknown",
    ],
)
async def test_nonlanguage_models_reject_before_engine_calls(tmp_path, engine_type):
    rt = runtime(tmp_path)
    rt.context.engine_pool._entries["model"].engine_type = engine_type
    calls = []

    async def runner(run, pool):
        calls.append("runner")

    svc = DiagnosticsService(
        rt, runners={"throughput": (BenchmarkRequest, BenchmarkRun, runner)}
    )
    assert svc.capabilities()["models"][0]["kinds"] == []
    with pytest.raises(ManagementError, match="language model"):
        await svc.start(request())
    assert calls == []
    assert svc.records == {}
    assert not rt.operation_lock.locked()


def test_nested_unknown_and_coerced_options_return_422(tmp_path):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_management_key] = lambda: None
    app.dependency_overrides[service] = lambda: DiagnosticsService(runtime(tmp_path))
    client = TestClient(app)
    for options in [
        {
            "model_id": "model",
            "prompt_lengths": [1024],
            "external": {
                "base_url": "https://example.invalid/v1",
                "model": "remote",
                "unknown": True,
            },
        },
        {"model_id": "model", "prompt_lengths": ["1024"]},
        {"model_id": "model", "prompt_lengths": [1024], "generation_length": True},
    ]:
        response = client.post(
            "/diagnostics/runs", json={"kind": "throughput", "options": options}
        )
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_real_batch_helper_failure_aborts_and_drains_siblings(monkeypatch):
    from omlx.services.diagnostics import benchmark

    monkeypatch.setattr(benchmark, "HAS_MLX", False)
    started, cleanup, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    aborted = []

    class Core:
        async def add_request(self, *, prompt, **kwargs):
            return str(prompt[0])

        async def stream_outputs(self, request_id):
            if request_id == "1":
                await started.wait()
                raise RuntimeError("stream failed")
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup.set()
                await finish.wait()
            yield None

        async def abort_request(self, request_id):
            aborted.append(request_id)

    task = asyncio.create_task(
        benchmark._run_batch_test(SimpleNamespace(_engine=Core()), [[1], [2]], 1, 1, 2)
    )
    await cleanup.wait()
    assert not task.done()
    task.cancel()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    finish.set()
    with pytest.raises(RuntimeError, match="stream failed"):
        await task
    assert set(aborted) == {"1", "2"}


@pytest.mark.asyncio
async def test_real_external_batch_helper_drains_failed_siblings():
    from omlx.services.diagnostics import benchmark

    started, cleanup, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class Client:
        async def stream_chat_completion(self, *, messages, **kwargs):
            if messages[0]["content"] == "fail":
                await started.wait()
                raise RuntimeError("upstream failed")
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup.set()
                await finish.wait()

    task = asyncio.create_task(
        benchmark._run_external_batch_test(Client(), ["fail", "slow"], 1, 2)
    )
    await cleanup.wait()
    assert not task.done()
    finish.set()
    with pytest.raises(RuntimeError, match="upstream failed"):
        await task


@pytest.mark.parametrize("backend", ["qwen", "k2"])
def test_real_tuning_snapshot_exports_tested_prerequisites(backend):
    from omlx.services.diagnostics import ane_tuning

    run = ane_tuning.ANETuningRun(
        "run", ane_tuning.ANETuningRequest(model_id="model", backend=backend)
    )
    run.recommendation = {"enabled": True, "mlp_fraction": 0.5}
    exported = ane_tuning.run_snapshot(run)["recommendation"]
    assert exported["dflash_enabled"] is False
    assert exported["specprefill_enabled"] is False
    if backend == "k2":
        assert exported["mtp_enabled"] is False
        assert exported["vlm_mtp_enabled"] is False


@pytest.mark.asyncio
async def test_real_native_abort_drain_survives_repeated_cancel():
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from omlx.services.diagnostics import benchmark

    native_started, native_finish = threading.Event(), threading.Event()
    scheduler = SimpleNamespace(
        _pending_abort_ids=set(), requests={"request": object()}
    )
    with ThreadPoolExecutor(max_workers=1) as executor:

        def deferred_abort():
            native_started.set()
            native_finish.wait()
            scheduler._pending_abort_ids.remove("request")
            scheduler.requests.pop("request")

        class Core:
            async def abort_request(self, request_id):
                scheduler._pending_abort_ids.add(request_id)
                asyncio.get_running_loop().run_in_executor(executor, deferred_abort)

        core = Core()
        core.scheduler = scheduler
        core._mlx_executor = executor
        task = asyncio.create_task(
            benchmark._await_drain(benchmark._abort_batch_request(core, "request"))
        )
        await asyncio.to_thread(native_started.wait)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert scheduler.requests
        native_finish.set()
        await task
        assert not scheduler.requests
        assert not scheduler._pending_abort_ids


@pytest.mark.asyncio
async def test_actual_idle_scheduler_removed_request_abort_never_orphans_pending():
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    from omlx.scheduler import Scheduler
    from omlx.services.diagnostics import benchmark

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.requests = {}
    scheduler.waiting = deque()
    scheduler.prefilling = deque()
    scheduler.running = {}
    scheduler._pending_async_removes = []
    scheduler._deferred_clear_at = None
    scheduler._pending_reclaim_request = False
    scheduler._pending_pressure_clear = False
    scheduler._pending_abort_ids = set()
    assert not scheduler.has_requests()
    with ThreadPoolExecutor(max_workers=1) as executor:

        class Core:
            async def abort_request(self, request_id):
                scheduler.abort_request(request_id)

        core = Core()
        core.scheduler = scheduler
        core._mlx_executor = executor
        await asyncio.wait_for(
            benchmark._abort_batch_request(core, "already_removed"), 1
        )
        assert not scheduler._pending_abort_ids
        # Reproduce removal racing with the initial presence check. Actual
        # abort_request enqueues pending work while has_requests stays false.
        scheduler.requests["racing"] = object()

        async def racing_abort(request_id):
            scheduler.requests.pop(request_id)
            scheduler.abort_request(request_id)
            assert not scheduler.has_requests()

        core.abort_request = racing_abort
        await asyncio.wait_for(benchmark._abort_batch_request(core, "racing"), 1)
        assert not scheduler._pending_abort_ids
        assert not scheduler.has_requests()
