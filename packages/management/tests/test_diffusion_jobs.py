"""Preparation job lifecycle tests use fake workflows, never model weights."""

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from molto_contracts.management import (
    DiffusionCalibrationRequest,
    DiffusionQuantizationRequest,
)
from molto_management import diffusion_jobs as module
from molto_management.diffusion_jobs import DiffusionJobs
from molto_server.api.management_routes import router
from molto_server.auth import AuthContext
from pydantic import ValidationError


class Pool:
    def __init__(self, source):
        self.entry = SimpleNamespace(
            model_path=source, engine_type="image_generation", source_type="local"
        )
        self.active = False
        self.entries = 0

    def get_model_view(self, model_id):
        return self.entry if model_id == "local" else None

    @asynccontextmanager
    async def exclusive_preparation(self, check_cancel):
        check_cancel()
        assert not self.active
        self.active = True
        self.entries += 1
        try:
            yield 123456
        finally:
            self.active = False


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = SimpleNamespace(path=source, base_model="flux2-klein-4b")
    monkeypatch.setattr(module, "_source", lambda path: checkpoint)
    monkeypatch.setattr(module, "read_quantization", lambda path: None)
    pool = Pool(source)
    jobs = DiffusionJobs(pool, tmp_path / "reports", tmp_path / "outputs")
    return jobs, pool


def request(**kwargs):
    return DiffusionCalibrationRequest(
        model_id="local", tasks=[{"prompt": "a teapot", "steps": 4}], **kwargs
    )


async def drain(jobs):
    await asyncio.gather(*list(jobs._workers.values()))


@pytest.mark.asyncio
async def test_workflow_progress_persistence_and_quantization(setup, monkeypatch):
    jobs, pool = setup
    refreshed = []

    async def refresh():
        refreshed.append(True)

    jobs.on_complete = refresh

    def calibrate(source, output, tasks, *, observer, memory_limit_bytes, **kwargs):
        assert pool.active and memory_limit_bytes == 123456
        observer("calibrating", 0.5, "sample")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"base_model": "flux2-klein-4b"}))
        return {"base_model": "flux2-klein-4b", "seconds": 1}

    def quantize(
        source, *, output, calibration, observer, memory_limit_bytes, **kwargs
    ):
        assert calibration.is_file() and pool.active and memory_limit_bytes == 123456
        observer("quantizing", 0.75, "packing")
        output.mkdir(parents=True)
        return {"base_model": "flux2-klein-4b"}

    monkeypatch.setattr(module, "calibrate_diffusion", calibrate)
    monkeypatch.setattr(module, "quantize_diffusion", quantize)
    calibration = jobs.start_calibration(request())
    assert calibration["status"] == "queued"
    await drain(jobs)
    assert jobs.get(calibration["id"])["status"] == "completed"
    quantization = jobs.start_quantization(
        DiffusionQuantizationRequest(
            model_id="local", calibration_job_id=calibration["id"]
        )
    )
    await drain(jobs)
    assert jobs.get(quantization["id"])["status"] == "completed"
    assert refreshed == [True] and pool.entries == 2 and not pool.active
    restored = DiffusionJobs(pool, jobs.artifacts_dir, jobs.output_dir)
    assert len(restored.list()["jobs"]) == 2


@pytest.mark.asyncio
async def test_enqueue_history_failure_does_not_leave_a_ghost_job(setup, monkeypatch):
    jobs, pool = setup

    def fail():
        raise OSError("disk full")

    monkeypatch.setattr(jobs, "_persist", fail)
    with pytest.raises(OSError, match="disk full"):
        jobs.start_calibration(request())
    assert jobs.list() == {"jobs": []}
    assert not jobs._workers and not jobs._cancel and not pool.active


@pytest.mark.asyncio
async def test_cancel_drains_worker_and_serializes_queued_job(setup, monkeypatch):
    jobs, pool = setup
    started, unblock, cleaned = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def calibrate(source, output, tasks, *, observer, **kwargs):
        calls.append(output)
        started.set()
        try:
            assert unblock.wait(5)
            observer("calibrating", 0.5, "checkpoint")
        finally:
            cleaned.set()

    monkeypatch.setattr(module, "calibrate_diffusion", calibrate)
    first = jobs.start_calibration(request())
    second = jobs.start_calibration(request())
    for _ in range(100):
        if started.is_set():
            break
        await asyncio.sleep(0.01)
    assert started.is_set() and pool.active
    jobs.cancel(first["id"])
    jobs.cancel(second["id"])
    assert jobs.get(second["id"])["status"] == "cancelled"
    assert jobs.get(first["id"])["status"] == "cancelling"
    assert pool.active and not cleaned.is_set()
    unblock.set()
    await drain(jobs)
    assert cleaned.is_set() and not pool.active and len(calls) == 1
    assert jobs.get(first["id"])["status"] == "cancelled"
    assert jobs.get(second["id"])["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_coordinator_waits_for_native_worker(setup, monkeypatch):
    jobs, pool = setup
    started, unblock = threading.Event(), threading.Event()

    def calibrate(source, output, tasks, *, observer, **kwargs):
        started.set()
        assert unblock.wait(5)
        observer("loading", 0.2, "loaded")

    monkeypatch.setattr(module, "calibrate_diffusion", calibrate)
    job = jobs.start_calibration(request())
    worker = jobs._workers[job["id"]]
    while not started.is_set():
        await asyncio.sleep(0.01)
    worker.cancel()
    await asyncio.sleep(0.01)
    assert not worker.done() and pool.active
    unblock.set()
    await worker
    assert jobs.get(job["id"])["status"] == "cancelled" and not pool.active


@pytest.mark.asyncio
async def test_publication_wins_late_cancel_and_shutdown_drains(setup, monkeypatch):
    jobs, pool = setup
    started, unblock = threading.Event(), threading.Event()

    def calibrate(source, output, tasks, *, observer, **kwargs):
        observer("publishing", 0.99, "publishing")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}")
        started.set()
        assert unblock.wait(5)
        return {}

    monkeypatch.setattr(module, "calibrate_diffusion", calibrate)
    job = jobs.start_calibration(request())
    while not started.is_set():
        await asyncio.sleep(0.01)
    shutdown = asyncio.create_task(jobs.shutdown())
    await asyncio.sleep(0.01)
    assert pool.active and not shutdown.done()
    unblock.set()
    await shutdown
    assert jobs.get(job["id"])["status"] == "completed"
    assert Path(jobs.get(job["id"])["result"]["output_path"]).exists()
    with pytest.raises(ValueError, match="shutting down"):
        jobs.start_calibration(request())


@pytest.mark.asyncio
async def test_failure_preserves_source_and_releases_gate(setup, monkeypatch):
    jobs, pool = setup
    sentinel = Path(pool.entry.model_path) / "weights"
    sentinel.write_text("original")

    def fail(*args, **kwargs):
        raise RuntimeError("native load failed")

    monkeypatch.setattr(module, "calibrate_diffusion", fail)
    job = jobs.start_calibration(request())
    await drain(jobs)
    result = jobs.get(job["id"])
    assert result["status"] == "failed" and result["error"] == "native load failed"
    assert not pool.active and sentinel.read_text() == "original"


def test_validation_before_enqueue(setup, monkeypatch):
    jobs, pool = setup
    with pytest.raises(KeyError):
        jobs.start_calibration(
            DiffusionCalibrationRequest(
                model_id="remote/repo", tasks=[{"prompt": "test"}]
            )
        )
    pool.entry.source_type = "cluster"
    with pytest.raises(ValueError, match="local"):
        jobs.start_calibration(request())
    pool.entry.source_type = "local"
    with pytest.raises(ValueError, match="negative_prompt"):
        jobs.start_calibration(
            DiffusionCalibrationRequest(
                model_id="local", tasks=[{"prompt": "test", "negative_prompt": "bad"}]
            )
        )
    monkeypatch.setattr(module, "read_quantization", lambda path: 4)
    with pytest.raises(ValueError, match="floating-point"):
        jobs.start_quantization(
            DiffusionQuantizationRequest(model_id="local", calibration_job_id="missing")
        )
    assert not jobs._workers and not jobs._jobs


def test_request_strict_options():
    with pytest.raises(ValidationError):
        DiffusionCalibrationRequest(
            model_id="local", tasks=[{"prompt": "test", "image_path": "source"}]
        )
    with pytest.raises(ValidationError):
        DiffusionQuantizationRequest(
            model_id="local",
            calibration_job_id="id",
            budget_bytes=100,
            budget_ratio=1.2,
        )
    with pytest.raises(ValidationError):
        DiffusionQuantizationRequest(model_id="local", calibration_job_id="id", bits=2)


@pytest.mark.asyncio
async def test_restart_fails_interrupted_job(setup):
    jobs, pool = setup
    job = jobs.start_calibration(request())
    # Simulate the persisted running snapshot without interrupting actual work.
    jobs._jobs[job["id"]]["status"] = "running"
    jobs._persist()
    restored = DiffusionJobs(pool, jobs.artifacts_dir, jobs.output_dir)
    assert restored.get(job["id"])["status"] == "failed"
    assert restored.get(job["id"])["phase"] == "interrupted"
    jobs.cancel(job["id"])
    await drain(jobs)


@pytest.mark.asyncio
async def test_api_contract_and_management_auth(setup, monkeypatch):
    jobs, _ = setup
    app = FastAPI()
    app.include_router(router)
    app.state.diffusion_jobs_provider = lambda: jobs
    app.state.management_auth_provider = lambda: AuthContext(
        "main-key", [SimpleNamespace(key="sub-key")], "127.0.0.1"
    )
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get("/management/v1/diffusion/jobs")).status_code == 401
        assert (
            await client.get(
                "/management/v1/diffusion/jobs",
                headers={"Authorization": "Bearer sub-key"},
            )
        ).status_code == 401
        client.headers["Authorization"] = "Bearer main-key"
        assert (await client.get("/management/v1/diffusion/jobs")).json() == {
            "jobs": []
        }
        assert (
            await client.get("/management/v1/diffusion/jobs/missing")
        ).status_code == 404
        assert (
            await client.post(
                "/management/v1/diffusion/calibrations",
                json={
                    "model_id": "local",
                    "tasks": [{"prompt": "test"}],
                    "download": True,
                },
            )
        ).status_code == 422
        response = await client.post(
            "/management/v1/diffusion/calibrations",
            json={"model_id": "local", "tasks": [{"prompt": "test"}]},
        )
        assert response.status_code == 202
        job = response.json()
        assert (
            await client.post(f"/management/v1/diffusion/jobs/{job['id']}/cancel")
        ).status_code == 200
        await drain(jobs)


@pytest.mark.asyncio
async def test_waiting_gate_cancellation_never_runs_native_worker(setup, monkeypatch):
    jobs, pool = setup
    calls = []

    @asynccontextmanager
    async def busy_gate(check_cancel):
        while True:
            check_cancel()
            await asyncio.sleep(0.01)
        yield

    monkeypatch.setattr(pool, "exclusive_preparation", busy_gate)
    monkeypatch.setattr(
        module, "calibrate_diffusion", lambda *args, **kwargs: calls.append(True)
    )
    job = jobs.start_calibration(request())
    await asyncio.sleep(0.01)
    assert jobs.get(job["id"])["status"] == "waiting"
    jobs.cancel(job["id"])
    await drain(jobs)
    assert jobs.get(job["id"])["status"] == "cancelled" and calls == []


@pytest.mark.asyncio
async def test_calibration_report_validation_and_source_output_overlap(
    setup, monkeypatch
):
    jobs, pool = setup

    def calibrate(source, output, tasks, **kwargs):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"base_model": "flux2-klein-4b"}))
        return {}

    monkeypatch.setattr(module, "calibrate_diffusion", calibrate)
    job = jobs.start_calibration(request())
    await drain(jobs)
    body = DiffusionQuantizationRequest(model_id="local", calibration_job_id=job["id"])
    report = jobs.artifacts_dir / f"{job['id']}.json"
    report.write_text("[]")
    with pytest.raises(ValueError, match="identities"):
        jobs.start_quantization(body)
    report.unlink()
    with pytest.raises(ValueError, match="no longer available"):
        jobs.start_quantization(body)
    jobs.artifacts_dir = Path(pool.entry.model_path)
    with pytest.raises(ValueError, match="separate"):
        jobs.start_calibration(request())
    assert len(jobs._jobs) == 1


@pytest.mark.asyncio
async def test_reporting_failure_cannot_release_gate_before_worker_drains(
    setup, monkeypatch
):
    jobs, pool = setup
    started, unblock, cleaned = threading.Event(), threading.Event(), threading.Event()

    def calibrate(source, output, tasks, *, observer, **kwargs):
        started.set()
        try:
            assert unblock.wait(5)
            observer("loading", 0.2, "loaded")
        finally:
            cleaned.set()

    monkeypatch.setattr(module, "calibrate_diffusion", calibrate)
    job = jobs.start_calibration(request())
    worker = jobs._workers[job["id"]]
    while not started.is_set():
        await asyncio.sleep(0.01)

    def disk_full():
        raise OSError("disk full")

    monkeypatch.setattr(jobs, "_persist", disk_full)
    worker.cancel()
    await asyncio.sleep(0.01)
    assert pool.active and not worker.done() and not cleaned.is_set()
    assert jobs.get(job["id"])["status"] == "cancelling"
    unblock.set()
    await worker
    assert cleaned.is_set() and not pool.active
    assert jobs.get(job["id"])["status"] == "cancelled"


@pytest.mark.asyncio
async def test_discovered_hf_cache_is_local_but_remote_sources_are_rejected(setup):
    jobs, pool = setup
    pool.entry.source_type = "hf_cache"
    job = jobs.start_calibration(request())
    assert job["status"] == "queued"
    jobs.cancel(job["id"])
    await asyncio.gather(*list(jobs._workers.values()), return_exceptions=True)
    for source_type in ("cluster", "remote"):
        pool.entry.source_type = source_type
        with pytest.raises(ValueError, match="local"):
            jobs.start_calibration(request())
    assert len(jobs._jobs) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_workflow_traceback_references_collected_before_gate_exit(
    setup, monkeypatch, cancelled
):
    import weakref

    jobs, pool = setup
    references = []
    exit_checks = []

    class NativePayload:
        pass

    def fail(source, output, tasks, **kwargs):
        native = NativePayload()
        native.cycle = native
        references.append(weakref.ref(native))
        if cancelled:
            raise module.PreparationCancelledError("cancelled")
        raise RuntimeError("native failed")

    @asynccontextmanager
    async def exclusive(check_cancel):
        pool.active = True
        try:
            yield None
        finally:
            exit_checks.append(references[0]() is None)
            pool.active = False

    monkeypatch.setattr(module, "calibrate_diffusion", fail)
    monkeypatch.setattr(pool, "exclusive_preparation", exclusive)
    job = jobs.start_calibration(request())
    await drain(jobs)
    assert exit_checks == [True]
    assert jobs.get(job["id"])["status"] == ("cancelled" if cancelled else "failed")


@pytest.mark.parametrize("root", [None, 1, {}, "invalid"])
def test_invalid_history_root_raises_value_error(setup, root):
    jobs, pool = setup
    jobs.artifacts_dir.mkdir(parents=True, exist_ok=True)
    jobs._history.write_text(json.dumps(root))
    with pytest.raises(ValueError, match="list"):
        DiffusionJobs(pool, jobs.artifacts_dir, jobs.output_dir)
