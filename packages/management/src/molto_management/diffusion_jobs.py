# SPDX-License-Identifier: Apache-2.0
"""Local diffusion preparation jobs, serialized with serving and safely drained."""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

from molto_contracts.management import (
    DiffusionCalibrationRequest,
    DiffusionJobView,
    DiffusionQuantizationRequest,
)
from molto_runtime.diffusion.checkpoint import read_quantization
from molto_runtime.diffusion.preparation import (
    _source,
    calibrate_diffusion,
    quantize_diffusion,
)
from molto_runtime.diffusion.registry import ImageTask, get_pipeline, validate_task
from molto_runtime.engine_core import get_mlx_executor

logger = logging.getLogger(__name__)
_TERMINAL = {"completed", "failed", "cancelled"}


class PreparationCancelledError(Exception):
    """Cooperative cancellation checked on the actual MLX worker."""


class DiffusionJobs:
    def __init__(self, pool, artifacts_dir: Path, output_dir: Path, on_complete=None):
        self.pool = pool
        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.on_complete = on_complete
        self._jobs: dict[str, dict] = {}
        self._workers: dict[str, asyncio.Task] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._serial = asyncio.Lock()
        self._closing = False
        self._history = self.artifacts_dir / "jobs.json"
        if self._history.exists():
            records = json.loads(self._history.read_text())
            if not isinstance(records, list):
                raise ValueError("Diffusion job history must contain a list")
            for record in records:
                job = DiffusionJobView.model_validate(record).model_dump()
                # Artifact paths are derived from our job ID, never trusted from disk.
                if len(job["id"]) != 32 or any(
                    c not in "0123456789abcdef" for c in job["id"]
                ):
                    raise ValueError("Invalid persisted diffusion job ID")
                if job["status"] not in _TERMINAL:
                    job.update(
                        status="failed",
                        phase="interrupted",
                        error="Server stopped before the job completed",
                        finished_at=time.time(),
                    )
                self._jobs[job["id"]] = job
            self._persist()

    def _persist(self):
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        temporary = self._history.with_suffix(".tmp")
        try:
            temporary.write_text(
                json.dumps(list(self._jobs.values()), allow_nan=False) + "\n"
            )
            os.replace(temporary, self._history)
        finally:
            temporary.unlink(missing_ok=True)

    def get(self, job_id):
        if job_id not in self._jobs:
            raise KeyError("Diffusion job not found")
        return DiffusionJobView.model_validate(self._jobs[job_id]).model_dump()

    def list(self):
        return {"jobs": [self.get(key) for key in reversed(self._jobs)]}

    def _checkpoint(self, model_id):
        entry = self.pool.get_model_view(model_id)
        if entry is None:
            raise KeyError("Discovered model not found")
        if entry.engine_type != "image_generation" or getattr(
            entry, "source_type", "local"
        ) not in {"local", "hf_cache"}:
            raise ValueError("Preparation requires a discovered local image model")
        return _source(entry.model_path)

    def start_calibration(self, body: DiffusionCalibrationRequest):
        checkpoint = self._checkpoint(body.model_id)
        tasks = [ImageTask(**task.model_dump()) for task in body.tasks]
        spec = get_pipeline(checkpoint.base_model)
        for task in tasks:
            validate_task(spec, task)
        return self._enqueue(
            "calibration",
            body.model_id,
            checkpoint,
            tasks=tasks,
            max_rows=body.max_rows,
        )

    def start_quantization(self, body: DiffusionQuantizationRequest):
        checkpoint = self._checkpoint(body.model_id)
        if read_quantization(checkpoint.path) is not None:
            raise ValueError(
                "Fresh quantization requires floating-point weights; packed checkpoints are calibration-only"
            )
        calibration = self.get(body.calibration_job_id)
        if (
            calibration["kind"] != "calibration"
            or calibration["status"] != "completed"
            or calibration["base_model"] != checkpoint.base_model
        ):
            raise ValueError(
                "A completed calibration job for the same checkpoint identity is required"
            )
        report = self.artifacts_dir / f"{calibration['id']}.json"
        if not report.is_file():
            raise ValueError("Calibration report is no longer available")
        data = json.loads(report.read_text())
        if (
            not isinstance(data, dict)
            or data.get("base_model") != checkpoint.base_model
        ):
            raise ValueError("Calibration report and checkpoint identities disagree")
        options = body.model_dump(exclude={"model_id", "calibration_job_id"})
        return self._enqueue(
            "quantization", body.model_id, checkpoint, calibration=report, **options
        )

    def _enqueue(self, kind, model_id, checkpoint, **options):
        if self._closing:
            raise ValueError("Diffusion preparation is shutting down")
        job_id = uuid.uuid4().hex
        output = (
            (self.artifacts_dir / f"{job_id}.json")
            if kind == "calibration"
            else (self.output_dir / f"{checkpoint.base_model}-oq-{job_id}")
        )
        source = checkpoint.path.resolve()
        resolved = output.resolve()
        if (
            resolved == source
            or source in resolved.parents
            or resolved in source.parents
        ):
            raise ValueError(
                "Preparation output must be separate from its source checkpoint"
            )
        if output.exists():
            raise ValueError("Preparation output already exists")
        self._jobs[job_id] = dict(
            id=job_id,
            kind=kind,
            model_id=model_id,
            base_model=checkpoint.base_model,
            status="queued",
            phase="queued",
            progress=0.0,
            detail="Waiting for preparation slot",
            created_at=time.time(),
            started_at=None,
            finished_at=None,
            result=None,
            error=None,
        )
        self._cancel[job_id] = threading.Event()
        try:
            self._persist()
        except OSError:
            self._jobs.pop(job_id, None)
            self._cancel.pop(job_id, None)
            raise
        self._workers[job_id] = asyncio.create_task(
            self._run(job_id, checkpoint.path, output, options)
        )
        return self.get(job_id)

    def _update(self, job_id, **values):
        self._jobs[job_id].update(values)
        try:
            self._persist()
        except OSError:
            # Reporting must never interrupt cancellation drainage or release
            # pool ownership while the native worker is still running. Keep
            # the authoritative in-memory state usable even if disk is full.
            logger.exception("Could not persist diffusion job %s", job_id)

    def cancel(self, job_id):
        self.get(job_id)
        if self._jobs[job_id]["status"] not in _TERMINAL:
            self._cancel[job_id].set()
            if self._jobs[job_id]["status"] == "queued":
                worker = self._workers[job_id]
                worker.cancel()
                self._update(
                    job_id,
                    status="cancelled",
                    phase="cancelled",
                    detail="Cancelled before preparation started",
                    finished_at=time.time(),
                )

                def remove(_):
                    self._cancel.pop(job_id, None)
                    self._workers.pop(job_id, None)

                worker.add_done_callback(remove)
            else:
                self._update(
                    job_id,
                    status="cancelling",
                    detail="Cancellation requested; waiting for worker cleanup",
                )
        return self.get(job_id)

    async def _run(self, job_id, source, output, options):
        loop = asyncio.get_running_loop()
        token = self._cancel[job_id]
        allowance = None
        last_progress = ("", -1.0, 0.0)

        def check_cancel():
            if token.is_set():
                raise PreparationCancelledError("Preparation cancelled")

        def progress(phase, amount, detail):
            nonlocal last_progress
            check_cancel()
            now = time.monotonic()
            previous_phase, previous_amount, previous_time = last_progress
            if (
                phase == previous_phase
                and abs(amount - previous_amount) < 0.01
                and now - previous_time < 0.25
            ):
                return
            last_progress = (phase, amount, now)

            def apply():
                if self._jobs[job_id]["status"] in _TERMINAL:
                    return
                self._update(
                    job_id,
                    phase=phase,
                    progress=max(0.0, min(1.0, amount)),
                    detail=detail,
                )

            loop.call_soon_threadsafe(apply)

        def perform():
            # Do not pass exception objects or native tracebacks to asyncio.
            # They retain workflow frames (and model tensors) even after its
            # finally block has released the backend's explicit references.
            try:
                check_cancel()
                if self._jobs[job_id]["kind"] == "calibration":
                    result = calibrate_diffusion(
                        source,
                        output,
                        observer=progress,
                        memory_limit_bytes=allowance,
                        **options,
                    )
                else:
                    result = quantize_diffusion(
                        source,
                        output=output,
                        observer=progress,
                        memory_limit_bytes=allowance,
                        **options,
                    )
                summary = {
                    "output_path": str(output),
                    "seconds": result.get("seconds"),
                    "base_model": result.get("base_model"),
                }
                return "completed", summary, None
            except PreparationCancelledError:
                return "cancelled", None, None
            except Exception as exc:
                return "failed", None, str(exc)

        def work():
            outcome = perform()
            # perform's frame and exception reference are gone before collection.
            # Cleanup runs on the shared MLX executor while the pool gate remains
            # held, including after cancellation and failed model operations.
            gc.collect()
            try:
                import mlx.core as mx

                mx.synchronize()
                mx.clear_cache()
            except Exception as exc:
                outcome = "failed", None, f"Preparation cleanup failed: {exc}"
            return outcome

        try:
            async with self._serial:
                check_cancel()
                self._update(
                    job_id,
                    status="waiting",
                    phase="waiting",
                    detail="Waiting for serving to drain",
                )
                async with self.pool.exclusive_preparation(check_cancel) as allowance:
                    check_cancel()
                    self._update(job_id, status="running", started_at=time.time())
                    future = loop.run_in_executor(get_mlx_executor(), work)
                    # Cancellation of this coordinator cannot release pool ownership while native work remains.
                    while True:
                        try:
                            status, summary, error = await asyncio.shield(future)
                            break
                        except asyncio.CancelledError:
                            token.set()
                            self._update(job_id, status="cancelling")
                    self._update(
                        job_id,
                        status=status,
                        phase=status,
                        progress=1.0
                        if status == "completed"
                        else self._jobs[job_id]["progress"],
                        detail=f"Preparation {status} after cleanup",
                        result=summary,
                        error=error,
                        finished_at=time.time(),
                    )
            if (
                self.on_complete is not None
                and self._jobs[job_id]["kind"] == "quantization"
                and self._jobs[job_id]["status"] == "completed"
            ):
                try:
                    await self.on_complete()
                except Exception:
                    logger.exception(
                        "Diffusion checkpoint saved but inventory refresh failed"
                    )
        except (PreparationCancelledError, asyncio.CancelledError):
            self._update(
                job_id,
                status="cancelled",
                phase="cancelled",
                detail="Preparation cancelled after cleanup",
                finished_at=time.time(),
            )
        except Exception as exc:
            self._update(
                job_id,
                status="failed",
                phase="failed",
                detail="Preparation failed after cleanup",
                error=str(exc),
                finished_at=time.time(),
            )
        finally:
            # Workflows clean their own staging and atomically publish outputs.
            self._cancel.pop(job_id, None)
            self._workers.pop(job_id, None)

    async def shutdown(self):
        self._closing = True
        for job_id in list(self._workers):
            self.cancel(job_id)
        if self._workers:
            await asyncio.gather(*list(self._workers.values()), return_exceptions=True)
