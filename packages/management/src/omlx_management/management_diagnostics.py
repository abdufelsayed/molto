"""Durable diagnostic orchestration with one shared inference admission lock."""

from __future__ import annotations

import asyncio
import copy
import json
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from omlx_runtime.diagnostics import (
    accuracy_benchmark,
    ane_tuning,
    benchmark,
    context_benchmark,
)
from omlx_runtime.diagnostics.lifecycle import (
    cancellation_requested,
    request_cancellation,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from omlx_management.management import ManagementError
from omlx_management.model_control import _atomic_json_write


class DiagnosticRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["throughput", "accuracy", "context", "ane"]
    options: dict[str, Any] = Field(default_factory=dict)


RUNNERS = {
    "throughput": (
        benchmark.BenchmarkRequest,
        benchmark.BenchmarkRun,
        benchmark.run_benchmark,
    ),
    "accuracy": (
        accuracy_benchmark.AccuracyBenchmarkRequest,
        accuracy_benchmark.AccuracyBenchmarkRun,
        accuracy_benchmark.run_accuracy_benchmark,
    ),
    "context": (
        context_benchmark.ContextBenchmarkRequest,
        context_benchmark.ContextBenchmarkRun,
        context_benchmark.run_context_benchmark,
    ),
    "ane": (
        ane_tuning.ANETuningRequest,
        ane_tuning.ANETuningRun,
        ane_tuning.run_tuning,
    ),
}


def now():
    return datetime.now(UTC).isoformat()


def redact(value):
    if isinstance(value, dict):
        return {
            k: redact(v)
            for k, v in value.items()
            if k.lower()
            not in {"api_key", "authorization", "token", "password", "secret"}
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


class DiagnosticsService:
    def __init__(self, runtime, *, runners=None, path: Path | None = None):
        self.runtime = runtime
        self.runners = runners or RUNNERS
        self.path = path or Path(runtime.settings.base_path) / "diagnostics.json"
        self.active = {}
        self.tasks = {}
        try:
            self.records = json.loads(self.path.read_text())
            if not isinstance(self.records, dict):
                raise ValueError("diagnostics history must be an object")
        except FileNotFoundError:
            self.records = {}
        except (ValueError, OSError) as exc:
            raise ManagementError(
                "persistence", "Cannot read diagnostics history"
            ) from exc
        recovered = False
        for record in self.records.values():
            if record["status"] in {"running", "cancelling", "queued"}:
                record.update(
                    status="error",
                    error="Server stopped before this run finished",
                    updated_at=now(),
                )
                recovered = True
        if recovered:
            self._save()

    def _save(self):
        try:
            _atomic_json_write(self.path, self.records)
        except (OSError, ValueError, TypeError) as exc:
            raise ManagementError(
                "persistence", "Cannot persist diagnostics history"
            ) from exc

    def _ane_unavailable(self, backend="qwen"):
        try:
            from omlx_runtime.custom_kernels.qwen35_prefill import fast

            if not fast.qwen35_ane_available():
                return "Native ANE runtime is unavailable on this host"
            if backend == "qwen" and not fast.qwen35_ane_bank_compiler_available():
                return "Native ANE procedure-bank compiler is unavailable"
            if backend == "k2" and not hasattr(fast._ext, "ane_compile_program"):
                return "K2 ANE compiler is unavailable"
        except (ImportError, AttributeError, RuntimeError):
            return "Native ANE extension is unavailable"
        return None

    def capabilities(self):
        models = []
        pool = self.runtime.context.engine_pool
        for model in pool.get_status().get("models", []):
            entry = pool.get_model_view(model["id"])
            engine_type = getattr(entry, "engine_type", "")
            model_type = getattr(entry, "config_model_type", "") or ""
            language = engine_type in {"batched", "simple", "vlm"}
            supported_ane = model_type in {"qwen3_5", "qwen3_5_moe", "k2_horizon"}
            kinds = ["throughput", "accuracy", "context"] if language else []
            ane_reason = (
                self._ane_unavailable("k2" if model_type == "k2_horizon" else "qwen")
                if supported_ane
                else "ANE tuning requires a supported Qwen 3.5 or K2 architecture"
            )
            if language and supported_ane and ane_reason is None:
                kinds.append("ane")
            models.append(
                {
                    "model_id": model["id"],
                    "kinds": kinds,
                    "disabled_reason": None
                    if language
                    else "Diagnostics require a language model",
                    "ane_disabled_reason": ane_reason,
                }
            )
        return {
            "models": models,
            "suites": [
                {"id": name, "label": name.replace("_", " ").upper()}
                for name in accuracy_benchmark.VALID_BENCHMARKS
            ],
            "kinds": [
                {
                    "kind": k,
                    "targets": ["local", "external"]
                    if k in {"throughput", "accuracy"}
                    else ["local"],
                    "request_schema": spec[0].model_json_schema(),
                }
                for k, spec in self.runners.items()
            ],
            "publication": False,
            "queue": False,
            "prompt_lengths": benchmark.VALID_PROMPT_LENGTHS,
            "batch_sizes": benchmark.VALID_BATCH_SIZES,
            "context_targets": context_benchmark.VALID_TARGET_TOKENS,
            "cancellation": "cooperative_drain",
        }

    def list(self):
        return {
            "runs": sorted(
                (copy.deepcopy(r) for r in self.records.values()),
                key=lambda r: r["created_at"],
                reverse=True,
            )
        }

    def get(self, run_id):
        if run_id not in self.records:
            raise ManagementError("not_found", "Diagnostic run not found")
        return copy.deepcopy(self.records[run_id])

    def results(self, run_id):
        record = self.get(run_id)
        result = {k: record[k] for k in ("id", "kind", "status", "results", "error")}
        if "recommendation" in record:
            result["recommendation"] = record["recommendation"]
        return result

    async def start(self, request: DiagnosticRequest):
        request_type, run_type, runner = self.runners[request.kind]
        unknown = set(request.options) - set(request_type.model_fields)
        if unknown:
            raise ManagementError(
                "invalid", f"Unknown diagnostic options: {', '.join(sorted(unknown))}"
            )
        try:
            options = request_type.model_validate_json(json.dumps(request.options))
        except ValidationError as exc:
            raise ManagementError("invalid", str(exc)) from exc
        if not options.model_id.strip():
            raise ManagementError("invalid", "model_id must not be empty")
        external = getattr(options, "external", None)
        if external is not None:
            parsed = urlsplit(external.base_url)
            if (
                not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ManagementError(
                    "invalid",
                    "Endpoint must have a hostname and no embedded credentials, query or fragment",
                )
        if external is None:
            entry = self.runtime.context.engine_pool.get_model_view(options.model_id)
            if entry is None:
                raise ManagementError("not_found", "Model not found")
            if request.kind == "ane" and runner is ane_tuning.run_tuning:
                mt = getattr(entry, "config_model_type", "")
                allowed = (
                    {"k2_horizon"}
                    if options.backend == "k2"
                    else {"qwen3_5", "qwen3_5_moe"}
                )
                reason = self._ane_unavailable(options.backend)
                if reason:
                    raise ManagementError("invalid", reason)
                if mt not in allowed:
                    raise ManagementError(
                        "invalid",
                        "Model architecture does not support the selected ANE backend",
                    )
            if getattr(entry, "engine_type", "") not in {"batched", "simple", "vlm"}:
                raise ManagementError("invalid", "Diagnostics require a language model")
        if (
            getattr(options, "generation_length", 1) < 1
            or getattr(options, "generation_length", 1) > 65536
        ):
            raise ManagementError(
                "invalid", "generation_length must be between 1 and 65536"
            )
        if self.runtime.operation_lock.locked():
            raise ManagementError(
                "busy", "Another inference or preparation operation is active"
            )
        await self.runtime.operation_lock.acquire()
        # Existing requests must finish before old runner lifecycle transitions.
        pool = self.runtime.context.engine_pool
        if external is None and pool.has_active_requests():
            self.runtime.operation_lock.release()
            raise ManagementError(
                "busy", "Wait for active inference requests to finish"
            )
        run_id = uuid.uuid4().hex
        try:
            # ANE creation populates candidate slots and backend-specific limits.
            run = (
                ane_tuning.create_run(options)
                if request.kind == "ane" and runner is ane_tuning.run_tuning
                else run_type(run_id, options)
            )
            if request.kind == "ane" and runner is ane_tuning.run_tuning:
                ane_tuning._runs.pop(run.tuning_id, None)
            record = {
                "id": run_id,
                "kind": request.kind,
                "status": "running",
                "created_at": now(),
                "updated_at": now(),
                "request": redact(options.model_dump(mode="json")),
                "progress": None,
                "results": [],
                "error": None,
            }
            self.records[run_id] = record
            self._save()
        except BaseException:
            self.records.pop(run_id, None)
            self.runtime.operation_lock.release()
            raise
        self.active[run_id] = run
        try:
            control = getattr(self.runtime, "control", None)
            if control is not None:

                async def execute(operation_id):
                    self.records[run_id]["operation_id"] = operation_id
                    await self._execute(run_id, run, runner)
                    final = self.get(run_id)
                    if final["status"] == "error":
                        raise ManagementError(
                            "diagnostic_failed", final["error"] or "Diagnostic failed"
                        )
                    if final["status"] == "cancelled":
                        raise asyncio.CancelledError()
                    return self.results(run_id)

                operation = control.start_operation(
                    "diagnostic_" + request.kind,
                    execute,
                    model_id=options.model_id,
                    payload={"run_id": run_id},
                    cancellable=True,
                )
                self.tasks[run_id] = control._tasks[operation["id"]]
            else:
                self.tasks[run_id] = asyncio.create_task(
                    self._execute(run_id, run, runner)
                )
        except Exception:
            self.active.pop(run_id, None)
            self.records[run_id].update(
                status="error", error="Cannot start diagnostic operation"
            )
            self.runtime.operation_lock.release()
            self._save()
            raise

        def cleanup_unstarted(task):
            if run_id in self.active:
                self.active.pop(run_id, None)
                self.records[run_id].update(status="cancelled", updated_at=now())
                self.runtime.operation_lock.release()
                self._save()

        self.tasks[run_id].add_done_callback(cleanup_unstarted)
        return self.get(run_id)

    def _snapshot(self, run_id, run):
        record = self.records[run_id]
        events = getattr(run, "events", [])
        if isinstance(run, ane_tuning.ANETuningRun):
            snapshot = ane_tuning.run_snapshot(run)
            record["recommendation"] = snapshot["recommendation"]
            record["results"] = snapshot["results"]
        progress = next(
            (e for e in reversed(events) if e.get("type") == "progress"), None
        )
        record.update(
            progress=redact(
                progress
                or {
                    "phase": getattr(run, "phase", "running"),
                    "progress": getattr(run, "progress", None),
                    "message": getattr(run, "message", ""),
                    "current": getattr(run, "current", None),
                    "total": getattr(run, "total", None),
                }
            ),
            results=redact(
                (
                    ane_tuning.run_snapshot(run)["results"]
                    if isinstance(run, ane_tuning.ANETuningRun)
                    else getattr(run, "results", None)
                )
                or ([run.result] if getattr(run, "result", None) else [])
            ),
            updated_at=now(),
        )
        self._save()
        operation_id = record.get("operation_id")
        if operation_id:
            progress = record["progress"] or {}
            current, total = progress.get("current"), progress.get("total")
            percent = (
                min(99.0, 100 * current / total)
                if isinstance(current, (int, float))
                and isinstance(total, (int, float))
                and total > 0
                else 5.0
            )
            self.runtime.control.update_operation(
                operation_id,
                stage=str(progress.get("phase") or "running"),
                progress=percent,
            )

    async def _execute(self, run_id, run, runner):
        async def admitted_runner():
            pool = self.runtime.context.engine_pool
            admission = getattr(pool, "exclusive_management", None)
            if getattr(run.request, "external", None) is None and admission is not None:

                def check_cancel():
                    if cancellation_requested(run):
                        raise asyncio.CancelledError()

                async with admission(check_cancel):
                    await runner(run, pool)
            else:
                await runner(run, pool)

        worker = asyncio.create_task(admitted_runner())
        try:
            while not worker.done():
                await asyncio.wait({worker}, timeout=0.25)
                self._snapshot(run_id, run)
            await worker
            self._snapshot(run_id, run)
            record = self.records[run_id]
            record["status"] = (
                "cancelled"
                if record["status"] == "cancelling"
                else getattr(run, "status", "completed")
            )
            record["error"] = getattr(run, "error_message", "") or None
            external = getattr(run.request, "external", None)
            if external is not None and record["error"]:
                key = external.api_key.get_secret_value()
                if key:
                    record["error"] = record["error"].replace(key, "[redacted]")
        except asyncio.CancelledError:
            # Shutdown or external cancellation cannot release GPU admission early.
            request_cancellation(run)
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            self.records[run_id].update(status="cancelled")
        except Exception as exc:
            # Never abort an inference worker because a history write failed.
            with suppress(BaseException):
                await asyncio.shield(worker)
            self.records[run_id].update(
                status="error",
                error=type(exc).__name__ + ": diagnostic execution failed",
            )
        finally:
            self.records[run_id]["updated_at"] = now()
            self.active.pop(run_id, None)
            self.runtime.operation_lock.release()
            try:
                self._save()
            except ManagementError:
                self.records[run_id].update(
                    status="error", error="Cannot persist final diagnostic results"
                )
                raise

    def cancel(self, run_id):
        record = self.get(run_id)
        if record["status"] not in {"running", "cancelling"}:
            return record
        previous = self.records[run_id]["status"]
        self.records[run_id]["status"] = "cancelling"
        try:
            self._save()
        except Exception:
            self.records[run_id]["status"] = previous
            raise
        run = self.active.get(run_id)
        if run is not None:
            request_cancellation(run)
        return self.get(run_id)

    async def shutdown(self):
        for run_id in list(self.active):
            self.cancel(run_id)
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
