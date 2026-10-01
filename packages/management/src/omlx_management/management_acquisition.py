"""Explicit discovery, preparation and publication workflows."""

from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path

from omlx_management.management import ManagementError, ManagementService
from omlx_management.model_control import (
    build_preparation_catalog,
    build_registry_record,
    discover_unmanaged_artifacts,
)

ACTIVE = {"queued", "running"}
RETRYABLE = {"failed", "cancelled", "interrupted"}
LEVELS = [2, 2.5, 2.7, 3, 3.5, 4, 5, 6, 8]


def repository_id(value):
    if (
        not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", value
        )
        or ".." in value
    ):
        raise ManagementError(
            "invalid_configuration", "Repository must have the form owner/model"
        )
    return value


def embedding_dtype(source):
    from safetensors import safe_open

    dtypes = set()
    for shard in source.glob("*.safetensors"):
        with safe_open(str(shard), framework="numpy") as weights:
            for name in weights.keys():  # noqa: SIM118 - safetensors reader is not iterable
                dtype = weights.get_slice(name).get_dtype()
                dtypes.add(dtype)
    mapping = {"F16": "float16", "BF16": "bfloat16", "F32": "float32"}
    if len(dtypes) != 1 or next(iter(dtypes)) not in mapping:
        raise ValueError(
            "Embedding conversion requires one supported floating-point precision; mixed precision cannot be preserved by this converter"
        )
    return mapping[next(iter(dtypes))]


def convert_local(adapter, source, output, on_stage):
    if adapter == "mflux":
        from omlx_runtime.mflux_conversion import convert_mflux_model

        return convert_mflux_model(str(source), output, None, on_stage=on_stage)
    import importlib

    modules = {
        "mlx-lm": "mlx_lm",
        "mlx-vlm": "mlx_vlm",
        "mlx-embeddings": "mlx_embeddings",
        "mlx-audio": "mlx_audio",
    }
    convert = importlib.import_module(f"{modules[adapter]}.convert").convert
    on_stage("converting")
    convert(
        hf_path=str(source),
        mlx_path=str(output),
        quantize=False,
        dtype=embedding_dtype(source) if adapter == "mlx-embeddings" else None,
    )
    return {
        "source": str(source),
        "output": str(output),
        "adapter": adapter,
        "quantize": False,
    }


class AcquisitionService:
    def __init__(self, runtime):
        self.runtime = runtime
        self.control = runtime.control

    def manager(self, provider):
        if provider not in {"hf", "ms"}:
            raise ManagementError("not_found", "Unknown acquisition provider")
        return getattr(self.runtime, provider)

    async def search(self, provider, **options):
        for low, high in (("min_params", "max_params"), ("min_size", "max_size")):
            if (
                options.get(low) is not None
                and options.get(high) is not None
                and options[low] > options[high]
            ):
                raise ManagementError(
                    "invalid_configuration", "Minimum filter exceeds maximum"
                )
        if provider == "ms":
            unsupported = {
                key: options.pop(key, None)
                for key in (
                    "min_params",
                    "max_params",
                    "min_size",
                    "max_size",
                    "sort_by_size",
                    "sort_ascending",
                )
            }
            if any(
                value is not None and value is not False
                for value in unsupported.values()
            ):
                raise ManagementError(
                    "invalid_configuration",
                    "ModelScope does not support size or parameter filtering",
                )
        return await self.manager(provider).search_models(**options)

    async def recommended(self, provider, **options):
        return await self.manager(provider).get_recommended_models(**options)

    async def info(self, provider, repo_id):
        return await self.manager(provider).get_model_info(repository_id(repo_id))

    def local_path(self, value):
        path = Path(value).expanduser().resolve()
        if not any(
            path.is_relative_to(root) and path != root for root in self.runtime.roots
        ):
            raise ManagementError(
                "invalid_configuration", "Select a model inside configured storage"
            )
        if not path.is_dir():
            raise ManagementError("not_found", "Model directory not found")
        # Include nested directory aliases without following external targets.
        if any(
            not any(child.is_relative_to(root) for root in self.runtime.roots)
            for child in self.runtime.normalized_paths([path])
        ):
            raise ManagementError(
                "invalid_configuration",
                "Model contains a link outside configured storage",
            )
        return path

    async def models(self):
        sources, all_models = await self.runtime.oq.list_quantizable_models()
        raw = ManagementService(self.runtime.context).inventory()["models"]
        known = {str(Path(item.get("model_path", "")).resolve()) for item in raw}
        raw += discover_unmanaged_artifacts(self.runtime.roots, known)
        records = [build_registry_record(item, self.control.store) for item in raw]
        models = build_preparation_catalog(records, sources, all_models)
        for item in models:
            if item["conversion"].get("adapter") == "mlx-embeddings" and item[
                "conversion"
            ].get("available"):
                try:
                    embedding_dtype(Path(item["path"]))
                except Exception as exc:
                    item["conversion"].update(available=False, reason=str(exc))
        return {"models": models, "options": self.options()}

    def options(self):
        return {
            "oq_levels": LEVELS,
            "group_sizes": [32, 64, 128],
            "dtypes": ["bfloat16", "float16"],
            "exclusive_busy": self.runtime.operation_lock.locked(),
        }

    async def selected(self, value, capability=None):
        path = self.local_path(value)
        item = next(
            (
                m
                for m in (await self.models())["models"]
                if Path(m["path"]).resolve() == path
            ),
            None,
        )
        if item is None:
            raise ManagementError(
                "invalid_configuration",
                "Select a local model from the preparation catalog",
            )
        if capability and not item[capability]["available"]:
            raise ManagementError(
                "conflict", item[capability].get("reason") or "Operation unavailable"
            )
        return item

    async def _wait_manager(self, operation_id, manager, task, kind):
        raw_id = task.task_id
        native = manager._active_tasks.get(raw_id)
        try:
            while native and not native.done():
                data = next(t for t in manager.get_tasks() if t["task_id"] == raw_id)
                self.control.update_operation(
                    operation_id,
                    progress=data.get("progress", 0),
                    stage=data.get("phase") or data["status"],
                )
                await asyncio.sleep(0.2)
            if native:
                await asyncio.shield(native)
        except BaseException:
            if kind == "quantize":
                await self.runtime.cancel_quantization(manager, raw_id)
            elif kind == "publish":
                # Upload SDK workers cannot be safely aborted. Drain them on shutdown.
                if native:
                    while not native.done():
                        try:
                            await asyncio.shield(native)
                        except asyncio.CancelledError:
                            continue
            else:
                manager._cancelled.add(raw_id)
                if native:
                    while not native.done():
                        try:
                            await asyncio.shield(native)
                        except asyncio.CancelledError:
                            continue
            raise
        data = next(t for t in manager.get_tasks() if t["task_id"] == raw_id)
        if data["status"] in {"failed", "cancelled"}:
            if data["status"] == "cancelled":
                raise asyncio.CancelledError
            raise RuntimeError(data.get("error") or "Worker failed")
        if data["status"] not in {"completed", "succeeded"}:
            raise RuntimeError(f"Worker stopped in state {data['status']}")
        return data

    def start_paths_operation(self, kind, runner, paths, **kwargs):
        owner = uuid.uuid4().hex
        self.runtime.reserve_paths(paths, owner)

        async def guarded(op):
            try:
                return await runner(op)
            finally:
                self.runtime.release_paths(owner)

        try:
            operation = self.control.start_operation(kind, guarded, **kwargs)
        except BaseException:
            self.runtime.release_paths(owner)
            raise
        self.control._tasks[operation["id"]].add_done_callback(
            lambda _: self.runtime.release_paths(owner)
        )
        return operation

    def download_destination(self, provider, repo_id):
        root = self.runtime.roots[0]
        destination = root / repo_id
        if not destination.resolve().is_relative_to(root):
            raise ManagementError(
                "invalid_configuration",
                "Download destination escapes configured storage",
            )
        if any(
            Path(model.get("model_path", "/unused"))
            .resolve()
            .is_relative_to(destination.resolve())
            or destination.resolve().is_relative_to(
                Path(model.get("model_path", "/unused")).resolve()
            )
            for model in self.runtime.context.engine_pool.get_status()["models"]
        ):
            raise ManagementError(
                "conflict", "A registered model already occupies this destination"
            )
        if destination.exists():
            self.local_path(str(destination))
            owned = any(
                op.get("kind") == f"download_{provider}"
                and op.get("status") in RETRYABLE
                and op.get("payload", {}).get("destination") == str(destination)
                for op in self.control.operations(limit=100000)
            )
            if not owned:
                raise ManagementError(
                    "conflict",
                    "Destination already exists and is not an interrupted management download",
                )
        return destination

    async def download(self, provider, repo_id, token=""):
        repository_id(repo_id)
        destination = self.download_destination(provider, repo_id)
        manager = self.manager(provider)

        async def run(op):
            try:
                task = await manager.start_download(repo_id, token)
                return await self._wait_manager(op, manager, task, "download")
            except Exception as exc:
                raise RuntimeError(
                    str(exc).replace(token, "[redacted]") if token else str(exc)
                ) from None

        return self.start_paths_operation(
            f"download_{provider}",
            run,
            [destination],
            model_id=repo_id,
            payload={
                "provider": provider,
                "repo_id": repo_id,
                "destination": str(destination),
            },
        )

    async def convert(self, model_path, output_name=None):
        item = await self.selected(model_path, "conversion")
        output_name = output_name or item["conversion"].get("output_name")
        if (
            not output_name
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", output_name)
            or ".." in output_name
        ):
            raise ManagementError(
                "invalid_configuration",
                "Output name must be a single model directory name",
            )
        output = self.runtime.roots[0] / output_name
        if output.is_symlink() or output.resolve().parent != self.runtime.roots[0]:
            raise ManagementError(
                "invalid_configuration", "Conversion output escapes configured storage"
            )
        if output.exists():
            raise ManagementError("conflict", "Output already exists")
        payload = {"model_path": item["path"], "output_name": output_name}

        async def run(op):
            def stage(name):
                self.control.update_operation(op, stage=name, progress=30)

            result = await self.runtime.run_native(
                convert_local,
                item["conversion"]["adapter"],
                Path(item["path"]),
                output,
                stage,
            )
            return result

        return await self.runtime.start_exclusive(
            "convert",
            run,
            model_id=item["id"],
            payload=payload,
            cancellable=False,
            paths=[item["path"], output],
        )

    async def estimate(self, model_path, oq_level, group_size=64, preserve_mtp=False):
        await self.selected(model_path, "quantization")
        self.validate_quant_options(oq_level, group_size)
        from omlx_runtime.oq import estimate_bpw_and_size

        path = self.local_path(model_path)
        owner = uuid.uuid4().hex
        self.runtime.reserve_paths([path], owner)
        try:
            return await self.runtime.run_native(
                estimate_bpw_and_size, str(path), oq_level, group_size, preserve_mtp
            )
        finally:
            self.runtime.release_paths(owner)

    def validate_quant_options(self, oq_level, group_size):
        if oq_level not in LEVELS or group_size not in {32, 64, 128}:
            raise ManagementError(
                "invalid_configuration", "Unsupported quantization level or group size"
            )

    async def quantize(self, **options):
        item = await self.selected(options["model_path"], "quantization")
        self.validate_quant_options(options["oq_level"], options.get("group_size", 64))
        for key in ("sensitivity_model_path", "mtp_assistant_model_path"):
            if options.get(key):
                options[key] = str(self.local_path(options[key]))
        if options.get("imatrix_cache_path"):
            path = Path(options["imatrix_cache_path"]).expanduser().resolve()
            if not any(path.is_relative_to(root) for root in self.runtime.roots):
                raise ManagementError(
                    "invalid_configuration",
                    "Calibration cache must be inside configured model storage",
                )
            options["imatrix_cache_path"] = str(path)
        options["model_path"] = item["path"]
        manager = self.runtime.oq

        async def run(op):
            task = await manager.start_quantization(**options)
            try:
                if (
                    Path(task.output_path).is_symlink()
                    or Path(task.output_path).resolve().parent != self.runtime.roots[0]
                ):
                    raise ManagementError(
                        "invalid_configuration",
                        "Quantization output escapes configured storage",
                    )
                if getattr(task, "imatrix_cache_path", "") and not any(
                    Path(task.imatrix_cache_path).resolve().is_relative_to(root)
                    for root in self.runtime.roots
                ):
                    raise ManagementError(
                        "invalid_configuration",
                        "Calibration cache escapes configured storage",
                    )
                self.control.update_operation(op, output_path=task.output_path)
                self.runtime.extend_operation_paths(
                    op,
                    [
                        task.output_path,
                        *(
                            [task.imatrix_cache_path]
                            if getattr(task, "imatrix_cache_path", "")
                            else []
                        ),
                    ],
                )
            except BaseException:
                await self.runtime.cancel_quantization(manager, task.task_id)
                raise
            return await self._wait_manager(op, manager, task, "quantize")

        return await self.runtime.start_exclusive(
            "quantize",
            run,
            model_id=item["id"],
            payload=options,
            paths=[
                item["path"],
                *[
                    options[key]
                    for key in (
                        "sensitivity_model_path",
                        "mtp_assistant_model_path",
                        "imatrix_cache_path",
                    )
                    if options.get(key)
                ],
            ],
        )

    async def validate_publish(self, token, model_path=None, repo_id=None):
        if model_path:
            path = self.local_path(model_path)
            if not (path / "config.json").is_file():
                raise ManagementError(
                    "invalid_configuration", "Publishing requires a model config.json"
                )
        if repo_id:
            repository_id(repo_id)
        if not token.strip():
            raise ManagementError(
                "invalid_configuration", "A Hugging Face write token is required"
            )
        try:
            result = await self.runtime.uploader.validate_token(token)
        except Exception as exc:
            raise ManagementError(
                "invalid_configuration", str(exc).replace(token, "[redacted]")
            ) from None
        # Return only the documented account/permission data, never token echoes.
        return {
            "valid": True,
            "can_write": True,
            "username": result.get("username", ""),
            "orgs": result.get("orgs", []),
        }

    async def publish(self, model_path, repo_id, token, **options):
        path = self.local_path(model_path)
        repository_id(repo_id)
        if not token.strip() or not (path / "config.json").is_file():
            raise ManagementError(
                "invalid_configuration",
                "Publishing requires config.json and a write token",
            )
        if options.get("readme_source_path"):
            options["readme_source_path"] = str(
                self.local_path(options["readme_source_path"])
            )
        manager = self.runtime.uploader

        async def run(op):
            try:
                task = await manager.start_upload(str(path), repo_id, token, **options)
                return await self._wait_manager(op, manager, task, "publish")
            except Exception as exc:
                raise RuntimeError(str(exc).replace(token, "[redacted]")) from None

        return self.start_paths_operation(
            "publish",
            run,
            [
                path,
                *(
                    [options["readme_source_path"]]
                    if options.get("readme_source_path")
                    else []
                ),
            ],
            model_id=path.name,
            payload={"model_path": str(path), "repo_id": repo_id, **options},
            cancellable=False,
        )

    def view(self, operation):
        operation = dict(operation)
        task = self.control._tasks.get(operation["id"])
        operation["actions"] = {
            "cancel": bool(
                task and not task.done() and operation.get("cancellable", True)
            ),
            "retry": operation["status"] in RETRYABLE
            and operation["kind"]
            in {"download_hf", "download_ms", "quantize", "convert"},
            "delete": operation["status"] not in ACTIVE
            and not (task and not task.done()),
        }
        if operation["actions"]["retry"] and operation["kind"] in {
            "convert",
            "quantize",
        }:
            output = operation.get("output_path")
            if operation["kind"] == "convert":
                output_name = operation.get("payload", {}).get("output_name")
                output = self.runtime.roots[0] / output_name if output_name else None
            if output and Path(output).exists():
                operation["actions"]["retry"] = False
                operation["retry_unavailable_reason"] = (
                    "An output directory already exists; inspect it in the library before starting again"
                )
        return operation

    def operations(self, limit=200, provider=None):
        values = self.control.operations(limit=limit)
        if provider:
            values = [v for v in values if v["kind"] == f"download_{provider}"]
        return {"operations": [self.view(v) for v in values]}

    def get(self, operation_id):
        operation = self.control.store.get("operations", operation_id)
        if not operation:
            raise ManagementError("not_found", "Operation not found")
        return self.view(operation)

    async def cancel(self, operation_id):
        operation = self.get(operation_id)
        if not operation["actions"]["cancel"]:
            raise ManagementError("conflict", "Operation cannot be cancelled")
        self.control.cancel_operation(operation_id)
        self.control.update_operation(operation_id, stage="cancelling")
        return self.get(operation_id)

    async def retry(self, operation_id, token=""):
        operation = self.get(operation_id)
        if not operation["actions"]["retry"]:
            raise ManagementError("conflict", "Operation cannot be retried")
        payload = operation["payload"]
        if operation["kind"].startswith("download_"):
            result = await self.download(payload["provider"], payload["repo_id"], token)
        elif operation["kind"] == "convert":
            result = await self.convert(**payload)
        else:
            result = await self.quantize(**payload)
        self.control.update_operation(result["id"], retry_of=operation_id)
        return self.get(result["id"])

    def delete(self, operation_id):
        operation = self.get(operation_id)
        if not operation["actions"]["delete"]:
            raise ManagementError("busy", "Active operation cannot be deleted")
        self.control.remove_operation(operation_id)
        return {"deleted": True, "id": operation_id}
