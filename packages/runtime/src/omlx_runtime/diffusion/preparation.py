"""Local-only calibration and transformer quantization preparation workflows."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import struct
import tempfile
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from omlx_runtime.diffusion.backend import MFluxBackend, _offline
from omlx_runtime.diffusion.checkpoint import (
    MANIFEST_NAME,
    detect_checkpoint,
    read_quantization,
)
from omlx_runtime.diffusion.registry import ImageTask, get_pipeline, validate_task

# Explicit first adapters. Adding a family requires checking its eager forward,
# native saver, and component policies. No inference-server conditionals.
QUANTIZATION_MODELS = ("flux2-klein-4b", "qwen-image-2.1")


class _Progress:
    """Synchronous notifications double as cooperative cancellation checks."""

    def __init__(self, observer):
        self.observer = observer
        self.value = 0.0

    def __call__(self, phase, progress, detail):
        self.value = max(self.value, min(1.0, max(0.0, progress)))
        if self.observer is not None:
            self.observer(phase, self.value, detail)


def _source(model):
    path = Path(model).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(
            "Diffusion preparation requires a local checkpoint directory; downloads are disabled"
        )
    checkpoint = detect_checkpoint(path)
    if checkpoint is None:
        raise ValueError(
            "Prepare a complete, identified local diffusion checkpoint first"
        )
    if checkpoint.base_model not in QUANTIZATION_MODELS:
        raise ValueError(
            "Calibrated quantization supports FLUX.2 Klein 4B and Qwen-Image-2.1 only"
        )
    return checkpoint


def _destination(output, source, *, directory=False):
    path = Path(output).expanduser().resolve()
    if path == source or source in path.parents or path in source.parents:
        raise ValueError("Output must be separate from the source checkpoint")
    if path.exists() and (not directory or not path.is_dir() or any(path.iterdir())):
        raise ValueError("Output already exists or is not an empty directory")
    return path


def _provenance(checkpoint):
    # Hash headers and file sizes, never load weight data just for provenance.
    digest = hashlib.sha256()
    weight_bytes = 0
    for path in sorted(checkpoint.path.rglob("*.safetensors")):
        size = path.stat().st_size
        weight_bytes += size
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            if header_size > 16 * 1024 * 1024:
                raise ValueError("Diffusion weight header exceeds metadata limit")
            header = stream.read(header_size)
            if len(header) != header_size:
                raise ValueError("Incomplete diffusion weight header")
        digest.update(str(path.relative_to(checkpoint.path)).encode())
        digest.update(str(size).encode())
        digest.update(header)
    return {
        "path": str(checkpoint.path),
        "base_model": checkpoint.base_model,
        "stored_quantization_bits": read_quantization(checkpoint.path),
        "header_size_fingerprint": digest.hexdigest(),
        "fingerprint_scope": "weight headers, relative paths and file sizes; not weight contents",
        "weight_file_bytes": weight_bytes,
    }


def _admit(provenance, *, memory_limit_bytes=None):
    import mlx.core as mx
    import psutil

    if memory_limit_bytes is not None and (
        type(memory_limit_bytes) is not int or memory_limit_bytes < 0
    ):
        raise ValueError("memory_limit_bytes must be a nonnegative integer")
    available = psutil.virtual_memory().available
    device = mx.device_info().get("max_recommended_working_set_size", available)
    capacity = min(available, max(0, device - mx.get_active_memory()))
    if memory_limit_bytes is not None:
        capacity = min(capacity, memory_limit_bytes)
    # Reserve space for activations and preparation temporaries. This is a
    # conservative admission estimate, not a prediction of generation peak RAM.
    required = provenance["weight_file_bytes"]
    if required > capacity * 0.60:
        raise ValueError("Local checkpoint exceeds the preparation memory allowance")
    return {
        "available_capacity_bytes": capacity,
        "checkpoint_file_bytes": required,
        "model_fraction_limit": 0.60,
        "memory_limit_bytes": memory_limit_bytes,
    }


def _write_report(path, data, *, before_publish=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
        # No callback may throw after publication.
        if before_publish is not None:
            before_publish()
        # Exclusive atomic publication also rejects a concurrently created file.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def calibrate_diffusion(
    model, output, tasks, *, max_rows=256, observer=None, memory_limit_bytes=None
):
    """Generate local text-to-image tasks and export per-linear input energy."""
    from omlx_runtime.diffusion.quantization import ActivationCollector

    checkpoint = _source(model)
    destination = _destination(output, checkpoint.path)
    spec = get_pipeline(checkpoint.base_model)
    if not tasks:
        raise ValueError("At least one calibration task is required")
    resolved_tasks = [validate_task(spec, task) for task in tasks]
    progress = _Progress(observer)
    progress("loading", 0.0, "Inspecting local checkpoint")
    expected_calls = sum(task["num_inference_steps"] for task in resolved_tasks)
    collector = ActivationCollector(
        max_rows,
        expected_calls=expected_calls,
        observer=lambda phase, fraction, detail: progress(
            phase, 0.15 + 0.65 * fraction, detail
        ),
    )
    provenance = _provenance(checkpoint)
    admission = (
        _admit(provenance)
        if memory_limit_bytes is None
        else _admit(provenance, memory_limit_bytes=memory_limit_bytes)
    )
    backend = MFluxBackend()
    native = None
    started = time.monotonic()
    try:
        with _offline():
            progress("loading", 0.05, "Loading native checkpoint")
            native = backend.load(checkpoint, spec.id)
            progress("loading", 0.15, "Native checkpoint loaded")
            images = []
            with collector.capture(native), backend.calibration_context(native):
                for index, task in enumerate(tasks):
                    progress(
                        "calibrating",
                        0.15 + 0.65 * index / len(tasks),
                        f"Calibration task {index + 1}/{len(tasks)}",
                    )
                    result = backend.generate(native, task, spec.id)
                    progress(
                        "calibrating",
                        0.15 + 0.65 * (index + 1) / len(tasks),
                        f"Calibration task {index + 1} finished",
                    )
                    images.append(
                        {
                            "size": [result.image.width, result.image.height],
                            "pixel_sha256": hashlib.sha256(
                                result.image.tobytes()
                            ).hexdigest(),
                        }
                    )
            report = collector.report()
        progress("saving", 0.85, "Preparing calibration report")
        report.update(
            {
                "base_model": checkpoint.base_model,
                "pipeline_id": spec.id,
                "source": provenance,
                "admission": admission,
                "tasks": [asdict(task) for task in tasks],
                "resolved_tasks": resolved_tasks,
                "outputs": images,
                "prediction_execution": "eager native denoising",
                "seconds": time.monotonic() - started,
                "versions": {name: version(name) for name in ("mlx", "mflux")},
            }
        )
        progress("saving", 0.95, "Writing calibration report")
        _write_report(
            destination,
            report,
            before_publish=lambda: progress(
                "saving", 0.99, "Publishing calibration report"
            ),
        )
        return report
    finally:
        if native is not None:
            backend.release(native)
        del native
        gc.collect()
        import mlx.core as mx

        mx.synchronize()
        mx.clear_cache()


def quantize_diffusion(
    model,
    calibration,
    output,
    *,
    bits=4,
    group_size=64,
    budget_bytes=None,
    budget_ratio=1.10,
    protected=(),
    observer=None,
    memory_limit_bytes=None,
):
    """Quantize a floating-point transformer and atomically publish native assets."""
    from omlx_runtime.diffusion.quantization import quantize_transformer
    from omlx_runtime.mflux_conversion import _copy_source_metadata

    checkpoint = _source(model)
    destination = _destination(output, checkpoint.path, directory=True)
    if read_quantization(checkpoint.path) is not None:
        raise ValueError(
            "Fresh quantization requires floating-point source weights; packed checkpoints are calibration-only"
        )
    report_path = Path(calibration).expanduser()
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    if (
        not isinstance(report, dict)
        or report.get("base_model") != checkpoint.base_model
    ):
        raise ValueError("Calibration and source checkpoint identities disagree")
    progress = _Progress(observer)
    progress("loading", 0.0, "Inspecting local checkpoint")
    provenance = _provenance(checkpoint)
    admission = (
        _admit(provenance)
        if memory_limit_bytes is None
        else _admit(provenance, memory_limit_bytes=memory_limit_bytes)
    )
    spec = get_pipeline(checkpoint.base_model)
    backend = MFluxBackend()
    native = None
    staging = None
    originals = None
    original_bits = None
    published = False
    try:
        with _offline():
            progress("loading", 0.05, "Loading native checkpoint")
            native = backend.load(checkpoint, spec.id)
            progress("loading", 0.15, "Native checkpoint loaded")
            import mlx.nn as nn
            from mlx.utils import tree_unflatten

            originals = [
                (name, module)
                for name, module in native.transformer.named_modules()
                if name and isinstance(module, (nn.Linear, nn.QuantizedLinear))
            ]
            original_bits = native.bits
            plan = quantize_transformer(
                native.transformer,
                report,
                bits=bits,
                group_size=group_size,
                budget_bytes=budget_bytes,
                budget_ratio=budget_ratio,
                protected=protected,
                observer=lambda phase, fraction, detail: progress(
                    phase, 0.15 + 0.60 * fraction, detail
                ),
            )
            progress("quantizing", 0.75, "Transformer quantization finished")
            native.bits = bits  # activates native mixed-layer reconstruction
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(
                tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
            )
            progress("saving", 0.80, "Preparing checkpoint assets")
            _copy_source_metadata(checkpoint.path, staging)
            progress("saving", 0.85, "Saving native checkpoint")
            manifest = backend.save(native, staging, pipeline_id=spec.id)
            progress("saving", 0.90, "Native checkpoint saved")
        plan.update(
            {
                "base_model": checkpoint.base_model,
                "source": provenance,
                "calibration_source": report.get("source"),
                "calibration_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                "calibration_tasks": report.get("tasks"),
                "admission": admission,
                "component_policy": "quantize transformer linears; preserve VAE and text encoder",
                "versions": {name: version(name) for name in ("mlx", "mflux")},
            }
        )
        (staging / "diffusion-quantization.json").write_text(
            json.dumps(plan, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        manifest["quantization_policy"] = {
            "method": "diffusion-oQe",
            "bits": bits,
            "scope": "transformer only; per-layer bits in diffusion-quantization.json",
            "report": "diffusion-quantization.json",
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        detect_checkpoint(staging)
        progress("saving", 0.99, "Publishing complete checkpoint")
        if destination.exists():
            if not destination.is_dir() or any(destination.iterdir()):
                raise ValueError("Output is no longer empty")
            destination.rmdir()
        staging.rename(destination)
        published = True
        return plan
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        if native is not None:
            if originals is not None and not published:
                native.transformer.update_modules(tree_unflatten(originals))
                native.bits = original_bits
            backend.release(native)
        # Drop restoration snapshots before clearing the allocator cache.
        originals = None
        del native
        gc.collect()
        import mlx.core as mx

        mx.synchronize()
        mx.clear_cache()


def cli_command(args):
    """Called by the main CLI without importing MLX during parser setup."""
    if args.command == "diffusion-calibrate":
        if not 0 <= args.seed < 2**32:
            raise ValueError("seed must be between 0 and 4294967295")
        tasks = [
            ImageTask(
                prompt=prompt,
                seed=(args.seed + i) % (2**32),
                width=args.width,
                height=args.height,
                steps=args.steps,
                guidance=args.guidance,
                negative_prompt=args.negative_prompt,
            )
            for i, prompt in enumerate(args.prompt)
        ]
        return calibrate_diffusion(
            args.model, args.output, tasks, max_rows=args.max_rows
        )
    return quantize_diffusion(
        args.model,
        args.calibration,
        args.output,
        bits=args.bits,
        group_size=args.group_size,
        budget_bytes=args.budget_bytes,
        budget_ratio=args.budget_ratio,
        protected=tuple(args.protect),
    )
