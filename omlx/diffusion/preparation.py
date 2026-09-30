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

from .backend import MFluxBackend, _offline
from .checkpoint import MANIFEST_NAME, detect_checkpoint, read_quantization
from .registry import ImageTask, get_pipeline, validate_task

# Explicit first adapters. Adding a family requires checking its eager forward,
# native saver, and component policies. No inference-server conditionals.
QUANTIZATION_MODELS = ("flux2-klein-4b", "qwen-image-2.1")


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


def _admit(provenance):
    import mlx.core as mx
    import psutil

    available = psutil.virtual_memory().available
    device = mx.device_info().get("max_recommended_working_set_size", available)
    capacity = min(available, max(0, device - mx.get_active_memory()))
    # Reserve space for activations and preparation temporaries. This is a
    # conservative admission estimate, not a prediction of generation peak RAM.
    required = provenance["weight_file_bytes"]
    if required > capacity * 0.60:
        raise ValueError("Local checkpoint exceeds the preparation memory allowance")
    return {
        "available_capacity_bytes": capacity,
        "checkpoint_file_bytes": required,
        "model_fraction_limit": 0.60,
    }


def _write_report(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
        # Exclusive atomic publication also rejects a concurrently created file.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def calibrate_diffusion(model, output, tasks, *, max_rows=256):
    """Generate local text-to-image tasks and export per-linear input energy."""
    from .quantization import ActivationCollector

    checkpoint = _source(model)
    destination = _destination(output, checkpoint.path)
    spec = get_pipeline(checkpoint.base_model)
    if not tasks:
        raise ValueError("At least one calibration task is required")
    resolved_tasks = [validate_task(spec, task) for task in tasks]
    collector = ActivationCollector(max_rows)
    provenance = _provenance(checkpoint)
    admission = _admit(provenance)
    backend = MFluxBackend()
    native = None
    started = time.monotonic()
    try:
        with _offline():
            native = backend.load(checkpoint, spec.id)
            images = []
            with collector.capture(native), backend.calibration_context(native):
                for task in tasks:
                    result = backend.generate(native, task, spec.id)
                    images.append(
                        {
                            "size": [result.image.width, result.image.height],
                            "pixel_sha256": hashlib.sha256(
                                result.image.tobytes()
                            ).hexdigest(),
                        }
                    )
            report = collector.report()
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
        _write_report(destination, report)
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
):
    """Quantize a floating-point transformer and atomically publish native assets."""
    from omlx.mflux_conversion import _copy_source_metadata

    from .quantization import quantize_transformer

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
    provenance = _provenance(checkpoint)
    admission = _admit(provenance)
    spec = get_pipeline(checkpoint.base_model)
    backend = MFluxBackend()
    native = None
    staging = None
    try:
        with _offline():
            native = backend.load(checkpoint, spec.id)
            plan = quantize_transformer(
                native.transformer,
                report,
                bits=bits,
                group_size=group_size,
                budget_bytes=budget_bytes,
                budget_ratio=budget_ratio,
                protected=protected,
            )
            native.bits = bits  # activates native mixed-layer reconstruction
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(
                tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
            )
            _copy_source_metadata(checkpoint.path, staging)
            manifest = backend.save(native, staging, pipeline_id=spec.id)
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
        if destination.exists():
            if not destination.is_dir() or any(destination.iterdir()):
                raise ValueError("Output is no longer empty")
            destination.rmdir()
        staging.rename(destination)
        return plan
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        if native is not None:
            backend.release(native)
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
