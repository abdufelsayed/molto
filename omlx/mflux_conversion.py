# SPDX-License-Identifier: Apache-2.0
"""mflux checkpoint acquisition and conversion for CLI model preparation."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _copy_source_metadata(source: Path, destination: Path) -> None:
    """Retain configuration and tokenizer metadata without replacing saved indexes."""
    if not source.is_dir():
        return
    for original in source.rglob("*"):
        if not original.is_file():
            continue
        relative = original.relative_to(source)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if original.name == "omlx-mflux.json" or original.name.endswith(
            ".safetensors.index.json"
        ):
            continue
        tokenizer = any(
            part.startswith(("tokenizer", "processor")) for part in relative.parts[:-1]
        )
        if original.suffix != ".json" and not tokenizer:
            continue
        # A tokenizer directory may contain cached weights; never duplicate them.
        if original.suffix in {".safetensors", ".bin", ".pt", ".pth"}:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)


def convert_mflux_model(
    source: str,
    output: str | Path,
    quantize: int | None = 8,
    *,
    base_model: str | None = None,
    pipeline: str | None = None,
    revision: str | None = None,
    on_stage: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Prepare a registry-selected model and publish only a complete checkpoint."""
    if quantize not in {None, 3, 4, 5, 6, 8}:
        raise ValueError("mflux quantization must be 3, 4, 5, 6, 8, or null")
    from .diffusion import (
        MFluxBackend,
        detect_checkpoint,
        get_pipeline,
        read_quantization,
        resolve_base_model,
    )

    source_path = Path(source).expanduser()
    checkpoint = detect_checkpoint(source_path) if source_path.is_dir() else None
    identity = (
        resolve_base_model(base_model)
        if base_model
        else (checkpoint.base_model if checkpoint else resolve_base_model(source))
    )
    if checkpoint and identity != checkpoint.base_model:
        raise ValueError(
            "The explicit base model conflicts with the checkpoint identity"
        )
    if base_model and not source_path.is_dir():
        try:
            source_identity = resolve_base_model(source)
        except ValueError:
            source_identity = None
        if source_identity is not None and source_identity != identity:
            raise ValueError(
                "The explicit base model conflicts with the source identity"
            )
    selected_pipeline = get_pipeline(
        pipeline or (checkpoint.default_pipeline if checkpoint else identity)
    )
    if selected_pipeline.base_model != identity:
        raise ValueError("The selected pipeline does not match the base model")
    if not selected_pipeline.save_supported:
        raise ValueError(selected_pipeline.save_unsupported_reason)
    if quantize is not None and quantize not in selected_pipeline.quantization_bits:
        raise ValueError(
            f"Pipeline {selected_pipeline.id} does not support {quantize}-bit quantization"
        )
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(f"Output path is not a directory: {destination}")
        if any(destination.iterdir()):
            raise ValueError(f"Output directory is not empty: {destination}")
    repository = None
    if not source_path.is_dir():
        from huggingface_hub import snapshot_download

        from .diffusion import download_patterns
        from .diffusion.registry import MODEL_IDENTITIES

        # Resolve built-in names exactly; third-party repositories require the
        # explicit identity already validated above. Never infer from substrings.
        repository = MODEL_IDENTITIES[identity][1] if not base_model else source
        try:
            if resolve_base_model(source) == identity:
                repository = MODEL_IDENTITIES[identity][1]
        except ValueError:
            pass
        if on_stage:
            on_stage("downloading checkpoint")
        source_path = Path(
            snapshot_download(
                repo_id=repository,
                revision=revision,
                allow_patterns=download_patterns(identity),
            )
        )
    acquired_checkpoint = detect_checkpoint(source_path)
    if acquired_checkpoint is not None and acquired_checkpoint.base_model != identity:
        raise ValueError("Downloaded checkpoint conflicts with the selected base model")
    if acquired_checkpoint is not None and pipeline is None:
        selected_pipeline = get_pipeline(acquired_checkpoint.default_pipeline)
    stored_bits = read_quantization(source_path)
    if stored_bits is None and acquired_checkpoint is not None:
        stored_bits = acquired_checkpoint.quantization
    if stored_bits is not None and quantize is not None and stored_bits != quantize:
        raise ValueError(
            f"Checkpoint is already quantized at {stored_bits} bits; requested {quantize} bits conflicts"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The hidden sibling is not discovered during saving and is removed on failure.
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        backend = MFluxBackend()
        if on_stage:
            on_stage("loading and quantizing" if quantize is not None else "converting")
        # Acquisition belongs to preparation; the backend receives only a local
        # snapshot and must reject every missing-weight remote fallback.
        model_path = str(source_path.resolve())
        model = backend.instantiate(
            identity,
            model_path=model_path,
            quantization=quantize,
            pipeline_id=selected_pipeline.id,
        )
        actual_bits = getattr(model, "bits", None)
        if quantize is not None and actual_bits is not None and actual_bits != quantize:
            raise ValueError(
                f"mflux loaded {actual_bits}-bit weights instead of requested {quantize}-bit weights"
            )
        if on_stage:
            on_stage("saving checkpoint")
        _copy_source_metadata(source_path, staging)
        backend.save(
            model, staging, pipeline_id=selected_pipeline.id, base_model=identity
        )
        prepared = detect_checkpoint(staging)
        if prepared is None:
            raise ValueError(
                "Saved output is not a complete supported image checkpoint"
            )
        manifest_path = staging / "omlx-mflux.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "source": source,
                "source_path": str(source_path.resolve()),
                "source_repo_id": repository,
                "source_revision": revision,
                "resolved_revision": source_path.name if repository else None,
                "requested_quantization_bits": quantize,
            }
        )
        pending_manifest = manifest_path.with_name(
            f".{manifest_path.name}.{os.getpid()}.tmp"
        )
        pending_manifest.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        pending_manifest.replace(manifest_path)
        detect_checkpoint(staging)
        # Recheck immediately before publishing so a concurrently populated output
        # cannot be replaced by this operation.
        if destination.exists():
            if not destination.is_dir() or any(destination.iterdir()):
                raise ValueError(f"Output directory is not empty: {destination}")
            destination.rmdir()
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "source": source,
        "output": str(destination),
        "quantize": quantize,
        "manifest": manifest,
    }
