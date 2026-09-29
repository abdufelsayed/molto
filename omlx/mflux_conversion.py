# SPDX-License-Identifier: Apache-2.0
"""Shared mflux conversion used by the CLI and web operation queue."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any


def convert_mflux_model(
    source: str,
    output: str | Path,
    quantize: int | None = 8,
    *,
    on_stage: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Load, optionally quantize, and save one supported diffusion model."""
    if quantize not in {None, 3, 4, 5, 6, 8}:
        raise ValueError("mflux quantization must be 3, 4, 5, 6, 8, or null")
    try:
        from mflux.models.common.config import ModelConfig
        from mflux.models.z_image import ZImageTurbo
    except ImportError as exc:
        raise ImportError(
            'mflux is not installed. Install it with: pip install "omlx[image]"'
        ) from exc

    destination = Path(output).expanduser().resolve()
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(f"Output path is not a directory: {destination}")
        if any(destination.iterdir()):
            raise ValueError(f"Output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    if on_stage:
        on_stage("loading and quantizing" if quantize is not None else "converting")
    model_path = None if source == "z-image-turbo" else source
    model = ZImageTurbo(
        model_config=ModelConfig.z_image_turbo(),
        model_path=model_path,
        quantize=quantize,
    )
    saved_bits = getattr(model, "bits", None)
    if on_stage:
        on_stage("saving checkpoint")
    model.save_model(str(destination))
    manifest = {
        "version": 1,
        "backend": "mflux",
        "model_family": "z-image-turbo",
        "source": source,
        # Saved weights already contain their quantization metadata.
        "quantize": None,
        "converted_quantization_bits": saved_bits,
    }
    manifest_path = destination / "omlx-mflux.json"
    temporary = manifest_path.with_name(f"{manifest_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return {
        "source": source,
        "output": str(destination),
        "quantize": quantize,
        "manifest": manifest,
    }
