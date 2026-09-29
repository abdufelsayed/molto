# SPDX-License-Identifier: Apache-2.0
"""mflux-backed image generation engine."""

from __future__ import annotations

import asyncio
import gc
import io
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..engine_core import get_mlx_executor
from ..model_discovery import read_mflux_manifest
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)


class MFluxImageEngine(BaseNonStreamingEngine):
    """Serve a local Z-Image Turbo checkpoint through mflux."""

    def __init__(self, model_name: str):
        super().__init__()
        self._model_name = model_name
        self._model: Any | None = None
        self._generation_lock = asyncio.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    async def start(self) -> None:
        if self._model is not None:
            return

        try:
            from mflux.models.common.config import ModelConfig
            from mflux.models.z_image import ZImageTurbo
        except ImportError as exc:
            raise ImportError(
                "mflux is required for image generation. "
                'Install it with: pip install "omlx[image]"'
            ) from exc

        model_path = Path(self._model_name)
        manifest = read_mflux_manifest(model_path) or {}
        quantize = manifest.get("quantize")
        if quantize is not None and quantize not in {3, 4, 5, 6, 8}:
            raise ValueError(
                f"Invalid mflux quantization level {quantize!r} in "
                f"{model_path / 'omlx-mflux.json'}"
            )

        def _load_sync():
            return ZImageTurbo(
                model_config=ModelConfig.z_image_turbo(),
                model_path=str(model_path),
                quantize=quantize,
            )

        logger.info("Starting mflux image engine: %s", self._model_name)
        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(get_mlx_executor(), _load_sync)
        logger.info("Started mflux image engine: %s", self._model_name)

    async def stop(self) -> None:
        if self._model is None:
            return
        logger.info("Stopping mflux image engine: %s", self._model_name)
        self._model = None
        gc.collect()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            get_mlx_executor(), lambda: (mx.synchronize(), mx.clear_cache())
        )
        logger.info("Stopped mflux image engine: %s", self._model_name)

    async def generate_image(
        self,
        *,
        prompt: str,
        seed: int,
        width: int,
        height: int,
        steps: int,
        guidance: float | None = None,
        negative_prompt: str | None = None,
    ) -> bytes:
        if self._model is None:
            raise RuntimeError("Engine not started. Call start() first.")

        async with self._generation_lock:
            model = self._model

            def _generate_sync() -> bytes:
                generated = model.generate_image(
                    seed=seed,
                    prompt=prompt,
                    num_inference_steps=steps,
                    width=width,
                    height=height,
                    guidance=guidance,
                    negative_prompt=negative_prompt,
                )
                image = getattr(generated, "image", generated)
                output = io.BytesIO()
                image.save(output, format="PNG")
                return output.getvalue()

            activity_id = self._begin_activity(
                "generating image",
                detail="Generating image",
                total_items=steps,
                metadata={
                    "width": width,
                    "height": height,
                    "seed": seed,
                    "steps": steps,
                },
            )
            try:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(get_mlx_executor(), _generate_sync)
            finally:
                await self._finish_activity(activity_id)

    def get_stats(self) -> dict[str, Any]:
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None,
            "backend": "mflux",
            "model_family": "z-image-turbo",
        }

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return f"<MFluxImageEngine model={self._model_name} status={status}>"
