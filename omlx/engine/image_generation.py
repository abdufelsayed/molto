# SPDX-License-Identifier: Apache-2.0
"""Managed diffusion pipelines on oMLX's shared Metal executor."""

from __future__ import annotations

import asyncio
import gc
import io
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..diffusion import (
    ImageTask,
    MFluxBackend,
    detect_checkpoint,
    get_pipeline,
    validate_task,
)
from ..engine_core import get_mlx_executor
from .base import BaseNonStreamingEngine

logger = logging.getLogger(__name__)


async def _await_completed(future, on_result=None):
    """Defer cancellation until work has actually stopped touching its resources."""
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(future)
            break
        except asyncio.CancelledError:
            cancelled = True
            if future.cancelled():
                raise
        except BaseException:
            if cancelled:
                raise asyncio.CancelledError() from None
            raise
    if on_result is not None:
        on_result(result)
    if cancelled:
        raise asyncio.CancelledError()
    return result


class DiffusionImageEngine(BaseNonStreamingEngine):
    """Load and run a detected checkpoint using its declared pipeline contract."""

    def __init__(self, model_name: str, initial_pipeline: str | None = None):
        super().__init__()
        self._model_name = model_name
        self._initial_pipeline = initial_pipeline
        self._model: Any | None = None
        self._started = False
        self._checkpoint = None
        self._pipeline = None
        self._backend = MFluxBackend()
        # Loading, generation, switching and unloading all share one lifecycle lock.
        self._generation_lock = asyncio.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    def _resolve_pipeline(self, pipeline: str | None = None):
        if self._checkpoint is None:
            self._checkpoint = detect_checkpoint(Path(self._model_name))
        if self._checkpoint is None:
            raise ValueError(
                f"Not a supported diffusion checkpoint: {self._model_name}"
            )
        spec = get_pipeline(pipeline or self._checkpoint.default_pipeline)
        if spec.local_unsupported_reason:
            raise ValueError(spec.local_unsupported_reason)
        if spec.base_model != self._checkpoint.base_model:
            raise ValueError(
                f"Pipeline '{spec.id}' is incompatible with this checkpoint"
            )
        return spec

    async def _executor_call(self, function, on_result=None):
        future = asyncio.get_running_loop().run_in_executor(
            get_mlx_executor(), function
        )
        return await _await_completed(future, on_result)

    def _release_sync(self):
        model = self._model
        self._model = None
        self._pipeline = None
        if model is not None:
            self._backend.release(model)
        del model
        gc.collect()
        mx.synchronize()
        mx.clear_cache()

    async def _load_locked(self, spec):
        if self._model is not None and self._pipeline.id == spec.id:
            return
        if self._model is not None:
            await self._executor_call(self._release_sync)

        def store_model(model):
            self._model = model
            self._pipeline = spec

        try:
            await self._executor_call(
                lambda: self._backend.load(self._checkpoint, spec.id), store_model
            )
        except asyncio.CancelledError:
            # The executor may have completed loading after cancellation arrived.
            # Drain and dispose of that model before permitting another lifecycle action.
            if self._model is not None:
                await self._executor_call(self._release_sync)
            raise

    async def start(self) -> None:
        async with self._generation_lock:
            if self._model is None:
                await self._load_locked(self._resolve_pipeline(self._initial_pipeline))
            self._started = True

    async def stop(self) -> None:
        async with self._generation_lock:
            self._started = False
            if self._model is not None:
                await self._executor_call(self._release_sync)

    async def generate_image(
        self,
        *,
        prompt: str | None,
        seed: int,
        width: int | None,
        height: int | None,
        steps: int | None = None,
        guidance: float | None = None,
        negative_prompt: str | None = None,
        pipeline: str | None = None,
        image_paths: tuple[str, ...] = (),
        mask_path: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> bytes:
        spec = self._resolve_pipeline(pipeline)
        task = ImageTask(
            prompt=prompt,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            negative_prompt=negative_prompt,
            image_paths=image_paths,
            mask_path=mask_path,
            options=options or {},
        )
        validate_task(spec, task)
        async with self._generation_lock:
            if not self._started:
                raise RuntimeError("Engine not started. Call start() first.")
            await self._load_locked(spec)

            def generate_sync():
                generated = self._backend.generate(self._model, task, spec.id)
                image = getattr(generated, "image", generated)
                output = io.BytesIO()
                image.save(output, format="PNG")
                return output.getvalue()

            activity_id = self._begin_activity(
                "generating image",
                detail=f"Running {spec.id}",
                metadata={
                    "width": width,
                    "height": height,
                    "seed": seed,
                    "steps": steps or spec.default_steps,
                    "pipeline": spec.id,
                },
            )
            try:
                return await self._executor_call(generate_sync)
            finally:
                # Activity/cache cleanup must also remain within the lifecycle lock.
                await _await_completed(
                    asyncio.create_task(self._finish_activity(activity_id))
                )

    def get_stats(self) -> dict[str, Any]:
        checkpoint = self._checkpoint
        return {
            "model_name": self._model_name,
            "loaded": self._model is not None,
            "backend": "mflux",
            "model_family": checkpoint.base_model if checkpoint else None,
            "pipeline": self._pipeline.id if self._pipeline else None,
            "capabilities": self._pipeline.metadata() if self._pipeline else None,
        }

    def __repr__(self) -> str:
        status = "running" if self._model is not None else "stopped"
        return f"<DiffusionImageEngine model={self._model_name} status={status}>"


# Existing integrations and pool imports retain their original name.
MFluxImageEngine = DiffusionImageEngine
