# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible local image generation endpoint."""

from __future__ import annotations

import base64
import secrets
import time

from fastapi import APIRouter, HTTPException

from ..engine.image_generation import MFluxImageEngine
from ..exceptions import (
    EnginePoolError,
    InsufficientMemoryError,
    ModelBusyError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelTooLargeError,
    ModelUnavailableError,
)
from ..server_metrics import get_server_metrics
from .image_models import ImageData, ImageGenerationRequest, ImageGenerationResponse

router = APIRouter()

_MIN_IMAGE_SIDE = 256
_MAX_IMAGE_SIDE = 2048
_IMAGE_SIDE_MULTIPLE = 16


def _get_engine_pool():
    from omlx.server import _server_state

    pool = _server_state.engine_pool
    if pool is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return pool


def _resolve_model(model_id: str) -> str:
    from omlx.server import resolve_model_id

    return resolve_model_id(model_id) or model_id


def _parse_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except (AttributeError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="'size' must use WIDTHxHEIGHT, for example '1024x1024'",
        ) from exc
    if not (
        _MIN_IMAGE_SIDE <= width <= _MAX_IMAGE_SIDE
        and _MIN_IMAGE_SIDE <= height <= _MAX_IMAGE_SIDE
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image dimensions must be between {_MIN_IMAGE_SIDE} and "
                f"{_MAX_IMAGE_SIDE} pixels"
            ),
        )
    if width % _IMAGE_SIDE_MULTIPLE or height % _IMAGE_SIDE_MULTIPLE:
        raise HTTPException(
            status_code=400,
            detail=f"Image dimensions must be multiples of {_IMAGE_SIDE_MULTIPLE}",
        )
    return width, height


@router.post("/v1/images/generations", response_model=ImageGenerationResponse)
async def create_image(request: ImageGenerationRequest) -> ImageGenerationResponse:
    """Generate one or more PNG images using a managed mflux engine."""
    started_at = time.monotonic()
    width, height = _parse_size(request.size)
    model_id = _resolve_model(request.model)
    pool = _get_engine_pool()
    base_seed = request.seed if request.seed is not None else secrets.randbits(32)

    try:
        async with pool.acquire(model_id) as engine:
            if not isinstance(engine, MFluxImageEngine):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Model '{model_id}' is not an image-generation model. "
                        "Use a discovered mflux Z-Image Turbo checkpoint."
                    ),
                )

            data: list[ImageData] = []
            for index in range(request.n):
                seed = (base_seed + index) % (2**32)
                png = await engine.generate_image(
                    prompt=request.prompt,
                    seed=seed,
                    width=width,
                    height=height,
                    steps=request.steps,
                    guidance=request.guidance,
                    negative_prompt=request.negative_prompt,
                )
                data.append(
                    ImageData(
                        b64_json=base64.b64encode(png).decode("ascii"),
                        seed=seed,
                    )
                )
    except HTTPException:
        raise
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ModelLoadingError, ModelBusyError, ModelUnavailableError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ModelTooLargeError, InsufficientMemoryError) as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc
    except EnginePoolError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    get_server_metrics().record_request_complete(
        prompt_tokens=0,
        completion_tokens=0,
        cached_tokens=0,
        generation_duration=time.monotonic() - started_at,
        model_id=model_id,
    )
    return ImageGenerationResponse(created=int(time.time()), data=data)
