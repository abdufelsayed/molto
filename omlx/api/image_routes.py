# SPDX-License-Identifier: Apache-2.0
"""OpenAI image routes with checkpoint preflight and bounded local media."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import secrets
import tempfile
import time
import warnings
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from ..diffusion import (
    ImageTask,
    detect_checkpoint,
    get_pipeline,
    pipelines_for_model,
    validate_task,
)
from ..engine.image_generation import MFluxImageEngine, _await_completed
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
from .image_models import (
    ImageData,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageOperationRequest,
)

router = APIRouter()
_MIN_IMAGE_SIDE = 256
_MAX_IMAGE_SIDE = 2048
_IMAGE_SIDE_MULTIPLE = 16
_MAX_MEDIA_BYTES = 8 * 1024 * 1024
_MAX_BODY_BYTES = 48 * 1024 * 1024
_MAX_INPUT_PIXELS = 16 * 1024 * 1024


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
            detail=f"Image dimensions must be between {_MIN_IMAGE_SIDE} and {_MAX_IMAGE_SIDE} pixels",
        )
    if width % _IMAGE_SIDE_MULTIPLE or height % _IMAGE_SIDE_MULTIPLE:
        raise HTTPException(
            status_code=400,
            detail=f"Image dimensions must be multiples of {_IMAGE_SIDE_MULTIPLE}",
        )
    return width, height


def _decode_image(value: str) -> bytes:
    if value.startswith("data:"):
        header, separator, value = value.partition(",")
        if not separator or header not in {
            "data:image/png;base64",
            "data:image/jpeg;base64",
            "data:image/webp;base64",
        }:
            raise ValueError("Images must use PNG, JPEG or WebP base64 data")
    if len(value) > ((_MAX_MEDIA_BYTES + 2) // 3) * 4:
        raise ValueError("Each image must be at most 8 MiB")
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(
            "Images must be base64 data; URLs and local paths are not accepted"
        ) from exc
    if not data or len(data) > _MAX_MEDIA_BYTES:
        raise ValueError("Each image must contain between 1 byte and 8 MiB")
    return data


def _normalize_image(value: str, path: Path) -> str:
    data = _decode_image(value)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {"PNG", "JPEG", "WEBP"}:
                    raise ValueError("Only PNG, JPEG and WebP images are supported")
                if (
                    image.width * image.height > _MAX_INPUT_PIXELS
                    or max(image.size) > 8192
                ):
                    raise ValueError(
                        "Input images must have at most 16 megapixels and 8192 pixels per side"
                    )
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Animated images are not supported")
                image.load()
                ImageOps.exif_transpose(image).convert(
                    "RGBA"
                    if "A" in image.getbands() or "transparency" in image.info
                    else "RGB"
                ).save(path, format="PNG")
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError("Invalid or oversized image data") from exc
    return str(path)


def _preflight(pool, model_id, request, operation, image_paths, mask_path):
    entry = pool.get_entry(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    if entry.engine_type != "image_generation":
        raise ValueError(f"Model '{model_id}' is not an image-generation model")
    checkpoint = detect_checkpoint(Path(entry.model_path))
    if checkpoint is None:
        raise ValueError(f"Model '{model_id}' is not a supported diffusion checkpoint")
    spec = get_pipeline(request.pipeline or checkpoint.default_pipeline)
    if request.pipeline is None and operation != "txt2img":
        operations = (
            {"img2img", "reference-edit", "inpaint"}
            if operation == "edit"
            else {operation}
        )
        if spec.operation not in operations:
            candidates = [
                candidate
                for candidate in pipelines_for_model(checkpoint.base_model)
                if candidate.operation in operations
                and candidate.image_min <= len(image_paths) <= candidate.image_max
                and (mask_path is None or candidate.supports_mask)
            ]
            if len(candidates) != 1:
                raise ValueError("Select an explicit pipeline for this image operation")
            spec = candidates[0]
    if spec.local_unsupported_reason:
        raise ValueError(spec.local_unsupported_reason)
    if spec.base_model != checkpoint.base_model:
        raise ValueError(f"Pipeline '{spec.id}' is incompatible with this checkpoint")
    if operation == "edit":
        if spec.operation not in {"img2img", "reference-edit", "inpaint"}:
            raise ValueError("Select an image editing pipeline for /v1/images/edits")
    elif spec.operation != operation:
        raise ValueError(
            f"Pipeline '{spec.id}' performs '{spec.operation}', not '{operation}'"
        )
    width, height = (
        _parse_size(request.size) if request.size is not None else (None, None)
    )
    task = ImageTask(
        prompt=request.prompt,
        seed=request.seed or 0,
        width=width,
        height=height,
        steps=request.steps,
        guidance=request.guidance,
        negative_prompt=request.negative_prompt,
        image_paths=image_paths,
        mask_path=mask_path,
        options=request.options,
    )
    native_options = validate_task(spec, task)
    if spec.operation == "upscale":
        with Image.open(image_paths[0]) as input_image:
            scale = native_options["resolution"] / min(input_image.size)
            output_pixels = input_image.width * input_image.height * scale * scale
            output_side = max(input_image.size) * scale
            if output_pixels > _MAX_INPUT_PIXELS or output_side > 8192:
                raise ValueError(
                    "Upscaled output must have at most 16 megapixels and 8192 pixels per side"
                )
    return spec, width, height


async def _serve_images(request, operation="txt2img", images=(), mask=None):
    started_at = time.monotonic()
    model_id = _resolve_model(request.model)
    pool = _get_engine_pool()
    base_seed = request.seed if request.seed is not None else secrets.randbits(32)
    try:
        with tempfile.TemporaryDirectory(prefix="omlx-images-") as directory:
            image_paths = tuple(
                _normalize_image(value, Path(directory) / f"image-{index}.png")
                for index, value in enumerate(images)
            )
            mask_path = (
                _normalize_image(mask, Path(directory) / "mask.png")
                if mask is not None
                else None
            )
            spec, width, height = _preflight(
                pool, model_id, request, operation, image_paths, mask_path
            )
            async with pool.acquire(model_id) as engine:
                if not isinstance(engine, MFluxImageEngine):
                    raise ValueError(
                        f"Model '{model_id}' is not an image-generation model"
                    )
                data = []
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
                        pipeline=spec.id,
                        image_paths=image_paths,
                        mask_path=mask_path,
                        options=request.options,
                    )
                    data.append(
                        ImageData(
                            b64_json=base64.b64encode(png).decode("ascii"), seed=seed
                        )
                    )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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


@router.post("/v1/images/generations", response_model=ImageGenerationResponse)
async def create_image(request: ImageGenerationRequest) -> ImageGenerationResponse:
    return await _serve_images(request)


async def _bounded_body(request: Request):
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > _MAX_BODY_BYTES:
            raise HTTPException(
                status_code=413, detail="Image request must be at most 48 MiB"
            )
        body.extend(chunk)
    request._body = bytes(body)


async def _parse_media_form(request: Request, model):
    # A form context closes every spooled upload even after validation failure.
    async with request.form(
        max_files=9, max_fields=20, max_part_size=_MAX_MEDIA_BYTES
    ) as form:
        values = {}
        images = []
        for key, value in form.multi_items():
            if key in {"image", "image[]"}:
                if not isinstance(value, UploadFile):
                    raise ValueError("Multipart images must be uploaded files")
                data = await value.read(_MAX_MEDIA_BYTES + 1)
                if len(data) > _MAX_MEDIA_BYTES:
                    raise ValueError("Each image must be at most 8 MiB")
                images.append(base64.b64encode(data).decode("ascii"))
                continue
            if key in values:
                raise ValueError(f"Duplicate field '{key}'")
            if isinstance(value, UploadFile):
                if key != "mask":
                    raise ValueError(f"Unexpected file field '{key}'")
                data = await value.read(_MAX_MEDIA_BYTES + 1)
                if len(data) > _MAX_MEDIA_BYTES:
                    raise ValueError("Each image must be at most 8 MiB")
                value = base64.b64encode(data).decode("ascii")
            if key == "options":
                value = json.loads(value)
            values[key] = value
        values["image"] = images
        return model.model_validate(values)


async def _media_request(request: Request, model, *, multipart=False):
    await _bounded_body(request)
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    try:
        if content_type == "application/json":
            return model.model_validate(await request.json())
        if not multipart or content_type != "multipart/form-data":
            raise HTTPException(
                status_code=415,
                detail="Use application/json"
                + (" or multipart/form-data" if multipart else ""),
            )
        # Finish parsing and close all spooled uploads before propagating cancellation.
        return await _await_completed(
            asyncio.create_task(_parse_media_form(request, model))
        )
    except ValidationError as exc:
        # Do not echo submitted image data in validation errors.
        raise HTTPException(
            status_code=422,
            detail=exc.errors(include_input=False, include_context=False),
        ) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _image_body_schema(model, *, multipart=False):
    content = {"application/json": {"schema": model.model_json_schema()}}
    if multipart:
        properties = dict(model.model_json_schema()["properties"])
        properties["image"] = {
            "anyOf": [
                {"type": "string", "format": "binary"},
                {
                    "type": "array",
                    "items": {"type": "string", "format": "binary"},
                    "maxItems": 8,
                },
            ],
        }
        properties["mask"] = {"type": "string", "format": "binary"}
        properties["options"] = {
            "type": "string",
            "description": "A JSON object of pipeline options",
        }
        content["multipart/form-data"] = {
            "schema": {
                "type": "object",
                "properties": properties,
                "required": ["model", "prompt", "image"],
                "additionalProperties": False,
            }
        }
    return {"requestBody": {"required": True, "content": content}}


@router.post(
    "/v1/images/edits",
    response_model=ImageGenerationResponse,
    openapi_extra=_image_body_schema(ImageEditRequest, multipart=True),
)
async def edit_image(request: Request) -> ImageGenerationResponse:
    parsed = await _media_request(request, ImageEditRequest, multipart=True)
    images = [parsed.image] if isinstance(parsed.image, str) else parsed.image
    return await _serve_images(parsed, "edit", images, parsed.mask)


@router.post(
    "/v1/images/operations",
    response_model=ImageGenerationResponse,
    openapi_extra=_image_body_schema(ImageOperationRequest),
)
async def operate_image(request: Request) -> ImageGenerationResponse:
    parsed = await _media_request(request, ImageOperationRequest)
    return await _serve_images(parsed, parsed.operation, parsed.images, parsed.mask)


@router.get("/v1/images/capabilities")
async def image_capabilities(model: str | None = None):
    """Describe available operations without loading model weights."""
    from ..diffusion import list_pipelines

    if model is None:
        return {"pipelines": [spec.metadata() for spec in list_pipelines()]}
    model_id = _resolve_model(model)
    entry = _get_engine_pool().get_entry(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    try:
        if entry.engine_type != "image_generation":
            raise ValueError(f"Model '{model_id}' is not an image-generation model")
        checkpoint = detect_checkpoint(Path(entry.model_path))
        if checkpoint is None:
            raise ValueError("Not a supported diffusion checkpoint")
        return {
            "model": model_id,
            "default_pipeline": checkpoint.default_pipeline,
            "pipelines": [
                spec.metadata() for spec in pipelines_for_model(checkpoint.base_model)
            ],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
