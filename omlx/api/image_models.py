# SPDX-License-Identifier: Apache-2.0
"""Strict request models for local image pipelines."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Base64 for an 8 MiB upload, including a small data URI header.
MAX_IMAGE_BASE64 = 11_185_000


class ImageGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=32_000)
    model: str = Field(min_length=1)
    pipeline: str | None = None
    n: int = Field(default=1, ge=1, le=4)
    batch_size: int = Field(default=1, ge=1, le=4)
    size: str = "1024x1024"
    response_format: Literal["b64_json"] = "b64_json"
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    steps: int | None = Field(default=None, ge=1, le=100)
    guidance: float | None = Field(default=None, ge=0, le=30, allow_inf_nan=False)
    negative_prompt: str | None = Field(default=None, max_length=32_000)
    options: dict[str, Any] = Field(default_factory=dict)


ImageBase64 = Annotated[str, Field(max_length=MAX_IMAGE_BASE64)]


class ImageOperationRequest(ImageGenerationRequest):
    prompt: str | None = Field(default=None, min_length=1, max_length=32_000)
    size: str | None = None
    operation: Literal[
        "txt2img", "img2img", "reference-edit", "inpaint", "controlnet", "upscale"
    ]
    images: list[ImageBase64] = Field(default_factory=list, max_length=8)
    mask: str | None = Field(default=None, max_length=MAX_IMAGE_BASE64)


class ImageEditRequest(ImageGenerationRequest):
    size: str | None = None
    image: ImageBase64 | Annotated[list[ImageBase64], Field(min_length=1, max_length=8)]
    mask: str | None = Field(default=None, max_length=MAX_IMAGE_BASE64)


class ImageData(BaseModel):
    b64_json: str
    revised_prompt: str | None = None
    seed: int | None = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[ImageData]
