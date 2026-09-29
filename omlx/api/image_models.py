# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible image generation request and response models."""

from typing import Literal

from pydantic import BaseModel, Field


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=32_000)
    model: str
    n: int = Field(default=1, ge=1, le=4)
    size: str = "1024x1024"
    response_format: Literal["b64_json"] = "b64_json"
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
    steps: int = Field(default=9, ge=1, le=100)
    guidance: float | None = Field(default=None, ge=0, le=30)
    negative_prompt: str | None = Field(default=None, max_length=32_000)


class ImageData(BaseModel):
    b64_json: str
    revised_prompt: str | None = None
    seed: int | None = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[ImageData]
