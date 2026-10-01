"""Authenticated management acquisition and durable operation endpoints."""

from typing import Literal

from fastapi import APIRouter, Depends, Query
from molto_management.management_acquisition import AcquisitionService
from pydantic import BaseModel, ConfigDict, Field

from molto_server.api.management_dependencies import get_runtime

router = APIRouter(tags=["management-acquisition"])


def service(runtime=Depends(get_runtime)):
    return AcquisitionService(runtime)


class StrictBody(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DownloadBody(StrictBody):
    repo_id: str = Field(min_length=3, max_length=256)
    token: str = Field(default="", max_length=4096, repr=False)


class ConvertBody(StrictBody):
    model_path: str
    output_name: str | None = None


class EstimateBody(StrictBody):
    model_path: str
    oq_level: float
    group_size: Literal[32, 64, 128] = 64
    preserve_mtp: bool = False


class QuantizeBody(EstimateBody):
    sensitivity_model_path: str = ""
    text_only: bool = False
    dtype: Literal["bfloat16", "float16"] = "bfloat16"
    auto_proxy_sensitivity: bool = True
    enhanced: bool = False
    imatrix_cache_path: str = ""
    imatrix_reuse_cache: bool = True
    imatrix_strict: bool = False
    imatrix_num_samples: int = Field(default=128, ge=1, le=4096)
    imatrix_seq_length: int = Field(default=512, ge=1, le=32768)
    mtp_assistant_model_path: str = ""


class PublishValidationBody(StrictBody):
    token: str = Field(min_length=1, max_length=4096, repr=False)
    model_path: str | None = None
    repo_id: str | None = None


class PublishBody(StrictBody):
    token: str = Field(min_length=1, max_length=4096, repr=False)
    model_path: str
    repo_id: str
    readme_source_path: str = ""
    auto_readme: bool = True
    redownload_notice: bool = False
    private: bool = False


class RetryBody(StrictBody):
    token: str = Field(default="", max_length=4096, repr=False)


@router.get("/acquisition/{provider}/search")
async def search(
    provider: Literal["hf", "ms"],
    query: str = Query(default="", max_length=256),
    sort: Literal["trending", "downloads", "created", "updated", "likes"] = "trending",
    limit: int = Query(default=100, ge=1, le=100),
    mlx_only: bool = True,
    min_params: int | None = Query(default=None, ge=0),
    max_params: int | None = Query(default=None, ge=0),
    min_size: int | None = Query(default=None, ge=0),
    max_size: int | None = Query(default=None, ge=0),
    sort_by_size: bool = False,
    sort_ascending: bool = False,
    svc=Depends(service),
):
    return await svc.search(
        provider,
        query=query,
        sort=sort,
        limit=limit,
        mlx_only=mlx_only,
        min_params=min_params,
        max_params=max_params,
        min_size=min_size,
        max_size=max_size,
        sort_by_size=sort_by_size,
        sort_ascending=sort_ascending,
    )


@router.get("/acquisition/{provider}/recommended")
async def recommended(
    provider: Literal["hf", "ms"],
    max_memory_bytes: int = Query(default=16 * 1024**3, gt=0),
    limit: int = Query(default=60, ge=1, le=100),
    result_limit: int = Query(default=50, ge=1, le=100),
    mlx_only: bool = True,
    svc=Depends(service),
):
    return await svc.recommended(
        provider,
        max_memory_bytes=max_memory_bytes,
        limit=limit,
        result_limit=result_limit,
        mlx_only=mlx_only,
    )


@router.get("/acquisition/{provider}/info")
async def info(provider: Literal["hf", "ms"], repo_id: str, svc=Depends(service)):
    return await svc.info(provider, repo_id)


@router.get("/acquisition/{provider}/downloads")
async def downloads(provider: Literal["hf", "ms"], svc=Depends(service)):
    return svc.operations(provider=provider)


@router.post("/acquisition/{provider}/downloads", status_code=202)
async def download(
    provider: Literal["hf", "ms"], body: DownloadBody, svc=Depends(service)
):
    return svc.view(await svc.download(provider, **body.model_dump()))


@router.get("/acquisition/prepare/models")
async def models(svc=Depends(service)):
    return await svc.models()


@router.get("/acquisition/prepare/options")
async def options(svc=Depends(service)):
    return svc.options()


@router.post("/acquisition/prepare/convert", status_code=202)
async def convert(body: ConvertBody, svc=Depends(service)):
    return svc.view(await svc.convert(**body.model_dump()))


@router.post("/acquisition/prepare/estimate")
async def estimate(body: EstimateBody, svc=Depends(service)):
    return await svc.estimate(**body.model_dump())


@router.post("/acquisition/prepare/quantize", status_code=202)
async def quantize(body: QuantizeBody, svc=Depends(service)):
    return svc.view(await svc.quantize(**body.model_dump()))


@router.post("/acquisition/publish/validate")
async def validate(body: PublishValidationBody, svc=Depends(service)):
    return await svc.validate_publish(**body.model_dump())


@router.post("/acquisition/publish/start", status_code=202)
async def publish(body: PublishBody, svc=Depends(service)):
    return svc.view(await svc.publish(**body.model_dump()))


@router.get("/operations")
async def operations(
    limit: int = Query(default=200, ge=1, le=1000), svc=Depends(service)
):
    return svc.operations(limit)


@router.get("/operations/{operation_id}")
async def operation(operation_id: str, svc=Depends(service)):
    return svc.get(operation_id)


@router.post("/operations/{operation_id}/cancel", status_code=202)
async def cancel(operation_id: str, svc=Depends(service)):
    return await svc.cancel(operation_id)


@router.post("/operations/{operation_id}/retry", status_code=202)
async def retry(operation_id: str, body: RetryBody, svc=Depends(service)):
    return await svc.retry(operation_id, body.token)


@router.delete("/operations/{operation_id}")
async def delete(operation_id: str, svc=Depends(service)):
    return svc.delete(operation_id)
