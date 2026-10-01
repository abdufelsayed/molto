# SPDX-License-Identifier: Apache-2.0
"""HTTP mapping for engine management. Runtime work lives in services."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from molto_contracts.management import (
    CacheClearResponse,
    CacheResponse,
    DiffusionCalibrationRequest,
    DiffusionJobsResponse,
    DiffusionJobView,
    DiffusionQuantizationRequest,
    GlobalSettingsPatch,
    GlobalSettingsUpdateResponse,
    GlobalSettingsView,
    InventoryResponse,
    ModelOperationResponse,
    ModelSettingsPatch,
    ModelSettingsResponse,
    ModelSettingsUpdateResponse,
    ProfileDeleteResponse,
    ProfileResponse,
    ProfilesResponse,
    ProfileUpdate,
    ProfileWrite,
    RefreshResponse,
    StateResponse,
    StatsResponse,
)
from molto_management.management import ManagementContext, ManagementService

from molto_server.api.management_acquisition_routes import router as acquisition_router
from molto_server.api.management_diagnostics_routes import router as diagnostics_router
from molto_server.api.management_model_routes import router as model_options_router
from molto_server.api.management_monitoring_routes import router as monitoring_router
from molto_server.api.management_server_routes import router as server_settings_router
from molto_server.api.management_workspace_routes import router as workspace_router
from molto_server.auth import require_management_key

router = APIRouter(
    prefix="/management/v1",
    tags=["management"],
    dependencies=[Depends(require_management_key)],
)


def get_management_service(request: Request) -> ManagementService:
    provider: Callable[[], ManagementContext] | None = getattr(
        request.app.state, "management_context_provider", None
    )
    if provider is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return ManagementService(provider())


@router.get("/models", response_model=InventoryResponse)
async def list_models(service: ManagementService = Depends(get_management_service)):
    return service.inventory()


@router.get("/state", response_model=StateResponse)
async def get_state(service: ManagementService = Depends(get_management_service)):
    return service.state()


@router.post("/models/refresh", response_model=RefreshResponse)
async def refresh_models(service: ManagementService = Depends(get_management_service)):
    return await service.refresh()


@router.post("/models/{model_id:path}/load", response_model=ModelOperationResponse)
async def load_model(
    model_id: str, service: ManagementService = Depends(get_management_service)
):
    return await service.load(model_id)


@router.post("/models/{model_id:path}/unload", response_model=ModelOperationResponse)
async def unload_model(
    model_id: str,
    response: Response,
    service: ManagementService = Depends(get_management_service),
):
    result = await service.unload(model_id)
    response.status_code = 202 if result["status"] == "unloading" else 200
    return result


@router.get("/settings", response_model=GlobalSettingsView)
async def get_global_settings(
    service: ManagementService = Depends(get_management_service),
):
    return service.get_global_settings()


@router.patch("/settings", response_model=GlobalSettingsUpdateResponse)
async def update_global_settings(
    body: GlobalSettingsPatch,
    service: ManagementService = Depends(get_management_service),
):
    return await service.update_global_settings(body)


@router.get("/models/{model_id:path}/settings", response_model=ModelSettingsResponse)
async def get_model_settings(
    model_id: str, service: ManagementService = Depends(get_management_service)
):
    return service.get_model_settings(model_id)


@router.patch(
    "/models/{model_id:path}/settings", response_model=ModelSettingsUpdateResponse
)
async def update_model_settings(
    model_id: str,
    body: ModelSettingsPatch,
    service: ManagementService = Depends(get_management_service),
):
    return await service.update_model_settings(model_id, body)


@router.get("/models/{model_id:path}/profiles", response_model=ProfilesResponse)
async def list_profiles(
    model_id: str, service: ManagementService = Depends(get_management_service)
):
    return service.list_profiles(model_id)


@router.post("/models/{model_id:path}/profiles", response_model=ProfileResponse)
async def create_profile(
    model_id: str,
    body: ProfileWrite,
    service: ManagementService = Depends(get_management_service),
):
    return service.create_profile(model_id, body)


@router.put("/models/{model_id:path}/profiles/{name}", response_model=ProfileResponse)
async def update_profile(
    model_id: str,
    name: str,
    body: ProfileUpdate,
    service: ManagementService = Depends(get_management_service),
):
    return service.update_profile(model_id, name, body)


@router.delete(
    "/models/{model_id:path}/profiles/{name}", response_model=ProfileDeleteResponse
)
async def delete_profile(
    model_id: str,
    name: str,
    service: ManagementService = Depends(get_management_service),
):
    return service.delete_profile(model_id, name)


@router.post(
    "/models/{model_id:path}/profiles/{name}/apply",
    response_model=ModelSettingsUpdateResponse,
)
async def apply_profile(
    model_id: str,
    name: str,
    service: ManagementService = Depends(get_management_service),
):
    return await service.apply_profile(model_id, name)


@router.get("/stats", response_model=StatsResponse)
def get_stats(
    model_id: str = "",
    scope: str = Query(default="session", pattern="^(session|alltime)$"),
    service: ManagementService = Depends(get_management_service),
):
    return service.stats(model_id, scope)


@router.get("/cache", response_model=CacheResponse)
async def get_cache(service: ManagementService = Depends(get_management_service)):
    return service.cache()


@router.post("/cache/{kind}/clear", response_model=CacheClearResponse)
async def clear_cache(
    kind: str, service: ManagementService = Depends(get_management_service)
):
    return await service.clear_cache(kind)


def get_diffusion_jobs(request: Request):
    provider = getattr(request.app.state, "diffusion_jobs_provider", None)
    if provider is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    jobs = provider()
    if jobs is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return jobs


def _diffusion_job_call(function, *args):
    try:
        return function(*args)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/diffusion/calibrations", response_model=DiffusionJobView, status_code=202
)
async def start_diffusion_calibration(
    body: DiffusionCalibrationRequest, jobs=Depends(get_diffusion_jobs)
):
    return _diffusion_job_call(jobs.start_calibration, body)


@router.post(
    "/diffusion/quantizations", response_model=DiffusionJobView, status_code=202
)
async def start_diffusion_quantization(
    body: DiffusionQuantizationRequest, jobs=Depends(get_diffusion_jobs)
):
    return _diffusion_job_call(jobs.start_quantization, body)


@router.get("/diffusion/jobs", response_model=DiffusionJobsResponse)
async def list_diffusion_jobs(jobs=Depends(get_diffusion_jobs)):
    return jobs.list()


@router.get("/diffusion/jobs/{job_id}", response_model=DiffusionJobView)
async def get_diffusion_job(job_id: str, jobs=Depends(get_diffusion_jobs)):
    return _diffusion_job_call(jobs.get, job_id)


@router.post("/diffusion/jobs/{job_id}/cancel", response_model=DiffusionJobView)
async def cancel_diffusion_job(job_id: str, jobs=Depends(get_diffusion_jobs)):
    return _diffusion_job_call(jobs.cancel, job_id)


for feature_router in (
    acquisition_router,
    diagnostics_router,
    model_options_router,
    monitoring_router,
    server_settings_router,
    workspace_router,
):
    router.include_router(feature_router)
