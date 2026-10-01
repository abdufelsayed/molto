# SPDX-License-Identifier: Apache-2.0
"""Authenticated server and credential management feature routes."""

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from molto_management.management import ManagementContext, ManagementError
from molto_management.management_server import ServerManagementService
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr

from molto_server.api.management_dependencies import get_context, get_runtime

router = APIRouter()


class SubkeyWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: StrictStr = Field(default="", max_length=200)
    key: StrictStr | None = None


class SubkeyPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: StrictStr | None = Field(default=None, max_length=200)
    key: StrictStr | None = None


class MainKeyWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: StrictStr


class PolicyWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skip_api_key_verification: StrictBool | None = None
    allow_unauthenticated_inference: StrictBool | None = None


ERROR_STATUS = {
    "not_found": 404,
    "conflict": 409,
    "busy": 409,
    "invalid_configuration": 422,
    "unavailable": 503,
    "persistence_failed": 500,
    "runtime_failed": 500,
    "rollback_failed": 500,
}


def run(call, *args, **kwargs):
    try:
        return call(*args, **kwargs)
    except ManagementError as exc:
        raise HTTPException(
            ERROR_STATUS.get(exc.code, 500), {"code": exc.code, "message": exc.detail}
        ) from exc


def service(context: ManagementContext = Depends(get_context)):
    return ServerManagementService(context)


@router.get("/server/settings")
def settings(svc=Depends(service)):
    return run(svc.describe)


@router.get("/server/defaults")
def defaults(svc=Depends(service)):
    return run(svc.defaults)


@router.patch("/server/settings")
async def patch_settings(
    patch: dict[str, dict[str, Any]], svc=Depends(service), runtime=Depends(get_runtime)
):
    async with runtime.mutation_lock:
        return run(svc.patch, patch)


@router.get("/auth/keys")
def keys(svc=Depends(service)):
    return run(svc.keys)


@router.post("/auth/subkeys", status_code=201)
async def create_subkey(
    body: SubkeyWrite, svc=Depends(service), runtime=Depends(get_runtime)
):
    async with runtime.mutation_lock:
        return run(svc.subkey, body.model_dump(exclude_none=True))


@router.patch("/auth/subkeys/{key_id}")
async def edit_subkey(
    key_id: str, body: SubkeyPatch, svc=Depends(service), runtime=Depends(get_runtime)
):
    async with runtime.mutation_lock:
        return run(svc.subkey, body.model_dump(exclude_unset=True), key_id)


@router.delete("/auth/subkeys/{key_id}")
async def delete_subkey(
    key_id: str, svc=Depends(service), runtime=Depends(get_runtime)
):
    async with runtime.mutation_lock:
        return run(svc.subkey, {}, key_id, delete=True)


@router.patch("/auth/main-key")
async def main_key(
    body: MainKeyWrite, svc=Depends(service), runtime=Depends(get_runtime)
):
    async with runtime.mutation_lock:
        return run(svc.main_key, body.key)


@router.patch("/auth/policy")
async def policy(body: PolicyWrite, svc=Depends(service), runtime=Depends(get_runtime)):
    async with runtime.mutation_lock:
        return run(svc.policy, body.model_dump(exclude_unset=True))


@router.get("/server/resources")
def resources(
    tier: Literal["safe", "balanced", "aggressive", "custom"] | None = None,
    custom_ceiling_gb: float | None = Query(default=None, gt=0, allow_inf_nan=False),
    guard_enabled: bool | None = None,
    svc=Depends(service),
):
    return run(svc.resources, tier, custom_ceiling_gb, guard_enabled)


@router.get("/server/info")
def info(svc=Depends(service)):
    return run(svc.info)


@router.post("/server/restart", status_code=202)
async def restart(svc=Depends(service), runtime=Depends(get_runtime)):
    async with runtime.mutation_lock:
        if runtime.operation_lock.locked():
            raise HTTPException(
                409,
                {"code": "busy", "message": "An exclusive model operation is active"},
            )
        try:
            return await svc.restart()
        except ManagementError as exc:
            raise HTTPException(
                ERROR_STATUS.get(exc.code, 500),
                {"code": exc.code, "message": exc.detail},
            ) from exc


@router.get("/server/update")
async def update(channel: Literal["stable", "beta"] = "stable", svc=Depends(service)):
    return await svc.update(channel)


@router.get("/server/integrations")
def integrations(svc=Depends(service)):
    return run(svc.integrations)
