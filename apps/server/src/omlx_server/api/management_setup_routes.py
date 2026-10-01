# SPDX-License-Identifier: Apache-2.0
"""First-run setup. Mount independently from authenticated management routes."""

from fastapi import APIRouter, Depends, HTTPException, Request
from omlx_management.management import ManagementContext, ManagementError
from omlx_management.management_setup import ManagementSetupService
from pydantic import BaseModel, ConfigDict, StrictStr

from omlx_server.api.management_dependencies import get_context, get_runtime

router = APIRouter(prefix="/management/v1/setup")


class SetupStatus(BaseModel):
    setup_required: bool
    allowed: bool
    reason: str | None


class SetupWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: StrictStr
    confirmation: StrictStr


class SetupResult(BaseModel):
    configured: bool


ERROR_STATUS = {
    "conflict": 409,
    "forbidden": 403,
    "invalid_configuration": 422,
    "unavailable": 503,
    "persistence_failed": 500,
    "runtime_failed": 500,
    "rollback_failed": 500,
}


def service(context: ManagementContext = Depends(get_context)):
    return ManagementSetupService(context)


def error(exc: ManagementError):
    return HTTPException(
        ERROR_STATUS.get(exc.code, 500), {"code": exc.code, "message": exc.detail}
    )


@router.get("", response_model=SetupStatus)
def status(request: Request, svc=Depends(service)):
    peer = request.client.host if request.client is not None else None
    try:
        return svc.status(peer)
    except ManagementError as exc:
        raise error(exc) from exc


@router.post("", status_code=201, response_model=SetupResult)
async def create(
    request: Request,
    body: SetupWrite,
    svc=Depends(service),
    runtime=Depends(get_runtime),
):
    peer = request.client.host if request.client is not None else None
    try:
        return await svc.create(body.key, body.confirmation, peer, runtime)
    except ManagementError as exc:
        raise error(exc) from exc
