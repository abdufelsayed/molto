"""Authenticated management diagnostic endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from molto_management.management import ManagementError
from molto_management.management_diagnostics import (
    DiagnosticRequest,
    DiagnosticsService,
)

from molto_server.api.management_dependencies import get_runtime
from molto_server.auth import require_management_key

router = APIRouter(tags=["diagnostics"], dependencies=[Depends(require_management_key)])


def service(runtime=Depends(get_runtime)):
    if not hasattr(runtime, "diagnostics"):
        runtime.diagnostics = DiagnosticsService(runtime)
    return runtime.diagnostics


def call(fn, *args):
    try:
        return fn(*args)
    except ManagementError as exc:
        raise HTTPException(
            {"not_found": 404, "busy": 409, "invalid": 422, "persistence": 503}.get(
                exc.code, 400
            ),
            exc.detail,
        ) from exc


@router.get("/diagnostics/capabilities")
def capabilities(svc=Depends(service)):
    return call(svc.capabilities)


@router.get("/diagnostics/runs")
def runs(svc=Depends(service)):
    return call(svc.list)


@router.post("/diagnostics/runs", status_code=202)
async def start(body: DiagnosticRequest, svc=Depends(service)):
    try:
        return await svc.start(body)
    except ManagementError as exc:
        raise HTTPException(
            {"not_found": 404, "busy": 409, "invalid": 422, "persistence": 503}.get(
                exc.code, 400
            ),
            exc.detail,
        ) from exc


@router.get("/diagnostics/runs/{run_id}")
def run(run_id: str, svc=Depends(service)):
    return call(svc.get, run_id)


@router.post("/diagnostics/runs/{run_id}/cancel")
def cancel(run_id: str, svc=Depends(service)):
    return call(svc.cancel, run_id)


@router.get("/diagnostics/runs/{run_id}/results")
def results(run_id: str, svc=Depends(service)):
    return call(svc.results, run_id)
