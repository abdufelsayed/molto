"""Authenticated management monitoring endpoints."""

from typing import Any, Literal

from fastapi import APIRouter, Depends, Query
from molto_management.management_monitoring import MonitoringService
from pydantic import BaseModel, ConfigDict, Field

from molto_server.api.management_dependencies import get_context
from molto_server.auth import require_management_key

router = APIRouter(dependencies=[Depends(require_management_key)])


def service(context=Depends(get_context)):
    return MonitoringService(context)


class CacheProbeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1, max_length=512)
    messages: list[dict[str, Any]] = Field(min_length=1, max_length=256)
    tools: list[dict[str, Any]] | None = Field(default=None, max_length=128)
    chat_template_kwargs: dict[str, Any] | None = None
    thinking_budget: int | None = Field(default=None, ge=0, le=131072)


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scope: Literal["session", "alltime"] = "session"


@router.get("/monitoring/activity")
def activity(monitor=Depends(service)):
    return monitor.activity()


@router.get("/monitoring/usage")
def usage(
    range: Literal["today", "yesterday", "7d", "30d", "90d", "month"] = "today",
    model: str = Query(default="", max_length=512),
    include_details: bool = False,
    monitor=Depends(service),
):
    return monitor.usage(range, model, include_details)


@router.get("/monitoring/logs")
def logs(
    lines: int = Query(default=100, ge=1, le=10000),
    file: str | None = None,
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None = None,
    monitor=Depends(service),
):
    return monitor.logs(lines, file, level)


@router.get("/monitoring/versions")
def versions(monitor=Depends(service)):
    return monitor.versions()


@router.post("/monitoring/stats/reset")
def reset(body: ResetRequest, monitor=Depends(service)):
    return monitor.reset_stats(body.scope)


@router.post("/monitoring/cache/probe")
def probe(body: CacheProbeRequest, monitor=Depends(service)):
    return monitor.probe(body)
