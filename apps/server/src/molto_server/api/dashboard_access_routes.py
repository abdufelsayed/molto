# SPDX-License-Identifier: Apache-2.0
"""Private credential handoff to the launcher-owned local dashboard."""

import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from molto_config.utils.network import is_loopback_bind, is_loopback_bind_host

from molto_server.auth import _auth_context, compare_keys

router = APIRouter(include_in_schema=False)


@router.post("/_internal/dashboard-key")
def dashboard_key(request: Request):
    expected = os.environ.get("MOLTO_LOCAL_ACCESS_TOKEN", "")
    provided = request.headers.get("x-molto-local-access", "")
    peer = request.client.host if request.client is not None else None
    forwarded = any(
        name in {"forwarded", "x-real-ip"} or name.startswith("x-forwarded-")
        for name in request.headers
    )
    if (
        os.environ.get("MOLTO_SUPERVISED") != "application"
        or not expected
        or not provided
        or not compare_keys(provided, expected)
        or not is_loopback_bind_host(peer)
        or forwarded
    ):
        raise HTTPException(403, "Local dashboard access unavailable")
    context = _auth_context(request)
    if not is_loopback_bind(context.bind_host):
        raise HTTPException(403, "Local dashboard access unavailable")
    if not context.main_key:
        raise HTTPException(409, "Create the first main key before connecting")
    return JSONResponse(
        {"key": context.main_key}, headers={"Cache-Control": "no-store"}
    )
