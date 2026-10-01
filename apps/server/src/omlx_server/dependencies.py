"""Request-bound access to the application's inference controller."""

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer(auto_error=False)


def get_controller(request: Request):
    return request.app.state.controller


async def verify_api_key(
    request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)
):
    return await get_controller(request).verify_api_key(request, credentials)


async def verify_inference_api_key(
    request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)
):
    return await get_controller(request).verify_inference_api_key(request, credentials)


async def require_distributed_inference_enabled(request: Request):
    return await get_controller(request).require_distributed_inference_enabled()


def get_engine_pool(request: Request):
    return get_controller(request).get_engine_pool()
