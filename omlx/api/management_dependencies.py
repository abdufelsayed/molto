"""Shared request dependencies for the management feature routers."""
import threading

from fastapi import HTTPException, Request

from ..services.management import ManagementContext, ManagementService

_runtime_lock = threading.RLock()


def get_context(request: Request) -> ManagementContext:
    provider = getattr(request.app.state, "management_context_provider", None)
    if provider is None:
        raise HTTPException(503, "Server not initialized")
    return provider()


def get_service(request: Request) -> ManagementService:
    return ManagementService(get_context(request))


def get_runtime(request: Request):
    from ..services.management_runtime import ManagementRuntime

    context = get_context(request)
    with _runtime_lock:
        runtime = getattr(request.app.state, "management_runtime", None)
        if runtime is None or runtime.settings is not context.global_settings:
            runtime = ManagementRuntime(context)
            request.app.state.management_runtime = runtime
    return runtime
