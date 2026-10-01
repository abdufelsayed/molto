# SPDX-License-Identifier: Apache-2.0
"""Errors for the private inference application."""

import logging

from fastapi import HTTPException
from fastapi import Request as FastAPIRequest
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from omlx_management.management import ManagementError

# Import from new modular API
from omlx_runtime.engine.distributed import DistributedInferenceError
from omlx_runtime.exceptions import (
    InvalidRequestError,
    PrefillMemoryAbortedError,
    PrefillMemoryExceededError,
    SchedulerQueueFullError,
)

logger = logging.getLogger(__name__)


def _status_to_error_type(status_code: int) -> str:
    """Map HTTP status code to OpenAI error type string."""
    if status_code == 401:
        return "authentication_error"
    if status_code == 404:
        return "not_found_error"
    if status_code == 413:
        # Body-size rejections are still request-shape errors.
        return "invalid_request_error"
    if status_code == 429:
        return "rate_limit_error"
    if status_code >= 500:
        return "server_error"
    return "invalid_request_error"


def _is_api_route(request: FastAPIRequest) -> bool:
    """Check if request targets an OpenAI-compatible API route.

    Path-prefix only. This assumes the FastAPI app is mounted at root
    (the oMLX deployment shape) and that route paths are case-sensitive
    — both true today. If a future deployment mounts this app under a
    prefix (``app.mount("/api", ...)``), ``request.url.path`` returns
    the full mounted path and every ``/v1/...`` route would be
    classified as non-API. Switch to ``request.scope.get("route")``
    matching at that point.
    """

    return request.url.path.startswith("/v1/")


def _openai_error_body(message, status_code: int, param=None, code=None) -> dict:
    """Build an OpenAI-compatible error response body."""
    return {
        "error": {
            "message": message,
            "type": _status_to_error_type(status_code),
            "param": param,
            "code": code,
        }
    }


def _prefill_memory_error_detail(exc: PrefillMemoryExceededError) -> str:
    if isinstance(exc, PrefillMemoryAbortedError):
        # This request was admitted and then killed mid-prefill, so "rejected
        # this prompt" would be wrong. The message already names usage, the
        # watermark and the binding ceiling with its own advice, so the
        # generic ladder below is not appended.
        return f"oMLX memory guard aborted this request mid-prefill: {str(exc)}"
    return (
        "oMLX prefill memory guard rejected this prompt: "
        f"{str(exc)} "
        "To continue, set Memory Guard to aggressive, raise the custom "
        "memory guard ceiling, free system memory, or compact/reduce context."
    )


def _streaming_error_payload(e: Exception, context: str) -> dict:
    """Error body for a failure inside an OpenAI SSE generator.

    A prefill-guard rejection raised during streaming must keep its
    structured body (root ``type``, ``omlx_code``, byte fields) — the
    blanket ``except`` in the generators would otherwise flatten it to a
    generic ``server_error`` that clients cannot classify, and the correct
    handler in ``_with_sse_keepalive`` never sees the exception because the
    generator's own handler is the innermost one (#3036).
    """
    if isinstance(e, PrefillMemoryExceededError):
        logger.warning(f"{context} prefill rejected: {e}")
        return _prefill_memory_openai_error_body(e)
    logger.error(f"Error during {context}: {e}")
    return {"error": {"message": str(e), "type": "server_error"}}


def _prefill_memory_openai_error_body(
    exc: PrefillMemoryExceededError,
    *,
    status_code: int = 400,
) -> dict:
    code = (
        "prefill_memory_aborted"
        if isinstance(exc, PrefillMemoryAbortedError)
        else "prefill_memory_exceeded"
    )
    content = _openai_error_body(
        _prefill_memory_error_detail(exc),
        status_code,
        code=code,
    )
    content["type"] = "error"
    content["error"]["omlx_code"] = code
    if exc.estimated_bytes is not None:
        content["error"]["estimated_bytes"] = exc.estimated_bytes
    if exc.limit_bytes is not None:
        content["error"]["limit_bytes"] = exc.limit_bytes
    return content


class ErrorsController:
    async def management_error_handler(
        self, request: FastAPIRequest, exc: ManagementError
    ):
        status = {
            "not_found": 404,
            "invalid_configuration": 400,
            "invalid": 422,
            "busy": 409,
            "conflict": 409,
            "unavailable": 503,
            "persistence": 503,
            "persistence_failed": 503,
            "runtime_failed": 500,
            "rollback_failed": 500,
        }.get(exc.code, 500)
        return JSONResponse(status_code=status, content={"detail": exc.detail})

    async def http_exception_handler(self, request: FastAPIRequest, exc: HTTPException):
        """Log all HTTP errors (4xx/5xx) before returning the response."""
        logger.warning(
            "%s %s → %d: %s",
            request.method,
            request.url.path,
            exc.status_code,
            exc.detail,
        )
        if _is_api_route(request):
            content = _openai_error_body(
                exc.detail, exc.status_code, code=getattr(exc, "code", None)
            )
        else:
            content = {"detail": exc.detail}
        return JSONResponse(
            status_code=exc.status_code,
            content=content,
            headers=exc.headers,
        )

    async def validation_exception_handler(
        self, request: FastAPIRequest, exc: RequestValidationError
    ):
        """Log request validation errors (422) before returning the response."""
        logger.warning(
            "%s %s → 422: %s",
            request.method,
            request.url.path,
            exc.errors(),
        )
        if _is_api_route(request):
            errors = exc.errors()
            parts = []
            for err in errors:
                loc = " -> ".join(str(x) for x in err.get("loc", []))
                msg = err.get("msg", "")
                parts.append(f"{loc}: {msg}" if loc else msg)
            detail_str = "; ".join(parts)
            param = errors[0].get("loc", [None])[-1] if errors else None
            content = _openai_error_body(detail_str, 422, param=param)
        else:
            content = {"detail": jsonable_encoder(exc.errors())}
        return JSONResponse(status_code=422, content=content)

    async def invalid_request_error_handler(
        self, request: FastAPIRequest, exc: InvalidRequestError
    ):
        """Map internal request validation failures to OpenAI-compatible 400s."""
        logger.warning(
            "%s %s -> 400: %s",
            request.method,
            request.url.path,
            exc,
        )
        if _is_api_route(request):
            content = _openai_error_body(str(exc), 400, param=exc.field)
        else:
            content = {"detail": str(exc)}
        return JSONResponse(status_code=400, content=content)

    async def scheduler_queue_full_handler(
        self, request: FastAPIRequest, exc: SchedulerQueueFullError
    ):
        """Map scheduler queue cap exhaustion to HTTP 503 + Retry-After."""
        logger.warning(
            "%s %s → 503: %s",
            request.method,
            request.url.path,
            exc,
        )
        detail = (
            f"Scheduler waiting queue full ({exc.current_depth}/{exc.max_depth}). "
            f"Try again shortly."
        )
        if _is_api_route(request):
            content = _openai_error_body(detail, 503)
        else:
            content = {"detail": detail}
        return JSONResponse(
            status_code=503,
            content=content,
            headers={"Retry-After": "1"},
        )

    async def distributed_unavailable_handler(
        self, request: FastAPIRequest, exc: DistributedInferenceError
    ):
        """Map a failed distributed cluster check to a clean HTTP 503.

        The distributed engine's preflight health gate raises this before the
        StreamingResponse is built, which is what turns a dead or half-dead
        cluster into an honest error instead of an empty 200 whose failure only
        surfaces mid-stream (#2708). Errors raised after streaming has started
        still land in-band; this handler covers every pre-commit path.
        """
        logger.warning(
            "%s %s -> 503: %s",
            request.method,
            request.url.path,
            exc,
        )
        if _is_api_route(request):
            content = _openai_error_body(str(exc), 503)
        else:
            content = {"detail": str(exc)}
        return JSONResponse(status_code=503, content=content)

    async def prefill_memory_exceeded_handler(
        self, request: FastAPIRequest, exc: PrefillMemoryExceededError
    ):
        """Map prefill peak overshoot to HTTP 400 with a clear JSON body.

        The synchronous prefill memory guard in ``Scheduler.add_request`` raises
        this when the estimated KV+SDPA peak for a request would push memory
        past the user-configured memory guard ceiling. The caller's prompt
        fits in the model's context window but is too large for the host's
        headroom.

        This is an actionable request rejection, not an HTTP body-size
        rejection. HTTP 400 also prevents Anthropic clients from collapsing
        this oMLX memory-guard failure into Anthropic's generic
        "Request too large (max 32MB)" body-size error.
        """
        detail = _prefill_memory_error_detail(exc)
        status_code = 400
        logger.warning(
            "%s %s → %d: %s",
            request.method,
            request.url.path,
            status_code,
            detail,
        )
        if _is_api_route(request):
            # code="prefill_memory_exceeded" lets OpenAI-SDK clients branch
            # on the failure mode. Without it, "context window too small"
            # and "host has no memory headroom" both surface as
            # invalid_request_error with code=None and clients can only
            # tell the user "shorten your prompt" — which is wrong when
            # the actual fix is to loosen the memory guard.
            # Surface the structured fields so clients can branch on
            # numeric values instead of regex-matching the human message.
            # OpenAI clients ignore unknown error fields so this is a
            # forward-compatible extension.
            content = _prefill_memory_openai_error_body(exc, status_code=status_code)
        else:
            content = {
                "detail": detail,
                "omlx_code": "prefill_memory_exceeded",
            }
            if exc.estimated_bytes is not None:
                content["estimated_bytes"] = exc.estimated_bytes
            if exc.limit_bytes is not None:
                content["limit_bytes"] = exc.limit_bytes
        return JSONResponse(status_code=status_code, content=content)

    async def unhandled_exception_handler(
        self, request: FastAPIRequest, exc: Exception
    ):
        """Log unhandled exceptions as 500 errors."""
        logger.error(
            "%s %s → 500 (unhandled): %s",
            request.method,
            request.url.path,
            exc,
            exc_info=exc,
        )
        if _is_api_route(request):
            content = _openai_error_body("Internal server error", 500)
        else:
            content = {"detail": "Internal server error"}
        return JSONResponse(status_code=500, content=content)
