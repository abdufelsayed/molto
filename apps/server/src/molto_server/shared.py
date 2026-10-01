# SPDX-License-Identifier: Apache-2.0
"""Shared for the private inference application."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from fastapi import HTTPException

# Import from new modular API
from molto_runtime.engine_pool import EnginePool

logger = logging.getLogger(__name__)


_TEXTUAL_BODY_CONTENT_TYPES = (
    "application/json",
    "application/x-www-form-urlencoded",
    "text/",
)


class DebugRequestLoggingMiddleware:
    """Pure ASGI middleware for trace-level request body logging.

    Uses raw ASGI protocol instead of BaseHTTPMiddleware to avoid
    wrapping StreamingResponse in an intermediate pipe layer, which
    causes connection corruption on HTTP keep-alive connections.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or not logger.isEnabledFor(5)
            or scope.get("method") != "POST"
        ):
            await self.app(scope, receive, send)
            return

        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope.get("headers", [])
        }
        content_type = headers.get("content-type", "")
        if not _is_textual_body(content_type):
            # Multipart / binary uploads (audio files can be 100 MB):
            # dumping the raw bytes garbles the terminal and buffering
            # the whole body just for logging wastes memory — log a
            # summary and stream the body through untouched.
            logger.log(
                5,
                "Incoming %s %s — body: <%s, %s bytes omitted>",
                scope["method"],
                scope["path"],
                content_type or "unknown content-type",
                headers.get("content-length", "?"),
            )
            await self.app(scope, receive, send)
            return

        # Read and cache the request body for logging
        body_parts = []
        while True:
            message = await receive()
            body_parts.append(message)
            if not message.get("more_body", False):
                break

        body = b"".join(part.get("body", b"") for part in body_parts)
        logger.log(
            5,
            "Incoming %s %s — body: %s",
            scope["method"],
            scope["path"],
            body.decode("utf-8", errors="replace"),
        )

        # Replay cached body for inner app, then forward real receive
        body_sent = False

        async def cached_receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, cached_receive, send)


_DISCONNECT_SIGNAL_SCOPE_KEY = "molto_runtime.client_disconnect_signal"


class _ClientDisconnectSignal:
    """Fan one ASGI disconnect message out to request-owned cleanup hooks.

    Starlette's ``StreamingResponse`` and ``Request.is_disconnected()`` both
    consume the same one-shot ``http.disconnect`` receive message.  Whichever
    one wins used to hide it from the other.  In particular, a real Uvicorn
    socket could close while a long distributed prefill kept running because
    the response generator never reached its nested ``aclose()`` chain.

    The outer ASGI middleware records the message before either consumer sees
    it, then schedules request-scoped callbacks outside Starlette's response
    cancellation scope.  A callback is keyed by the inference request id, so
    cancelling one client can never fall back to aborting every active request.
    """

    def __init__(self, tasks: set[asyncio.Task] | None = None) -> None:
        self._tasks = tasks if tasks is not None else set()
        self._disconnected = False
        self._next_token = 0
        self._callbacks: dict[int, Callable[[], Awaitable[None]]] = {}

    def register(self, callback: Callable[[], Awaitable[None]]) -> int:
        self._next_token += 1
        token = self._next_token
        if self._disconnected:
            self._spawn(callback)
        else:
            self._callbacks[token] = callback
        return token

    def unregister(self, token: int) -> None:
        self._callbacks.pop(token, None)

    def disconnect(self) -> None:
        if self._disconnected:
            return
        self._disconnected = True
        callbacks = tuple(self._callbacks.values())
        self._callbacks.clear()
        for callback in callbacks:
            self._spawn(callback)

    def _spawn(self, callback: Callable[[], Awaitable[None]]) -> None:
        async def run_callback() -> None:
            try:
                await callback()
            except Exception:
                logger.exception("Request-scoped disconnect callback failed")

        task = asyncio.create_task(run_callback())
        # asyncio only holds weak task references.  Retain the detached abort
        # until it completes; Starlette may already be cancelling the response
        # task that observed the disconnect.
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


class ClientDisconnectTrackingMiddleware:
    """Record ``http.disconnect`` before competing ASGI consumers receive it."""

    def __init__(self, app):
        self.app = app
        self._tasks: set[asyncio.Task] = set()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        signal = _ClientDisconnectSignal(self._tasks)
        scope[_DISCONNECT_SIGNAL_SCOPE_KEY] = signal

        async def tracked_receive():
            message = await receive()
            if message.get("type") == "http.disconnect":
                signal.disconnect()
            return message

        await self.app(scope, tracked_receive, send)


@dataclass
class _LLMEngineLease:
    """Release handle for an LLM engine lease taken from EnginePool."""

    pool: EnginePool | None = None
    model_id: str | None = None
    released: bool = False

    async def release(self) -> None:
        if self.released:
            return
        self.released = True
        if self.model_id is not None:
            await self.pool.release_engine(self.model_id)

    def abort_requested(self) -> bool:
        return self.abort_reason() is not None

    def abort_reason(self) -> str | None:
        if self.model_id is None or self.released:
            return None
        pool = self.pool
        if pool is None:
            return None
        get_reason = getattr(pool, "get_abort_requested_reason", None)
        if callable(get_reason):
            return get_reason(self.model_id)
        is_abort_requested = getattr(pool, "is_abort_requested", None)
        if not callable(is_abort_requested):
            return None
        return "hard memory pressure" if is_abort_requested(self.model_id) else None


_KEEPALIVE_SENTINEL = object()


_KEEPALIVE_COMMENT = ": keep-alive\n\n"


_KEEPALIVE_CHAT_CHUNK = (
    'data: {"id":"chatcmpl-keepalive","object":"chat.completion.chunk",'
    '"created":0,"model":"keepalive",'
    '"choices":[{"index":0,"delta":{"role":"assistant","content":""},'
    '"finish_reason":null}]}\n\n'
)


_KEEPALIVE_COMPLETION_CHUNK = (
    'data: {"id":"cmpl-keepalive","object":"text_completion","created":0,'
    '"model":"keepalive",'
    '"choices":[{"index":0,"text":"","logprobs":null,"finish_reason":null}]}\n\n'
)


_KEEPALIVE_ANTHROPIC_PING = 'event: ping\ndata: {"type":"ping"}\n\n'


_JSON_KEEPALIVE_GRACE_S = 2.0


class _ToolCallGenerationError(HTTPException):
    """Keep generation failure codes through JSON keepalive responses."""

    def __init__(self, error: dict):
        super().__init__(status_code=500, detail=error["message"])
        self.code = error["code"]


def _is_textual_body(content_type: str) -> bool:
    """True when a request body is safe to dump into the log as text."""
    ct = content_type.split(";", 1)[0].strip().lower()
    return ct.startswith(_TEXTUAL_BODY_CONTENT_TYPES) or ct.endswith("+json")
