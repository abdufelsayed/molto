# SPDX-License-Identifier: Apache-2.0
"""Transport for the private inference application."""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Optional

from fastapi import HTTPException, Response
from fastapi import Request as FastAPIRequest
from fastapi.responses import JSONResponse, StreamingResponse

# Import from new modular API
from molto_runtime.engine import BaseEngine
from molto_runtime.exceptions import PrefillMemoryExceededError

from .engine_requests import _release_after_stream
from .errors import _openai_error_body, _prefill_memory_openai_error_body
from .shared import (
    _DISCONNECT_SIGNAL_SCOPE_KEY,
    _JSON_KEEPALIVE_GRACE_S,
    _KEEPALIVE_ANTHROPIC_PING,
    _KEEPALIVE_CHAT_CHUNK,
    _KEEPALIVE_COMMENT,
    _KEEPALIVE_COMPLETION_CHUNK,
    _KEEPALIVE_SENTINEL,
    _ClientDisconnectSignal,
    _LLMEngineLease,
)

logger = logging.getLogger(__name__)


def _completion_keepalive_chunk(response_id: str) -> str:
    """Keepalive frame that shares the stream's completion id."""
    return (
        'data: {"id":"' + response_id + '","object":"text_completion","created":0,'
        '"model":"keepalive",'
        '"choices":[{"index":0,"text":"","logprobs":null,"finish_reason":null}]}\n\n'
    )


def _chat_keepalive_chunk(response_id: str) -> str:
    """Keepalive frame that shares the stream's completion id.

    The static ``_KEEPALIVE_CHAT_CHUNK`` carries a sentinel id
    (``chatcmpl-keepalive``) that differs from the real completion chunks.
    Strict OpenAI stream accumulators (e.g. the official ``openai-go`` SDK)
    assume every chunk in one streamed completion shares a single ``id``: they
    latch the first chunk's id and silently drop later chunks whose id differs,
    discarding the real ``tool_calls``/``finish_reason``/``usage``. Emitting the
    keepalive with the stream's own ``response_id`` makes it a true no-op for
    those clients while remaining a parseable data event for clients that can't
    handle SSE comment lines.

    The delta must also carry ``"role":"assistant"`` — see the comment on
    ``_KEEPALIVE_CHAT_CHUNK`` (accumulators that type the stream from the
    first chunk's role drop tool_call_chunks without it, #2074).
    """
    return (
        'data: {"id":"' + response_id + '","object":"chat.completion.chunk",'
        '"created":0,"model":"keepalive",'
        '"choices":[{"index":0,"delta":{"role":"assistant","content":""},'
        '"finish_reason":null}]}\n\n'
    )


async def _safe_anext(ait):
    """Wrapper for __anext__ that converts StopAsyncIteration to a sentinel.

    StopAsyncIteration cannot propagate through asyncio.Task (raises RuntimeError),
    so we catch it here and return a sentinel value instead.
    """
    try:
        return await ait.__anext__()
    except StopAsyncIteration:
        return _KEEPALIVE_SENTINEL


async def _aclose_async_iterator(iterator: object) -> None:
    """Close an async generator when the response transport ends."""

    close = getattr(iterator, "aclose", None)
    if callable(close):
        await close()


def _request_abort_id(engine: BaseEngine) -> str | None:
    """Mint an opaque id only for engines with targeted abort semantics."""

    if not getattr(engine, "supports_request_scoped_abort", False):
        return None
    abort = getattr(engine, "abort_request", None)
    if not callable(abort):
        return None
    return f"transport-{uuid.uuid4().hex}"


async def _with_request_disconnect_abort(
    generator: AsyncIterator[str],
    http_request: FastAPIRequest,
    engine: BaseEngine,
    request_id: str | None,
) -> AsyncIterator[str]:
    """Bind one public response transport to one inference request."""

    signal = http_request.scope.get(_DISCONNECT_SIGNAL_SCOPE_KEY)
    token: int | None = None
    if request_id is not None and isinstance(signal, _ClientDisconnectSignal):
        abort = getattr(engine, "abort_request")

        async def abort_disconnected_request() -> None:
            await abort(
                request_id,
                reason="public client transport disconnected",
                error_code="client_disconnected",
            )

        token = signal.register(abort_disconnected_request)

    try:
        async for chunk in generator:
            yield chunk
    finally:
        if token is not None:
            signal.unregister(token)
        await _aclose_async_iterator(generator)


async def _with_sse_keepalive(
    generator: AsyncIterator[str],
    http_request: Optional["FastAPIRequest"] = None,
    interval: float = 10.0,
    disconnect_poll: float = 2.0,
    keepalive_chunk: str | None = _KEEPALIVE_COMMENT,
) -> AsyncIterator[str]:
    """Wrap an SSE generator to send periodic keepalive frames.

    During long prefill (e.g. 90k tokens), no SSE events are emitted,
    causing clients with read timeouts (like Claude Code) to disconnect.
    This wrapper periodically yields a keepalive frame to hold the
    connection open. The frame format depends on caller-supplied
    keepalive_chunk: a legacy SSE comment, a protocol-aware no-op event,
    or None to disable emission entirely.

    When http_request is provided, also polls for client disconnect
    between prefill steps. This detects cancellation during long prefills
    where uvicorn's ASGI disconnect message is not delivered until after
    the generator yields.
    """
    ait = generator.__aiter__()
    task = None
    keepalive_elapsed = 0.0
    next_disconnect_check = (
        time.monotonic() + disconnect_poll if http_request is not None else None
    )

    async def client_disconnected() -> bool:
        try:
            disconnected = await http_request.is_disconnected()
        except Exception as e:
            logger.debug(f"is_disconnected() check failed: {e}")
            return False  # is_disconnected() can fail if scope is already closed
        if disconnected:
            logger.info(
                "Client disconnected during streaming (is_disconnected), cancelling"
            )
        return disconnected

    # Send initial keepalive immediately so clients with short read
    # timeouts (e.g. openclaw ~15s) don't disconnect during prefill.
    if keepalive_chunk is not None:
        yield keepalive_chunk

    try:
        while True:
            # A continuously-ready token stream never enters the timeout branch
            # below. Probe on elapsed wall time as well so aborting a fast client
            # still closes the upstream generator and its inference request.
            if (
                next_disconnect_check is not None
                and time.monotonic() >= next_disconnect_check
            ):
                next_disconnect_check = time.monotonic() + disconnect_poll
                if await client_disconnected():
                    return
            task = asyncio.ensure_future(_safe_anext(ait))
            keepalive_elapsed = 0.0
            while not task.done():
                # Use shorter poll interval for disconnect detection,
                # accumulate time for keepalive emission
                wait_time = disconnect_poll if http_request else interval
                done, _ = await asyncio.wait({task}, timeout=wait_time)
                if done:
                    break
                # Check for client disconnect
                if http_request is not None:
                    next_disconnect_check = time.monotonic() + disconnect_poll
                    if await client_disconnected():
                        task.cancel()
                        try:
                            await task
                        except (asyncio.CancelledError, StopAsyncIteration):
                            pass
                        return
                # Send keepalive at the configured interval
                keepalive_elapsed += wait_time
                if keepalive_elapsed >= interval:
                    keepalive_elapsed = 0.0
                    if keepalive_chunk is not None:
                        yield keepalive_chunk
            if task.done():
                try:
                    result = task.result()
                except Exception as e:
                    if isinstance(e, PrefillMemoryExceededError):
                        logger.warning(f"SSE generator prefill rejected: {e}")
                        error_data = _prefill_memory_openai_error_body(e)
                    else:
                        logger.error(f"SSE generator error: {e}")
                        error_data = {
                            "error": {"message": str(e), "type": "server_error"}
                        }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                if result is _KEEPALIVE_SENTINEL:
                    return
                yield result
    finally:
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, StopAsyncIteration):
                pass
        if hasattr(ait, "aclose"):
            await ait.aclose()


async def _run_with_disconnect_guard(
    http_request: FastAPIRequest,
    coro,
    poll_interval: float = 1.0,
):
    """Run a coroutine with client disconnect detection.

    For non-streaming requests, FastAPI/uvicorn does NOT automatically cancel
    the handler coroutine when a client disconnects. This helper polls
    is_disconnected() periodically and cancels the task on disconnect,
    which triggers CancelledError -> abort_request() in EngineCore.generate()
    to free scheduler/GPU resources.
    """
    task = asyncio.create_task(coro)
    while not task.done():
        done, _ = await asyncio.wait({task}, timeout=poll_interval)
        if done:
            break
        if await http_request.is_disconnected():
            logger.info("Client disconnected, cancelling generation task")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return None
    return task.result()


async def _with_json_keepalive(
    http_request: FastAPIRequest,
    coro_or_task,
    interval: float = 10.0,
    disconnect_poll: float = 2.0,
) -> AsyncIterator[str]:
    """Send keepalive spaces while waiting for a coroutine or task.

    For non-streaming requests, the HTTP response body is buffered until
    generation finishes, causing client read timeouts on long prefills.
    This wrapper uses StreamingResponse to send space characters as
    keepalive. JSON parsers ignore leading whitespace, so the final
    response parses normally.

    Callers reaching this generator via ``_json_response_or_keepalive``
    have already confirmed the task is still running past the grace period
    -- from this point on, HTTP has committed the response status to 200
    (the status line ships with the first byte), so a failure discovered
    here can only be reported through the JSON body, never the status code.
    ``asyncio.ensure_future`` is a no-op when given an already-scheduled
    task, so passing either shape is safe.
    """
    task = asyncio.ensure_future(coro_or_task)
    keepalive_elapsed = 0.0

    yield " "

    try:
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=disconnect_poll)
            if done:
                break
            if http_request is not None:
                try:
                    disconnected = await http_request.is_disconnected()
                    if disconnected:
                        logger.info(
                            "Client disconnected during non-streaming response, cancelling"
                        )
                        task.cancel()
                        try:
                            await task
                        except (asyncio.CancelledError, StopAsyncIteration):
                            pass
                        return
                except Exception:
                    pass
            keepalive_elapsed += disconnect_poll
            if keepalive_elapsed >= interval:
                keepalive_elapsed = 0.0
                yield " "
        try:
            result = task.result()
        except PrefillMemoryExceededError as e:
            logger.warning(f"JSON keepalive prefill rejected: {e}")
            yield json.dumps(_prefill_memory_openai_error_body(e))
            return
        except HTTPException as e:
            # Headers are already sent; preserve the API error in the body.
            logger.warning(
                "JSON keepalive request failed (%d): %s", e.status_code, e.detail
            )
            yield json.dumps(
                _openai_error_body(
                    e.detail, e.status_code, code=getattr(e, "code", None)
                )
            )
            return
        if result is not None:
            yield result
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, StopAsyncIteration):
                pass


async def _json_response_or_keepalive(
    http_request: FastAPIRequest,
    coro,
    *,
    media_type: str = "application/json",
    headers: dict | None = None,
    lease: "_LLMEngineLease | None" = None,
) -> Response:
    """Resolve a non-streaming JSON-body coroutine, preferring a real HTTP
    status code over the keepalive-streaming fallback.

    Most rejections (validation, guard checks) resolve in well under a
    second. Racing the coroutine against ``_JSON_KEEPALIVE_GRACE_S`` lets
    those return a plain ``Response``/``JSONResponse`` with the correct
    status code (e.g. 400 for a memory-guard rejection) instead of the
    keepalive wrapper's forced 200 -- a request that fails fast must not
    look like a success to the client. Only requests still running past
    the grace period fall back to keepalive streaming, where any later
    failure can only be signaled via the JSON body: the status line ships
    with the first keepalive byte and cannot be revised afterward, an
    HTTP/ASGI constraint, not a choice.
    """
    task = asyncio.ensure_future(coro)
    try:
        done, _pending = await asyncio.wait({task}, timeout=_JSON_KEEPALIVE_GRACE_S)
    except BaseException:
        # The handler can unwind during the grace period (server shutdown,
        # middleware cancellation, or another request-level abort). The task
        # was scheduled independently by ensure_future(), so cancellation does
        # not propagate into it automatically. Drain it before releasing the
        # lease; otherwise inference can continue after ModelRegistry considers
        # the engine idle and eligible for unload.
        task.cancel()
        with suppress(BaseException):
            await task
        if lease is not None:
            await lease.release()
        raise
    if done:
        if lease is not None:
            await lease.release()
        try:
            result = task.result()
        except PrefillMemoryExceededError as e:
            logger.warning(f"JSON keepalive prefill rejected (fast path): {e}")
            return JSONResponse(
                status_code=400,
                content=_prefill_memory_openai_error_body(e),
                headers=headers,
            )
        return Response(content=result, media_type=media_type, headers=headers)

    generator = _with_json_keepalive(http_request, task)
    if lease is not None:
        generator = _release_after_stream(generator, lease)
    return StreamingResponse(generator, media_type=media_type, headers=headers)


class TransportController:
    def _resolve_keepalive(self, protocol: str) -> str | None:
        """Pick a wire-level keepalive frame for the given API protocol.

        Returns None when the configured mode disables keepalive for this protocol.
        Modes: "chunk" (default, protocol-aware), "comment" (legacy SSE comment),
        "off" (no keepalive). Some clients (e.g. OpenClaw / WorkBuddy) cannot parse
        SSE comment lines, so the chunk mode emits valid no-op events instead.
        """
        global_settings = self.state.global_settings
        mode = "chunk"
        if global_settings is not None:
            mode = getattr(global_settings.server, "sse_keepalive_mode", "chunk")
        if mode == "off":
            return None
        if mode == "comment":
            return _KEEPALIVE_COMMENT
        if protocol == "openai_chat":
            return _KEEPALIVE_CHAT_CHUNK
        if protocol == "openai_completion":
            return _KEEPALIVE_COMPLETION_CHUNK
        if protocol == "anthropic":
            return _KEEPALIVE_ANTHROPIC_PING
        if protocol == "openai_responses":
            return None
        return None
