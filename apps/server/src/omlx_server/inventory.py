# SPDX-License-Identifier: Apache-2.0
"""Inventory for the private inference application."""

import logging
import uuid
from collections.abc import AsyncIterator

from fastapi import Depends, HTTPException, Response
from fastapi import Request as FastAPIRequest
from fastapi.responses import StreamingResponse
from omlx_config._version import __version__

# Import from new modular API
from omlx_contracts.api.openai_models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelInfo,
    ModelsResponse,
    Usage,
)
from omlx_runtime.documents.markitdown import (
    MARKITDOWN_MODEL_ID,
    MarkItDownRequestError,
    convert_messages_to_markdown_async,
    is_markitdown_model,
    markitdown_model_visible,
    preprocess_markitdown_file_parts_async,
    request_has_file_parts,
    stream_messages_to_markdown_async,
)
from omlx_runtime.exceptions import (
    EnginePoolError,
    InsufficientMemoryError,
    ModelBusyError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelTooLargeError,
    ModelUnavailableError,
)

from omlx_server.auth import require_management_key

from .dependencies import verify_api_key, verify_inference_api_key
from .shared import _KEEPALIVE_CHAT_CHUNK
from .transport import (
    _chat_keepalive_chunk,
    _json_response_or_keepalive,
    _with_sse_keepalive,
)

logger = logging.getLogger(__name__)


def _ane_prefill_status(pool) -> dict:
    if pool is None:
        return {"patch_available": False, "configured_models": 0, "models": []}
    return pool.get_ane_prefill_status()


def _markitdown_virtual_model_status() -> dict:
    return {
        "id": MARKITDOWN_MODEL_ID,
        "model_path": "builtin://markitdown",
        "loaded": True,
        "is_loading": False,
        "loading_started_at": None,
        "estimated_size": 0,
        "actual_size": 0,
        "pinned": False,
        "engine_type": "markitdown",
        "model_type": "markitdown",
        "config_model_type": "markitdown",
        "thinking_default": None,
        "preserve_thinking_default": None,
        "source_type": "builtin",
        "source_repo_id": None,
        "last_access": None,
    }


def _build_markitdown_chat_response(
    request: ChatCompletionRequest,
    markdown: str,
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        model=request.model,
        choices=[
            ChatCompletionChoice(
                message=AssistantMessage(content=markdown),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )


async def _stream_markitdown_chat_response(
    request: ChatCompletionRequest,
    markdown_chunks: AsyncIterator[str],
    response_id: str | None = None,
) -> AsyncIterator[str]:
    response_id = response_id or f"chatcmpl-{uuid.uuid4().hex[:8]}"
    role_chunk = ChatCompletionChunk(
        id=response_id,
        model=request.model,
        choices=[
            ChatCompletionChunkChoice(
                delta=ChatCompletionChunkDelta(role="assistant"),
            )
        ],
    )
    yield f"data: {role_chunk.model_dump_json(exclude_none=True)}\n\n"

    emitted = False
    async for markdown in markdown_chunks:
        if not markdown:
            continue
        emitted = True
        content_chunk = ChatCompletionChunk(
            id=response_id,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(content=markdown),
                )
            ],
        )
        yield f"data: {content_chunk.model_dump_json(exclude_none=True)}\n\n"

    if not emitted:
        raise MarkItDownRequestError(
            "No text or supported file content found for MarkItDown.",
            status_code=400,
        )

    final_chunk = ChatCompletionChunk(
        id=response_id,
        model=request.model,
        choices=[
            ChatCompletionChunkChoice(
                delta=ChatCompletionChunkDelta(),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"

    if request.stream_options and request.stream_options.include_usage:
        usage_chunk = ChatCompletionChunk(
            id=response_id,
            model=request.model,
            choices=[],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
        yield f"data: {usage_chunk.model_dump_json(exclude_none=True)}\n\n"

    yield "data: [DONE]\n\n"


class InventoryController:
    async def health(self, response: Response):
        """Health check endpoint.

        Answers 503 with status "loading" while the startup pinned-model
        preload is still running: the port is already bound (liveness for
        watchdogs, #2184) but the server is not ready to serve those models.
        """
        mcp_info = None
        if self.state.mcp_manager is not None:
            connected = sum(
                1
                for s in self.state.mcp_manager.get_server_status()
                if s.state.value == "connected"
            )
            total = len(self.state.mcp_manager.get_server_status())
            mcp_info = {
                "enabled": True,
                "servers_connected": connected,
                "servers_total": total,
                "tools_available": len(self.state.mcp_manager.get_all_tools()),
            }

        pool_status = None
        if self.state.engine_pool is not None:
            enforcer = self.state.process_memory_enforcer
            ceiling = 0
            if enforcer is not None:
                try:
                    ceiling = enforcer.get_final_ceiling()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Health memory ceiling unavailable: %s", exc)
            pool_status = {
                "model_count": self.state.engine_pool.model_count,
                "loaded_count": self.state.engine_pool.loaded_model_count,
                "final_ceiling": ceiling,
                "current_model_memory": self.state.engine_pool.current_model_memory,
            }

        loading = not self.state.pinned_preload_complete
        if loading:
            response.status_code = 503
        return {
            "status": "loading" if loading else "healthy",
            "default_model": self.state.default_model,
            "engine_pool": pool_status,
            "mcp": mcp_info,
        }

    async def server_status(self, _: bool = Depends(verify_api_key)):
        """Lightweight status endpoint for external tool polling (statuslines, scripts)."""
        from omlx_runtime.custom_kernels import native_kernel_status
        from omlx_runtime.model_discovery import format_size

        metrics = self.get_server_metrics()
        snapshot = metrics.get_snapshot()

        pool = self.state.engine_pool

        models_discovered = 0
        models_loaded = 0
        models_loading = 0
        loaded_models = []
        model_memory_used = 0
        model_memory_max = None

        if pool is not None:
            models_discovered = pool.model_count
            models_loaded = pool.loaded_model_count
            loaded_models = pool.get_loaded_model_ids()
            model_memory_used = pool.current_model_memory
            enforcer = self.state.process_memory_enforcer
            if enforcer is not None:
                try:
                    model_memory_max = enforcer.get_final_ceiling()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Status memory ceiling unavailable: %s", exc)
            models_loading = sum(
                view.is_loading
                for model_id in pool.get_model_ids()
                if (view := pool.get_model_view(model_id)) is not None
            )

        active_requests, waiting_requests = (
            pool.get_request_counts() if pool is not None else (0, 0)
        )

        return {
            "status": "ok",
            "version": __version__,
            "uptime_seconds": snapshot["uptime_seconds"],
            "models_discovered": models_discovered,
            "models_loaded": models_loaded,
            "models_loading": models_loading,
            "default_model": self.state.default_model,
            "loaded_models": loaded_models,
            "total_requests": snapshot["total_requests"],
            "active_requests": active_requests,
            "waiting_requests": waiting_requests,
            "total_prompt_tokens": snapshot["total_prompt_tokens"],
            "total_completion_tokens": snapshot["total_completion_tokens"],
            "total_cached_tokens": snapshot["total_cached_tokens"],
            "cache_efficiency": snapshot["cache_efficiency"],
            "avg_prefill_tps": snapshot["avg_prefill_tps"],
            "avg_generation_tps": snapshot["avg_generation_tps"],
            "model_memory_used": model_memory_used,
            "model_memory_max": model_memory_max,
            "model_memory_used_formatted": (
                format_size(model_memory_used) if model_memory_used else "0B"
            ),
            "model_memory_max_formatted": (
                format_size(model_memory_max) if model_memory_max else "unlimited"
            ),
            "custom_kernels": native_kernel_status(),
            "ane_prefill": _ane_prefill_status(pool),
        }

    def _markitdown_is_visible(self) -> bool:
        return markitdown_model_visible(self.state.global_settings)

    def _with_markitdown_status(self, status: dict) -> dict:
        if not self._markitdown_is_visible():
            return status

        augmented = dict(status)
        models = list(augmented.get("models", []))
        if not any(m.get("id") == MARKITDOWN_MODEL_ID for m in models):
            models.append(_markitdown_virtual_model_status())
        augmented["models"] = models
        augmented["model_count"] = len(models)
        augmented["loaded_count"] = sum(1 for m in models if m.get("loaded"))
        return augmented

    def _with_exposed_profile_status(self, status: dict) -> dict:
        settings_manager = self.state.settings_manager
        if settings_manager is None:
            return status

        list_profiles = getattr(settings_manager, "list_exposed_profile_models", None)
        if not callable(list_profiles):
            return status

        augmented = dict(status)
        models = [dict(m) for m in augmented.get("models", [])]
        by_id = {m.get("id"): m for m in models}
        existing_ids = set(by_id)
        for profile in list_profiles():
            source_model_id = profile.get("source_model_id")
            profile_model_id = profile.get("model_id")
            if (
                not source_model_id
                or not profile_model_id
                or source_model_id not in by_id
                or profile_model_id in existing_ids
            ):
                continue
            profile_status = dict(by_id[source_model_id])
            profile_status.update(
                {
                    "id": profile_model_id,
                    "source_model_id": source_model_id,
                    "profile_name": profile.get("name"),
                    "profile_api_name": profile.get("api_name"),
                    "profile_display_name": profile.get("display_name"),
                }
            )
            models.append(profile_status)
            existing_ids.add(profile_model_id)

        augmented["models"] = models
        augmented["model_count"] = len(models)
        augmented["loaded_count"] = sum(1 for m in models if m.get("loaded"))
        return augmented

    async def _preprocess_markitdown_files_for_llm(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionRequest:
        if not request_has_file_parts(request.messages):
            return request

        try:
            messages = await preprocess_markitdown_file_parts_async(
                request.messages,
                global_settings=self.state.global_settings,
                engine_pool=self.state.engine_pool,
                settings_manager=self.state.settings_manager,
                get_sampling_params=self.get_sampling_params,
                fail_when_disabled=True,
                allow_missing_historical_files=True,
            )
        except MarkItDownRequestError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return request.model_copy(update={"messages": messages})

    async def _create_markitdown_chat_completion(
        self,
        request: ChatCompletionRequest,
        http_request: FastAPIRequest,
    ):
        if not self._markitdown_is_visible():
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {MARKITDOWN_MODEL_ID}",
            )

        if request.stream:
            response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            keepalive = self._resolve_keepalive("openai_chat")
            if keepalive == _KEEPALIVE_CHAT_CHUNK:
                keepalive = _chat_keepalive_chunk(response_id)
            markdown_chunks = stream_messages_to_markdown_async(
                request.messages,
                global_settings=self.state.global_settings,
                engine_pool=self.state.engine_pool,
                settings_manager=self.state.settings_manager,
                get_sampling_params=self.get_sampling_params,
                latest_user_only=True,
            )
            return StreamingResponse(
                _with_sse_keepalive(
                    _stream_markitdown_chat_response(
                        request,
                        markdown_chunks,
                        response_id=response_id,
                    ),
                    http_request=http_request,
                    keepalive_chunk=keepalive,
                ),
                media_type="text/event-stream",
                headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
            )

        async def _build_markitdown_completion():
            try:
                markdown = await convert_messages_to_markdown_async(
                    request.messages,
                    global_settings=self.state.global_settings,
                    engine_pool=self.state.engine_pool,
                    settings_manager=self.state.settings_manager,
                    get_sampling_params=self.get_sampling_params,
                    latest_user_only=True,
                )
            except MarkItDownRequestError as exc:
                raise HTTPException(
                    status_code=exc.status_code, detail=exc.detail
                ) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

            if not markdown:
                raise HTTPException(
                    status_code=400,
                    detail="No text or supported file content found for MarkItDown.",
                )

            logger.info("MarkItDown completion converted request to markdown")
            return _build_markitdown_chat_response(
                request,
                markdown,
            ).model_dump_json(exclude_none=True)

        return await _json_response_or_keepalive(
            http_request, _build_markitdown_completion()
        )

    async def list_models(
        self, _: bool = Depends(verify_inference_api_key)
    ) -> ModelsResponse:
        """List all available models with load status."""
        models = []
        favorite_ids: set[str] = set()

        if self.state.engine_pool is not None:
            status = self.state.engine_pool.get_status()
            settings_manager = self.state.settings_manager

            hide_helpers = bool(
                self.state.global_settings is not None
                and self.state.global_settings.model.hide_helper_models
            )
            # Set of draft-model references (paths / repo ids) pointed at by other
            # models' speculative settings — used to flag "helper" drafters that
            # only differ from a chat model by being referenced elsewhere.
            referenced_drafts: set[str] = set()
            if hide_helpers and settings_manager:
                for _ms in settings_manager.get_all_settings().values():
                    for ref in (
                        _ms.specprefill_draft_model,
                        _ms.dflash_draft_model,
                        _ms.vlm_mtp_draft_model,
                    ):
                        if ref:
                            referenced_drafts.add(ref)

            excluded_model_ids: set[str] = set()
            for m in status["models"]:
                model_id = m["id"]
                display_id = model_id
                ms = None
                if settings_manager:
                    ms = settings_manager.get_settings(model_id)
                    if ms.model_alias:
                        display_id = ms.model_alias
                # Per-model hide: user-selected, always applied.
                is_hidden = ms is not None and ms.is_hidden
                # Global helper hide: skip drafters when the toggle is on. A model
                # is a drafter if intrinsically flagged at discovery (config marker)
                # or referenced as another model's draft.
                is_hidden_helper = hide_helpers and (
                    m.get("is_helper")
                    or model_id in referenced_drafts
                    or m.get("model_path") in referenced_drafts
                    or (m.get("source_repo_id") in referenced_drafts)
                )
                if is_hidden or is_hidden_helper:
                    excluded_model_ids.add(model_id)
                    continue
                if ms is not None and ms.is_favorite:
                    favorite_ids.add(display_id)
                models.append(
                    ModelInfo(
                        id=display_id,
                        owned_by="omlx",
                        max_model_len=self.get_max_context_window(model_id),
                    )
                )
            if settings_manager:
                physical_ids = {m["id"] for m in status["models"]}
                existing_ids = {m.id for m in models}
                for profile in settings_manager.list_exposed_profile_models():
                    source_model_id = profile["source_model_id"]
                    profile_model_id = profile["model_id"]
                    if (
                        source_model_id not in physical_ids
                        or source_model_id in excluded_model_ids
                        or profile_model_id in existing_ids
                    ):
                        continue
                    models.append(
                        ModelInfo(
                            id=profile_model_id,
                            owned_by="omlx",
                            max_model_len=self.get_max_context_window(profile_model_id),
                        )
                    )
                    existing_ids.add(profile_model_id)

        if self._markitdown_is_visible() and not any(
            m.id == MARKITDOWN_MODEL_ID for m in models
        ):
            models.append(ModelInfo(id=MARKITDOWN_MODEL_ID, owned_by="omlx"))

        # Favorites first; stable sort keeps alphabetical order within groups.
        if favorite_ids:
            models.sort(key=lambda m: m.id not in favorite_ids)

        return ModelsResponse(data=models)

    async def list_models_status(self, _: bool = Depends(verify_api_key)):
        """
        List all available models with detailed status.

        Extended endpoint that provides more information than /v1/models.
        """
        if self.state.engine_pool is None:
            raise HTTPException(status_code=503, detail="Server not initialized")

        status = self._with_exposed_profile_status(
            self._with_markitdown_status(self.state.engine_pool.get_status())
        )
        for m in status["models"]:
            model_id = m["id"]
            if is_markitdown_model(model_id):
                m["max_context_window"] = None
                m["max_tokens"] = None
                m["is_favorite"] = False
                m["is_hidden"] = False
                continue

            m["max_context_window"] = self.get_max_context_window(model_id)
            source_model_id = m.get("source_model_id") or model_id

            # Resolve effective max_tokens: model setting > global default
            max_tokens = self.state.sampling.max_tokens
            if self.state.settings_manager:
                sm = self.state.settings_manager
                if hasattr(sm, "get_settings_for_request"):
                    ms = sm.get_settings_for_request(
                        model_id,
                        resolved_model_id=source_model_id,
                    )
                else:
                    ms = sm.get_settings(source_model_id)
                base_ms = sm.get_settings(source_model_id)
                if base_ms and base_ms.model_alias and source_model_id == model_id:
                    m["model_alias"] = base_ms.model_alias
                m["is_favorite"] = base_ms is not None and base_ms.is_favorite
                m["is_hidden"] = base_ms is not None and base_ms.is_hidden
                if ms and ms.max_tokens is not None:
                    max_tokens = ms.max_tokens
            else:
                m["is_favorite"] = False
                m["is_hidden"] = False
            m["max_tokens"] = max_tokens
        return status

    async def unload_model(
        self, model_id: str, _: bool = Depends(require_management_key)
    ):
        """Manually unload a model from memory."""
        if self.state.engine_pool is None:
            raise HTTPException(status_code=503, detail="Server not initialized")

        entry = self.state.engine_pool.get_model_view(model_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
        if not entry.loaded:
            raise HTTPException(status_code=400, detail=f"Model not loaded: {model_id}")

        await self.state.engine_pool.unload_engine(model_id)
        return {"status": "ok", "model_id": model_id}

    async def load_model_public(self, model_id: str, _: bool = Depends(verify_api_key)):
        """Load a discovered model into memory. Blocks until loading completes."""
        if self.state.engine_pool is None:
            raise HTTPException(status_code=503, detail="Server not initialized")

        entry = self.state.engine_pool.get_model_view(model_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
        if entry.loaded:
            return {
                "status": "ok",
                "model_id": model_id,
                "message": f"Already loaded: {model_id}",
            }

        try:
            await self.state.engine_pool.get_engine(model_id)
        except ModelNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except ModelTooLargeError as e:
            raise HTTPException(status_code=507, detail=str(e)) from e
        except InsufficientMemoryError as e:
            raise HTTPException(status_code=507, detail=str(e)) from e
        except ModelUnavailableError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        except ModelLoadingError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        except ModelBusyError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        except EnginePoolError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

        return {"status": "ok", "model_id": model_id, "message": f"Loaded {model_id}"}
