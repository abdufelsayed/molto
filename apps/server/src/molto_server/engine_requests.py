# SPDX-License-Identifier: Apache-2.0
"""Engine requests for the private inference application."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import HTTPException
from molto_runtime.engine import BaseEngine
from molto_runtime.engine.embedding import EmbeddingEngine
from molto_runtime.engine.reranker import RerankerEngine
from molto_runtime.exceptions import (
    EnginePoolError,
    InsufficientMemoryError,
    ModelBusyError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelTooLargeError,
    ModelUnavailableError,
)

# Import from new modular API
from molto_runtime.generation.utils import has_nonleading_system_message

from .shared import _LLMEngineLease
from .state import EngineType

logger = logging.getLogger(__name__)


def _suggest_endpoint_for_engine(engine: object) -> str:
    """Return a one-line hint pointing at the correct endpoint for a non-LLM engine."""
    # Import audio engine classes lazily so that Molto without the [audio]
    # extra still imports this module.
    try:
        from molto_runtime.engine.stt import STTEngine as stt_engine_cls
    except Exception:  # pragma: no cover - defensive
        stt_engine_cls = None
    try:
        from molto_runtime.engine.tts import TTSEngine as tts_engine_cls
    except Exception:  # pragma: no cover - defensive
        tts_engine_cls = None
    try:
        from molto_runtime.engine.sts import STSEngine as sts_engine_cls
    except Exception:  # pragma: no cover - defensive
        sts_engine_cls = None
    try:
        from molto_runtime.engine.image_generation import (
            MFluxImageEngine as image_engine_cls,
        )
    except Exception:  # pragma: no cover - defensive
        image_engine_cls = None

    if stt_engine_cls is not None and isinstance(engine, stt_engine_cls):
        return "Use /v1/audio/transcriptions for speech-to-text models."
    if tts_engine_cls is not None and isinstance(engine, tts_engine_cls):
        return "Use /v1/audio/speech for text-to-speech models."
    if sts_engine_cls is not None and isinstance(engine, sts_engine_cls):
        return "Use /v1/audio/process for speech-to-speech / audio processing models."
    if image_engine_cls is not None and isinstance(engine, image_engine_cls):
        return "Use /v1/images/generations, /v1/images/edits or /v1/images/operations for image models."
    if isinstance(engine, EmbeddingEngine):
        return "Use /v1/embeddings for embedding models."
    if isinstance(engine, RerankerEngine):
        return "Use /v1/rerank for reranker models."
    return "Use the model's dedicated endpoint (see /v1/models)."


async def _raise_if_llm_lease_abort_requested(lease: _LLMEngineLease) -> None:
    reason = lease.abort_reason()
    if reason == "manual management unload":
        raise HTTPException(
            status_code=409,
            detail="Request aborted because this model is being unloaded.",
        )
    if reason is not None:
        raise HTTPException(
            status_code=507,
            detail=(
                "Request aborted before scheduling because process memory "
                "pressure requested this model to unload. Retry with a shorter "
                "context or after memory pressure drops."
            ),
        )


async def _release_after_stream(
    generator: AsyncIterator[str],
    lease: _LLMEngineLease,
) -> AsyncIterator[str]:
    try:
        await _raise_if_llm_lease_abort_requested(lease)
        async for chunk in generator:
            yield chunk
    finally:
        await lease.release()


def _strip_synthetic_think_prefix(chunk_text: str, think_tag: str) -> str:
    """Drop the scheduler's synthetic think opener from a raw completions chunk.

    Raw completions are a pure continuation of the prompt. When the prompt
    itself ends with an open think tag, the scheduler still prepends a
    synthetic ``"<think>\\n"`` to the first streamed chunk (chat streams rely
    on it to rebuild the reasoning block), but the opener belongs to the
    prompt and the non-streaming completions path never returns it. Stripping
    it keeps both completion paths returning the same continuation.
    """
    prefix = f"{think_tag}\n"
    return chunk_text[len(prefix) :] if chunk_text.startswith(prefix) else chunk_text


async def _ensure_tokenizer_for_system_probe(
    engine: BaseEngine, messages: list
) -> None:
    """Load lazy engines before probing mid-conversation system placement."""
    if not has_nonleading_system_message(messages):
        return
    if getattr(engine, "tokenizer", None) is not None:
        return
    await engine.start()


def _format_generation_speed_for_log(
    output,
    tokens_per_sec: float,
    *,
    is_diffusion: bool,
) -> str:
    if not is_diffusion:
        return f"{tokens_per_sec:.1f} tok/s"

    parts = [f"{tokens_per_sec:.1f} tok/s e2e"]
    output_tps = float(getattr(output, "generation_tps", 0.0) or 0.0)
    if output_tps > 0:
        parts.append(f"output={output_tps:.1f} tok/s")
    canvas_tps = float(getattr(output, "diffusion_canvas_tps", 0.0) or 0.0)
    if canvas_tps > 0:
        parts.append(f"canvas={canvas_tps:.1f} tok/s")
    prompt_tps = float(getattr(output, "prompt_tps", 0.0) or 0.0)
    if prompt_tps > 0:
        parts.append(f"prompt={prompt_tps:.1f} tok/s")
    work_tps = float(getattr(output, "diffusion_work_tps", 0.0) or 0.0)
    if work_tps > 0:
        parts.append(f"work={work_tps:.1f} tok/s")
    steps = int(getattr(output, "diffusion_denoising_steps", 0) or 0)
    if steps > 0:
        parts.append(f"steps={steps}")
    return ", ".join(parts)


def _resolve_metric_durations(
    output,
    *,
    is_diffusion: bool,
    prefill_duration: float = 0.0,
    generation_duration: float = 0.0,
) -> tuple[float, float]:
    if not is_diffusion:
        return prefill_duration, generation_duration

    prompt_tps = float(getattr(output, "prompt_tps", 0.0) or 0.0)
    if prompt_tps > 0:
        prefill_duration = output.prompt_tokens / prompt_tps

    generation_tps = float(getattr(output, "generation_tps", 0.0) or 0.0)
    if generation_tps > 0:
        generation_duration = output.completion_tokens / generation_tps

    return prefill_duration, generation_duration


class EngineRequestsController:
    def _wake_process_memory_enforcer(self, *, active: bool = False) -> None:
        enforcer = self.state.process_memory_enforcer
        wake = getattr(enforcer, "wake", None) if enforcer is not None else None
        if callable(wake):
            wake(active=active)

    async def get_engine(
        self,
        model_id: str | None = None,
        engine_type: EngineType = EngineType.LLM,
        _lease: bool = False,
        _leased_out: list | None = None,
    ) -> BaseEngine | EmbeddingEngine | RerankerEngine:
        """
        Get engine for the specified model and type.

        This is the unified engine getter that handles LLM, embedding, and reranker models.

        Args:
            model_id: Model ID to get engine for, or None for default (LLM only)
            engine_type: Type of engine to retrieve (LLM, EMBEDDING, or RERANKER)
            _lease: When True, take an atomic in-use lease on the engine that the
                pool actually loaded (eviction-proof until released). The caller
                MUST release exactly one lease per successful leased call.
            _leased_out: When _lease is True, the EXACT pool model_id that was
                leased is appended to this list. Release using that id (not the
                request model) so the lease/release ids always match even when the
                pool falls back to the default model.

        Returns:
            The loaded engine of the appropriate type

        Raises:
            HTTPException: If model not found, wrong type, or memory error
        """
        pool = self.get_engine_pool()

        # Default model only applies to LLM
        if model_id is None:
            if engine_type != EngineType.LLM:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model ID is required for {engine_type.value} engines",
                )
            model_id = self.state.default_model

        if model_id is None:
            raise HTTPException(
                status_code=400, detail="No model specified and no default model set"
            )

        # Resolve alias/profile request to the physical model. Exposed profiles
        # may carry engine-construction settings (MTP/DFlash/etc.); pass those
        # transient settings to the pool so the loaded variant can switch without
        # mutating the base model's persisted settings.
        requested_model_id = model_id
        runtime_settings = None
        sm = self.state.settings_manager
        if (
            engine_type == EngineType.LLM
            and sm is not None
            and hasattr(sm, "get_exposed_profile_runtime_settings_for_request")
        ):
            runtime = sm.get_exposed_profile_runtime_settings_for_request(
                requested_model_id
            )
            if runtime is not None:
                model_id, runtime_settings = runtime
            else:
                model_id = pool.resolve_model_id(model_id, sm)
        else:
            model_id = pool.resolve_model_id(model_id, sm)
        self._wake_process_memory_enforcer(active=True)

        # Only thread optional kwargs through when they are needed, so the common
        # path keeps the original pool.get_engine(model_id) call shape.
        _lease_kwargs = {"_lease": True} if _lease else {}
        if runtime_settings is not None:
            _lease_kwargs["runtime_settings"] = runtime_settings
        try:
            engine = await pool.get_engine(model_id, **_lease_kwargs)
            if _lease and _leased_out is not None:
                _leased_out.append(model_id)
        except ModelNotFoundError as e:
            # Fallback to default model if enabled (LLM only)
            if (
                engine_type == EngineType.LLM
                and self.state.global_settings
                and self.state.global_settings.model.model_fallback
                and self.state.default_model
            ):
                logger.info(
                    f"Model '{model_id}' not found, falling back to "
                    f"default model '{self.state.default_model}'"
                )
                try:
                    self._wake_process_memory_enforcer(active=True)
                    _fallback_kwargs = {"_lease": True} if _lease else {}
                    fb_engine = await pool.get_engine(
                        self.state.default_model, **_fallback_kwargs
                    )
                    if _lease and _leased_out is not None:
                        _leased_out.append(self.state.default_model)
                    return fb_engine
                except Exception:
                    pass  # Fall through to original 404

            # Show aliases instead of directory names for user-friendly display
            available = e.available_models
            sm = self.state.settings_manager
            if sm:
                display = []
                for mid in available:
                    ms = sm.get_settings(mid)
                    display.append(ms.model_alias if ms.model_alias else mid)
                available = display
            detail = (
                f"Model '{model_id}' not found. "
                f"Available models: {', '.join(available) if available else '(none)'}"
            )
            raise HTTPException(status_code=404, detail=detail)
        except ModelTooLargeError as e:
            raise HTTPException(status_code=507, detail=str(e))
        except InsufficientMemoryError as e:
            raise HTTPException(status_code=507, detail=str(e))
        except ModelUnavailableError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        except ModelLoadingError as e:
            raise HTTPException(status_code=409, detail=str(e))
        except ModelBusyError as e:
            raise HTTPException(status_code=409, detail=str(e))
        except EnginePoolError as e:
            raise HTTPException(status_code=500, detail=str(e))

        # Validate engine type. If a lease was taken above but validation fails,
        # release it before raising so a rejected request never leaks an in_use
        # count (which would pin the engine non-evictable forever).
        try:
            if engine_type == EngineType.EMBEDDING:
                if not isinstance(engine, EmbeddingEngine):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Model '{model_id}' is not an embedding model. "
                        f"Use /v1/chat/completions for LLM models.",
                    )
            elif engine_type == EngineType.RERANKER:
                if not isinstance(engine, RerankerEngine):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Model '{model_id}' is not a reranker model. "
                        f"Use a SequenceClassification model for reranking.",
                    )
            elif engine_type == EngineType.LLM:
                # #507: non-LLM engines (STT/TTS/STS/Embedding/Reranker) previously
                # fell through and crashed on `engine.model_type` with an unhandled
                # 500. Reject with a clear 400 pointing the caller at the right
                # endpoint.
                if not isinstance(engine, BaseEngine):
                    _endpoint_hint = _suggest_endpoint_for_engine(engine)
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Model '{model_id}' is not an LLM / chat model. "
                            f"{_endpoint_hint}"
                        ),
                    )
        except BaseException:
            if _lease and _leased_out:
                await pool.release_engine(_leased_out.pop())
            raise

        return engine

    async def get_engine_for_model(
        self,
        model: str | None = None,
        *,
        lease: _LLMEngineLease | None = None,
    ) -> BaseEngine:
        """
        Get LLM engine for the specified model (or default).

        This is a convenience wrapper around get_engine() for LLM models.

        Args:
            model: Model ID to get engine for, or None for default

        Returns:
            The loaded engine

        Raises:
            HTTPException: If model not found or memory error
        """
        if lease is None:
            return await self.get_engine(model, EngineType.LLM)

        leased: list[str] = []
        engine = await self.get_engine(
            model,
            EngineType.LLM,
            _lease=True,
            _leased_out=leased,
        )
        if leased:
            lease.model_id = leased[0]
        return engine

    def _serving_model_id(
        self,
        lease: _LLMEngineLease,
        requested_model: str | None,
    ) -> str | None:
        """Return the exact pool model selected for an LLM request."""
        return (
            lease.model_id or self.resolve_model_id(requested_model) or requested_model
        )

    async def get_embedding_engine(self, model: str) -> EmbeddingEngine:
        """
        Get embedding engine for the specified model.

        This is a convenience wrapper around get_engine() for embedding models.

        Args:
            model: Model ID to get engine for

        Returns:
            The loaded embedding engine

        Raises:
            HTTPException: If model not found, is not an embedding model, or memory error
        """
        return await self.get_engine(model, EngineType.EMBEDDING)

    async def get_reranker_engine(self, model: str) -> RerankerEngine:
        """
        Get reranker engine for the specified model.

        This is a convenience wrapper around get_engine() for reranker models.

        Args:
            model: Model ID to get engine for

        Returns:
            The loaded reranker engine

        Raises:
            HTTPException: If model not found, is not a reranker model, or memory error
        """
        return await self.get_engine(model, EngineType.RERANKER)

    @asynccontextmanager
    async def acquire_embedding_engine(self, model: str):
        """Acquire an embedding engine with an atomic, eviction-proof in-use lease.

        Resolves + loads + validates exactly like get_embedding_engine, but holds
        the engine non-evictable for the duration of the request and releases the
        lease on the EXACT pool model_id the pool loaded (handles default-model
        fallback) in finally.
        """
        leased: list = []
        engine = await self.get_engine(
            model, EngineType.EMBEDDING, _lease=True, _leased_out=leased
        )
        try:
            yield engine
        finally:
            if leased:
                await self.get_engine_pool().release_engine(leased[0])

    @asynccontextmanager
    async def acquire_reranker_engine(self, model: str):
        """Acquire a reranker engine with an atomic, eviction-proof in-use lease.

        See acquire_embedding_engine for the lease/release contract.
        """
        leased: list = []
        engine = await self.get_engine(
            model, EngineType.RERANKER, _lease=True, _leased_out=leased
        )
        try:
            yield engine
        finally:
            if leased:
                await self.get_engine_pool().release_engine(leased[0])

    def get_sampling_params(
        self,
        req_temperature: float | None,
        req_top_p: float | None,
        model_id: str | None = None,
        req_top_k: int | None = None,
        req_repetition_penalty: float | None = None,
        req_min_p: float | None = None,
        req_presence_penalty: float | None = None,
        req_frequency_penalty: float | None = None,
        req_max_tokens: int | None = None,
        ocr_defaults: dict | None = None,
        req_xtc_probability: float | None = None,
        req_xtc_threshold: float | None = None,
        sampling_override: bool = False,
    ) -> tuple[float, float, int, float, float, float, float, int, float, float]:
        """
        Get effective sampling parameters with per-model settings support.

        Priority:
        - If force_sampling is True (global or model level): force sampling knobs
          that affect token selection.
        - max_tokens is an output length cap, so it always uses
          request > model settings > ocr_defaults > global defaults.
        - Otherwise: request > model settings > ocr_defaults > global defaults.

        Returns:
            tuple of (temperature, top_p, top_k, repetition_penalty, min_p, presence_penalty, frequency_penalty, max_tokens, xtc_probability, xtc_threshold)
        """
        global_sampling = self.state.sampling

        # Get per-model (or exposed-profile) settings if available
        model_settings = self.get_model_settings_for_request(model_id)

        # Resolve alias so physical-model defaults can still be found by real model ID
        model_id = self.resolve_model_id(model_id)

        # Resolve OCR defaults if not provided by caller
        if ocr_defaults is None and model_id:
            ocr_defaults = self._get_ocr_defaults(model_id)

        # Check force at any level
        force = not sampling_override and (
            global_sampling.force_sampling
            or (model_settings and model_settings.force_sampling)
        )

        if force:
            # Forced mode: use model settings if available, else global
            if model_settings and model_settings.temperature is not None:
                temperature = model_settings.temperature
            elif ocr_defaults and "temperature" in ocr_defaults:
                temperature = ocr_defaults["temperature"]
            else:
                temperature = global_sampling.temperature

            if model_settings and model_settings.top_p is not None:
                top_p = model_settings.top_p
            else:
                top_p = global_sampling.top_p

            if model_settings and model_settings.top_k is not None:
                top_k = model_settings.top_k
            else:
                top_k = global_sampling.top_k
        else:
            # Normal mode: priority request > model > ocr_defaults > global
            if req_temperature is not None:
                temperature = req_temperature
            elif model_settings and model_settings.temperature is not None:
                temperature = model_settings.temperature
            elif ocr_defaults and "temperature" in ocr_defaults:
                temperature = ocr_defaults["temperature"]
            else:
                temperature = global_sampling.temperature

            if req_top_p is not None:
                top_p = req_top_p
            elif model_settings and model_settings.top_p is not None:
                top_p = model_settings.top_p
            else:
                top_p = global_sampling.top_p

            if req_top_k is not None:
                top_k = req_top_k
            elif model_settings and model_settings.top_k is not None:
                top_k = model_settings.top_k
            elif ocr_defaults and "top_k" in ocr_defaults:
                top_k = ocr_defaults["top_k"]
            else:
                top_k = global_sampling.top_k

        # Repetition penalty: request > model settings > ocr_defaults > global (1.0)
        if req_repetition_penalty is not None:
            repetition_penalty = req_repetition_penalty
        elif model_settings and model_settings.repetition_penalty is not None:
            repetition_penalty = model_settings.repetition_penalty
        elif ocr_defaults and "repetition_penalty" in ocr_defaults:
            repetition_penalty = ocr_defaults["repetition_penalty"]
        else:
            repetition_penalty = getattr(global_sampling, "repetition_penalty", 1.0)

        # Min P: request > model settings > default (0.0)
        if req_min_p is not None:
            min_p = req_min_p
        elif model_settings and getattr(model_settings, "min_p", None) is not None:
            min_p = model_settings.min_p
        else:
            min_p = 0.0

        # Presence penalty: request > model settings > default (0.0)
        if req_presence_penalty is not None:
            presence_penalty = req_presence_penalty
        elif (
            model_settings
            and getattr(model_settings, "presence_penalty", None) is not None
        ):
            presence_penalty = model_settings.presence_penalty
        else:
            presence_penalty = 0.0

        # Frequency penalty: request > model settings > default (0.0)
        if req_frequency_penalty is not None:
            frequency_penalty = req_frequency_penalty
        elif (
            model_settings
            and getattr(model_settings, "frequency_penalty", None) is not None
        ):
            frequency_penalty = model_settings.frequency_penalty
        else:
            frequency_penalty = 0.0

        # Max tokens is an output length cap, not a sampling knob. Honor request
        # bounds even when force_sampling pins token-selection parameters.
        if req_max_tokens is not None:
            max_tokens = req_max_tokens
        elif model_settings and model_settings.max_tokens is not None:
            max_tokens = model_settings.max_tokens
        elif ocr_defaults and "max_tokens" in ocr_defaults:
            max_tokens = ocr_defaults["max_tokens"]
        else:
            max_tokens = global_sampling.max_tokens

        # XTC probability: request > default (0.0 = disabled)
        xtc_probability = (
            req_xtc_probability if req_xtc_probability is not None else 0.0
        )

        # XTC threshold: request > default (0.1 = safe default when probability is set)
        xtc_threshold = req_xtc_threshold if req_xtc_threshold is not None else 0.1

        logger.debug(
            f"Sampling params: temperature={temperature}, top_p={top_p}, top_k={top_k}, "
            f"repetition_penalty={repetition_penalty}, min_p={min_p}, presence_penalty={presence_penalty}, "
            f"frequency_penalty={frequency_penalty}, max_tokens={max_tokens}, "
            f"xtc_probability={xtc_probability}, xtc_threshold={xtc_threshold}"
            f"{' (forced)' if force else ''}"
            f"{f' (model: {model_id})' if model_id else ''}"
        )
        return (
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            min_p,
            presence_penalty,
            frequency_penalty,
            max_tokens,
            xtc_probability,
            xtc_threshold,
        )

    def _resolve_thinking_budget(self, request, model_id: str | None) -> int | None:
        """Resolve thinking budget: request param > model settings > None."""
        # Check request-level override (OpenAI format)
        req_budget = getattr(request, "thinking_budget", None)
        # For Anthropic: check thinking.budget_tokens
        if req_budget is None and hasattr(request, "thinking") and request.thinking:
            req_budget = getattr(request.thinking, "budget_tokens", None)
        if req_budget is not None:
            return req_budget
        ms = self.get_model_settings_for_request(model_id)
        if ms and ms.thinking_budget_enabled and ms.thinking_budget_tokens:
            return ms.thinking_budget_tokens
        return None

    def get_model_settings_for_request(self, model_id: str | None):
        """Return settings for the requested API model name via ModelSettingsManager."""
        sm = self.state.settings_manager
        if not model_id or sm is None:
            return None

        resolved_model_id = self.resolve_model_id(model_id)
        if not hasattr(sm, "get_settings_for_request"):
            return sm.get_settings(resolved_model_id or model_id)

        return sm.get_settings_for_request(
            model_id,
            resolved_model_id=resolved_model_id,
        )

    def resolve_model_id(self, model_id: str | None) -> str | None:
        """Resolve a model alias to its real model ID.

        Returns the resolved ID, or the original value if no alias match.
        """
        if model_id is None:
            return None
        pool = self.state.engine_pool
        if pool is None:
            return model_id
        return pool.resolve_model_id(model_id, self.state.settings_manager)

    def _unsupported_mid_system_policy(self) -> str:
        settings = self.state.global_settings
        preserve_cache = True
        if settings is not None:
            preserve_cache = bool(
                getattr(settings.server, "preserve_mid_system_cache", True)
            )
        return "user_note_safe" if preserve_cache else "strict"

    def _get_ocr_defaults(self, model_id: str | None) -> dict | None:
        """Get OCR generation defaults for a model, or None if not an OCR model."""
        if model_id is None:
            return None
        pool = self.state.engine_pool
        if pool is None:
            return None
        entry = pool.get_entry(model_id)
        if entry is None:
            return None
        from molto_runtime.engine.vlm import (
            OCR_MODEL_GENERATION_DEFAULTS,
            OCR_MODEL_TYPES,
        )

        cmt = getattr(entry, "config_model_type", "")
        if cmt in OCR_MODEL_TYPES:
            return OCR_MODEL_GENERATION_DEFAULTS.get(cmt)
        return None

    def get_max_context_window(self, model_id: str | None = None) -> int | None:
        """
        Get effective max context window limit.

        Resolution:
            1. **Per-model override** (management API / model settings) — always
               wins. An operator who has set a per-model number knows what
               they want; ``max_context_window_policy`` does not clamp it.
            2. **Model-config-discovered native context length** (#1308),
               optionally clamped by the operator policy: if
               ``sampling.max_context_window_policy`` is set, return
               ``min(native, policy)``; otherwise return ``native`` as-is.
            3. **Fallback default** from ``SamplingSettings.max_context_window``
               — only used when neither tier 1 nor tier 2 yields a value.
               Treated as a default, NOT capped by the policy; existing
               ``settings.json`` files carrying the historical ``32768``
               default keep working unchanged after upgrade.

        The policy field is intentionally nullable and unset by default so
        no existing install behavior shifts. Setting it engages
        ``min(native, policy)`` across every model whose native context is
        discoverable; per-model overrides remain the operator's escape
        hatch for individual models that should exceed the policy.

        Returns:
            Max context window token count, or ``None`` if no tier resolves
            (only possible when neither the model nor the global default
            provides a value, which shouldn't happen in practice).
        """
        # Resolve alias for physical model metadata, but keep requested alias settings.
        requested_model_id = model_id
        model_settings = self.get_model_settings_for_request(requested_model_id)
        model_id = self.resolve_model_id(model_id)

        # Priority 1: explicit per-model override (not capped by policy)
        if model_settings and model_settings.max_context_window is not None:
            return model_settings.max_context_window

        # Priority 2: model-native context, optionally clamped by policy
        pool = self.state.engine_pool
        if model_id and pool is not None:
            entry = pool.get_entry(model_id)
            if entry is not None and entry.model_context_length is not None:
                native = entry.model_context_length
                policy = getattr(self.state.sampling, "max_context_window_policy", None)
                if policy is not None and policy > 0:
                    return min(native, policy)
                return native

        # Priority 3: fallback default (not capped — preserves legacy
        # settings.json behavior).
        return self.state.sampling.max_context_window

    def get_embedding_max_length(
        self,
        model_id: str | None = None,
        request_max_length: int | None = None,
    ) -> int | None:
        """Get max token length for embedding requests.

        Returns ``None`` when neither the request nor the server's
        ``max_context_window`` pins a limit, so the embedding model resolves its
        own configured context length (``max_position_embeddings`` / tokenizer
        ``model_max_length`` in ``MLXEmbeddingModel._resolve_max_length``) instead
        of re-truncating long-context models at the legacy 512-token cap (#1687).
        """
        if request_max_length is not None:
            return request_max_length

        return self.get_max_context_window(model_id)

    def validate_context_window(
        self, num_prompt_tokens: int, model_id: str | None = None
    ) -> None:
        """
        Validate that prompt token count does not exceed max context window.

        Raises HTTPException 400 if the prompt is too long.
        """
        max_ctx = self.get_max_context_window(model_id)
        if max_ctx and num_prompt_tokens > max_ctx:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Prompt too long: {num_prompt_tokens} tokens exceeds "
                    f"max context window of {max_ctx} tokens"
                ),
            )
