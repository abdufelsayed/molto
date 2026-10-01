# SPDX-License-Identifier: Apache-2.0
"""Openai for the private inference application."""

import inspect
import json
import logging
import time
import uuid

from fastapi import Depends, HTTPException
from fastapi import Request as FastAPIRequest
from fastapi.responses import StreamingResponse
from molto_config.model_settings import merge_chat_template_request_kwargs
from molto_contracts.api.embedding_models import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)

# Import from new modular API
from molto_contracts.api.openai_models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    PromptTokensDetails,
    Usage,
)
from molto_contracts.api.rerank_models import (
    RerankRequest,
    RerankResponse,
    RerankResult,
    RerankUsage,
)
from molto_runtime.documents.markitdown import is_markitdown_model
from molto_runtime.engine import VLMBatchedEngine
from molto_runtime.exceptions import InvalidRequestError
from molto_runtime.generation.thinking import extract_thinking
from molto_runtime.generation.tool_calling import (
    build_json_system_prompt,
    convert_tools_for_template,
    enrich_tool_params_for_gemma4,
    extract_tool_calls_with_thinking,
    parse_json_output,
    restore_gemma4_param_names,
    sanitize_tool_call_markup,
)
from molto_runtime.generation.utils import (
    cache_reasoning_output,
    clean_special_tokens,
    detect_and_strip_partial,
    extract_multimodal_content,
    extract_text_content,
    merge_reasoning_effort_chat_template_kwargs,
    prepare_system_messages_for_template,
    uses_native_reasoning_content,
)

from molto_server.api.embedding_utils import (
    encode_embedding_base64,
    find_non_finite_embeddings,
    normalize_embedding_items,
    normalize_input,
    truncate_embedding,
)
from molto_server.api.parser_tool_calls import (
    convert_parser_tool_calls as _convert_parser_tool_calls,
)

from .dependencies import verify_inference_api_key
from .engine_requests import (
    _ensure_tokenizer_for_system_probe,
    _format_generation_speed_for_log,
    _raise_if_llm_lease_abort_requested,
    _release_after_stream,
    _resolve_metric_durations,
)
from .openai_streaming import _tool_call_failure
from .shared import (
    _KEEPALIVE_CHAT_CHUNK,
    _KEEPALIVE_COMPLETION_CHUNK,
    _LLMEngineLease,
    _ToolCallGenerationError,
)
from .structured_output import (
    _compile_grammar_for_request,
    _effective_guided_grammar,
    _inject_json_instruction,
    _normalize_structured_outputs,
    _reject_diffusion_structured_outputs,
    _response_format_requests_grammar,
    _response_format_warning_header,
    _settings_guided_grammar,
)
from .transport import (
    _chat_keepalive_chunk,
    _completion_keepalive_chunk,
    _json_response_or_keepalive,
    _request_abort_id,
    _with_request_disconnect_abort,
    _with_sse_keepalive,
)

logger = logging.getLogger(__name__)


def normalize_documents(documents: list[str] | list[dict]) -> list[str]:
    """Normalize document input to list of strings."""
    result = []
    for doc in documents:
        if isinstance(doc, str):
            result.append(doc)
        elif isinstance(doc, dict):
            result.append(doc.get("text", ""))
        else:
            result.append(str(doc))
    return result


class OpenaiController:
    async def create_embeddings(
        self,
        request: EmbeddingRequest,
        http_request: FastAPIRequest,
        _: bool = Depends(verify_inference_api_key),
    ):
        """
        Create embeddings for input text(s).

        OpenAI-compatible endpoint for generating text embeddings.

        Example request:
        ```json
        {
            "model": "all-MiniLM-L6-v2",
            "input": ["Hello, world!", "How are you?"],
            "encoding_format": "float"
        }
        ```

        Supports:
        - Single text or list of texts
        - float or base64 encoding format
        - Optional dimension reduction (with renormalization)
        """
        oq_manager = getattr(self.state, "oq_manager", None)
        if oq_manager and oq_manager.is_quantizing:
            raise HTTPException(
                status_code=503,
                detail="Server is busy with oQ quantization. Please try again after quantization completes.",
            )

        # Validate the model up front (resolves + loads + type-checks) so a bad
        # model still 400/404s before we start the streaming response. The actual
        # eviction-proof lease is taken inside _build_embeddings, which is where
        # the engine is used (the StreamingResponse runs that coroutine later).
        await self.get_embedding_engine(request.model)

        if request.items is not None:
            embedding_inputs = normalize_embedding_items(request.items)
        elif request.input is not None:
            embedding_inputs = normalize_input(request.input)
        else:
            embedding_inputs = []

        if not embedding_inputs:
            raise HTTPException(status_code=400, detail="Input cannot be empty")

        max_length = self.get_embedding_max_length(request.model, request.max_length)

        async def _build_embeddings():
            start_time = time.perf_counter()
            try:
                async with self.acquire_embedding_engine(request.model) as engine:
                    output = await engine.embed(
                        embedding_inputs,
                        max_length=max_length,
                        truncation=request.truncation,
                    )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except TypeError as e:
                raise HTTPException(status_code=400, detail=str(e))

            elapsed = time.perf_counter() - start_time
            resolved_model = self.resolve_model_id(request.model) or request.model

            # A NaN/Inf vector has no JSON representation: FastAPI would ship it
            # as null-filled arrays inside a 200, and a RAG pipeline stores the
            # corrupt vectors without noticing. Fail the request instead.
            non_finite = find_non_finite_embeddings(output.embeddings)
            if non_finite:
                logger.error(
                    f"Embedding: model={resolved_model} returned non-finite values "
                    f"for input item(s) {non_finite} of {len(embedding_inputs)}"
                )
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Embedding model returned non-finite (NaN/Inf) values for "
                        f"input item(s) {non_finite}. The response was rejected "
                        "instead of returning null vectors; try sending the affected "
                        "inputs one per request or batching inputs of equal length."
                    ),
                )

            logger.info(
                f"Embedding: model={resolved_model}, "
                f"{len(embedding_inputs)} inputs, {output.dimensions} dims, "
                f"{output.total_tokens} tokens, max_length={max_length}, "
                f"truncation={request.truncation} in {elapsed:.3f}s"
            )
            self.get_server_metrics().record_request_complete(
                prompt_tokens=output.total_tokens,
                completion_tokens=0,
                cached_tokens=0,
                prefill_duration=elapsed,
                model_id=resolved_model,
                request_duration=elapsed,
            )

            data = []
            for i, embedding in enumerate(output.embeddings):
                if request.dimensions and request.dimensions < len(embedding):
                    embedding = truncate_embedding(embedding, request.dimensions)

                if request.encoding_format == "base64":
                    formatted_embedding = encode_embedding_base64(embedding)
                else:
                    formatted_embedding = embedding

                data.append(
                    EmbeddingData(
                        index=i,
                        embedding=formatted_embedding,
                    )
                )

            return EmbeddingResponse(
                data=data,
                model=request.model,
                usage=EmbeddingUsage(
                    prompt_tokens=output.total_tokens,
                    total_tokens=output.total_tokens,
                ),
            ).model_dump_json()

        return await _json_response_or_keepalive(http_request, _build_embeddings())

    async def create_rerank(
        self,
        request: RerankRequest,
        _: bool = Depends(verify_inference_api_key),
    ) -> RerankResponse:
        """
        Rerank documents by relevance to a query.

        Cohere/Jina-compatible endpoint for document reranking.

        Example request:
        ```json
        {
            "model": "bge-reranker-v2-m3",
            "query": "What is machine learning?",
            "documents": [
                "Machine learning is a subset of AI...",
                "The weather today is sunny...",
                "Deep learning uses neural networks..."
            ],
            "top_n": 2
        }
        ```

        Supports:
        - String documents or dict documents with 'text' field
        - Optional top_n to limit results
        - Optional return_documents to include document text in response
        """
        if self.state.oq_manager and self.state.oq_manager.is_quantizing:
            raise HTTPException(
                status_code=503,
                detail="Server is busy with oQ quantization. Please try again after quantization completes.",
            )

        # Validate the model up front (resolves + loads + type-checks). The
        # eviction-proof lease is held only around the actual rerank() call below.
        await self.get_reranker_engine(request.model)

        # Preserve original structure for the engine (multimodal rerankers need
        # dicts with 'image'), but keep a normalized text view for logging and
        # emptiness checks.
        documents_raw = request.documents
        documents_text = normalize_documents(documents_raw)

        if not documents_text:
            raise HTTPException(status_code=400, detail="Documents cannot be empty")

        if not request.query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Perform reranking
        start_time = time.perf_counter()

        async with self.acquire_reranker_engine(request.model) as engine:
            output = await engine.rerank(
                query=request.query,
                documents=documents_raw,
                top_n=request.top_n,
            )

        elapsed = time.perf_counter() - start_time
        resolved_model = self.resolve_model_id(request.model) or request.model
        logger.info(
            f"Rerank: model={resolved_model}, {len(documents_raw)} docs, "
            f"{output.total_tokens} tokens in {elapsed:.3f}s"
        )
        self.get_server_metrics().record_request_complete(
            prompt_tokens=output.total_tokens,
            completion_tokens=0,
            cached_tokens=0,
            prefill_duration=elapsed,
            model_id=resolved_model,
            request_duration=elapsed,
        )

        # Format response - results sorted by score (descending). Strings wrap
        # into {"text": "..."}; dict inputs pass through as-is so multimodal
        # callers get their original 'image' back.
        results = []
        for idx in output.indices:
            if request.return_documents:
                orig = documents_raw[idx]
                display_doc = orig if isinstance(orig, dict) else {"text": orig}
            else:
                display_doc = None
            result = RerankResult(
                index=idx,
                relevance_score=output.scores[idx],
                document=display_doc,
            )
            results.append(result)

        return RerankResponse(
            results=results,
            model=request.model,
            usage=RerankUsage(total_tokens=output.total_tokens),
        )

    async def create_completion(
        self,
        request: CompletionRequest,
        http_request: FastAPIRequest,
        _: bool = Depends(verify_inference_api_key),
    ):
        """Create a text completion."""
        if self.state.oq_manager and self.state.oq_manager.is_quantizing:
            raise HTTPException(
                status_code=503,
                detail="Server is busy with oQ quantization. Please try again after quantization completes.",
            )
        lease = _LLMEngineLease(pool=self.state.engine_pool)
        try:
            load_start = time.perf_counter()
            engine = await self.get_engine_for_model(request.model, lease=lease)
            model_load_duration = time.perf_counter() - load_start
            resolved_model = self._serving_model_id(lease, request.model)

            # Handle single prompt or list of prompts
            prompts = (
                request.prompt if isinstance(request.prompt, list) else [request.prompt]
            )

            # Validate context window for each prompt
            prompt_token_ids_by_prompt = []
            for prompt in prompts:
                prompt_token_ids = list(engine.tokenizer.encode(prompt))
                prompt_token_ids_by_prompt.append(prompt_token_ids)
                self.validate_context_window(len(prompt_token_ids), request.model)

            # Pre-flight prefill memory guard — see create_chat_completion for
            # the reason this must precede any StreamingResponse return.
            # Thread the client-provided X-Request-ID when present so the 400
            # log line and the FastAPI handler trace correlate with whatever
            # the client is using on its side.
            upstream_request_id = http_request.headers.get("x-request-id")
            await _raise_if_llm_lease_abort_requested(lease)
            for prompt in prompts:
                await engine.preflight_completion(
                    prompt, request_id=upstream_request_id
                )
            await _raise_if_llm_lease_abort_requested(lease)
            inference_request_id = _request_abort_id(engine)

            if request.stream:
                response_id = f"cmpl-{uuid.uuid4().hex[:8]}"
                keepalive = self._resolve_keepalive("openai_completion")
                if keepalive == _KEEPALIVE_COMPLETION_CHUNK:
                    keepalive = _completion_keepalive_chunk(response_id)
                return StreamingResponse(
                    _release_after_stream(
                        _with_request_disconnect_abort(
                            _with_sse_keepalive(
                                self.stream_completion(
                                    engine,
                                    prompts[0],
                                    request,
                                    model_load_duration=model_load_duration,
                                    prompt_token_ids=prompt_token_ids_by_prompt[0],
                                    resolved_model=resolved_model,
                                    response_id=response_id,
                                    inference_request_id=inference_request_id,
                                ),
                                http_request=http_request,
                                keepalive_chunk=keepalive,
                            ),
                            http_request,
                            engine,
                            inference_request_id,
                        ),
                        lease,
                    ),
                    media_type="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
                )

            # Non-streaming response with keepalive during prefill
            async def _build_completion():
                await _raise_if_llm_lease_abort_requested(lease)
                start_time = time.perf_counter()
                choices = []
                total_completion_tokens = 0
                total_prompt_tokens = 0
                total_cached_tokens = 0

                (
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
                ) = self.get_sampling_params(
                    request.temperature,
                    request.top_p,
                    request.model,
                    req_top_k=getattr(request, "top_k", None),
                    req_repetition_penalty=getattr(request, "repetition_penalty", None),
                    req_min_p=getattr(request, "min_p", None),
                    req_presence_penalty=getattr(request, "presence_penalty", None),
                    req_frequency_penalty=getattr(request, "frequency_penalty", None),
                    req_max_tokens=request.max_tokens,
                    req_xtc_probability=getattr(request, "xtc_probability", None),
                    req_xtc_threshold=getattr(request, "xtc_threshold", None),
                )

                gen_kwargs = {}
                thinking_budget = self._resolve_thinking_budget(request, request.model)
                if thinking_budget is not None:
                    gen_kwargs["thinking_budget"] = thinking_budget
                if inference_request_id is not None:
                    gen_kwargs["_request_id"] = inference_request_id
                # Widen the repetition-penalty look-back window when the client
                # asks for it (mlx-lm default window is 20 tokens).
                repetition_context_size = getattr(
                    request, "repetition_context_size", None
                )
                if repetition_context_size is not None:
                    gen_kwargs["repetition_context_size"] = repetition_context_size

                # First prompt's first-token timestamp only: later prompts start
                # after earlier generations, so their first_token_at would count
                # prior generation time as prefill.
                first_token_at = None
                for i, prompt in enumerate(prompts):
                    output = await engine.generate(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        min_p=min_p,
                        repetition_penalty=repetition_penalty,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                        xtc_probability=xtc_probability,
                        xtc_threshold=xtc_threshold,
                        stop=request.stop,
                        seed=request.seed,
                        **gen_kwargs,
                    )
                    if i == 0:
                        first_token_at = getattr(output, "first_token_at", None)

                    choices.append(
                        CompletionChoice(
                            index=i,
                            text=output.text,
                            finish_reason=output.finish_reason,
                        )
                    )
                    total_completion_tokens += output.completion_tokens
                    total_prompt_tokens += output.prompt_tokens
                    total_cached_tokens += output.cached_tokens

                elapsed = time.perf_counter() - start_time
                tokens_per_sec = total_completion_tokens / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Completion: model={resolved_model}, "
                    f"{total_completion_tokens} tokens in {elapsed:.2f}s "
                    f"({tokens_per_sec:.1f} tok/s), prompt: {total_prompt_tokens}"
                )

                prefill_duration = (
                    (first_token_at - start_time) if first_token_at is not None else 0.0
                )
                gen_duration = (
                    elapsed - prefill_duration if prefill_duration > 0 else elapsed
                )
                self.get_server_metrics().record_request_complete(
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    cached_tokens=total_cached_tokens,
                    prefill_duration=prefill_duration,
                    generation_duration=gen_duration,
                    model_id=resolved_model,
                    request_duration=elapsed,
                )

                return CompletionResponse(
                    model=request.model,
                    choices=choices,
                    usage=Usage(
                        prompt_tokens=total_prompt_tokens,
                        completion_tokens=total_completion_tokens,
                        total_tokens=total_prompt_tokens + total_completion_tokens,
                        prompt_tokens_details=PromptTokensDetails(
                            cached_tokens=total_cached_tokens,
                        ),
                        model_load_duration=(
                            round(model_load_duration, 2)
                            if model_load_duration > 1.0
                            else None
                        ),
                        total_time=round(elapsed, 2),
                    ),
                ).model_dump_json(exclude_none=True)

            return await _json_response_or_keepalive(
                http_request, _build_completion(), lease=lease
            )
        except BaseException:
            await lease.release()
            raise

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        http_request: FastAPIRequest,
        _: bool = Depends(verify_inference_api_key),
    ):
        """
        Create a chat completion.

        Structured output (JSON mode):
        ```json
        response_format={"type": "json_object"}
        ```

        Structured output (JSON Schema):
        ```json
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "my_schema",
                "schema": {"type": "object", "properties": {...}}
            }
        }
        ```
        """
        # Log incoming request summary at debug, message content at trace
        logger.debug(
            f"Chat completion request received: model={request.model}, "
            f"messages={len(request.messages)}, stream={request.stream}, "
            f"max_tokens={request.max_tokens}, temp={request.temperature}"
        )
        if logger.isEnabledFor(5):
            for i, msg in enumerate(request.messages):
                content_preview = str(msg.content)[:200] if msg.content else "(empty)"
                logger.log(
                    5,
                    "  Message[%d]: role=%s, content=%s...",
                    i,
                    msg.role,
                    content_preview,
                )

        if is_markitdown_model(request.model):
            return await self._create_markitdown_chat_completion(request, http_request)

        request = await self._preprocess_markitdown_files_for_llm(request)

        # Block inference during quantization to prevent GPU Metal errors
        if self.state.oq_manager and self.state.oq_manager.is_quantizing:
            raise HTTPException(
                status_code=503,
                detail="Server is busy with oQ quantization. Please try again after quantization completes.",
            )

        lease = _LLMEngineLease(pool=self.state.engine_pool)
        try:
            load_start = time.perf_counter()
            engine = await self.get_engine_for_model(request.model, lease=lease)
            model_load_duration = time.perf_counter() - load_start

            # Use the exact model selected by the pool, including fallback.
            resolved_model = self._serving_model_id(lease, request.model)

            # Get per-model settings
            max_tool_result_tokens = None
            reasoning_parser = None
            settings_guided_grammar = None
            ms = self.get_model_settings_for_request(request.model)
            if ms:
                max_tool_result_tokens = ms.max_tool_result_tokens
                reasoning_parser = ms.reasoning_parser
                settings_guided_grammar = _settings_guided_grammar(ms)
            merged_ct_kwargs = merge_chat_template_request_kwargs(
                ms,
                merge_reasoning_effort_chat_template_kwargs(
                    request.chat_template_kwargs,
                    request.reasoning_effort,
                ),
            )

            # Extract messages - different engines need different content handling.
            # Templates that expose message.reasoning_content natively (Qwen 3.6+)
            # get reasoning as a separate field; others fall back to <think> inlined
            # in content.
            _entry = self.get_engine_pool().get_entry(resolved_model)
            native_reasoning = uses_native_reasoning_content(
                resolved_model,
                config_model_type=(
                    getattr(_entry, "config_model_type", None)
                    if _entry is not None
                    else None
                ),
                engine_model_type=getattr(engine, "model_type", None),
                preserve_thinking_default=(
                    getattr(_entry, "preserve_thinking_default", None)
                    if _entry is not None
                    else None
                ),
            )
            is_vlm = isinstance(engine, VLMBatchedEngine)
            is_dflash_vlm = not is_vlm and getattr(
                engine, "supports_multimodal_fallback", False
            )
            extractor = getattr(engine, "message_extractor", None)
            merge_system_fallback_roles = not (is_vlm or is_dflash_vlm)
            if extractor is not None:
                extractor_kwargs = {}
                try:
                    if (
                        "consolidate_system_messages"
                        in inspect.signature(extractor).parameters
                    ):
                        extractor_kwargs["consolidate_system_messages"] = False
                except (TypeError, ValueError):
                    pass
                messages = extractor(
                    request.messages,
                    max_tool_result_tokens,
                    engine.tokenizer,
                    **extractor_kwargs,
                )
                merge_system_fallback_roles = True
            elif is_vlm or is_dflash_vlm:
                # VLM or DFlash with VLM fallback: preserve image_url content parts
                messages = extract_multimodal_content(
                    request.messages,
                    max_tool_result_tokens,
                    engine.tokenizer,
                    native_reasoning_content=native_reasoning,
                    consolidate_system_messages=False,
                )
            else:
                messages = extract_text_content(
                    request.messages,
                    max_tool_result_tokens,
                    engine.tokenizer,
                    native_reasoning_content=native_reasoning,
                    consolidate_system_messages=False,
                )

            # Detect and strip partial mode at the API boundary — exactly once,
            # before any chat template application.  The boolean result is forwarded
            # as an explicit parameter so the engine never has to re-derive it.
            is_partial = detect_and_strip_partial(messages)

            # Compile grammar for structured output (logit-level enforcement).
            # Grammar compilation needs the tokenizer, so ensure the engine is loaded.
            response_format = request.response_format
            guided_grammar = _effective_guided_grammar(
                structured_outputs=request.structured_outputs,
                response_format=response_format,
                request_guided_grammar=request.guided_grammar,
                settings_guided_grammar=settings_guided_grammar,
            )
            structured_outputs = _normalize_structured_outputs(
                request.structured_outputs,
                guided_grammar,
            )
            _reject_diffusion_structured_outputs(
                engine,
                response_format=response_format,
                structured_outputs=structured_outputs,
                guided_grammar=guided_grammar,
            )
            if structured_outputs is not None or response_format:
                await engine.start()
            compiled_grammar = _compile_grammar_for_request(
                engine,
                structured_outputs=structured_outputs,
                response_format=response_format,
                chat_template_kwargs=merged_ct_kwargs or None,
                reasoning_parser=reasoning_parser,
            )
            # Fall back to prompt injection when grammar is not compiled. The degrade
            # is also surfaced to the caller as a Warning response header (#1241).
            # Only response formats that actually request grammar-constrained JSON
            # (json_object / json_schema) can be "unenforced"; a plain text format
            # never asked for enforcement, so it must not warn (#1241 review).
            response_format_warning = None
            if compiled_grammar is None and _response_format_requests_grammar(
                response_format
            ):
                response_format_warning = _response_format_warning_header(
                    response_format
                )
                json_instruction = build_json_system_prompt(response_format)
                if json_instruction:
                    messages = _inject_json_instruction(messages, json_instruction)

            # Merge MCP tools with user-provided tools unless the request explicitly
            # disables tool use.
            tools_disabled = request.tool_choice == "none"
            if getattr(engine, "is_diffusion_model", False) and not getattr(
                engine, "supports_tool_calling", False
            ):
                if request.tools and not tools_disabled:
                    raise InvalidRequestError(
                        "Tool calling is not supported for this diffusion model "
                        "(no tool parser matched its chat template).",
                        field="tools",
                    )
                tools_disabled = True
            effective_tools = None if tools_disabled else request.tools
            if (
                self.state.mcp_manager
                and not tools_disabled
                and request.include_mcp_tools
                and self.mcp_tools_exposed()
            ):
                # Convert Pydantic ToolDefinition models to dicts for merge_tools
                user_tools_dicts = (
                    [t.model_dump() for t in request.tools] if request.tools else None
                )
                effective_tools = self.state.mcp_manager.get_merged_tools(
                    user_tools_dicts
                )

            # Validate context window before sending to model
            tools_for_template = (
                convert_tools_for_template(effective_tools) if effective_tools else None
            )
            # Gemma 4 drops required params that lack descriptions — enrich them
            if tools_for_template and "gemma" in (resolved_model or "").lower():
                tools_for_template = enrich_tool_params_for_gemma4(tools_for_template)
            await _ensure_tokenizer_for_system_probe(engine, messages)
            messages = prepare_system_messages_for_template(
                messages,
                engine.tokenizer,
                tools=tools_for_template,
                chat_template_kwargs=merged_ct_kwargs or None,
                is_partial=is_partial,
                merge_consecutive_roles=merge_system_fallback_roles,
                unsupported_mid_system_policy=self._unsupported_mid_system_policy(),
            )
            try:
                num_prompt_tokens = engine.count_chat_tokens(
                    messages,
                    tools_for_template,
                    chat_template_kwargs=merged_ct_kwargs or None,
                    is_partial=is_partial,
                )
            except Exception as e:
                # Catch chat template rendering failures: Jinja2 TemplateError,
                # AssertionError from strict role validation, ValueError, etc.
                err_name = type(e).__name__.lower()
                err_msg = str(e).lower()
                if (
                    "template" in err_name
                    or "template" in err_msg
                    or isinstance(e, (AssertionError, ValueError))
                ):
                    raise HTTPException(
                        status_code=400, detail=f"Chat template error: {e}"
                    )
                raise
            self.validate_context_window(num_prompt_tokens, request.model)

            # Prepare kwargs
            (
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
            ) = self.get_sampling_params(
                request.temperature,
                request.top_p,
                request.model,
                req_top_k=getattr(request, "top_k", None),
                req_repetition_penalty=getattr(request, "repetition_penalty", None),
                req_min_p=getattr(request, "min_p", None),
                req_presence_penalty=getattr(request, "presence_penalty", None),
                req_frequency_penalty=getattr(request, "frequency_penalty", None),
                req_max_tokens=request.max_tokens,
                sampling_override=request.sampling_override,
                req_xtc_probability=getattr(request, "xtc_probability", None),
                req_xtc_threshold=getattr(request, "xtc_threshold", None),
            )
            chat_kwargs = {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "repetition_penalty": repetition_penalty,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "xtc_probability": xtc_probability,
                "xtc_threshold": xtc_threshold,
            }

            # Widen the repetition-penalty look-back window when the client
            # asks for it (mlx-lm default window is 20 tokens).
            repetition_context_size = getattr(request, "repetition_context_size", None)
            if repetition_context_size is not None:
                chat_kwargs["repetition_context_size"] = repetition_context_size

            # Add seed for reproducible generation (best-effort)
            if request.seed is not None:
                chat_kwargs["seed"] = request.seed

            # Add thinking budget if applicable
            thinking_budget = self._resolve_thinking_budget(request, request.model)
            if thinking_budget is not None:
                chat_kwargs["thinking_budget"] = thinking_budget

            # Auto-set enable_thinking in chat template kwargs when a positive thinking
            # budget is active (from request or model settings).  Some chat
            # templates (e.g. Gemma 4) explicitly suppress thinking unless this
            # kwarg is True.
            if (
                thinking_budget is not None
                and thinking_budget > 0
                and "enable_thinking" not in merged_ct_kwargs
            ):
                merged_ct_kwargs["enable_thinking"] = True

            # Auto-set preserve_thinking only when the template advertises support
            # for it (Qwen 3.6+). Other templates silently ignore unknown kwargs
            # today but strict templates could raise, so gate on the detected flag.
            _entry = self.get_engine_pool().get_entry(resolved_model)
            if (
                _entry is not None
                and _entry.preserve_thinking_default is True
                and merged_ct_kwargs.get("enable_thinking") is not False
                and "preserve_thinking" not in merged_ct_kwargs
            ):
                merged_ct_kwargs["preserve_thinking"] = True

            # Add compiled grammar for logit-level structured output.
            # When a reasoning_parser is configured, the structural tag includes
            # a thinking phase — auto-set a thinking_budget so the model exits
            # the reasoning phase and the grammar can activate.
            if compiled_grammar is not None:
                chat_kwargs["compiled_grammar"] = compiled_grammar
                if reasoning_parser and "thinking_budget" not in chat_kwargs:
                    default_budget = min(max_tokens // 2, 4096)
                    chat_kwargs["thinking_budget"] = default_budget
                    logger.debug(
                        "Auto-set thinking_budget=%d for grammar-constrained request",
                        default_budget,
                    )

            # Add tools if provided (includes MCP tools)
            if tools_for_template:
                chat_kwargs["tools"] = tools_for_template

            # Add chat template kwargs
            if merged_ct_kwargs:
                chat_kwargs["chat_template_kwargs"] = merged_ct_kwargs

            # Forward partial-mode decision to the engine explicitly
            chat_kwargs["is_partial"] = is_partial
            chat_kwargs["preserve_reasoning"] = cache_reasoning_output(
                ms,
                native_reasoning=native_reasoning,
                chat_template_kwargs=merged_ct_kwargs,
            )

            # SpecPrefill: per-request overrides (fall back to model_settings)
            if request.specprefill is not None:
                chat_kwargs["specprefill"] = request.specprefill
            if request.specprefill_keep_pct is not None:
                chat_kwargs["specprefill_keep_pct"] = request.specprefill_keep_pct
            elif self.state.settings_manager and ms.specprefill_keep_pct is not None:
                chat_kwargs["specprefill_keep_pct"] = ms.specprefill_keep_pct
            if getattr(request, "specprefill_threshold", None) is not None:
                chat_kwargs["specprefill_threshold"] = request.specprefill_threshold
            elif self.state.settings_manager and ms.specprefill_threshold is not None:
                chat_kwargs["specprefill_threshold"] = ms.specprefill_threshold

            if request.stop:
                chat_kwargs["stop"] = request.stop

            # Pre-flight prefill memory guard. Must run BEFORE either branch wraps
            # the response in a StreamingResponse — starlette emits
            # http.response.start (status 200) before iterating the body generator,
            # so a typed exception thrown later by add_request lands as "Caught
            # handled exception, but response already started" and the client sees
            # an incomplete chunked read. Running the check here lets
            # prefill_memory_exceeded_handler return a clean HTTP 400.
            await _raise_if_llm_lease_abort_requested(lease)
            await engine.preflight_chat(
                messages,
                request_id=http_request.headers.get("x-request-id"),
                **chat_kwargs,
            )

            await _raise_if_llm_lease_abort_requested(lease)
            inference_request_id = _request_abort_id(engine)
            if inference_request_id is not None:
                chat_kwargs["_request_id"] = inference_request_id

            if request.stream:
                # Pre-mint the completion id so the keepalive frame (emitted before the
                # generator starts) can share it. See _chat_keepalive_chunk.
                response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                keepalive = self._resolve_keepalive("openai_chat")
                if keepalive == _KEEPALIVE_CHAT_CHUNK:
                    keepalive = _chat_keepalive_chunk(response_id)
                sse_headers = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
                if response_format_warning:
                    sse_headers["Warning"] = response_format_warning
                return StreamingResponse(
                    _release_after_stream(
                        _with_request_disconnect_abort(
                            _with_sse_keepalive(
                                self.stream_chat_completion(
                                    engine,
                                    messages,
                                    request,
                                    model_load_duration=model_load_duration,
                                    resolved_model=resolved_model,
                                    response_id=response_id,
                                    **chat_kwargs,
                                ),
                                http_request=http_request,
                                keepalive_chunk=keepalive,
                            ),
                            http_request,
                            engine,
                            inference_request_id,
                        ),
                        lease,
                    ),
                    media_type="text/event-stream",
                    headers=sse_headers,
                )

            # Non-streaming response with keepalive during prefill
            async def _build_chat_completion():
                await _raise_if_llm_lease_abort_requested(lease)
                start_time = time.perf_counter()

                output = await engine.chat(messages=messages, **chat_kwargs)

                elapsed = time.perf_counter() - start_time
                tokens_per_sec = (
                    output.completion_tokens / elapsed if elapsed > 0 else 0
                )
                is_diffusion = getattr(engine, "is_diffusion_model", False)
                speed_text = _format_generation_speed_for_log(
                    output,
                    tokens_per_sec,
                    is_diffusion=is_diffusion,
                )
                logger.info(
                    f"Chat completion: model={resolved_model}, "
                    f"{output.completion_tokens} tokens in {elapsed:.2f}s "
                    f"({speed_text}), prompt: {output.prompt_tokens}, "
                    f"finish_reason={output.finish_reason}, max_tokens={max_tokens}, "
                    f"request_max_tokens={request.max_tokens}"
                )
                first_token_at = getattr(output, "first_token_at", None)
                ttft = (
                    (first_token_at - start_time) if first_token_at is not None else 0.0
                )
                gen_duration = elapsed - ttft if ttft > 0 else elapsed
                metric_prefill_duration, metric_gen_duration = (
                    _resolve_metric_durations(
                        output,
                        is_diffusion=is_diffusion,
                        prefill_duration=ttft,
                        generation_duration=gen_duration,
                    )
                )

                self.get_server_metrics().record_request_complete(
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    cached_tokens=output.cached_tokens,
                    prefill_duration=metric_prefill_duration,
                    generation_duration=metric_gen_duration,
                    model_id=resolved_model,
                    request_duration=elapsed,
                )

                # Separate thinking from content
                raw_text = clean_special_tokens(output.text) if output.text else ""
                thinking_content, regular_content = extract_thinking(
                    raw_text, truncated=output.finish_reason == "length"
                )
                cleaned_thinking = sanitize_tool_call_markup(
                    thinking_content, engine.tokenizer
                )

                # Protocol parsers can return structured tool_calls directly.
                if output.tool_calls:
                    tool_calls = _convert_parser_tool_calls(output.tool_calls)
                    cleaned_text = regular_content
                else:
                    extraction = extract_tool_calls_with_thinking(
                        thinking_content,
                        regular_content,
                        tokenizer=engine.tokenizer,
                        tools=tools_for_template,
                        finish_reason=output.finish_reason,
                    )
                    cleaned_text = extraction.cleaned_text
                    tool_calls = extraction.tool_calls
                    if failure := _tool_call_failure(extraction):
                        raise _ToolCallGenerationError(failure["error"])
                    cleaned_thinking = extraction.cleaned_thinking

                # Process response_format if specified
                if response_format and not tool_calls:
                    cleaned_text, parsed_json, is_valid, error = parse_json_output(
                        cleaned_text or regular_content, response_format
                    )
                    if parsed_json is not None:
                        cleaned_text = json.dumps(parsed_json)
                    if not is_valid:
                        logger.warning(f"JSON validation failed: {error}")

                # Reverse Gemma 4 parameter renaming (param_description -> description)
                if tool_calls and "gemma" in (resolved_model or "").lower():
                    for tc in tool_calls:
                        if tc.function and tc.function.arguments:
                            try:
                                args = json.loads(tc.function.arguments)
                                args = restore_gemma4_param_names(args)
                                tc.function.arguments = json.dumps(
                                    args, ensure_ascii=False
                                )
                            except (json.JSONDecodeError, AttributeError):
                                pass

                finish_reason = "tool_calls" if tool_calls else output.finish_reason

                return ChatCompletionResponse(
                    model=request.model,
                    choices=[
                        ChatCompletionChoice(
                            message=AssistantMessage(
                                content=cleaned_text.strip() if cleaned_text else None,
                                reasoning_content=(
                                    cleaned_thinking if cleaned_thinking else None
                                ),
                                tool_calls=tool_calls,
                            ),
                            finish_reason=finish_reason,
                        )
                    ],
                    usage=Usage(
                        prompt_tokens=output.prompt_tokens,
                        completion_tokens=output.completion_tokens,
                        total_tokens=output.prompt_tokens + output.completion_tokens,
                        prompt_tokens_details=PromptTokensDetails(
                            cached_tokens=output.cached_tokens,
                        ),
                        model_load_duration=(
                            round(model_load_duration, 2)
                            if model_load_duration > 1.0
                            else None
                        ),
                        total_time=round(elapsed, 2),
                    ),
                ).model_dump_json(exclude_none=True)

            json_headers = (
                {"Warning": response_format_warning}
                if response_format_warning
                else None
            )
            return await _json_response_or_keepalive(
                http_request,
                _build_chat_completion(),
                lease=lease,
                headers=json_headers,
            )

        except BaseException:
            await lease.release()
            raise
