# SPDX-License-Identifier: Apache-2.0
"""Responses for the private inference application."""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import Depends, HTTPException
from fastapi import Request as FastAPIRequest
from fastapi.responses import StreamingResponse
from molto_config.model_settings import merge_chat_template_request_kwargs
from molto_contracts.api.responses_models import (
    OutputItem,
    ResponseObject,
    ResponsesRequest,
)
from molto_runtime.engine import BaseEngine, VLMBatchedEngine
from molto_runtime.exceptions import InvalidRequestError, PrefillMemoryExceededError
from molto_runtime.generation.thinking import (
    ThinkingParser,
    extract_thinking,
    prompt_opens_thinking,
)
from molto_runtime.generation.tool_calling import (
    ToolCallStreamFilter,
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
    merge_reasoning_effort_chat_template_kwargs,
    prepare_system_messages_for_template,
    uses_native_reasoning_content,
)

# Import from new modular API
from molto_server.api.parser_tool_calls import (
    convert_parser_tool_calls as _convert_parser_tool_calls,
)
from molto_server.api.responses_utils import (
    ResponseStateCorruptError,
    ResponseStateNotFoundError,
    apply_namespace_tool_aliases,
    build_function_call_output_item,
    build_message_output_item,
    build_reasoning_output_item,
    build_response_store_record,
    build_response_usage,
    convert_responses_input_to_messages,
    convert_responses_tools,
    format_sse_event,
    normalize_response_output_to_messages,
    split_namespace_tool_name,
)

from .dependencies import verify_inference_api_key
from .engine_requests import (
    _ensure_tokenizer_for_system_probe,
    _raise_if_llm_lease_abort_requested,
    _release_after_stream,
)
from .errors import _prefill_memory_openai_error_body
from .openai_streaming import (
    _render_chat_prompt_for_thinking_detection,
    _tool_call_failure,
)
from .shared import _LLMEngineLease, _ToolCallGenerationError
from .structured_output import (
    _compile_grammar_for_request,
    _inject_json_instruction,
    _reject_diffusion_structured_outputs,
    _response_format_warning_header,
)
from .transport import (
    _aclose_async_iterator,
    _json_response_or_keepalive,
    _request_abort_id,
    _with_request_disconnect_abort,
    _with_sse_keepalive,
)

logger = logging.getLogger(__name__)


def _should_store_response(store_flag: bool | None) -> bool:
    """OpenAI Responses defaults to storing responses unless explicitly disabled."""
    return store_flag is not False


class ResponsesController:
    def _resolve_previous_response_messages(
        self, previous_response_id: str
    ) -> list[dict]:
        """Resolve a previous_response_id chain into chat messages."""
        try:
            return self.state.responses_store.resolve_chain_messages(
                previous_response_id
            )
        except ResponseStateNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Response state not found for previous_response_id. "
                    "It may have been deleted, evicted, or lost after restart."
                ),
            ) from exc
        except ResponseStateCorruptError as exc:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Stored response state is incomplete or corrupted for "
                    "previous_response_id."
                ),
            ) from exc

    def _store_response_state(
        self,
        public_response: dict,
        input_messages: list[dict],
    ) -> None:
        """Persist the response object and the normalized conversation state."""
        output_messages = normalize_response_output_to_messages(
            public_response.get("output", [])
        )
        record = build_response_store_record(
            public_response,
            input_messages=input_messages,
            output_messages=output_messages,
        )
        self.state.responses_store.put(public_response["id"], record)

    async def create_response(
        self,
        request: ResponsesRequest,
        http_request: FastAPIRequest,
        _: bool = Depends(verify_inference_api_key),
    ):
        """Create a response (OpenAI Responses API)."""
        if self.state.oq_manager and self.state.oq_manager.is_quantizing:
            raise HTTPException(
                status_code=503,
                detail="Server is busy with oQ quantization. Please try again after quantization completes.",
            )

        logger.debug(
            f"Responses API request: model={request.model}, stream={request.stream}"
        )

        load_start = time.perf_counter()
        lease = _LLMEngineLease(pool=self.state.engine_pool)
        try:
            engine = await self.get_engine_for_model(request.model, lease=lease)
            model_load_duration = time.perf_counter() - load_start

            resolved_model = self._serving_model_id(lease, request.model)

            # Images in function_call_output lists survive only for engines that
            # can extract them; text engines get a placeholder instead so base64
            # payloads never reach the prompt (#2989).
            preserve_tool_images = isinstance(engine, VLMBatchedEngine) or getattr(
                engine, "supports_multimodal_fallback", False
            )

            current_input_messages = convert_responses_input_to_messages(
                request.input,
                consolidate_system_messages=False,
                preserve_images=preserve_tool_images,
            )

            # Build previous context from previous_response_id
            previous_messages = None
            if request.previous_response_id:
                previous_messages = self._resolve_previous_response_messages(
                    request.previous_response_id
                )

            # Convert Responses API input → internal messages
            messages = convert_responses_input_to_messages(
                request.input,
                request.instructions,
                previous_messages,
                consolidate_system_messages=False,
                preserve_images=preserve_tool_images,
            )

            # Convert tools: flat → nested. namespace_aliases maps each expanded
            # namespace member's wire name back for the return path.
            namespace_aliases: dict = {}
            openai_tools = convert_responses_tools(request.tools, namespace_aliases)
            apply_namespace_tool_aliases(messages, namespace_aliases)
            if (
                getattr(engine, "is_diffusion_model", False)
                and not getattr(engine, "supports_tool_calling", False)
                and openai_tools
            ):
                raise InvalidRequestError(
                    "Tool calling is not supported for this diffusion model "
                    "(no tool parser matched its chat template).",
                    field="tools",
                )

            # Get per-model settings
            reasoning_parser = None
            ms = self.get_model_settings_for_request(request.model)
            if ms:
                reasoning_parser = ms.reasoning_parser
            merged_ct_kwargs = merge_chat_template_request_kwargs(
                ms,
                merge_reasoning_effort_chat_template_kwargs(
                    request.chat_template_kwargs,
                    (
                        request.reasoning.get("effort")
                        if isinstance(request.reasoning, dict)
                        else None
                    ),
                ),
            )

            _entry = self.get_engine_pool().get_entry(resolved_model)

            # Note: extract_text_content/extract_harmony_messages/extract_multimodal_content
            # are NOT called here because convert_responses_input_to_messages() already
            # returns plain dicts in {"role": str, "content": str} format.
            # Those extract functions expect Pydantic Message objects from OpenAI/Anthropic requests.

            # Handle text.format (structured output)
            response_format = None
            compiled_grammar = None
            response_format_warning = None
            if request.text and request.text.format:
                fmt = request.text.format
                if fmt.type == "json_object":
                    response_format = {"type": "json_object"}
                elif fmt.type == "json_schema":
                    response_format = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": fmt.name or "response",
                            "schema": fmt.schema_ or {},
                            "strict": fmt.strict or False,
                        },
                    }
                if response_format:
                    from molto_contracts.api.openai_models import ResponseFormat

                    _reject_diffusion_structured_outputs(
                        engine,
                        response_format=response_format,
                    )
                    await engine.start()
                    rf = ResponseFormat(**response_format)
                    compiled_grammar = _compile_grammar_for_request(
                        engine,
                        response_format=rf,
                        chat_template_kwargs=merged_ct_kwargs or None,
                        reasoning_parser=reasoning_parser,
                    )
                    if compiled_grammar is None:
                        # Non-strict formats still degrade to prompt injection, so
                        # surface it to the caller with the same Warning response
                        # header /v1/chat/completions uses; the log line alone only
                        # ever reaches the operator (#1241).
                        response_format_warning = _response_format_warning_header(rf)
                        json_instruction = build_json_system_prompt(rf)
                        if json_instruction:
                            messages = _inject_json_instruction(
                                messages, json_instruction
                            )
                else:
                    compiled_grammar = None

            # Merge MCP tools, matching create_chat_completion's tools_disabled
            # semantics: tool_choice="none" suppresses tool exposure to the chat
            # template, and the MCP merge itself must run even when the client
            # sent no tools of its own, since MCP servers can offer tools the
            # client never listed.
            tools_disabled = request.tool_choice == "none" or (
                getattr(engine, "is_diffusion_model", False)
                and not getattr(engine, "supports_tool_calling", False)
            )
            effective_tools = None if tools_disabled else openai_tools
            if (
                self.state.mcp_manager
                and not tools_disabled
                and self.mcp_tools_exposed()
            ):
                effective_tools = self.state.mcp_manager.get_merged_tools(openai_tools)

            # Convert tools for chat template
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
                is_partial=False,
                merge_consecutive_roles=True,
                unsupported_mid_system_policy=self._unsupported_mid_system_policy(),
            )

            # Validate context window
            try:
                num_prompt_tokens = engine.count_chat_tokens(
                    messages,
                    tools_for_template,
                    chat_template_kwargs=merged_ct_kwargs or None,
                )
            except Exception as e:
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

            # Build sampling kwargs
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
                req_max_tokens=request.max_output_tokens,
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

            # Auto-set enable_thinking when a positive thinking budget is active.
            if (
                thinking_budget is not None
                and thinking_budget > 0
                and "enable_thinking" not in merged_ct_kwargs
            ):
                merged_ct_kwargs["enable_thinking"] = True

            # Auto-set preserve_thinking only when the template advertises support
            # for it (Qwen 3.6+). Gated on detection so other templates don't
            # receive an unknown kwarg.
            native_reasoning = bool(_entry and _entry.preserve_thinking_default is True)
            if (
                native_reasoning
                and merged_ct_kwargs.get("enable_thinking") is not False
                and "preserve_thinking" not in merged_ct_kwargs
            ):
                merged_ct_kwargs["preserve_thinking"] = True

            # Add compiled grammar for logit-level structured output.
            if compiled_grammar is not None:
                chat_kwargs["compiled_grammar"] = compiled_grammar
                if reasoning_parser and "thinking_budget" not in chat_kwargs:
                    default_budget = min(max_tokens // 2, 4096)
                    chat_kwargs["thinking_budget"] = default_budget
                    logger.debug(
                        "Auto-set thinking_budget=%d for grammar-constrained request",
                        default_budget,
                    )

            if tools_for_template:
                chat_kwargs["tools"] = tools_for_template
            if merged_ct_kwargs:
                chat_kwargs["chat_template_kwargs"] = merged_ct_kwargs

            chat_kwargs["preserve_reasoning"] = cache_reasoning_output(
                ms,
                native_reasoning=uses_native_reasoning_content(
                    resolved_model,
                    config_model_type=getattr(_entry, "config_model_type", None),
                    engine_model_type=getattr(engine, "model_type", None),
                    preserve_thinking_default=getattr(
                        _entry, "preserve_thinking_default", None
                    ),
                ),
                chat_template_kwargs=merged_ct_kwargs,
            )

            # Pre-flight prefill memory guard — must precede any StreamingResponse
            # return so PrefillMemoryExceededError can be mapped to HTTP 400.
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
                sse_headers = {"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
                if response_format_warning:
                    sse_headers["Warning"] = response_format_warning
                return StreamingResponse(
                    _release_after_stream(
                        _with_request_disconnect_abort(
                            _with_sse_keepalive(
                                self.stream_responses_api(
                                    engine,
                                    messages,
                                    request,
                                    input_messages=current_input_messages,
                                    store_response=_should_store_response(
                                        request.store
                                    ),
                                    model_load_duration=model_load_duration,
                                    resolved_model=resolved_model,
                                    response_format=response_format,
                                    native_reasoning=native_reasoning,
                                    namespace_aliases=namespace_aliases,
                                    **chat_kwargs,
                                ),
                                http_request=http_request,
                                keepalive_chunk=self._resolve_keepalive(
                                    "openai_responses"
                                ),
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

            # Non-streaming with keepalive during prefill
            async def _build_responses_api():
                await _raise_if_llm_lease_abort_requested(lease)
                start_time = time.perf_counter()
                output = await engine.chat(messages=messages, **chat_kwargs)

                elapsed = time.perf_counter() - start_time
                tokens_per_sec = (
                    output.completion_tokens / elapsed if elapsed > 0 else 0
                )
                logger.info(
                    f"Responses API: model={resolved_model}, "
                    f"{output.completion_tokens} tokens in {elapsed:.2f}s "
                    f"({tokens_per_sec:.1f} tok/s)"
                )

                first_token_at = getattr(output, "first_token_at", None)
                prefill_duration = (
                    (first_token_at - start_time) if first_token_at is not None else 0.0
                )
                gen_duration = (
                    elapsed - prefill_duration if prefill_duration > 0 else elapsed
                )
                self.get_server_metrics().record_request_complete(
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    cached_tokens=output.cached_tokens,
                    prefill_duration=prefill_duration,
                    generation_duration=gen_duration,
                    model_id=resolved_model,
                    request_duration=elapsed,
                )

                # Process output text
                raw_text = clean_special_tokens(output.text) if output.text else ""
                thinking_content, regular_content = extract_thinking(
                    raw_text, truncated=output.finish_reason == "length"
                )

                # Parse tool calls
                if output.tool_calls:
                    tool_calls = _convert_parser_tool_calls(output.tool_calls)
                    cleaned_text = regular_content
                    cleaned_thinking = sanitize_tool_call_markup(
                        thinking_content, engine.tokenizer
                    )
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

                # Reverse Gemma 4 parameter renaming
                if tool_calls and "gemma" in (resolved_model or "").lower():
                    for tc in tool_calls:
                        fn = getattr(tc, "function", None)
                        if fn and fn.arguments:
                            try:
                                args = json.loads(fn.arguments)
                                args = restore_gemma4_param_names(args)
                                fn.arguments = json.dumps(args, ensure_ascii=False)
                            except (json.JSONDecodeError, AttributeError):
                                pass

                # Process response_format if specified
                if response_format and not tool_calls:
                    cleaned_text, parsed_json, is_valid, error = parse_json_output(
                        cleaned_text or regular_content, response_format
                    )
                    if parsed_json is not None:
                        cleaned_text = json.dumps(parsed_json)
                    if not is_valid:
                        logger.warning(f"JSON validation failed: {error}")

                # Build output items
                output_items: list[OutputItem] = []
                reasoning_text = (cleaned_thinking or "").strip()
                if reasoning_text:
                    output_items.append(build_reasoning_output_item(reasoning_text))
                output_items.append(
                    build_message_output_item(
                        cleaned_text.strip() if cleaned_text else ""
                    )
                )

                if tool_calls:
                    for tc in tool_calls:
                        if hasattr(tc, "function"):
                            call_id = tc.id
                            name = tc.function.name
                            arguments = tc.function.arguments
                        elif isinstance(tc, dict):
                            call_id = tc.get(
                                "call_id", tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
                            )
                            name = tc.get("name", "")
                            arguments = tc.get("arguments", "{}")
                        else:
                            continue
                        namespace, name = split_namespace_tool_name(
                            name, namespace_aliases
                        )
                        output_items.append(
                            build_function_call_output_item(
                                name=name,
                                arguments=arguments,
                                call_id=call_id,
                                namespace=namespace,
                            )
                        )

                reasoning_token_count = (
                    len(engine.tokenizer.encode(reasoning_text))
                    if reasoning_text
                    else 0
                )
                usage = build_response_usage(
                    input_tokens=output.prompt_tokens,
                    output_tokens=output.completion_tokens,
                    reasoning_tokens=reasoning_token_count,
                    cached_tokens=output.cached_tokens,
                )

                # Surface max_output_tokens truncation so clients can tell an
                # incomplete turn from a natural stop. The Responses API has no
                # finish_reason field; status + incomplete_details is the signal.
                truncated = getattr(output, "finish_reason", None) == "length"
                response_obj = ResponseObject(
                    model=request.model,
                    status="incomplete" if truncated else "completed",
                    output=output_items,
                    usage=usage,
                    tools=request.tools or [],
                    tool_choice=request.tool_choice or "auto",
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=request.max_output_tokens,
                    previous_response_id=request.previous_response_id,
                    incomplete_details={"reason": "max_output_tokens"}
                    if truncated
                    else None,
                )

                # Store response
                if _should_store_response(request.store):
                    self._store_response_state(
                        response_obj.model_dump(exclude_none=True),
                        input_messages=current_input_messages,
                    )

                return response_obj.model_dump_json()

            json_headers = (
                {"Warning": response_format_warning}
                if response_format_warning
                else None
            )
            return await _json_response_or_keepalive(
                http_request, _build_responses_api(), lease=lease, headers=json_headers
            )

        except BaseException:
            await lease.release()
            raise

    async def stream_responses_api(
        self,
        engine: BaseEngine,
        messages: list,
        request: ResponsesRequest,
        input_messages: list[dict] | None = None,
        store_response: bool = True,
        model_load_duration: float = 0.0,
        resolved_model: str | None = None,
        response_format=None,
        native_reasoning: bool = False,
        namespace_aliases: dict | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream Responses API events (SSE with named event types)."""
        from molto_contracts.api.shared_models import IDPrefix, generate_id

        start_time = time.perf_counter()
        first_token_time = None
        last_output = None
        accumulated_text = ""
        accumulated_reasoning = ""
        has_tools = bool(kwargs.get("tools"))
        # Some templates open the thinking block in the prompt itself, so the
        # generated text starts with reasoning body and only later emits </think>.
        start_in_thinking = native_reasoning
        if not start_in_thinking:
            try:
                tokenizer = getattr(engine, "tokenizer", None)
                if tokenizer is not None:
                    prompt, prompt_token_ids = (
                        _render_chat_prompt_for_thinking_detection(
                            engine, messages, kwargs
                        )
                    )
                    start_in_thinking, _ = prompt_opens_thinking(
                        tokenizer, prompt, prompt_token_ids=prompt_token_ids
                    )
            except Exception as exc:
                logger.debug(
                    "Could not detect Responses stream thinking state: %s", exc
                )
        thinking_parser = ThinkingParser(start_in_thinking=start_in_thinking)
        seq = 0

        response_id = generate_id(IDPrefix.RESPONSE)
        msg_id = generate_id(IDPrefix.MESSAGE)
        reasoning_id = generate_id(IDPrefix.REASONING)

        # Lazy item emission state — items are opened on first token
        reasoning_opened = False
        reasoning_closed = False
        message_opened = False
        next_output_index = 0
        reasoning_output_index: int | None = None  # captured when reasoning opens
        msg_output_index: int | None = None  # captured when message opens

        # Build initial response object (in_progress, empty output)
        initial_response = ResponseObject(
            id=response_id,
            model=request.model,
            status="in_progress",
            output=[],
            tools=request.tools or [],
            tool_choice=request.tool_choice or "auto",
            temperature=request.temperature,
            top_p=request.top_p,
            max_output_tokens=request.max_output_tokens,
            previous_response_id=request.previous_response_id,
        )
        initial_data = initial_response.model_dump(exclude_none=True)

        # 1. response.created
        seq += 1
        yield format_sse_event(
            "response.created",
            {
                "type": "response.created",
                "response": initial_data,
                "sequence_number": seq,
            },
        )

        # 2. response.in_progress
        seq += 1
        yield format_sse_event(
            "response.in_progress",
            {
                "type": "response.in_progress",
                "response": initial_data,
                "sequence_number": seq,
            },
        )

        # --- helper closures for lazy item emission ----------------------
        def _open_reasoning():
            nonlocal seq, reasoning_opened, reasoning_output_index, next_output_index
            if reasoning_opened:
                return []
            reasoning_opened = True
            reasoning_output_index = next_output_index
            next_output_index += 1
            events = []
            seq += 1
            events.append(
                format_sse_event(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": reasoning_output_index,
                        "item": {
                            "type": "reasoning",
                            "id": reasoning_id,
                            "status": "in_progress",
                            "summary": [],
                        },
                        "sequence_number": seq,
                    },
                )
            )
            seq += 1
            events.append(
                format_sse_event(
                    "response.reasoning_summary_part.added",
                    {
                        "type": "response.reasoning_summary_part.added",
                        "item_id": reasoning_id,
                        "output_index": reasoning_output_index,
                        "summary_index": 0,
                        "part": {"type": "summary_text", "text": ""},
                        "sequence_number": seq,
                    },
                )
            )
            return events

        def _close_reasoning():
            nonlocal seq, reasoning_closed
            if reasoning_closed or not reasoning_opened:
                return []
            reasoning_closed = True
            reasoning_text = accumulated_reasoning
            events = []
            seq += 1
            events.append(
                format_sse_event(
                    "response.reasoning_summary_text.done",
                    {
                        "type": "response.reasoning_summary_text.done",
                        "item_id": reasoning_id,
                        "output_index": reasoning_output_index,
                        "summary_index": 0,
                        "text": reasoning_text,
                        "sequence_number": seq,
                    },
                )
            )
            seq += 1
            events.append(
                format_sse_event(
                    "response.reasoning_summary_part.done",
                    {
                        "type": "response.reasoning_summary_part.done",
                        "item_id": reasoning_id,
                        "output_index": reasoning_output_index,
                        "summary_index": 0,
                        "part": {"type": "summary_text", "text": reasoning_text},
                        "sequence_number": seq,
                    },
                )
            )
            seq += 1
            events.append(
                format_sse_event(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": reasoning_output_index,
                        "item": {
                            "type": "reasoning",
                            "id": reasoning_id,
                            "status": "completed",
                            "summary": [
                                {"type": "summary_text", "text": reasoning_text}
                            ],
                        },
                        "sequence_number": seq,
                    },
                )
            )
            return events

        def _open_message():
            nonlocal seq, message_opened, next_output_index, msg_output_index
            if message_opened:
                return []
            message_opened = True
            msg_output_index = next_output_index
            next_output_index += 1
            events = []
            seq += 1
            events.append(
                format_sse_event(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": msg_output_index,
                        "item": {
                            "type": "message",
                            "id": msg_id,
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        },
                        "sequence_number": seq,
                    },
                )
            )
            seq += 1
            events.append(
                format_sse_event(
                    "response.content_part.added",
                    {
                        "type": "response.content_part.added",
                        "item_id": msg_id,
                        "output_index": msg_output_index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                        "sequence_number": seq,
                    },
                )
            )
            return events

        def _emit_reasoning_delta(delta: str):
            nonlocal seq, accumulated_reasoning
            if not delta:
                return []
            accumulated_reasoning += delta
            events = []
            events.extend(_open_reasoning())
            seq += 1
            events.append(
                format_sse_event(
                    "response.reasoning_summary_text.delta",
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": reasoning_id,
                        "output_index": reasoning_output_index,
                        "summary_index": 0,
                        "delta": delta,
                        "sequence_number": seq,
                    },
                )
            )
            return events

        # -----------------------------------------------------------------

        # Open message/reasoning items lazily so non-native <think> blocks can still
        # become a leading Responses reasoning item.

        # Stream tokens
        tool_filter = None
        thinking_filter = None
        stream_content = True
        if has_tools:
            _content_filter = ToolCallStreamFilter(
                engine.tokenizer,
                tools=kwargs.get("tools"),
            )
            # The thinking channel never contains a separator-prefixed DSML
            # block; holding trailing newlines would flush them as a late
            # reasoning delta after the channel closed.
            _thinking_filter = ToolCallStreamFilter(
                engine.tokenizer,
                tools=kwargs.get("tools"),
                consume_dsml_separator=False,
            )
            if _content_filter.active:
                tool_filter = _content_filter
                thinking_filter = _thinking_filter
            else:
                stream_content = False

        engine_stream = engine.stream_chat(messages=messages, **kwargs)
        try:
            async for output in engine_stream:
                if first_token_time is None and output.new_text:
                    first_token_time = time.perf_counter()
                last_output = output
                if output.new_text:
                    accumulated_text += output.new_text

                if stream_content and output.new_text:
                    thinking_delta, content_delta = thinking_parser.feed(
                        output.new_text
                    )

                    if thinking_delta:
                        if thinking_filter:
                            thinking_delta = thinking_filter.feed(thinking_delta)
                        for ev in _emit_reasoning_delta(thinking_delta):
                            yield ev

                    if content_delta:
                        if reasoning_opened and not reasoning_closed:
                            for ev in _close_reasoning():
                                yield ev
                        for ev in _open_message():
                            yield ev
                        if tool_filter:
                            content_delta = tool_filter.feed(content_delta)
                        if content_delta:
                            seq += 1
                            yield format_sse_event(
                                "response.output_text.delta",
                                {
                                    "type": "response.output_text.delta",
                                    "item_id": msg_id,
                                    "output_index": msg_output_index,
                                    "content_index": 0,
                                    "delta": content_delta,
                                    "sequence_number": seq,
                                },
                            )
        except Exception as e:
            if isinstance(e, PrefillMemoryExceededError):
                # Same shadowing as the chat generator (#3036): surface the
                # guard's code and message in the response.failed error object
                # instead of failing with no error at all.
                logger.warning(f"Responses API streaming prefill rejected: {e}")
                guard_body = _prefill_memory_openai_error_body(e)["error"]
                failure_error = {
                    "code": guard_body.get("molto_code", "prefill_memory_exceeded"),
                    "message": guard_body["message"],
                }
            else:
                logger.error(f"Error during Responses API streaming: {e}")
                failure_error = {"code": "server_error", "message": str(e)}
            seq += 1
            yield format_sse_event(
                "response.failed",
                {
                    "type": "response.failed",
                    "response": {
                        **initial_data,
                        "status": "failed",
                        "error": failure_error,
                    },
                    "sequence_number": seq,
                },
            )
            return

        finally:
            await _aclose_async_iterator(engine_stream)

        # Flush remaining content from parsers
        if stream_content:
            thinking_delta, content_delta = thinking_parser.finish(
                truncated=last_output is not None
                and last_output.finish_reason == "length"
            )
            if thinking_delta:
                if thinking_filter:
                    thinking_delta = thinking_filter.feed(thinking_delta)
                for ev in _emit_reasoning_delta(thinking_delta):
                    yield ev
            if thinking_filter:
                remaining_thinking = thinking_filter.finish()
                for ev in _emit_reasoning_delta(remaining_thinking):
                    yield ev
            if content_delta:
                if reasoning_opened and not reasoning_closed:
                    for ev in _close_reasoning():
                        yield ev
                for ev in _open_message():
                    yield ev
                if tool_filter:
                    content_delta = tool_filter.feed(content_delta)
                if content_delta:
                    seq += 1
                    yield format_sse_event(
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "item_id": msg_id,
                            "output_index": msg_output_index,
                            "content_index": 0,
                            "delta": content_delta,
                            "sequence_number": seq,
                        },
                    )
            if tool_filter:
                remaining = tool_filter.finish()
                if remaining:
                    if reasoning_opened and not reasoning_closed:
                        for ev in _close_reasoning():
                            yield ev
                    for ev in _open_message():
                        yield ev
                    seq += 1
                    yield format_sse_event(
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "item_id": msg_id,
                            "output_index": msg_output_index,
                            "content_index": 0,
                            "delta": remaining,
                            "sequence_number": seq,
                        },
                    )

        # Parse tool calls from accumulated text
        tool_calls = None
        tool_failure = None
        cleaned_text = accumulated_text
        if last_output and last_output.tool_calls:
            tool_calls = _convert_parser_tool_calls(last_output.tool_calls)
            cleaned_text = ""
        elif has_tools and accumulated_text:
            thinking_content, regular_content = extract_thinking(
                accumulated_text,
                truncated=last_output is not None
                and last_output.finish_reason == "length",
            )
            extraction = extract_tool_calls_with_thinking(
                thinking_content,
                regular_content,
                tokenizer=engine.tokenizer,
                tools=kwargs.get("tools"),
                finish_reason=last_output.finish_reason if last_output else "stop",
            )
            cleaned_text = extraction.cleaned_text
            tool_calls = extraction.tool_calls
            tool_failure = _tool_call_failure(extraction)
            if not stream_content:
                cleaned_thinking = (extraction.cleaned_thinking or "").strip()
                for ev in _emit_reasoning_delta(cleaned_thinking):
                    yield ev
                if reasoning_opened and not reasoning_closed:
                    for ev in _close_reasoning():
                        yield ev
            if not stream_content and cleaned_text:
                for ev in _open_message():
                    yield ev
                seq += 1
                yield format_sse_event(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": msg_id,
                        "output_index": msg_output_index,
                        "content_index": 0,
                        "delta": cleaned_text,
                        "sequence_number": seq,
                    },
                )
        else:
            # No tools — use raw accumulated text minus thinking.
            thinking_content, regular_content = extract_thinking(
                accumulated_text,
                truncated=last_output is not None
                and last_output.finish_reason == "length",
            )
            cleaned_text = (
                clean_special_tokens(regular_content) if regular_content else ""
            )

        recovered_thinking = (
            thinking_filter.take_recovery_candidate() if thinking_filter else ""
        )
        recovered_content = tool_filter.take_recovery_candidate() if tool_filter else ""
        if not tool_calls and not tool_failure:
            for ev in _emit_reasoning_delta(recovered_thinking):
                yield ev
            if recovered_content:
                if reasoning_opened and not reasoning_closed:
                    for ev in _close_reasoning():
                        yield ev
                for ev in _open_message():
                    yield ev
                seq += 1
                yield format_sse_event(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": msg_id,
                        "output_index": msg_output_index,
                        "content_index": 0,
                        "delta": recovered_content,
                        "sequence_number": seq,
                    },
                )

        # Reverse Gemma 4 parameter renaming
        if tool_calls and "gemma" in (resolved_model or request.model or "").lower():
            for tc in tool_calls:
                fn = getattr(tc, "function", None)
                if fn and fn.arguments:
                    try:
                        args = json.loads(fn.arguments)
                        args = restore_gemma4_param_names(args)
                        fn.arguments = json.dumps(args, ensure_ascii=False)
                    except (json.JSONDecodeError, AttributeError):
                        pass

        final_text = cleaned_text.strip() if cleaned_text else ""

        # Process response_format if specified
        if response_format and not tool_calls:
            _, parsed_json, is_valid, error = parse_json_output(
                final_text, response_format
            )
            if parsed_json is not None:
                final_text = json.dumps(parsed_json)
            if not is_valid:
                logger.warning(f"JSON validation failed: {error}")

        if reasoning_opened and not reasoning_closed:
            for ev in _close_reasoning():
                yield ev

        # Ensure message item is opened (even if no content was streamed).
        for ev in _open_message():
            yield ev

        # response.output_text.done
        seq += 1
        yield format_sse_event(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": msg_id,
                "output_index": msg_output_index,
                "content_index": 0,
                "text": final_text,
                "sequence_number": seq,
            },
        )

        # response.content_part.done
        seq += 1
        yield format_sse_event(
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "item_id": msg_id,
                "output_index": msg_output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": final_text, "annotations": []},
                "sequence_number": seq,
            },
        )

        # response.output_item.done (message)
        seq += 1
        yield format_sse_event(
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": msg_output_index,
                "item": {
                    "type": "message",
                    "id": msg_id,
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": final_text, "annotations": []}
                    ],
                },
                "sequence_number": seq,
            },
        )

        # Build output items for final response
        output_items = []
        reasoning_text = accumulated_reasoning
        if reasoning_text:
            output_items.append(
                {
                    "type": "reasoning",
                    "id": reasoning_id,
                    "status": "completed",
                    "summary": [{"type": "summary_text", "text": reasoning_text}],
                }
            )
        output_items.append(
            {
                "type": "message",
                "id": msg_id,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": final_text, "annotations": []}
                ],
            }
        )

        # Emit function call items if present
        if tool_calls:
            output_index = next_output_index
            for tc in tool_calls:
                if hasattr(tc, "function"):
                    call_id = tc.id
                    name = tc.function.name
                    arguments = tc.function.arguments
                elif isinstance(tc, dict):
                    call_id = tc.get(
                        "call_id", tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
                    )
                    name = tc.get("name", "")
                    arguments = tc.get("arguments", "{}")
                else:
                    continue

                namespace, name = split_namespace_tool_name(name, namespace_aliases)
                fc_id = generate_id(IDPrefix.FUNCTION_CALL)
                fc_item = {
                    "type": "function_call",
                    "id": fc_id,
                    "call_id": call_id,
                    "name": name,
                    "arguments": "",
                    "status": "in_progress",
                }
                if namespace:
                    fc_item["namespace"] = namespace

                # output_item.added
                seq += 1
                yield format_sse_event(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": fc_item,
                        "sequence_number": seq,
                    },
                )

                # function_call_arguments.delta
                seq += 1
                yield format_sse_event(
                    "response.function_call_arguments.delta",
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": fc_id,
                        "output_index": output_index,
                        "delta": arguments,
                        "sequence_number": seq,
                    },
                )

                # function_call_arguments.done
                seq += 1
                yield format_sse_event(
                    "response.function_call_arguments.done",
                    {
                        "type": "response.function_call_arguments.done",
                        "item_id": fc_id,
                        "output_index": output_index,
                        "arguments": arguments,
                        "sequence_number": seq,
                    },
                )

                # output_item.done
                completed_fc = {
                    "type": "function_call",
                    "id": fc_id,
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                    "status": "completed",
                }
                if namespace:
                    completed_fc["namespace"] = namespace
                seq += 1
                yield format_sse_event(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": output_index,
                        "item": completed_fc,
                        "sequence_number": seq,
                    },
                )

                output_items.append(completed_fc)
                output_index += 1
                next_output_index = output_index

        if tool_failure:
            seq += 1
            yield format_sse_event(
                "response.failed",
                {
                    "type": "response.failed",
                    "response": {
                        **initial_data,
                        "status": "failed",
                        "output": output_items,
                        "error": tool_failure["error"],
                    },
                    "sequence_number": seq,
                },
            )
            return

        # Record metrics
        usage_data = None
        if last_output and last_output.finished:
            end_time = time.perf_counter()
            total_duration = end_time - start_time
            ttft = (
                (first_token_time - start_time) if first_token_time else total_duration
            )
            if getattr(engine, "is_diffusion_model", False):
                gen_duration = total_duration
            else:
                gen_duration = end_time - (first_token_time or start_time)
            serving_model = resolved_model or request.model
            self.get_server_metrics().record_request_complete(
                prompt_tokens=last_output.prompt_tokens,
                completion_tokens=last_output.completion_tokens,
                cached_tokens=last_output.cached_tokens,
                prefill_duration=ttft,
                generation_duration=gen_duration,
                model_id=serving_model,
                request_duration=total_duration,
            )
            tokens_per_sec = (
                last_output.completion_tokens / total_duration
                if total_duration > 0
                else 0
            )
            logger.info(
                f"Responses API: model={serving_model}, "
                f"{last_output.completion_tokens} tokens in {total_duration:.2f}s "
                f"({tokens_per_sec:.1f} tok/s)"
            )
            reasoning_token_count = (
                len(engine.tokenizer.encode(reasoning_text)) if reasoning_text else 0
            )
            usage_data = {
                "input_tokens": last_output.prompt_tokens,
                "output_tokens": last_output.completion_tokens,
                "total_tokens": last_output.prompt_tokens
                + last_output.completion_tokens,
                "input_tokens_details": {"cached_tokens": last_output.cached_tokens},
                "output_tokens_details": {"reasoning_tokens": reasoning_token_count},
            }

        # 13. Emit the terminal event matching the final response status.
        truncated = getattr(last_output, "finish_reason", None) == "length"
        terminal_event = "response.incomplete" if truncated else "response.completed"
        final_response = {
            "id": response_id,
            "object": "response",
            "created_at": initial_response.created_at,
            "model": request.model,
            "status": "incomplete" if truncated else "completed",
            "output": output_items,
            "usage": usage_data,
            "tool_choice": request.tool_choice or "auto",
            "tools": (
                [t.model_dump(exclude_none=True) for t in request.tools]
                if request.tools
                else []
            ),
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_output_tokens": request.max_output_tokens,
        }
        if truncated:
            final_response["incomplete_details"] = {"reason": "max_output_tokens"}
        if request.previous_response_id:
            final_response["previous_response_id"] = request.previous_response_id

        seq += 1
        yield format_sse_event(
            terminal_event,
            {
                "type": terminal_event,
                "response": final_response,
                "sequence_number": seq,
            },
        )

        # Store for future previous_response_id usage
        if store_response:
            self._store_response_state(
                final_response, input_messages=input_messages or []
            )

    async def get_response(
        self,
        response_id: str,
        _: bool = Depends(verify_inference_api_key),
    ):
        """Retrieve a stored response."""
        data = self.state.responses_store.get(response_id)
        if data is None:
            raise HTTPException(status_code=404, detail="Response not found")
        return data

    async def delete_response(
        self,
        response_id: str,
        _: bool = Depends(verify_inference_api_key),
    ):
        """Delete a stored response."""
        if not self.state.responses_store.delete(response_id):
            raise HTTPException(status_code=404, detail="Response not found")
        return {"id": response_id, "object": "response.deleted", "deleted": True}
