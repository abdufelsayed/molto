# SPDX-License-Identifier: Apache-2.0
"""Anthropic for the private inference application."""

import inspect
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import Depends, HTTPException
from fastapi import Request as FastAPIRequest
from fastapi.responses import StreamingResponse
from omlx_config.model_settings import (
    forced_ct_keys,
    merge_chat_template_request_kwargs,
)
from omlx_contracts.api.anthropic_models import (
    MessagesRequest as AnthropicMessagesRequest,
)
from omlx_contracts.api.anthropic_models import TokenCountRequest, TokenCountResponse
from omlx_runtime.engine import BaseEngine, VLMBatchedEngine
from omlx_runtime.exceptions import InvalidRequestError, PrefillMemoryExceededError
from omlx_runtime.generation.anthropic_utils import (
    convert_anthropic_to_internal,
    convert_anthropic_to_internal_harmony,
    convert_anthropic_tools_to_internal,
    convert_internal_to_anthropic_response,
    create_content_block_start_event,
    create_content_block_stop_event,
    create_error_event,
    create_input_json_delta_event,
    create_message_delta_event,
    create_message_start_event,
    create_message_stop_event,
    create_text_delta_event,
    create_thinking_delta_event,
    map_finish_reason_to_stop_reason,
    request_has_cache_control,
)
from omlx_runtime.generation.thinking import (
    ThinkingParser,
    extract_thinking,
    prompt_opens_thinking,
)
from omlx_runtime.generation.tool_calling import (
    ToolCallStreamFilter,
    enrich_tool_params_for_gemma4,
    extract_tool_calls_with_thinking,
    restore_gemma4_param_names,
    sanitize_tool_call_markup,
)
from omlx_runtime.generation.utils import (
    cache_reasoning_output,
    clean_special_tokens,
    detect_and_strip_partial,
    prepare_system_messages_for_template,
    uses_native_reasoning_content,
)

# Import from new modular API
from omlx_server.api.parser_tool_calls import (
    convert_parser_tool_calls as _convert_parser_tool_calls,
)
from omlx_server.api.responses_utils import format_sse_event

from .dependencies import verify_inference_api_key
from .engine_requests import (
    _ensure_tokenizer_for_system_probe,
    _raise_if_llm_lease_abort_requested,
    _release_after_stream,
)
from .errors import _prefill_memory_error_detail
from .openai_streaming import (
    _render_chat_prompt_for_thinking_detection,
    _tool_call_failure,
)
from .shared import _LLMEngineLease, _ToolCallGenerationError
from .transport import (
    _aclose_async_iterator,
    _json_response_or_keepalive,
    _request_abort_id,
    _with_request_disconnect_abort,
    _with_sse_keepalive,
)

logger = logging.getLogger(__name__)


class AnthropicController:
    async def stream_anthropic_messages(
        self,
        engine: BaseEngine,
        messages: list,
        request: AnthropicMessagesRequest,
        resolved_model: str | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Stream Anthropic Messages API response.

        For Harmony models (gpt-oss), separates analysis and final channels:
        - index=0: analysis channel (<think>...</think>) - displayed as thinking
        - index=1: final channel (response text) - displayed as message

        For other models:
        - index=0: all text

        Emits events in Anthropic SSE format:
        1. message_start - Initial message
        2. content_block_start - Start block(s)
        3. content_block_delta - Text chunks
        4. content_block_stop - End block(s)
        5. (tool blocks if present)
        6. message_delta - Final stop_reason and usage
        7. message_stop - End marker
        """
        start_time = time.perf_counter()
        first_token_time = None

        message_id = f"msg_{uuid.uuid4().hex[:24]}"
        accumulated_text = ""

        # Track content blocks with thinking separation. Some templates open the
        # thinking block in the prompt itself, so the generated text starts with
        # reasoning body and only later emits </think>.
        start_in_thinking = False
        try:
            tokenizer = getattr(engine, "tokenizer", None)
            if tokenizer is not None:
                prompt, prompt_token_ids = _render_chat_prompt_for_thinking_detection(
                    engine, messages, kwargs
                )
                start_in_thinking, _ = prompt_opens_thinking(
                    tokenizer, prompt, prompt_token_ids=prompt_token_ids
                )
        except Exception as exc:
            logger.debug("Could not detect Anthropic stream thinking state: %s", exc)
        thinking_parser = ThinkingParser(start_in_thinking=start_in_thinking)
        thinking_block_started = False
        text_block_started = False
        block_index = 0
        last_output = None  # Track last output for tool_calls and token counts

        # Filter tool-call markup from streamed content when tools are present.
        has_tools = bool(kwargs.get("tools"))
        tool_filter = None
        thinking_filter = None
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

        # Does the client opt into Anthropic's cache_control accounting?
        # When yes, message_start.input_tokens reports the post-partition value
        # (0 here, since we approximate the whole prompt as belonging to the
        # cache_control region — the final message_delta refines with the real
        # cache hit count). When no, input_tokens carries the full prompt count.
        uses_cache_control = request_has_cache_control(request)

        # Calculate input tokens before streaming starts
        # This is needed for message_start event
        estimated_input_tokens = 0
        try:
            if hasattr(engine, "tokenizer") and engine.tokenizer is not None:
                # Build the prompt using chat template
                template_kwargs = {"tokenize": False, "add_generation_prompt": True}
                if kwargs.get("tools"):
                    template_kwargs["tools"] = kwargs["tools"]
                if kwargs.get("chat_template_kwargs"):
                    template_kwargs.update(kwargs["chat_template_kwargs"])
                prompt = engine.tokenizer.apply_chat_template(
                    messages, **template_kwargs
                )
                # Tokenize to count
                tokens = engine.tokenizer.encode(prompt)
                estimated_input_tokens = len(tokens)
        except Exception as e:
            logger.debug(f"Could not estimate input tokens: {e}")

        # 1. Send message_start with estimated input tokens
        yield create_message_start_event(
            message_id=message_id,
            model=request.model,
            input_tokens=(0 if uses_cache_control else estimated_input_tokens),
        )

        # 3. Stream content with thinking/content separation
        engine_stream = engine.stream_chat(messages=messages, **kwargs)
        try:
            async for output in engine_stream:
                last_output = output  # Keep reference for tool_calls and token counts

                if first_token_time is None and output.new_text:
                    first_token_time = time.perf_counter()

                if output.new_text:
                    accumulated_text += output.new_text
                    thinking_delta, content_delta = thinking_parser.feed(
                        output.new_text
                    )

                    # Emit thinking content as thinking block
                    if thinking_delta:
                        if thinking_filter:
                            thinking_delta = thinking_filter.feed(thinking_delta)
                        if thinking_delta:
                            # Close any open text block before starting a new
                            # thinking block at a fresh index. Anthropic SDKs
                            # reject mixed-type content_block events at the same
                            # index — this transition handles a model that emits
                            # a second thinking section after some text.
                            if text_block_started:
                                yield create_content_block_stop_event(index=block_index)
                                block_index += 1
                                text_block_started = False
                            if not thinking_block_started:
                                yield create_content_block_start_event(
                                    index=block_index, block_type="thinking"
                                )
                                thinking_block_started = True
                            yield create_thinking_delta_event(
                                index=block_index, thinking=thinking_delta
                            )

                    # Emit regular content as text block — filter tool-call
                    # markup when a known start marker is available.
                    if content_delta:
                        if tool_filter:
                            content_delta = tool_filter.feed(content_delta)
                        if content_delta:
                            # When tools are requested AND we haven't yet opened
                            # a text block, drop pure-whitespace deltas. Most
                            # models emit a leading newline around <tool_call>
                            # envelopes that tool_filter passes through (their
                            # whitespace isn't part of the envelope markers;
                            # DeepSeek V4's separator is, and the filter
                            # consumes it itself). Without this guard, the `\n`
                            # opens a text block that then holds only
                            # whitespace — surfacing as a phantom empty-ish
                            # text block before the tool_use blocks.
                            if (
                                not text_block_started
                                and kwargs.get("tools")
                                and not content_delta.strip()
                            ):
                                pass  # drop leading whitespace adjacent to tool envelopes
                            else:
                                # Close thinking block if transitioning to text
                                if thinking_block_started and not text_block_started:
                                    yield create_content_block_stop_event(
                                        index=block_index
                                    )
                                    block_index += 1
                                    thinking_block_started = False
                                if not text_block_started:
                                    yield create_content_block_start_event(
                                        index=block_index, block_type="text"
                                    )
                                    text_block_started = True
                                yield create_text_delta_event(
                                    index=block_index, text=content_delta
                                )

                if output.finished:
                    break
        except Exception as e:
            if isinstance(e, PrefillMemoryExceededError):
                # Same shadowing as the OpenAI generators (#3036): keep the
                # rejection classifiable. invalid_request_error is the Anthropic
                # terminal request-error type — a retry without shrinking the
                # prompt cannot succeed.
                logger.warning(f"Anthropic streaming prefill rejected: {e}")
                yield create_error_event(
                    "invalid_request_error", _prefill_memory_error_detail(e)
                )
            else:
                logger.error(f"Error during Anthropic streaming: {e}")
                yield create_error_event("api_error", str(e))
            yield create_message_stop_event()
            return

        finally:
            await _aclose_async_iterator(engine_stream)

        # Flush remaining buffered content from thinking parser
        thinking_delta, content_delta = thinking_parser.finish(
            truncated=last_output is not None and last_output.finish_reason == "length"
        )
        if thinking_delta:
            if thinking_filter:
                thinking_delta = thinking_filter.feed(thinking_delta)
            if thinking_delta:
                if text_block_started:
                    yield create_content_block_stop_event(index=block_index)
                    block_index += 1
                    text_block_started = False
                if not thinking_block_started:
                    yield create_content_block_start_event(
                        index=block_index, block_type="thinking"
                    )
                    thinking_block_started = True
                yield create_thinking_delta_event(
                    index=block_index, thinking=thinking_delta
                )
        if thinking_filter:
            remaining_thinking = thinking_filter.finish()
            if remaining_thinking:
                if text_block_started:
                    yield create_content_block_stop_event(index=block_index)
                    block_index += 1
                    text_block_started = False
                if not thinking_block_started:
                    yield create_content_block_start_event(
                        index=block_index, block_type="thinking"
                    )
                    thinking_block_started = True
                yield create_thinking_delta_event(
                    index=block_index, thinking=remaining_thinking
                )
        if content_delta:
            if tool_filter:
                content_delta = tool_filter.feed(content_delta)
            if content_delta:
                if thinking_block_started and not text_block_started:
                    yield create_content_block_stop_event(index=block_index)
                    block_index += 1
                    thinking_block_started = False
                if not text_block_started:
                    yield create_content_block_start_event(
                        index=block_index, block_type="text"
                    )
                    text_block_started = True
                yield create_text_delta_event(index=block_index, text=content_delta)

        # Flush any remaining buffered content from the tool-call filter
        if tool_filter:
            remaining = tool_filter.finish()
            # Same guard as the delta path above: the filter can now flush
            # held newlines here, and pure whitespace must not open a
            # phantom text block.
            if remaining and not (
                not text_block_started and kwargs.get("tools") and not remaining.strip()
            ):
                if not text_block_started:
                    if thinking_block_started:
                        yield create_content_block_stop_event(index=block_index)
                        block_index += 1
                        thinking_block_started = False
                    yield create_content_block_start_event(
                        index=block_index, block_type="text"
                    )
                    text_block_started = True
                yield create_text_delta_event(index=block_index, text=remaining)

        # 5. Handle tool calls (moved before block-closing so empty-text-block
        # emission can skip when tool_use blocks will follow).
        # For Harmony models, use tool_calls from output (parsed by HarmonyStreamingParser)
        # For other models, parse from accumulated text
        tool_calls = None
        tool_failure = None
        if last_output and last_output.tool_calls:
            # Protocol parser already extracted structured tool calls.
            tool_calls = _convert_parser_tool_calls(last_output.tool_calls)
        elif kwargs.get("tools"):
            # Non-Harmony: separate thinking, then parse tool calls from content
            # (falls back to thinking content for small models)
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
            tool_calls = extraction.tool_calls
            tool_failure = _tool_call_failure(extraction)

        recovered_thinking = (
            thinking_filter.take_recovery_candidate() if thinking_filter else ""
        )
        recovered_content = tool_filter.take_recovery_candidate() if tool_filter else ""
        if not tool_calls and not tool_failure:
            if recovered_thinking:
                if text_block_started:
                    yield create_content_block_stop_event(index=block_index)
                    block_index += 1
                    text_block_started = False
                if not thinking_block_started:
                    yield create_content_block_start_event(
                        index=block_index, block_type="thinking"
                    )
                    thinking_block_started = True
                yield create_thinking_delta_event(
                    index=block_index, thinking=recovered_thinking
                )
            if recovered_content:
                if thinking_block_started and not text_block_started:
                    yield create_content_block_stop_event(index=block_index)
                    block_index += 1
                    thinking_block_started = False
                if not text_block_started:
                    yield create_content_block_start_event(
                        index=block_index, block_type="text"
                    )
                    text_block_started = True
                yield create_text_delta_event(index=block_index, text=recovered_content)

        # 4. Close open blocks
        if thinking_block_started and not text_block_started:
            # Only thinking was emitted. Keep block_index on the just-closed
            # block so following tool_use blocks start at the next contiguous index.
            yield create_content_block_stop_event(index=block_index)
        if text_block_started:
            yield create_content_block_stop_event(index=block_index)
        elif not thinking_block_started and not tool_calls:
            # No content AND no tool_calls — emit an empty text block so the
            # message is well-formed. When tool_calls will follow, skip this —
            # the tool_use blocks carry the semantic content, and an empty
            # preceding text block confuses SDK clients that treat content[0]
            # as authoritative.
            yield create_content_block_start_event(index=block_index, block_type="text")
            yield create_content_block_stop_event(index=block_index)

        # Reverse Gemma 4 parameter renaming
        if tool_calls and "gemma" in (resolved_model or request.model or "").lower():
            for tc in tool_calls:
                if tc.function and tc.function.arguments:
                    try:
                        args = json.loads(tc.function.arguments)
                        args = restore_gemma4_param_names(args)
                        tc.function.arguments = json.dumps(args, ensure_ascii=False)
                    except (json.JSONDecodeError, AttributeError):
                        pass

        # Emit tool_use blocks if present
        # When neither text nor thinking was streamed AND the empty-text-block
        # emission was skipped (because tool_calls are about to follow), the
        # tool_use block takes index 0. Otherwise it follows the last emitted
        # text/thinking block at block_index+1.
        if not text_block_started and not thinking_block_started:
            tool_block_start = 0
        else:
            tool_block_start = block_index + 1
        if tool_calls:
            for i, tc in enumerate(tool_calls, start=tool_block_start):
                # Start tool_use block
                yield create_content_block_start_event(
                    index=i,
                    block_type="tool_use",
                    id=tc.id,
                    name=tc.function.name,
                )
                # Send input as delta
                yield create_input_json_delta_event(
                    index=i, partial_json=tc.function.arguments
                )
                # Close tool block
                yield create_content_block_stop_event(index=i)

        if tool_failure:
            error = tool_failure["error"]
            yield format_sse_event(
                "error",
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": error["message"],
                        "code": error["code"],
                    },
                },
            )
            yield create_message_stop_event()
            return

        # 6. Send message_delta with stop_reason and actual token counts
        stop_reason = map_finish_reason_to_stop_reason(
            output.finish_reason if output else "stop", bool(tool_calls)
        )
        # Use actual token counts from the last output
        actual_input_tokens = last_output.prompt_tokens if last_output else 0
        actual_output_tokens = last_output.completion_tokens if last_output else 0
        actual_cached_tokens = last_output.cached_tokens if last_output else 0
        yield create_message_delta_event(
            stop_reason=stop_reason,
            output_tokens=actual_output_tokens,
            input_tokens=actual_input_tokens,
            cached_tokens=actual_cached_tokens,
            request_uses_cache_control=uses_cache_control,
        )

        # Record metrics
        if last_output:
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
                f"Anthropic message: model={serving_model}, "
                f"{last_output.completion_tokens} tokens in {total_duration:.2f}s "
                f"({tokens_per_sec:.1f} tok/s)"
            )

        # 7. Send message_stop
        yield create_message_stop_event()

    async def create_anthropic_message(
        self,
        request: AnthropicMessagesRequest,
        http_request: FastAPIRequest,
        _: bool = Depends(verify_inference_api_key),
    ):
        """
        Create a message using Anthropic Messages API format.

        This endpoint provides compatibility with Anthropic's Messages API,
        allowing clients that use Anthropic SDK to work with oMLX.

        Example request:
        ```json
        {
            "model": "claude-3-sonnet",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ]
        }
        ```

        Streaming is supported with `stream: true`.
        """
        logger.debug(
            f"Anthropic Messages request: model={request.model}, "
            f"messages={len(request.messages)}, stream={request.stream}, "
            f"max_tokens={request.max_tokens}"
        )

        if self.state.oq_manager and self.state.oq_manager.is_quantizing:
            raise HTTPException(
                status_code=503,
                detail="Server is busy with oQ quantization. Please try again after quantization completes.",
            )

        lease = _LLMEngineLease(pool=self.state.engine_pool)
        try:
            engine = await self.get_engine_for_model(request.model, lease=lease)

            # Use the exact model selected by the pool, including fallback.
            resolved_model = self._serving_model_id(lease, request.model)

            # Get per-model settings
            max_tool_result_tokens = None
            ms = self.get_model_settings_for_request(request.model)
            if ms:
                max_tool_result_tokens = ms.max_tool_result_tokens
            merged_ct_kwargs = merge_chat_template_request_kwargs(
                ms,
                request.chat_template_kwargs,
            )
            forced_keys = forced_ct_keys(ms)

            # Pass Anthropic thinking config to chat template (except forced keys)
            if hasattr(request, "thinking") and request.thinking:
                if "enable_thinking" not in forced_keys:
                    thinking_type = getattr(request.thinking, "type", None)
                    if thinking_type in ("enabled", "adaptive"):
                        merged_ct_kwargs["enable_thinking"] = True
                    elif thinking_type == "disabled":
                        merged_ct_kwargs["enable_thinking"] = False

            _entry = self.get_engine_pool().get_entry(resolved_model)

            logger.debug(
                f"Tool result truncation config: max_tokens={max_tool_result_tokens}, "
                f"has_tokenizer={engine.tokenizer is not None}"
            )

            # Convert Anthropic format to internal format
            # Harmony models need special handling to preserve tool format
            is_vlm = isinstance(engine, VLMBatchedEngine)
            is_dflash_vlm = not is_vlm and getattr(
                engine, "supports_multimodal_fallback", False
            )
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
            if engine.model_type == "gpt_oss":
                messages = convert_anthropic_to_internal_harmony(
                    request,
                    max_tool_result_tokens,
                    engine.tokenizer,
                    consolidate_system_messages=False,
                )
            else:
                messages = convert_anthropic_to_internal(
                    request,
                    max_tool_result_tokens,
                    engine.tokenizer,
                    preserve_images=is_vlm or is_dflash_vlm,
                    native_reasoning_content=native_reasoning,
                    consolidate_system_messages=False,
                )

            # Apply model-specific message extraction (e.g. Gemma 4 converts
            # role=tool messages into tool_responses on assistant turns).
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
                    messages,
                    max_tool_result_tokens,
                    engine.tokenizer,
                    **extractor_kwargs,
                )
                merge_system_fallback_roles = True

            # Detect and strip partial mode at the API boundary — exactly once.
            is_partial = detect_and_strip_partial(messages)

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
                req_max_tokens=request.max_tokens,
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

            # Add thinking budget if applicable
            thinking_budget = self._resolve_thinking_budget(request, request.model)
            if thinking_budget is not None:
                chat_kwargs["thinking_budget"] = thinking_budget

            # Auto-set enable_thinking in chat template kwargs when a positive thinking
            # budget is active but enable_thinking was not already set (e.g. via
            # the Anthropic thinking.type field above or model settings).
            if (
                thinking_budget is not None
                and thinking_budget > 0
                and "enable_thinking" not in merged_ct_kwargs
            ):
                merged_ct_kwargs["enable_thinking"] = True

            # Auto-set preserve_thinking only when the template advertises support
            # for it (Qwen 3.6+). Gated on detection so other templates don't
            # receive an unknown kwarg.
            _entry = self.get_engine_pool().get_entry(resolved_model)
            if (
                _entry is not None
                and _entry.preserve_thinking_default is True
                and merged_ct_kwargs.get("enable_thinking") is not False
                and "preserve_thinking" not in merged_ct_kwargs
            ):
                merged_ct_kwargs["preserve_thinking"] = True

            # Merge MCP tools with user-provided Anthropic tools
            user_internal = convert_anthropic_tools_to_internal(request.tools)
            if getattr(engine, "is_diffusion_model", False) and not getattr(
                engine, "supports_tool_calling", False
            ):
                if user_internal:
                    raise InvalidRequestError(
                        "Tool calling is not supported for this diffusion model "
                        "(no tool parser matched its chat template).",
                        field="tools",
                    )
                internal_tools = None
            elif self.state.mcp_manager and self.mcp_tools_exposed():
                mcp_openai_tools = self.state.mcp_manager.get_all_tools_openai()
                combined = (mcp_openai_tools or []) + (user_internal or [])
                # Deduplicate by function name (user tools take precedence)
                if combined:
                    seen = {}
                    for tool in combined:
                        name = tool.get("function", {}).get("name", "")
                        seen[name] = tool
                    internal_tools = list(seen.values())
                else:
                    internal_tools = None
            else:
                internal_tools = user_internal
            # Gemma 4 drops required params that lack descriptions — enrich them
            if internal_tools and "gemma" in (resolved_model or "").lower():
                internal_tools = enrich_tool_params_for_gemma4(internal_tools)
            if internal_tools:
                chat_kwargs["tools"] = internal_tools

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

            await _ensure_tokenizer_for_system_probe(engine, messages)
            messages = prepare_system_messages_for_template(
                messages,
                engine.tokenizer,
                tools=internal_tools,
                chat_template_kwargs=merged_ct_kwargs or None,
                is_partial=is_partial,
                merge_consecutive_roles=merge_system_fallback_roles,
                unsupported_mid_system_policy=self._unsupported_mid_system_policy(),
            )

            # Validate context window before sending to model
            try:
                num_prompt_tokens = engine.count_chat_tokens(
                    messages,
                    internal_tools,
                    chat_template_kwargs=merged_ct_kwargs or None,
                    is_partial=is_partial,
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

            # Add stop sequences
            if request.stop_sequences:
                chat_kwargs["stop"] = request.stop_sequences

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
                return StreamingResponse(
                    _release_after_stream(
                        _with_request_disconnect_abort(
                            _with_sse_keepalive(
                                self.stream_anthropic_messages(
                                    engine,
                                    messages,
                                    request,
                                    resolved_model=resolved_model,
                                    **chat_kwargs,
                                ),
                                http_request=http_request,
                                keepalive_chunk=self._resolve_keepalive("anthropic"),
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
            async def _build_anthropic_message():
                await _raise_if_llm_lease_abort_requested(lease)
                start_time = time.perf_counter()

                output = await engine.chat(messages=messages, **chat_kwargs)

                elapsed = time.perf_counter() - start_time
                tokens_per_sec = (
                    output.completion_tokens / elapsed if elapsed > 0 else 0
                )
                logger.info(
                    f"Anthropic message: model={resolved_model}, "
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
                        tools=internal_tools,
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
                        if tc.function and tc.function.arguments:
                            try:
                                args = json.loads(tc.function.arguments)
                                args = restore_gemma4_param_names(args)
                                tc.function.arguments = json.dumps(
                                    args, ensure_ascii=False
                                )
                            except (json.JSONDecodeError, AttributeError):
                                pass

                response = convert_internal_to_anthropic_response(
                    text=cleaned_text.strip() if cleaned_text else "",
                    model=request.model,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    finish_reason=output.finish_reason,
                    tool_calls=tool_calls,
                    thinking=cleaned_thinking if cleaned_thinking else None,
                    cached_tokens=output.cached_tokens,
                    request_uses_cache_control=request_has_cache_control(request),
                )

                return response.model_dump_json()

            return await _json_response_or_keepalive(
                http_request, _build_anthropic_message(), lease=lease
            )

        except BaseException:
            await lease.release()
            raise

    async def count_anthropic_tokens(
        self,
        request: TokenCountRequest,
        _: bool = Depends(verify_inference_api_key),
    ):
        """
        Count tokens in a message request.

        Uses the loaded model's tokenizer to accurately count tokens
        including system prompt, messages, and tools.

        This is compatible with Anthropic's token counting API.
        """
        if self.state.oq_manager and self.state.oq_manager.is_quantizing:
            raise HTTPException(
                status_code=503,
                detail="Server is busy with oQ quantization. Please try again after quantization completes.",
            )

        lease = _LLMEngineLease(pool=self.state.engine_pool)
        try:
            engine = await self.get_engine_for_model(request.model, lease=lease)
            await _raise_if_llm_lease_abort_requested(lease)

            # Convert Anthropic format to internal format
            # Create a temporary MessagesRequest to reuse existing conversion logic
            temp_request = AnthropicMessagesRequest(
                model=request.model,
                max_tokens=1,  # Dummy value, not used for token counting
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                thinking=request.thinking,
            )
            messages = convert_anthropic_to_internal(temp_request)

            # Convert tools if present
            internal_tools = convert_anthropic_tools_to_internal(request.tools)

            # Apply chat template to get prompt
            tokenizer = engine.tokenizer
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if internal_tools:
                template_kwargs["tools"] = internal_tools

            try:
                prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
            except Exception as e:
                logger.warning(
                    f"Failed to apply chat template: {e}, using simple concatenation"
                )
                # Fallback: simple concatenation
                prompt = "\n".join(
                    f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                    for msg in messages
                )

            # Tokenize to count tokens
            if isinstance(prompt, str):
                token_ids = tokenizer.encode(prompt)
            else:
                token_ids = prompt  # Already tokenized

            input_tokens = len(token_ids)
            logger.debug(
                f"Token count: {input_tokens} tokens for {len(messages)} messages"
            )

            return TokenCountResponse(input_tokens=input_tokens)

        finally:
            await lease.release()
