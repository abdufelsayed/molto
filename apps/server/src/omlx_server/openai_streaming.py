# SPDX-License-Identifier: Apache-2.0
"""Openai streaming for the private inference application."""

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

# Import from new modular API
from omlx_contracts.api.openai_models import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionRequest,
    CompletionRequest,
    PromptTokensDetails,
    Usage,
)
from omlx_runtime.engine import BaseEngine
from omlx_runtime.generation.thinking import (
    ThinkingParser,
    extract_thinking,
    prompt_opens_thinking,
)
from omlx_runtime.generation.tool_calling import (
    ToolCallExtraction,
    ToolCallStreamFilter,
    ToolCallStreamSegment,
    extract_tool_calls_with_thinking,
    parse_json_output,
    parse_qwen_tool_calls,
    restore_gemma4_param_names,
)

from omlx_server.api.parser_tool_calls import (
    convert_parser_tool_calls as _convert_parser_tool_calls,
)

from .engine_requests import (
    _format_generation_speed_for_log,
    _resolve_metric_durations,
    _strip_synthetic_think_prefix,
)
from .errors import _openai_error_body, _streaming_error_payload
from .transport import _aclose_async_iterator

logger = logging.getLogger(__name__)


def _copy_chat_template_messages(messages: list) -> list:
    return [
        dict(message) if isinstance(message, dict) else message for message in messages
    ]


def _render_chat_prompt_for_thinking_detection(
    engine: BaseEngine,
    messages: list,
    kwargs: dict,
) -> tuple[str, list[int] | None]:
    tokenizer = getattr(engine, "tokenizer", None)
    if tokenizer is None:
        return "", None

    template_messages = _copy_chat_template_messages(messages)
    tools = kwargs.get("tools")
    chat_template_kwargs = kwargs.get("chat_template_kwargs")
    is_partial = kwargs.get("is_partial")
    engine_renderer = getattr(engine, "_apply_chat_template", None)

    if is_partial is not None:
        for message in template_messages:
            if isinstance(message, dict):
                message.pop("partial", None)

    if callable(engine_renderer):
        prompt = engine_renderer(
            template_messages,
            tools,
            chat_template_kwargs=chat_template_kwargs,
            is_partial=is_partial,
        )
    else:
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": not bool(is_partial),
        }
        if is_partial:
            template_kwargs["continue_final_message"] = True
        if tools:
            template_kwargs["tools"] = tools
        if chat_template_kwargs:
            template_kwargs.update(chat_template_kwargs)

        try:
            prompt = tokenizer.apply_chat_template(template_messages, **template_kwargs)
        except TypeError:
            if chat_template_kwargs:
                for key in chat_template_kwargs:
                    template_kwargs.pop(key, None)
            template_kwargs.pop("tools", None)
            template_kwargs.pop("enable_thinking", None)
            prompt = tokenizer.apply_chat_template(template_messages, **template_kwargs)

    if isinstance(prompt, str):
        return prompt, None
    if isinstance(prompt, list):
        try:
            return "", [int(token_id) for token_id in prompt]
        except (TypeError, ValueError):
            return str(prompt), None
    return str(prompt), None


def _tool_call_failure(extraction: ToolCallExtraction) -> dict | None:
    failed = extraction.parse_errors
    if not failed:
        return None
    code = "incomplete_tool_call" if "incomplete" in failed else "invalid_tool_call"
    message = (
        "Model output contains an unrecoverable tool call. "
        "Previously delivered tool calls must not be executed again on retry."
    )
    logger.warning(
        "Tool call generation failed: code=%s, failed_calls=%d", code, len(failed)
    )
    return _openai_error_body(message, 500, code=code)


def _registered_tool_names(tools: object) -> set[str]:
    """Return nonempty function names explicitly registered by the request."""

    names: set[str] = set()
    for tool in tools or []:
        function = (
            tool.get("function")
            if isinstance(tool, dict)
            else getattr(tool, "function", None)
        )
        name = (
            function.get("name")
            if isinstance(function, dict)
            else getattr(function, "name", None)
        )
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _tool_call_semantic_key(tool_call: object) -> tuple[str, str] | None:
    """Canonical name/JSON-object identity, or ``None`` when malformed.

    Unknown names remain callable output for client-side error feedback.
    This is syntactic validation, not full JSON Schema argument validation.
    """

    function = getattr(tool_call, "function", None)
    name = getattr(function, "name", None)
    arguments = getattr(function, "arguments", None)
    if not isinstance(name, str) or not name or not isinstance(arguments, str):
        return None
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    canonical = json.dumps(
        parsed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return name, canonical


def _chat_can_stream_qwen_tool_envelopes(engine: BaseEngine) -> bool:
    """Narrow Chat-only early-tool gate.

    The engine capability is authoritative and defaults false. The tokenizer
    must independently expose mlx-lm's qwen3_coder parser; all other APIs and
    parser families remain terminal-buffered.
    """

    if getattr(engine, "supports_early_tool_call_streaming", False) is not True:
        return False
    tokenizer = getattr(engine, "tokenizer", None)
    parser = getattr(tokenizer, "tool_parser", None)
    try:
        from mlx_lm.tool_parsers.qwen3_coder import (
            parse_tool_call as expected_parser,
        )
    except ImportError:
        return False
    return bool(
        parser is expected_parser
        and getattr(parser, "__name__", None) == "parse_tool_call"
        and getattr(parser, "__module__", None) == "mlx_lm.tool_parsers.qwen3_coder"
    )


def _merge_streamed_tool_call_prefix(streamed: list, terminal: list | None) -> list:
    """Keep validated early calls as an occurrence-aware semantic prefix."""

    remaining = list(terminal or [])
    merged = list(streamed)
    for early in streamed:
        key = _tool_call_semantic_key(early)
        if key is None:
            continue
        for index, candidate in enumerate(remaining):
            if _tool_call_semantic_key(candidate) == key:
                remaining.pop(index)
                break
    merged.extend(remaining)
    return merged


class OpenaiStreamingController:
    async def stream_chat_completion(
        self,
        engine: BaseEngine,
        messages: list,
        request: ChatCompletionRequest,
        model_load_duration: float = 0.0,
        resolved_model: str | None = None,
        response_id: str | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream chat completion response.

        Streams content tokens with reasoning/thinking separation, then at
        completion parses tool calls from accumulated text and emits them
        as structured tool_calls chunks (OpenAI streaming format).
        """
        start_time = time.perf_counter()
        first_token_time = None
        first_visible_time = None
        last_output = None
        accumulated_text = ""
        has_tools = bool(kwargs.get("tools"))
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
            logger.debug("Could not detect chat stream thinking state: %s", exc)
        thinking_parser = ThinkingParser(start_in_thinking=start_in_thinking)

        def mark_visible_delta() -> None:
            nonlocal first_visible_time
            if first_visible_time is None:
                first_visible_time = time.perf_counter()

        # Reuse the id pre-minted by the caller (so the keepalive frame can share
        # it); otherwise mint one for direct/non-streaming callers.
        response_id = response_id or f"chatcmpl-{uuid.uuid4().hex[:8]}"

        # First chunk with role
        first_chunk = ChatCompletionChunk(
            id=response_id,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(role="assistant"),
                )
            ],
        )
        yield f"data: {first_chunk.model_dump_json(exclude_none=True)}\n\n"

        # Stream content token-by-token. When tools are present, a
        # ToolCallStreamFilter suppresses known tool-call control markup so
        # clients do not see raw envelopes/tags in assistant content deltas.
        tool_filter = None
        thinking_filter = None
        streamed_tool_calls = []
        stream_tool_sequence_safe = True
        stream_completed_qwen_tools = False
        qwen_tool_envelope_streaming_capable = False
        registered_tool_names: set[str] = set()
        stream_content = True
        if has_tools:
            registered_tool_names = _registered_tool_names(kwargs.get("tools"))
            qwen_tool_envelope_streaming_capable = bool(
                registered_tool_names and _chat_can_stream_qwen_tool_envelopes(engine)
            )
            stream_completed_qwen_tools = qwen_tool_envelope_streaming_capable
            _content_filter = ToolCallStreamFilter(
                engine.tokenizer,
                tools=kwargs.get("tools"),
                capture_ordered_segments=stream_completed_qwen_tools,
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
                if first_token_time is None:
                    produced_at = getattr(output, "first_token_at", None)
                    if produced_at is None:
                        produced_at = getattr(output, "generated_at", None)
                    if produced_at is not None:
                        first_token_time = float(produced_at)
                    elif getattr(output, "completion_tokens", 0) > 0 or output.new_text:
                        # Engines without producer timestamps can only expose the
                        # exact API-observation time. Never substitute end-of-turn.
                        first_token_time = time.perf_counter()
                last_output = output
                if output.tool_calls:
                    # A structured producer is authoritative. Correct engines keep
                    # the explicit capability false; this guard also prevents a
                    # same-output raw envelope from racing its structured result.
                    stream_completed_qwen_tools = False
                    stream_tool_sequence_safe = False
                if output.new_text:
                    accumulated_text += output.new_text

                if stream_content and output.new_text:
                    thinking_delta, content_delta = thinking_parser.feed(
                        output.new_text
                    )

                    # Emit reasoning_content delta
                    if thinking_delta:
                        if thinking_filter:
                            thinking_delta = thinking_filter.feed(thinking_delta)
                            # Thinking-channel calls are terminal fallback only:
                            # content-channel calls take precedence, so they cannot
                            # be streamed safely before the turn finishes.
                            thinking_filter.take_completed_envelopes()
                        chunk = ChatCompletionChunk(
                            id=response_id,
                            model=request.model,
                            choices=[
                                ChatCompletionChunkChoice(
                                    delta=ChatCompletionChunkDelta(
                                        reasoning_content=thinking_delta
                                    ),
                                    finish_reason=None,
                                )
                            ],
                        )
                        if thinking_delta:
                            event = (
                                f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                            )
                            mark_visible_delta()
                            yield event

                    # Emit content delta — filter out tool-call markup when
                    # tools are present so clients see clean streamed text.
                    if content_delta:
                        ordered_segments: list[ToolCallStreamSegment] = []
                        if tool_filter:
                            content_delta = tool_filter.feed(content_delta)
                            if stream_completed_qwen_tools:
                                ordered_segments = tool_filter.take_ordered_segments()
                                # The legacy completed queue shares the same string
                                # objects; drain it so ordered capture adds no
                                # retained duplicate state.
                                tool_filter.take_completed_envelopes()
                                if tool_filter.completed_envelope_overflowed:
                                    logger.warning(
                                        "Early qwen tool streaming disabled for this "
                                        "Chat turn: completed-envelope queue exceeded "
                                        "its count or byte bound"
                                    )
                                    stream_completed_qwen_tools = False
                                    stream_tool_sequence_safe = False
                                    ordered_segments = [
                                        segment
                                        for segment in ordered_segments
                                        if segment.kind == "content"
                                    ]
                            else:
                                tool_filter.take_ordered_segments()
                                tool_filter.take_completed_envelopes()
                        if not ordered_segments and content_delta:
                            ordered_segments = [
                                ToolCallStreamSegment("content", content_delta)
                            ]

                        for segment in ordered_segments:
                            if segment.kind == "content":
                                chunk = ChatCompletionChunk(
                                    id=response_id,
                                    model=request.model,
                                    choices=[
                                        ChatCompletionChunkChoice(
                                            delta=ChatCompletionChunkDelta(
                                                content=segment.text
                                            ),
                                            finish_reason=None,
                                        )
                                    ],
                                )
                                event = (
                                    f"data: {chunk.model_dump_json(exclude_none=True)}"
                                    "\n\n"
                                )
                                mark_visible_delta()
                                yield event
                                continue

                            if segment.kind != "envelope":
                                stream_tool_sequence_safe = False
                                continue
                            _, completed_calls, _ = parse_qwen_tool_calls(
                                segment.text,
                                engine.tokenizer,
                                kwargs.get("tools"),
                                finish_reason="stop",
                            )
                            completed_calls = completed_calls or []
                            completed_keys = [
                                _tool_call_semantic_key(tc) for tc in completed_calls
                            ]
                            if not completed_calls or any(
                                key is None for key in completed_keys
                            ):
                                stream_tool_sequence_safe = False
                                continue
                            if not stream_tool_sequence_safe:
                                continue
                            for tc in completed_calls:
                                index = len(streamed_tool_calls)
                                streamed_tool_calls.append(tc)
                                tc_chunk = ChatCompletionChunk(
                                    id=response_id,
                                    model=request.model,
                                    choices=[
                                        ChatCompletionChunkChoice(
                                            delta=ChatCompletionChunkDelta(
                                                tool_calls=[
                                                    {
                                                        "index": index,
                                                        "id": tc.id,
                                                        "type": "function",
                                                        "function": {
                                                            "name": tc.function.name,
                                                            "arguments": (
                                                                tc.function.arguments
                                                            ),
                                                        },
                                                    }
                                                ]
                                            ),
                                            finish_reason=None,
                                        )
                                    ],
                                )
                                event = (
                                    f"data: {tc_chunk.model_dump_json(exclude_none=True)}"
                                    "\n\n"
                                )
                                mark_visible_delta()
                                yield event
        except Exception as e:
            error_data = _streaming_error_payload(e, "chat streaming")
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"
            return

        finally:
            await _aclose_async_iterator(engine_stream)

        # Flush remaining buffered content from thinking/tool-call parsers
        if stream_content:
            thinking_delta, content_delta = thinking_parser.finish(
                truncated=last_output is not None
                and last_output.finish_reason == "length"
            )
            if thinking_delta:
                if thinking_filter:
                    thinking_delta = thinking_filter.feed(thinking_delta)
                if thinking_delta:
                    chunk = ChatCompletionChunk(
                        id=response_id,
                        model=request.model,
                        choices=[
                            ChatCompletionChunkChoice(
                                delta=ChatCompletionChunkDelta(
                                    reasoning_content=thinking_delta
                                ),
                                finish_reason=None,
                            )
                        ],
                    )
                    event = f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                    mark_visible_delta()
                    yield event
            if thinking_filter:
                remaining_thinking = thinking_filter.finish()
                if remaining_thinking:
                    chunk = ChatCompletionChunk(
                        id=response_id,
                        model=request.model,
                        choices=[
                            ChatCompletionChunkChoice(
                                delta=ChatCompletionChunkDelta(
                                    reasoning_content=remaining_thinking
                                ),
                                finish_reason=None,
                            )
                        ],
                    )
                    event = f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                    mark_visible_delta()
                    yield event
            if content_delta:
                if tool_filter:
                    content_delta = tool_filter.feed(content_delta)
                if content_delta:
                    chunk = ChatCompletionChunk(
                        id=response_id,
                        model=request.model,
                        choices=[
                            ChatCompletionChunkChoice(
                                delta=ChatCompletionChunkDelta(content=content_delta),
                                finish_reason=None,
                            )
                        ],
                    )
                    event = f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                    mark_visible_delta()
                    yield event

            if tool_filter:
                remaining = tool_filter.finish()
                if remaining:
                    chunk = ChatCompletionChunk(
                        id=response_id,
                        model=request.model,
                        choices=[
                            ChatCompletionChunkChoice(
                                delta=ChatCompletionChunkDelta(content=remaining),
                                finish_reason=None,
                            )
                        ],
                    )
                    event = f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                    mark_visible_delta()
                    yield event

        # Parse tool calls from accumulated text
        tool_calls = None
        tool_failure = None
        cleaned_text = accumulated_text
        terminal_tool_calls_authoritative = bool(last_output and last_output.tool_calls)
        if last_output and last_output.tool_calls:
            # Protocol parser already extracted structured tool calls.
            tool_calls = _convert_parser_tool_calls(last_output.tool_calls)
            cleaned_text = ""
        elif has_tools and accumulated_text:
            # Separate thinking from content, then parse tool calls from content
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
            cleaned_text = extraction.cleaned_text
            tool_calls = extraction.tool_calls
            tool_failure = _tool_call_failure(extraction)
            cleaned_thinking = extraction.cleaned_thinking
            # Process response_format if specified
            if request.response_format and not tool_calls:
                cleaned_text, parsed_json, is_valid, error = parse_json_output(
                    cleaned_text, request.response_format
                )
                if parsed_json is not None:
                    cleaned_text = json.dumps(parsed_json)
                if not is_valid:
                    logger.warning(f"JSON validation failed: {error}")

            # Buffered mode: emit thinking and cleaned content now
            if not stream_content:
                if cleaned_thinking:
                    chunk = ChatCompletionChunk(
                        id=response_id,
                        model=request.model,
                        choices=[
                            ChatCompletionChunkChoice(
                                delta=ChatCompletionChunkDelta(
                                    reasoning_content=cleaned_thinking
                                ),
                                finish_reason=None,
                            )
                        ],
                    )
                    event = f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                    mark_visible_delta()
                    yield event
                if cleaned_text:
                    chunk = ChatCompletionChunk(
                        id=response_id,
                        model=request.model,
                        choices=[
                            ChatCompletionChunkChoice(
                                delta=ChatCompletionChunkDelta(content=cleaned_text),
                                finish_reason=None,
                            )
                        ],
                    )
                    event = f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                    mark_visible_delta()
                    yield event

        # Surface an unterminated paired envelope only when final parsing could not
        # recover a structured tool call. The candidate begins at the opening marker,
        # so prose already streamed before it is never duplicated.
        recovered_thinking = (
            thinking_filter.take_recovery_candidate() if thinking_filter else ""
        )
        recovered_content = tool_filter.take_recovery_candidate() if tool_filter else ""
        if not tool_calls and not tool_failure:
            if recovered_thinking:
                chunk = ChatCompletionChunk(
                    id=response_id,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(
                                reasoning_content=recovered_thinking
                            ),
                            finish_reason=None,
                        )
                    ],
                )
                event = f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                mark_visible_delta()
                yield event
            if recovered_content:
                chunk = ChatCompletionChunk(
                    id=response_id,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(content=recovered_content),
                            finish_reason=None,
                        )
                    ],
                )
                event = f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                mark_visible_delta()
                yield event

        # A qwen3_coder raw-envelope stream has no engine-side structured parser,
        # so preserve each already-emitted validated occurrence even if malformed
        # later markup makes terminal extraction partial. Structured engine output
        # is authoritative and, by capability contract, can never race this path.
        if streamed_tool_calls and not terminal_tool_calls_authoritative:
            tool_calls = _merge_streamed_tool_call_prefix(
                streamed_tool_calls,
                tool_calls,
            )

        # Reconcile by canonical JSON semantics plus occurrence—not raw argument
        # formatting or list position. Repeated identical calls remain distinct.
        streamed_by_fingerprint: dict[tuple[str, str], list] = {}
        reconcilable_streamed = (
            [] if terminal_tool_calls_authoritative else streamed_tool_calls
        )
        for streamed in reconcilable_streamed:
            fingerprint = _tool_call_semantic_key(streamed)
            if fingerprint is not None:
                streamed_by_fingerprint.setdefault(fingerprint, []).append(streamed)
        streamed_tool_call_ids: set[str] = set()
        for final in tool_calls or []:
            fingerprint = _tool_call_semantic_key(final)
            if fingerprint is None:
                continue
            candidates = streamed_by_fingerprint.get(fingerprint) or []
            if candidates:
                streamed = candidates.pop(0)
                final.id = streamed.id
                streamed_tool_call_ids.add(streamed.id)

        # Reverse Gemma 4 parameter renaming for streaming path
        if tool_calls and "gemma" in (resolved_model or request.model or "").lower():
            for tc in tool_calls:
                if tc.function and tc.function.arguments:
                    try:
                        args = json.loads(tc.function.arguments)
                        args = restore_gemma4_param_names(args)
                        tc.function.arguments = json.dumps(args, ensure_ascii=False)
                    except (json.JSONDecodeError, AttributeError):
                        pass

        # Emit tool call chunks if found
        if tool_calls:
            for i, tc in enumerate(tool_calls):
                if tc.id in streamed_tool_call_ids:
                    continue
                tc_chunk = ChatCompletionChunk(
                    id=response_id,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            delta=ChatCompletionChunkDelta(
                                tool_calls=[
                                    {
                                        "index": i,
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments,
                                        },
                                    }
                                ],
                            ),
                        )
                    ],
                )
                event = f"data: {tc_chunk.model_dump_json(exclude_none=True)}\n\n"
                mark_visible_delta()
                yield event

        if tool_failure:
            yield f"data: {json.dumps(tool_failure)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Final chunk with finish_reason
        finish_reason = (
            "tool_calls"
            if tool_calls
            else (last_output.finish_reason if last_output else "stop")
        )
        final_chunk = ChatCompletionChunk(
            id=response_id,
            model=request.model,
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChunkDelta(),
                    finish_reason=finish_reason,
                )
            ],
        )
        yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"

        # Record metrics and emit usage chunk
        if last_output and last_output.finished:
            end_time = time.perf_counter()
            total_duration = end_time - start_time
            model_ttft = (
                max(0.0, first_token_time - start_time)
                if first_token_time is not None
                else None
            )
            visible_ttft = (
                max(0.0, first_visible_time - start_time)
                if first_visible_time is not None
                else None
            )
            is_diffusion = getattr(engine, "is_diffusion_model", False)
            if is_diffusion:
                gen_duration = total_duration
            else:
                gen_duration = max(
                    0.0,
                    end_time
                    - (
                        first_token_time if first_token_time is not None else start_time
                    ),
                )
            metric_prefill_duration, metric_gen_duration = _resolve_metric_durations(
                last_output,
                is_diffusion=is_diffusion,
                prefill_duration=(
                    model_ttft if model_ttft is not None else total_duration
                ),
                generation_duration=gen_duration,
            )
            self.get_server_metrics().record_request_complete(
                prompt_tokens=last_output.prompt_tokens,
                completion_tokens=last_output.completion_tokens,
                cached_tokens=last_output.cached_tokens,
                prefill_duration=metric_prefill_duration,
                generation_duration=metric_gen_duration,
                model_id=resolved_model or request.model,
                request_duration=total_duration,
            )
            speed_duration = total_duration if is_diffusion else gen_duration
            tokens_per_sec = (
                last_output.completion_tokens / speed_duration
                if speed_duration > 0
                else 0
            )
            speed_text = _format_generation_speed_for_log(
                last_output,
                tokens_per_sec,
                is_diffusion=is_diffusion,
            )
            model_ttft_text = (
                f"{model_ttft:.2f}s" if model_ttft is not None else "unavailable"
            )
            visible_ttft_text = (
                f"{visible_ttft:.2f}s" if visible_ttft is not None else "unavailable"
            )
            logger.info(
                f"Chat completion: model={resolved_model or request.model}, "
                f"{last_output.completion_tokens} tokens in "
                f"{total_duration:.2f}s ({speed_text}), "
                f"prompt: {last_output.prompt_tokens}, finish_reason={finish_reason}, "
                f"max_tokens={kwargs.get('max_tokens')}, "
                f"request_max_tokens={request.max_tokens}, "
                f"stream_model_ttft={model_ttft_text}, "
                f"stream_visible_ttft={visible_ttft_text}"
            )

            # Emit usage chunk if requested
            if request.stream_options and request.stream_options.include_usage:
                total_time = end_time - start_time
                pt = last_output.prompt_tokens
                ct = last_output.completion_tokens
                usage_chunk = ChatCompletionChunk(
                    id=response_id,
                    model=request.model,
                    choices=[],
                    usage=Usage(
                        prompt_tokens=pt,
                        completion_tokens=ct,
                        total_tokens=pt + ct,
                        prompt_tokens_details=PromptTokensDetails(
                            cached_tokens=last_output.cached_tokens,
                        ),
                        model_load_duration=(
                            round(model_load_duration, 2)
                            if model_load_duration > 1.0
                            else None
                        ),
                        time_to_first_token=(
                            round(model_ttft, 2) if model_ttft is not None else None
                        ),
                        time_to_first_visible_token=(
                            round(visible_ttft, 2) if visible_ttft is not None else None
                        ),
                        total_time=round(total_time, 2),
                        prompt_eval_duration=round(metric_prefill_duration, 2),
                        generation_duration=round(metric_gen_duration, 2),
                        prompt_tokens_per_second=(
                            round(pt / metric_prefill_duration, 2)
                            if metric_prefill_duration > 0
                            else None
                        ),
                        generation_tokens_per_second=(
                            round(ct / metric_gen_duration, 2)
                            if metric_gen_duration > 0
                            else None
                        ),
                    ),
                )
                yield f"data: {usage_chunk.model_dump_json(exclude_none=True)}\n\n"

        yield "data: [DONE]\n\n"

    async def stream_completion(
        self,
        engine: BaseEngine,
        prompt: str,
        request: CompletionRequest,
        model_load_duration: float = 0.0,
        prompt_token_ids: list[int] | None = None,
        resolved_model: str | None = None,
        response_id: str | None = None,
        inference_request_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream completion response."""
        response_id = response_id or f"cmpl-{uuid.uuid4().hex[:8]}"
        start_time = time.perf_counter()
        first_token_time = None
        last_output = None
        # Parity with the non-streaming path: when the prompt opens a thinking
        # block, the first chunk carries the scheduler's synthetic think opener;
        # strip it once so the stream is a pure continuation of the prompt.
        pending_think_prefix_strip, think_tag = prompt_opens_thinking(
            getattr(engine, "tokenizer", None),
            prompt,
            prompt_token_ids=prompt_token_ids,
        )

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
        repetition_context_size = getattr(request, "repetition_context_size", None)
        if repetition_context_size is not None:
            gen_kwargs["repetition_context_size"] = repetition_context_size
        try:
            async for output in engine.stream_generate(
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
            ):
                if first_token_time is None and output.new_text:
                    first_token_time = time.perf_counter()
                last_output = output

                chunk_text = output.new_text
                if pending_think_prefix_strip and chunk_text:
                    chunk_text = _strip_synthetic_think_prefix(chunk_text, think_tag)
                    pending_think_prefix_strip = False

                data = {
                    "id": response_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "text": chunk_text,
                            "finish_reason": (
                                output.finish_reason if output.finished else None
                            ),
                        }
                    ],
                }
                yield f"data: {json.dumps(data)}\n\n"
        except Exception as e:
            error_data = _streaming_error_payload(e, "completion streaming")
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Record metrics
        if last_output and last_output.finished:
            end_time = time.perf_counter()
            total_duration = end_time - start_time
            ttft = (
                (first_token_time - start_time) if first_token_time else total_duration
            )
            is_diffusion = getattr(engine, "is_diffusion_model", False)
            if is_diffusion:
                gen_duration = total_duration
            else:
                gen_duration = end_time - (first_token_time or start_time)
            metric_prefill_duration, metric_gen_duration = _resolve_metric_durations(
                last_output,
                is_diffusion=is_diffusion,
                prefill_duration=ttft,
                generation_duration=gen_duration,
            )
            serving_model = (
                resolved_model or self.resolve_model_id(request.model) or request.model
            )
            self.get_server_metrics().record_request_complete(
                prompt_tokens=last_output.prompt_tokens,
                completion_tokens=last_output.completion_tokens,
                cached_tokens=last_output.cached_tokens,
                prefill_duration=metric_prefill_duration,
                generation_duration=metric_gen_duration,
                model_id=serving_model,
                request_duration=total_duration,
            )
            speed_duration = total_duration if is_diffusion else gen_duration
            tokens_per_sec = (
                last_output.completion_tokens / speed_duration
                if speed_duration > 0
                else 0
            )
            speed_text = _format_generation_speed_for_log(
                last_output,
                tokens_per_sec,
                is_diffusion=is_diffusion,
            )
            logger.info(
                f"Completion: model={serving_model}, "
                f"{last_output.completion_tokens} tokens in "
                f"{total_duration:.2f}s ({speed_text}), "
                f"prompt: {last_output.prompt_tokens}"
            )

            # Emit usage chunk if requested
            if request.stream_options and request.stream_options.include_usage:
                total_time = end_time - start_time
                pt = last_output.prompt_tokens
                ct = last_output.completion_tokens
                usage_data = {
                    "id": response_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [],
                    "usage": Usage(
                        prompt_tokens=pt,
                        completion_tokens=ct,
                        total_tokens=pt + ct,
                        prompt_tokens_details=PromptTokensDetails(
                            cached_tokens=last_output.cached_tokens,
                        ),
                        model_load_duration=(
                            round(model_load_duration, 2)
                            if model_load_duration > 1.0
                            else None
                        ),
                        time_to_first_token=round(ttft, 2),
                        total_time=round(total_time, 2),
                        prompt_eval_duration=round(metric_prefill_duration, 2),
                        generation_duration=round(metric_gen_duration, 2),
                        prompt_tokens_per_second=(
                            round(pt / metric_prefill_duration, 2)
                            if metric_prefill_duration > 0
                            else None
                        ),
                        generation_tokens_per_second=(
                            round(ct / metric_gen_duration, 2)
                            if metric_gen_duration > 0
                            else None
                        ),
                    ).model_dump(exclude_none=True),
                }
                yield f"data: {json.dumps(usage_data)}\n\n"

        yield "data: [DONE]\n\n"
