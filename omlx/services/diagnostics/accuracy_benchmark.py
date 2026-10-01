# SPDX-License-Identifier: Apache-2.0
"""Accuracy benchmark execution logic for oMLX diagnostics.

Orchestrates MMLU, HellaSwag, TruthfulQA, GSM8K, and LiveCodeBench
evaluations with real-time progress reporting via SSE events.

Supports server-side queue and persistent result accumulation.
Results survive browser close and persist until explicitly reset.
"""

import asyncio
import contextlib
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .external_api import (
    ExternalAPIClient,
    ExternalChatAdapter,
    ExternalEndpointConfig,
)

logger = logging.getLogger(__name__)

VALID_BENCHMARKS = [
    "mmlu",
    "mmlu_pro",
    "kmmlu",
    "cmmlu",
    "jmmlu",
    "hellaswag",
    "truthfulqa",
    "arc_challenge",
    "winogrande",
    "gsm8k",
    "mathqa",
    "humaneval",
    "mbpp",
    "livecodebench",
    "bbq",
    "safetybench",
]

# Sampling profile for an accuracy run. "deterministic" (default) runs greedy
# (temperature 0) so saved scores stay reproducible; "model_settings" opts in to
# the model's configured sampling (temperature, top_p, …) for a real-world score.
SamplingProfile = Literal["deterministic", "model_settings"]


class AccuracyBenchmarkRequest(BaseModel):
    """Request model for starting an accuracy benchmark."""

    model_config = ConfigDict(extra="forbid", strict=True)

    model_id: str
    benchmarks: dict[str, int]  # name -> sample_size (0 = full dataset)
    batch_size: int = 1
    enable_thinking: bool = False
    sampling_profile: SamplingProfile = "deterministic"
    # When set, the benchmark runs against a remote OpenAI-compatible
    # endpoint instead of a local engine and model_id is the remote
    # model name (not validated against the local catalog).
    external: ExternalEndpointConfig | None = None

    @model_validator(mode="after")
    def _force_thinking_off_for_external(self) -> "AccuracyBenchmarkRequest":
        # enable_thinking is a local chat-template kwarg; external requests
        # never send it, so keep the stored flag honest.
        if self.external is not None:
            self.enable_thinking = False
        return self

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v not in (1, 2, 4, 8, 16, 32):
            raise ValueError("batch_size must be 1, 2, 4, 8, 16, or 32")
        return v

    @field_validator("benchmarks")
    @classmethod
    def validate_benchmarks(cls, v: dict[str, int]) -> dict[str, int]:
        if not v:
            raise ValueError("At least one benchmark is required")
        for name, size in v.items():
            if name not in VALID_BENCHMARKS:
                raise ValueError(
                    f"Invalid benchmark '{name}'. Must be one of {VALID_BENCHMARKS}"
                )
            if size < 0:
                raise ValueError(f"Sample size for '{name}' must be >= 0")
        return v


@dataclass
class AccuracyBenchmarkRun:
    """Tracks the state of a running accuracy benchmark.

    SSE delivery model mirrors `BenchmarkRun`: append-only `events`
    log + `cond` for live notification + `terminal` flag set on the
    final event. See benchmark.py for the rationale.
    """

    bench_id: str
    request: AccuracyBenchmarkRequest
    status: str = "running"  # running, completed, cancelled, error
    events: list[dict] = field(default_factory=list)
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)
    terminal: bool = False
    task: asyncio.Task | None = None
    results: list[dict] = field(default_factory=list)
    error_message: str = ""
    last_progress: dict | None = None  # last progress event for reconnect
    # Finer-grained lifecycle than `status` — surfaces the difference between
    # "still scoring questions" and "cleaning up after the last result was
    # emitted". The serialization gate (_queue_running) stays True across
    # both, but a UI rendering the running row wants to hide it once
    # phase=="unloading" so the user isn't told "still running" when the
    # result card has already appeared on screen. Transitions:
    #   pending → loading → evaluating → unloading → completed
    # (cancelled / error replace the terminal phase on those branches.)
    phase: str = "pending"


# Accuracy stream closes on `done` (run finished) or `error`. Unlike the
# throughput bench there's no post-done upload phase to ride out: each
# suite's `upload` event is emitted right after its `result`, before `done`.
_ACCURACY_TERMINAL_TYPES = frozenset({"done", "error"})


# --- Run management ---


# --- Accumulated results ---


# --- Queue management ---


# --- SSE ---


async def _send_event(run: AccuracyBenchmarkRun, event: dict) -> None:
    """Append an event to the run's log and wake subscribers.

    Updates `last_progress` (used by the REST `queue/status` endpoint
    for reconnect hints) and sets `run.terminal` on the final event.
    """
    if getattr(run, "_cancel_requested", False):
        run._cancel_requested = False
        raise asyncio.CancelledError()
    if event.get("type") == "progress":
        run.last_progress = event
    async with run.cond:
        run.events.append(event)
        if event.get("type") in _ACCURACY_TERMINAL_TYPES:
            run.terminal = True
        run.cond.notify_all()


# --- Benchmark execution ---


async def run_accuracy_benchmark(run: AccuracyBenchmarkRun, engine_pool: Any) -> None:
    """Execute accuracy benchmark run.

    Phases:
    1. Unload all models (a reusable resident target stays loaded)
    2. Load target model
    3. For each selected benchmark: load data, evaluate, report
    4. Unload model
    5. Send done event
    """
    from ...eval import BENCHMARKS

    request = run.request

    # Suppress TTL auto-unload during benchmark (local engines only)
    if request.external is None:
        engine_pool._suppress_ttl = True
    start_time = time.time()
    client: ExternalAPIClient | None = None
    leased_model_id: str | None = None

    try:
        run.phase = "loading"
        if request.external is not None:
            # External endpoint: no local model lifecycle. The adapter owns
            # the sampling-profile mapping, so sampling_kwargs stays empty
            # (enable_thinking is already forced off by request validation).
            await _send_event(
                run,
                {
                    "type": "progress",
                    "phase": "connect",
                    "model_id": request.model_id,
                    "benchmark": "",
                    "message": f"Connecting to {request.external.base_url}...",
                    "current": 0,
                    "total": len(request.benchmarks),
                },
            )
            client = ExternalAPIClient(request.external)
            engine = ExternalChatAdapter(client, request.sampling_profile)
            # Fail fast on auth/URL/model errors so a wrong API key cannot
            # silently produce a 0% score.
            await engine.preflight()
            sampling_kwargs = {}
        else:
            # Phase 1: Unload all models except a resident target that the
            # LM load below would reuse as-is.
            loaded_ids = [
                model_id
                for model_id in engine_pool.get_loaded_model_ids()
                if model_id != request.model_id
                or engine_pool._force_lm_replaces_engine(model_id)
            ]
            if loaded_ids:
                await _send_event(
                    run,
                    {
                        "type": "progress",
                        "phase": "unload",
                        "model_id": request.model_id,
                        "benchmark": "",
                        "message": f"Unloading {len(loaded_ids)} model(s)...",
                        "current": 0,
                        "total": len(request.benchmarks),
                    },
                )
                for model_id in loaded_ids:
                    try:
                        await engine_pool._unload_engine(model_id)
                    except Exception as e:
                        logger.warning(f"Failed to unload {model_id}: {e}")

            # Phase 2: Load target model
            await _send_event(
                run,
                {
                    "type": "progress",
                    "phase": "load",
                    "model_id": request.model_id,
                    "benchmark": "",
                    "message": f"Loading {request.model_id}...",
                    "current": 0,
                    "total": len(request.benchmarks),
                },
            )

            # Force LM engine for accuracy benchmarks — text-only tasks
            # don't need VLM and the VLM adapter can produce empty responses.
            # The lease keeps settings saves and eviction from unloading the
            # engine between eval batches.
            engine = await engine_pool.get_engine(
                request.model_id, force_lm=True, _lease=True
            )
            leased_model_id = request.model_id

            # Load model sampling settings. Under the default "deterministic"
            # profile sampling params are not read — the benchmark runs greedy
            # (temperature 0) so saved scores stay reproducible. Only the
            # explicit "model_settings" opt-in honors the model's configured
            # sampling. chat_template_kwargs is prompt construction, not
            # sampling, so it is forwarded in both profiles.
            sampling_kwargs = {}
            if engine_pool._settings_manager is not None:
                ms = engine_pool._settings_manager.get_settings(request.model_id)
                if ms.chat_template_kwargs:
                    sampling_kwargs["chat_template_kwargs"] = ms.chat_template_kwargs
                if request.sampling_profile == "model_settings":
                    if ms.temperature is not None:
                        sampling_kwargs["temperature"] = ms.temperature
                    if ms.top_p is not None:
                        sampling_kwargs["top_p"] = ms.top_p
                    if ms.top_k is not None:
                        sampling_kwargs["top_k"] = ms.top_k
                    if ms.min_p is not None:
                        sampling_kwargs["min_p"] = ms.min_p
                    if ms.repetition_penalty is not None:
                        sampling_kwargs["repetition_penalty"] = ms.repetition_penalty
                    if ms.presence_penalty is not None:
                        sampling_kwargs["presence_penalty"] = ms.presence_penalty

        # Phase 3: Run each benchmark
        run.phase = "evaluating"
        completed = 0
        for bench_name, sample_size in request.benchmarks.items():
            if run.status == "cancelled":
                break

            bench_cls = BENCHMARKS.get(bench_name)
            if bench_cls is None:
                logger.warning(f"Unknown benchmark: {bench_name}")
                continue

            evaluator = bench_cls()

            # Load dataset
            await _send_event(
                run,
                {
                    "type": "progress",
                    "phase": "download",
                    "model_id": request.model_id,
                    "benchmark": bench_name,
                    "message": f"Loading {bench_name} dataset...",
                    "current": completed,
                    "total": len(request.benchmarks),
                },
            )

            try:
                items = await evaluator.load_dataset(sample_size=sample_size)
            except Exception as e:
                logger.error(f"Failed to load {bench_name} dataset: {e}")
                await _send_event(
                    run,
                    {
                        "type": "error",
                        "message": f"Failed to load {bench_name} dataset: {e}",
                    },
                )
                run.status = "error"
                run.error_message = str(e)
                return

            # Run evaluation with progress
            total_items = len(items)

            async def on_progress(
                current: int, total: int, bench_name=bench_name, completed=completed
            ) -> None:
                if run.status == "cancelled":
                    raise asyncio.CancelledError()
                await _send_event(
                    run,
                    {
                        "type": "progress",
                        "phase": "eval",
                        "model_id": request.model_id,
                        "benchmark": bench_name,
                        "message": f"Evaluating {bench_name} ({current}/{total})...",
                        "current": completed,
                        "total": len(request.benchmarks),
                        "bench_current": current,
                        "bench_total": total,
                    },
                )

            await _send_event(
                run,
                {
                    "type": "progress",
                    "phase": "eval",
                    "model_id": request.model_id,
                    "benchmark": bench_name,
                    "message": f"Evaluating {bench_name} (0/{total_items})...",
                    "current": completed,
                    "total": len(request.benchmarks),
                    "bench_current": 0,
                    "bench_total": total_items,
                },
            )

            try:
                result = await evaluator.run(
                    engine,
                    items,
                    on_progress,
                    batch_size=request.batch_size,
                    sampling_kwargs=sampling_kwargs,
                    enable_thinking=request.enable_thinking,
                )
            except asyncio.CancelledError:
                run.status = "cancelled"
                await _send_event(
                    run,
                    {
                        "type": "error",
                        "message": "Benchmark cancelled",
                    },
                )
                return
            except Exception as e:
                logger.error(f"Error running {bench_name}: {e}")
                await _send_event(
                    run,
                    {
                        "type": "error",
                        "message": f"Error running {bench_name}: {e}",
                    },
                )
                run.status = "error"
                run.error_message = str(e)
                return

            question_results = []
            for qr in result.question_results:
                question_data = {
                    "id": qr.question_id,
                    "correct": qr.correct,
                    "expected": qr.expected,
                    "predicted": qr.predicted,
                    "question": qr.question_text,
                    "raw_response": qr.raw_response,
                    "category": qr.category,
                    "time_s": round(qr.time_seconds, 3),
                }
                if request.external is not None:
                    question_data.update(
                        {
                            "status": qr.status,
                            "finish_reason": qr.finish_reason,
                            "reasoning_fields_present": qr.reasoning_fields_present,
                            "reasoning_fields_nonempty": qr.reasoning_fields_nonempty,
                            "prompt_tokens": qr.prompt_tokens,
                            "completion_tokens": qr.completion_tokens,
                            "error_message": qr.error_message,
                        }
                    )
                else:
                    question_data.update(
                        {
                            "finish_reason": qr.finish_reason,
                            "prompt_tokens": qr.prompt_tokens,
                            "completion_tokens": qr.completion_tokens,
                        }
                    )
                question_results.append(question_data)

            result_data = {
                "model_id": request.model_id,
                "external": request.external is not None,
                "benchmark": result.benchmark_name,
                "accuracy": round(result.accuracy, 4),
                "thinking_used": result.thinking_used,
                "total": result.total_questions,
                "correct": result.correct_count,
                "time_s": round(result.time_seconds, 1),
                "dataset_total": evaluator.dataset_total,
                "sampling_profile": request.sampling_profile,
                "question_results": question_results,
            }
            if request.external is not None:
                status_counts = Counter(
                    qr.status or "invalid_response" for qr in result.question_results
                )
                valid_responses = status_counts["correct"] + status_counts["wrong"]
                total_questions = result.total_questions
                result_data.update(
                    {
                        "valid_response_count": valid_responses,
                        "empty_content_count": status_counts["empty_content"],
                        "truncated_count": status_counts["truncated"],
                        "timeout_count": status_counts["timeout"],
                        "http_error_count": status_counts["http_error"],
                        "connection_error_count": status_counts["connection_error"],
                        "invalid_response_count": status_counts["invalid_response"],
                        "parse_error_count": status_counts["parse_error"],
                        "wrong_count": status_counts["wrong"],
                        "valid_response_rate": round(
                            valid_responses / total_questions
                            if total_questions > 0
                            else 0.0,
                            4,
                        ),
                        "valid_answer_accuracy": round(
                            result.correct_count / valid_responses
                            if valid_responses > 0
                            else 0.0,
                            4,
                        ),
                        "reliability_warning": any(
                            status_counts[name] > 0
                            for name in (
                                "empty_content",
                                "truncated",
                                "timeout",
                                "http_error",
                                "connection_error",
                                "invalid_response",
                                "parse_error",
                            )
                        ),
                    }
                )
            else:
                # Answers that hit the token limit (max_tokens) are scored on
                # incomplete output; count them so the headline is not read as
                # clean (#3772). "Finished" means a normal stop, so errored or
                # aborted answers count as neither. Accuracy on finished answers
                # describes only those (None when there are none); it is not a
                # corrected score.
                questions = result.question_results
                truncated = [qr for qr in questions if qr.finish_reason == "length"]
                finished = [
                    qr for qr in questions if qr.finish_reason in ("stop", "tool_calls")
                ]
                result_data.update(
                    {
                        "truncated_count": len(truncated),
                        "truncated_correct_count": sum(qr.correct for qr in truncated),
                        "finished_count": len(finished),
                        "finished_accuracy": (
                            round(sum(qr.correct for qr in finished) / len(finished), 4)
                            if finished
                            else None
                        ),
                    }
                )
            if result.category_scores:
                result_data["category_scores"] = {
                    k: round(v, 4) for k, v in result.category_scores.items()
                }

            # Accumulate persistently

            run.results.append(result_data)
            completed += 1

            await _send_event(
                run,
                {
                    "type": "result",
                    "data": result_data,
                },
            )

            # done event so the SSE terminal contract stays unchanged.
            # result_data is the same object stored in _accumulated_results,
            # so the outcome is visible to polling clients and SSE replay
            # without any extra state. Never fails the benchmark.

        # Phase 4: Unload model. The result(s) are already emitted by now,
        # so flip phase so polling clients hide the running indicator
        # (the result card has already appeared on screen — telling the
        # user "still running" while we clean up reads as a bug).
        run.phase = "unloading"
        if request.external is None:
            if leased_model_id is not None:
                await engine_pool.release_engine(leased_model_id)
                leased_model_id = None
            with contextlib.suppress(Exception):
                await engine_pool._unload_engine(request.model_id)

        # Phase 5: Done
        total_time = time.time() - start_time
        run.status = "completed"
        run.phase = "completed"

        await _send_event(
            run,
            {
                "type": "done",
                "summary": {
                    "model_id": request.model_id,
                    "total_time": round(total_time, 1),
                    "benchmarks_completed": completed,
                },
            },
        )

    except asyncio.CancelledError:
        run.status = "cancelled"
        run.phase = "cancelled"
        await _send_event(
            run,
            {
                "type": "error",
                "message": "Benchmark cancelled",
            },
        )
    except Exception as e:
        logger.exception(f"Accuracy benchmark error: {e}")
        run.status = "error"
        run.phase = "error"
        run.error_message = str(e)
        await _send_event(
            run,
            {
                "type": "error",
                "message": str(e),
            },
        )
    finally:
        if leased_model_id is not None:
            await engine_pool.release_engine(leased_model_id)
        # Re-enable TTL auto-unload
        engine_pool._suppress_ttl = False
        if client is not None:
            await client.aclose()
