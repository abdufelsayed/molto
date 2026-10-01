"""Inference-owned cache and activity observations, independent of HTTP."""

import asyncio
import copy
import gc
import json
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from omlx_contracts.runtime import RuntimeOperationError


def scheduler_for(engine):
    wrapper = getattr(engine, "_engine", None)
    core = getattr(wrapper, "engine", None)
    return getattr(core, "scheduler", None) or getattr(engine, "scheduler", None)


def loaded_cores(pool):
    for model_id in pool.get_loaded_model_ids():
        entry = pool.get_entry(model_id)
        if entry is None or entry.engine is None:
            continue
        engine = entry.engine
        core = getattr(engine, "engine", engine)
        scheduler = getattr(core, "scheduler", None)
        yield model_id, engine, core, scheduler


def cache_status(pool, cache_dir) -> dict[str, Any]:
    models = []
    for model_id, engine, _, scheduler in loaded_cores(pool):
        get_stats = getattr(engine, "get_runtime_cache_stats", None)
        if not callable(get_stats):
            get_stats = getattr(engine, "get_cache_stats", None)
        if not callable(get_stats) and scheduler is not None:
            get_stats = getattr(scheduler, "get_ssd_cache_stats", None)
        stats = get_stats() if callable(get_stats) else None
        if stats is not None:
            models.append({"model_id": model_id, "stats": stats})
    return {
        "models": models,
        "ssd_cache_dir": str(cache_dir) if cache_dir else None,
    }


async def clear_cache(pool, kind: str, cache_dir=None) -> dict[str, Any]:
    if kind not in {"hot", "ssd"}:
        raise RuntimeOperationError("not_found", "Unknown cache kind")
    busy = [
        model_id
        for model_id in pool.get_loaded_model_ids()
        if pool.is_model_busy(model_id)
    ]
    if busy:
        raise RuntimeOperationError(
            "busy",
            f"Cannot clear cache while requests are active: {', '.join(busy)}",
        )
    total = 0
    failures = []
    reclaim = []
    for model_id, _, core, scheduler in loaded_cores(pool):
        distributed = getattr(core, "clear_prompt_caches", None)
        if callable(distributed):
            try:
                report = await distributed(**{kind: True})
                total += int(
                    report.get(f"{kind}_cleared", report.get("ssd_deleted", 0))
                )
            except Exception as exc:
                failures.append(f"{model_id}: {exc}")
            continue
        if scheduler is None:
            continue
        manager = getattr(scheduler, "paged_ssd_cache_manager", None)
        if manager is not None:
            clear = getattr(
                manager, "clear_hot_cache" if kind == "hot" else "clear", None
            )
            if callable(clear):
                total += int(clear() or 0)
        if kind == "hot":
            tracker = getattr(scheduler, "_cache_rate_tracker", None)
            if tracker is not None:
                tracker.clear()
            executor = getattr(core, "_mlx_executor", None)
            if executor is not None:
                reclaim.append((executor, getattr(scheduler, "_stream", None)))
    if kind == "hot":
        from omlx_runtime.engine_core import get_mlx_executor
        from omlx_runtime.scheduler import _sync_and_clear_cache

        budget = getattr(
            pool._scheduler_config,
            "hot_cache_budget",
            None,
        )
        if budget is not None and hasattr(budget, "clear_all_owners"):
            total += int(budget.clear_all_owners() or 0)
        gc.collect()
        loop = asyncio.get_running_loop()
        for executor, stream in reclaim or [(get_mlx_executor(), None)]:
            await loop.run_in_executor(executor, _sync_and_clear_cache, stream)
    else:
        # The manager handles loaded models. Also clear known persisted
        # cache files for models that are currently unloaded.
        if cache_dir is not None:
            root = Path(cache_dir)
            for base in (root, root / "deepseek_v41_ced_v1"):
                for bucket in "0123456789abcdef":
                    for file in (base / bucket).glob("*.safetensors"):
                        try:
                            file.unlink()
                            total += 1
                        except OSError as exc:
                            failures.append(f"{file}: {exc}")
    if failures:
        raise RuntimeOperationError("unavailable", "; ".join(failures)[:1000])
    return {"status": "ok", "kind": kind, "total_cleared": total}


def activity(pool, metrics, enforcer=None):
    from omlx_runtime.prefill_progress import get_prefill_tracker

    now = time.monotonic()
    status = pool.get_status()
    models = []
    for info in status.get("models", []):
        if not (info.get("loaded") or info.get("is_loading")):
            continue
        engine = getattr(pool.get_entry(info["id"]), "engine", None)
        scheduler = scheduler_for(engine)
        snap = (
            scheduler.snapshot_for_admin()
            if scheduler and hasattr(scheduler, "snapshot_for_admin")
            else None
        )
        if snap is None and scheduler is not None:
            # Current schedulers expose containers; copy them before formatting.
            try:
                snap = {
                    "running_by_id": getattr(scheduler, "running", {}).copy(),
                    "waiting": list(getattr(scheduler, "waiting", ()).copy()),
                }
            except (RuntimeError, AttributeError):
                snap = None
        running = snap.get("running_by_id", {}) if snap else {}
        queue = snap.get("waiting", []) if snap else []
        waiting_ids = {r.request_id for r in queue}
        prefill = get_prefill_tracker().get_model_progress(info["id"])
        prefill_ids = {r["request_id"] for r in prefill}
        active_ids = set(running) | prefill_ids
        if snap is None:
            core = getattr(getattr(engine, "_engine", None), "engine", None)
            with suppress(RuntimeError):
                active_ids |= set(getattr(core, "_output_collectors", {})) - waiting_ids
        own = (
            engine.get_activity_snapshot()
            if hasattr(engine, "get_activity_snapshot")
            else {}
        )
        active = len(active_ids) + own.get("active_requests", 0)
        generating = []
        for rid in sorted(active_ids - prefill_ids - waiting_ids):
            req = running.get(rid)
            started = getattr(req, "generation_started_at", None)
            elapsed = max(0, now - started) if started else None
            tokens = getattr(req, "num_output_tokens", 0)
            generating.append(
                {
                    "request_id": rid,
                    "generated_tokens": tokens,
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": tokens / elapsed if elapsed else None,
                }
            )
        live = None
        if hasattr(engine, "get_live_metrics"):
            try:
                live = engine.get_live_metrics()
            except Exception:
                live = {"stale": True, "unavailable": True}
        if live and not live.get("stale"):
            active = live.get("metrics", {}).get("active_requests", 0)
            last = live.get("metrics", {}).get("last_request")
            if isinstance(last, dict) and last.get("status") == "running":
                progress = last.get("prefill_progress")
                if isinstance(progress, dict) and progress.get("active"):
                    prefill.append({"request_id": "rank0", **progress})
                elif last.get("decode_tps"):
                    generating.append(
                        {
                            "request_id": "rank0",
                            "generated_tokens": last.get("completion_tokens", 0),
                            "elapsed_seconds": last.get("elapsed_seconds"),
                            "tokens_per_second": last["decode_tps"],
                        }
                    )
        started = info.get("loading_started_at")
        models.append(
            {
                **info,
                "active_requests": active,
                "waiting_requests": len(queue),
                "prefilling": prefill,
                "generating": generating,
                "waiting": [
                    {
                        "request_id": r.request_id,
                        "queue_position": i,
                        "elapsed_seconds": max(0, now - r.arrival_time),
                        "prompt_tokens": getattr(r, "num_prompt_tokens", 0),
                    }
                    for i, r in enumerate(queue, 1)
                ],
                "activities": own.get("activities", []),
                "cluster_live": live,
                "stage": "loading"
                if info.get("is_loading")
                else "prefilling"
                if prefill
                else "generating"
                if generating
                else "active"
                if active
                else "idle",
                "loading_elapsed_seconds": max(0, now - started) if started else None,
            }
        )
    return {
        "models": models,
        "total_active_requests": sum(m["active_requests"] for m in models),
        "total_waiting_requests": sum(m["waiting_requests"] for m in models),
        "model_memory_used": status.get("current_model_memory", 0),
        "model_memory_max": status.get("final_ceiling", 0),
        "memory_pressure": enforcer.get_status() if enforcer else None,
        "uptime_seconds": metrics.get_snapshot().get("uptime_seconds"),
    }


def probe(pool, request, settings):
    entry = pool.get_entry(request.model_id)
    if entry is None:
        raise RuntimeOperationError("not_found", "Model not found")
    if getattr(entry, "is_loading", False) or getattr(
        entry, "pending_unload_reason", None
    ):
        raise RuntimeOperationError("busy", "Model is changing state")
    engine = entry.engine
    if engine is None:
        return {
            "model_id": request.model_id,
            "model_loaded": False,
            "reason": "Load the model to probe its cache",
        }
    scheduler = scheduler_for(engine)
    if scheduler is None:
        raise RuntimeOperationError(
            "invalid_configuration", "This engine does not expose a prefix cache"
        )
    tokenizer = getattr(engine, "_tokenizer", None)
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeOperationError(
            "invalid_configuration", "Chat templating unavailable"
        )
    from omlx_config.model_settings import merge_chat_template_kwargs

    from omlx_runtime.generation.tool_calling import convert_tools_for_template

    messages = copy.deepcopy(request.messages)
    for message in messages:
        for call in message.get("tool_calls", []):
            function = call.get("function", {})
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                with suppress(ValueError):
                    function["arguments"] = json.loads(arguments)
    if hasattr(engine, "_preprocess_messages"):
        messages = engine._preprocess_messages(messages)
    kwargs = merge_chat_template_kwargs(
        settings,
        request.chat_template_kwargs,
        thinking_budget=request.thinking_budget,
        preserve_thinking_default=getattr(entry, "preserve_thinking_default", None),
    )
    try:
        tools = convert_tools_for_template(request.tools) if request.tools else None
        prompt = (
            engine._apply_chat_template(
                messages, tools, chat_template_kwargs=kwargs or None
            )
            if hasattr(engine, "_apply_chat_template")
            else tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kwargs
            )
        )
        tokens = list(tokenizer.encode(prompt))
    except Exception as exc:
        raise RuntimeOperationError(
            "invalid_configuration", "Failed to tokenize probe messages"
        ) from exc
    if len(tokens) > 131072:
        raise RuntimeOperationError(
            "invalid_configuration", "Probe exceeds 131072 tokens"
        )
    prefix = getattr(scheduler, "block_aware_cache", None)
    size = getattr(
        getattr(scheduler, "config", None), "paged_cache_block_size", 0
    ) or getattr(prefix, "block_size", 0)
    if not size:
        raise RuntimeOperationError("invalid_configuration", "Paged cache is disabled")
    from omlx_runtime.cache.paged_cache import compute_block_hash

    ssd = getattr(scheduler, "paged_ssd_cache_manager", None)
    index = getattr(ssd, "_index", None)
    hot = getattr(ssd, "_hot_cache", None)
    model_name = getattr(
        getattr(scheduler, "paged_cache_manager", None), "model_name", None
    )
    parent = b""
    hot_count = disk_count = hit_tokens = 0
    for start in range(0, len(tokens), size):
        block = tokens[start : start + size]
        parent = compute_block_hash(
            parent, block, extra_keys=None, model_name=model_name
        )
        in_hot = hot is not None and parent in hot
        in_disk = index is not None and index.contains(parent)
        if not (in_hot or in_disk):
            break
        hot_count += int(in_hot)
        disk_count += int(not in_hot)
        hit_tokens += len(block)
    blocks = (len(tokens) + size - 1) // size
    return {
        "model_id": request.model_id,
        "model_loaded": True,
        "total_tokens": len(tokens),
        "block_size": size,
        "total_blocks": blocks,
        "blocks_ssd_hot": hot_count,
        "blocks_ssd_disk": disk_count,
        "blocks_cold": blocks - hot_count - disk_count,
        "ssd_hit_tokens": hit_tokens,
        "cold_tokens": len(tokens) - hit_tokens,
    }
