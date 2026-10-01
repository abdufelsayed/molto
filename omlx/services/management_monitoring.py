"""Read-only runtime observations and explicitly requested cache diagnostics."""

from __future__ import annotations

import copy
import importlib.metadata
import json
import platform
import re
import time
from contextlib import suppress
from pathlib import Path

from fastapi import HTTPException

from ..server_metrics import get_server_metrics


def scheduler_for(engine):
    wrapper = getattr(engine, "_engine", None)
    core = getattr(wrapper, "engine", None)
    return getattr(core, "scheduler", None) or getattr(engine, "scheduler", None)


class MonitoringService:
    def __init__(self, context):
        self.context = context
        self.pool = context.engine_pool

    @property
    def metrics(self):
        return (
            getattr(self.context.runtime_state, "server_metrics", None)
            or get_server_metrics()
        )

    def activity(self):
        from ..prefill_progress import get_prefill_tracker

        now = time.monotonic()
        status = self.pool.get_status()
        models = []
        for info in status.get("models", []):
            if not (info.get("loaded") or info.get("is_loading")):
                continue
            engine = getattr(self.pool.get_entry(info["id"]), "engine", None)
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
                    active_ids |= (
                        set(getattr(core, "_output_collectors", {})) - waiting_ids
                    )
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
                    "loading_elapsed_seconds": max(0, now - started)
                    if started
                    else None,
                }
            )
        enforcer = getattr(self.context.runtime_state, "process_memory_enforcer", None)
        return {
            "models": models,
            "total_active_requests": sum(m["active_requests"] for m in models),
            "total_waiting_requests": sum(m["waiting_requests"] for m in models),
            "model_memory_used": status.get("current_model_memory", 0),
            "model_memory_max": status.get("final_ceiling", 0),
            "memory_pressure": enforcer.get_status() if enforcer else None,
            "uptime_seconds": self.metrics.get_snapshot().get("uptime_seconds"),
        }

    def usage(self, period, model, include_details):
        # Historical canonical IDs can outlive the discovered model registry.
        if model and (
            model.startswith("/") or "\\" in model or ".." in model.split("/")
        ):
            raise HTTPException(400, "Use a canonical model ID")
        history = self.metrics.usage_history
        if history is None:
            return {
                "state": "unavailable",
                "enabled": False,
                "available": False,
                "dropped_requests": 0,
                "range": period,
                "totals": None,
                "models": [],
                "heatmap": [],
            }
        try:
            result = history.query(period, model, include_details=include_details)
        except Exception:
            return {
                "state": "unavailable",
                "enabled": history.enabled,
                "available": False,
                "dropped_requests": history.dropped_requests,
                "range": period,
                "totals": None,
                "models": [],
                "heatmap": [],
            }
        return {
            **result,
            "state": "disabled"
            if not result["enabled"]
            else "available"
            if result["available"]
            else "unavailable",
        }

    def logs(self, lines=100, file=None, level=None):
        settings = self.context.global_settings
        if settings is None:
            raise HTTPException(503, "Server settings unavailable")
        directory = settings.logging.get_log_dir(settings.base_path).resolve()
        files = sorted(
            p.name
            for p in directory.glob("server.log*")
            if p.is_file()
            and not p.is_symlink()
            and re.fullmatch(r"server\.log(?:\.\d+|\.\d{4}-\d{2}-\d{2})?", p.name)
        )
        name = file or "server.log"
        if not re.fullmatch(r"server\.log(?:\.\d+|\.\d{4}-\d{2}-\d{2})?", name):
            raise HTTPException(400, "Invalid log file name")
        path = directory / name
        if path.is_symlink() or path.resolve().parent != directory:
            raise HTTPException(400, "Invalid log file")
        if file and name not in files:
            raise HTTPException(404, "Log file not found")
        max_scan_bytes = 1024 * 1024
        data = b""
        end = 0
        scan_truncated = False
        try:
            if path.exists():
                with path.open("rb") as stream:
                    stream.seek(0, 2)
                    end = stream.tell()
                    position = end
                    chunks = []
                    remaining = max_scan_bytes
                    while position and remaining:
                        count = min(65536, position, remaining)
                        position -= count
                        stream.seek(position)
                        chunks.append(stream.read(count))
                        remaining -= count
                    data = b"".join(reversed(chunks))
                    scan_truncated = position > 0
                    if scan_truncated:
                        # The first fragment may begin inside a physical line.
                        boundary = data.find(b"\n")
                        data = data[boundary + 1 :] if boundary >= 0 else b""
        except OSError as exc:
            raise HTTPException(503, "Log file unavailable") from exc
        physical_lines = data.decode("utf-8", errors="replace").splitlines(
            keepends=True
        )
        severity = {
            "TRACE": 5,
            "DEBUG": 10,
            "INFO": 20,
            "WARNING": 30,
            "ERROR": 40,
            "CRITICAL": 50,
        }
        matches = []
        record_level = None
        # Continuations inherit the most recent parsed record. An initial orphan
        # continuation is excluded from filtered output because its level is unknown.
        for line in physical_lines:
            parsed_level = None
            if line.startswith("{"):
                with suppress(ValueError, TypeError):
                    record = json.loads(line)
                    if isinstance(record, dict):
                        parsed_level = record.get("level")
            else:
                match = re.match(
                    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - [\w.]+ - (TRACE|DEBUG|INFO|WARNING|ERROR|CRITICAL) - ",
                    line,
                )
                if match:
                    parsed_level = match.group(1)
            if parsed_level in severity:
                record_level = parsed_level
            if level is None or (
                record_level is not None and severity[record_level] >= severity[level]
            ):
                matches.append(line)
        return {
            "logs": "".join(matches[-lines:]),
            "total_lines": None if scan_truncated else len(physical_lines),
            "matched_lines": len(matches),
            "scanned_lines": len(physical_lines),
            "scanned_bytes": min(end, max_scan_bytes),
            "scan_truncated": scan_truncated,
            "scan_message": "Only the newest 1 MiB was searched; older log content and an incomplete leading line were excluded."
            if scan_truncated
            else None,
            "log_file": name,
            "available_files": files,
        }

    def versions(self):
        engines = {}
        bundled = Path(__file__).parents[1] / "_engine_commits.json"
        try:
            fallback = json.loads(bundled.read_text())
        except (OSError, ValueError):
            fallback = {}
        declared = {}
        try:
            project = (Path(__file__).parents[2] / "pyproject.toml").read_text()
            for name, url, commit in re.findall(
                r'"([\w-]+)\s*@\s*git\+(https://[^@"]+)@([0-9a-f]{7,40})"', project
            ):
                declared[name] = {"commit": commit, "url": url}
        except OSError:
            pass
        for package in (
            "omlx",
            "mlx",
            "mlx-lm",
            "mlx-vlm",
            "mlx-embeddings",
            "mlx-audio",
            "mflux",
        ):
            info = {
                "name": package,
                "version": None,
                "commit": None,
                "url": None,
                "source": None,
            }
            try:
                dist = importlib.metadata.distribution(package)
                info["version"] = dist.version
                direct = json.loads(dist.read_text("direct_url.json") or "{}")
                vcs = direct.get("vcs_info", {})
                info.update(
                    commit=vcs.get("commit_id"),
                    source="direct_url" if direct else "distribution",
                )
                # Do not expose credentials embedded in source URLs.
                url = direct.get("url", "")
                from urllib.parse import urlsplit, urlunsplit

                parsed = urlsplit(url)
                if parsed.scheme in ("https", "http"):
                    info["url"] = urlunsplit(
                        (parsed.scheme, parsed.hostname or "", parsed.path, "", "")
                    )
                if not info["commit"] and package in fallback:
                    info.update(commit=fallback[package].get("commit"), source="bundle")
            except (importlib.metadata.PackageNotFoundError, ValueError, OSError):
                pass
            info["declared_dependency"] = declared.get(package)
            engines[package] = info
        return {
            "engines": engines,
            "python": platform.python_version(),
            "platform": platform.platform(),
        }

    def reset_stats(self, scope):
        if scope == "alltime":
            path = getattr(self.metrics, "_stats_path", None)
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    raise HTTPException(
                        503, "Persisted stats could not be reset"
                    ) from exc
            self.metrics.clear_alltime_metrics()
        else:
            self.metrics.clear_metrics()
        return {"status": "ok", "scope": scope, "history_preserved": True}

    def probe(self, request):
        entry = self.pool.get_entry(request.model_id)
        if entry is None:
            raise HTTPException(404, "Model not found")
        if getattr(entry, "is_loading", False) or getattr(
            entry, "pending_unload_reason", None
        ):
            raise HTTPException(409, "Model is changing state")
        engine = entry.engine
        if engine is None:
            return {
                "model_id": request.model_id,
                "model_loaded": False,
                "reason": "Load the model to probe its cache",
            }
        scheduler = scheduler_for(engine)
        if scheduler is None:
            raise HTTPException(400, "This engine does not expose a prefix cache")
        tokenizer = getattr(engine, "_tokenizer", None)
        if not hasattr(tokenizer, "apply_chat_template"):
            raise HTTPException(400, "Chat templating unavailable")
        from ..api.tool_calling import convert_tools_for_template
        from ..model_settings import merge_chat_template_kwargs

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
        settings = self.context.settings_manager.get_settings(request.model_id)
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
            raise HTTPException(400, "Failed to tokenize probe messages") from exc
        if len(tokens) > 131072:
            raise HTTPException(400, "Probe exceeds 131072 tokens")
        prefix = getattr(scheduler, "block_aware_cache", None)
        size = getattr(
            getattr(scheduler, "config", None), "paged_cache_block_size", 0
        ) or getattr(prefix, "block_size", 0)
        if not size:
            raise HTTPException(400, "Paged cache is disabled")
        from ..cache.paged_cache import compute_block_hash

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
