# SPDX-License-Identifier: Apache-2.0
"""Engine management operations, independent of HTTP and server globals."""

from __future__ import annotations

import asyncio
import copy
import gc
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from ..engine_pool import EnginePool
from ..model_profiles import filter_profile_fields
from ..model_settings import (
    ModelSettings,
    ModelSettingsManager,
    validate_ane_prefill,
    validate_moe_expert_offload,
)
from ..server_metrics import get_server_metrics
from ..settings import GlobalSettings, SamplingSettings, SchedulerSettings
from .management_models import (
    GlobalSettingsPatch,
    ModelSettingsPatch,
    ProfileUpdate,
    ProfileWrite,
)


class ManagementError(Exception):
    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(detail)


@dataclass(frozen=True)
class ManagementContext:
    engine_pool: EnginePool
    settings_manager: ModelSettingsManager
    global_settings: GlobalSettings | None
    get_default_model: Callable[[], str | None]
    set_default_model: Callable[[str | None], None]
    apply_sampling: Callable[[], None]


class ManagementService:
    def __init__(self, context: ManagementContext):
        self.context = context

    @property
    def pool(self) -> EnginePool:
        return self.context.engine_pool

    @property
    def manager(self) -> ModelSettingsManager:
        return self.context.settings_manager

    def _model(self, model_id: str):
        entry = self.pool.get_entry(model_id)
        if entry is None:
            raise ManagementError("not_found", f"Model not found: {model_id}")
        return entry

    def inventory(self) -> dict[str, Any]:
        """Report every discovered model and its persisted engine settings."""
        status = self.pool.get_status()
        models = []
        for model in status["models"]:
            settings = asdict(self.manager.get_settings(model["id"]))
            # Do not expose the local trust decision as a general API toggle.
            settings.pop("trust_remote_code", None)
            entry = self.pool.get_entry(model["id"])
            is_unloading = (
                bool(entry is not None and entry.pending_unload_reason)
                or model["id"] in self.pool._unloading_models
            )
            image_details = {}
            if model.get("engine_type") == "image_generation" and entry is not None:
                from .model_control import diffusion_metadata

                image_details["diffusion"] = diffusion_metadata(Path(entry.model_path))
            models.append(
                {
                    **model,
                    **image_details,
                    "is_unloading": is_unloading,
                    "settings": settings,
                }
            )
        return {"models": models, "model_count": status["model_count"]}

    def state(self) -> dict[str, Any]:
        status = self.pool.get_status()
        models = []
        for model in status["models"]:
            entry = self.pool.get_entry(model["id"])
            models.append(
                {
                    "id": model["id"],
                    "loaded": model["loaded"],
                    "is_loading": model["is_loading"],
                    "is_unloading": bool(
                        entry is not None and entry.pending_unload_reason
                    )
                    or model["id"] in self.pool._unloading_models,
                    "load_failed": model["load_failed"],
                }
            )
        return {
            "default_model": self.context.get_default_model(),
            "preparation_active": status.get("preparation_active", False),
            "model_count": status["model_count"],
            "loaded_count": sum(1 for model in models if model["loaded"]),
            "current_model_memory": status["current_model_memory"],
            "final_ceiling": status["final_ceiling"],
            "models": models,
        }

    async def load(self, model_id: str) -> dict[str, Any]:
        entry = self._model(model_id)
        if entry.engine is not None:
            return {"status": "ok", "model_id": model_id, "message": "Already loaded"}
        if entry.is_loading:
            raise ManagementError("busy", f"Model is already loading: {model_id}")
        await self.pool.get_engine(model_id)
        return {"status": "ok", "model_id": model_id, "message": "Loaded"}

    async def unload(self, model_id: str) -> dict[str, Any]:
        entry = self._model(model_id)
        if entry.engine is None:
            raise ManagementError(
                "invalid_configuration", f"Model not loaded: {model_id}"
            )
        if entry.is_loading:
            raise ManagementError("busy", f"Model still loading: {model_id}")
        unloaded = await self.pool.request_unload(
            model_id, reason="manual management unload"
        )
        if not unloaded:
            return {"status": "unloading", "model_id": model_id}
        return {"status": "ok", "model_id": model_id}

    async def refresh(self) -> dict[str, Any]:
        """Re-read model settings and discover current model directories."""
        if getattr(self.pool, "_preparation_active", False) is True:
            raise ManagementError("busy", "Diffusion preparation is active")
        self.manager._load()
        global_settings = self.context.global_settings
        model_dirs = (
            [str(path) for path in global_settings.get_effective_model_dirs()]
            if global_settings is not None
            else [str(path) for path in getattr(self.pool, "_model_dirs", [])]
        )
        if not model_dirs:
            raise ManagementError("unavailable", "No model directories configured")
        self.pool.discover_models(model_dirs, self.manager.get_pinned_model_ids())
        self.pool.apply_settings_overrides(self.manager)
        available = self.pool.get_model_ids()
        preferred = self.manager.get_default_model_id()
        current = self.context.get_default_model()
        self.context.set_default_model(
            preferred
            if preferred in available
            else current
            if current in available
            else available[0]
            if available
            else None
        )
        return {"status": "ok", "model_count": self.pool.model_count}

    def get_model_settings(self, model_id: str) -> dict[str, Any]:
        self._model(model_id)
        settings = asdict(self.manager.get_settings(model_id))
        settings.pop("trust_remote_code", None)
        return {"model_id": model_id, "settings": settings}

    async def update_model_settings(
        self, model_id: str, patch: ModelSettingsPatch
    ) -> dict[str, Any]:
        entry = self._model(model_id)
        current = self.manager.get_settings(model_id)
        old_type = (entry.model_type, entry.engine_type)
        old_signature = self.pool._engine_runtime_signature(model_id, current)
        values = current.to_dict()
        defaults = ModelSettings()
        for name in patch.model_fields_set:
            value = getattr(patch, name)
            values[name] = getattr(defaults, name) if value is None else value
        alias = values.get("model_alias")
        if alias:
            other_ids = set(self.pool.get_model_ids()) - {model_id}
            other_aliases = {
                settings.model_alias
                for mid, settings in self.manager.get_all_settings().items()
                if mid != model_id and settings.model_alias
            }
            exposed = self.manager.get_exposed_profile_model_ids()
            if alias in other_ids | other_aliases | exposed:
                raise ManagementError(
                    "conflict", f"Model alias already in use: {alias}"
                )
        try:
            updated = ModelSettings.from_dict(values)
            self._validate_model_settings(entry, updated)
        except (TypeError, ValueError) as exc:
            raise ManagementError("invalid_configuration", str(exc)) from exc

        detected_type = None
        if (
            "model_type_override" in patch.model_fields_set
            and not updated.model_type_override
        ):
            from ..model_discovery import detect_model_type

            detected_type = detect_model_type(Path(entry.model_path))

        try:
            self.manager.set_settings(model_id, updated)
        except OSError as exc:
            raise ManagementError("unavailable", str(exc)) from exc

        entry.is_pinned = updated.is_pinned
        if "model_type_override" in patch.model_fields_set:
            if updated.model_type_override:
                entry.model_type = updated.model_type_override
                entry.engine_type = self.pool._MODEL_TYPE_TO_ENGINE[
                    updated.model_type_override
                ]
            else:
                entry.model_type = detected_type
                entry.engine_type = self.pool._MODEL_TYPE_TO_ENGINE.get(
                    detected_type, "batched"
                )
        else:
            self.pool.apply_settings_overrides(self.manager)
        if updated.is_default:
            self.context.set_default_model(model_id)
        elif current.is_default and not updated.is_default:
            available = [mid for mid in self.pool.get_model_ids() if mid != model_id]
            self.context.set_default_model(available[0] if available else None)
        requires_reload = entry.engine is not None and (
            old_type != (entry.model_type, entry.engine_type)
            or old_signature != self.pool._engine_runtime_signature(model_id, updated)
            or bool(
                patch.model_fields_set
                & {"index_cache_freq", "dflash_enabled", "dflash_draft_model"}
            )
        )
        auto_unloaded = auto_reloaded = reload_deferred = False
        reload_error = None
        if requires_reload:
            try:
                auto_unloaded = await self.pool.request_unload(
                    model_id, reason="settings changed", abort_active=False
                )
                reload_deferred = not auto_unloaded
                if auto_unloaded and entry.is_pinned:
                    await self.pool.get_engine(model_id)
                    auto_reloaded = True
            except Exception as exc:
                reload_error = str(exc)
        return {
            "model_id": model_id,
            "settings": self.get_model_settings(model_id)["settings"],
            "requires_reload": requires_reload,
            "auto_unloaded": auto_unloaded,
            "auto_reloaded": auto_reloaded,
            "reload_deferred": reload_deferred,
            "reload_error": reload_error,
        }

    @staticmethod
    def _validate_model_settings(entry, settings: ModelSettings) -> None:
        values = settings.to_dict()
        validate_moe_expert_offload(
            values, model_type=getattr(entry, "config_model_type", None)
        )
        validate_ane_prefill(values, getattr(entry, "config_model_type", None))
        if settings.moe_expert_offload_enabled:
            from ..patches.moe_offload_compat import moe_offload_compatibility

            supported, reason = moe_offload_compatibility(entry.model_path)
            if not supported:
                raise ValueError(reason)
        if settings.mtp_enabled:
            from .model_compat import mtp_compatibility

            compatible, reason = mtp_compatibility(entry.model_path)
            if not compatible:
                raise ValueError(reason)
        if entry.config_model_type == "k2_horizon":
            from ..patches.k2_horizon import validate_chat_template_kwargs

            kwargs = dict(settings.chat_template_kwargs or {})
            if settings.enable_thinking is not None:
                kwargs["enable_thinking"] = settings.enable_thinking
            validate_chat_template_kwargs(kwargs)

    def get_global_settings(self) -> dict[str, Any]:
        settings = self.context.global_settings
        if settings is None:
            raise ManagementError("unavailable", "Global settings unavailable")
        sampling = settings.sampling.to_dict()
        scheduler = settings.scheduler.to_dict()
        return {
            "sampling": sampling,
            "scheduler": {
                key: scheduler[key]
                for key in (
                    "max_concurrent_requests",
                    "embedding_batch_size",
                    "chunked_prefill",
                    "prefill_priority",
                    "decode_fairness",
                )
            },
        }

    async def update_global_settings(
        self, patch: GlobalSettingsPatch
    ) -> dict[str, Any]:
        settings = self.context.global_settings
        if settings is None:
            raise ManagementError("unavailable", "Global settings unavailable")
        scheduler_fields = {
            "max_concurrent_requests",
            "embedding_batch_size",
            "chunked_prefill",
            "prefill_priority",
            "decode_fairness",
        }
        old_sampling = copy.deepcopy(settings.sampling)
        old_scheduler = copy.deepcopy(settings.scheduler)
        sampling_defaults = SamplingSettings()
        scheduler_defaults = SchedulerSettings()
        for name in patch.model_fields_set:
            value = getattr(patch, name)
            target = (
                settings.scheduler if name in scheduler_fields else settings.sampling
            )
            default = (
                scheduler_defaults if name in scheduler_fields else sampling_defaults
            )
            setattr(target, name, getattr(default, name) if value is None else value)
        try:
            settings.save()
        except OSError as exc:
            settings.sampling = old_sampling
            settings.scheduler = old_scheduler
            raise ManagementError("unavailable", str(exc)) from exc
        try:
            self.context.apply_sampling()
            if "embedding_batch_size" in patch.model_fields_set:
                await self.pool.apply_embedding_batch_size(
                    settings.scheduler.embedding_batch_size
                )
        except Exception as exc:
            settings.sampling = old_sampling
            settings.scheduler = old_scheduler
            settings.save()
            self.context.apply_sampling()
            raise ManagementError("unavailable", str(exc)) from exc
        # Scheduler construction fields are persisted for subsequent loads.
        return {
            **self.get_global_settings(),
            "requires_restart": bool(
                patch.model_fields_set & (scheduler_fields - {"embedding_batch_size"})
            ),
        }

    def list_profiles(self, model_id: str) -> dict[str, Any]:
        self._model(model_id)
        return {"profiles": self.manager.list_profiles(model_id)}

    def _profile_settings(
        self, model_id: str, patch: ModelSettingsPatch
    ) -> dict[str, Any]:
        fields = patch.model_dump(exclude_unset=True)
        filtered = filter_profile_fields(fields)
        base = self.manager.get_settings(model_id).to_dict()
        try:
            merged = ModelSettings.from_dict({**base, **filtered})
            self._validate_model_settings(self._model(model_id), merged)
        except (TypeError, ValueError) as exc:
            raise ManagementError("invalid_configuration", str(exc)) from exc
        return cast(dict[str, Any], filtered)

    def create_profile(self, model_id: str, body: ProfileWrite) -> dict[str, Any]:
        self._model(model_id)
        try:
            profile = self.manager.save_profile(
                model_id=model_id,
                name=body.name,
                display_name=body.display_name,
                description=body.description,
                settings=self._profile_settings(model_id, body.settings),
                expose_as_model=body.expose_as_model,
                api_name=body.api_name,
                reserved_model_ids=set(self.pool.get_model_ids()),
            )
        except ValueError as exc:
            raise ManagementError("conflict", str(exc)) from exc
        return {"profile": profile}

    def update_profile(
        self, model_id: str, name: str, body: ProfileUpdate
    ) -> dict[str, Any]:
        self._model(model_id)
        changes = body.model_dump(exclude_unset=True)
        if body.settings is not None:
            changes["settings"] = self._profile_settings(model_id, body.settings)
        try:
            profile = self.manager.update_profile(
                model_id,
                name,
                reserved_model_ids=set(self.pool.get_model_ids()),
                **changes,
            )
        except ValueError as exc:
            raise ManagementError("conflict", str(exc)) from exc
        if profile is None:
            raise ManagementError("not_found", f"Profile not found: {name}")
        return {"profile": profile}

    def delete_profile(self, model_id: str, name: str) -> dict[str, Any]:
        self._model(model_id)
        if not self.manager.delete_profile(model_id, name):
            raise ManagementError("not_found", f"Profile not found: {name}")
        return {"deleted": True, "name": name}

    def apply_profile(self, model_id: str, name: str) -> dict[str, Any]:
        entry = self._model(model_id)

        def validate(values: dict[str, Any]) -> None:
            self._validate_model_settings(entry, ModelSettings.from_dict(values))

        try:
            settings = self.manager.apply_profile(
                model_id, name, settings_sanitizer=validate
            )
        except ValueError as exc:
            raise ManagementError("invalid_configuration", str(exc)) from exc
        if settings is None:
            raise ManagementError("not_found", f"Profile not found: {name}")
        self.pool.apply_settings_overrides(self.manager)
        return {"model_id": model_id, "settings": settings.to_dict()}

    def stats(self, model_id: str, scope: str) -> dict[str, Any]:
        if scope not in {"session", "alltime"}:
            raise ManagementError("invalid_configuration", "Invalid stats scope")
        return cast(
            dict[str, Any],
            get_server_metrics().get_snapshot(model_id=model_id, scope=scope),
        )

    def _loaded_cores(self):
        for model_id in self.pool.get_loaded_model_ids():
            entry = self.pool.get_entry(model_id)
            if entry is None or entry.engine is None:
                continue
            engine = entry.engine
            core = getattr(engine, "engine", engine)
            scheduler = getattr(core, "scheduler", None)
            yield model_id, engine, core, scheduler

    def cache(self) -> dict[str, Any]:
        models = []
        for model_id, engine, _, scheduler in self._loaded_cores():
            get_stats = getattr(engine, "get_runtime_cache_stats", None)
            if not callable(get_stats):
                get_stats = getattr(engine, "get_cache_stats", None)
            if not callable(get_stats) and scheduler is not None:
                get_stats = getattr(scheduler, "get_ssd_cache_stats", None)
            stats = get_stats() if callable(get_stats) else None
            if stats is not None:
                models.append({"model_id": model_id, "stats": stats})
        settings = self.context.global_settings
        cache_dir = (
            settings.cache.get_ssd_cache_dir(settings.base_path)
            if settings is not None
            else None
        )
        return {
            "models": models,
            "ssd_cache_dir": str(cache_dir) if cache_dir else None,
        }

    async def clear_cache(self, kind: str) -> dict[str, Any]:
        if kind not in {"hot", "ssd"}:
            raise ManagementError("not_found", "Unknown cache kind")
        busy = [
            model_id
            for model_id in self.pool.get_loaded_model_ids()
            if (entry := self.pool.get_entry(model_id)) is not None
            and self.pool._entry_is_busy(entry)
        ]
        if busy:
            raise ManagementError(
                "busy",
                f"Cannot clear cache while requests are active: {', '.join(busy)}",
            )
        total = 0
        failures = []
        reclaim = []
        for model_id, _, core, scheduler in self._loaded_cores():
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
            from ..engine_core import get_mlx_executor
            from ..scheduler import _sync_and_clear_cache

            budget = getattr(
                getattr(self.pool, "_scheduler_config", None), "hot_cache_budget", None
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
            settings = self.context.global_settings
            if settings is not None:
                root = Path(settings.cache.get_ssd_cache_dir(settings.base_path))
                for base in (root, root / "deepseek_v41_ced_v1"):
                    for bucket in "0123456789abcdef":
                        for file in (base / bucket).glob("*.safetensors"):
                            try:
                                file.unlink()
                                total += 1
                            except OSError as exc:
                                failures.append(f"{file}: {exc}")
        if failures:
            raise ManagementError("unavailable", "; ".join(failures)[:1000])
        return {"status": "ok", "kind": kind, "total_cleared": total}
