# SPDX-License-Identifier: Apache-2.0
"""Engine management operations, independent of HTTP and server globals."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from omlx_config.model_profiles import (
    PROFILE_FIELDS_SET,
    UNIVERSAL_FIELDS_SET,
    filter_profile_fields,
)
from omlx_config.model_settings import (
    ModelSettings,
    ModelSettingsManager,
    validate_ane_prefill,
    validate_moe_expert_offload,
)
from omlx_config.settings import GlobalSettings, SamplingSettings, SchedulerSettings
from omlx_contracts.management import (
    GlobalSettingsPatch,
    ModelSettingsPatch,
    ProfileUpdate,
    ProfileWrite,
)
from omlx_contracts.runtime import ManagementEngine, RuntimeOperationError
from omlx_runtime.server_metrics import get_server_metrics


class ManagementError(Exception):
    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(detail)


@dataclass(frozen=True)
class ManagementContext:
    engine_pool: ManagementEngine
    settings_manager: ModelSettingsManager
    global_settings: GlobalSettings | None
    get_default_model: Callable[[], str | None]
    set_default_model: Callable[[str | None], None]
    apply_sampling: Callable[[], None]
    get_api_key: Callable[[], str | None] | None = None
    set_api_key: Callable[[str], None] | None = None
    get_bind_host: Callable[[], str] | None = None
    get_server_info: Callable[[], dict[str, Any]] | None = None
    runtime_state: Any | None = None


class ManagementService:
    def __init__(self, context: ManagementContext):
        self.context = context

    @property
    def pool(self) -> ManagementEngine:
        return self.context.engine_pool

    @property
    def manager(self) -> ModelSettingsManager:
        return self.context.settings_manager

    def _require_mutation_admission(self) -> None:
        blocked = not self.pool.management_operation_allowed()
        if blocked:
            raise ManagementError("busy", "Exclusive engine management work is active")

    def _model(self, model_id: str):
        entry = self.pool.get_model_view(model_id)
        if entry is None:
            raise ManagementError("not_found", f"Model not found: {model_id}")
        return entry

    def inventory(self) -> dict[str, Any]:
        """Report every discovered model and its persisted engine settings."""
        status = self.pool.get_status()
        models = []
        for model in status["models"]:
            settings = asdict(self.manager.get_settings(model["id"]))
            entry = self.pool.get_model_view(model["id"])
            is_unloading = bool(
                entry is not None and entry.pending_unload_reason
            ) or self.pool.is_model_unloading(model["id"])
            image_details = {}
            if model.get("engine_type") == "image_generation" and entry is not None:
                from omlx_management.model_control import diffusion_metadata

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
            entry = self.pool.get_model_view(model["id"])
            models.append(
                {
                    "id": model["id"],
                    "loaded": model["loaded"],
                    "is_loading": model["is_loading"],
                    "is_unloading": bool(
                        entry is not None and entry.pending_unload_reason
                    )
                    or self.pool.is_model_unloading(model["id"]),
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
        self._require_mutation_admission()
        entry = self._model(model_id)
        if entry.loaded:
            return {"status": "ok", "model_id": model_id, "message": "Already loaded"}
        if entry.is_loading:
            raise ManagementError("busy", f"Model is already loading: {model_id}")
        await self.pool.load_model(model_id)
        return {"status": "ok", "model_id": model_id, "message": "Loaded"}

    async def unload(self, model_id: str) -> dict[str, Any]:
        self._require_mutation_admission()
        entry = self._model(model_id)
        if not entry.loaded:
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
        self._require_mutation_admission()
        self.manager.reload()
        global_settings = self.context.global_settings
        model_dirs = (
            [str(path) for path in global_settings.get_effective_model_dirs()]
            if global_settings is not None
            else [str(path) for path in getattr(self.pool, "model_directories", [])]
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
        return {"model_id": model_id, "settings": settings}

    async def update_model_settings(
        self, model_id: str, patch: ModelSettingsPatch
    ) -> dict[str, Any]:
        self._require_mutation_admission()
        entry = self._model(model_id)
        if entry.is_loading or self.pool.is_model_unloading(model_id):
            raise ManagementError("busy", "Model is loading or unloading")
        current = self.manager.get_settings(model_id)
        old_type = (entry.model_type, entry.engine_type)
        old_signature = self.pool.runtime_signature(model_id, current)
        values = current.to_dict()
        if patch.model_fields_set == set(ModelSettingsPatch.model_fields):
            values["active_profile_name"] = None
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
            updated = ModelSettings(**values)
            self._validate_model_settings(entry, updated)
            self._validate_drafts(entry, updated)
        except (TypeError, ValueError) as exc:
            raise ManagementError("invalid_configuration", str(exc)) from exc

        detected_type = None
        if (
            "model_type_override" in patch.model_fields_set
            and not updated.model_type_override
        ):
            from omlx_runtime.model_discovery import detect_model_type

            detected_type = detect_model_type(Path(entry.model_path))

        try:
            self.manager.set_settings(model_id, updated)
        except OSError as exc:
            raise ManagementError("unavailable", str(exc)) from exc

        self.pool.set_model_pinned(model_id, updated.is_pinned)
        if "model_type_override" in patch.model_fields_set:
            if updated.model_type_override:
                self.pool.set_model_type(model_id, updated.model_type_override)
            else:
                self.pool.set_model_type(model_id, detected_type)
        else:
            self.pool.apply_settings_overrides(self.manager)
        if updated.is_default:
            self.context.set_default_model(model_id)
        elif current.is_default and not updated.is_default:
            available = [mid for mid in self.pool.get_model_ids() if mid != model_id]
            self.context.set_default_model(available[0] if available else None)
        transition = await self._reload_after_settings_change(
            model_id,
            entry,
            old_type,
            old_signature,
            force_reload=bool(
                patch.model_fields_set
                & {"index_cache_freq", "dflash_enabled", "dflash_draft_model"}
            ),
        )
        return {
            "model_id": model_id,
            "settings": self.get_model_settings(model_id)["settings"],
            **transition,
        }

    async def _reload_after_settings_change(
        self,
        model_id: str,
        entry,
        old_type,
        old_signature,
        *,
        force_reload: bool = False,
    ) -> dict[str, Any]:
        updated = self.manager.get_settings(model_id)
        requires_reload = entry.loaded and (
            old_type
            != (self._model(model_id).model_type, self._model(model_id).engine_type)
            or old_signature != self.pool.runtime_signature(model_id, updated)
            or force_reload
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
                    await self.pool.load_model(model_id)
                    auto_reloaded = True
            except Exception as exc:
                reload_error = str(exc)
        return {
            "requires_reload": requires_reload,
            "auto_unloaded": auto_unloaded,
            "auto_reloaded": auto_reloaded,
            "reload_deferred": reload_deferred,
            "reload_error": reload_error,
        }

    def _validate_drafts(self, entry, updated):
        if updated.dflash_ssd_cache and not self.pool.ssd_cache_enabled:
            raise ValueError(
                "DFlash SSD cache requires a configured paged SSD cache directory"
            )
        for enabled, key in (
            ("dflash_enabled", "dflash_draft_model"),
            ("specprefill_enabled", "specprefill_draft_model"),
            ("vlm_mtp_enabled", "vlm_mtp_draft_model"),
        ):
            if not getattr(updated, enabled):
                continue
            reference = getattr(updated, key)
            draft = self.pool.get_model_view(reference) if reference else None
            if draft is None:
                draft = next(
                    (
                        self.pool.get_model_view(mid)
                        for mid in self.pool.get_model_ids()
                        if self.pool.get_model_view(mid).model_path == reference
                    ),
                    None,
                )
            if draft is None or not (Path(draft.model_path) / "config.json").is_file():
                raise ValueError(
                    f"{key} must identify an installed model with config.json"
                )
            if enabled == "vlm_mtp_enabled":
                import json

                target = json.loads(
                    (Path(entry.model_path) / "config.json").read_text()
                )
                config = json.loads(
                    (Path(draft.model_path) / "config.json").read_text()
                )
                if not isinstance(target, dict) or not isinstance(config, dict):
                    raise ValueError("Target and draft configs must be objects")
                target_family = target.get("model_type", "")
                draft_family = config.get("model_type", "")
                compatible = (
                    draft_family in ("gemma4_assistant", "gemma4_unified_assistant")
                    if target_family.startswith("gemma4")
                    else target_family.startswith(("qwen3_5", "qwen3_6", "qwen3_8"))
                    and draft_family == "qwen3_5_mtp"
                )
                if not compatible:
                    raise ValueError(
                        "VLM MTP target and draft families are incompatible"
                    )
                for dimension in ("hidden_size", "vocab_size"):
                    target_dim = (target.get("text_config") or target).get(dimension)
                    draft_dim = (
                        config.get("backbone_hidden_size")
                        if dimension == "hidden_size"
                        and draft_family.startswith("gemma4")
                        else (config.get("text_config") or config).get(dimension)
                    )
                    if (
                        target_dim is not None
                        and draft_dim is not None
                        and target_dim != draft_dim
                    ):
                        raise ValueError(f"VLM MTP target and draft {dimension} differ")

    @staticmethod
    def _validate_model_settings(entry, settings: ModelSettings) -> None:
        from omlx_management.management_model_options import validate_candidate

        ModelSettingsPatch.model_validate(
            {
                key: value
                for key, value in asdict(settings).items()
                if key in ModelSettingsPatch.model_fields
            }
        )
        validate_candidate(entry, settings)
        import json

        values = settings.to_dict()
        json.dumps(values, allow_nan=False)
        validate_moe_expert_offload(
            values, model_type=getattr(entry, "config_model_type", None)
        )
        validate_ane_prefill(values, getattr(entry, "config_model_type", None))
        if settings.moe_expert_offload_enabled:
            from omlx_runtime.patches.moe_offload_compat import (
                moe_offload_compatibility,
            )

            supported, reason = moe_offload_compatibility(entry.model_path)
            if not supported:
                raise ValueError(reason)
        if settings.mtp_enabled:
            from omlx_management.model_compat import mtp_compatibility

            compatible, reason = mtp_compatibility(entry.model_path)
            if not compatible:
                raise ValueError(reason)
        if entry.config_model_type == "k2_horizon":
            from omlx_runtime.patches.k2_horizon import validate_chat_template_kwargs

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
        self._require_mutation_admission()
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
        base = {
            k: v
            for k, v in self.manager.get_settings(model_id).to_dict().items()
            if k not in UNIVERSAL_FIELDS_SET
        }
        try:
            defaults = asdict(ModelSettings())
            merged = ModelSettings(
                **{
                    **base,
                    **{
                        key: defaults[key] if value is None else value
                        for key, value in filtered.items()
                    },
                }
            )
            self._validate_model_settings(self._model(model_id), merged)
            self._validate_drafts(self._model(model_id), merged)
        except (TypeError, ValueError) as exc:
            raise ManagementError("invalid_configuration", str(exc)) from exc
        return {
            key: value for key, value in fields.items() if key in PROFILE_FIELDS_SET
        }

    def create_profile(self, model_id: str, body: ProfileWrite) -> dict[str, Any]:
        self._model(model_id)
        snapshot = self.manager.snapshot_profiles()
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
        except OSError as exc:
            self.manager.restore_profiles(snapshot)
            raise ManagementError("unavailable", str(exc)) from exc
        return {"profile": profile}

    def update_profile(
        self, model_id: str, name: str, body: ProfileUpdate
    ) -> dict[str, Any]:
        self._model(model_id)
        changes = body.model_dump(exclude_unset=True)
        if body.settings is not None:
            changes["settings"] = self._profile_settings(model_id, body.settings)
        else:
            existing = self.manager.get_profile(model_id, name)
            if existing is None:
                raise ManagementError("not_found", f"Profile not found: {name}")
            try:
                self._profile_settings(
                    model_id,
                    ModelSettingsPatch.model_validate(existing.get("settings", {})),
                )
            except ValueError as exc:
                raise ManagementError("invalid_configuration", str(exc)) from exc
        snapshot = self.manager.snapshot_profiles()
        try:
            profile = self.manager.update_profile(
                model_id,
                name,
                reserved_model_ids=set(self.pool.get_model_ids()),
                **changes,
            )
        except ValueError as exc:
            raise ManagementError("conflict", str(exc)) from exc
        except OSError as exc:
            self.manager.restore_profiles(snapshot)
            raise ManagementError("unavailable", str(exc)) from exc
        if profile is None:
            raise ManagementError("not_found", f"Profile not found: {name}")
        return {"profile": profile}

    def delete_profile(self, model_id: str, name: str) -> dict[str, Any]:
        self._model(model_id)
        try:
            deleted = self.manager.delete_profile(model_id, name)
        except OSError as exc:
            raise ManagementError("unavailable", str(exc)) from exc
        if not deleted:
            raise ManagementError("not_found", f"Profile not found: {name}")
        return {"deleted": True, "name": name}

    async def apply_profile(self, model_id: str, name: str) -> dict[str, Any]:
        self._require_mutation_admission()
        entry = self._model(model_id)
        if entry.is_loading or self.pool.is_model_unloading(model_id):
            raise ManagementError("busy", "Model is loading or unloading")
        previous = self.manager.get_settings(model_id)
        old_type = (entry.model_type, entry.engine_type)
        old_signature = self.pool.runtime_signature(model_id, previous)

        def validate(values: dict[str, Any]) -> None:
            ModelSettingsPatch(
                **{
                    key: value
                    for key, value in values.items()
                    if key in ModelSettingsPatch.model_fields
                }
            )
            candidate = ModelSettings(**values)
            self._validate_model_settings(entry, candidate)
            self._validate_drafts(entry, candidate)

        try:
            settings = self.manager.apply_profile(
                model_id, name, settings_sanitizer=validate
            )
        except ValueError as exc:
            raise ManagementError("invalid_configuration", str(exc)) from exc
        except OSError as exc:
            raise ManagementError("unavailable", str(exc)) from exc
        if settings is None:
            raise ManagementError("not_found", f"Profile not found: {name}")
        self.pool.apply_settings_overrides(self.manager)
        transition = await self._reload_after_settings_change(
            model_id,
            entry,
            old_type,
            old_signature,
            force_reload=any(
                getattr(previous, field) != getattr(settings, field)
                for field in (
                    "index_cache_freq",
                    "dflash_enabled",
                    "dflash_draft_model",
                )
            ),
        )
        return {"model_id": model_id, "settings": settings.to_dict(), **transition}

    def stats(self, model_id: str, scope: str) -> dict[str, Any]:
        if scope not in {"session", "alltime"}:
            raise ManagementError("invalid_configuration", "Invalid stats scope")
        return cast(
            dict[str, Any],
            (
                getattr(self.context.runtime_state, "metrics", None)
                or get_server_metrics()
            ).get_snapshot(model_id=model_id, scope=scope),
        )

    def cache(self) -> dict[str, Any]:
        settings = self.context.global_settings
        directory = (
            settings.cache.get_ssd_cache_dir(settings.base_path) if settings else None
        )
        return cast(dict[str, Any], self.pool.cache_status(directory))

    async def clear_cache(self, kind: str) -> dict[str, Any]:
        self._require_mutation_admission()
        settings = self.context.global_settings
        directory = (
            settings.cache.get_ssd_cache_dir(settings.base_path) if settings else None
        )
        try:
            return cast(dict[str, Any], await self.pool.clear_cache(kind, directory))
        except RuntimeOperationError as exc:
            raise ManagementError(exc.code, exc.detail) from exc
