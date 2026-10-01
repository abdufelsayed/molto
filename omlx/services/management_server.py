# SPDX-License-Identifier: Apache-2.0
"""Server configuration and credentials, without server-global imports."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import shlex
import tempfile
import uuid
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, get_type_hints

import httpx
from packaging.version import InvalidVersion, Version
from pydantic import TypeAdapter, ValidationError

from .. import __version__
from ..auth import validate_api_key
from ..settings import GlobalSettings, SubKeyEntry
from ..utils.network import network_auth_error
from .management import ManagementContext, ManagementError

SECTIONS = tuple(
    f.name for f in fields(GlobalSettings) if f.name not in {"base_path", "auth"}
)
CHOICES = {
    "server.log_level": ["trace", "debug", "info", "warning", "error", "critical"],
    "server.sse_keepalive_mode": ["chunk", "comment", "off"],
    "server.burst_decode_mode": ["off", "light", "balanced", "aggressive"],
    "memory.memory_guard_tier": ["safe", "balanced", "aggressive", "custom"],
    "scheduler.prefill_priority": ["context", "speed"],
    "cache.gdn_sidecar_state_dtype": ["fp32", "bf16", "int8", "rht_int8", "rht_int16"],
    "claude_code.mode": ["local", "cloud"],
    "integrations.openclaw_tools_profile": ["minimal", "coding", "messaging", "full"],
    "integrations.web_search_provider": [
        "ddgs",
        "ddgs_custom",
        "duckduckgo",
        "brave",
        "searxng",
    ],
    "integrations.web_search_content_mode": ["snippet", "full"],
}
BOUNDS = {
    "server.port": (1, 65535),
    "server.max_image_side_length": (0, None),
    "server.gpu_keep_warm_interval": (0, None),
    "memory.memory_guard_custom_ceiling_gb": (0, None),
    "memory.soft_threshold": (0.01, 1),
    "memory.hard_threshold": (0.01, 1),
    "memory.prefill_safe_zone_ratio": (0.5, 0.99),
    "memory.prefill_min_chunk_tokens": (1, 1024),
    "scheduler.max_concurrent_requests": (1, None),
    "scheduler.embedding_batch_size": (1, None),
    "cache.initial_cache_blocks": (1, None),
    "sampling.max_context_window": (1, None),
    "sampling.max_context_window_policy": (1, None),
    "sampling.max_tokens": (1, None),
    "sampling.temperature": (0, 2),
    "sampling.top_p": (0, 1),
    "sampling.top_k": (0, None),
    "sampling.repetition_penalty": (0.01, None),
    "logging.retention_days": (1, None),
    "idle_timeout.idle_timeout_seconds": (0, None),
    "integrations.markitdown_max_file_size_mb": (1, None),
    "integrations.markitdown_max_files_per_request": (1, None),
    "integrations.web_search_max_results": (1, 10),
    "integrations.web_search_content_max_chars": (1, None),
}
# These consumers look up the current dataclass on each request. All engine,
# environment, memory, directory, logging and network changes require restart.
LIVE = {
    "model.model_fallback",
    "model.hide_helper_models",
    "mcp.expose_tools",
    "idle_timeout.idle_timeout_seconds",
}
DESCRIPTIONS = {
    "model.model_dirs": "Ordered model roots. An empty list uses the models directory under the base path.",
    "model.model_dir": "Legacy primary model root. Prefer model_dirs.",
    "cache.ssd_cache_max_size": "SSD cache limit such as 100GB, or auto for half the available space.",
    "cache.hot_cache_max_size": "RAM cache limit such as 8GB. Use 0 to disable.",
    "network.ca_bundle": "Path to a readable PEM CA bundle used by outbound clients after restart.",
    "mcp.config_path": "Path to a JSON MCP configuration loaded at server startup.",
    "server.host": "Single public bind address. Use 0.0.0.0 for all IPv4 interfaces. Non-loopback binds require API key verification.",
    "integrations.web_search_brave_api_key": "Brave Search API subscription key.",
    "sampling.max_context_window_policy": "Optional global cap on discovered native context lengths.",
}


def invalid(detail: str):
    raise ManagementError("invalid_configuration", detail)


class ServerManagementService:
    def __init__(self, context: ManagementContext):
        self.context = context

    @property
    def settings(self) -> GlobalSettings:
        if self.context.global_settings is None:
            raise ManagementError("unavailable", "Global settings unavailable")
        return self.context.global_settings

    def _live(self, path: str) -> bool:
        return path in LIVE or path.startswith("sampling.")

    def describe(self, settings: GlobalSettings | None = None) -> dict[str, Any]:
        settings = settings or self.settings
        defaults = GlobalSettings(base_path=settings.base_path)
        sections = {
            name: {
                f.name: copy.deepcopy(getattr(getattr(settings, name), f.name))
                for f in fields(getattr(settings, name))
            }
            for name in SECTIONS
        }
        metadata = []
        for section in SECTIONS:
            default_section = getattr(defaults, section)
            hints = get_type_hints(type(default_section))
            for field in fields(default_section):
                path = f"{section}.{field.name}"
                schema = TypeAdapter(hints[field.name]).json_schema()
                alternatives = schema.get("anyOf", [schema])
                kind = next(
                    (
                        part.get("type")
                        for part in alternatives
                        if part.get("type") != "null"
                    ),
                    "string",
                )
                minimum, maximum = BOUNDS.get(path, (None, None))
                metadata.append(
                    {
                        "key": field.name,
                        "section": section,
                        "label": field.name.replace("_", " ").capitalize(),
                        "description": DESCRIPTIONS.get(
                            path,
                            f"{field.name.replace('_', ' ').capitalize()} for {section.replace('_', ' ')}.",
                        ),
                        "type": kind,
                        "nullable": any(
                            part.get("type") == "null" for part in alternatives
                        ),
                        "default": getattr(default_section, field.name),
                        "choices": CHOICES.get(path, schema.get("enum")),
                        "minimum": minimum,
                        "maximum": maximum,
                        "secret": "api_key" in field.name,
                        "restart_required": not self._live(path),
                    }
                )
        return {
            "base_path": str(settings.base_path),
            "sections": sections,
            "fields": metadata,
            "effective_model_dirs": [
                str(p) for p in settings.get_effective_model_dirs()
            ],
        }

    def defaults(self):
        return self.describe(GlobalSettings(base_path=self.settings.base_path))

    def _candidate(self, patch: dict[str, Any]) -> tuple[GlobalSettings, list[str]]:
        if set(patch) == {"sections"}:
            patch = patch["sections"]
        if not isinstance(patch, dict):
            invalid("Settings patch must be an object")
        candidate = copy.deepcopy(self.settings)
        changed = []
        requested = []
        for section, values in patch.items():
            if section not in SECTIONS or not isinstance(values, dict):
                invalid(f"Unknown settings section or invalid object: {section}")
            target = getattr(candidate, section)
            hints = get_type_hints(type(target))
            for key, value in values.items():
                if key not in hints:
                    invalid(f"Unknown field: {section}.{key}")
                path = f"{section}.{key}"
                try:
                    value = TypeAdapter(hints[key]).validate_python(value, strict=True)
                except ValidationError:
                    invalid(f"Invalid type or value for {path}")
                if isinstance(value, float) and not math.isfinite(value):
                    invalid(f"{path} must be finite")
                if path in CHOICES and value not in CHOICES[path]:
                    invalid(f"{path} must be one of {', '.join(CHOICES[path])}")
                low, high = BOUNDS.get(path, (None, None))
                if value is not None and (
                    (low is not None and value < low)
                    or (high is not None and value > high)
                ):
                    invalid(f"{path} is outside the allowed range")
                if path == "server.host":
                    value = value.strip()
                    if "," in value:
                        invalid(
                            "server.host must contain exactly one bind address; use 0.0.0.0 to listen on all IPv4 interfaces"
                        )
                if path == "network.ca_bundle" and value:
                    try:
                        value = str(Path(value).expanduser().resolve())
                    except (OSError, ValueError) as exc:
                        invalid(f"Invalid CA bundle path: {exc}")
                requested.append(path)
                if getattr(target, key) != value:
                    setattr(target, key, value)
                    changed.append(path)
        if "model.model_dirs" in requested:
            candidate.model.model_dir = (
                candidate.model.model_dirs[0] if candidate.model.model_dirs else None
            )
        self._validate(candidate)
        persisted = self._persisted()
        for path in requested:
            section, _ = path.split(".", 1)
            if path not in changed and any(
                persisted.get(section, {}).get(key) != value
                for key, value in self._serialized_fields(candidate, path).items()
            ):
                changed.append(path)
        return candidate, changed

    def _validate(self, candidate: GlobalSettings):
        if isinstance(candidate.server.host, str) and "," in candidate.server.host:
            invalid(
                "server.host must contain exactly one bind address; use 0.0.0.0 to listen on all IPv4 interfaces"
            )
        errors = candidate.validate()
        if not candidate.memory.soft_threshold < candidate.memory.hard_threshold:
            errors.append("memory.soft_threshold must be below memory.hard_threshold")
        for root in candidate.model.model_dirs:
            if not root.strip() or "\x00" in root:
                errors.append("Model roots must be nonempty paths")
        for section, key in [("network", "ca_bundle"), ("mcp", "config_path")]:
            raw = getattr(getattr(candidate, section), key)
            if raw and not Path(raw).expanduser().is_file():
                errors.append(f"{section}.{key} must name an existing file")
        from urllib.parse import urlsplit

        for name, raw in [
            (
                "integrations.web_search_searxng_url",
                candidate.integrations.web_search_searxng_url,
            ),
            ("huggingface.endpoint", candidate.huggingface.endpoint),
            ("modelscope.endpoint", candidate.modelscope.endpoint),
            ("network.http_proxy", candidate.network.http_proxy),
            ("network.https_proxy", candidate.network.https_proxy),
        ]:
            if raw:
                try:
                    parsed = urlsplit(raw)
                    _ = parsed.port
                    valid = parsed.scheme in {"http", "https"} and bool(parsed.hostname)
                except ValueError:
                    valid = False
                if not valid:
                    errors.append(
                        f"{name} must be an HTTP(S) URL with a valid host and port"
                    )
        for section in SECTIONS:
            for field in fields(getattr(candidate, section)):
                value = getattr(getattr(candidate, section), field.name)
                if isinstance(value, str) and "\x00" in value:
                    errors.append(
                        f"{section}.{field.name} cannot contain a null character"
                    )
        from ..websearch import DDGS_TEXT_BACKENDS

        unknown = set(
            filter(
                None,
                (
                    p.strip()
                    for p in candidate.integrations.web_search_ddgs_backends.split(",")
                ),
            )
        ) - set(DDGS_TEXT_BACKENDS)
        if unknown:
            errors.append("Unknown DDGS backends: " + ", ".join(sorted(unknown)))
        host = (
            self.context.get_bind_host()
            if self.context.get_bind_host
            else self.settings.server.host
        )
        active_error = network_auth_error(
            host, candidate.auth.api_key, candidate.auth.skip_api_key_verification
        )
        if active_error:
            errors.append("Active bind: " + active_error)
        if errors:
            invalid("; ".join(errors))

    def _persisted(self) -> dict[str, Any]:
        path = self.settings.base_path / "settings.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("Invalid settings object")
                return data
            except (OSError, ValueError) as exc:
                raise ManagementError(
                    "persistence_failed", "Cannot read persisted settings"
                ) from exc
        data = GlobalSettings(base_path=self.settings.base_path).to_dict()
        data.pop("base_path", None)
        return data

    @staticmethod
    def _serialized_fields(candidate: GlobalSettings, path: str) -> dict[str, Any]:
        section, key = path.split(".", 1)
        serialized = getattr(candidate, section).to_dict()
        if path == "cache.gdn_sidecar_state_dtype":
            return {"gdn_sidecar_precision": serialized["gdn_sidecar_precision"]}
        if path == "cache.gdn_ssd_split_enabled":
            return {
                key: serialized[key],
                "gdn_snapshot_storage": serialized["gdn_snapshot_storage"],
            }
        if path == "model.model_dirs":
            return {key: serialized[key], "model_dir": serialized["model_dir"]}
        return {key: serialized[key]}

    def _save(self, candidate: GlobalSettings, persisted_fields: list[str]):
        # Start from disk, never the effective CLI/environment configuration.
        data = self._persisted()
        for path in persisted_fields:
            section, _ = path.split(".", 1)
            data.setdefault(section, {}).update(
                self._serialized_fields(candidate, path)
            )
        candidate.base_path.mkdir(parents=True, exist_ok=True)
        candidate._save_data(data)

    def _commit(
        self,
        candidate: GlobalSettings,
        persisted_fields: list[str],
        *,
        main_key: bool = False,
        sampling: bool = False,
    ):
        old = copy.deepcopy(self.settings)
        settings_path = self.settings.base_path / "settings.json"
        existed = settings_path.exists()
        persisted_before = self._persisted()
        try:
            self._save(candidate, persisted_fields)
        except Exception as exc:
            raise ManagementError(
                "persistence_failed", "Settings could not be saved"
            ) from exc
        # Keep the GlobalSettings identity used by existing consumers.
        for field in fields(GlobalSettings):
            if field.name != "base_path":
                setattr(self.settings, field.name, getattr(candidate, field.name))
        try:
            if main_key and self.context.set_api_key:
                self.context.set_api_key(candidate.auth.api_key or "")
            if sampling:
                self.context.apply_sampling()
        except Exception as exc:
            for field in fields(GlobalSettings):
                if field.name != "base_path":
                    setattr(self.settings, field.name, getattr(old, field.name))
            rollback_errors = []
            if main_key and self.context.set_api_key:
                try:
                    self.context.set_api_key(old.auth.api_key or "")
                except Exception as rollback:
                    rollback_errors.append(("main API key callback", rollback))
            if sampling:
                try:
                    self.context.apply_sampling()
                except Exception as rollback:
                    rollback_errors.append(("sampling callback", rollback))
            # Disk restoration must run even when a runtime callback fails.
            try:
                if existed:
                    old._save_data(persisted_before)
                else:
                    settings_path.unlink(missing_ok=True)
            except Exception as rollback:
                rollback_errors.append(("persisted settings", rollback))
            if rollback_errors:
                failed = ", ".join(component for component, _ in rollback_errors)
                disk_restored = not any(
                    component == "persisted settings"
                    for component, _ in rollback_errors
                )
                disk_status = (
                    "persisted settings restored"
                    if disk_restored
                    else "persisted settings restoration failed"
                )
                raise ManagementError(
                    "rollback_failed",
                    f"Runtime update failed; rollback failed for {failed}; {disk_status}",
                ) from rollback_errors[0][1]
            raise ManagementError(
                "runtime_failed", "Runtime update failed; settings restored"
            ) from exc

    def patch(self, patch: dict[str, Any]):
        candidate, changed = self._candidate(patch)
        if changed:
            self._commit(
                candidate,
                changed,
                sampling=any(p.startswith("sampling.") for p in changed),
            )
        return {
            **self.describe(),
            "changed": changed,
            "live_applied": [p for p in changed if self._live(p)],
            "restart_required": [p for p in changed if not self._live(p)],
        }

    def _ids(self) -> dict[str, str]:
        path = self.settings.base_path / "management-key-ids.json"
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in data.items()
            ):
                raise ValueError("Invalid registry")
            return data
        except (OSError, ValueError) as exc:
            raise ManagementError(
                "persistence_failed", "Cannot read key identifiers"
            ) from exc

    @staticmethod
    def _fingerprint(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def _id(self, entry: SubKeyEntry, ids: dict[str, str]) -> str:
        fingerprint = self._fingerprint(entry.key)
        # Legacy keys get opaque deterministic identifiers without a GET mutation.
        return ids.get(
            fingerprint,
            str(
                uuid.uuid5(
                    uuid.NAMESPACE_OID, str(self.settings.base_path) + fingerprint
                )
            ),
        )

    def _write_ids(self, ids: dict[str, str]):
        self.settings.base_path.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(
            prefix=".management-key-ids-", dir=self.settings.base_path
        )
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(ids, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.settings.base_path / "management-key-ids.json")
        finally:
            Path(name).unlink(missing_ok=True)

    def keys(self):
        ids = self._ids()
        key = (
            self.context.get_api_key()
            if self.context.get_api_key
            else self.settings.auth.api_key
        )
        return {
            "main_key": key or "",
            "sub_keys": [
                {"id": self._id(e, ids), **e.to_dict()}
                for e in self.settings.auth.sub_keys
            ],
            "policy": {
                "skip_api_key_verification": self.settings.auth.skip_api_key_verification,
                "allow_unauthenticated_inference": self.settings.auth.allow_unauthenticated_inference,
            },
        }

    def _validate_key(
        self, key: str, candidate: GlobalSettings, exclude: int | None = None
    ):
        valid, message = validate_api_key(key)
        if not valid:
            invalid(message)
        main = (
            self.context.get_api_key()
            if self.context.get_api_key
            else candidate.auth.api_key
        )
        if (
            key == main
            or key == candidate.auth.api_key
            or any(
                e.key == key
                for i, e in enumerate(candidate.auth.sub_keys)
                if i != exclude
            )
        ):
            raise ManagementError("conflict", "API key already exists")

    def subkey(
        self, values: dict[str, Any], key_id: str | None = None, delete: bool = False
    ):
        if set(values) - {"name", "key"}:
            invalid("Unknown subkey fields")
        if "name" in values and (
            not isinstance(values["name"], str) or len(values["name"]) > 200
        ):
            invalid("Subkey name must be a string of at most 200 characters")
        candidate = copy.deepcopy(self.settings)
        ids = self._ids()
        original_ids = dict(ids)
        index = None
        if key_id:
            index = next(
                (
                    i
                    for i, e in enumerate(candidate.auth.sub_keys)
                    if self._id(e, ids) == key_id
                ),
                None,
            )
            if index is None:
                raise ManagementError("not_found", "Subkey not found")
        if delete:
            candidate.auth.sub_keys.pop(index)
        else:
            key = values.get(
                "key",
                (
                    uuid.uuid4().hex
                    if index is None
                    else candidate.auth.sub_keys[index].key
                ),
            )
            if not isinstance(key, str):
                invalid("API key must be a string")
            self._validate_key(key, candidate, index)
            if index is None:
                entry = SubKeyEntry(
                    key=key,
                    name=values.get("name", ""),
                    created_at=datetime.now(UTC).isoformat(),
                )
                candidate.auth.sub_keys.append(entry)
                key_id = str(uuid.uuid4())
            else:
                entry = candidate.auth.sub_keys[index]
                entry.key = key
                entry.name = values.get("name", entry.name)
            ids[self._fingerprint(key)] = key_id
        # Publish identifier aliases first. If saving settings fails, extra aliases
        # are harmless: credentials and authorization remain unchanged.
        try:
            self._write_ids(ids)
        except Exception as exc:
            raise ManagementError(
                "persistence_failed", "Key identifiers could not be saved"
            ) from exc
        try:
            self._commit(candidate, ["auth.sub_keys"])
        except ManagementError:
            # The credential transaction did not commit. Restore registry aliases.
            try:
                self._write_ids(original_ids)
            except Exception as exc:
                raise ManagementError(
                    "rollback_failed",
                    "Credentials unchanged; identifier rollback failed",
                ) from exc
            raise
        return (
            {"deleted": True, "id": key_id}
            if delete
            else {"sub_key": {"id": key_id, **entry.to_dict()}}
        )

    def main_key(self, key: str):
        valid, message = validate_api_key(key)
        if not valid:
            invalid(message)
        if any(e.key == key for e in self.settings.auth.sub_keys):
            raise ManagementError("conflict", "API key already exists")
        if not self.context.set_api_key:
            raise ManagementError("unavailable", "Live main-key rotation unavailable")
        candidate = copy.deepcopy(self.settings)
        candidate.auth.api_key = key
        self._validate(candidate)
        self._commit(candidate, ["auth.api_key"], main_key=True)
        return {"main_key": key, "live_applied": True}

    def policy(self, values: dict[str, Any]):
        if not values or set(values) - {
            "skip_api_key_verification",
            "allow_unauthenticated_inference",
        }:
            invalid("Unknown or empty authentication policy")
        candidate = copy.deepcopy(self.settings)
        for key, value in values.items():
            if type(value) is not bool:
                invalid(f"{key} must be a boolean")
            setattr(candidate.auth, key, value)
        self._validate(candidate)
        self._commit(candidate, [f"auth.{key}" for key in values])
        return {"policy": self.keys()["policy"], "live_applied": True}

    def resources(
        self,
        tier: str | None = None,
        custom_ceiling_gb: float | None = None,
        guard_enabled: bool | None = None,
    ):
        """Read hardware and preview limits without starting or changing an enforcer."""
        from .. import process_memory_enforcer as memory
        from ..settings import detect_system_memory
        from ..utils import psutil_compat

        if guard_enabled is not None and type(guard_enabled) is not bool:
            invalid("guard_enabled must be a boolean")
        selected_guard = (
            self.settings.memory.prefill_memory_guard
            if guard_enabled is None
            else guard_enabled
        )
        selected_tier = tier or self.settings.memory.memory_guard_tier
        selected_custom = (
            self.settings.memory.memory_guard_custom_ceiling_gb
            if custom_ceiling_gb is None
            else custom_ceiling_gb
        )
        if selected_tier not in {"safe", "balanced", "aggressive", "custom"}:
            invalid("Resource tier must be safe, balanced, aggressive, or custom")
        if (
            not math.isfinite(selected_custom)
            or selected_custom < 0
            or (selected_tier == "custom" and selected_custom <= 0)
        ):
            invalid("A custom ceiling must be finite and greater than zero")
        warnings = []

        def read_size(label, callback):
            try:
                value = int(callback())
                if value <= 0:
                    raise ValueError("Unavailable value")
                return value
            except Exception:
                warnings.append(f"{label} unavailable")
                return None

        physical = read_size("Physical memory", detect_system_memory)
        available = read_size(
            "Available memory", lambda: psutil_compat.virtual_memory().available
        )
        metal_cap = read_size(
            "Metal allocation cap", memory.get_effective_metal_cap_bytes
        )
        try:
            # Tier math uses the historical runtime RAM fallback internally.
            # Never present those guessed ceilings as a hardware preview.
            previews = memory.preview_tier_ceilings() if physical is not None else {}
        except Exception:
            previews = {}
            warnings.append("Tier previews unavailable")

        def preview(tier_name, custom_gb, guard_enabled):
            source = previews.get(tier_name)
            if source is None:
                return None
            result = copy.deepcopy(source)
            if tier_name == "custom":
                from decimal import Decimal

                result["dynamic_bytes"] = int(Decimal(str(custom_gb)) * 1024**3)
                components = {
                    name: result.get(f"{name}_bytes", 0)
                    for name in ("static", "dynamic", "metal_cap")
                    if result.get(f"{name}_bytes", 0) > 0
                }
                result["ceiling_bytes"] = min(components.values()) if components else 0
                result["binding"] = next(
                    (
                        name
                        for name, value in components.items()
                        if value == result["ceiling_bytes"]
                    ),
                    "",
                )
            if not guard_enabled:
                result["ceiling_bytes"] = 0
                result["binding"] = "disabled"
            return result

        defaults = GlobalSettings(base_path=self.settings.base_path).memory
        saved_memory = self._persisted().get("memory", {})
        saved_tier = saved_memory.get("memory_guard_tier", defaults.memory_guard_tier)
        saved_custom = saved_memory.get(
            "memory_guard_custom_ceiling_gb", defaults.memory_guard_custom_ceiling_gb
        )
        saved_guard = saved_memory.get(
            "prefill_memory_guard", defaults.prefill_memory_guard
        )
        saved = {
            "tier": saved_tier,
            "custom_ceiling_gb": saved_custom,
            "guard_enabled": saved_guard,
            "preview": preview(saved_tier, saved_custom, saved_guard),
        }
        draft = {
            "tier": selected_tier,
            "custom_ceiling_gb": selected_custom,
            "guard_enabled": selected_guard,
            "preview": preview(selected_tier, selected_custom, selected_guard),
        }
        state = self.context.runtime_state
        enforcer = getattr(state, "process_memory_enforcer", None)
        if enforcer is None:
            enforcer = getattr(
                self.context.engine_pool, "_process_memory_enforcer", None
            )
        runtime = {
            "available": False,
            "tier": None,
            "custom_ceiling_gb": None,
            "guard_enabled": None,
            "ceiling_bytes": None,
            "breakdown": None,
            "wired_limit_request_bytes": 0,
        }
        if enforcer is not None:
            try:
                breakdown = enforcer.get_ceiling_breakdown()
                runtime = {
                    "available": True,
                    "tier": enforcer._memory_guard_tier,
                    "custom_ceiling_gb": enforcer._memory_guard_custom_ceiling_bytes
                    / 1024**3,
                    "guard_enabled": enforcer._prefill_memory_guard,
                    "ceiling_bytes": int(breakdown["hard_limit"]),
                    "breakdown": breakdown,
                    "wired_limit_request_bytes": int(
                        getattr(enforcer, "_metal_wired_limit_request", 0) or 0
                    ),
                }
            except Exception:
                warnings.append("Active memory enforcer status unavailable")
        request = runtime["wired_limit_request_bytes"]
        limited = metal_cap < request if metal_cap is not None and request > 0 else None
        recommended = None
        if limited and physical is not None:
            # Retained recommendation leaves 5% RAM and floors to whole MiB.
            try:
                suggestion = memory._wired_limit_suggestion_bytes(request)
                recommended = (
                    max(0, min(suggestion, physical - physical // 20))
                    // 1024**2
                    * 1024**2
                )
                if recommended <= metal_cap:
                    recommended = None
            except Exception:
                warnings.append("Wired-memory recommendation unavailable")
        command = None
        recommended_mib = None
        if recommended is not None:
            recommended_mib = recommended // 1024**2
            command = f"sudo sysctl iogpu.wired_limit_mb={recommended_mib}"
        if limited:
            warnings.append(
                "The Metal allocation cap is below the active enforcer's wired-memory request"
            )
        return {
            "hardware": {
                "physical_memory_bytes": physical,
                "available_memory_bytes": available,
                "metal_cap_bytes": metal_cap,
            },
            "tier_previews": previews,
            "saved": saved,
            "runtime": runtime,
            "draft": draft,
            "wired_limit": {
                "limited": limited,
                "recommended_bytes": recommended,
                "recommended_mib": recommended_mib,
                "command": command,
                "copy_only": True,
            },
            "warnings": warnings,
        }

    def info(self):
        supplied = (
            self.context.get_server_info() if self.context.get_server_info else {}
        )
        state = self.context.runtime_state
        request_restart = getattr(state, "request_restart", None)
        return {
            "version": __version__,
            "base_path": str(self.settings.base_path),
            "host": (
                self.context.get_bind_host()
                if self.context.get_bind_host
                else self.settings.server.host
            ),
            "port": getattr(state, "bind_port", None) or self.settings.server.port,
            "configured_port": self.settings.server.port,
            "restart_supported": callable(request_restart),
            **supplied,
        }

    async def restart(self):
        callback = getattr(self.context.runtime_state, "request_restart", None)
        if not callable(callback) or not self.info()["restart_supported"]:
            raise ManagementError(
                "unavailable", "Restart requires an active supervisor"
            )
        result = callback()
        if inspect.isawaitable(result):
            result = await result
        if result is False:
            raise ManagementError("busy", "Restart already pending or server busy")
        return {"status": "restarting"}

    def integrations(self):
        from ..integrations import INTEGRATIONS
        from ..utils.install import get_cli_command_prefix

        prefix = get_cli_command_prefix()
        commands = []
        info = self.info()
        host = str(info["host"])
        port = str(info["port"])
        for name, integration in INTEGRATIONS.items():
            model = getattr(self.settings.integrations, f"{name}_model", None)
            command = f"{prefix} launch {shlex.quote(name)}"
            if model:
                command += " --model " + shlex.quote(model)
            command += " --host " + shlex.quote(host) + " --port " + shlex.quote(port)
            if name == "openclaw":
                command += " --tools-profile " + shlex.quote(
                    self.settings.integrations.openclaw_tools_profile
                )
            if name == "claude":
                for tier in ("opus", "sonnet", "haiku"):
                    tier_model = getattr(self.settings.claude_code, f"{tier}_model")
                    if tier_model:
                        command += f" --{tier} " + shlex.quote(tier_model)
            commands.append(
                {
                    "id": name,
                    "name": getattr(integration, "display_name", name),
                    "model": model,
                    "command": command,
                    "description": "Run locally in a terminal. Launch may update the integration's configuration.",
                }
            )
        return {"integrations": commands}

    async def web_search_test(self, values: dict[str, Any]):
        allowed = {
            "provider",
            "brave_api_key",
            "searxng_url",
            "ddgs_backends",
            "max_results",
        }
        if set(values) - allowed:
            invalid("Unknown web-search test fields")
        mapping = {
            "provider": "web_search_provider",
            "brave_api_key": "web_search_brave_api_key",
            "searxng_url": "web_search_searxng_url",
            "ddgs_backends": "web_search_ddgs_backends",
            "max_results": "web_search_max_results",
        }
        pending = {mapping[k]: v for k, v in values.items()}
        candidate, _ = self._candidate({"integrations": pending})
        from ..websearch import run_web_search_test

        args = {k: getattr(candidate.integrations, v) for k, v in mapping.items()}
        return await run_web_search_test(**args)

    async def update(self, channel: str = "stable"):
        if channel not in {"stable", "beta"}:
            invalid("Update channel must be stable or beta")
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(
                    "https://api.github.com/repos/jundot/omlx/releases",
                    params={"per_page": 20},
                )
                response.raise_for_status()
                releases = response.json()
            if not isinstance(releases, list):
                raise ValueError("Invalid releases response")
            candidates = []
            for release in releases:
                if release.get("draft"):
                    continue
                try:
                    version = Version(release["tag_name"].lstrip("v"))
                except (InvalidVersion, KeyError):
                    continue
                if channel == "stable" and (
                    release.get("prerelease")
                    or version.is_prerelease
                    or version.is_devrelease
                ):
                    continue
                candidates.append((version, release))
            if not candidates:
                return {
                    "status": "unavailable",
                    "update_available": None,
                    "update_channel": channel,
                    "error": "No eligible releases",
                }
            version, release = max(candidates, key=lambda entry: entry[0])
            return {
                "status": "checked",
                "update_available": version > Version(__version__),
                "current_version": __version__,
                "latest_version": str(version),
                "release_url": release.get("html_url"),
                "update_channel": channel,
            }
        except Exception:
            return {
                "status": "failed",
                "update_available": None,
                "update_channel": channel,
                "error": "Release check failed",
            }
