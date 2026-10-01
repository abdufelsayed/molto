# SPDX-License-Identifier: Apache-2.0
"""Loopback-only initial credential setup, independent of server globals."""

from __future__ import annotations

import copy
import json
from typing import Any, cast

from omlx_config.auth import validate_api_key
from omlx_config.utils.network import is_loopback_bind, is_loopback_bind_host

from omlx_management.management import ManagementContext, ManagementError


class ManagementSetupService:
    def __init__(self, context: ManagementContext):
        self.context = context

    @property
    def settings(self):
        if self.context.global_settings is None:
            raise ManagementError("unavailable", "Server settings unavailable")
        return self.context.global_settings

    def _persisted(self) -> dict[str, Any]:
        path = self.settings.base_path / "settings.json"
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(data.get("auth", {}), dict):
                raise ValueError("Invalid settings")
            return data
        except (OSError, ValueError) as exc:
            raise ManagementError(
                "unavailable", "Persisted settings unavailable"
            ) from exc

    def _effective_key(self):
        return (
            self.context.get_api_key()
            if self.context.get_api_key
            else self.settings.auth.api_key
        )

    def _configured(self, persisted):
        return bool(
            self._effective_key()
            or self.settings.auth.api_key
            or persisted.get("auth", {}).get("api_key")
        )

    def _loopback(self, peer_host: str | None) -> bool:
        bind = (
            self.context.get_bind_host()
            if self.context.get_bind_host
            else self.settings.server.host
        )
        return is_loopback_bind(bind) and is_loopback_bind_host(peer_host)

    def status(self, peer_host: str | None):
        persisted = self._persisted()
        if self._configured(persisted):
            return {
                "setup_required": False,
                "allowed": False,
                "reason": "already_configured",
            }
        if not self._loopback(peer_host):
            return {
                "setup_required": True,
                "allowed": False,
                "reason": "loopback_required",
            }
        if self.context.set_api_key is None:
            return {
                "setup_required": True,
                "allowed": False,
                "reason": "activation_unavailable",
            }
        return {"setup_required": True, "allowed": True, "reason": None}

    async def create(self, key: str, confirmation: str, peer_host: str | None, runtime):
        async with runtime.mutation_lock:
            # Recheck inside the lock so only one first-run request can succeed.
            persisted = self._persisted()
            if self._configured(persisted):
                raise ManagementError(
                    "conflict", "A main API key is already configured"
                )
            if not self._loopback(peer_host):
                raise ManagementError(
                    "forbidden",
                    "Initial setup requires a loopback bind and loopback client",
                )
            if self.context.set_api_key is None:
                raise ManagementError("unavailable", "Live key activation unavailable")
            if (
                not isinstance(key, str)
                or not isinstance(confirmation, str)
                or key != confirmation
            ):
                raise ManagementError(
                    "invalid_configuration", "Key and confirmation must match"
                )
            valid, message = validate_api_key(key)
            if not valid:
                raise ManagementError("invalid_configuration", message)
            if any(entry.key == key for entry in self.settings.auth.sub_keys):
                raise ManagementError(
                    "conflict", "Main key must differ from existing subkeys"
                )
            path = self.settings.base_path / "settings.json"
            existed = path.exists()
            old_effective = self._effective_key()
            old_settings_key = self.settings.auth.api_key
            candidate = copy.deepcopy(self.settings)
            candidate.auth.api_key = key
            data = copy.deepcopy(persisted)
            data.setdefault("auth", {})["api_key"] = key
            try:
                candidate.base_path.mkdir(parents=True, exist_ok=True)
                candidate._save_data(data)
            except Exception as exc:
                raise ManagementError(
                    "persistence_failed", "Initial key could not be saved"
                ) from exc
            self.settings.auth.api_key = key
            try:
                self.context.set_api_key(key)
            except Exception as exc:
                self.settings.auth.api_key = old_settings_key
                failures = []
                try:
                    # The initial runtime key is None. Restore that exact state.
                    cast(Any, self.context.set_api_key)(old_effective)
                except Exception as rollback:
                    failures.append(("key activation callback", rollback))
                try:
                    if existed:
                        candidate._save_data(persisted)
                    else:
                        path.unlink(missing_ok=True)
                except Exception as rollback:
                    failures.append(("persisted settings", rollback))
                if failures:
                    names = ", ".join(name for name, _ in failures)
                    raise ManagementError(
                        "rollback_failed",
                        f"Initial activation failed; rollback failed for {names}",
                    ) from failures[0][1]
                raise ManagementError(
                    "runtime_failed", "Initial key activation failed; settings restored"
                ) from exc
            return {"configured": True}
