"""Model library, storage and workspace operations without server globals."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
import shutil
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, fields
from datetime import UTC, datetime
from pathlib import Path

from omlx_config.model_settings import ModelSettings
from omlx_contracts.management import ModelSettingsPatch, ProfileWrite

from omlx_management.management import ManagementError, ManagementService
from omlx_management.model_control import (
    CONTROL_SCHEMA_VERSION,
    build_lineage,
    build_registry_record,
    derive_policy,
    directory_usage,
    discover_unmanaged_artifacts,
    plan_resources,
    verify_model_files,
)


def now():
    return datetime.now(UTC).isoformat()


def root_id(path: Path) -> str:
    return hashlib.sha256(str(path).encode()).hexdigest()[:16]


class WorkspaceService:
    def __init__(self, runtime):
        self.runtime = runtime
        self.context = runtime.context
        self.control = runtime.control
        self.service = ManagementService(self.context)
        self.pool = self.context.engine_pool
        self.manager = self.context.settings_manager

    def roots(self) -> list[Path]:
        settings = self.context.global_settings
        values = (
            settings.get_effective_model_dirs()
            if settings
            else getattr(self.pool, "model_directories", [])
        )
        return list(
            dict.fromkeys(Path(value).expanduser().resolve() for value in values)
        )

    def confined(self, value: str | Path, *, inspect_links=False) -> Path:
        path = Path(value).expanduser().resolve()
        roots = self.roots()
        if not any(path != root and path.is_relative_to(root) for root in roots):
            raise ManagementError(
                "invalid_configuration",
                "Model path must be below a configured model root",
            )
        if inspect_links and path.is_dir():
            # Use the same expanded ownership walk as acquisition reservations.
            # rglob alone does not enter directory links, so a second link below
            # an internal target could otherwise reach files outside storage.
            for item in self.runtime.normalized_paths([path]):
                if not any(item.resolve().is_relative_to(root) for root in roots):
                    raise ManagementError(
                        "invalid_configuration",
                        "Model contains a link outside configured roots",
                    )
            for index in path.rglob("*.safetensors.index.json"):
                try:
                    mapping = json.loads(index.read_text()).get("weight_map", {})
                except (ValueError, AttributeError):
                    continue
                if isinstance(mapping, dict):
                    for shard in mapping.values():
                        if (
                            not isinstance(shard, str)
                            or Path(shard).is_absolute()
                            or ".." in Path(shard).parts
                        ):
                            raise ManagementError(
                                "invalid_configuration",
                                "Weight index contains an unsafe shard path",
                            )
        return path

    async def rows(self):
        rows = self.service.inventory()["models"]
        for row in rows:
            row["is_default"] = row["id"] == self.context.get_default_model() or bool(
                row.get("settings", {}).get("is_default")
            )
            row["pinned"] = bool(
                row.get("pinned") or row.get("settings", {}).get("is_pinned")
            )
            row["exposed_profiles"] = (
                self.manager.list_profiles(row["id"]) if not row.get("virtual") else []
            )
        by_id = {row["id"]: row for row in rows}
        for profile in self.manager.list_exposed_profile_models():
            source_id = profile["source_model_id"]
            source = by_id.get(source_id)
            if source is None:
                continue
            api_name = profile.get("api_name") or self.manager.profile_api_name(profile)
            canonical_id = self.manager.profile_model_id(source_id, api_name)
            if canonical_id in by_id:
                continue
            virtual = {
                **source,
                "id": canonical_id,
                "display_name": profile.get("display_name") or canonical_id,
                "source_model_id": source_id,
                "virtual": True,
                "estimated_size": 0,
                "resident_estimated_size": 0,
                "actual_size": 0,
                "loaded": False,
                "is_loading": False,
                "pinned": False,
                "is_default": False,
                "settings": profile.get("settings", {}),
                "exposed_profiles": [],
            }
            rows.append(virtual)
            by_id[canonical_id] = virtual
        known = {
            str(Path(row["model_path"]).resolve())
            for row in rows
            if row.get("model_path")
            and not str(row["model_path"]).startswith("builtin://")
        }
        # Prefix unmanaged IDs with the root identity so duplicate relative names
        # in separate storage roots cannot address the wrong artifact.
        for root in self.roots():
            artifacts = await self.runtime.run_native(
                discover_unmanaged_artifacts, [root], known
            )
            for row in artifacts:
                try:
                    self.confined(row["model_path"])
                except ManagementError:
                    continue
                row["id"] = (
                    f"{row['id'].split(':', 1)[0]}:{root_id(root)}:{row['id'].split(':', 1)[1]}"
                )
                known.add(str(Path(row["model_path"]).resolve()))
                rows.append(row)
        return rows

    async def registry(self):
        rows = await self.rows()
        lineage = build_lineage(rows)
        records = [
            build_registry_record(row, self.control.store, **lineage[row["id"]])
            for row in rows
        ]
        for record, row in zip(records, rows):
            record["runtime"]["unloading"] = bool(row.get("is_unloading"))
            record["kind"] = (
                "incomplete" if row.get("preparation_blocker") else record["kind"]
            )
        return {
            "schema_version": CONTROL_SCHEMA_VERSION,
            "models": records,
            "summary": {
                "total": len(records),
                "physical": sum(r["kind"] == "physical" for r in records),
                "exposed": sum(r["kind"] == "virtual" for r in records),
                "incomplete": sum(r["kind"] == "incomplete" for r in records),
            },
        }

    async def model(self, model_id):
        row = next((r for r in await self.rows() if r["id"] == model_id), None)
        if row is None:
            raise ManagementError("not_found", f"Model not found: {model_id}")
        return row

    async def storage(self):
        roots = []
        for root in self.roots():
            ancestor = root
            while not ancestor.exists() and ancestor != ancestor.parent:
                ancestor = ancestor.parent
            usage = await self.runtime.run_native(directory_usage, root)
            disk = await self.runtime.run_native(shutil.disk_usage, ancestor)
            roots.append(
                {
                    "id": root_id(root),
                    "path": str(root),
                    "exists": root.exists(),
                    **usage,
                    "disk_total_bytes": disk.total,
                    "disk_used_bytes": disk.used,
                    "disk_free_bytes": disk.free,
                }
            )
        records = (await self.registry())["models"]
        return {
            "roots": roots,
            "models": [
                {
                    "id": r["id"],
                    "path": r["path"],
                    **r["storage"],
                    "revision": r["source"]["revision"],
                    "cached_revisions": r["source"]["cached_revisions"],
                }
                for r in records
                if r["kind"] != "virtual"
            ],
        }

    def plan(self, model_ids):
        status = copy.deepcopy(self.pool.get_status())
        # The current default and draining engines must not be eviction victims.
        for row in status["models"]:
            entry = self.pool.get_model_view(row["id"])
            if row["id"] == self.context.get_default_model() or (
                entry and getattr(entry, "pending_unload_reason", None)
            ):
                row["pinned"] = True
        try:
            return plan_resources(
                model_ids,
                status,
                in_use={
                    r["id"]: getattr(self.pool.get_model_view(r["id"]), "in_use", 0)
                    for r in status["models"]
                },
            )
        except ValueError as exc:
            raise ManagementError("invalid_configuration", str(exc)) from exc

    @asynccontextmanager
    async def mutation(self):
        if (
            self.runtime.mutation_lock.locked()
            or self.runtime.operation_lock.locked()
            or getattr(self.pool, "preparation_active", False)
        ):
            raise ManagementError("busy", "Another management operation is active")
        async with (
            self.runtime.operation_lock,
            self.runtime.mutation_lock,
            self.pool.exclusive_management(),
        ):
            yield

    @asynccontextmanager
    async def file_reservation(self, *paths):
        owner = f"workspace:{uuid.uuid4().hex}"
        self.runtime.reserve_paths(
            [path for path in paths if path and not str(path).startswith("builtin://")],
            owner,
        )
        try:
            yield
        finally:
            self.runtime.release_paths(owner)

    def busy(self, model_id):
        return self.pool.is_model_busy(model_id)

    async def blockers(self, row, *, allow_loaded=False):
        model_id = row["id"]
        records = (await self.registry())["models"]
        record = next(r for r in records if r["id"] == model_id)
        reasons = []
        if row.get("virtual") or str(row.get("model_path", "")).startswith(
            "builtin://"
        ):
            reasons.append("Model has no independently managed files")
        if self.busy(model_id):
            reasons.append("Model has active requests or a lifecycle transition")
        if record["runtime"]["loaded"] and not allow_loaded:
            reasons.append("Model is loaded")
        if record["runtime"]["pinned"]:
            reasons.append("Model is pinned")
        if record["runtime"]["default"]:
            reasons.append("Model is the default")
        # Draft dependency points from the consumer to the draft; profiles and
        # variants point from the physical base to their exposed child.
        dependents = [
            p
            for p in record["lineage"]["parents"]
            if p["relation"] not in {"profile", "variant"}
        ]
        dependents += [
            c
            for c in record["lineage"]["children"]
            if c["relation"] in {"profile", "variant"}
        ]
        dependents += [
            {"id": p.get("model_id") or p.get("name"), "relation": "profile"}
            for p in record["lineage"]["profiles"]
            if p.get("expose_as_model")
        ]
        for other in records:
            if (
                other["id"] != model_id
                and other["path"]
                and record["path"]
                and (
                    Path(other["path"]).resolve() == Path(record["path"]).resolve()
                    or (
                        Path(record["path"]).parent.name == "snapshots"
                        and Path(other["path"])
                        .resolve()
                        .is_relative_to(Path(record["path"]).resolve().parent.parent)
                    )
                )
            ):
                dependents.append({"id": other["id"], "relation": "shared-files"})
        for collection in self.control.store.section("collections").values():
            if model_id in collection.get("model_ids", []):
                dependents.append({"id": collection["id"], "relation": "collection"})
        if dependents:
            reasons.append("Registered assets depend on this model")
        return reasons, dependents

    async def guard(self, row, *, drain=False):
        path = self.confined(row["model_path"], inspect_links=True)
        scope = path.parent.parent if path.parent.name == "snapshots" else path
        self.runtime.assert_paths_idle([scope])
        reasons, _ = await self.blockers(row, allow_loaded=drain)
        if reasons:
            raise ManagementError("busy", "; ".join(reasons))
        entry = self.pool.get_model_view(row["id"])
        if entry and entry.loaded:
            if not drain:
                raise ManagementError("busy", "Model is loaded")
            unloaded = await self.pool.request_unload(
                row["id"], reason="workspace file mutation", abort_active=False
            )
            if not unloaded or self.pool.get_model_view(row["id"]).loaded:
                raise ManagementError(
                    "busy", "Model did not finish unloading; retry after it drains"
                )

    def collections(self):
        return {"collections": list(self.control.store.section("collections").values())}

    async def save_collection(self, collection_id, body):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", collection_id):
            raise ManagementError("invalid_configuration", "Invalid collection ID")
        if not body["name"].strip():
            raise ManagementError(
                "invalid_configuration", "Collection name is required"
            )
        ids = list(dict.fromkeys(body["model_ids"]))
        if not ids:
            raise ManagementError("invalid_configuration", "Select at least one model")
        self.plan(ids)
        record = {"id": collection_id, **body, "model_ids": ids, "updated_at": now()}
        async with self.mutation():
            self.plan(ids)
            if not body.get("preload"):
                self.write_store("collections", collection_id, record)
            else:
                # Preload is persisted through the existing pinned-model startup
                # mechanism. Saving does not start inference or loading.
                old_settings = self.manager.snapshot_settings()
                old_control = self.control.store.export_state()
                try:
                    for model_id in ids:
                        if self.busy(model_id):
                            raise ManagementError("busy", "Selected model is busy")
                        settings = copy.deepcopy(self.manager.get_settings(model_id))
                        settings.is_pinned = True
                        self.manager.set_settings(model_id, settings)
                    self.write_store("collections", collection_id, record)
                    self.pool.apply_settings_overrides(self.manager)
                except BaseException:
                    self.manager.restore_settings(old_settings, persist=True)
                    self.control.store._data = old_control
                    self.control.store._save_locked()
                    self.pool.apply_settings_overrides(self.manager)
                    raise
        return {"collection": record}

    def write_store(self, section, key, value=None, *, remove=False):
        store = self.control.store
        with store._lock:
            snapshot = copy.deepcopy(store._data)
            try:
                if remove:
                    return store.remove(section, key)
                store.put(section, key, value)
            except BaseException:
                store._data = snapshot
                raise

    async def delete_collection(self, collection_id):
        async with self.mutation():
            if not self.write_store("collections", collection_id, remove=True):
                raise ManagementError("not_found", "Collection not found")
        return {"success": True}

    async def load_collection(self, collection_id):
        async with self.mutation():
            collection = self.control.store.get("collections", collection_id)
            if collection is None:
                raise ManagementError("not_found", "Collection not found")
            plan = self.plan(collection["model_ids"])
            if not plan["fits"]:
                raise ManagementError(
                    "busy", "Collection exceeds the available memory ceiling"
                )
            for model_id in collection["model_ids"]:
                if self.busy(model_id):
                    raise ManagementError("busy", f"Model is busy: {model_id}")
            async with self.file_reservation(
                *(
                    self.pool.get_model_view(mid).model_path
                    for mid in collection["model_ids"]
                )
            ):
                loaded = []
                try:
                    for victim in plan["evictions"]:
                        if not await self.pool.request_unload(
                            victim["id"],
                            reason="collection admission",
                            abort_active=False,
                        ):
                            raise ManagementError(
                                "busy", "An eviction did not finish draining"
                            )
                    for model_id in collection["model_ids"]:
                        entry = self.pool.get_model_view(model_id)
                        was_loaded = entry.loaded
                        await drain_async(self.pool.load_model(model_id))
                        if not was_loaded:
                            loaded.append(model_id)
                except BaseException:
                    for model_id in reversed(loaded):
                        await self.pool.request_unload(
                            model_id,
                            reason="collection load rollback",
                            abort_active=False,
                        )
                    raise
                return {
                    "collection_id": collection_id,
                    "loaded": collection["model_ids"],
                    "plan": plan,
                }

    async def export(self):
        rows = await self.rows()
        models = []
        for row in rows:
            if row.get("virtual") or self.pool.get_model_view(row["id"]) is None:
                continue
            settings = self.manager.get_settings(row["id"]).to_dict()
            settings.pop("trust_remote_code", None)
            models.append(
                {
                    "id": row["id"],
                    "settings": settings,
                    "profiles": [
                        {
                            **profile,
                            "settings": {
                                key: value
                                for key, value in profile.get("settings", {}).items()
                                if key != "trust_remote_code"
                            },
                        }
                        for profile in self.manager.list_profiles(row["id"])
                    ],
                    "policy": self.control.store.get("policies", row["id"]),
                }
            )
        return {
            "schema_version": CONTROL_SCHEMA_VERSION,
            "exported_at": now(),
            "models": models,
            "collections": self.collections()["collections"],
        }

    def validate_bundle(self, bundle):
        if bundle.get("schema_version") != CONTROL_SCHEMA_VERSION:
            raise ManagementError(
                "invalid_configuration", "Unsupported workspace bundle version"
            )
        models, collections = bundle.get("models", []), bundle.get("collections", [])
        if not isinstance(models, list) or not isinstance(collections, list):
            raise ManagementError(
                "invalid_configuration", "Models and collections must be lists"
            )
        known = set(self.pool.get_model_ids())
        seen = set()
        allowed = {field.name for field in fields(ModelSettings)} - {
            "trust_remote_code"
        }
        normalized = []
        try:
            for item in models:
                if (
                    not isinstance(item, dict)
                    or item.get("id") not in known
                    or item["id"] in seen
                ):
                    raise ValueError("Unknown or duplicate model ID in bundle")
                model_id = item["id"]
                seen.add(model_id)
                entry = self.pool.get_model_view(model_id)
                if getattr(entry, "virtual", False):
                    raise ValueError("Import settings on the physical model")
                settings = item.get("settings", {})
                if not isinstance(settings, dict) or set(settings) - allowed:
                    raise ValueError("Unknown or protected model setting")
                ModelSettingsPatch.model_validate(
                    {
                        k: v
                        for k, v in settings.items()
                        if k in ModelSettingsPatch.model_fields
                    }
                )
                candidate = ModelSettings.from_dict(settings)
                candidate.trust_remote_code = self.manager.get_settings(
                    model_id
                ).trust_remote_code
                self.service._validate_model_settings(entry, candidate)
                profiles = item.get("profiles", [])
                if not isinstance(profiles, list):
                    raise ValueError("Profiles must be a list")
                profile_names = set()
                for profile in profiles:
                    if (
                        not isinstance(profile, dict)
                        or not profile.get("name")
                        or profile["name"] in profile_names
                    ):
                        raise ValueError("Invalid or duplicate profile")
                    profile_names.add(profile["name"])
                    if "trust_remote_code" in profile.get("settings", {}):
                        raise ValueError(
                            "Profile import cannot change executable-code trust"
                        )
                    patch = ModelSettingsPatch.model_validate(
                        profile.get("settings", {})
                    )
                    self.service._profile_settings(model_id, patch)
                    ProfileWrite.model_validate(
                        {
                            k: profile[k]
                            for k in ProfileWrite.model_fields
                            if k in profile
                        }
                    )
                policy = item.get("policy")
                if policy is not None and (
                    not isinstance(policy, dict)
                    or policy.get("mode")
                    not in {
                        "on_demand",
                        "always_resident",
                        "keep_warm",
                        "unload_after_request",
                    }
                ):
                    raise ValueError("Invalid policy")
                normalized.append({**item, "candidate": candidate})
            collection_ids = set()
            for collection in collections:
                if (
                    not isinstance(collection, dict)
                    or not re.fullmatch(
                        r"[A-Za-z0-9_-]{1,80}", str(collection.get("id", ""))
                    )
                    or collection["id"] in collection_ids
                ):
                    raise ValueError("Invalid or duplicate collection ID")
                collection_ids.add(collection["id"])
                if (
                    not isinstance(collection.get("name"), str)
                    or not collection["name"].strip()
                    or not isinstance(collection.get("model_ids"), list)
                    or not collection["model_ids"]
                    or any(mid not in known for mid in collection["model_ids"])
                ):
                    raise ValueError("Invalid collection or unknown collection model")
            if sum(item["candidate"].is_default for item in normalized) > 1:
                raise ValueError("Bundle selects multiple default models")
        except (ValueError, TypeError, KeyError) as exc:
            raise ManagementError("invalid_configuration", str(exc)) from exc
        return normalized, collections

    def import_plan(self, models, collections, *, check_global=True):
        changes = []
        changed_models = []
        settings_count = profiles_count = 0
        for item in models:
            model_id = item["id"]
            before = asdict(self.manager.get_settings(model_id))
            settings_changed = False
            for field, after in asdict(item["candidate"]).items():
                if field != "trust_remote_code" and before.get(field) != after:
                    settings_changed = True
                    changes.append(
                        {
                            "model_id": model_id,
                            "field": f"settings.{field}",
                            "before": before.get(field),
                            "after": after,
                        }
                    )
            policy_changed = False
            if item.get("policy") is not None:
                old_policy = derive_policy(
                    {"settings": before}, self.control.store.get("policies", model_id)
                )
                policy_changed = any(
                    old_policy.get(key) != item["policy"].get(key)
                    for key in ("mode", "ttl_seconds")
                )
                if policy_changed:
                    changes.append(
                        {
                            "model_id": model_id,
                            "field": "policy",
                            "before": old_policy,
                            "after": item["policy"],
                        }
                    )
            changed_profiles = []
            for profile in item.get("profiles", []):
                old_profile = self.manager.get_profile(model_id, profile["name"])
                candidate = (
                    copy.deepcopy(old_profile)
                    if old_profile
                    else {"name": profile["name"]}
                )
                candidate.update(
                    {
                        "display_name": profile.get("display_name") or profile["name"],
                        "settings": {
                            key: value
                            for key, value in profile.get("settings", {}).items()
                            if key != "trust_remote_code"
                        },
                        "expose_as_model": profile.get("expose_as_model", False),
                    }
                )
                if profile.get("description") is not None or not old_profile:
                    candidate["description"] = profile.get("description")
                if profile.get("api_name") is not None:
                    candidate["api_name"] = profile["api_name"]
                comparable_old = copy.deepcopy(old_profile)
                if comparable_old:
                    comparable_old["settings"] = {
                        key: value
                        for key, value in comparable_old.get("settings", {}).items()
                        if key != "trust_remote_code"
                    }
                if candidate != comparable_old:
                    changed_profiles.append(profile)
                    changes.append(
                        {
                            "model_id": model_id,
                            "field": f"profiles.{profile['name']}",
                            "before": comparable_old,
                            "after": candidate,
                        }
                    )
            if settings_changed or policy_changed or changed_profiles:
                changed_models.append(
                    {
                        **item,
                        "settings_changed": settings_changed,
                        "policy_changed": policy_changed,
                        "profiles": changed_profiles,
                    }
                )
                settings_count += settings_changed
                profiles_count += len(changed_profiles)
        # Selecting a new default also clears the previous persisted default.
        # Include that setter side effect in preview admission and field diffs.
        previous_default = self.manager.get_default_model_id()
        selects_new_default = any(
            item["candidate"].is_default and item["id"] != previous_default
            for item in changed_models
        )
        if (
            selects_new_default
            and previous_default
            and self.pool.get_model_view(previous_default) is not None
            and previous_default not in {item["id"] for item in models}
        ):
            previous_settings = self.manager.get_settings(previous_default)
            candidate = copy.deepcopy(previous_settings)
            candidate.is_default = False
            changed_models.append(
                {
                    "id": previous_default,
                    "candidate": candidate,
                    "settings_changed": True,
                    "policy_changed": False,
                    "profiles": [],
                }
            )
            changes.append(
                {
                    "model_id": previous_default,
                    "field": "settings.is_default",
                    "before": True,
                    "after": False,
                }
            )
            settings_count += 1
        changed_collections = []
        for collection in collections:
            previous = self.control.store.get("collections", collection["id"])
            if previous != collection:
                changed_collections.append(collection)
                changes.append(
                    {
                        "model_id": None,
                        "field": f"collections.{collection['id']}",
                        "before": previous,
                        "after": collection,
                    }
                )
        blockers = []
        for item in changed_models:
            model_id = item["id"]
            entry = self.pool.get_model_view(model_id)
            if self.busy(model_id):
                blockers.append(
                    {
                        "model_id": model_id,
                        "code": "busy",
                        "reason": f"Model {model_id} has active requests or a lifecycle transition",
                    }
                )
            if entry.loaded:
                blockers.append(
                    {
                        "model_id": model_id,
                        "code": "loaded",
                        "reason": f"Unload model {model_id} before applying its imported changes",
                    }
                )
            try:
                self.runtime.assert_paths_idle([entry.model_path])
            except ManagementError as exc:
                blockers.append(
                    {"model_id": model_id, "code": "files_in_use", "reason": exc.detail}
                )
        if (
            check_global
            and changes
            and (
                self.runtime.operation_lock.locked()
                or self.runtime.mutation_lock.locked()
                or getattr(self.pool, "preparation_active", False)
            )
        ):
            blockers.append(
                {
                    "model_id": None,
                    "code": "busy",
                    "reason": "Another management operation is active",
                }
            )
        plan = {
            "matched_models": [item["id"] for item in models],
            "missing_models": [],
            "settings_updates": settings_count,
            "profile_updates": profiles_count,
            "collection_updates": len(changed_collections),
            "changes": changes,
            "affected_model_ids": [item["id"] for item in changed_models],
            "blockers": blockers,
            "can_apply": not blockers,
        }
        return plan, changed_models, changed_collections

    async def import_bundle(self, bundle, *, dry_run=True):
        models, collections = self.validate_bundle(bundle)
        plan, models, collections = self.import_plan(models, collections)
        if dry_run or not plan["changes"]:
            return {"dry_run": dry_run, "plan": plan}
        if plan["blockers"]:
            raise ManagementError(
                "busy", "; ".join(blocker["reason"] for blocker in plan["blockers"])
            )
        async with self.mutation():
            # Repeat preview validation under management admission. A previously
            # clean preview cannot authorize newly loaded or changed models.
            all_models, all_collections = self.validate_bundle(bundle)
            plan, models, collections = self.import_plan(
                all_models, all_collections, check_global=False
            )
            if plan["blockers"]:
                raise ManagementError(
                    "busy", "; ".join(blocker["reason"] for blocker in plan["blockers"])
                )
            async with self.file_reservation(
                *(self.pool.get_model_view(item["id"]).model_path for item in models)
            ):
                manager, store = self.manager, self.control.store
                with store._lock:
                    old_settings, old_profiles = (
                        manager.snapshot_settings(),
                        manager.snapshot_profiles(),
                    )
                    old_control = copy.deepcopy(store._data)
                    default = self.context.get_default_model()
                    try:
                        for item in models:
                            if item["settings_changed"]:
                                manager.set_settings(item["id"], item["candidate"])
                            if item["policy_changed"]:
                                store.put("policies", item["id"], item["policy"])
                            for profile in item.get("profiles", []):
                                kwargs = {
                                    "display_name": profile.get("display_name")
                                    or profile["name"],
                                    "description": profile.get("description"),
                                    "settings": profile.get("settings", {}),
                                    "expose_as_model": profile.get(
                                        "expose_as_model", False
                                    ),
                                    "api_name": profile.get("api_name"),
                                    "reserved_model_ids": set(
                                        self.pool.get_model_ids()
                                    ),
                                }
                                existing_profile = manager.get_profile(
                                    item["id"], profile["name"]
                                )
                                if existing_profile:
                                    if "trust_remote_code" in existing_profile.get(
                                        "settings", {}
                                    ):
                                        kwargs["settings"] = {
                                            **kwargs["settings"],
                                            "trust_remote_code": existing_profile[
                                                "settings"
                                            ]["trust_remote_code"],
                                        }
                                    manager.update_profile(
                                        item["id"], profile["name"], **kwargs
                                    )
                                else:
                                    manager.save_profile(
                                        item["id"], profile["name"], **kwargs
                                    )
                        for collection in collections:
                            store.put("collections", collection["id"], collection)
                        self.pool.apply_settings_overrides(manager)
                        self.context.set_default_model(manager.get_default_model_id())
                        self.context.apply_sampling()
                    except BaseException:
                        manager.restore_settings(old_settings, persist=True)
                        manager.restore_profiles(old_profiles, persist=True)
                        store._data = old_control
                        store._save_locked()
                        self.pool.apply_settings_overrides(manager)
                        self.context.set_default_model(default)
                        raise
        return {"dry_run": False, "plan": plan}

    async def delete_plan(self, model_id):
        row = await self.model(model_id)
        path = self.confined(row["model_path"], inspect_links=True)
        if path.parent.name == "snapshots":
            path = self.confined(path.parent.parent, inspect_links=True)
        reasons, dependents = await self.blockers(row)
        usage = await self.runtime.run_native(directory_usage, path)
        token = hashlib.sha256(
            json.dumps(
                {
                    "id": model_id,
                    "path": str(path),
                    "stat": path.stat().st_mtime_ns,
                    "usage": usage,
                    "manifest": await self.runtime.run_native(file_manifest, path),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        return {
            "model_id": model_id,
            "path": str(path),
            **usage,
            "plan_token": token,
            "loaded": bool(row.get("loaded")),
            "pinned": bool(row.get("pinned")),
            "dependents": dependents,
            "protected_reasons": reasons,
            "safe": not reasons,
            "can_drain": bool(row.get("loaded")) and reasons == ["Model is loaded"],
        }

    async def refresh_files(self):
        """Refresh while this service owns the pool preparation admission.

        Discovery runs while exclusive_management closes external admission,
        after every existing lease has drained.
        """
        self.manager.reload()
        self.pool.discover_models(
            [str(root) for root in self.roots()], self.manager.get_pinned_model_ids()
        )
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

    async def delete(self, model_id, *, plan_token, drain=False):
        async with self.mutation():
            plan = await self.delete_plan(model_id)
            if plan_token != plan["plan_token"]:
                raise ManagementError(
                    "busy", "Deletion plan changed; review a fresh plan"
                )
            row = await self.model(model_id)
            await self.guard(row, drain=drain)
            path = Path(plan["path"])
            # Rename first so a refresh or persistence failure can restore files.
            async with self.file_reservation(path):
                tombstone = path.with_name(f".workspace-delete-{uuid.uuid4().hex}")
                old_settings = self.manager.snapshot_settings()
                old_profiles = self.manager.snapshot_profiles()
                old_control = self.control.store.export_state()
                try:
                    await self.runtime.run_native(path.rename, tombstone)
                    self.manager.delete_settings(model_id)
                    for section in ("policies", "health", "updates"):
                        self.write_store(section, model_id, remove=True)
                    await self.refresh_files()
                except BaseException:
                    if tombstone.exists():
                        await self.runtime.run_native(tombstone.rename, path)
                    self.manager.restore_settings(old_settings, persist=True)
                    self.manager.restore_profiles(old_profiles, persist=True)
                    self.control.store._data = old_control
                    self.control.store._save_locked()
                    await self.refresh_files()
                    raise
                await self.runtime.run_native(shutil.rmtree, tombstone)
                self.control.invalidate_storage_cache()
                return {
                    "model_id": model_id,
                    "deleted": True,
                    "removed_logical_bytes": plan["logical_bytes"],
                }

    async def move_guard(self, rows, source, *, drain):
        affected = [
            row
            for row in rows
            if not row.get("virtual")
            and row.get("model_path")
            and not str(row["model_path"]).startswith("builtin://")
            and Path(row["model_path"]).resolve().is_relative_to(source)
        ]
        for row in affected:
            row["_move_serving"] = self.pool.get_model_view(row["id"]) is not None
        affected_ids = {row["id"] for row in affected}
        for other in rows:
            if not other.get("model_path") or str(other["model_path"]).startswith(
                "builtin://"
            ):
                continue
            other_path = Path(other["model_path"]).resolve()
            settings_records = [other.get("settings", {})] + [
                profile.get("settings", {})
                for profile in other.get("exposed_profiles", [])
            ]
            for settings in settings_records:
                for key in (
                    "specprefill_draft_model",
                    "dflash_draft_model",
                    "vlm_mtp_draft_model",
                ):
                    reference = settings.get(key)
                    if (
                        reference
                        and reference not in affected_ids
                        and Path(reference)
                        .expanduser()
                        .resolve()
                        .is_relative_to(source)
                    ):
                        raise ManagementError(
                            "busy",
                            f"Model {other['id']} depends on an explicit path inside the moved container",
                        )
            if other.get("virtual"):
                continue
            if not other_path.is_relative_to(source) and any(
                path.resolve().is_relative_to(source)
                for path in self.runtime.normalized_paths([other_path])
            ):
                raise ManagementError(
                    "busy",
                    f"Model {other['id']} links to files inside the moved container",
                )
        for row in affected:
            self.confined(row["model_path"], inspect_links=True)
            if self.busy(row["id"]):
                raise ManagementError(
                    "busy",
                    f"Model {row['id']} has active requests or a lifecycle transition",
                )
            if (
                row.get("loaded")
                or row.get("pinned")
                or row["id"] == self.context.get_default_model()
                or row.get("settings", {}).get("is_default")
            ) and not drain:
                raise ManagementError(
                    "busy",
                    f"Model {row['id']} is loaded, pinned, or default; explicitly drain it before moving",
                )
        for row in affected:
            entry = self.pool.get_model_view(row["id"])
            if entry and entry.loaded:
                unloaded = await drain_async(
                    self.pool.request_unload(
                        row["id"], reason="workspace storage move", abort_active=False
                    )
                )
                if not unloaded or self.pool.get_model_view(row["id"]).loaded:
                    raise ManagementError(
                        "busy", f"Model {row['id']} did not finish unloading"
                    )
        return affected

    async def verify_move_destination(
        self, affected, source, destination, destination_root
    ):
        discovered_rows = None
        moved_ids = []
        for row in affected:
            path = destination / Path(row["model_path"]).resolve().relative_to(source)
            self.confined(path, inspect_links=True)
            entry = self.pool.get_model_view(row["id"])
            if entry is not None:
                if Path(entry.model_path).resolve() != path.resolve():
                    raise ManagementError(
                        "invalid_configuration",
                        f"Destination discovery did not preserve model ID {row['id']}",
                    )
                moved_ids.append(row["id"])
            else:
                if row["_move_serving"]:
                    raise ManagementError(
                        "invalid_configuration",
                        f"Destination discovery did not preserve model ID {row['id']}",
                    )
                if discovered_rows is None:
                    discovered_rows = await self.rows()
                expected_id = f"{row['id'].split(':', 1)[0]}:{root_id(destination_root)}:{path.relative_to(destination_root).as_posix()}"
                if not any(
                    candidate["id"] == expected_id
                    and Path(candidate["model_path"]).resolve() == path.resolve()
                    for candidate in discovered_rows
                ):
                    raise ManagementError(
                        "invalid_configuration",
                        "Destination artifact was not rediscovered",
                    )
                moved_ids.append(expected_id)
        return moved_ids

    async def move(self, model_id, *, destination_root_id, drain=False):
        # Reject current requests before management admission could drain them.
        # Recheck inside the gate as well, where no new inference can start.
        initial = await self.model(model_id)
        if not initial.get("virtual") and not str(
            initial.get("model_path", "")
        ).startswith("builtin://"):
            initial_path = self.confined(initial["model_path"], inspect_links=True)
            initial_scope = (
                initial_path.parent.parent
                if initial_path.parent.name == "snapshots"
                else initial_path
            )
            for candidate in self.pool.get_status()["models"]:
                if (
                    candidate.get("model_path")
                    and not str(candidate["model_path"]).startswith("builtin://")
                    and Path(candidate["model_path"])
                    .resolve()
                    .is_relative_to(initial_scope)
                    and self.busy(candidate["id"])
                ):
                    raise ManagementError(
                        "busy",
                        f"Model {candidate['id']} has active requests or a lifecycle transition",
                    )
        async with self.mutation():
            row = await self.model(model_id)
            if row.get("virtual") or str(row.get("model_path", "")).startswith(
                "builtin://"
            ):
                raise ManagementError(
                    "invalid_configuration", "Model has no independently managed files"
                )
            checkpoint = self.confined(row["model_path"], inspect_links=True)
            source = self.confined(
                checkpoint.parent.parent
                if checkpoint.parent.name == "snapshots"
                else checkpoint,
                inspect_links=True,
            )
            self.runtime.assert_paths_idle([source])
            root = next(
                (
                    candidate
                    for candidate in self.roots()
                    if root_id(candidate) == destination_root_id
                ),
                None,
            )
            if root is None:
                raise ManagementError(
                    "invalid_configuration", "Select a configured destination root"
                )
            source_root = next(
                candidate
                for candidate in self.roots()
                if source.is_relative_to(candidate)
            )
            destination = root / source.relative_to(source_root)
            self.confined(destination)
            if destination.is_relative_to(source) or source.is_relative_to(destination):
                raise ManagementError(
                    "invalid_configuration",
                    "Source and destination containers cannot overlap",
                )
            if destination.exists() or destination.is_symlink():
                raise ManagementError("busy", "Destination already exists")
            rows = await self.rows()
            staging = destination.parent / f".workspace-move-{uuid.uuid4().hex}"
            tombstone = source.parent / f".workspace-original-{uuid.uuid4().hex}"
            async with self.file_reservation(source, destination, staging, tombstone):
                affected = await self.move_guard(rows, source, drain=drain)
                destination.parent.mkdir(parents=True, exist_ok=True)
                cleanup_started = False
                try:
                    await self.runtime.run_native(
                        copy_model_container, source, staging, destination
                    )
                    await self.runtime.run_native(staging.rename, destination)
                    self.confined(destination, inspect_links=True)
                    # Hide the original reversibly, so first-root duplicate-ID
                    # precedence cannot conceal failed destination discovery.
                    await self.runtime.run_native(source.rename, tombstone)
                    await self.refresh_files()
                    moved_ids = await self.verify_move_destination(
                        affected, source, destination, root
                    )
                    # Root-scoped unmanaged IDs change with their storage root.
                    # Migrate all their control metadata in one atomic write.
                    with self.control.store._lock:
                        current_control = copy.deepcopy(self.control.store._data)
                        try:
                            migrated = False
                            for asset, new_id in zip(affected, moved_ids):
                                if asset["id"] == new_id:
                                    continue
                                for section in ("policies", "health", "updates"):
                                    values = self.control.store._data[section]
                                    if asset["id"] in values:
                                        values[new_id] = values.pop(asset["id"])
                                        migrated = True
                            if migrated:
                                self.control.store._save_locked()
                        except BaseException:
                            self.control.store._data = current_control
                            raise
                    cleanup_started = True
                    await self.runtime.run_native(shutil.rmtree, tombstone)
                except BaseException:
                    if cleanup_started:
                        raise
                    # Once original cleanup starts, a partial deletion cannot be
                    # rolled back. Keep the already verified destination intact.
                    if tombstone.exists():
                        await self.runtime.run_native(tombstone.rename, source)
                    if source.exists():
                        for temporary in (staging, destination):
                            if temporary.exists():
                                await self.runtime.run_native(shutil.rmtree, temporary)
                        await self.refresh_files()
                    raise
                self.control.invalidate_storage_cache()
                return {
                    "model_id": model_id,
                    "path": str(destination / checkpoint.relative_to(source)),
                    "container_path": str(destination),
                    "moved_model_ids": moved_ids,
                    "moved": True,
                }

    async def activate_revision(self, model_id, revision, *, drain=False):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", revision):
            raise ManagementError("invalid_configuration", "Invalid revision")
        async with self.mutation():
            row = await self.model(model_id)
            path = self.confined(row["model_path"], inspect_links=True)
            if path.parent.name != "snapshots":
                raise ManagementError(
                    "invalid_configuration", "Model is not backed by a snapshot cache"
                )
            candidate = self.confined(path.parent / revision, inspect_links=True)
            report = await self.runtime.run_native(
                verify_model_files, {**row, "model_path": str(candidate)}
            )
            if report["status"] != "ready":
                raise ManagementError(
                    "invalid_configuration",
                    "Cached revision is missing or failed structural verification",
                )
            await self.guard(row, drain=drain)
            ref = self.confined(path.parent.parent / "refs" / "main")
            async with self.file_reservation(path.parent.parent):
                previous = ref.read_bytes() if ref.exists() else None
                ref.parent.mkdir(parents=True, exist_ok=True)
                temp = ref.with_name(f".main-{uuid.uuid4().hex}")
                try:
                    temp.write_text(revision, encoding="utf-8")
                    temp.replace(ref)
                    await self.refresh_files()
                    update = self.control.store.get("updates", model_id, {})
                    self.write_store(
                        "updates",
                        model_id,
                        {
                            **update,
                            "local_revision": revision,
                            "activated_at": now(),
                            "status": "current"
                            if update.get("remote_revision") == revision
                            else "rolled_back",
                        },
                    )
                except BaseException:
                    if previous is None:
                        ref.unlink(missing_ok=True)
                    else:
                        ref.write_bytes(previous)
                    await self.refresh_files()
                    raise
                finally:
                    temp.unlink(missing_ok=True)
                return {"model_id": model_id, "revision": revision, "activated": True}

    async def verify(self, model_id, *, mode="structural"):
        row = await self.model(model_id)
        if not row.get("virtual") and not str(row.get("model_path", "")).startswith(
            "builtin://"
        ):
            self.confined(row["model_path"], inspect_links=True)
        if mode == "smoke" and row.get("model_type", "llm") not in {
            "llm",
            "vlm",
            "embedding",
            "reranker",
            "audio_tts",
            "image_generation",
        }:
            raise ManagementError(
                "unavailable",
                "No inference smoke probe is available for this model type",
            )

        async def run(operation_id):
            async with self.mutation(), self.file_reservation(row.get("model_path")):
                current = await self.model(model_id)
                if current.get("model_path") != row.get("model_path"):
                    raise ManagementError(
                        "busy", "Model path changed before verification started"
                    )
                self.control.update_operation(
                    operation_id, stage="checking files", progress=20
                )
                report = await self.runtime.run_native(verify_model_files, current)
                report["mode"] = mode
                report["smoke_verified"] = False
                if mode == "smoke" and report["status"] == "ready":
                    if self.busy(model_id):
                        raise ManagementError("busy", "Model is busy")
                    entry = self.pool.get_model_view(model_id)
                    if entry is None:
                        raise ManagementError(
                            "unavailable", "Artifact has no serving engine"
                        )
                    was_loaded = entry.loaded
                    try:
                        self.control.update_operation(
                            operation_id, stage="smoke inference", progress=65
                        )
                        report["probe"] = await drain_async(
                            self.pool.smoke_model(
                                model_id, current.get("model_type", "llm")
                            )
                        )
                        report["smoke_verified"] = True
                    except Exception as exc:
                        report["status"] = "smoke_failed"
                        report["summary"] = str(exc)
                        self.write_store("health", model_id, report)
                        raise
                    finally:
                        if (
                            not was_loaded
                            and self.pool.get_model_view(model_id).loaded
                            and not await self.pool.request_unload(
                                model_id,
                                reason="workspace smoke complete",
                                abort_active=False,
                            )
                        ):
                            raise ManagementError(
                                "busy", "Probe engine did not finish draining"
                            )
                self.write_store("health", model_id, report)
                return report

        operation = self.control.start_operation(
            "verify", run, model_id=model_id, payload={"mode": mode}, cancellable=False
        )
        operation = self.control.update_operation(operation["id"], retryable=False)
        return {"operation": operation}

    async def check_update(self, model_id):
        row = await self.model(model_id)
        if not row.get("source_repo_id"):
            raise ManagementError(
                "invalid_configuration", "Model has no Hugging Face source repository"
            )

        async def run(operation_id):
            from huggingface_hub import HfApi

            info = await self.runtime.run_native(
                HfApi().model_info, row["source_repo_id"]
            )
            local = (
                Path(row["model_path"]).name
                if Path(row["model_path"]).parent.name == "snapshots"
                else None
            )
            remote = getattr(info, "sha", None)
            report = {
                "repo_id": row["source_repo_id"],
                "local_revision": local,
                "remote_revision": remote,
                "status": "unknown"
                if not local or not remote
                else "current"
                if local == remote
                else "update_available",
                "checked_at": now(),
            }
            self.write_store("updates", model_id, report)
            return report

        operation = self.control.start_operation(
            "update_check", run, model_id=model_id, cancellable=False
        )
        return {
            "operation": self.control.update_operation(operation["id"], retryable=False)
        }

    async def stage_update(self, model_id):
        row = await self.model(model_id)
        path = self.confined(row["model_path"], inspect_links=True)
        update = self.control.store.get("updates", model_id, {})
        revision = update.get("remote_revision")
        repo_id = row.get("source_repo_id")
        if (
            path.parent.name != "snapshots"
            or not repo_id
            or not isinstance(revision, str)
            or not re.fullmatch(r"[0-9a-fA-F]{7,64}", revision)
        ):
            raise ManagementError(
                "invalid_configuration",
                "Check updates on a cached Hugging Face model before staging",
            )
        cache_dir = self.confined(path.parent.parent).parent

        async def run(operation_id):
            from huggingface_hub import snapshot_download

            async with self.mutation(), self.file_reservation(path.parent.parent):
                if self.busy(model_id):
                    raise ManagementError("busy", "Model is busy")
                self.control.update_operation(
                    operation_id, stage="downloading cached revision", progress=20
                )
                staged = await self.runtime.run_native(
                    snapshot_download,
                    repo_id=repo_id,
                    revision=revision,
                    cache_dir=str(cache_dir),
                )
                candidate = self.confined(staged, inspect_links=True)
                if candidate != path.parent / revision:
                    raise RuntimeError(
                        "Download produced an unexpected cached revision"
                    )
                report = await self.runtime.run_native(
                    verify_model_files, {**row, "model_path": str(candidate)}
                )
                if report["status"] != "ready":
                    raise RuntimeError(
                        "Downloaded revision failed structural verification"
                    )
                # snapshot_download updates refs/main for branch downloads only;
                # the explicit commit leaves the active revision unchanged.
                self.write_store(
                    "updates",
                    model_id,
                    {**update, "staged_revision": revision, "staged_at": now()},
                )
                self.control.invalidate_storage_cache()
                return {
                    "model_id": model_id,
                    "revision": revision,
                    "path": str(candidate),
                    "activated": False,
                }

        operation = self.control.start_operation(
            "stage_update",
            run,
            model_id=model_id,
            payload={"revision": revision},
            cancellable=False,
        )
        return {
            "operation": self.control.update_operation(operation["id"], retryable=False)
        }


def copy_model_container(source, staging, destination):
    shutil.copytree(source, staging, symlinks=True)
    for item in source.rglob("*"):
        if not item.is_symlink():
            continue
        target = item.resolve()
        copied_link = staging / item.relative_to(source)
        final_link = destination / item.relative_to(source)
        if target.is_relative_to(source):
            mapped_target = destination / target.relative_to(source)
            if not item.readlink().is_absolute():
                continue
        else:
            mapped_target = target
        copied_link.unlink()
        copied_link.symlink_to(
            os.path.relpath(mapped_target, final_link.parent),
            target_is_directory=target.is_dir(),
        )


async def drain_async(awaitable):
    """Keep management admission until an engine's async worker has stopped."""
    worker = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if worker.done() and not worker.cancelled():
            worker.exception()
        raise


def file_manifest(path):
    result = []
    for item in sorted(path.rglob("*")):
        stat = item.lstat()
        result.append(
            (
                str(item.relative_to(path)),
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                str(item.readlink()) if item.is_symlink() else None,
            )
        )
    return result
