"""Model setting metadata and explicit helpers. Never load model weights."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from dataclasses import asdict
from pathlib import Path

import requests
from molto_config.model_profiles import PROFILE_FIELDS_SET, UNIVERSAL_FIELDS_SET
from molto_config.model_settings import ModelSettings, ane_prefill_backend
from molto_contracts.management import ModelSettingsPatch

from molto_management.management import ManagementError, ManagementService
from molto_management.recipe import (
    SNAPSHOT_EXCLUDED_FIELDS,
    clean_snapshot,
    decode_recipe,
)

PRESET_URL = "https://omlx.ai/assets/omlx_preset.json"
BENCHMARK_URL = "https://omlx.ai/api/benchmarks"


def validate_candidate(entry, settings):
    family = getattr(entry, "config_model_type", None) or ""
    from molto_contracts.runtime import engine_type_for

    engine_type = engine_type_for(
        settings.model_type_override, getattr(entry, "engine_type", "batched")
    )
    checks = {
        "qwen4_ple_ssd_offload": family == "qwen4_exp",
        "deepseek_v41_engram_ssd_offload": family == "deepseek_v41",
        "deepseek_v41_ced_prefill_enabled": family == "deepseek_v41",
        "qwen35_oq_a8_enabled": family.startswith(("qwen3_5", "qwen3_6", "qwen3_8")),
        "vlm_mtp_enabled": engine_type == "vlm",
        "dflash_enabled": engine_type in ("batched", "vlm"),
        "specprefill_enabled": engine_type == "batched",
        "turboquant_kv_enabled": engine_type in ("batched", "vlm"),
        "mtp_enabled": engine_type in ("batched", "vlm"),
    }
    if (
        ane_prefill_backend(family) == "k2"
        and settings.qwen35_ane_prefill_enabled
        and (
            settings.qwen35_ane_prefill_cpu_enabled
            or settings.qwen35_ane_prefill_fused_down
        )
    ):
        raise ValueError(
            "K2 ANE prefill does not support Qwen CPU or fused-down controls"
        )
    for name, supported in checks.items():
        if getattr(settings, name) and not supported:
            raise ValueError(f"{name} is unavailable for this model")
    if (
        settings.specprefill_keep_pct is not None
        and not 0.1 <= settings.specprefill_keep_pct <= 0.5
    ):
        raise ValueError("specprefill_keep_pct must be between 0.1 and 0.5")
    if settings.dflash_ssd_cache and not settings.dflash_in_memory_cache:
        raise ValueError("DFlash SSD cache requires its in-memory cache")
    if settings.guided_grammar_enabled and not (settings.guided_grammar or "").strip():
        raise ValueError("Guided grammar requires a grammar")
    if settings.guided_grammar_enabled:
        try:
            from molto_runtime._torch_stub import install

            install()
            import xgrammar as xgr

            xgr.Grammar.from_ebnf(settings.guided_grammar)
        except Exception as exc:
            raise ValueError(f"Invalid or unavailable guided grammar: {exc}") from exc
    if settings.reasoning_parser and settings.reasoning_parser not in parser_choices():
        raise ValueError("Unknown or unavailable reasoning parser")
    if settings.qwen35_oq_a8_enabled and not oq_a8_available():
        raise ValueError("oQ A8 prefill requires available native INT8 tensor kernels")
    if settings.qwen35_ane_prefill_enabled and ane_prefill_backend(family) == "qwen":
        fraction = (
            settings.qwen35_ane_prefill_fraction
            if settings.qwen35_ane_prefill_fraction is not None
            else 0.53
        )
        fused = settings.qwen35_ane_prefill_fused_down
        if fused and fraction > 0.5:
            raise ValueError(
                "Fused ANE down projection requires an MLP fraction at most 0.5"
            )
        if settings.qwen35_ane_prefill_cpu_enabled:
            if (
                fraction * (2 if fused else 1)
                + settings.qwen35_ane_prefill_cpu_fraction
                >= 1
            ):
                raise ValueError("ANE and CPU MLP fractions must total less than 1")
            if (
                settings.qwen35_ane_prefill_gdn
                and settings.qwen35_ane_prefill_gdn_fraction
                + settings.qwen35_ane_prefill_cpu_gdn_fraction
                >= 1
            ):
                raise ValueError("ANE and CPU GDN fractions must total less than 1")
    if settings.dflash_enabled:
        from molto_runtime.engine.dflash import is_dflash_compatible

        supported, reason = is_dflash_compatible(entry.model_path)
        if not supported:
            raise ValueError(reason)


def oq_a8_available():
    try:
        from molto_runtime.custom_kernels.qwen35_prefill import fast

        return bool(fast.oq_a8_available())
    except Exception:
        return False


def field_descriptions():
    descriptions = {}
    current = None
    for line in (inspect.getdoc(ModelSettings) or "").splitlines():
        match = re.match(r"    (\w+): (.*)", line)
        if match:
            current = match[1]
            descriptions[current] = match[2]
        elif current and line.startswith("        "):
            descriptions[current] += " " + line.strip()
    descriptions.update(
        {
            "is_hidden": "Hide from the inference model list while retaining management access.",
            "is_favorite": "List this model first in inference and management model lists.",
            "qwen4_ple_ssd_offload": "Keep Qwen4-Exp PLE N-gram tables on SSD and gather rows through mmap.",
            "deepseek_v41_engram_ssd_offload": "Keep DeepSeek V4.1 Engram tables on SSD through mmap.",
            "deepseek_v41_ced_prefill_enabled": "Limit DeepSeek V4.1 decoder prefill to the final window using CED.",
            "mtp_adaptive_max_depth": "Maximum adaptive MTP draft tokens per verify cycle, from 1 to 8.",
            "mtp_fixed_depth": "Draft exactly this many MTP tokens per cycle; null selects adaptive depth.",
            "preserve_thinking": "Keep reasoning blocks in historical conversation turns when the template supports it.",
            "cache_reasoning_output": "Cache reasoning output for subsequent turns; null selects automatic behavior.",
        }
    )
    return descriptions


def parser_choices():
    try:
        from molto_runtime._torch_stub import install

        install()
        from xgrammar.builtin_structural_tag import _structural_tag_registry

        return sorted(_structural_tag_registry)
    except Exception:
        try:
            from xgrammar import get_builtin_structural_tag_supported_models

            return sorted(get_builtin_structural_tag_supported_models())
        except Exception:
            return []


def options(service: ManagementService, model_id=None):
    defaults = asdict(ModelSettings())
    schema = ModelSettingsPatch.model_json_schema()["properties"]
    fields = []
    descriptions = field_descriptions()
    for key, definition in schema.items():
        leaf = next(
            (v for v in definition.get("anyOf", []) if v.get("type") != "null"),
            definition,
        )
        fields.append(
            {
                "key": key,
                "label": key.replace("_", " ").capitalize(),
                "type": leaf.get("type", "string"),
                "default": defaults[key],
                "minimum": leaf.get("minimum"),
                "maximum": leaf.get("maximum"),
                "choices": leaf.get("enum"),
                "profile": key in PROFILE_FIELDS_SET,
                "template": key in UNIVERSAL_FIELDS_SET,
                "description": "Allows model repository Python to execute on the next load. Enable only for a trusted repository."
                if key == "trust_remote_code"
                else descriptions.get(key, "Configure " + key.replace("_", " ") + "."),
            }
        )
    choices = {
        "turboquant_kv_bits": [2, 2.5, 3, 3.5, 4, 6, 8],
        "dflash_verify_mode": ["dflash", "adaptive", "ddtree", "off"],
        "reasoning_parser": parser_choices(),
        "dflash_draft_quant_weight_bits": [2, 4, 8],
        "dflash_draft_quant_activation_bits": [16, 32],
        "dflash_draft_quant_group_size": [32, 64, 128],
    }
    for field in fields:
        key = field["key"]
        field["group"] = (
            "advanced"
            if key.startswith(
                (
                    "qwen",
                    "deepseek",
                    "mtp",
                    "vlm_mtp",
                    "dflash",
                    "specprefill",
                    "turboquant",
                    "moe",
                    "index_cache",
                )
            )
            else "general"
        )
        field["advanced"] = field["group"] == "advanced"
        field["reload_required"] = field["advanced"] or key in (
            "trust_remote_code",
            "model_type_override",
        )
        field["supported"] = True
        if key in choices:
            field["choices"] = choices[key]
        if key == "ttl_seconds":
            field["description"] = (
                "Unload after this many idle seconds. Zero expires immediately when idle; null uses the global timeout. Pinned models do not expire."
            )
    result = {
        "fields": fields,
        "profile_fields": sorted(PROFILE_FIELDS_SET),
        "template_fields": sorted(UNIVERSAL_FIELDS_SET),
        "defaults": defaults,
    }
    if model_id is not None:
        entry = service._model(model_id)
        result.update(
            model_id=model_id,
            family=entry.config_model_type,
            capabilities={
                "ane_prefill": ane_prefill_backend(entry.config_model_type) is not None,
                "vlm_mtp": entry.engine_type == "vlm",
                "dflash": entry.engine_type == "batched",
            },
        )
        family = entry.config_model_type or ""
        gates = {
            "qwen4_ple_ssd_offload": family == "qwen4_exp",
            "deepseek_v41_engram_ssd_offload": family == "deepseek_v41",
            "deepseek_v41_ced_prefill_enabled": family == "deepseek_v41",
            "qwen35_ane_prefill": ane_prefill_backend(family) is not None,
            "qwen35_oq_a8": family.startswith(("qwen3_5", "qwen3_6", "qwen3_8"))
            and oq_a8_available(),
            "vlm_mtp": entry.engine_type == "vlm",
            "specprefill": entry.engine_type == "batched",
            "dflash": entry.engine_type == "batched",
        }
        try:
            from molto_management.model_compat import mtp_compatibility

            gates["mtp"], mtp_reason = mtp_compatibility(entry.model_path)
        except (ImportError, OSError, ValueError):
            gates["mtp"], mtp_reason = (
                False,
                "MTP compatibility could not be established",
            )
        try:
            from molto_runtime.engine.dflash import is_dflash_compatible

            gates["dflash"], dflash_reason = is_dflash_compatible(entry.model_path)
        except ImportError:
            gates["dflash"], dflash_reason = False, "DFlash runtime is unavailable"
        from molto_runtime.patches.moe_offload_compat import moe_offload_compatibility

        gates["moe_expert_offload"], offload_reason = moe_offload_compatibility(
            entry.model_path
        )
        for field in fields:
            for prefix, supported in gates.items():
                if field["key"].startswith(prefix):
                    field["supported"] = supported
                    field["unsupported_reason"] = (
                        None
                        if supported
                        else mtp_reason
                        if prefix == "mtp"
                        else dflash_reason
                        if prefix == "dflash"
                        else offload_reason
                        if prefix == "moe_expert_offload"
                        else "Unavailable for this model family or engine"
                    )
        result["capabilities"].update(gates)
        result["mtplx_sidecar_detected"] = mtplx_detected(Path(entry.model_path))
        try:
            mtplx_path(service, model_id)
            result["capabilities"]["mtplx_import"] = True
            result["mtplx_import_reason"] = None
        except ManagementError as exc:
            result["capabilities"]["mtplx_import"] = False
            result["mtplx_import_reason"] = exc.detail
        settings = asdict(service.manager.get_settings(model_id))
        active = settings.get("active_profile_name")
        profile = service.manager.get_profile(model_id, active) if active else None
        result["active_profile"] = active
        profile_values = profile.get("settings", {}) if profile else {}
        expected = {key: defaults[key] for key in UNIVERSAL_FIELDS_SET}
        expected.update(
            {
                key: defaults[key] if value is None else value
                for key, value in profile_values.items()
            }
        )
        result["profile_drift"] = bool(
            profile
            and any(settings.get(key) != value for key, value in expected.items())
        )
        source = profile.get("source_template") if profile else None
        template = service.manager.get_template(source) if source else None
        result["template_drift"] = bool(
            template
            and template.get("settings", {})
            != {
                key: value
                for key, value in profile_values.items()
                if key in UNIVERSAL_FIELDS_SET
            }
        )
    return result


def mtplx_detected(path: Path) -> bool:
    """Distinguish an MTPLX export from an ordinary incompatible checkpoint."""
    try:
        if any(
            (path / name).is_file()
            for name in (
                "mtp.safetensors",
                "mtp/weights.safetensors",
                "model-mtp.safetensors",
                "mtplx_runtime.json",
            )
        ):
            return True
        config = json.loads((path / "config.json").read_text())
        if not isinstance(config, dict):
            return False
        extra = config.get("mlx_lm_extra_tensors")
        return bool(
            (isinstance(extra, dict) and extra.get("mtp_file"))
            or "mtplx_mtp_contract" in config
            or "mtplx_mtp_payload_audit" in config
        )
    except (OSError, ValueError, RuntimeError):
        return False


def mtplx_path(service, model_id):
    from molto_runtime.oq import (
        _resolve_mtplx_sidecar,
        _validate_mtplx_runtime_contract,
    )

    entry = service._model(model_id)
    path = Path(entry.model_path).resolve()
    try:
        config = json.loads((path / "config.json").read_text())
        if not isinstance(config, dict):
            raise ValueError("Model config must be an object")
        sidecar = _resolve_mtplx_sidecar(path, config)
        if sidecar is None:
            raise ValueError("No MTPLX sidecar found")
        paths = [
            sidecar,
            path / "config.json",
            path / "mtplx_runtime.json",
            path / "model.safetensors.index.json",
            path / "model-mtp.safetensors",
            *path.glob("*.safetensors"),
        ]
        if any(not item.resolve().is_relative_to(path) for item in paths):
            raise ValueError(
                "MTPLX import files must remain inside the model directory"
            )
        runtime = json.loads((path / "mtplx_runtime.json").read_text())
        if not isinstance(runtime, dict):
            raise ValueError("MTPLX runtime contract must be an object")
        index = path / "model.safetensors.index.json"
        if index.exists() and not isinstance(json.loads(index.read_text()), dict):
            raise ValueError("Model weight index must be an object")
        _validate_mtplx_runtime_contract(path, config)
        audit = config.get("mtplx_mtp_payload_audit")
        if isinstance(audit, dict) and not bool(audit.get("passed", True)):
            raise ValueError("MTPLX payload audit failed")
    except (OSError, ValueError, RuntimeError) as exc:
        raise ManagementError("invalid_configuration", str(exc)) from exc
    return path


class ModelHelpers:
    def __init__(self, service):
        self.service = service
        self.manager = service.manager

    def templates(self):
        return {"templates": self.manager.list_templates()}

    def write_template(self, body, name=None):
        changes = body.model_dump(exclude_unset=True)
        if name is None:
            changes.setdefault("display_name", body.display_name)
            changes.setdefault("description", body.description)
        settings_patch = body.settings
        if settings_patch is None and name is not None:
            existing = self.manager.get_template(name)
            if existing is None:
                raise ManagementError("not_found", "Template not found")
            try:
                settings_patch = ModelSettingsPatch.model_validate(
                    existing.get("settings", {})
                )
            except ValueError as exc:
                raise ManagementError("invalid_configuration", str(exc)) from exc
        if settings_patch is not None:
            values = settings_patch.model_dump(exclude_unset=True)
            if set(values) - UNIVERSAL_FIELDS_SET:
                raise ManagementError(
                    "invalid_configuration", "Templates accept only universal fields"
                )
            defaults = asdict(ModelSettings())
            changes["settings"] = values
            # Universal templates must validate before any mutation.
            from types import SimpleNamespace

            try:
                candidate = ModelSettings(
                    **{k: defaults[k] if v is None else v for k, v in values.items()}
                )
                self.service._validate_model_settings(
                    SimpleNamespace(
                        config_model_type=None, engine_type="batched", model_path=""
                    ),
                    candidate,
                )
            except (TypeError, ValueError) as exc:
                raise ManagementError("invalid_configuration", str(exc)) from exc
        before = self.manager.snapshot_templates()
        try:
            result = (
                self.manager.save_template(**changes)
                if name is None
                else self.manager.update_template(name, **changes)
            )
        except ValueError as exc:
            raise ManagementError("conflict", str(exc)) from exc
        except OSError as exc:
            self.manager.restore_templates(before)
            raise ManagementError("unavailable", str(exc)) from exc
        if result is None:
            raise ManagementError("not_found", "Template not found")
        return {"template": result}

    def delete_template(self, name):
        try:
            deleted = self.manager.delete_template(name)
        except OSError as exc:
            raise ManagementError("unavailable", str(exc)) from exc
        if not deleted:
            raise ManagementError("not_found", "Template not found")
        return {"deleted": True, "name": name}

    async def apply_template(self, model_id, name):
        service = self.service
        service._require_mutation_admission()
        entry = service._model(model_id)
        if entry.is_loading or service.pool.is_model_unloading(model_id):
            raise ManagementError("busy", "Model is loading or unloading")
        previous = self.manager.get_settings(model_id)
        old_type = (entry.model_type, entry.engine_type)
        signature = service.pool.runtime_signature(model_id, previous)

        def validate(values):
            candidate = ModelSettings(**values)
            service._validate_model_settings(entry, candidate)
            service._validate_drafts(entry, candidate)

        try:
            result = self.manager.apply_template(
                model_id,
                name,
                validate,
            )
        except ValueError as exc:
            raise ManagementError("invalid_configuration", str(exc)) from exc
        except OSError as exc:
            raise ManagementError("unavailable", str(exc)) from exc
        if result is None:
            raise ManagementError("not_found", "Template not found")
        transition = await service._reload_after_settings_change(
            model_id, entry, old_type, signature
        )
        return {"model_id": model_id, "settings": asdict(result), **transition}

    async def reset(self, model_id):
        return await self.service.update_model_settings(
            model_id,
            ModelSettingsPatch(
                **{key: None for key in ModelSettingsPatch.model_fields}
            ),
        )

    def generation_config(self, model_id):
        path = Path(self.service._model(model_id).model_path)
        result = {}
        for filename in ("generation_config.json", "config.json"):
            file = path / filename
            if not file.exists():
                continue
            try:
                data = json.loads(file.read_text())
                if not isinstance(data, dict):
                    raise ValueError("Configuration must be an object")
            except (OSError, ValueError) as exc:
                raise ManagementError("invalid_configuration", str(exc)) from exc
            if filename == "generation_config.json":
                result.update(
                    {
                        k: data[k]
                        for k in (
                            "temperature",
                            "top_p",
                            "top_k",
                            "min_p",
                            "repetition_penalty",
                            "max_tokens",
                        )
                        if k in data
                    }
                )
                if data.get("do_sample") is False:
                    result["temperature"] = 0.0
            else:
                config = data.get("text_config") or data
                for key in (
                    "max_position_embeddings",
                    "max_seq_len",
                    "seq_length",
                    "n_positions",
                ):
                    if config.get(key):
                        result["max_context_window"] = config[key]
                        break
        if not result:
            raise ManagementError("not_found", "No generation defaults found")
        return {"settings": ModelSettingsPatch(**result).model_dump(exclude_unset=True)}

    async def snapshot(self, model_id, snapshot):
        values = clean_snapshot(snapshot)
        if not values:
            raise ManagementError(
                "invalid_configuration", "Snapshot contains no applicable settings"
            )
        patch = {
            key: values.get(key)
            for key in PROFILE_FIELDS_SET
            if key in ModelSettingsPatch.model_fields
            and key not in SNAPSHOT_EXCLUDED_FIELDS
        }
        return await self.service.update_model_settings(
            model_id, ModelSettingsPatch(**patch)
        )

    async def recipe(self, model_id, recipe):
        try:
            snapshot = decode_recipe(recipe)
        except ValueError as exc:
            raise ManagementError("invalid_configuration", str(exc)) from exc
        return await self.snapshot(model_id, snapshot)

    async def fetch(self, url):
        def request():
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            if len(response.content) > 2_000_000:
                raise ValueError("Remote response is too large")
            return response.json()

        try:
            return await asyncio.to_thread(request)
        except (requests.RequestException, ValueError) as exc:
            raise ManagementError("unavailable", str(exc)) from exc

    async def refresh_presets(self):
        bundle = await self.fetch(PRESET_URL)
        if not isinstance(bundle, dict) or not isinstance(bundle.get("presets"), list):
            raise ManagementError("unavailable", "Invalid preset bundle")
        try:
            for preset in bundle["presets"]:
                if not isinstance(preset, dict) or not isinstance(
                    preset.get("name"), str
                ):
                    raise ValueError("Preset requires a name")
                patch = ModelSettingsPatch(**preset.get("settings", {}))
                if patch.model_fields_set - UNIVERSAL_FIELDS_SET:
                    raise ValueError("Preset contains model-specific fields")
        except (TypeError, ValueError) as exc:
            raise ManagementError("unavailable", str(exc)) from exc
        return bundle

    async def optimal(self, model_id, benchmark_id):
        self.service._model(model_id)
        row = await self.fetch(f"{BENCHMARK_URL}/{benchmark_id}")
        snapshot = row.get("model_settings") if isinstance(row, dict) else None
        if not isinstance(snapshot, dict) or not snapshot:
            raise ManagementError(
                "invalid_configuration", "Benchmark has no settings snapshot"
            )
        return await self.snapshot(model_id, snapshot)

    async def optimal_candidates(self, model_id):
        from urllib.parse import urlencode

        from molto_runtime.utils.hardware import (
            get_chip_name,
            get_total_memory_gb,
            parse_chip_info,
        )

        self.service._model(model_id)
        chip, variant = parse_chip_info(get_chip_name())
        device = {
            "chip_name": chip,
            "chip_variant": variant,
            "memory_gb": round(get_total_memory_gb()),
        }
        groups = {}
        for sort in ("pp", "tg"):
            params = {
                "chip": chip,
                "variant": variant,
                "memory_gb": device["memory_gb"],
                "model": model_id.split("/")[-1],
                "context": 4096,
                "limit": 3,
                "sort": sort,
            }
            data = await self.fetch(f"{BENCHMARK_URL}/best?{urlencode(params)}")
            rows = data.get("results", []) if isinstance(data, dict) else []
            groups[sort] = [
                {
                    "benchmark_id": row["id"],
                    **{
                        key: row.get(key)
                        for key in (
                            "pp_tps",
                            "tg_tps",
                            "quantization",
                            "omlx_version",
                            "created_at",
                            "memory_gb",
                        )
                    },
                }
                for row in rows
                if isinstance(row, dict) and row.get("id")
            ]
        return {
            "found": bool(groups["pp"] or groups["tg"]),
            "device": device,
            "context_length": 4096,
            "by_pp": groups["pp"],
            "by_tg": groups["tg"],
        }

    async def apply_preset(self, model_id, name):
        bundle = json.loads(
            Path(__file__).with_name("management_presets.json").read_text()
        )
        preset = next(
            (item for item in bundle["presets"] if item["name"] == name), None
        )
        if preset is None:
            raise ManagementError("not_found", "Preset not found")
        return await self.service.update_model_settings(
            model_id, ModelSettingsPatch(**preset["settings"])
        )

    async def import_mtplx(self, model_id, runtime):
        import uuid

        from molto_runtime.oq import import_mtplx_sidecar

        service = self.service
        service._require_mutation_admission()
        entry = service._model(model_id)

        def assert_idle():
            if (
                entry.loaded
                or entry.is_loading
                or getattr(entry, "in_use", 0)
                or getattr(entry, "pending_unload_reason", None)
                or service.pool.is_model_unloading(model_id)
            ):
                raise ManagementError(
                    "busy",
                    "Unload the model and wait for active work before importing MTPLX",
                )

        assert_idle()
        path = mtplx_path(service, model_id)
        if not any(path.is_relative_to(root) for root in runtime.roots):
            raise ManagementError(
                "invalid_configuration",
                "Model checkpoint is outside configured model roots",
            )
        if runtime.operation_lock.locked() or runtime.mutation_lock.locked():
            raise ManagementError(
                "busy", "Another native management operation is active"
            )
        owner = f"mtplx:{uuid.uuid4().hex}"
        runtime.reserve_paths([path], owner)
        try:
            async with (
                runtime.operation_lock,
                runtime.mutation_lock,
                service.pool.exclusive_management(),
            ):
                assert_idle()
                path = mtplx_path(service, model_id)
                try:
                    result = await runtime.run_native(import_mtplx_sidecar, str(path))
                except ValueError as exc:
                    raise ManagementError("invalid_configuration", str(exc)) from exc
                except (OSError, RuntimeError) as exc:
                    raise ManagementError("unavailable", str(exc)) from exc
                return {
                    "status": "ok",
                    "model_id": model_id,
                    **result,
                    "options": options(service, model_id),
                }
        finally:
            runtime.release_paths(owner)
