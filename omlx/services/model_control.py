# SPDX-License-Identifier: Apache-2.0
"""Shared model-management primitives for management clients.

The inference engines keep their own runtime state.  This module adds the
durable, engine-neutral state needed by control clients: policies,
health reports, update reports, operation history, and model collections.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import re
import threading
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CONTROL_SCHEMA_VERSION = 1
_ACTIVE_OPERATION_STATES = {"queued", "running"}


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _empty_document() -> dict[str, Any]:
    return {
        "version": CONTROL_SCHEMA_VERSION,
        "policies": {},
        "health": {},
        "updates": {},
        "operations": {},
        "collections": {},
    }


class ModelControlStore:
    """Atomic JSON persistence for the web model control center."""

    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path).expanduser()
        self.path = self.base_path / "model_control.json"
        self._lock = threading.RLock()
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        document = _empty_document()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return document
            for key in document:
                if key == "version":
                    continue
                value = raw.get(key)
                if isinstance(value, dict):
                    document[key] = value
        except (OSError, json.JSONDecodeError):
            pass

        # Running work cannot survive a process exit.  Keep the record and make
        # the retry explicit rather than leaving a task stuck as "running".
        for operation in document["operations"].values():
            if operation.get("status") in _ACTIVE_OPERATION_STATES:
                operation["status"] = "interrupted"
                operation["error"] = "Server restarted before the operation finished"
                operation["finished_at"] = _utcnow()
        return document

    def _save_locked(self) -> None:
        _atomic_json_write(self.path, self._data)

    def section(self, name: str) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._data.get(name, {})))

    def get(self, section: str, key: str, default: Any = None) -> Any:
        with self._lock:
            value = self._data.get(section, {}).get(key, default)
            return json.loads(json.dumps(value))

    def put(self, section: str, key: str, value: Any) -> None:
        with self._lock:
            self._data.setdefault(section, {})[key] = value
            self._save_locked()

    def remove(self, section: str, key: str) -> bool:
        with self._lock:
            target = self._data.setdefault(section, {})
            if key not in target:
                return False
            del target[key]
            self._save_locked()
            return True

    def merge(self, section: str, values: dict[str, Any]) -> None:
        if not values:
            return
        with self._lock:
            self._data.setdefault(section, {}).update(values)
            self._save_locked()

    def export_state(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._data))


@dataclass(frozen=True)
class CapabilitySet:
    tasks: tuple[str, ...]
    endpoints: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    dependency: str | None = None


_CAPABILITIES: dict[str, CapabilitySet] = {
    "llm": CapabilitySet(
        ("chat", "text-generation"),
        ("/v1/chat/completions", "/v1/completions", "/v1/responses"),
        ("text",),
        ("text",),
        "mlx-lm",
    ),
    "vlm": CapabilitySet(
        ("chat", "vision-language", "text-generation"),
        ("/v1/chat/completions", "/v1/responses"),
        ("text", "image", "video"),
        ("text",),
        "mlx-vlm",
    ),
    "embedding": CapabilitySet(
        ("embedding",), ("/v1/embeddings",), ("text",), ("vector",), "mlx-embeddings"
    ),
    "reranker": CapabilitySet(
        ("reranking",), ("/v1/rerank",), ("text",), ("scores",), "mlx-embeddings"
    ),
    "audio_stt": CapabilitySet(
        ("speech-to-text",),
        ("/v1/audio/transcriptions", "/v1/audio/translations"),
        ("audio",),
        ("text",),
        "mlx-audio",
    ),
    "audio_tts": CapabilitySet(
        ("text-to-speech",), ("/v1/audio/speech",), ("text",), ("audio",), "mlx-audio"
    ),
    "audio_sts": CapabilitySet(
        ("speech-to-speech",),
        ("/v1/audio/speech-to-speech",),
        ("audio",),
        ("audio",),
        "mlx-audio",
    ),
    "image_generation": CapabilitySet(
        ("text-to-image",),
        ("/v1/images/generations",),
        ("text",),
        ("image",),
        "mflux",
    ),
    "markitdown": CapabilitySet(
        ("document-to-text",),
        ("/v1/chat/completions", "/v1/responses"),
        ("document",),
        ("text",),
        "markitdown",
    ),
    "adapter": CapabilitySet(("adapter",), (), ("weights",), ("model-variant",), None),
    "other": CapabilitySet((), (), (), (), None),
}


def diffusion_metadata(path: Path) -> dict[str, Any] | None:
    """Describe a local image checkpoint without importing its runtime."""
    from ..diffusion import detect_checkpoint, pipelines_for_model
    from ..diffusion.preparation import QUANTIZATION_MODELS

    try:
        checkpoint = detect_checkpoint(path)
    except ValueError as exc:
        return {"supported": False, "reason": str(exc), "pipelines": []}
    if checkpoint is None:
        return None
    pipelines = pipelines_for_model(checkpoint.base_model)
    calibrated = checkpoint.base_model in QUANTIZATION_MODELS
    fresh = calibrated and checkpoint.quantization is None
    return {
        **checkpoint.metadata(),
        "supported": any(spec.local_unsupported_reason is None for spec in pipelines),
        "calibration": {
            "available": calibrated,
            "operations": ["txt2img"] if calibrated else [],
            "reason": None
            if calibrated
            else "Calibrated preparation is not integrated for this checkpoint identity",
        },
        "calibrated_quantization": {
            "available": fresh,
            "method": "diffusion-oQe",
            "scope": "transformer linears",
            "reason": None
            if fresh
            else (
                "Fresh quantization requires floating-point source weights"
                if calibrated
                else "Calibrated preparation is not integrated for this checkpoint identity"
            ),
        },
    }


def capabilities_for(model_type: str, model_path: Path | None = None) -> dict[str, Any]:
    if model_type == "image_generation":
        from ..diffusion.registry import PIPELINES

        details = diffusion_metadata(model_path) if model_path is not None else None
        pipelines = (
            (details or {}).get("pipelines", [])
            if model_path is not None
            else [spec.metadata() for spec in PIPELINES.values()]
        )
        supported = [
            spec for spec in pipelines if spec["local_unsupported_reason"] is None
        ]
        operations = sorted({spec["operation"] for spec in supported})
        endpoints = []
        if "txt2img" in operations:
            endpoints.append("/v1/images/generations")
        if set(operations) & {"img2img", "reference-edit", "inpaint"}:
            endpoints.append("/v1/images/edits")
        if operations:
            endpoints.append("/v1/images/operations")
        return {
            "tasks": operations,
            "endpoints": endpoints,
            "inputs": ["text", "image"]
            if any(spec["image_max"] for spec in supported)
            else ["text"],
            "outputs": ["image"],
            "dependency": "mflux",
            "pipelines": pipelines,
        }
    capability = _CAPABILITIES.get(model_type, _CAPABILITIES["llm"])
    return {
        "tasks": list(capability.tasks),
        "endpoints": list(capability.endpoints),
        "inputs": list(capability.inputs),
        "outputs": list(capability.outputs),
        "dependency": capability.dependency,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _weight_metadata(path: Path) -> dict[str, str]:
    """Read metadata from the first usable weight shard without loading it."""
    try:
        from safetensors import safe_open
    except ImportError:
        return {}
    candidates = sorted(path.glob("model*.safetensors"))
    if not candidates:
        candidates = sorted(path.glob("*/*.safetensors"))
    for shard in candidates:
        try:
            with safe_open(str(shard), framework="numpy") as stream:
                return dict(stream.metadata() or {})
        except Exception:
            continue
    return {}


def artifact_characteristics(path: Path) -> dict[str, Any]:
    """Describe the checkpoint format and precision used for operation routing."""
    config = _read_json(path / "config.json")
    manifest = _read_json(path / "omlx-mflux.json")
    metadata = _weight_metadata(path)

    if manifest.get("backend") == "mflux" or metadata.get("mflux_version"):
        model_format = "mflux"
    elif str(metadata.get("format", "")).lower() == "mlx":
        model_format = "mlx"
    else:
        model_format = "huggingface"

    bits = manifest.get(
        "converted_quantization_bits", manifest.get("quantization_bits")
    )
    if bits is None:
        bits = metadata.get("quantization_level")
    quantization = config.get("quantization")
    if bits is None and isinstance(quantization, dict):
        bits = quantization.get("bits")
    try:
        bits = int(bits) if bits is not None else None
    except (TypeError, ValueError):
        bits = None

    quantization_config = config.get("quantization_config")
    quant_method = None
    if isinstance(quantization_config, dict):
        quant_method = str(quantization_config.get("quant_method") or "").lower()
    if bits is not None:
        precision = f"{bits}-bit"
        quantized = True
    elif quant_method in {"fp8", "mxfp8", "compressed-tensors"}:
        precision = "FP8"
        quantized = False
    else:
        raw_dtype = config.get("torch_dtype") or config.get("dtype")
        precision = str(raw_dtype).upper() if raw_dtype else "Full precision"
        quantized = False

    return {
        "format": model_format,
        "precision": precision,
        "quantized": quantized,
        "quantization_bits": bits,
    }


def source_metadata(model_path: str, source_repo_id: str | None) -> dict[str, Any]:
    path = Path(model_path)
    characteristics = artifact_characteristics(path)
    revision = None
    cached_revisions: list[dict[str, Any]] = []
    cache_root = None
    if path.parent.name == "snapshots":
        cache_root = path.parent.parent
        revision = path.name
        for snapshot in sorted(path.parent.iterdir()) if path.parent.is_dir() else []:
            if snapshot.is_dir():
                try:
                    modified = snapshot.stat().st_mtime
                except OSError:
                    modified = 0
                cached_revisions.append(
                    {
                        "revision": snapshot.name,
                        "active": snapshot == path,
                        "modified_at": modified,
                    }
                )
    config = _read_json(path / "config.json")
    model_index = _read_json(path / "model_index.json")
    mflux_manifest = _read_json(path / "omlx-mflux.json")
    adapter_config = _read_json(path / "adapter_config.json")
    quantization = config.get("quantization") or config.get("quantization_config")
    if not isinstance(quantization, dict):
        quantization = {}
    card_metadata: dict[str, str] = {}
    try:
        card = (path / "README.md").read_text(encoding="utf-8")
        if card.startswith("---"):
            frontmatter = card.split("---", 2)[1]
            for line in frontmatter.splitlines():
                key, separator, value = line.partition(":")
                if separator and key.strip() in {"license", "gated", "base_model"}:
                    card_metadata[key.strip()] = value.strip().strip("'\"")
    except OSError:
        pass
    details = diffusion_metadata(path) if (model_index or mflux_manifest) else None
    return {
        "repo_id": source_repo_id,
        "revision": revision,
        "cache_root": str(cache_root) if cache_root else None,
        "cached_revisions": cached_revisions,
        "declared_source": config.get("_name_or_path")
        or model_index.get("_name_or_path"),
        "architectures": config.get("architectures") or [],
        "quantization": quantization,
        "license": card_metadata.get("license"),
        "gated": card_metadata.get("gated", "false").lower()
        in {"true", "yes", "manual"},
        "base_model": card_metadata.get("base_model"),
        "adapter_base_model": adapter_config.get("base_model_name_or_path"),
        "adapter_type": adapter_config.get("peft_type"),
        "model_family": mflux_manifest.get(
            "model_family", mflux_manifest.get("base_model")
        ),
        **({"diffusion": details} if details is not None else {}),
        **characteristics,
    }


_CONVERSION_ADAPTERS: dict[str, tuple[str, str, str]] = {
    "llm": ("mlx-lm", "mlx_lm", "mlx"),
    "vlm": ("mlx-vlm", "mlx_vlm", "mlx"),
    "embedding": ("mlx-embeddings", "mlx_embeddings", "mlx"),
    "reranker": ("mlx-embeddings", "mlx_embeddings", "mlx"),
    "audio_stt": ("mlx-audio", "mlx_audio", "mlx"),
    "audio_tts": ("mlx-audio", "mlx_audio", "mlx"),
    "audio_sts": ("mlx-audio", "mlx_audio", "mlx"),
    "image_generation": ("mflux", "mflux", "mflux"),
}


def _preparation_modality(model_type: str) -> str:
    if model_type == "image_generation":
        return "image"
    if model_type.startswith("audio_"):
        return "audio"
    if model_type == "vlm":
        return "multimodal"
    if model_type in {"embedding", "reranker", "llm"}:
        return "text"
    if model_type == "markitdown":
        return "document"
    return "other"


def _classify_preparation_path(path: Path) -> tuple[str, str]:
    """Classify raw artifacts without pretending every config is a text LLM."""
    from ..model_discovery import detect_model_type

    detected = detect_model_type(path)
    config = _read_json(path / "config.json")
    config_type = str(config.get("model_type") or "").lower()
    architectures = [str(value).lower() for value in config.get("architectures", [])]
    audio_hint = any(
        token in config_type
        for token in ("wav2vec", "wavlm", "hubert", "audio", "speech")
    ) or any(
        token in architecture
        for architecture in architectures
        for token in ("wav2vec", "wavlm", "hubert", "audio", "speech")
    )
    text_generation_hint = any(
        "causallm" in architecture or "conditionalgeneration" in architecture
        for architecture in architectures
    )
    if detected == "llm" and audio_hint:
        return "other", "audio"
    if detected == "llm" and architectures and not text_generation_hint:
        return "other", "other"
    return detected, _preparation_modality(detected)


def _format_preparation_size(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    if value < 1024**2:
        return f"{value / 1024:.1f} KB"
    if value < 1024**3:
        return f"{value / 1024**2:.1f} MB"
    return f"{value / 1024**3:.1f} GB"


def _conversion_capability(model_type: str, source_format: str) -> dict[str, Any]:
    route = _CONVERSION_ADAPTERS.get(model_type)
    if route is None:
        return {
            "available": False,
            "adapter": None,
            "target_format": None,
            "reason": f"No converter is registered for {model_type.replace('_', ' ')} models",
        }
    adapter, module_name, target_format = route
    if source_format == target_format:
        target_label = (
            target_format.upper() if target_format == "mlx" else target_format
        )
        return {
            "available": False,
            "adapter": adapter,
            "target_format": target_format,
            "reason": f"This model is already in {target_label} format",
        }
    if importlib.util.find_spec(module_name) is None:
        return {
            "available": False,
            "adapter": adapter,
            "target_format": target_format,
            "reason": f"{adapter} is not installed",
        }
    return {
        "available": True,
        "adapter": adapter,
        "target_format": target_format,
        "reason": None,
    }


def build_preparation_catalog(
    records: list[dict[str, Any]],
    oq_source_models: list[dict[str, Any]],
    oq_all_models: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge discovery and oQ scans into one source-model catalog."""
    catalog: dict[str, dict[str, Any]] = {}

    def path_key(value: str) -> str:
        return str(Path(value).expanduser().resolve())

    for record in records:
        raw_path = str(record.get("path") or "")
        if not raw_path or record.get("kind") == "virtual":
            continue
        key = path_key(raw_path)
        source = record.get("source") or {}
        size = int((record.get("storage") or {}).get("estimated_bytes") or 0)
        model_type = str(record.get("model_type") or "other")
        catalog[key] = {
            "id": record.get("id") or key,
            "name": record.get("display_name") or Path(key).name,
            "path": key,
            "source_repo_id": source.get("repo_id"),
            "model_type": model_type,
            "config_model_type": record.get("config_model_type") or "",
            "modality": record.get("preparation_modality")
            or _preparation_modality(model_type),
            "format": source.get("format") or "unknown",
            "precision": source.get("precision") or "Unknown",
            "quantized": bool(source.get("quantized")),
            "size": size,
            "size_formatted": _format_preparation_size(size),
            "blocker": record.get("preparation_blocker"),
            "oq": None,
        }

    for oq_model in oq_all_models:
        raw_path = str(oq_model.get("path") or "")
        if not raw_path:
            continue
        key = path_key(raw_path)
        current = catalog.get(key)
        if current is None:
            model_type, modality = _classify_preparation_path(Path(key))
            size = int(oq_model.get("size") or 0)
            current = {
                "id": oq_model.get("source_repo_id") or oq_model.get("name") or key,
                "name": oq_model.get("source_repo_id")
                or oq_model.get("name")
                or Path(key).name,
                "path": key,
                "source_repo_id": oq_model.get("source_repo_id"),
                "model_type": model_type,
                "config_model_type": oq_model.get("model_type") or "",
                "modality": modality,
                "format": oq_model.get("format") or "unknown",
                "precision": oq_model.get("precision") or "Unknown",
                "quantized": bool(oq_model.get("is_quantized")),
                "size": size,
                "size_formatted": oq_model.get("size_formatted")
                or _format_preparation_size(size),
                "blocker": None,
                "oq": None,
            }
            catalog[key] = current
        elif not current["size"]:
            current["size"] = int(oq_model.get("size") or 0)
            current["size_formatted"] = oq_model.get(
                "size_formatted"
            ) or _format_preparation_size(current["size"])
        current["oq"] = oq_model

    quantizable = {
        path_key(str(model["path"])): model
        for model in oq_source_models
        if model.get("path")
    }
    for key, item in catalog.items():
        oq_model = quantizable.get(key)
        if oq_model is not None:
            item["oq"] = oq_model
            quantization = {
                "available": True,
                "adapter": "oQ",
                "requires_conversion": bool(oq_model.get("conversion_required")),
                "reason": None,
            }
        elif item.get("blocker"):
            quantization = {
                "available": False,
                "adapter": "oQ",
                "requires_conversion": False,
                "reason": item["blocker"],
            }
        elif item["quantized"] or (item.get("oq") or {}).get("is_quantized"):
            quantization = {
                "available": False,
                "adapter": "oQ",
                "requires_conversion": False,
                "reason": "oQ quantization requires a full-precision source model",
            }
        elif item["model_type"] == "image_generation":
            quantization = {
                "available": False,
                "adapter": "oQ",
                "requires_conversion": False,
                "reason": "oQ does not yet have a diffusion-model tensor adapter",
            }
        elif item["model_type"].startswith("audio_"):
            quantization = {
                "available": False,
                "adapter": "oQ",
                "requires_conversion": False,
                "reason": "oQ does not yet have an audio-model tensor adapter",
            }
        else:
            quantization = {
                "available": False,
                "adapter": "oQ",
                "requires_conversion": False,
                "reason": "This checkpoint layout is not supported by oQ",
            }

        conversion = _conversion_capability(item["model_type"], item["format"])
        if item["model_type"] == "image_generation":
            details = diffusion_metadata(Path(key))
            item["diffusion"] = details
            pipeline = next(
                (
                    spec
                    for spec in (details or {}).get("pipelines", [])
                    if spec["id"] == details.get("default_pipeline")
                ),
                None,
            )
            reason = (
                item.get("blocker")
                or ((details or {}).get("reason"))
                or (
                    "No complete supported diffusion checkpoint was identified"
                    if pipeline is None
                    else None
                )
                or (pipeline.get("local_unsupported_reason") if pipeline else None)
                or (pipeline.get("save_unsupported_reason") if pipeline else None)
            )
            if reason:
                conversion.update(available=False, reason=reason)
            quant_reason = reason
            if quant_reason is None and importlib.util.find_spec("mflux") is None:
                quant_reason = "mflux is not installed"
            if quant_reason is None and item["quantized"]:
                quant_reason = "Quantization requires a full-precision source; stored quantization cannot be replaced"
            quantization = {
                "available": quant_reason is None,
                "adapter": "mflux",
                "requires_conversion": item["format"] != "mflux",
                "reason": quant_reason,
                "bits": pipeline["quantization_bits"] if pipeline else [],
            }
        if item.get("blocker"):
            conversion["available"] = False
            conversion["reason"] = item["blocker"]
        target = conversion.get("target_format")
        base = (
            re.sub(
                r"[^a-z0-9._-]+", "-", str(item["name"]).strip(), flags=re.IGNORECASE
            )
            .strip("-")
            .lower()
        )
        conversion["output_name"] = f"{base or 'model'}-{target}" if target else None
        item["conversion"] = conversion
        item["quantization"] = quantization
        item.pop("blocker", None)
        item.pop("oq", None)

    return sorted(
        catalog.values(), key=lambda item: (str(item["name"]).lower(), item["path"])
    )


def derive_policy(
    model: dict[str, Any], stored: dict[str, Any] | None
) -> dict[str, Any]:
    if stored:
        return stored
    settings = model.get("settings") or {}
    pinned = bool(model.get("pinned") or settings.get("is_pinned"))
    ttl = settings.get("ttl_seconds")
    if pinned:
        mode = "always_resident"
    elif ttl == 1:
        mode = "unload_after_request"
    elif isinstance(ttl, int) and ttl > 1:
        mode = "keep_warm"
    else:
        mode = "on_demand"
    return {"mode": mode, "ttl_seconds": ttl, "updated_at": None}


def build_registry_record(
    model: dict[str, Any],
    store: ModelControlStore,
    *,
    children: list[dict[str, Any]] | None = None,
    parents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    model_id = model["id"]
    source = source_metadata(model.get("model_path", ""), model.get("source_repo_id"))
    health = store.get("health", model_id, {"status": "unverified", "checked_at": None})
    if model.get("load_failed"):
        health = {
            **health,
            "status": "load_failed",
            "summary": model.get("load_failure_message") or "The last load failed",
        }
    if model.get("virtual"):
        kind = "virtual"
    elif model.get("unsupported_reason"):
        kind = "artifact"
    else:
        kind = "physical"
    if model.get("unsupported_reason"):
        health = {
            "status": "incompatible",
            "summary": model["unsupported_reason"],
            "checked_at": None,
        }
    return {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "id": model_id,
        "display_name": model.get("display_name") or model_id,
        "path": model.get("model_path"),
        "kind": kind,
        "engine": model.get("engine_type"),
        "model_type": model.get("model_type", "llm"),
        "config_model_type": model.get("config_model_type", ""),
        "preparation_modality": model.get("preparation_modality"),
        "preparation_blocker": model.get("preparation_blocker"),
        "capabilities": capabilities_for(
            model.get("model_type", "llm"), Path(model.get("model_path", ""))
        ),
        "source": source,
        "storage": {
            "estimated_bytes": int(model.get("estimated_size") or 0),
            "resident_bytes": int(
                model.get("resident_estimated_size")
                or model.get("actual_size")
                or model.get("estimated_size")
                or 0
            ),
            "actual_bytes": int(model.get("actual_size") or 0),
        },
        "runtime": {
            "loaded": bool(model.get("loaded")),
            "loading": bool(model.get("is_loading")),
            "pinned": bool(model.get("pinned")),
            "default": bool(model.get("is_default")),
            "hidden": bool(model.get("is_hidden")),
            "favorite": bool(model.get("is_favorite")),
            "helper": bool(model.get("is_helper")),
            "last_access": model.get("last_access"),
            "distributed": bool(model.get("distributed")),
            "cluster": model.get("cluster"),
        },
        "policy": derive_policy(model, store.get("policies", model_id)),
        "health": health,
        "update": store.get("updates", model_id, {"status": "unchecked"}),
        "lineage": {
            "parents": parents or [],
            "children": children or [],
            "profiles": model.get("exposed_profiles") or [],
        },
    }


def build_lineage(
    models: list[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result = {m["id"]: {"parents": [], "children": []} for m in models}
    by_reference: dict[str, str] = {}
    for model in models:
        for reference in (
            model.get("id"),
            model.get("model_path"),
            model.get("source_repo_id"),
        ):
            if reference:
                by_reference[str(reference)] = model["id"]
    reference_fields = {
        "specprefill_draft_model": "speculative-draft",
        "dflash_draft_model": "dflash-draft",
        "vlm_mtp_draft_model": "mtp-draft",
    }
    for model in models:
        source_model_id = model.get("source_model_id")
        if source_model_id in result and source_model_id != model["id"]:
            child = {"id": model["id"], "relation": "profile"}
            parent = {"id": source_model_id, "relation": "profile"}
            result[source_model_id]["children"].append(child)
            result[model["id"]]["parents"].append(parent)
        settings = model.get("settings") or {}
        for field, relation in reference_fields.items():
            target = by_reference.get(str(settings.get(field) or ""))
            if not target or target == model["id"] or target not in result:
                continue
            child = {"id": target, "relation": relation}
            parent = {"id": model["id"], "relation": relation}
            if child not in result[model["id"]]["children"]:
                result[model["id"]]["children"].append(child)
            if parent not in result[target]["parents"]:
                result[target]["parents"].append(parent)
        source = source_metadata(
            str(model.get("model_path") or ""), model.get("source_repo_id")
        )
        base_reference = (
            source.get("adapter_base_model")
            or source.get("base_model")
            or source.get("declared_source")
        )
        base = by_reference.get(str(base_reference or ""))
        if base and base != model["id"]:
            child = {"id": model["id"], "relation": "variant"}
            parent = {"id": base, "relation": "variant"}
            if child not in result[base]["children"]:
                result[base]["children"].append(child)
            if parent not in result[model["id"]]["parents"]:
                result[model["id"]]["parents"].append(parent)
    return result


def discover_unmanaged_artifacts(
    roots: list[str | Path], known_paths: set[str]
) -> list[dict[str, Any]]:
    """Find complete, unsupported, and incomplete artifacts omitted by serving."""
    from ..model_discovery import _resolve_hf_cache_entry

    def checkpoint_blocker(path: Path) -> str | None:
        indexes = sorted(path.rglob("*.safetensors.index.json"))
        missing: list[str] = []
        for index_path in indexes:
            index = _read_json(index_path)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict):
                return f"Invalid weight index: {index_path.name}"
            for shard in {str(value) for value in weight_map.values()}:
                if not (index_path.parent / shard).is_file():
                    missing.append(shard)
        if missing:
            return f"Checkpoint is incomplete: {len(set(missing))} weight shard(s) are missing"
        if not any(path.rglob("*.safetensors")):
            return "No safetensors weights were found"
        return None

    artifacts: list[dict[str, Any]] = []
    resolved_known = {str(Path(path).resolve()) for path in known_paths if path}
    seen: set[str] = set()
    for configured_root in roots:
        root = Path(configured_root).expanduser()
        if not root.is_dir():
            continue
        candidates: list[tuple[Path, str | None]] = []
        try:
            for child in root.iterdir():
                if not child.is_dir() or child.name.startswith("."):
                    continue
                if child.name.startswith("models--"):
                    resolved_cache = _resolve_hf_cache_entry(child)
                    if resolved_cache is not None:
                        candidates.append(
                            (
                                resolved_cache.snapshot_path,
                                resolved_cache.source_repo_id,
                            )
                        )
                    continue
                candidates.append((child, None))
                candidates.extend(
                    (grandchild, None)
                    for grandchild in child.iterdir()
                    if grandchild.is_dir() and not grandchild.name.startswith(".")
                )
        except OSError:
            continue
        for path, source_repo_id in candidates:
            resolved = str(path.resolve())
            if resolved in seen or resolved in resolved_known:
                continue
            seen.add(resolved)
            adapter = _read_json(path / "adapter_config.json")
            has_descriptor = any(
                (path / name).is_file()
                for name in ("config.json", "model_index.json", "omlx-mflux.json")
            )
            if not adapter and not has_descriptor:
                continue
            try:
                relative = path.relative_to(root)
            except ValueError:
                relative = Path(path.name)
            if adapter:
                model_type = "adapter"
                modality = "other"
                config_model_type = str(
                    adapter.get("peft_type") or "peft_adapter"
                ).lower()
                blocker = None
                reason = "LoRA/PEFT adapters are stored but this oMLX engine cannot load them"
                prefix = "adapter"
            else:
                model_type, modality = _classify_preparation_path(path)
                config = _read_json(path / "config.json")
                config_model_type = str(config.get("model_type") or "")
                blocker = checkpoint_blocker(path)
                reason = blocker or (
                    "This checkpoint is stored locally but no compatible oMLX serving engine discovered it"
                )
                prefix = "artifact"
            artifacts.append(
                {
                    "id": f"{prefix}:{str(relative).replace(os.sep, '/')}",
                    "display_name": source_repo_id or path.name,
                    "model_path": str(path),
                    "model_type": model_type,
                    "engine_type": "unsupported",
                    "config_model_type": config_model_type,
                    "preparation_modality": modality,
                    "source_type": "huggingface" if source_repo_id else "local",
                    "source_repo_id": source_repo_id,
                    "estimated_size": directory_usage(path)["logical_bytes"],
                    "resident_estimated_size": 0,
                    "actual_size": 0,
                    "loaded": False,
                    "is_loading": False,
                    "pinned": False,
                    "settings": {},
                    "preparation_blocker": blocker,
                    "unsupported_reason": reason,
                }
            )
    return artifacts


def directory_usage(path: str | Path) -> dict[str, int]:
    root = Path(path)
    logical = allocated = files = broken_links = 0
    physical = 0
    seen_inodes: set[tuple[int, int]] = set()
    if not root.exists():
        return {
            "logical_bytes": 0,
            "allocated_bytes": 0,
            "physical_bytes": 0,
            "file_count": 0,
            "broken_links": 0,
        }
    for item in root.rglob("*"):
        if item.is_symlink() and not item.exists():
            broken_links += 1
            continue
        try:
            if not item.is_file():
                continue
            stat = item.stat()
        except OSError:
            continue
        files += 1
        logical += stat.st_size
        allocated += int(getattr(stat, "st_blocks", 0)) * 512
        inode = (stat.st_dev, stat.st_ino)
        if inode not in seen_inodes:
            physical += stat.st_size
            seen_inodes.add(inode)
    return {
        "logical_bytes": logical,
        "allocated_bytes": allocated,
        "physical_bytes": physical,
        "file_count": files,
        "broken_links": broken_links,
    }


def verify_model_files(
    model: dict[str, Any], *, checksum_small_files: bool = True
) -> dict[str, Any]:
    if model.get("virtual") or str(model.get("model_path", "")).startswith(
        "builtin://"
    ):
        return {
            "status": "ready",
            "summary": "Built-in model is available",
            "checks": [{"name": "builtin", "status": "passed"}],
            "checked_at": _utcnow(),
        }

    path = Path(model.get("model_path") or "")
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append(
            {"name": name, "status": "passed" if ok else "failed", "detail": detail}
        )
        if not ok:
            errors.append(detail)

    record(
        "directory",
        path.is_dir(),
        f"Model directory {'exists' if path.is_dir() else 'is missing'}: {path}",
    )
    if not path.is_dir():
        return {
            "status": "corrupt",
            "summary": errors[0],
            "checks": checks,
            "errors": errors,
            "warnings": warnings,
            "checked_at": _utcnow(),
        }

    config_path = path / "config.json"
    index_path = path / "model_index.json"
    has_descriptor = config_path.is_file() or index_path.is_file()
    record(
        "descriptor",
        has_descriptor,
        "Found config.json or model_index.json"
        if has_descriptor
        else "No config.json or model_index.json found",
    )
    config = (
        _read_json(config_path) if config_path.is_file() else _read_json(index_path)
    )
    if has_descriptor and not config:
        record("descriptor_json", False, "Model descriptor is not valid JSON")

    index_files = sorted(path.rglob("*.safetensors.index.json"))
    missing_shards: list[str] = []
    indexed_shards = 0
    for index_file in index_files:
        index = _read_json(index_file)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            missing_shards.append(f"Invalid weight map: {index_file.relative_to(path)}")
            continue
        shards = {str(value) for value in weight_map.values()}
        indexed_shards += len(shards)
        for shard in shards:
            if not (index_file.parent / shard).is_file():
                missing_shards.append(
                    str((index_file.parent / shard).relative_to(path))
                )
    record(
        "weight_indexes",
        not missing_shards,
        f"Verified {indexed_shards} indexed weight shards"
        if not missing_shards
        else f"Missing weight shards: {', '.join(missing_shards[:8])}",
    )

    weight_files = list(path.rglob("*.safetensors"))
    record(
        "weights",
        bool(weight_files),
        f"Found {len(weight_files)} safetensors file(s)"
        if weight_files
        else "No safetensors weights found",
    )
    broken = [
        str(item.relative_to(path))
        for item in path.rglob("*")
        if item.is_symlink() and not item.exists()
    ]
    record(
        "links",
        not broken,
        "No broken links" if not broken else f"Broken links: {', '.join(broken[:8])}",
    )

    dependency = capabilities_for(model.get("model_type", "llm")).get("dependency")
    module_name = {
        "mlx-lm": "mlx_lm",
        "mlx-vlm": "mlx_vlm",
        "mlx-embeddings": "mlx_embeddings",
        "mlx-audio": "mlx_audio",
        "mflux": "mflux",
        "markitdown": "markitdown",
    }.get(dependency or "")
    if module_name:
        installed = importlib.util.find_spec(module_name) is not None
        checks.append(
            {
                "name": "dependency",
                "status": "passed" if installed else "failed",
                "detail": f"{dependency} is {'installed' if installed else 'not installed'}",
            }
        )
        if not installed:
            errors.append(f"Required package is not installed: {dependency}")

    checksums: dict[str, str] = {}
    if checksum_small_files:
        for item in (
            config_path,
            index_path,
            path / "tokenizer_config.json",
            path / "generation_config.json",
        ):
            try:
                if item.is_file() and item.stat().st_size <= 16 * 1024 * 1024:
                    checksums[str(item.relative_to(path))] = hashlib.sha256(
                        item.read_bytes()
                    ).hexdigest()
            except OSError:
                warnings.append(f"Could not checksum {item.name}")

    usage = directory_usage(path)
    status = "ready" if not errors else "corrupt"
    return {
        "status": status,
        "summary": "All file and dependency checks passed"
        if status == "ready"
        else errors[0],
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "checksums": checksums,
        "storage": usage,
        "checked_at": _utcnow(),
    }


def plan_resources(
    model_ids: list[str],
    pool_status: dict[str, Any],
    *,
    in_use: dict[str, int] | None = None,
) -> dict[str, Any]:
    in_use = in_use or {}
    models = {model["id"]: model for model in pool_status.get("models", [])}
    unknown = [model_id for model_id in model_ids if model_id not in models]
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}")
    targets = [models[model_id] for model_id in dict.fromkeys(model_ids)]
    ceiling = int(pool_status.get("final_ceiling") or 0)
    current = int(pool_status.get("current_model_memory") or 0)
    additional = sum(
        int(model.get("resident_estimated_size") or model.get("estimated_size") or 0)
        for model in targets
        if not model.get("loaded")
    )
    required_free = max(0, current + additional - ceiling) if ceiling else 0
    target_ids = {model["id"] for model in targets}
    candidates = sorted(
        (
            model
            for model in models.values()
            if model.get("loaded")
            and model["id"] not in target_ids
            and not model.get("pinned")
            and not model.get("is_loading")
            and not in_use.get(model["id"], 0)
        ),
        key=lambda model: model.get("last_access") or 0,
    )
    victims: list[dict[str, Any]] = []
    freed = 0
    for model in candidates:
        if freed >= required_free:
            break
        size = int(
            model.get("actual_size")
            or model.get("resident_estimated_size")
            or model.get("estimated_size")
            or 0
        )
        victims.append({"id": model["id"], "bytes": size})
        freed += size
    fits = not ceiling or current + additional - freed <= ceiling
    return {
        "model_ids": list(dict.fromkeys(model_ids)),
        "ceiling_bytes": ceiling,
        "current_bytes": current,
        "additional_bytes": additional,
        "required_free_bytes": required_free,
        "planned_free_bytes": freed,
        "fits": fits,
        "evictions": victims,
        "steps": [
            *[{"action": "unload", "model_id": victim["id"]} for victim in victims],
            *[
                {
                    "action": "reuse" if model.get("loaded") else "load",
                    "model_id": model["id"],
                }
                for model in targets
            ],
        ],
    }


_EXTERNAL_STATUS = {
    "pending": "queued",
    "loading": "running",
    "downloading": "running",
    "quantizing": "running",
    "saving": "running",
    "uploading": "running",
    "completed": "succeeded",
    "failed": "failed",
    "cancelled": "cancelled",
}


class ModelControl:
    """Durable operation and registry coordinator."""

    def __init__(self, base_path: str | Path):
        self.store = ModelControlStore(base_path)
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self.storage_cache: tuple[float, list[dict[str, Any]]] | None = None

    def invalidate_storage_cache(self) -> None:
        self.storage_cache = None

    def start_operation(
        self,
        kind: str,
        runner: Callable[[str], Awaitable[dict[str, Any]]],
        *,
        model_id: str | None = None,
        payload: dict[str, Any] | None = None,
        cancellable: bool = True,
    ) -> dict[str, Any]:
        operation_id = uuid.uuid4().hex
        operation = {
            "id": operation_id,
            "kind": kind,
            "source": "control",
            "model_id": model_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "payload": payload or {},
            "result": None,
            "error": None,
            "cancellable": cancellable,
            "created_at": _utcnow(),
            "started_at": None,
            "finished_at": None,
        }
        self.store.put("operations", operation_id, operation)

        async def execute() -> None:
            self.update_operation(
                operation_id,
                status="running",
                stage="running",
                progress=5.0,
                started_at=_utcnow(),
            )
            try:
                result = await runner(operation_id)
            except asyncio.CancelledError:
                self.update_operation(
                    operation_id,
                    status="cancelled",
                    stage="cancelled",
                    finished_at=_utcnow(),
                )
                raise
            except Exception as exc:  # noqa: BLE001 - operation error belongs in history
                self.update_operation(
                    operation_id,
                    status="failed",
                    stage="failed",
                    error=str(exc),
                    finished_at=_utcnow(),
                )
            else:
                self.update_operation(
                    operation_id,
                    status="succeeded",
                    stage="complete",
                    progress=100.0,
                    result=result,
                    finished_at=_utcnow(),
                )
            finally:
                self._tasks.pop(operation_id, None)

        self._tasks[operation_id] = asyncio.create_task(execute())
        return operation

    def update_operation(self, operation_id: str, **updates: Any) -> dict[str, Any]:
        operation = self.store.get("operations", operation_id)
        if operation is None:
            raise KeyError(operation_id)
        operation.update(updates)
        self.store.put("operations", operation_id, operation)
        return operation

    def cancel_operation(self, operation_id: str) -> bool:
        operation = self.store.get("operations", operation_id)
        if not operation or operation.get("cancellable", True) is False:
            return False
        task = self._tasks.get(operation_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    def sync_external(self, source: str, tasks: list[dict[str, Any]]) -> None:
        updates: dict[str, dict[str, Any]] = {}
        for raw in tasks:
            raw_id = str(raw.get("task_id") or raw.get("id") or "")
            if not raw_id:
                continue
            operation_id = f"{source}:{raw_id}"
            status = _EXTERNAL_STATUS.get(
                str(raw.get("status")), str(raw.get("status") or "unknown")
            )
            operation = {
                "id": operation_id,
                "raw_id": raw_id,
                "kind": source,
                "source": source,
                "model_id": raw.get("model_name")
                or raw.get("repo_id")
                or raw.get("model_id"),
                "status": status,
                "stage": raw.get("phase") or raw.get("status") or status,
                "progress": float(raw.get("progress") or 0.0),
                "payload": raw,
                "result": raw if status == "succeeded" else None,
                "error": raw.get("error") or None,
                "created_at": raw.get("created_at"),
                "started_at": raw.get("started_at"),
                "finished_at": raw.get("completed_at"),
            }
            updates[operation_id] = operation
        self.store.merge("operations", updates)

    def operations(self, *, limit: int = 200) -> list[dict[str, Any]]:
        values = list(self.store.section("operations").values())

        def created_at(item: dict[str, Any]) -> float:
            value = item.get("created_at")
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value).timestamp()
                except ValueError:
                    return 0.0
            return 0.0

        values.sort(key=created_at, reverse=True)
        return values[:limit]

    def remove_operation(self, operation_id: str) -> bool:
        operation = self.store.get("operations", operation_id)
        if operation and operation.get("status") in _ACTIVE_OPERATION_STATES:
            return False
        return self.store.remove("operations", operation_id)
