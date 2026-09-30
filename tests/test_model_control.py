# SPDX-License-Identifier: Apache-2.0
"""Tests for model-control services."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from omlx.services.model_control import (
    ModelControl,
    ModelControlStore,
    artifact_characteristics,
    build_lineage,
    build_preparation_catalog,
    build_registry_record,
    capabilities_for,
    directory_usage,
    discover_unmanaged_artifacts,
    plan_resources,
    source_metadata,
    verify_model_files,
)


def _model(path: Path, **overrides):
    value = {
        "id": "org--model",
        "display_name": "Model",
        "model_path": str(path),
        "model_type": "llm",
        "engine_type": "batched",
        "config_model_type": "llama",
        "source_repo_id": "org/model",
        "estimated_size": 100,
        "actual_size": 0,
        "loaded": False,
        "is_loading": False,
        "pinned": False,
        "settings": {},
    }
    value.update(overrides)
    return value


def test_capabilities_cover_every_engine_model_type():
    expected = {
        "llm": "/v1/chat/completions",
        "vlm": "/v1/responses",
        "embedding": "/v1/embeddings",
        "reranker": "/v1/rerank",
        "audio_stt": "/v1/audio/transcriptions",
        "audio_tts": "/v1/audio/speech",
        "audio_sts": "/v1/audio/speech-to-speech",
        "image_generation": "/v1/images/generations",
        "markitdown": "/v1/responses",
    }
    for model_type, endpoint in expected.items():
        assert endpoint in capabilities_for(model_type)["endpoints"]


def test_control_store_persists_and_marks_abandoned_operation(tmp_path):
    store = ModelControlStore(tmp_path)
    store.put("policies", "model", {"mode": "keep_warm"})
    store.put("operations", "op", {"id": "op", "status": "running"})

    restored = ModelControlStore(tmp_path)

    assert restored.get("policies", "model") == {"mode": "keep_warm"}
    assert restored.get("operations", "op")["status"] == "interrupted"


def test_hf_source_metadata_lists_active_and_rollback_revisions(tmp_path):
    cache = tmp_path / "models--org--model"
    active = cache / "snapshots" / "aaaaaaaa"
    previous = cache / "snapshots" / "bbbbbbbb"
    active.mkdir(parents=True)
    previous.mkdir()
    (active / "config.json").write_text(json.dumps({"model_type": "llama"}))

    source = source_metadata(str(active), "org/model")

    assert source["revision"] == "aaaaaaaa"
    assert {item["revision"] for item in source["cached_revisions"]} == {
        "aaaaaaaa",
        "bbbbbbbb",
    }
    assert (
        next(item for item in source["cached_revisions"] if item["active"])["revision"]
        == "aaaaaaaa"
    )


def test_mflux_source_metadata_exposes_compatible_model_family(tmp_path):
    (tmp_path / "omlx-mflux.json").write_text(
        json.dumps({"backend": "mflux", "model_family": "z-image-turbo"})
    )

    source = source_metadata(str(tmp_path), None)

    assert source["model_family"] == "z-image-turbo"


def test_artifact_characteristics_detect_mflux_quantization_metadata(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file

    component = tmp_path / "transformer"
    component.mkdir()
    save_file(
        {"weight": np.zeros((1,), dtype=np.float32)},
        component / "0.safetensors",
        metadata={"mflux_version": "0.13.0", "quantization_level": "4"},
    )

    identity = artifact_characteristics(tmp_path)

    assert identity == {
        "format": "mflux",
        "precision": "4-bit",
        "quantized": True,
        "quantization_bits": 4,
    }


def test_image_capabilities_follow_checkpoint_operations(tmp_path):
    import struct

    from omlx.diffusion import get_pipeline

    spec = get_pipeline("flux2-klein-4b")
    for component in spec.components:
        directory = tmp_path / component
        directory.mkdir()
        header = json.dumps({"__metadata__": {"quantization_level": "4"}}).encode()
        (directory / "0.safetensors").write_bytes(
            struct.pack("<Q", len(header)) + header
        )
    (tmp_path / "tokenizer").mkdir()
    (tmp_path / "tokenizer" / "tokenizer.json").write_text("{}")
    (tmp_path / "omlx-mflux.json").write_text(
        json.dumps(
            {
                "version": 2,
                "backend": "mflux",
                "base_model": spec.base_model,
                "format": "mflux",
                "components": list(spec.components),
                "quantization_bits": 4,
            }
        )
    )

    capabilities = capabilities_for("image_generation", tmp_path)

    assert set(capabilities["tasks"]) == {"txt2img", "img2img", "reference-edit"}
    assert "/v1/images/edits" in capabilities["endpoints"]
    assert {pipeline["base_model"] for pipeline in capabilities["pipelines"]} == {
        spec.base_model
    }
    source = source_metadata(str(tmp_path), None)
    assert source["diffusion"]["quantization_bits"] == 4
    assert source["model_family"] == "flux2-klein-4b"


def test_incomplete_image_checkpoint_has_no_advertised_operations(tmp_path):
    (tmp_path / "model_index.json").write_text('{"_class_name": "UnsupportedPipeline"}')
    assert capabilities_for("image_generation", tmp_path)["tasks"] == []


def test_preparation_catalog_keeps_every_model_in_one_inventory(tmp_path):
    text_path = tmp_path / "text"
    image_path = tmp_path / "image"
    audio_path = tmp_path / "audio"
    raw_audio_path = tmp_path / "raw-audio"
    for path in (text_path, image_path, audio_path, raw_audio_path):
        path.mkdir()
    (raw_audio_path / "config.json").write_text(
        json.dumps({"model_type": "wav2vec2", "architectures": ["Wav2Vec2ForCTC"]})
    )
    records = [
        {
            "id": "text",
            "display_name": "Text model",
            "path": str(text_path),
            "kind": "physical",
            "model_type": "llm",
            "config_model_type": "llama",
            "source": {
                "format": "huggingface",
                "precision": "BFLOAT16",
                "quantized": False,
            },
            "storage": {"estimated_bytes": 100},
        },
        {
            "id": "image",
            "display_name": "Image model",
            "path": str(image_path),
            "kind": "physical",
            "model_type": "image_generation",
            "config_model_type": "z_image_turbo",
            "source": {
                "format": "huggingface",
                "precision": "FLOAT16",
                "quantized": False,
            },
            "storage": {"estimated_bytes": 200},
        },
        {
            "id": "audio",
            "display_name": "Audio model",
            "path": str(audio_path),
            "kind": "physical",
            "model_type": "audio_stt",
            "config_model_type": "wav2vec2",
            "source": {"format": "mlx", "precision": "4-bit", "quantized": True},
            "storage": {"estimated_bytes": 300},
        },
    ]
    oq_text = {
        "name": "Text model",
        "path": str(text_path),
        "model_type": "llama",
        "format": "huggingface",
        "precision": "BFLOAT16",
        "is_quantized": False,
        "conversion_required": True,
        "size": 100,
        "size_formatted": "100 B",
        "is_vlm": False,
    }
    oq_audio = {
        "name": "Raw audio",
        "path": str(raw_audio_path),
        "model_type": "wav2vec2",
        "format": "huggingface",
        "precision": "FLOAT32",
        "is_quantized": False,
        "conversion_required": True,
        "size": 400,
        "size_formatted": "400 B",
        "is_vlm": False,
    }

    with patch(
        "omlx.services.model_control.importlib.util.find_spec", return_value=object()
    ):
        catalog = build_preparation_catalog(
            records, [oq_text, oq_audio], [oq_text, oq_audio]
        )

    assert {model["id"] for model in catalog} == {
        "text",
        "image",
        "audio",
        "Raw audio",
    }
    text_model = next(model for model in catalog if model["id"] == "text")
    image_model = next(model for model in catalog if model["id"] == "image")
    audio_model = next(model for model in catalog if model["id"] == "audio")
    assert text_model["conversion"]["adapter"] == "mlx-lm"
    assert text_model["quantization"] == {
        "available": True,
        "adapter": "oQ",
        "requires_conversion": True,
        "reason": None,
    }
    assert image_model["conversion"]["adapter"] == "mflux"
    assert image_model["quantization"]["available"] is False
    assert (
        "complete supported diffusion checkpoint"
        in image_model["quantization"]["reason"]
    )
    assert image_model["quantization"]["adapter"] == "mflux"
    assert audio_model["conversion"]["available"] is False
    assert "already in MLX" in audio_model["conversion"]["reason"]
    assert "full-precision" in audio_model["quantization"]["reason"]
    raw_audio = next(model for model in catalog if model["id"] == "Raw audio")
    assert raw_audio["model_type"] == "other"
    assert raw_audio["modality"] == "audio"
    assert raw_audio["conversion"]["available"] is False
    assert "No converter is registered" in raw_audio["conversion"]["reason"]


def test_lineage_resolves_draft_by_path_and_repo(tmp_path):
    base = _model(
        tmp_path / "base",
        id="base",
        settings={"dflash_draft_model": "org/draft"},
    )
    draft = _model(
        tmp_path / "draft",
        id="draft",
        source_repo_id="org/draft",
        is_helper=True,
    )

    lineage = build_lineage([base, draft])

    assert lineage["base"]["children"] == [{"id": "draft", "relation": "dflash-draft"}]
    assert lineage["draft"]["parents"] == [{"id": "base", "relation": "dflash-draft"}]


def test_unmanaged_lora_adapter_is_visible_as_incompatible_artifact(tmp_path):
    base_path = tmp_path / "base"
    adapter_path = tmp_path / "sql-adapter"
    base_path.mkdir()
    adapter_path.mkdir()
    (adapter_path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": "org/base"})
    )
    (adapter_path / "adapter_model.safetensors").write_bytes(b"adapter")

    artifacts = discover_unmanaged_artifacts([tmp_path], {str(base_path)})
    control = ModelControl(tmp_path / "state")
    record = build_registry_record(artifacts[0], control.store)

    assert record["kind"] == "artifact"
    assert record["model_type"] == "adapter"
    assert record["health"]["status"] == "incompatible"
    assert record["source"]["adapter_base_model"] == "org/base"


def test_incomplete_model_stays_visible_with_operation_blocker(tmp_path):
    model = tmp_path / "incomplete"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]})
    )
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": "missing.safetensors"}})
    )

    [artifact] = discover_unmanaged_artifacts([tmp_path], set())
    control = ModelControl(tmp_path / "state")
    record = build_registry_record(artifact, control.store)
    with patch(
        "omlx.services.model_control.importlib.util.find_spec", return_value=object()
    ):
        [prepared] = build_preparation_catalog([record], [], [])

    assert record["kind"] == "artifact"
    assert "incomplete" in record["health"]["summary"].lower()
    assert prepared["conversion"]["available"] is False
    assert prepared["quantization"]["available"] is False
    assert "weight shard" in prepared["conversion"]["reason"]


def test_directory_usage_deduplicates_hardlinked_physical_bytes(tmp_path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"x" * 1024)
    second.hardlink_to(first)

    usage = directory_usage(tmp_path)

    assert usage["logical_bytes"] == 2048
    assert usage["physical_bytes"] == 1024
    assert usage["file_count"] == 2


def test_verify_model_detects_missing_indexed_shard(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": "missing.safetensors"}})
    )

    report = verify_model_files(_model(tmp_path))

    assert report["status"] == "corrupt"
    assert any("Missing weight shards" in error for error in report["errors"])
    assert report["checksums"]["config.json"]


def test_registry_combines_capabilities_policy_health_and_revision(tmp_path):
    snapshot = tmp_path / "models--org--model" / "snapshots" / "deadbeef"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps({"model_type": "llama", "quantization_config": {"bits": 4}})
    )
    control = ModelControl(tmp_path / "state")
    control.store.put(
        "health",
        "org--model",
        {"status": "ready", "checked_at": "now"},
    )

    record = build_registry_record(_model(snapshot), control.store)

    assert record["capabilities"]["tasks"] == ["chat", "text-generation"]
    assert record["health"]["status"] == "ready"
    assert record["policy"]["mode"] == "on_demand"
    assert record["source"]["revision"] == "deadbeef"
    assert record["source"]["quantization"] == {"bits": 4}


def test_resource_plan_uses_lru_and_preserves_pinned_models():
    status = {
        "final_ceiling": 100,
        "current_model_memory": 80,
        "models": [
            {
                "id": "target",
                "loaded": False,
                "estimated_size": 50,
                "resident_estimated_size": 50,
            },
            {
                "id": "old",
                "loaded": True,
                "actual_size": 40,
                "pinned": False,
                "last_access": 1,
            },
            {
                "id": "pinned",
                "loaded": True,
                "actual_size": 40,
                "pinned": True,
                "last_access": 0,
            },
        ],
    }

    plan = plan_resources(["target"], status)

    assert plan["fits"] is True
    assert plan["required_free_bytes"] == 30
    assert plan["evictions"] == [{"id": "old", "bytes": 40}]


@pytest.mark.asyncio
async def test_control_operation_records_success(tmp_path):
    control = ModelControl(tmp_path)

    async def run(operation_id):
        control.update_operation(operation_id, stage="working", progress=50)
        return {"ok": True}

    operation = control.start_operation("verify", run, model_id="model")
    await control._tasks[operation["id"]]

    saved = control.store.get("operations", operation["id"])
    assert saved["status"] == "succeeded"
    assert saved["result"] == {"ok": True}
    assert saved["progress"] == 100.0
