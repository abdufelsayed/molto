"""Preparation is transactional and uses the same identity as image serving."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omlx_runtime.mflux_conversion import convert_mflux_model
from omlx_runtime.model_discovery import detect_model_type, discover_models


@pytest.fixture
def preparation_backend(monkeypatch, tmp_path):
    import omlx_runtime.diffusion as diffusion

    state = SimpleNamespace(
        loads=[], saves=[], fail=False, bits=8, rewrite_tokenizer=False
    )
    spec = SimpleNamespace(
        id="qwen-image",
        base_model="qwen-image",
        save_supported=True,
        quantization_bits=(3, 4, 5, 6, 8),
    )

    def detect(path):
        path = Path(path)
        if (path / "omlx-mflux.json").exists():
            data = json.loads((path / "omlx-mflux.json").read_text())
            if data.get("backend") != "mflux":
                raise ValueError("Invalid explicit manifest")
            return SimpleNamespace(
                base_model=data["base_model"],
                quantization=data.get("quantization_bits"),
                default_pipeline=data.get("pipeline_id", data["base_model"]),
            )
        return None

    class Backend:
        def instantiate(self, base_model, **kwargs):
            state.loads.append((base_model, kwargs))
            return SimpleNamespace(bits=state.bits)

        def save(self, model, path, **kwargs):
            state.saves.append(kwargs)
            if state.rewrite_tokenizer:
                (Path(path) / "tokenizer" / "tokenizer.json").write_text(
                    '{"resaved": true}'
                )
            (Path(path) / "weights.safetensors").write_bytes(b"weights")
            if state.fail:
                raise RuntimeError("save failed")
            (Path(path) / "omlx-mflux.json").write_text(
                json.dumps(
                    {
                        "version": 2,
                        "backend": "mflux",
                        "base_model": "qwen-image",
                        "pipeline_id": "qwen-image",
                        "quantization_bits": model.bits,
                    }
                )
            )

    def resolve(name):
        if name not in {"qwen-image", "qwen", "Qwen/Qwen-Image"}:
            raise ValueError("Pass --base-model for an unknown checkpoint")
        return "qwen-image"

    monkeypatch.setattr(diffusion, "MFluxBackend", Backend)
    monkeypatch.setattr(diffusion, "detect_checkpoint", detect)
    monkeypatch.setattr(diffusion, "get_pipeline", lambda name: spec)
    monkeypatch.setattr(diffusion, "resolve_base_model", resolve)
    monkeypatch.setattr(
        diffusion,
        "read_quantization",
        lambda path: state.bits if (Path(path) / "stored").exists() else None,
    )
    monkeypatch.setattr(
        diffusion, "download_patterns", lambda model: ["transformer/*", "tokenizer/*"]
    )
    import huggingface_hub

    cache = tmp_path / "cache"
    cache.mkdir()
    state.cache = cache
    state.downloads = []

    def snapshot(**kwargs):
        state.downloads.append(kwargs)
        return str(cache)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    return state


def test_prepares_registry_alias_and_reports_actual_precision(
    tmp_path, preparation_backend
):
    result = convert_mflux_model("qwen", tmp_path / "saved", 8)
    assert preparation_backend.loads == [
        (
            "qwen-image",
            {
                "model_path": str(preparation_backend.cache),
                "quantization": 8,
                "pipeline_id": "qwen-image",
            },
        )
    ]
    assert result["manifest"]["quantization_bits"] == 8
    assert (tmp_path / "saved" / "omlx-mflux.json").is_file()


def test_third_party_requires_identity_before_output_mutation(
    tmp_path, preparation_backend
):
    with pytest.raises(ValueError, match="base-model"):
        convert_mflux_model("someone/arbitrary-model", tmp_path / "saved")
    assert not (tmp_path / "saved").exists()
    assert not preparation_backend.loads


def test_failure_cleans_hidden_staging_and_preserves_empty_output(
    tmp_path, preparation_backend
):
    preparation_backend.fail = True
    output = tmp_path / "saved"
    output.mkdir()
    with pytest.raises(RuntimeError, match="save failed"):
        convert_mflux_model("qwen-image", output)
    assert output.is_dir() and not list(output.iterdir())
    assert set(tmp_path.iterdir()) == {output, preparation_backend.cache}


def test_existing_output_never_loaded_or_overwritten(tmp_path, preparation_backend):
    output = tmp_path / "saved"
    output.mkdir()
    original = output / "keep"
    original.write_text("original")
    with pytest.raises(ValueError, match="not empty"):
        convert_mflux_model("qwen-image", output)
    assert original.read_text() == "original"
    assert not preparation_backend.loads


def test_stored_quantization_conflict_fails_before_mutation(
    tmp_path, preparation_backend
):
    source = tmp_path / "custom"
    source.mkdir()
    (source / "stored").touch()
    preparation_backend.bits = 4
    with pytest.raises(ValueError, match="already quantized"):
        convert_mflux_model(str(source), tmp_path / "saved", 8, base_model="qwen")
    assert not preparation_backend.loads
    assert not (tmp_path / "saved").exists()


def test_conversion_preserves_configs_and_tokenizers_without_original_indexes(
    tmp_path, preparation_backend
):
    source = tmp_path / "custom"
    (source / "transformer").mkdir(parents=True)
    (source / "transformer" / "config.json").write_text('{"geometry": 1}')
    (source / "transformer" / "model.safetensors.index.json").write_text("{}")
    (source / "tokenizer").mkdir()
    (source / "tokenizer" / "merges.txt").write_text("tokens")
    output = tmp_path / "saved"
    convert_mflux_model(str(source), output, base_model="qwen")
    assert (output / "transformer" / "config.json").read_text() == '{"geometry": 1}'
    assert not (output / "transformer" / "model.safetensors.index.json").exists()
    assert (output / "tokenizer" / "merges.txt").read_text() == "tokens"
    assert (source / "transformer" / "model.safetensors.index.json").is_file()


def test_unsupported_original_is_image_artifact_and_never_served(tmp_path):
    source = tmp_path / "unsupported"
    source.mkdir()
    (source / "model_index.json").write_text('{"_class_name": "UnsupportedPipeline"}')
    (source / "config.json").write_text('{"model_type": "llama"}')
    (source / "model.safetensors").write_bytes(b"weights")
    assert detect_model_type(source) == "image_generation"
    assert not discover_models(tmp_path)


def test_invalid_manifest_never_falls_back_to_text(tmp_path):
    source = tmp_path / "bad"
    source.mkdir()
    (source / "omlx-mflux.json").write_text('{"backend": "wrong"}')
    (source / "config.json").write_text('{"model_type": "llama"}')
    (source / "model.safetensors").write_bytes(b"weights")
    assert detect_model_type(source) == "image_generation"
    assert not discover_models(tmp_path)


def _complete_layout(path, base_model):
    from omlx_runtime.diffusion import get_pipeline

    spec = get_pipeline(base_model)
    for component in spec.components:
        directory = path / component
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "0.safetensors").write_bytes(b"weights")
    for tokenizer in spec.tokenizers:
        directory = path / tokenizer
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tokenizer.json").write_text("{}")
    return spec


def test_legacy_v1_manifest_is_discovered_without_name_hint(tmp_path):
    source = tmp_path / "old-conversion"
    _complete_layout(source, "z-image-turbo")
    (source / "omlx-mflux.json").write_text(
        json.dumps(
            {
                "version": 1,
                "backend": "mflux",
                "model_family": "z-image-turbo",
                "quantize": None,
            }
        )
    )
    model = discover_models(tmp_path)[source.name]
    assert model.config_model_type == "z_image_turbo"


def test_cached_filipstrand_zimage_remains_discoverable(tmp_path):
    entry = tmp_path / "models--filipstrand--Z-Image-Turbo-mflux-4bit"
    source = entry / "snapshots" / "arbitrary-commit"
    _complete_layout(source, "z-image-turbo")
    model = discover_models(tmp_path)["filipstrand--Z-Image-Turbo-mflux-4bit"]
    assert model.source_repo_id == "filipstrand/Z-Image-Turbo-mflux-4bit"
    assert model.model_type == "image_generation"


def test_original_flux2_checkpoint_uses_identity_without_directory_hint(tmp_path):
    source = tmp_path / "arbitrary-original"
    _complete_layout(source, "flux2-klein-4b")
    (source / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "Flux2KleinPipeline",
                "_name_or_path": "black-forest-labs/FLUX.2-klein-4B",
            }
        )
    )
    model = discover_models(tmp_path)[source.name]
    assert model.config_model_type == "flux2_klein_4b"


def test_prepared_qwen_without_root_config_uses_manifest_identity(tmp_path):
    source = tmp_path / "arbitrary-prepared"
    spec = _complete_layout(source, "qwen-image")
    (source / "omlx-mflux.json").write_text(
        json.dumps(
            {
                "version": 2,
                "backend": "mflux",
                "base_model": "qwen-image",
                "pipeline_id": "qwen-image",
                "format": "mflux",
                "components": list(spec.components),
                "quantization_bits": None,
            }
        )
    )
    model = discover_models(tmp_path)[source.name]
    assert model.config_model_type == "qwen_image"


def test_incomplete_original_is_not_discovered(tmp_path):
    source = tmp_path / "incomplete"
    source.mkdir()
    (source / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "Flux2KleinPipeline",
                "_name_or_path": "black-forest-labs/FLUX.2-klein-4B",
            }
        )
    )
    assert detect_model_type(source) == "image_generation"
    assert not discover_models(tmp_path)


def test_acquisition_is_bounded_and_passes_only_local_snapshot(
    tmp_path, preparation_backend
):
    convert_mflux_model(
        "someone/custom", tmp_path / "saved", base_model="qwen", revision="commit"
    )
    assert preparation_backend.downloads == [
        {
            "repo_id": "someone/custom",
            "revision": "commit",
            "allow_patterns": ["transformer/*", "tokenizer/*"],
        }
    ]
    assert preparation_backend.loads[0][1]["model_path"] == str(
        preparation_backend.cache
    )


def test_runtime_precision_conflict_never_saves_or_publishes(
    tmp_path, preparation_backend
):
    preparation_backend.bits = 4
    with pytest.raises(ValueError, match="instead of requested"):
        convert_mflux_model("qwen-image", tmp_path / "saved", 8)
    assert not preparation_backend.saves
    assert not (tmp_path / "saved").exists()


def test_no_quantize_reports_runtime_preserved_precision(tmp_path, preparation_backend):
    preparation_backend.bits = 4
    result = convert_mflux_model("qwen-image", tmp_path / "saved", None)
    assert result["manifest"]["quantization_bits"] == 4
    assert result["manifest"]["requested_quantization_bits"] is None


def test_readonly_cache_assets_can_be_resaved_without_changing_source(
    tmp_path, preparation_backend
):
    source = tmp_path / "source"
    tokenizer = source / "tokenizer" / "tokenizer.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text('{"original": true}')
    tokenizer.chmod(0o444)
    preparation_backend.rewrite_tokenizer = True

    convert_mflux_model(str(source), tmp_path / "saved", base_model="qwen-image")

    assert tokenizer.read_text() == '{"original": true}'
    assert tokenizer.stat().st_mode & 0o777 == 0o444
    assert json.loads(
        (tmp_path / "saved" / "tokenizer" / "tokenizer.json").read_text()
    ) == {"resaved": True}
