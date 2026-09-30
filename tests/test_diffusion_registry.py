"""Contract tests use tiny local artifacts and fake native models, never weights downloads."""

import importlib
import inspect
import json
import struct
import subprocess
import sys
from unittest.mock import Mock

import pytest

from omlx.diffusion import (
    MODEL_IDENTITIES,
    PIPELINES,
    Checkpoint,
    ImageTask,
    MFluxBackend,
    detect_checkpoint,
    get_pipeline,
    read_quantization,
    resolve_base_model,
    validate_checkpoint,
    validate_task,
)


def artifact(tmp_path, base="z-image-turbo", bits=None):
    spec = get_pipeline(base)
    for component in spec.components:
        root = tmp_path / component
        root.mkdir(parents=True, exist_ok=True)
        header = json.dumps(
            {
                "__metadata__": {
                    "mflux_version": "0.20.0",
                    "quantization_level": str(bits),
                },
                "test.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
            }
        ).encode()
        (root / "0.safetensors").write_bytes(
            struct.pack("<Q", len(header)) + header + b"\0" * 4
        )
    for component in spec.tokenizers:
        root = tmp_path / component
        root.mkdir(parents=True, exist_ok=True)
        (root / "tokenizer.json").write_text("{}")
    return Checkpoint(tmp_path, base, bits, "mflux", spec.components)


def test_registry_import_does_not_load_optional_backend():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import omlx.diffusion; assert not any(n == 'mflux' or n.startswith('mflux.') for n in sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_acquisition_preserves_remote_checkpoint_identity():
    from huggingface_hub.utils import filter_repo_objects

    from omlx.diffusion import download_patterns

    selected = list(
        filter_repo_objects(
            ["omlx-mflux.json", "transformer/0.safetensors", "unrelated.bin"],
            allow_patterns=download_patterns("flux2-klein-4b"),
        )
    )
    assert selected == ["omlx-mflux.json", "transformer/0.safetensors"]


def test_registry_covers_installed_canonical_models():
    pytest.importorskip("mflux")
    from mflux.models.common.config.model_config import AVAILABLE_MODELS

    assert set(MODEL_IDENTITIES) == set(AVAILABLE_MODELS)
    assert {spec.base_model for spec in PIPELINES.values()} == set(AVAILABLE_MODELS)
    for key, config in AVAILABLE_MODELS.items():
        for alias in config.aliases:
            assert resolve_base_model(alias) == key
        assert get_pipeline(key).operation in (
            "txt2img",
            "reference-edit",
            "inpaint",
            "controlnet",
            "upscale",
        )


def test_ambiguous_repository_needs_identity():
    with pytest.raises(ValueError, match="unambiguous"):
        resolve_base_model("black-forest-labs/FLUX.1-dev")
    with pytest.raises(ValueError):
        resolve_base_model("/tmp/looks-like-qwen-image-but-is-not")
    assert resolve_base_model("klein-base-4B") == "flux2-klein-base-4b"


@pytest.mark.parametrize("key", list(PIPELINES))
def test_translated_calls_match_installed_native_api(key):
    pytest.importorskip("mflux")
    spec = get_pipeline(key)
    module, name = spec.backend_class.rsplit(".", 1)
    cls = getattr(importlib.import_module(module), name)
    images = tuple("input.png" for _ in range(spec.image_min))
    options = (
        {"control_types": ["canny"] * len(images)}
        if spec.image_argument == "controls"
        else {}
    )
    prompt = (
        None
        if spec.operation == "upscale"
        else json.dumps({"edit_instruction": "add a tree"})
        if spec.prompt_format.startswith("json")
        else "a tree"
    )
    task = ImageTask(
        prompt=prompt,
        image_paths=images,
        mask_path="mask.png" if spec.operation == "inpaint" else None,
        options=options,
    )
    kwargs = validate_task(spec, task)
    if spec.image_argument == "controls":
        kwargs.pop("control_types")
    inspect.signature(cls.generate_image).bind(None, **kwargs)
    inspect.signature(cls).bind(
        model_config=object(), model_path="/local", quantize=None
    )


@pytest.mark.parametrize(
    "pipeline",
    ["dev", "schnell", "z-image-turbo", "boogu-image-turbo", "flux2-klein-4b"],
)
def test_unsupported_explicit_empty_negative_rejected(pipeline):
    with pytest.raises(ValueError, match="negative_prompt"):
        validate_task(
            get_pipeline(pipeline), ImageTask(prompt="tree", negative_prompt="")
        )


def test_required_media_and_unknown_options_fail_before_backend():
    with pytest.raises(ValueError, match="requires"):
        validate_task(get_pipeline("qwen-image-edit"), ImageTask(prompt="edit"))
    with pytest.raises(ValueError, match="options"):
        validate_task(
            get_pipeline("qwen-image"),
            ImageTask(prompt="tree", options={"made_up": True}),
        )
    with pytest.raises(ValueError, match="mask"):
        validate_task(
            get_pipeline("dev-fill"), ImageTask(prompt="edit", image_paths=("a.png",))
        )


def test_img2img_has_effective_image_strength():
    kwargs = validate_task(
        get_pipeline("z-image-turbo/img2img"),
        ImageTask(prompt="tree", image_paths=("a.png",)),
    )
    assert kwargs["image_path"] == "a.png"
    assert kwargs["image_strength"] == 0.4


def test_distilled_and_base_defaults_differ():
    distilled = get_pipeline("flux2-klein-4b")
    base = get_pipeline("flux2-klein-base-4b")
    assert distilled.default_steps == 4
    assert base.default_steps == 50
    with pytest.raises(ValueError, match="requires guidance"):
        validate_task(distilled, ImageTask(prompt="tree", guidance=3.0))
    assert (
        validate_task(base, ImageTask(prompt="tree", guidance=3.0))["guidance"] == 3.0
    )
    assert "use_kv_cache" not in get_pipeline("flux2-klein-4b/edit").options
    assert "use_kv_cache" in get_pipeline("flux2-klein-9b-kv/edit").options


def test_negative_prompt_requires_effective_cfg():
    with pytest.raises(ValueError, match="guidance"):
        validate_task(
            get_pipeline("qwen-image-2.1"),
            ImageTask(prompt="tree", negative_prompt="blur"),
        )
    assert (
        validate_task(
            get_pipeline("qwen-image-2.1"),
            ImageTask(prompt="tree", negative_prompt="blur", guidance=2.0),
        )["negative_prompt"]
        == "blur"
    )


def test_ideogram_preserves_preset_schedule():
    kwargs = validate_task(
        get_pipeline("ideogram-4-fp8"),
        ImageTask(prompt="tree", options={"preset": "V4_TURBO_12"}),
    )
    assert "guidance" not in kwargs and "num_inference_steps" not in kwargs
    with pytest.raises(ValueError, match="explicit steps"):
        validate_task(
            get_pipeline("ideogram-4-fp8"), ImageTask(prompt="tree", guidance=7.0)
        )


def test_upscale_rejects_ignored_generation_options():
    with pytest.raises(ValueError, match="accepts no prompt"):
        validate_task(
            get_pipeline("seedvr2-3b"),
            ImageTask(prompt="tree", image_paths=("input.png",)),
        )
    assert validate_task(
        get_pipeline("seedvr2-3b"), ImageTask(image_paths=("input.png",))
    ) == {"seed": 0, "resolution": 384, "image_path": "input.png"}


def test_fake_generation_records_only_valid_translation():
    model = Mock()
    model.generate_image.return_value = "output"
    backend = MFluxBackend()
    result = backend.generate(
        model, ImageTask(prompt="tree", seed=42, steps=9), "z-image-turbo"
    )
    assert result == "output"
    model.generate_image.assert_called_once_with(
        seed=42, prompt="tree", num_inference_steps=9, width=1024, height=1024
    )
    with pytest.raises(ValueError):
        backend.generate(
            model, ImageTask(prompt="tree", options={"guidance": 10}), "z-image-turbo"
        )
    assert model.generate_image.call_count == 1


def test_checkpoint_manifest_roundtrip_and_legacy_migration(tmp_path):
    cp = artifact(tmp_path, bits=4)
    backend = MFluxBackend()
    model = Mock(bits=4)
    backend.save(model, tmp_path, base_model=cp.base_model)
    found = detect_checkpoint(tmp_path)
    assert found.quantization == 4 and found.default_pipeline == "z-image-turbo"
    assert found.metadata()["base_model"] == "z-image-turbo"
    (tmp_path / "omlx-mflux.json").write_text(
        json.dumps(
            {
                "version": 1,
                "backend": "mflux",
                "model_family": "z-image-turbo",
                "quantize": None,
                "converted_quantization_bits": 4,
            }
        )
    )
    assert detect_checkpoint(tmp_path).quantization == 4


@pytest.mark.parametrize(
    "changes",
    [
        {"version": 99},
        {"base_model": "future-model"},
        {"quantization_bits": True},
        {"components": ["transformer"]},
        {"quantization_bits": None},
    ],
)
def test_invalid_manifest_never_falls_back_to_directory_name(tmp_path, changes):
    path = tmp_path / "z-image-turbo-4bit"
    artifact(path, bits=4)
    manifest = {
        "version": 2,
        "backend": "mflux",
        "base_model": "z-image-turbo",
        "pipeline_id": "z-image-turbo",
        "format": "mflux",
        "quantization_bits": 4,
        "components": list(get_pipeline("z-image-turbo").components),
    }
    manifest.update(changes)
    (path / "omlx-mflux.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        detect_checkpoint(path)


def test_incomplete_shards_and_tokenizer_reject(tmp_path):
    cp = artifact(tmp_path)
    (tmp_path / "transformer" / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "missing.safetensors"}})
    )
    with pytest.raises(ValueError, match="shard"):
        validate_checkpoint(cp)
    (tmp_path / "transformer" / "model.safetensors.index.json").unlink()
    (tmp_path / "tokenizer" / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="tokenizer"):
        validate_checkpoint(cp)


def test_precision_manifest_cannot_relabel_weights(tmp_path):
    cp = artifact(tmp_path)
    from dataclasses import replace

    with pytest.raises(ValueError, match="quantization"):
        validate_checkpoint(replace(cp, quantization=4))
    assert read_quantization(tmp_path) is None


def test_offline_load_does_not_acquire_components(tmp_path):
    artifact(tmp_path)
    from omlx.diffusion.backend import _offline

    pytest.importorskip("mflux")
    from mflux.models.common.resolution.path_resolution import PathResolution
    from mflux.models.common.weights.loading.weight_loader import WeightLoader

    with _offline():
        assert PathResolution.resolve(str(tmp_path)) == tmp_path
        with pytest.raises(ValueError, match="prepared component"):
            PathResolution.resolve("org/missing")
        with pytest.raises(ValueError, match="download"):
            WeightLoader._download_from_url("https://invalid", "aux")


def test_fibo_lite_rejects_ignored_cfg_controls():
    for pipeline in ("fibo-lite", "fibo-lite/img2img"):
        spec = get_pipeline(pipeline)
        assert spec.fixed_guidance == 1.0 and not spec.supports_negative_prompt
        images = ("a.png",) if spec.image_min else ()
        with pytest.raises(ValueError, match="requires guidance"):
            validate_task(
                spec, ImageTask(prompt="{}", image_paths=images, guidance=4.0)
            )
        with pytest.raises(ValueError, match="negative_prompt"):
            validate_task(
                spec, ImageTask(prompt="{}", image_paths=images, negative_prompt="")
            )


def test_upscaler_rejects_ignored_guidance():
    with pytest.raises(ValueError, match="does not support guidance"):
        validate_task(
            get_pipeline("dev-controlnet-upscaler"),
            ImageTask(prompt="tree", image_paths=("a.png",), guidance=3.5),
        )


def test_pid_options_reject_before_auxiliary_acquisition():
    for spec in PIPELINES.values():
        assert (
            "pid_decode" not in spec.options and "pid_degrade_sigma" not in spec.options
        )
    for options in (
        {"pid_decode": True},
        {"pid_degrade_sigma": 0.3},
        {"pid_decode": True, "pid_degrade_sigma": 0.8},
    ):
        with pytest.raises(ValueError, match="does not support options"):
            validate_task(
                get_pipeline("z-image-turbo"), ImageTask(prompt="tree", options=options)
            )


@pytest.mark.parametrize("name", ["fibo", "fibo-lite", "fibo-edit", "fibo-edit-rmbg"])
def test_fibo_prompt_preflight_matches_json_backend(name):
    spec = get_pipeline(name)
    images = ("input.png",) if spec.image_min else ()
    with pytest.raises(ValueError, match="JSON object"):
        validate_task(spec, ImageTask(prompt="a tree", image_paths=images))
    prompt = json.dumps({"edit_instruction": "add a tree"})
    assert (
        validate_task(spec, ImageTask(prompt=prompt, image_paths=images))["prompt"]
        == prompt
    )
    if spec.prompt_format == "json-edit":
        with pytest.raises(ValueError, match="edit_instruction"):
            validate_task(spec, ImageTask(prompt="{}", image_paths=images))


def test_packed_weights_without_precision_metadata_reject(tmp_path):
    artifact(tmp_path)
    header = json.dumps(
        {
            "linear.weight": {"dtype": "U32", "shape": [1], "data_offsets": [0, 4]},
            "linear.scales": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
            "linear.biases": {"dtype": "F32", "shape": [1], "data_offsets": [8, 12]},
        }
    ).encode()
    (tmp_path / "transformer" / "0.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\0" * 12
    )
    with pytest.raises(ValueError, match="lack stored quantization metadata"):
        read_quantization(tmp_path)


def test_qwen21_guidance_requires_effective_true_cfg():
    spec = get_pipeline("qwen-image-2.1")
    for task in (
        ImageTask(prompt="tree", guidance=4.0),
        ImageTask(prompt="tree", guidance=0.5),
        ImageTask(prompt="tree", guidance=4.0, negative_prompt=""),
    ):
        with pytest.raises(ValueError, match="guidance|CFG"):
            validate_task(spec, task)
    assert (
        validate_task(spec, ImageTask(prompt="tree", guidance=1.0))["guidance"] == 1.0
    )
    assert (
        validate_task(
            spec, ImageTask(prompt="tree", guidance=4.0, negative_prompt="blur")
        )["guidance"]
        == 4.0
    )


@pytest.mark.parametrize("mode", ["depth", "hed", "pose"])
def test_union_rejects_modes_with_unprepared_auxiliary_models(mode):
    spec = get_pipeline("z-image-turbo-controlnet-union-2.1")
    assert spec.metadata()["supported_control_types"] == ["canny", "mlsd"]
    with pytest.raises(
        ValueError, match="explicit local auxiliary preprocessor preparation"
    ):
        validate_task(
            spec,
            ImageTask(
                prompt="tree",
                image_paths=("input.png",),
                options={"control_types": [mode]},
            ),
        )


@pytest.mark.parametrize("mode", ["canny", "mlsd"])
def test_union_accepts_modes_without_auxiliary_models(mode):
    spec = get_pipeline("z-image-turbo-controlnet-union-2.1")
    assert validate_task(
        spec,
        ImageTask(
            prompt="tree", image_paths=("input.png",), options={"control_types": [mode]}
        ),
    )["control_types"] == [mode]
    assert spec.metadata()["defaults_source"] == "oMLX defaults informed by mflux APIs"


def test_legacy_cache_identity_survives_symlink_alias(tmp_path):
    cache = (
        tmp_path
        / "models--filipstrand--Z-Image-Turbo-mflux-4bit"
        / "snapshots"
        / "arbitrary-commit"
    )
    artifact(cache, bits=4)
    alias = tmp_path / "friendly-local-name"
    alias.symlink_to(cache, target_is_directory=True)
    cp = detect_checkpoint(alias)
    assert cp.path == cache.resolve()
    assert cp.base_model == "z-image-turbo" and cp.quantization == 4


def test_dev_depth_declares_missing_local_preprocessor_and_rejects_before_construction(
    tmp_path, monkeypatch
):
    from omlx.diffusion import backend as backend_module

    spec = get_pipeline("dev-depth")
    metadata = spec.metadata()
    assert not metadata["serving_supported"]
    assert not metadata["preparation_supported"]
    assert not metadata["download_supported"]
    assert "DepthPro" in metadata["local_unsupported_reason"]
    checkpoint = artifact(tmp_path, base="dev-depth")
    native_symbol = Mock(
        side_effect=AssertionError("Native class must not be imported or constructed")
    )
    monkeypatch.setattr(backend_module, "_symbol", native_symbol)
    with pytest.raises(ValueError, match="local DepthPro preparation"):
        MFluxBackend().load(checkpoint)
    native_symbol.assert_not_called()


def test_catvton_rejects_ordinary_inpainting_and_incomplete_acquisition(
    tmp_path, monkeypatch
):
    from omlx.diffusion import backend as backend_module
    from omlx.diffusion import download_patterns

    spec = get_pipeline("dev-fill-catvton")
    metadata = spec.metadata()
    assert not metadata["serving_supported"]
    assert not metadata["preparation_supported"]
    assert not metadata["download_supported"]
    assert "two-image" in metadata["local_unsupported_reason"]
    checkpoint = artifact(tmp_path, base="dev-fill-catvton")
    native_symbol = Mock(
        side_effect=AssertionError("CatVTON must not load an ordinary fill model")
    )
    monkeypatch.setattr(backend_module, "_symbol", native_symbol)
    with pytest.raises(ValueError, match="virtual try-on adapter"):
        MFluxBackend().load(checkpoint)
    with pytest.raises(ValueError, match="custom transformer acquisition"):
        download_patterns("dev-fill-catvton")
    native_symbol.assert_not_called()


def test_krea_download_filters_skip_duplicate_turbo_layout():
    from huggingface_hub.utils import filter_repo_objects

    from omlx.diffusion import download_patterns

    files = [
        "turbo.safetensors",
        "raw.safetensors",
        "0.safetensors",
        "12.safetensors",
        "model.safetensors.index.json",
        "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "transformer/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "text_encoder/model.safetensors",
        "text_encoder/config.json",
        "tokenizer/tokenizer.json",
        "unrelated/0.safetensors",
        "unrelated/model.safetensors.index.json",
    ]
    turbo = set(filter_repo_objects(files, allow_patterns=download_patterns("krea-2")))
    assert turbo == {
        "turbo.safetensors",
        "0.safetensors",
        "12.safetensors",
        "model.safetensors.index.json",
        "vae/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "text_encoder/model.safetensors",
        "text_encoder/config.json",
        "tokenizer/tokenizer.json",
    }
    raw = set(
        filter_repo_objects(files, allow_patterns=download_patterns("krea-2-raw"))
    )
    assert "raw.safetensors" in raw and "turbo.safetensors" not in raw
    assert "transformer/diffusion_pytorch_model-00001-of-00003.safetensors" in raw
    assert "transformer/diffusion_pytorch_model.safetensors.index.json" in raw
    assert {"0.safetensors", "12.safetensors", "model.safetensors.index.json"} <= raw
    assert not any(name.startswith("unrelated/") for name in raw)


def test_backend_save_does_not_recursively_copy_nested_source_staging(tmp_path):
    source = tmp_path / "source"
    artifact(source)
    (source / "source-only.json").write_text("{}")
    destination = source / ".hidden-staging"
    artifact(destination)
    (destination / "prepared-only.json").write_text("{}")
    spec = get_pipeline("z-image-turbo")
    backend = MFluxBackend()
    model = Mock(bits=None)
    backend._models[id(model)] = (spec, source)
    backend.save(model, destination)
    model.save_model.assert_called_once_with(str(destination))
    assert not (destination / "source-only.json").exists()
    assert not (destination / ".hidden-staging").exists()
    assert (destination / "prepared-only.json").is_file()
    assert detect_checkpoint(destination).base_model == "z-image-turbo"
