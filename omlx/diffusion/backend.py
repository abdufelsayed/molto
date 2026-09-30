"""Localized native-MLX adapters. Imports and all allocations occur on demand."""

import json
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

from .checkpoint import (
    MANIFEST_NAME,
    Checkpoint,
    read_quantization,
    validate_checkpoint,
)
from .registry import (
    ImageTask,
    get_pipeline,
    resolve_base_model,
    validate_task,
)


def _symbol(name):
    module, attr = name.rsplit(".", 1)
    return getattr(import_module(module), attr)


def download_patterns(base_model: str) -> list[str]:
    spec = get_pipeline(base_model)
    if spec.local_unsupported_reason:
        raise ValueError(spec.local_unsupported_reason)
    if spec.base_model in (
        "dev-redux",
        "dev-controlnet-canny",
        "schnell-controlnet-canny",
        "dev-controlnet-upscaler",
        "z-image-turbo-controlnet-union-2.1",
    ):
        raise ValueError(
            "This pipeline requires a complete local checkpoint containing its auxiliary weights; a single repository is insufficient"
        )
    if spec.base_model in ("krea-2", "krea-2-raw"):
        # Match the native Krea2WeightDefinition recipe without root wildcards:
        # HF uses fnmatch, where '*' also crosses directory separators. Turbo
        # repositories include a redundant ~26 GB transformer/ shard layout.
        shared = [
            "vae/*.safetensors",
            "vae/*.json",
            "text_encoder/*.safetensors",
            "text_encoder/*.json",
            "tokenizer/**",
            "added_tokens.json",
            "chat_template.jinja",
            "config.json",
            "omlx-mflux.json",
        ]
        prepared = ["[0-9]" * digits + ".safetensors" for digits in range(1, 7)]
        prepared.append("model.safetensors.index.json")
        if spec.base_model == "krea-2-raw":
            return [
                "raw.safetensors",
                "transformer/*.safetensors",
                "transformer/*.json",
                "model_index.json",
                *prepared,
                *shared,
            ]
        return ["turbo.safetensors", *prepared, *shared]
    patterns = ["model_index.json", "config.json", MANIFEST_NAME]
    for directory in spec.components:
        prefix = "" if directory == "." else directory + "/"
        patterns += [prefix + "*.safetensors", prefix + "*.json"]
    patterns += [directory + "/**" for directory in spec.tokenizers]
    return patterns


@contextmanager
def _offline():
    """Prevent backend fallback acquisition, including auxiliary components.

    mflux's local-path resolver otherwise falls back to remote repositories when a
    component is absent. Loading is serialized on oMLX's existing MLX executor.
    """
    resolution = import_module(
        "mflux.models.common.resolution.path_resolution"
    ).PathResolution
    loader = import_module(
        "mflux.models.common.weights.loading.weight_loader"
    ).WeightLoader

    def local(path, patterns=None):
        if path is None:
            return None
        root = Path(path).expanduser()
        if not root.exists():
            raise ValueError(
                f"Local diffusion load requires prepared component: {path}"
            )
        return root

    def no_url(*args, **kwargs):
        raise ValueError("Local diffusion loading cannot download auxiliary weights")

    # Tokenizers are always pointed at validated local directories. The hub guard
    # additionally catches backend-specific model/tokenizer fallback paths.
    with (
        patch.object(resolution, "resolve", side_effect=local),
        patch.object(loader, "_download_from_url", side_effect=no_url),
        patch("huggingface_hub.file_download.hf_hub_download", side_effect=no_url),
        patch("huggingface_hub.hf_hub_download", side_effect=no_url),
        patch("huggingface_hub.snapshot_download", side_effect=no_url),
    ):
        yield


class MFluxBackend:
    def __init__(self):
        self._models = {}

    def instantiate(
        self,
        base_model: str,
        model_path: str | None = None,
        quantization: int | None = None,
        pipeline_id: str | None = None,
    ):
        base_model = resolve_base_model(base_model)
        spec = get_pipeline(pipeline_id or base_model)
        if spec.base_model != base_model:
            raise ValueError("pipeline is incompatible with base_model")
        if quantization is not None and (
            type(quantization) is not int or quantization not in spec.quantization_bits
        ):
            raise ValueError("Unsupported quantization")
        if model_path is None or not Path(model_path).is_dir():
            raise ValueError(
                "Prepare a complete local checkpoint before constructing the diffusion model"
            )
        checkpoint = Checkpoint(
            Path(model_path),
            base_model,
            quantization,
            components=spec.components,
            pipeline_id=spec.id,
        )
        validate_checkpoint(checkpoint)
        if spec.local_unsupported_reason:
            raise ValueError(spec.local_unsupported_reason)
        model_class = _symbol(spec.backend_class)
        config_module = import_module("mflux.models.common.config.model_config")
        config = config_module.AVAILABLE_MODELS[base_model]
        if base_model == "ideogram-4-fp8":
            definition = _symbol(
                "mflux.models.ideogram4.weights.ideogram4_weight_definition.Ideogram4WeightDefinition"
            )
            config = definition.resolve_inference_config(
                str(model_path), base_model, str(model_path)
            )
        with _offline():
            model = model_class(
                model_config=config, model_path=str(model_path), quantize=quantization
            )
        self._models[id(model)] = (spec, Path(model_path))
        return model

    def load(self, checkpoint: Checkpoint, pipeline_id: str | None = None):
        validate_checkpoint(checkpoint)
        # Stored quantization is self-describing and must not be applied again.
        stored = read_quantization(checkpoint.path)
        requested = (
            None
            if stored is not None or checkpoint.format == "mflux"
            else checkpoint.quantization
        )
        return self.instantiate(
            checkpoint.base_model,
            str(checkpoint.path),
            requested,
            pipeline_id or checkpoint.default_pipeline,
        )

    def generate(self, model, task: ImageTask, pipeline_id: str | None = None):
        entry = self._models.get(id(model))
        if entry is None and pipeline_id is None:
            raise ValueError(
                "Pipeline identity is required for an externally supplied model"
            )
        spec = get_pipeline(pipeline_id) if pipeline_id else entry[0]
        if entry is not None and entry[0].id != spec.id:
            raise ValueError("Loaded pipeline differs from the requested pipeline")
        kwargs = validate_task(spec, task)
        if spec.image_argument == "controls":
            types = _symbol(
                "mflux.models.z_image.variants.controlnet.control_types.ControlType"
            )
            control = _symbol(
                "mflux.models.z_image.variants.controlnet.control_types.ControlSpec"
            )
            names = kwargs.pop("control_types")
            strengths = kwargs.pop("control_strengths", [1.0] * len(task.image_paths))
            kwargs["controls"] = [
                control(types(name), path, strength)
                for name, path, strength in zip(names, task.image_paths, strengths)
            ]
        return model.generate_image(**kwargs)

    def save(
        self,
        model,
        path: Path,
        pipeline_id: str | None = None,
        base_model: str | None = None,
    ):
        entry = self._models.get(id(model))
        spec = (
            get_pipeline(pipeline_id or base_model)
            if pipeline_id or base_model
            else entry[0]
            if entry
            else None
        )
        if spec is None:
            raise ValueError("Model identity is required for saving")
        if not spec.save_supported:
            raise ValueError(spec.save_unsupported_reason)
        if entry is not None and entry[0].base_model != spec.base_model:
            raise ValueError("Save identity differs from loaded model")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if (
            spec.backend_class
            == "mflux.models.flux2.variants.edit.flux2_klein_edit.Flux2KleinEdit"
        ):
            # The edit class owns identical weights but omits the saver method.
            saver = _symbol(
                "mflux.models.flux2.variants.txt2img.flux2_klein.Flux2Klein"
            )
            saver.save_model(model, str(path))
        else:
            model.save_model(str(path))
        bits = read_quantization(path)
        runtime_bits = getattr(model, "bits", None)
        if bits is None:
            bits = runtime_bits
        manifest = {
            "version": 2,
            "schema_version": 2,
            "backend": "mflux",
            "base_model": spec.base_model,
            "pipeline_id": spec.id,
            "format": "mflux",
            "quantization_bits": bits,
            "components": list(spec.components),
            "quantization_policy": {
                "bits": bits,
                "scope": "mflux component and layer predicates; not uniform precision",
            },
        }
        validate_checkpoint(
            Checkpoint(path, spec.base_model, bits, "mflux", spec.components, spec.id)
        )
        temporary = path / (MANIFEST_NAME + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path / MANIFEST_NAME)
        return manifest

    def release(self, model):
        self._models.pop(id(model), None)
