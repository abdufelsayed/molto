"""Local identity and completeness checks, independent of optional inference packages."""

import json
import re
import struct
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from omlx_runtime.diffusion.registry import (
    get_pipeline,
    pipelines_for_model,
    resolve_base_model,
)

MANIFEST_NAME = "omlx-mflux.json"


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    base_model: str
    quantization: int | None = None
    format: str = "huggingface"
    components: tuple[str, ...] = ()
    pipeline_id: str | None = None

    @property
    def default_pipeline(self):
        return self.pipeline_id or self.base_model

    def metadata(self):
        return model_capabilities(self)


def _json(path):
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid diffusion metadata {path.name}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"{path.name} must contain an object")
    return obj


def _bits(value):
    if value in (None, "None", "null"):
        return None
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if type(value) is not int or value not in (3, 4, 5, 6, 8):
        raise ValueError("Invalid diffusion quantization bits")
    return value


def read_quantization(path: Path) -> int | None:
    """Stored precision only; never infer it from the folder name."""
    path = Path(path)
    found = set()
    packed = False
    for index in path.rglob("*.safetensors.index.json"):
        index_data = _json(index)
        metadata = index_data.get("metadata", {})
        packed = packed or any(
            key.endswith((".scales", ".biases"))
            for key in index_data.get("weight_map", {})
        )
        if "quantization_level" in metadata:
            found.add(_bits(metadata["quantization_level"]))
    for weight in path.rglob("*.safetensors"):
        try:
            with weight.open("rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                if size > 16 * 1024 * 1024:
                    continue
                header = json.loads(f.read(size))
        except (OSError, ValueError, struct.error):
            continue
        packed = packed or any(key.endswith((".scales", ".biases")) for key in header)
        metadata = header.get("__metadata__", {})
        if "quantization_level" in metadata:
            found.add(_bits(metadata["quantization_level"]))
    # mflux leaves VAE full precision. The global level describes quantized components.
    found.discard(None)
    if packed and not found:
        raise ValueError(
            "Packed affine diffusion weights lack stored quantization metadata"
        )
    if len(found) > 1:
        raise ValueError(
            "Conflicting stored quantization levels across diffusion components"
        )
    return next(iter(found), None)


def validate_checkpoint(checkpoint: Checkpoint) -> Checkpoint:
    path = Path(checkpoint.path)
    spec = get_pipeline(checkpoint.default_pipeline)
    if spec.base_model != checkpoint.base_model:
        raise ValueError("Manifest pipeline and base_model disagree")
    if not path.is_dir():
        raise ValueError("Diffusion checkpoint must be a local directory")
    for subdir in spec.components:
        directory = path / subdir
        # Krea Raw uses transformer/ whereas Turbo uses a root-level single file.
        if (
            subdir == "."
            and spec.base_model == "krea-2-raw"
            and (path / "transformer").is_dir()
        ):
            directory = path / "transformer"
        if not any(
            p.is_file() and p.stat().st_size > 0
            for p in directory.glob("*.safetensors")
        ):
            raise ValueError(
                f"Incomplete {spec.base_model} checkpoint: missing weights for {subdir}"
            )
    if (
        spec.base_model == "z-image-turbo-controlnet-union-2.1"
        and not (path / "controlnet" / "config.json").is_file()
    ):
        raise ValueError(
            "Incomplete ZImage ControlNet checkpoint: missing controlnet/config.json"
        )
    if spec.base_model in ("seedvr2-3b", "seedvr2-7b"):
        variant = "7b" if spec.base_model == "seedvr2-7b" else "3b"
        for file in (
            f"seedvr2_ema_{variant}_fp16.safetensors",
            "ema_vae_fp16.safetensors",
        ):
            if not (path / file).is_file():
                raise ValueError(f"Incomplete SeedVR2 checkpoint: missing {file}")
    for subdir in spec.tokenizers:
        directory = path / subdir
        if not directory.is_dir() or not any(
            (directory / name).is_file() and (directory / name).stat().st_size > 0
            for name in (
                "tokenizer.json",
                "tokenizer.model",
                "spiece.model",
                "vocab.json",
                "vocab.txt",
            )
        ):
            raise ValueError(
                f"Incomplete checkpoint: missing tokenizer content in {subdir}"
            )
    for index in path.rglob("*.safetensors.index.json"):
        weight_map = _json(index).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid shard index: {index.name}")
        for filename in weight_map.values():
            if (
                not isinstance(filename, str)
                or Path(filename).is_absolute()
                or ".." in Path(filename).parts
                or not (index.parent / filename).is_file()
            ):
                raise ValueError(
                    f"Incomplete shard index: missing or invalid {filename!r}"
                )
    actual = read_quantization(path)
    if checkpoint.format == "mflux" and checkpoint.quantization != actual:
        raise ValueError("Manifest and stored quantization disagree")
    if (
        actual is not None
        and checkpoint.quantization is not None
        and actual != checkpoint.quantization
    ):
        raise ValueError("Manifest and stored quantization disagree")
    return checkpoint


def detect_checkpoint(path: Path) -> Checkpoint | None:
    path = Path(path).expanduser().resolve()
    manifest = path / MANIFEST_NAME
    if manifest.exists():
        data = _json(manifest)
        version = data.get("version", data.get("schema_version"))
        if (
            type(version) is not int
            or version not in (1, 2)
            or data.get("backend") != "mflux"
        ):
            raise ValueError("Unsupported diffusion manifest backend or version")
        if version == 1:
            # Only the historic ZImageTurbo schema is accepted as a migration.
            if (
                data.get("model_family") != "z-image-turbo"
                or data.get("quantize") is not None
            ):
                raise ValueError("Invalid legacy mflux manifest")
            base = "z-image-turbo"
            bits = _bits(data.get("converted_quantization_bits"))
            if bits is None:
                bits = read_quantization(path)
            pipeline = base
            components = get_pipeline(base).components
        else:
            base = resolve_base_model(data.get("base_model", ""))
            if "quantization_bits" not in data:
                raise ValueError("Manifest must declare stored quantization_bits")
            bits = _bits(data["quantization_bits"])
            pipeline = data.get("pipeline_id", base)
            spec = get_pipeline(pipeline)
            components = data.get("components")
            if not isinstance(components, list) or components != list(spec.components):
                raise ValueError(
                    "Manifest must declare the complete pipeline component list"
                )
            components = tuple(components)
            if data.get("format") != "mflux":
                raise ValueError("Unsupported diffusion checkpoint format")
        return validate_checkpoint(
            Checkpoint(path, base, bits, "mflux", components, pipeline)
        )
    # Identity comes from explicit source config or pipeline class, never path substrings.
    index = path / "model_index.json"
    if not index.is_file():
        # Historic mflux checkpoints predate identity manifests. Accept the exact
        # known ZImage naming convention only after complete component validation.
        names = [path.name]
        if path.parent.name == "snapshots":
            names.append(
                path.parent.parent.name.removeprefix("models--")
                .replace("--", "/")
                .split("/")[-1]
            )
        for name in names:
            if re.fullmatch(r"z-image-turbo(?:-mflux)?-(?:3|4|5|6|8)bit", name, re.I):
                spec = get_pipeline("z-image-turbo")
                return validate_checkpoint(
                    Checkpoint(
                        path,
                        spec.base_model,
                        read_quantization(path),
                        "mflux",
                        spec.components,
                    )
                )
        return None
    data = _json(index)
    hint = data.get("base_model") or data.get("_name_or_path")
    base = None
    if hint:
        with suppress(ValueError):
            base = resolve_base_model(hint)
    classname = data.get("_class_name")
    class_models = {
        "ZImageTurboPipeline": "z-image-turbo",
        "QwenImagePipeline": "qwen-image",
        "QwenImageEditPipeline": "qwen-image-edit",
        "QwenImageEditPlusPipeline": "qwen-image-edit",
        "FluxKontextPipeline": "dev-kontext",
        "FluxFillPipeline": "dev-fill",
        "FluxDepthPipeline": "dev-depth",
        "FIBOPipeline": "fibo",
        "FIBOEditPipeline": "fibo-edit",
        "ErnieImagePipeline": "ernie-image",
    }
    if base is None:
        # Exact canonical identity hints are allowed; ambiguous source repositories
        # such as FLUX.1-dev plus its ControlNet adapters require base_model.
        names = [path.name]
        if path.parent.name == "snapshots":
            names.append(
                path.parent.parent.name.removeprefix("models--").replace("--", "/")
            )
        for name in names:
            try:
                base = resolve_base_model(name)
                break
            except ValueError:
                pass
    if base is None:
        base = class_models.get(classname)
    # A ZImagePipeline class alone cannot distinguish distilled and base ZImage.
    if classname == "ZImagePipeline" and not hint and base is None:
        return None
    if base is None:
        return None
    spec = get_pipeline(base)
    return validate_checkpoint(
        Checkpoint(path, base, read_quantization(path), "huggingface", spec.components)
    )


def model_capabilities(value: Path | Checkpoint) -> dict:
    checkpoint = (
        value if isinstance(value, Checkpoint) else detect_checkpoint(Path(value))
    )
    if checkpoint is None:
        raise ValueError("Not a supported diffusion checkpoint")
    return {
        "backend": "mflux",
        "base_model": checkpoint.base_model,
        "default_pipeline": checkpoint.default_pipeline,
        "format": checkpoint.format,
        "quantization_bits": checkpoint.quantization,
        "pipelines": [
            spec.metadata() for spec in pipelines_for_model(checkpoint.base_model)
        ],
    }
