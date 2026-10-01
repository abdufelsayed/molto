"""Explicit, import-free mflux 0.20 pipeline capabilities.

Pipeline identity describes an operation, not just the architecture of its weights.
Backend symbols are strings so discovery does not import MLX or mflux.
"""

import json
import math
from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class ImageTask:
    prompt: str | None = None
    seed: int = 0
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    guidance: float | None = None
    negative_prompt: str | None = None
    image_paths: tuple[str, ...] = ()
    mask_path: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineSpec:
    id: str
    base_model: str
    operation: str
    backend_class: str
    components: tuple[str, ...]
    tokenizers: tuple[str, ...]
    default_steps: int | None = 4
    default_guidance: float | None = 4.0
    default_width: int | None = 1024
    default_height: int | None = 1024
    supports_negative_prompt: bool = False
    image_min: int = 0
    image_max: int = 0
    image_argument: str = "image_path"
    mask_argument: str | None = None
    options: tuple[str, ...] = ()
    save_unsupported_reason: str | None = None
    local_unsupported_reason: str | None = None
    quantization_bits: tuple[int, ...] = (3, 4, 5, 6, 8)
    fixed_guidance: float | None = None
    prompt_format: str = "text"
    supported_control_types: tuple[str, ...] = ()

    @property
    def supports_mask(self):
        return self.mask_argument is not None

    @property
    def save_supported(self):
        return self.save_unsupported_reason is None

    @property
    def max_batch_size(self):
        return (
            4
            if self.backend_class
            in {
                "mflux.models.flux2.variants.txt2img.flux2_klein.Flux2Klein",
                "mflux.models.flux2.variants.edit.flux2_klein_edit.Flux2KleinEdit",
            }
            and self.operation in {"txt2img", "reference-edit"}
            else 1
        )

    @property
    def supported_parameters(self):
        params = ["seed"]
        if self.operation != "upscale":
            params += ["prompt", "width", "height", "steps"]
        if self.default_guidance is not None:
            params.append("guidance")
        if self.supports_negative_prompt:
            params.append("negative_prompt")
        if self.image_max:
            params.append("image_paths")
        if self.supports_mask:
            params.append("mask_path")
        return tuple(params) + self.options

    def metadata(self):
        return dict(
            id=self.id,
            pipeline=self.id,
            base_model=self.base_model,
            operation=self.operation,
            prompt_format=self.prompt_format,
            supported_parameters=list(self.supported_parameters),
            options=list(self.options),
            unsupported_options={
                "pid_decode": "PiD auxiliary decoder and text encoder local preparation is not integrated",
                "pid_degrade_sigma": "PiD local preparation is not integrated",
            },
            performance={
                "batching": {
                    "max_size": self.max_batch_size,
                    "requires_matching_prompt_settings_and_images": True,
                },
                "reference_kv_cache": self.base_model == "flux2-klein-9b-kv"
                and self.operation == "reference-edit",
                "reference_kv_cache_reason": None
                if self.base_model == "flux2-klein-9b-kv"
                and self.operation == "reference-edit"
                else "Requires the FLUX.2 Klein 9B-KV checkpoint and reference-edit pipeline",
            },
            image_min=self.image_min,
            image_max=self.image_max,
            supports_mask=self.supports_mask,
            defaults_source="oMLX defaults informed by mflux APIs",
            supported_control_types=list(self.supported_control_types),
            unsupported_control_types={
                name: "Requires explicit local auxiliary preprocessor preparation"
                for name in ("depth", "hed", "pose")
            }
            if self.image_argument == "controls"
            else {},
            default_steps=self.default_steps,
            default_guidance=self.default_guidance,
            default_width=self.default_width,
            default_height=self.default_height,
            save_supported=self.save_supported,
            save_unsupported_reason=self.save_unsupported_reason,
            local_unsupported_reason=self.local_unsupported_reason,
            quantization_bits=list(self.quantization_bits),
            fixed_guidance=self.fixed_guidance,
            serving_supported=self.local_unsupported_reason is None,
            preparation_supported=self.save_supported
            and self.local_unsupported_reason is None,
            download_supported=self.local_unsupported_reason is None
            and self.base_model not in AUXILIARY_ACQUISITION_GAPS,
            download_unsupported_reason=self.local_unsupported_reason
            or AUXILIARY_ACQUISITION_GAPS.get(self.base_model),
        )


AUXILIARY_ACQUISITION_GAPS = {
    name: "Requires complete local auxiliary components; single-repository acquisition is insufficient"
    for name in (
        "dev-redux",
        "dev-controlnet-canny",
        "schnell-controlnet-canny",
        "dev-controlnet-upscaler",
        "z-image-turbo-controlnet-union-2.1",
    )
}

PIPELINES: dict[str, PipelineSpec] = {}


def _add(
    names,
    symbol,
    *,
    operation="txt2img",
    components=("vae", "transformer", "text_encoder"),
    tokenizers=("tokenizer",),
    steps=4,
    guidance=4.0,
    negative=False,
    images=0,
    image_argument="image_path",
    mask=None,
    options=(),
    save_reason=None,
    local_reason=None,
    img2img=False,
    control_types=(),
):
    for name in names.split():
        spec = PipelineSpec(
            name,
            name,
            operation,
            "mflux.models." + symbol,
            components,
            tokenizers,
            steps,
            guidance,
            supports_negative_prompt=negative,
            image_min=1 if images else 0,
            image_max=images,
            image_argument=image_argument,
            mask_argument=mask,
            options=options,
            save_unsupported_reason=save_reason,
            local_unsupported_reason=local_reason,
            supported_control_types=control_types,
        )
        PIPELINES[name] = spec
        if img2img:
            PIPELINES[name + "/img2img"] = replace(
                spec,
                id=name + "/img2img",
                operation="img2img",
                image_min=1,
                image_max=1,
                options=options + ("image_strength",),
            )


_FLUX = ("vae", "transformer", "text_encoder", "text_encoder_2")
_FTOK = ("tokenizer", "tokenizer_2")
_add(
    "dev krea-dev",
    "flux.variants.txt2img.flux.Flux1",
    components=_FLUX,
    tokenizers=_FTOK,
    steps=28,
    guidance=3.5,
    options=("scheduler", "pid_decode", "pid_degrade_sigma"),
    img2img=True,
)
_add(
    "schnell",
    "flux.variants.txt2img.flux.Flux1",
    components=_FLUX,
    tokenizers=_FTOK,
    guidance=None,
    options=("scheduler", "pid_decode", "pid_degrade_sigma"),
    img2img=True,
)
_add(
    "dev-kontext",
    "flux.variants.kontext.flux_kontext.Flux1Kontext",
    operation="reference-edit",
    components=_FLUX,
    tokenizers=_FTOK,
    images=1,
    steps=28,
    guidance=2.5,
    options=("scheduler", "image_strength"),
)
_add(
    "dev-fill dev-fill-catvton",
    "flux.variants.fill.flux_fill.Flux1Fill",
    operation="inpaint",
    components=_FLUX,
    tokenizers=_FTOK,
    images=1,
    mask="masked_image_path",
    steps=28,
    guidance=30.0,
    options=("scheduler", "image_strength"),
)
_add(
    "dev-depth",
    "flux.variants.depth.flux_depth.Flux1Depth",
    operation="controlnet",
    components=_FLUX,
    tokenizers=_FTOK,
    images=1,
    image_argument="depth_image_path",
    steps=28,
    guidance=10.0,
    options=("scheduler",),
)
_add(
    "dev-redux",
    "flux.variants.redux.flux_redux.Flux1Redux",
    operation="reference-edit",
    components=_FLUX + ("image_encoder", "image_embedder"),
    tokenizers=_FTOK,
    images=8,
    image_argument="redux_image_paths",
    steps=28,
    guidance=3.5,
    options=("scheduler", "redux_image_strengths"),
)
_add(
    "dev-controlnet-canny",
    "flux.variants.controlnet.flux_controlnet.Flux1Controlnet",
    operation="controlnet",
    components=_FLUX + ("transformer_controlnet",),
    tokenizers=_FTOK,
    images=1,
    image_argument="controlnet_image_path",
    steps=28,
    guidance=3.5,
    options=("scheduler", "controlnet_strength"),
)
_add(
    "schnell-controlnet-canny dev-controlnet-upscaler",
    "flux.variants.controlnet.flux_controlnet.Flux1Controlnet",
    operation="controlnet",
    components=_FLUX + ("transformer_controlnet",),
    tokenizers=_FTOK,
    images=1,
    image_argument="controlnet_image_path",
    guidance=None,
    options=("scheduler", "controlnet_strength"),
)
_add(
    "flux2-klein-4b flux2-klein-9b flux2-klein-9b-kv flux2-klein-base-4b flux2-klein-base-9b",
    "flux2.variants.txt2img.flux2_klein.Flux2Klein",
    guidance=1.0,
    options=("scheduler", "pid_decode", "pid_degrade_sigma"),
    img2img=True,
)
for name in tuple(PIPELINES):
    if name.startswith("flux2-") and "/" not in name:
        PIPELINES[name + "/edit"] = replace(
            PIPELINES[name],
            id=name + "/edit",
            operation="reference-edit",
            backend_class="mflux.models.flux2.variants.edit.flux2_klein_edit.Flux2KleinEdit",
            image_min=1,
            image_max=8,
            image_argument="image_paths",
            options=("scheduler", "image_strength", "use_kv_cache"),
        )
_add(
    "qwen-image",
    "qwen.variants.txt2img.qwen_image.QwenImage",
    steps=50,
    negative=True,
    options=("scheduler", "pid_decode", "pid_degrade_sigma"),
    img2img=True,
)
_add(
    "qwen-image-edit",
    "qwen.variants.edit.qwen_image_edit.QwenImageEdit",
    operation="reference-edit",
    images=3,
    image_argument="image_paths",
    steps=40,
    negative=True,
    options=("scheduler",),
)
PIPELINES["qwen-image-edit"] = replace(
    PIPELINES["qwen-image-edit"], default_width=None, default_height=None
)
_add(
    "qwen-image-2.1",
    "qwen21.variants.txt2img.qwen_image_21.QwenImage21",
    tokenizers=("processor",),
    steps=40,
    guidance=1.0,
    negative=True,
    options=("scheduler",),
    img2img=True,
)
_add(
    "fibo fibo-lite",
    "fibo.variants.txt2img.fibo.FIBO",
    steps=50,
    negative=True,
    options=("scheduler",),
    img2img=True,
)
_add(
    "fibo-edit fibo-edit-rmbg",
    "fibo.variants.edit.fibo_edit.FIBOEdit",
    operation="reference-edit",
    images=1,
    mask="mask_path",
    steps=50,
    negative=True,
    options=("scheduler",),
)
_add(
    "z-image",
    "z_image.variants.z_image.ZImage",
    steps=50,
    guidance=4.0,
    negative=True,
    options=("scheduler", "pid_decode", "pid_degrade_sigma"),
    img2img=True,
)
_add(
    "z-image-turbo",
    "z_image.variants.z_image.ZImage",
    steps=9,
    guidance=None,
    options=("scheduler", "pid_decode", "pid_degrade_sigma"),
    img2img=True,
)
_add(
    "ernie-image",
    "ernie_image.variants.txt2img.ernie_image.ErnieImage",
    steps=50,
    guidance=4.0,
    negative=True,
    options=("scheduler", "pid_decode", "pid_degrade_sigma"),
    img2img=True,
)
_add(
    "ernie-image-turbo",
    "ernie_image.variants.txt2img.ernie_image.ErnieImage",
    steps=8,
    guidance=None,
    options=("scheduler", "pid_decode", "pid_degrade_sigma"),
    img2img=True,
)
_add(
    "krea-2 krea-2-raw",
    "krea2.variants.txt2img.krea2.Krea2",
    components=("vae", ".", "text_encoder"),
    steps=8,
    guidance=1.0,
    negative=True,
    options=("scheduler", "pid_decode", "pid_degrade_sigma"),
    img2img=True,
)
_add(
    "ideogram-4-fp8",
    "ideogram4.variants.txt2img.ideogram4.Ideogram4",
    components=("vae", "transformer", "unconditional_transformer", "text_encoder"),
    steps=28,
    guidance=4.0,
    options=(
        "preset",
        "strict_caption_validation",
        "warn_on_caption_issues",
        "cfg_end",
        "pid_decode",
        "pid_degrade_sigma",
    ),
)
_add(
    "boogu-image-turbo",
    "boogu.variants.txt2img.boogu_image.BooguImage",
    components=("vae", "transformer", "mllm"),
    tokenizers=("mllm",),
    guidance=None,
    options=("conditioning_sigma",),
)
_add(
    "lens-turbo",
    "lens.variants.txt2img.lens_image.LensImage",
    guidance=None,
    components=(".",),
    tokenizers=(),
    save_reason="LensImage has no save_model in mflux 0.20",
    local_reason="LensImage hardcodes an external VAE repository; mflux has no complete local checkpoint interface",
)
_add(
    "seedvr2-3b seedvr2-7b",
    "seedvr2.variants.upscale.seedvr2.SeedVR2",
    operation="upscale",
    components=(".",),
    tokenizers=(),
    steps=None,
    guidance=None,
    images=1,
    options=("resolution", "softness"),
    save_reason="SeedVR2 has no save_model in mflux 0.20",
)
_add(
    "z-image-turbo-controlnet-union-2.1",
    "z_image.variants.controlnet.z_image_turbo_controlnet.ZImageTurboControlnet",
    operation="controlnet",
    components=("vae", "transformer", "text_encoder", "controlnet"),
    images=8,
    image_argument="controls",
    steps=9,
    guidance=None,
    options=("scheduler", "controlnet_strength", "control_types", "control_strengths"),
    control_types=("canny", "mlsd"),
)


# Per-checkpoint oMLX defaults informed by mflux APIs.
_DEFAULT_STEPS = {
    "dev": 25,
    "krea-dev": 25,
    "dev-kontext": 25,
    "dev-fill": 25,
    "dev-fill-catvton": 25,
    "dev-depth": 25,
    "dev-redux": 25,
    "dev-controlnet-canny": 25,
    "dev-controlnet-upscaler": 25,
    "fibo-lite": 8,
    "fibo-edit-rmbg": 10,
    "flux2-klein-base-4b": 50,
    "flux2-klein-base-9b": 50,
    "qwen-image": 20,
    "qwen-image-edit": 20,
    "z-image-turbo-controlnet-union-2.1": 8,
}
for _id, _spec in tuple(PIPELINES.items()):
    _changes = {
        "default_steps": _DEFAULT_STEPS.get(_spec.base_model, _spec.default_steps)
    }
    if _spec.base_model in (
        "flux2-klein-4b",
        "flux2-klein-9b",
        "flux2-klein-9b-kv",
        "ernie-image-turbo",
    ):
        _changes.update(default_guidance=1.0, fixed_guidance=1.0)
    if _spec.base_model == "fibo-lite":
        _changes.update(
            default_guidance=1.0, fixed_guidance=1.0, supports_negative_prompt=False
        )
    if _spec.base_model.startswith("flux2-klein-base-"):
        _changes.update(default_guidance=4.0)
    if (
        _spec.base_model.startswith("flux2-")
        and _spec.base_model != "flux2-klein-9b-kv"
    ):
        _changes["options"] = tuple(o for o in _spec.options if o != "use_kv_cache")
    if _spec.base_model == "ideogram-4-fp8":
        _changes.update(default_steps=None, default_guidance=7.0)
    if _spec.base_model == "z-image":
        _changes["default_guidance"] = 0.0
    if _spec.base_model == "dev-depth":
        _changes["local_unsupported_reason"] = (
            "mflux FluxInitializer.init_depth unconditionally constructs DepthPro using Apple CDN weights; a local DepthPro preparation and constructor interface is not available"
        )
    if _spec.base_model == "dev-fill-catvton":
        _changes["local_unsupported_reason"] = (
            "CatVTON requires a two-image in-context virtual try-on adapter and custom transformer acquisition, which are not integrated; use dev-fill for ordinary inpainting"
        )
    if _spec.base_model == "dev-controlnet-upscaler":
        _changes["default_guidance"] = None
    if _spec.base_model in ("fibo", "fibo-lite", "fibo-edit", "fibo-edit-rmbg"):
        _changes["prompt_format"] = (
            "json-edit"
            if _spec.base_model in ("fibo-edit", "fibo-edit-rmbg")
            else "json"
        )
    _changes["options"] = tuple(
        option
        for option in _changes.get("options", _spec.options)
        if option not in ("pid_decode", "pid_degrade_sigma")
    )
    PIPELINES[_id] = replace(_spec, **_changes)


def resolve_base_model(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("base_model must be a nonempty string")
    if name in PIPELINES:
        return PIPELINES[name].base_model
    matches = {
        key
        for key, (aliases, repo) in MODEL_IDENTITIES.items()
        if name.lower() in [a.lower() for a in aliases]
    }
    if not matches:
        matches = {
            key
            for key, (_, repo) in MODEL_IDENTITIES.items()
            if name.lower() == repo.lower()
        }
    if len(matches) != 1:
        raise ValueError(
            f"Cannot resolve unambiguous base model for {name!r}; supply a canonical base_model"
        )
    return matches.pop()


def get_pipeline(pipeline_id: str) -> PipelineSpec:
    return PIPELINES.get(pipeline_id) or PIPELINES[resolve_base_model(pipeline_id)]


def pipelines_for_model(base_model: str) -> tuple[PipelineSpec, ...]:
    base_model = resolve_base_model(base_model)
    return tuple(spec for spec in PIPELINES.values() if spec.base_model == base_model)


def validate_task(spec: PipelineSpec, task: ImageTask) -> dict[str, Any]:
    if type(task.seed) is not int or not 0 <= task.seed <= 2**32 - 1:
        raise ValueError("seed must be an integer between 0 and 4294967295")
    if not isinstance(task.options, dict):
        raise ValueError("options must be an object")
    unknown = set(task.options) - set(spec.options)
    if unknown:
        raise ValueError(
            f"{spec.id} does not support options: {', '.join(sorted(unknown))}"
        )
    if not spec.image_min <= len(task.image_paths) <= spec.image_max:
        raise ValueError(
            f"{spec.id} requires {spec.image_min}..{spec.image_max} images"
        )
    if task.mask_path is not None and not spec.supports_mask:
        raise ValueError(f"{spec.id} does not support masks")
    if spec.operation == "inpaint" and task.mask_path is None:
        raise ValueError(f"{spec.id} requires a mask")
    out = {"seed": task.seed}
    if spec.operation == "upscale":
        if any(
            value is not None
            for value in (
                task.prompt,
                task.width,
                task.height,
                task.steps,
                task.guidance,
                task.negative_prompt,
            )
        ):
            raise ValueError(
                f"{spec.id} accepts no prompt, dimensions, steps, guidance or negative_prompt; use resolution"
            )
        out["resolution"] = task.options.get("resolution", 384)
    else:
        if not isinstance(task.prompt, str) or not task.prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        if spec.prompt_format.startswith("json"):
            try:
                structured = json.loads(task.prompt)
            except ValueError as exc:
                raise ValueError(
                    "FIBO prompt must be a JSON object; plain text requires a separately prepared VLM conversion"
                ) from exc
            if not isinstance(structured, dict):
                raise ValueError("FIBO prompt must be a JSON object")
            if spec.prompt_format == "json-edit" and (
                not isinstance(structured.get("edit_instruction"), str)
                or not structured["edit_instruction"].strip()
            ):
                raise ValueError(
                    "FIBO edit prompt JSON must include a nonempty edit_instruction string"
                )
        out["prompt"] = task.prompt
        for key, default in (
            ("width", spec.default_width),
            ("height", spec.default_height),
            ("steps", spec.default_steps),
        ):
            value = getattr(task, key)
            value = default if value is None else value
            if value is not None:
                if (
                    type(value) is not int
                    or value <= 0
                    or (key != "steps" and value % 16)
                ):
                    raise ValueError(
                        f"{key} must be a positive integer"
                        + (" multiple of 16" if key != "steps" else "")
                    )
                if (
                    key == "steps"
                    and value > 100
                    or key != "steps"
                    and not 256 <= value <= 2048
                ):
                    raise ValueError(f"{key} exceeds supported bounds")
                out["num_inference_steps" if key == "steps" else key] = value
        if task.guidance is not None and spec.default_guidance is None:
            raise ValueError(f"{spec.id} does not support guidance")
        if spec.default_guidance is not None:
            guidance = spec.default_guidance if task.guidance is None else task.guidance
            if (
                isinstance(guidance, bool)
                or not isinstance(guidance, (float, int))
                or not math.isfinite(guidance)
                or guidance < 0
            ):
                raise ValueError("guidance must be a finite nonnegative number")
            out["guidance"] = guidance
        if task.negative_prompt is not None:
            if not spec.supports_negative_prompt:
                raise ValueError(f"{spec.id} does not support negative_prompt")
            if not isinstance(task.negative_prompt, str):
                raise ValueError("negative_prompt must be a string")
            if (
                out.get("guidance", 0) <= 1
                and spec.base_model not in ("krea-2", "krea-2-raw")
                or out.get("guidance", 0) == 1
            ):
                raise ValueError("negative_prompt requires guidance greater than 1")
            out["negative_prompt"] = task.negative_prompt
    if spec.base_model == "qwen-image-2.1":
        active_cfg = out.get("guidance", 1.0) > 1.0 and bool(task.negative_prompt)
        if task.guidance is not None and task.guidance != 1.0 and not active_cfg:
            raise ValueError(
                "Qwen Image 2.1 nonneutral guidance requires a nonempty negative_prompt and guidance greater than 1"
            )
        if task.negative_prompt is not None and not active_cfg:
            raise ValueError(
                "Qwen Image 2.1 negative_prompt requires a nonempty string and active CFG"
            )
    if (
        spec.fixed_guidance is not None
        and task.guidance is not None
        and task.guidance != spec.fixed_guidance
    ):
        raise ValueError(f"{spec.id} requires guidance {spec.fixed_guidance}")
    if spec.base_model == "ideogram-4-fp8":
        if task.steps is None:
            if task.guidance is not None:
                raise ValueError(
                    "Ideogram custom guidance requires explicit steps; presets define their own guidance"
                )
            out.pop("guidance", None)
        elif "preset" in task.options:
            raise ValueError("Ideogram preset and custom steps cannot be combined")
    if task.image_paths:
        out[spec.image_argument] = (
            list(task.image_paths)
            if spec.image_argument in ("image_paths", "redux_image_paths", "controls")
            else task.image_paths[0]
        )
    if spec.operation == "img2img":
        out["image_strength"] = 0.4
    if task.mask_path is not None:
        out[spec.mask_argument] = task.mask_path
    for key, value in task.options.items():
        if key in (
            "image_strength",
            "controlnet_strength",
            "softness",
            "conditioning_sigma",
            "pid_degrade_sigma",
            "cfg_end",
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{key} must be a finite nonnegative number")
            if key in ("image_strength", "softness", "cfg_end") and value > 1:
                raise ValueError(f"{key} must be at most 1")
        elif key in (
            "pid_decode",
            "use_kv_cache",
            "strict_caption_validation",
            "warn_on_caption_issues",
        ):
            if type(value) is not bool:
                raise ValueError(f"{key} must be a boolean")
        elif key == "scheduler":
            allowed = (
                ("linear", "er_sde", "euler")
                if spec.base_model in ("krea-2", "krea-2-raw")
                else ("linear", "flow_match_euler_discrete")
            )
            if value not in allowed:
                raise ValueError(
                    "scheduler must be linear or flow_match_euler_discrete"
                )
        elif key == "resolution":
            if type(value) is not int or not 1 <= value <= 2048:
                raise ValueError("resolution must be an integer between 1 and 2048")
        elif key in ("control_types", "control_strengths", "redux_image_strengths"):
            if not isinstance(value, (list, tuple)) or len(value) != len(
                task.image_paths
            ):
                raise ValueError(f"{key} must have one entry per input image")
            if key == "control_types" and any(
                v not in spec.supported_control_types for v in value
            ):
                raise ValueError(
                    "Unsupported control type; depth, hed and pose require explicit local auxiliary preprocessor preparation. Supported types: "
                    + ", ".join(spec.supported_control_types)
                )
            if key != "control_types" and any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or v < 0
                for v in value
            ):
                raise ValueError(f"{key} entries must be finite nonnegative numbers")
        elif key == "preset" and (
            not isinstance(value, str)
            or value.upper() not in ("V4_TURBO_12", "V4_DEFAULT_20", "V4_QUALITY_48")
        ):
            raise ValueError(
                "preset must be V4_TURBO_12, V4_DEFAULT_20 or V4_QUALITY_48"
            )
        out[key] = value
    if spec.image_argument == "controls" and "control_types" not in task.options:
        raise ValueError("control_types is required for each control image")
    return out


# Canonical identities from mflux 0.20 AVAILABLE_MODELS.
MODEL_IDENTITIES = {
    "krea-2": (("krea-2", "krea2"), "krea/Krea-2-Turbo"),
    "krea-2-raw": (("krea-2-raw", "krea2-raw"), "krea/Krea-2-Raw"),
    "dev": (("dev",), "black-forest-labs/FLUX.1-dev"),
    "schnell": (("schnell",), "black-forest-labs/FLUX.1-schnell"),
    "dev-kontext": (("dev-kontext",), "black-forest-labs/FLUX.1-Kontext-dev"),
    "dev-fill": (("dev-fill",), "black-forest-labs/FLUX.1-Fill-dev"),
    "dev-redux": (("dev-redux",), "black-forest-labs/FLUX.1-Redux-dev"),
    "dev-depth": (("dev-depth",), "black-forest-labs/FLUX.1-Depth-dev"),
    "dev-controlnet-canny": (("dev-controlnet-canny",), "black-forest-labs/FLUX.1-dev"),
    "schnell-controlnet-canny": (
        ("schnell-controlnet-canny",),
        "black-forest-labs/FLUX.1-schnell",
    ),
    "dev-controlnet-upscaler": (
        ("dev-controlnet-upscaler",),
        "black-forest-labs/FLUX.1-dev",
    ),
    "dev-fill-catvton": (("dev-fill-catvton",), "black-forest-labs/FLUX.1-Fill-dev"),
    "krea-dev": (("krea-dev", "dev-krea"), "black-forest-labs/FLUX.1-Krea-dev"),
    "flux2-klein-4b": (
        ("flux2-klein-4b", "flux2-klein-4B", "flux2-klein", "klein-4b", "klein-4B"),
        "black-forest-labs/FLUX.2-klein-4B",
    ),
    "flux2-klein-9b": (
        ("flux2-klein-9b", "flux2-klein-9B", "klein-9b", "klein-9B"),
        "black-forest-labs/FLUX.2-klein-9B",
    ),
    "flux2-klein-9b-kv": (
        (
            "flux2-klein-9b-kv",
            "flux2-klein-9B-kv",
            "flux2-klein-9b-KV",
            "klein-9b-kv",
            "klein-9B-kv",
        ),
        "black-forest-labs/FLUX.2-klein-9b-kv",
    ),
    "flux2-klein-base-4b": (
        (
            "flux2-klein-base-4b",
            "flux2-klein-base-4B",
            "flux2-base-4b",
            "flux2-base-4B",
            "klein-base-4b",
            "klein-base-4B",
        ),
        "black-forest-labs/FLUX.2-klein-base-4B",
    ),
    "flux2-klein-base-9b": (
        (
            "flux2-klein-base-9b",
            "flux2-klein-base-9B",
            "flux2-base-9b",
            "flux2-base-9B",
            "klein-base-9b",
            "klein-base-9B",
        ),
        "black-forest-labs/FLUX.2-klein-base-9B",
    ),
    "qwen-image": (
        ("qwen-image", "qwen", "qwen-image-2512", "qwen-2512"),
        "Qwen/Qwen-Image-2512",
    ),
    "qwen-image-edit": (
        (
            "qwen-image-edit",
            "qwen-edit",
            "qwen-edit-plus",
            "qwen-edit-2509",
            "qwen-edit-2511",
            "qwen-image-edit-2511",
        ),
        "Qwen/Qwen-Image-Edit-2509",
    ),
    "fibo": (("fibo",), "briaai/FIBO"),
    "fibo-lite": (("fibo-lite", "fibo_lite"), "briaai/Fibo-lite"),
    "fibo-edit": (("fibo-edit", "fiboedit"), "briaai/Fibo-Edit"),
    "fibo-edit-rmbg": (("fibo-edit-rmbg", "fiboedit-rmbg"), "briaai/Fibo-Edit-RMBG"),
    "z-image": (("z-image", "zimage"), "Tongyi-MAI/Z-Image"),
    "lens-turbo": (("lens-turbo", "lens"), "Comfy-Org/Lens"),
    "z-image-turbo-controlnet-union-2.1": (
        (
            "z-image-turbo-controlnet-union-2.1",
            "z-image-controlnet-union-2.1",
            "z-image-controlnet",
            "z-image-turbo-controlnet",
        ),
        "Tongyi-MAI/Z-Image-Turbo",
    ),
    "z-image-turbo": (("z-image-turbo", "zimage-turbo"), "Tongyi-MAI/Z-Image-Turbo"),
    "seedvr2-3b": (("seedvr2-3b", "seedvr2"), "numz/SeedVR2_comfyUI"),
    "ernie-image": (("ernie-image",), "baidu/ERNIE-Image"),
    "ernie-image-turbo": (("ernie-image-turbo",), "baidu/ERNIE-Image-Turbo"),
    "seedvr2-7b": (("seedvr2-7b", "seedvr2-7B"), "numz/SeedVR2_comfyUI"),
    "ideogram-4-fp8": (
        ("ideogram-4-fp8", "ideogram4-fp8", "ideogram4", "ideogram-4", "ideogram"),
        "ideogram-ai/ideogram-4-fp8",
    ),
    "boogu-image-turbo": (
        ("boogu-image-turbo", "boogu-turbo", "boogu-image", "boogu"),
        "Boogu/Boogu-Image-0.1-Turbo",
    ),
    "qwen-image-2.1": (
        ("qwen-image-2.1", "qwen-2.1", "qwen-image-21"),
        "Qwen/Qwen-Image-2.1",
    ),
}
