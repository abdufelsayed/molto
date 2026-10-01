"""Request-local Flux2 batching around the installed native denoising loop.

The caller must serialize access to the model on oMLX's MLX executor. This
adapter temporarily wraps instance hooks, never changes native module globals,
and restores existing hooks, including prompt-cache wrappers, on every exit.
"""

from contextlib import ExitStack, contextmanager
from dataclasses import replace
from importlib import import_module
from types import FunctionType

import mlx.core as mx

from .registry import ImageTask, PipelineSpec, validate_task

_TXT_BACKEND = "mflux.models.flux2.variants.txt2img.flux2_klein.Flux2Klein"
_EDIT_BACKEND = "mflux.models.flux2.variants.edit.flux2_klein_edit.Flux2KleinEdit"
_MAX_BATCH_SIZE = 4
_MISSING = object()


@contextmanager
def _instance_hook(model, name, replacement):
    previous = vars(model).get(name, _MISSING)
    # MLX Module.__setattr__ manages parameter dictionaries. Hooks are ordinary
    # Python attributes and must not alter those dictionaries during a request.
    object.__setattr__(model, name, replacement)
    try:
        yield
    finally:
        if previous is _MISSING:
            object.__delattr__(model, name)
        else:
            object.__setattr__(model, name, previous)


def _native_function(model, spec):
    expected_operation = {
        _TXT_BACKEND: "txt2img",
        _EDIT_BACKEND: "reference-edit",
    }.get(spec.backend_class)
    if expected_operation is None or spec.operation != expected_operation:
        raise ValueError(
            "Vectorized batching supports only Flux2 txt2img and reference-edit"
        )
    module_name, class_name = spec.backend_class.rsplit(".", 1)
    native_class = getattr(import_module(module_name), class_name)
    method = model.generate_image
    function = getattr(method, "__func__", None)
    if (
        not isinstance(model, native_class)
        or function is not native_class.generate_image
    ):
        raise ValueError(
            "The loaded model does not expose the expected native Flux2 generation method"
        )
    required_globals = {"ImageUtil", "Config", "mx"}
    required_calls = {"to_image", "_encode_prompt_pair", "decode_packed_latents"}
    if spec.operation == "reference-edit":
        required_globals.add("_Flux2KleinEditHelpers")
        required_calls.update(
            {"prepare_generation_latents", "prepare_reference_image_conditioning"}
        )
    else:
        required_calls.add("_prepare_generation_latents")
    if not required_globals <= function.__globals__.keys() or not required_calls <= set(
        function.__code__.co_names
    ):
        raise ValueError(
            "Installed mflux Flux2 generation hooks changed; vectorized batching is unavailable"
        )
    if not callable(getattr(function.__globals__["ImageUtil"], "to_image", None)):
        raise ValueError("Installed mflux image conversion hook changed")
    return function


def _stacked_latents(seeds, *, height, width):
    from mflux.models.flux2.latent_creator.flux2_latent_creator import (
        Flux2LatentCreator,
    )

    # A single random draw with shape [B, ...] changes even the first seed's
    # noise. Draw each request seed with its native singleton shape instead.
    prepared = [
        Flux2LatentCreator.prepare_packed_latents(
            seed=seed,
            height=height,
            width=width,
            batch_size=1,
        )
        for seed in seeds
    ]
    latent_height, latent_width = prepared[0][2:]
    for latents, ids, row_height, row_width in prepared:
        if (
            latents.ndim != 3
            or ids.ndim != 3
            or latents.shape[0] != 1
            or ids.shape[0] != 1
            or ids.shape[1] != latents.shape[1]
            or ids.shape[2] != 4
            or (row_height, row_width) != (latent_height, latent_width)
        ):
            raise ValueError("Installed mflux singleton latent shapes changed")
    return (
        mx.concatenate([row[0] for row in prepared], axis=0),
        mx.concatenate([row[1] for row in prepared], axis=0),
        latent_height,
        latent_width,
    )


def generate_batch(
    model,
    tasks: list[ImageTask] | tuple[ImageTask, ...],
    spec: PipelineSpec,
    *,
    reference_conditioning=None,
):
    """Run 1..4 seed variants of one task in a native vectorized denoising loop.

    All settings and reference images must match. This preserves singleton RNG
    draws and output order; floating-point kernel differences may still make
    batched denoising differ numerically from serial generation.
    """
    if not 1 <= len(tasks) <= _MAX_BATCH_SIZE:
        raise ValueError("A vectorized image batch must contain 1..4 tasks")
    if any(not isinstance(task, ImageTask) for task in tasks):
        raise ValueError("Every batch entry must be an ImageTask")
    first = tasks[0]
    homogeneous = replace(first, seed=0)
    kwargs = validate_task(spec, first)
    for task in tasks[1:]:
        validate_task(spec, task)
        if replace(task, seed=0) != homogeneous:
            raise ValueError(
                "Batched tasks must have identical prompt, settings and media except seed"
            )
    function = _native_function(model, spec)
    seeds = tuple(task.seed for task in tasks)
    batch_size = len(seeds)
    native_globals = function.__globals__.copy()
    native_image_util = native_globals["ImageUtil"]

    class BatchImageUtil:
        @staticmethod
        def to_image(*, decoded_latents, **metadata):
            if (
                not isinstance(decoded_latents, mx.array)
                or decoded_latents.ndim not in (4, 5)
                or decoded_latents.shape[0] != batch_size
            ):
                raise ValueError(
                    "Native Flux2 decoded batch shape differs from the requested image count"
                )
            return [
                native_image_util.to_image(
                    decoded_latents=decoded_latents[index : index + 1],
                    **{**metadata, "seed": seed},
                )
                for index, seed in enumerate(seeds)
            ]

    native_globals["ImageUtil"] = BatchImageUtil
    if spec.operation == "reference-edit":
        native_helpers = native_globals["_Flux2KleinEditHelpers"]
        if not callable(getattr(native_helpers, "prepare_generation_latents", None)):
            raise ValueError("Installed mflux Flux2 edit latent hook changed")

        class BatchEditHelpers(native_helpers):
            @staticmethod
            def prepare_generation_latents(*, seed, height, width):
                return _stacked_latents(seeds, height=height, width=width)

            @staticmethod
            def prepare_reference_image_conditioning(**arguments):
                prepare = (
                    reference_conditioning
                    if reference_conditioning is not None
                    else native_helpers.prepare_reference_image_conditioning
                )
                conditioned = prepare(**arguments)
                if not isinstance(conditioned, tuple) or len(conditioned) != 2:
                    raise ValueError(
                        "Flux2 reference conditioning must return latents and position IDs"
                    )
                latents, ids = conditioned
                if latents is None and ids is None:
                    return conditioned
                if (
                    not isinstance(latents, mx.array)
                    or not isinstance(ids, mx.array)
                    or latents.ndim != 3
                    or ids.ndim != 3
                    or latents.shape[0] != arguments["batch_size"]
                    or ids.shape[0] != arguments["batch_size"]
                    or ids.shape[1] != latents.shape[1]
                    or ids.shape[2] != 4
                ):
                    raise ValueError(
                        "Flux2 reference conditioning must match the requested batch and position shapes"
                    )
                return conditioned

        native_globals["_Flux2KleinEditHelpers"] = BatchEditHelpers

    # Bind the existing function code to a private globals dictionary. The
    # upstream denoising loop, scheduler and edit KV cache remain unchanged.
    native_generate = FunctionType(
        function.__code__,
        native_globals,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    native_generate.__kwdefaults__ = function.__kwdefaults__
    original_encode = model._encode_prompt_pair

    def encode_prompt_pair(**arguments):
        encoded = original_encode(**arguments)
        if not isinstance(encoded, tuple) or len(encoded) != 4:
            raise ValueError("Installed mflux Flux2 prompt hook changed")
        if (
            encoded[0] is None
            or encoded[1] is None
            or (encoded[2] is None) != (encoded[3] is None)
        ):
            raise ValueError("Flux2 prompt embeddings and position IDs must be paired")
        for embedding, ids in (encoded[:2], encoded[2:]):
            if embedding is not None and (
                not isinstance(embedding, mx.array)
                or not isinstance(ids, mx.array)
                or embedding.ndim != 3
                or ids.ndim != 3
                or embedding.shape[1] != ids.shape[1]
                or ids.shape[2] != 4
            ):
                raise ValueError(
                    "Flux2 prompt embeddings and position IDs must share sequence positions"
                )
        expanded = []
        for value in encoded:
            if value is None:
                expanded.append(None)
                continue
            if (
                not isinstance(value, mx.array)
                or value.ndim != 3
                or value.shape[0] != 1
            ):
                raise ValueError(
                    "Flux2 batching requires singleton prompt embeddings and position IDs"
                )
            expanded.append(mx.repeat(value, batch_size, axis=0))
        return tuple(expanded)

    def prepare_generation_latents(*, seed, config):
        if config.image_path is not None:
            raise ValueError("Vectorized Flux2 img2img is not supported")
        return _stacked_latents(seeds, height=config.height, width=config.width)

    with ExitStack() as hooks:
        hooks.enter_context(
            _instance_hook(model, "_encode_prompt_pair", encode_prompt_pair)
        )
        if spec.operation == "txt2img":
            hooks.enter_context(
                _instance_hook(
                    model, "_prepare_generation_latents", prepare_generation_latents
                )
            )
        generated = native_generate(model, **kwargs)
    if not isinstance(generated, list) or len(generated) != batch_size:
        raise ValueError("Native Flux2 did not serialize every requested batch member")
    return generated
