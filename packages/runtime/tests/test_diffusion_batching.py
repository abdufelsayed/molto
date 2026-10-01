"""CPU-only batch proofs. Native generation code runs with tiny fake components."""

import io
from dataclasses import replace
from types import MethodType, SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from omlx_runtime.diffusion import ImageTask, get_pipeline
from omlx_runtime.diffusion.batching import generate_batch
from PIL import Image


@pytest.fixture(autouse=True)
def cpu_stream():
    # This suite never loads checkpoint weights or submits full generation to Metal.
    with mx.stream(mx.cpu):
        yield


class Context:
    def __init__(self, state):
        self.state = state

    def before_loop(self, latents):
        self.state["initial"] = latents

    def in_loop(self, step, latents):
        self.state.setdefault("steps", []).append((step, latents.shape))

    def after_loop(self, latents):
        self.state["final"] = latents


class Callbacks:
    def __init__(self, state):
        self.state = state

    def start(self, seed, prompt, config):
        self.state["callback_metadata"] = (seed, prompt)
        return Context(self.state)


class TinyVAE:
    def __init__(self, state):
        self.state = state
        self.bn = SimpleNamespace(
            running_mean=mx.zeros((128,)), running_var=mx.ones((128,)), eps=1e-4
        )

    def encode(self, image):
        self.state["reference_encodes"] = self.state.get("reference_encodes", 0) + 1
        return mx.zeros((1, 32, image.shape[2] // 8, image.shape[3] // 8))

    def decode_packed_latents(self, packed_latents, tiling_config):
        self.state["decode_shape"] = packed_latents.shape
        if self.state.get("fail_decode"):
            raise RuntimeError("Decode failed")
        # Seed-dependent pixels expose output slicing and ordering mistakes.
        return mx.broadcast_to(
            packed_latents[:, :3, :1, :1], (packed_latents.shape[0], 3, 4, 4)
        )


def tiny_native_model(edit=False, *, cached_hooks=True, kv=False):
    from mflux.models.common.config.model_config import AVAILABLE_MODELS
    from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    native_class = Flux2KleinEdit if edit else Flux2Klein
    model = native_class.__new__(native_class)
    nn.Module.__init__(model)  # Skip the native initializer and all checkpoint loading.
    state = {}
    model.model_config = AVAILABLE_MODELS[
        "flux2-klein-9b-kv" if kv else "flux2-klein-4b"
    ]
    model.callbacks = Callbacks(state)
    model.vae = TinyVAE(state)
    model.transformer = SimpleNamespace(
        transformer_blocks=[None], single_transformer_blocks=[None]
    )
    model.tiling_config = None
    model.bits = None
    model.lora_paths = None
    model.lora_scales = None

    def encode_prompt_pair(self, **kwargs):
        state["encode_calls"] = state.get("encode_calls", 0) + 1
        embeddings = mx.arange(18, dtype=mx.float32).reshape(1, 3, 6)
        ids = mx.zeros((1, 3, 4), dtype=mx.int32)
        return embeddings, ids, None, None

    def predict(**kwargs):
        latents = kwargs["latents"]
        state.setdefault("predict_shapes", []).append(latents.shape)
        assert kwargs["prompt_embeds"].shape[0] == latents.shape[0]
        assert kwargs["text_ids"].shape[0] == latents.shape[0]
        assert kwargs["latent_ids"].shape[0] == latents.shape[0]
        assert bool(mx.array_equal(kwargs["text_ids"][0], kwargs["text_ids"][-1]))
        if "image_latents" in kwargs:
            assert kwargs["image_latents"].shape[0] == latents.shape[0]
            assert kwargs["image_latent_ids"].shape[0] == latents.shape[0]
            state["reference_shape"] = kwargs["image_latents"].shape
        cache = kwargs.get("kv_cache")
        if cache is not None:
            state.setdefault("cache_modes", []).append(cache.mode)
        if state.get("fail_predict"):
            raise RuntimeError("Prediction failed")
        return mx.zeros_like(latents)

    model._predict = lambda transformer: predict
    if edit:
        model._cached_predict = lambda transformer: predict
    if cached_hooks:
        # This models a resident per-model prompt-cache wrapper. Restoration must
        # preserve its exact function object, not replace it with a native method.
        model._encode_prompt_pair = MethodType(encode_prompt_pair, model)
    else:
        # A subclass supplies the fake encoder without an instance attribute.
        class TinyModel(native_class):
            _encode_prompt_pair = encode_prompt_pair

        model.__class__ = TinyModel
    return model, state


def task_variants(count=2, *, edit=False, image_path=None, kv=False):
    base = "flux2-klein-9b-kv" if kv else "flux2-klein-4b"
    spec = get_pipeline(base + "/edit" if edit else base)
    task = ImageTask(
        prompt="A bird",
        seed=7,
        width=256,
        height=256,
        steps=2,
        image_paths=(str(image_path),) if edit else (),
    )
    return [replace(task, seed=7 + index) for index in range(count)], spec


def reference_image(tmp_path):
    path = tmp_path / "reference.png"
    Image.new("RGB", (32, 32), "red").save(path)
    return path


@pytest.mark.parametrize("count", [2, 4])
@pytest.mark.parametrize("edit", [False, True])
def test_native_loop_vectorizes_and_serializes_each_seed_in_order(
    count, edit, tmp_path
):
    from mflux.models.flux2.latent_creator.flux2_latent_creator import (
        Flux2LatentCreator,
    )
    from mflux.utils.image_util import ImageUtil

    model, state = tiny_native_model(edit)
    tasks, spec = task_variants(count, edit=edit, image_path=reference_image(tmp_path))
    original_hook = model.__dict__["_encode_prompt_pair"]
    original_globals = model.generate_image.__func__.__globals__.copy()
    generated = generate_batch(model, tasks, spec)
    assert len(generated) == count
    assert [result.seed for result in generated] == [task.seed for task in tasks]
    assert state["encode_calls"] == 1
    assert state["initial"].shape == (count, 256, 128)
    assert state["predict_shapes"] == [(count, 256, 128)] * 2
    assert state["decode_shape"] == (count, 128, 16, 16)
    for index, task in enumerate(tasks):
        single, _, _, _ = Flux2LatentCreator.prepare_packed_latents(
            seed=task.seed, height=256, width=256, batch_size=1
        )
        assert bool(mx.array_equal(state["initial"][index : index + 1], single))
        packed = single.reshape(1, 16, 16, 128).transpose(0, 3, 1, 2)
        expected = ImageUtil.to_pil(
            mx.broadcast_to(packed[:, :3, :1, :1], (1, 3, 4, 4))
        )
        assert np.array_equal(np.array(generated[index].image), np.array(expected))
        stream = io.BytesIO()
        generated[index].image.save(stream, format="PNG")
        assert stream.getvalue().startswith(b"\x89PNG")
    if edit:
        assert state["reference_encodes"] == 1
        assert state["reference_shape"][0] == count
    assert model.__dict__["_encode_prompt_pair"] is original_hook
    assert "_prepare_generation_latents" not in model.__dict__
    assert all(
        model.generate_image.__func__.__globals__[key] is value
        for key, value in original_globals.items()
    )


def test_real_rng_batched_shape_is_not_independent_seed_sequence():
    from mflux.models.flux2.latent_creator.flux2_latent_creator import (
        Flux2LatentCreator,
    )

    batch, _, _, _ = Flux2LatentCreator.prepare_packed_latents(
        seed=7, height=256, width=256, batch_size=2
    )
    singles = [
        Flux2LatentCreator.prepare_packed_latents(
            seed=seed, height=256, width=256, batch_size=1
        )[0]
        for seed in (7, 8)
    ]
    assert not bool(mx.array_equal(batch[0:1], singles[0]))
    assert not bool(mx.array_equal(batch[1:2], singles[1]))


@pytest.mark.parametrize("failure", ["fail_predict", "fail_decode"])
@pytest.mark.parametrize("edit", [False, True])
def test_exception_restores_exact_hooks_and_native_globals(failure, edit, tmp_path):
    model, state = tiny_native_model(edit)
    tasks, spec = task_variants(edit=edit, image_path=reference_image(tmp_path))
    state[failure] = True
    hook = model.__dict__["_encode_prompt_pair"]
    original_latent_hook = MethodType(lambda self, **kwargs: None, model)
    if not edit:
        model._prepare_generation_latents = original_latent_hook
    globals_before = model.generate_image.__func__.__globals__.copy()
    with pytest.raises(RuntimeError):
        generate_batch(model, tasks, spec)
    assert model.__dict__["_encode_prompt_pair"] is hook
    if not edit:
        assert model.__dict__["_prepare_generation_latents"] is original_latent_hook
    assert all(
        model.generate_image.__func__.__globals__[key] is value
        for key, value in globals_before.items()
    )


def test_absent_instance_prompt_hook_is_absent_after_generation():
    model, _ = tiny_native_model(cached_hooks=False)
    tasks, spec = task_variants()
    assert "_encode_prompt_pair" not in model.__dict__
    generate_batch(model, tasks, spec)
    assert "_encode_prompt_pair" not in model.__dict__


@pytest.mark.parametrize(
    "change",
    [
        {"prompt": "Another prompt"},
        {"width": 512},
        {"steps": 3},
        {"options": {"scheduler": "linear"}},
        {"image_paths": ("other.png",)},
    ],
)
def test_heterogeneous_tasks_rejected_before_native_execution(change):
    model, state = tiny_native_model()
    tasks, spec = task_variants()
    tasks[1] = replace(tasks[1], **change)
    with pytest.raises(ValueError):
        generate_batch(model, tasks, spec)
    assert not state


@pytest.mark.parametrize("count", [0, 5])
def test_batch_count_is_bounded(count):
    model, state = tiny_native_model()
    tasks, spec = task_variants(count)
    with pytest.raises(ValueError, match="1..4"):
        generate_batch(model, tasks, spec)
    assert not state


def test_img2img_and_other_backend_are_rejected():
    model, state = tiny_native_model()
    tasks, _ = task_variants()
    tasks = [replace(task, image_paths=("input.png",)) for task in tasks]
    with pytest.raises(ValueError, match="supports only"):
        generate_batch(model, tasks, get_pipeline("flux2-klein-4b/img2img"))
    tasks, _ = task_variants()
    with pytest.raises(ValueError, match="supports only"):
        generate_batch(model, tasks, get_pipeline("z-image-turbo"))
    assert not state


def test_native_dependency_guard_fails_before_hooks_change(monkeypatch):
    model, state = tiny_native_model()
    tasks, spec = task_variants()
    original_hook = model.__dict__["_encode_prompt_pair"]
    monkeypatch.delitem(model.generate_image.__func__.__globals__, "ImageUtil")
    with pytest.raises(ValueError, match="hooks changed"):
        generate_batch(model, tasks, spec)
    assert model.__dict__["_encode_prompt_pair"] is original_hook
    assert not state


def test_already_batched_prompt_cache_result_rejected_and_wrapper_restored():
    model, _ = tiny_native_model()

    def wrapper(**kwargs):
        return mx.zeros((2, 3, 6)), mx.zeros((2, 3, 4)), None, None

    model._encode_prompt_pair = wrapper
    tasks, spec = task_variants()
    with pytest.raises(ValueError, match="singleton prompt"):
        generate_batch(model, tasks, spec)
    assert model.__dict__["_encode_prompt_pair"] is wrapper


def test_edit_native_kv_mode_switches_are_preserved(tmp_path):
    model, state = tiny_native_model(edit=True, kv=True)
    tasks, spec = task_variants(
        edit=True, image_path=reference_image(tmp_path), kv=True
    )
    generated = generate_batch(model, tasks, spec)
    assert len(generated) == 2
    assert state["cache_modes"] == ["extract", "cached"]


@pytest.mark.parametrize("count", [1, 2, 4])
def test_reference_conditioning_callback_reuses_singleton_then_broadcasts(
    count, tmp_path
):
    from mflux.models.flux2.variants.edit.flux2_klein_edit_helpers import (
        _Flux2KleinEditHelpers,
    )

    model, state = tiny_native_model(edit=True)
    tasks, spec = task_variants(count, edit=True, image_path=reference_image(tmp_path))
    cached = []
    requested_batches = []

    def reference_conditioning(*, vae, tiling_config, image_paths, batch_size):
        assert vae is model.vae
        assert image_paths == list(tasks[0].image_paths)
        requested_batches.append(batch_size)
        if not cached:
            cached.append(
                _Flux2KleinEditHelpers.prepare_reference_image_conditioning(
                    vae=vae,
                    tiling_config=tiling_config,
                    image_paths=image_paths,
                    batch_size=1,
                )
            )
        assert cached[0][0].shape[0] == cached[0][1].shape[0] == 1
        return tuple(
            mx.broadcast_to(value, (batch_size, *value.shape[1:]))
            for value in cached[0]
        )

    first = generate_batch(
        model, tasks, spec, reference_conditioning=reference_conditioning
    )
    second = generate_batch(
        model, tasks, spec, reference_conditioning=reference_conditioning
    )
    assert len(first) == len(second) == count
    assert state["reference_encodes"] == 1
    assert requested_batches == [count, count]
    assert state["reference_shape"][0] == count
    assert (
        _Flux2KleinEditHelpers.prepare_generation_latents.__globals__[
            "_Flux2KleinEditHelpers"
        ]
        is _Flux2KleinEditHelpers
    )


def test_bad_reference_callback_batch_restores_hooks(tmp_path):
    model, _ = tiny_native_model(edit=True)
    hook = model.__dict__["_encode_prompt_pair"]
    tasks, spec = task_variants(edit=True, image_path=reference_image(tmp_path))

    def reference_conditioning(**kwargs):
        return mx.zeros((1, 4, 128)), mx.zeros((1, 4, 4))

    with pytest.raises(ValueError, match="reference conditioning must match"):
        generate_batch(
            model, tasks, spec, reference_conditioning=reference_conditioning
        )
    assert model.__dict__["_encode_prompt_pair"] is hook


def test_mismatched_prompt_positions_rejected_before_denoising():
    model, state = tiny_native_model()

    def encode(**kwargs):
        return mx.zeros((1, 3, 6)), mx.zeros((1, 2, 4)), None, None

    model._encode_prompt_pair = encode
    tasks, spec = task_variants()
    with pytest.raises(ValueError, match="share sequence positions"):
        generate_batch(model, tasks, spec)
    assert model.__dict__["_encode_prompt_pair"] is encode
    assert not state


@pytest.mark.parametrize("count", [2, 4])
def test_tiny_native_transformer_batch_rows_remain_independent(count):
    from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer

    # Random tiny parameters, not checkpoint weights. This exercises actual
    # Flux2 attention/modulation/position code on CPU at a few tokens per row.
    transformer = Flux2Transformer(
        in_channels=8,
        out_channels=8,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=12,
        timestep_guidance_channels=8,
        mlp_ratio=2.0,
        axes_dims_rope=(2, 2, 2, 2),
    )
    hidden = mx.arange(count * 32, dtype=mx.float32).reshape(count, 4, 8) / (count * 32)
    prompt = mx.broadcast_to(
        mx.arange(36, dtype=mx.float32).reshape(1, 3, 12) / 36, (count, 3, 12)
    )
    image_ids = mx.zeros((count, 4, 4), dtype=mx.int32)
    text_ids = mx.zeros((count, 3, 4), dtype=mx.int32)
    batch = transformer(
        hidden_states=hidden,
        encoder_hidden_states=prompt,
        timestep=0.5,
        img_ids=image_ids,
        txt_ids=text_ids,
    )
    serial = mx.concatenate(
        [
            transformer(
                hidden_states=hidden[index : index + 1],
                encoder_hidden_states=prompt[index : index + 1],
                timestep=0.5,
                img_ids=image_ids[index : index + 1],
                txt_ids=text_ids[index : index + 1],
            )
            for index in range(count)
        ]
    )
    assert batch.shape == (count, 4, 8)
    assert bool(mx.allclose(batch, serial, rtol=1e-5, atol=1e-5))
