"""Exact cache tests use tiny MLX arrays and fake native methods, never weights."""

from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from omlx.diffusion.backend import MFluxBackend
from omlx.diffusion.cache import PromptCache, PromptCacheBinding
from omlx.diffusion.registry import get_pipeline


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = mx.array([1.0, 2.0])
        self.prompt_cache = {}
        self.encodes = 0
        self.factories = 0

    def _encode_prompt_pair(self, *, prompt, negative_prompt=None, guidance=4.0):
        self.encodes += 1
        if prompt == "fail":
            raise RuntimeError("encoder failed")
        value = len(prompt) + len(negative_prompt or "") + guidance
        return mx.array([value]), mx.array([1]), None, None

    def _predict(self, transformer):
        self.factories += 1
        return lambda value: transformer(value)

    def _cached_predict(self, transformer):
        self.factories += 1
        return lambda value: transformer(value)

    def generate(self, prompt, seed):
        embeds = self._encode_prompt_pair(prompt=prompt)[0]
        return embeds + seed


def test_nested_tensor_bytes_materialization_and_no_prompt_stats():
    cache = PromptCache(max_entries=3, max_bytes=64)
    value = mx.arange(4, dtype=mx.float32) * 2
    result = {"embeds": (value, None), "duplicate": [value]}
    with patch.object(mx, "eval", wraps=mx.eval) as evaluate:
        cache["private prompt"] = result
        evaluate.assert_called_once()
    assert cache.stats()["bytes"] == 16
    assert cache["private prompt"] is result
    assert cache.stats()["hits"] == 1
    assert "private prompt" not in str(cache.stats())


def test_lru_entry_and_byte_limits_and_replacement():
    cache = PromptCache(max_entries=2, max_bytes=16)
    cache["a"] = mx.zeros((2,))
    cache["b"] = mx.ones((2,))
    assert "a" in cache
    assert cache["a"].size == 2
    cache["c"] = mx.ones((2,))
    assert list(cache) == ["a", "c"] and cache.stats()["evictions"] == 1
    cache["a"] = mx.zeros((4,))
    assert list(cache) == ["a"] and cache.stats()["bytes"] == 16
    cache.clear()
    assert cache.stats()["bytes"] == 0 and len(cache) == 0


def test_oversized_result_is_not_admitted():
    cache = PromptCache(max_entries=2, max_bytes=8)
    cache["large"] = mx.ones((3,))
    assert len(cache) == 0 and cache.stats()["bytes"] == 0
    with pytest.raises(KeyError):
        cache["large"]
    assert cache.stats()["misses"] == 1


def test_flux_exact_semantics_hit_across_seed_and_miss_changed_arguments():
    model = Tiny()
    binding = PromptCacheBinding(model, "flux2-klein-4b", "flux2-klein-4b")
    first = model.generate("teapot", seed=1)
    second = model.generate("teapot", seed=2)
    assert model.encodes == 1
    assert mx.array_equal(second, first + 1).item()
    model._encode_prompt_pair(prompt="teapot", negative_prompt="red", guidance=4)
    model._encode_prompt_pair(prompt="teapot", negative_prompt="blue", guidance=4)
    model._encode_prompt_pair(prompt="teapot", negative_prompt="blue", guidance=5)
    assert model.encodes == 4
    assert binding.stats()["hits"] == 1 and binding.stats()["misses"] == 4
    # Delimiter-containing strings remain distinct complete semantic arguments.
    model._encode_prompt_pair(prompt="a|NEG|b", negative_prompt="c", guidance=4)
    model._encode_prompt_pair(prompt="a", negative_prompt="b|NEG|c", guidance=4)
    assert model.encodes == 6


def test_no_parameter_leak_and_release_restores_original_attributes():
    model = Tiny()
    original_cache = model.prompt_cache
    original_encode = model._encode_prompt_pair
    parameter_names = [name for name, _ in tree_flatten(model.parameters())]
    binding = PromptCacheBinding(model, "flux2-klein-4b", "flux2-klein-4b")
    assert "prompt_cache" not in model
    model._encode_prompt_pair(prompt="hello")
    assert [name for name, _ in tree_flatten(model.parameters())] == parameter_names
    assert not any("prompt" in name for name, _ in tree_flatten(model.parameters()))
    binding.release()
    assert model.prompt_cache is original_cache and original_cache == {}
    assert model._encode_prompt_pair == original_encode
    assert "_encode_prompt_pair" not in vars(model)
    assert "_predict" not in vars(model)
    assert binding.stats()["entries"] == 0


def test_encoder_failure_does_not_insert_and_clear_preserves_wrapper():
    model = Tiny()
    binding = PromptCacheBinding(model, "flux2-klein-4b", "flux2-klein-4b")
    with pytest.raises(RuntimeError):
        model._encode_prompt_pair(prompt="fail")
    assert binding.stats()["entries"] == 0
    model._encode_prompt_pair(prompt="good")
    wrapper = model._encode_prompt_pair
    binding.clear()
    assert model._encode_prompt_pair is wrapper
    model._encode_prompt_pair(prompt="good")
    assert model.encodes == 3


def test_factory_reuse_and_transformer_change_and_clear():
    model = Tiny()
    binding = PromptCacheBinding(model, "flux2-klein-4b", "flux2-klein-4b")

    def transformer(value):
        return value + 1

    first = model._predict(transformer)
    assert model._predict(transformer) is first
    assert first(2) == 3

    def other(value):
        return value + 2

    second = model._predict(other)
    assert second is not first and second(2) == 4
    assert len(binding._factories) == 1
    assert binding.stats()["prediction_factory_builds"] == 2
    assert binding.stats()["prediction_factory_reuses"] == 1
    binding.clear()
    assert model._predict(other) is not second
    binding.release()
    assert not binding._factories


def test_zimage_prompt_boundary_and_legacy_qwen_cache_bypass():
    class Z(Tiny):
        def _encode_prompts(self, *, prompt, negative_prompt, guidance):
            self.encodes += 1
            return mx.array([len(prompt), guidance]), None

    z = Z()
    binding = PromptCacheBinding(z, "z-image-turbo", "z-image-turbo")
    first = z._encode_prompts(prompt="cat", negative_prompt=None, guidance=1)
    second = z._encode_prompts(prompt="cat", negative_prompt=None, guidance=1)
    assert first is second and z.encodes == 1
    assert "_predict" not in vars(z)
    binding.release()
    qwen = Tiny()
    qwen_binding = PromptCacheBinding(qwen, "qwen-image", "qwen-image")
    qwen.prompt_cache["a|NEG|b"] = mx.array([1])
    assert "a|NEG|b" not in qwen.prompt_cache
    assert qwen_binding.stats()["entries"] == 0


def test_backend_stats_clear_release_and_external_model_compatibility():
    model = Tiny()
    backend = MFluxBackend()
    backend._cache_bindings[id(model)] = PromptCacheBinding(
        model, "flux2-klein-4b", "flux2-klein-4b"
    )
    model._encode_prompt_pair(prompt="hello")
    assert backend.get_cache_stats(model)["entries"] == 1
    assert backend.clear_cache(model) == 1
    assert backend.get_cache_stats(model)["entries"] == 0
    backend.release(model)
    assert not backend._cache_bindings
    assert backend.get_cache_stats(model)["entries"] == 0
    assert model.prompt_cache == {}


def test_calibration_bypasses_cached_factory_and_restores_exact_wrapper(monkeypatch):
    utility = SimpleNamespace(is_m1_or_m2=lambda: False)

    class Calibration(Tiny):
        def __init__(self):
            super().__init__()
            self.transformer = lambda value: value

        @staticmethod
        def _predict(transformer):
            mode = "eager" if utility.is_m1_or_m2() else "compiled"
            return lambda value: mode

    model = Calibration()
    backend = MFluxBackend()
    backend._models[id(model)] = get_pipeline("flux2-klein-4b"), None
    binding = PromptCacheBinding(model, "flux2-klein-4b", "flux2-klein-4b")
    backend._cache_bindings[id(model)] = binding
    monkeypatch.setattr("omlx.diffusion.backend._symbol", lambda name: utility)
    original_wrapper = model._predict
    assert model._predict(model.transformer)(0) == "compiled"
    model._encode_prompt_pair(prompt="hello")
    with pytest.raises(RuntimeError), backend.calibration_context(model):
        assert model._predict(model.transformer)(0) == "eager"
        assert not binding.cache.enabled
        model._encode_prompt_pair(prompt="hello")
        model._encode_prompt_pair(prompt="hello")
        raise RuntimeError("calibration failed")
    assert model._predict is original_wrapper
    assert binding.cache.enabled and binding.stats()["entries"] == 0
    assert model._predict(model.transformer)(0) == "compiled"
    assert model.encodes == 3


def test_stats_snapshot_survives_concurrent_lru_eviction(monkeypatch):
    import sys
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    # Isolate the map race from GPU operations. Before the tuple snapshot,
    # ordinary eviction repeatedly invalidated stats' live dictionary iterator.
    monkeypatch.setattr("omlx.diffusion.cache._arrays", lambda value: [])
    binding = PromptCacheBinding(SimpleNamespace(prompt_cache={}), "dev", "dev")
    start = Event()

    def mutate():
        start.wait()
        for index in range(20000):
            binding.cache[index] = None

    def read():
        start.wait()
        for _ in range(20000):
            stats = binding.stats()
            assert stats["reference_entries"] == 0
            assert stats["bytes"] == 0

    previous = sys.getswitchinterval()
    try:
        sys.setswitchinterval(0.000001)
        with ThreadPoolExecutor(max_workers=2) as pool:
            writer = pool.submit(mutate)
            reader = pool.submit(read)
            start.set()
            writer.result(timeout=10)
            reader.result(timeout=10)
    finally:
        sys.setswitchinterval(previous)


def test_native_compile_retains_only_current_shape_dtype_and_scalar_signature():
    import weakref

    built = []

    class Compiled(Tiny):
        def _predict(self, transformer):
            self.factories += 1

            def prediction(values, *, scale=1.0, configuration=None):
                return transformer(values) * scale

            native = mx.compile(prediction)
            # mlx.gc_func itself cannot be weak-referenced; it owns the
            # Python function, whose lifetime tracks its compiled wrapper.
            built.append(weakref.ref(prediction))
            return native

    model = Compiled()
    binding = PromptCacheBinding(model, "flux2-klein-4b", "flux2-klein-4b")

    def transformer(values):
        return values + 2

    proxy = model._predict(transformer)
    for size in (2, 3, 2):
        values = mx.arange(size, dtype=mx.float32)
        actual = proxy(values, scale=1.0)
        mx.eval(actual)
        assert mx.array_equal(actual, values + 2).item()
        assert binding.stats()["prediction_signatures"] == 1
        assert sum(reference() is not None for reference in built) == 1
    assert model.factories == 3
    assert binding.stats()["prediction_shape_rebuilds"] == 2
    # New tensor contents and identity reuse the current native callable.
    assert mx.array_equal(
        proxy(mx.array([8.0, 9.0]), scale=1.0), mx.array([10.0, 11.0])
    ).item()
    assert model.factories == 3
    assert binding.stats()["prediction_signature_reuses"] == 1
    # Python static values and tensor dtype are separate specializations.
    proxy(mx.array([8.0, 9.0]), scale=2.0)
    proxy(mx.array([8, 9], dtype=mx.int32), scale=2.0)
    assert model.factories == 5
    binding.clear()
    assert binding.stats()["prediction_signatures"] == 0
    assert all(reference() is None for reference in built)


def test_mutable_prediction_arguments_bypass_retention_and_native_errors_preserved():
    model = Tiny()
    binding = PromptCacheBinding(model, "flux2-klein-4b", "flux2-klein-4b")

    class Mutable:
        value = 1

    def transformer(value):
        return value.value

    proxy = model._predict(transformer)
    state = Mutable()
    assert proxy(state) == 1
    state.value = 2
    assert proxy(state) == 2
    assert binding.stats()["prediction_signatures"] == 0
    assert binding.stats()["prediction_signature_bypasses"] == 2
    assert model.factories == 2
