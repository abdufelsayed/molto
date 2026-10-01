"""Reference cache tests use tiny arrays and fake VAE conditioning, no models."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import mlx.core as mx
import pytest
from molto_runtime.diffusion.cache import PromptCacheBinding


@dataclass(frozen=True, slots=True)
class Tiling:
    vae_encode_tiled: bool = True
    vae_encode_tile_size: int = 512


@pytest.fixture
def setup(tmp_path, monkeypatch):
    first = tmp_path / "first.png"
    first.write_bytes(b"same-image-content")
    model = SimpleNamespace(prompt_cache={})
    binding = PromptCacheBinding(model, "flux2-klein-4b/edit", "flux2-klein-4b")
    calls = []

    def compute(*, vae, tiling_config, image_paths, batch_size):
        calls.append((vae, tiling_config, image_paths, batch_size))
        assert batch_size == 1
        return mx.array([[[1.0, 2.0], [3.0, 4.0]]]), mx.array(
            [[[10, 0, 0], [10, 0, 1]]]
        )

    monkeypatch.setattr("molto_runtime.diffusion.cache._reference_compute", compute)
    return binding, first, object(), Tiling(), calls


def reference(binding, path, vae, tiling, batch_size=1):
    return binding.reference_conditioning(
        vae=vae, tiling_config=tiling, image_paths=[path], batch_size=batch_size
    )


def test_identical_content_different_temporary_path_hits_and_broadcasts(
    setup, tmp_path
):
    binding, first, vae, tiling, calls = setup
    original = reference(binding, first, vae, tiling)
    second = tmp_path / "different-upload-name.png"
    second.write_bytes(first.read_bytes())
    repeated = reference(binding, second, vae, tiling, batch_size=3)
    assert len(calls) == 1
    assert repeated[0].shape == (3, 2, 2) and repeated[1].shape == (3, 2, 3)
    assert mx.array_equal(repeated[0][0:1], original[0]).item()
    assert mx.array_equal(repeated[1][2:3], original[1]).item()
    assert binding.stats()["reference_hits"] == 1
    assert binding.stats()["reference_misses"] == 1
    assert binding.stats()["reference_entries"] == 1


def test_content_vae_tiling_and_order_change_invalidate(setup, tmp_path):
    binding, first, vae, tiling, calls = setup
    reference(binding, first, vae, tiling)
    first.write_bytes(b"edited-content")
    reference(binding, first, vae, tiling)
    reference(binding, first, vae, replace(tiling, vae_encode_tile_size=256))
    other_vae = object()
    reference(binding, first, other_vae, tiling)
    second = tmp_path / "second.png"
    second.write_bytes(b"other-reference")
    binding.reference_conditioning(
        vae=other_vae, tiling_config=tiling, image_paths=[first, second], batch_size=1
    )
    binding.reference_conditioning(
        vae=other_vae, tiling_config=tiling, image_paths=[second, first], batch_size=1
    )
    assert len(calls) == 6


def test_reference_shares_prompt_budget_and_release_drops_entries(setup):
    binding, first, vae, tiling, calls = setup
    binding.cache.max_bytes = 48  # reference = 40 bytes; prompt = 16 bytes
    binding.cache["prompt"] = mx.zeros((4,))
    reference(binding, first, vae, tiling)
    assert binding.stats()["bytes"] == 40
    assert binding.stats()["entries"] == 1 and binding.stats()["evictions"] == 1
    binding.release()
    assert binding.stats()["bytes"] == 0
    assert binding.stats()["reference_entries"] == 0
    assert binding._reference_vae is None


def test_calibration_bypass_computes_every_time_and_clears(setup):
    binding, first, vae, tiling, calls = setup
    reference(binding, first, vae, tiling)
    with binding.bypass():
        reference(binding, first, vae, tiling)
        reference(binding, first, vae, tiling)
        assert binding.stats()["entries"] == 0
    assert len(calls) == 3 and binding.stats()["entries"] == 0
    reference(binding, first, vae, tiling)
    assert len(calls) == 4


def test_file_changes_during_compute_are_not_admitted(setup, monkeypatch):
    binding, first, vae, tiling, calls = setup

    def changing(**kwargs):
        first.write_bytes(b"changed-during-encode")
        return mx.ones((1, 2, 2)), mx.ones((1, 2, 3))

    monkeypatch.setattr("molto_runtime.diffusion.cache._reference_compute", changing)
    reference(binding, first, vae, tiling)
    assert binding.stats()["reference_entries"] == 0


def test_failed_or_oversized_conditioning_not_admitted(setup, monkeypatch):
    binding, first, vae, tiling, calls = setup
    binding.cache.max_bytes = 1
    reference(binding, first, vae, tiling)
    assert binding.stats()["reference_entries"] == 0

    def fail(**kwargs):
        raise RuntimeError("VAE failed")

    monkeypatch.setattr("molto_runtime.diffusion.cache._reference_compute", fail)
    with pytest.raises(RuntimeError, match="VAE"):
        reference(binding, first, vae, tiling)
    assert binding.stats()["entries"] == 0


def test_backend_single_edit_and_batch_use_same_reference_callback(monkeypatch):
    from molto_runtime.diffusion.backend import MFluxBackend
    from molto_runtime.diffusion.registry import ImageTask, get_pipeline

    calls = []
    model = SimpleNamespace(prompt_cache={})
    spec = get_pipeline("flux2-klein-4b/edit")
    backend = MFluxBackend()
    backend._models[id(model)] = spec, None
    binding = PromptCacheBinding(model, spec.id, spec.base_model)
    backend._cache_bindings[id(model)] = binding

    def generate(model_arg, tasks, spec_arg, *, reference_conditioning):
        calls.append((model_arg, tasks, spec_arg, reference_conditioning))
        return [f"image-{task.seed}" for task in tasks]

    monkeypatch.setattr("molto_runtime.diffusion.batching.generate_batch", generate)
    task = ImageTask(prompt="edit", image_paths=("local-reference.png",), seed=1)
    assert backend.generate(model, task) == "image-1"
    assert backend.generate_batch(model, (task, replace(task, seed=2))) == [
        "image-1",
        "image-2",
    ]
    assert len(calls) == 2
    assert all(call[0] is model and call[2] is spec for call in calls)
    assert all(call[3] == binding.reference_conditioning for call in calls)
