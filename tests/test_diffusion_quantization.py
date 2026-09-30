"""Numerical MLX tests use small generated float weights, not model downloads."""

# ruff: noqa: E402 -- optional MLX must be checked before importing consumers

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
from mlx.utils import tree_flatten

from omlx.diffusion.quantization import (
    ActivationCollector,
    affine_bytes,
    quantize_transformer,
)
from omlx.quantization.affine import weighted_affine_quantize


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(64, 64)
        self.b = nn.Linear(64, 64)
        self.keep = nn.Linear(64, 64)

    def __call__(self, x):
        return self.keep(self.b(nn.silu(self.a(x))))


def collect(model):
    pipeline = SimpleNamespace(transformer=model)
    collector = ActivationCollector(max_rows=8)
    with collector.capture(pipeline):
        for _ in range(3):
            mx.eval(pipeline.transformer(mx.random.normal((2, 16, 64))))
    return collector.report()


def test_collector_preserves_output_parameters_and_observes_every_forward():
    mx.random.seed(11)
    model = TinyTransformer()
    before = dict(tree_flatten(model.parameters()))
    x = mx.random.normal((2, 17, 64))
    expected = model(x)
    pipeline = SimpleNamespace(transformer=model)
    collector = ActivationCollector(max_rows=8)
    with collector.capture(pipeline):
        for _ in range(4):
            result = pipeline.transformer(x)
            assert mx.array_equal(result, expected).item()
        with pytest.raises(ValueError, match="Restore"):
            collector.report()
    report = collector.report()
    assert pipeline.transformer is model
    assert report["transformer_calls"] == 4
    assert set(report["layers"]) == {"a", "b", "keep"}
    assert report["layers"]["a"]["rows"] <= 4 * 8
    assert report["layers"]["a"]["forward_calls"] == [1, 2, 3, 4]
    # Expected energy is computed independently from the sampled input.
    sampled = x.reshape(-1, 64)[::5]
    expected_energy = mx.mean(sampled.astype(mx.float32) ** 2, axis=0)
    assert mx.allclose(
        mx.array(report["layers"]["a"]["mean_square"]), expected_energy
    ).item()
    after = dict(tree_flatten(model.parameters()))
    assert before.keys() == after.keys()
    assert all(after[k] is v for k, v in before.items())


def test_exception_restores_wrappers():
    model = TinyTransformer()
    original = model.a
    pipeline = SimpleNamespace(transformer=model)
    collector = ActivationCollector()
    with pytest.raises(RuntimeError, match="failed"), collector.capture(pipeline):
        raise RuntimeError("failed")
    assert pipeline.transformer is model and model.a is original


def test_quantized_collection_is_allowed_but_fresh_quantization_is_rejected():
    model = TinyTransformer()
    nn.quantize(model, bits=4, group_size=64)
    report = collect(model)
    assert report["layers"]["a"]["shape"] == [64, 64]
    assert report["layers"]["a"]["source_bits"] == 4
    with pytest.raises(ValueError, match="floating-point"):
        quantize_transformer(model, report)


@pytest.mark.parametrize("bits", [3, 4, 5, 6, 8])
def test_weighted_packing_is_native_affine_and_tracks_error(bits):
    mx.random.seed(9)
    weights = mx.random.normal((8, 128)).astype(mx.float16)
    importance = mx.arange(1, 129, dtype=mx.float32)
    packed, scales, biases = weighted_affine_quantize(weights, 64, bits, importance)
    assert packed.dtype == mx.uint32
    restored = mx.dequantize(packed, scales, biases, group_size=64, bits=bits)
    assert restored.shape == weights.shape
    assert packed.nbytes + scales.nbytes + biases.nbytes == affine_bytes(
        weights.shape, bits, 64, 2
    )
    layer = nn.QuantizedLinear(128, 8, False, 64, bits)
    layer.weight, layer.scales, layer.biases = packed, scales, biases
    x = mx.random.normal((3, 128)).astype(mx.float16)
    assert mx.allclose(layer(x), x @ restored.T, atol=0.03, rtol=0.01).item()


def test_weighted_clipping_reduces_error_for_low_energy_outlier():
    weights = mx.concatenate(
        [
            mx.full((4, 1), 20.0),
            mx.broadcast_to(mx.linspace(-1, 1, 63), (4, 63)),
        ],
        axis=1,
    )
    importance = mx.concatenate([mx.array([1e-5]), mx.ones((63,))])
    packed, scales, biases = weighted_affine_quantize(weights, 64, 4, importance)
    weighted = mx.dequantize(packed, scales, biases, 64, 4)
    plain = mx.dequantize(*mx.quantize(weights, 64, 4), group_size=64, bits=4)
    weighted_error = mx.sum(importance * (weights - weighted) ** 2).item()
    plain_error = mx.sum(importance * (weights - plain) ** 2).item()
    assert weighted_error < plain_error * 0.5


def test_budget_selects_mixed_precision_and_preserves_protected_layer():
    mx.random.seed(3)
    model = TinyTransformer()
    report = collect(model)
    original = model.keep
    original_parameters = dict(tree_flatten(original.parameters()))
    baseline = 2 * (affine_bytes((64, 64), 4, 64, 4) + 64 * 4) + sum(
        w.nbytes for w in original_parameters.values()
    )
    plan = quantize_transformer(
        model, report, budget_bytes=baseline + 512, protected=("keep",)
    )
    assert plan["actual_transformer_bytes"] <= baseline + 512
    assert len({m.bits for m in (model.a, model.b)}) == 2
    assert model.keep is original
    assert plan["retained_linear_layers"] == {"keep": "explicitly protected"}
    assert plan["actual_transformer_bytes"] == sum(
        w.nbytes for _, w in tree_flatten(model.parameters())
    )
    for layer in plan["layers"].values():
        assert layer["weighted_error"] <= layer["candidate_errors"]["4"]
    assert all(
        dict(tree_flatten(model.keep.parameters()))[k] is v
        for k, v in original_parameters.items()
    )


@pytest.mark.parametrize("defect", ["missing", "shape", "nan", "zero", "rows"])
def test_bad_calibration_is_rejected_before_mutation(defect):
    model = TinyTransformer()
    report = collect(model)
    if defect == "missing":
        del report["layers"]["b"]
    elif defect == "shape":
        report["layers"]["b"]["shape"] = [1, 64]
    elif defect == "nan":
        report["layers"]["b"]["mean_square"][0] = float("nan")
    elif defect == "zero":
        report["layers"]["b"]["mean_square"] = [0.0] * 64
    else:
        report["layers"]["b"]["rows"] = 0
    original = model.a
    with pytest.raises(ValueError, match="calibration|energy"):
        quantize_transformer(model, report)
    assert model.a is original


def test_too_small_budget_and_unmatched_protection_fail_without_mutation():
    model = TinyTransformer()
    report = collect(model)
    with pytest.raises(ValueError, match="below"):
        quantize_transformer(model, report, budget_bytes=1)
    with pytest.raises(ValueError, match="matches no"):
        quantize_transformer(model, report, protected=("typo",))
    assert isinstance(model.a, nn.Linear)


def test_native_mflux_disk_save_reload_preserves_mixed_precision(tmp_path):
    pytest.importorskip("mflux")
    from mflux.models.common.weights.loading.weight_applier import WeightApplier
    from mflux.models.common.weights.loading.weight_definition import (
        ComponentDefinition,
    )
    from mflux.models.common.weights.loading.weight_loader import WeightLoader
    from mflux.models.common.weights.saving.model_saver import ModelSaver

    mx.random.seed(13)
    model = TinyTransformer()
    report = collect(model)
    original_bytes = sum(w.nbytes for _, w in tree_flatten(model.parameters()))
    baseline = original_bytes - 2 * (64 * 64 * 4 - affine_bytes((64, 64), 4, 64, 4))
    quantize_transformer(
        model, report, budget_bytes=baseline + 512, protected=("keep",)
    )
    ModelSaver._save_weights(str(tmp_path), 4, model, "transformer")
    component = ComponentDefinition("transformer", "transformer")
    stored = WeightLoader.load_single_local(component, tmp_path)
    assert stored.meta_data.quantization_level == 4
    reloaded = TinyTransformer()
    WeightApplier.apply_and_quantize_single(stored, reloaded, component, None)
    assert reloaded.a.bits == model.a.bits
    assert reloaded.b.bits == model.b.bits
    assert isinstance(reloaded.keep, nn.Linear)
    before = dict(tree_flatten(model.parameters()))
    after = dict(tree_flatten(reloaded.parameters()))
    assert before.keys() == after.keys()
    assert all(mx.array_equal(before[k], after[k]).item() for k in before)
    x = mx.random.normal((2, 64))
    assert mx.array_equal(model(x), reloaded(x)).item()
