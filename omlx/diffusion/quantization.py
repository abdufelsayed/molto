"""Offline transformer calibration and activation-weighted quantization.

The collector accepts float or packed affine linears. Fresh quantization accepts
only floating-point transformer parameters and retains all other components.
Layer error is a diagonal activation-energy proxy, not an image-quality score.
"""

from __future__ import annotations

import fnmatch
import math
from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from omlx.quantization.affine import weighted_affine_quantize

GROUP_SIZES = (32, 64, 128)
BITS = (3, 4, 5, 6, 8)


def _linear_shape(module):
    output, width = module.weight.shape
    if isinstance(module, nn.QuantizedLinear):
        width = width * 32 // module.bits
    return output, width


class _CaptureLinear(nn.Module):
    def __init__(self, inner, name, collector):
        super().__init__()
        # Temporary instrumentation must never rename or serialize parameters.
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "collector", collector)
        object.__setattr__(self, "name", name)

    def __call__(self, *args, **kwargs):
        x = args[0] if args else kwargs.get("x")
        if x is None:
            raise ValueError(f"Cannot capture linear input for {self.name}")
        self.collector.record(self.name, self.inner, x)
        return self.inner(*args, **kwargs)


class _CaptureTransformer(nn.Module):
    def __init__(self, inner, collector):
        super().__init__()
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "collector", collector)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)

    def __call__(self, *args, **kwargs):
        # Keep every forward, including separate CFG branches. Do not interpret
        # family-specific timestep tensors as scheduler step numbers.
        self.collector.calls += 1
        return self.inner(*args, **kwargs)


class ActivationCollector:
    """Sample a bounded number of activation rows on every transformer call."""

    def __init__(self, max_rows: int = 256):
        if type(max_rows) is not int or max_rows <= 0:
            raise ValueError("max_rows must be a positive integer")
        self.max_rows = max_rows
        self.calls = 0
        self.entries = {}
        self._active = False

    def record(self, name, module, x):
        output, width = _linear_shape(module)
        if x.shape[-1] != width:
            raise ValueError(f"Activation width disagrees with {name}")
        rows = x.reshape(-1, width)
        stride = max(1, math.ceil(rows.shape[0] / self.max_rows))
        rows = rows[::stride].astype(mx.float32)
        energy = mx.sum(rows * rows, axis=0)
        mx.eval(energy)
        entry = self.entries.setdefault(
            name,
            {
                "shape": [output, width],
                "rows": 0,
                "observations": 0,
                "energy": mx.zeros((width,), dtype=mx.float32),
                "forward_calls": set(),
                "source_bits": getattr(module, "bits", None),
            },
        )
        entry["energy"] = entry["energy"] + energy
        mx.eval(entry["energy"])
        entry["rows"] += rows.shape[0]
        entry["observations"] += 1
        entry["forward_calls"].add(self.calls)

    @contextmanager
    def capture(self, pipeline):
        if self._active:
            raise ValueError("Collector is already installed")
        transformer = pipeline.transformer
        originals = [
            (name, module)
            for name, module in transformer.named_modules()
            if name and isinstance(module, (nn.Linear, nn.QuantizedLinear))
        ]
        if not originals:
            raise ValueError("Transformer has no supported linear modules")
        self._active = True
        try:
            transformer.update_modules(
                tree_unflatten(
                    [
                        (name, _CaptureLinear(module, name, self))
                        for name, module in originals
                    ]
                )
            )
            pipeline.transformer = _CaptureTransformer(transformer, self)
            yield self
        finally:
            pipeline.transformer = transformer
            transformer.update_modules(tree_unflatten(originals))
            self._active = False

    def report(self):
        if self._active:
            raise ValueError("Restore calibration wrappers before exporting")
        if not self.calls or not self.entries:
            raise ValueError("Calibration produced no transformer observations")
        layers = {}
        for name, entry in self.entries.items():
            values = (entry["energy"] / entry["rows"]).tolist()
            if not all(math.isfinite(v) and v >= 0 for v in values):
                raise ValueError(f"Nonfinite activation energy in {name}")
            layers[name] = {
                **{
                    k: v
                    for k, v in entry.items()
                    if k not in ("energy", "forward_calls")
                },
                "mean_square": values,
                "forward_calls": sorted(entry["forward_calls"]),
            }
        return {
            "version": 1,
            "method": "diffusion-input-energy",
            "max_rows_per_observation": self.max_rows,
            "transformer_calls": self.calls,
            "layers": layers,
        }


def _importance(report, name, shape):
    if report.get("version") != 1 or report.get("method") != "diffusion-input-energy":
        raise ValueError("Unsupported diffusion calibration report")
    entry = report.get("layers", {}).get(name)
    if not isinstance(entry, dict) or entry.get("shape") != list(shape):
        raise ValueError(f"Missing or incompatible calibration for {name}")
    values = entry.get("mean_square")
    if (
        not isinstance(values, list)
        or len(values) != shape[-1]
        or type(entry.get("rows")) is not int
        or entry["rows"] <= 0
        or not all(
            type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values
        )
        or not any(v > 0 for v in values)
    ):
        raise ValueError(f"Invalid activation energy for {name}")
    # Normalize magnitudes without erasing relative channel importance.
    imp = mx.array(values, dtype=mx.float32)
    return mx.maximum(imp / mx.mean(imp), mx.array(1e-8, mx.float32))


def affine_bytes(shape, bits, group_size, dtype_bytes):
    output, width = shape
    return output * width * bits // 8 + 2 * output * (width // group_size) * dtype_bytes


def _pack(weight, importance, bits, group_size):
    # Row chunks bound clipping-search temporaries independently of model size.
    chunk_rows = max(1, (1 << 18) // weight.shape[-1])
    pieces = []
    error = 0.0
    for start in range(0, weight.shape[0], chunk_rows):
        w = weight[start : start + chunk_rows]
        packed, scales, biases = weighted_affine_quantize(
            w, group_size, bits, importance
        )
        restored = mx.dequantize(packed, scales, biases, group_size, bits)
        loss = mx.sum(
            importance * (w.astype(mx.float32) - restored.astype(mx.float32)) ** 2
        )
        mx.eval(loss)
        error += float(loss.item())
        pieces.append((packed, scales, biases))
    tensors = tuple(mx.concatenate([p[i] for p in pieces], axis=0) for i in range(3))
    mx.eval(tensors)
    if not math.isfinite(error):
        raise ValueError("Nonfinite weighted reconstruction error")
    return tensors, error


def quantize_transformer(
    transformer,
    calibration,
    *,
    bits=4,
    group_size=64,
    budget_bytes=None,
    budget_ratio=1.10,
    protected=(),
):
    """Allocate upgrades by measured weighted-error reduction per added byte.

    The budget covers transformer parameter tensors, including retained float
    parameters. It excludes other components, file headers and runtime memory.
    Coverage and budget validation finish before any module is replaced.
    """
    if type(bits) is not int or bits not in BITS:
        raise ValueError("bits must be 3, 4, 5, 6, or 8")
    if type(group_size) is not int or group_size not in GROUP_SIZES:
        raise ValueError("group_size must be 32, 64, or 128")
    if not math.isfinite(budget_ratio) or budget_ratio < 1:
        raise ValueError("budget_ratio must be finite and at least 1")
    if budget_bytes is not None and (
        type(budget_bytes) is not int or budget_bytes <= 0
    ):
        raise ValueError("budget_bytes must be a positive integer")
    parameters = dict(tree_flatten(transformer.parameters()))
    if not parameters or any(
        not mx.issubdtype(w.dtype, mx.floating) for w in parameters.values()
    ):
        raise ValueError(
            "Fresh quantization requires floating-point transformer weights"
        )
    modules = dict(transformer.named_modules())
    linear = {
        name: m for name, m in modules.items() if name and isinstance(m, nn.Linear)
    }
    matched = {
        pattern
        for pattern in protected
        if any(fnmatch.fnmatchcase(n, pattern) for n in linear)
    }
    if matched != set(protected):
        raise ValueError("A protected-layer pattern matches no linear module")
    selected, retained = {}, {}
    original_bytes = sum(w.nbytes for w in parameters.values())
    baseline = original_bytes
    for name, module in linear.items():
        shape = tuple(module.weight.shape)
        reason = None
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in protected):
            reason = "explicitly protected"
        elif shape[-1] % group_size:
            reason = "input width is not divisible by group size"
        if reason:
            retained[name] = reason
            continue
        imp = _importance(calibration, name, shape)
        size = affine_bytes(shape, bits, group_size, module.weight.itemsize)
        baseline += size - module.weight.nbytes
        selected[name] = {
            "module": module,
            "importance": imp,
            "bits": bits,
            "bytes": size,
            "errors": {},
        }
    if not selected:
        raise ValueError("No eligible floating-point linear layers")
    budget = (
        budget_bytes
        if budget_bytes is not None
        else math.floor(baseline * budget_ratio)
    )
    if budget < baseline:
        raise ValueError(
            f"Transformer budget {budget} is below the base allocation {baseline}"
        )
    candidates = [b for b in BITS if b >= bits]
    for entry in selected.values():
        for candidate in candidates:
            tensors, error = _pack(
                entry["module"].weight, entry["importance"], candidate, group_size
            )
            entry["errors"][candidate] = error
            del tensors
    total = baseline
    while True:
        upgrades = []
        for name, entry in selected.items():
            for candidate in candidates:
                if candidate <= entry["bits"]:
                    continue
                size = affine_bytes(
                    entry["module"].weight.shape,
                    candidate,
                    group_size,
                    entry["module"].weight.itemsize,
                )
                added = size - entry["bytes"]
                gain = entry["errors"][entry["bits"]] - entry["errors"][candidate]
                if gain > 0 and total + added <= budget:
                    upgrades.append((gain / added, name, candidate, size))
        if not upgrades:
            break
        _, name, candidate, size = max(upgrades)
        entry = selected[name]
        total += size - entry["bytes"]
        entry.update(bits=candidate, bytes=size)
    replacements, layers = [], {}
    for name, entry in selected.items():
        module = entry["module"]
        tensors, _ = _pack(
            module.weight, entry["importance"], entry["bits"], group_size
        )
        output, width = module.weight.shape
        q = nn.QuantizedLinear(width, output, False, group_size, entry["bits"])
        q.weight, q.scales, q.biases = tensors
        if "bias" in module:
            q.bias = module.bias
        replacements.append((name, q))
        layers[name] = {
            "shape": [output, width],
            "bits": entry["bits"],
            "group_size": group_size,
            "weight_bytes": entry["bytes"],
            "weighted_error": entry["errors"][entry["bits"]],
            "candidate_errors": {str(b): e for b, e in entry["errors"].items()},
        }
    transformer.update_modules(tree_unflatten(replacements))
    actual = sum(w.nbytes for _, w in tree_flatten(transformer.parameters()))
    if actual != total or actual > budget:
        transformer.update_modules(
            tree_unflatten([(n, e["module"]) for n, e in selected.items()])
        )
        raise ValueError("Actual transformer allocation disagrees with the budget")
    return {
        "version": 1,
        "method": "diffusion-oQe",
        "nominal_bits": bits,
        "group_size": group_size,
        "original_transformer_bytes": original_bytes,
        "base_transformer_bytes": baseline,
        "budget_bytes": budget,
        "actual_transformer_bytes": actual,
        "layers": layers,
        "retained_linear_layers": retained,
        "error_metric": "diagonal input-energy weighted weight reconstruction; not image quality",
        "allocation": "greedy weighted-error reduction per added byte",
    }
