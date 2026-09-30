"""Transactional workflow tests use miniature MLX modules and fake pipelines."""

# ruff: noqa: E402 -- import optional runtime before importing its consumers
import json
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mflux")
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mlx.utils import tree_flatten
from PIL import Image

from omlx.diffusion import preparation
from omlx.diffusion.backend import MFluxBackend
from omlx.diffusion.checkpoint import detect_checkpoint
from omlx.diffusion.quantization import ActivationCollector
from omlx.diffusion.registry import ImageTask, get_pipeline


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)

    def __call__(self, x):
        return self.linear(x)


class TinyPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = TinyTransformer()
        self.text_encoder = nn.Linear(64, 64)
        self.vae = nn.Linear(64, 64)
        self.bits = None

    def save_model(self, path):
        definition = SimpleNamespace(
            get_components=lambda: [
                ComponentDefinition(name, name)
                for name in ("transformer", "text_encoder", "vae")
            ],
            get_tokenizers=lambda: [],
        )
        ModelSaver.save_model(self, self.bits, str(path), definition)

    def generate_image(self, **kwargs):
        for _ in range(kwargs["num_inference_steps"]):
            mx.eval(self.transformer(mx.ones((4, 64))))
        return SimpleNamespace(
            image=Image.new("RGB", (kwargs["width"], kwargs["height"]))
        )


@pytest.fixture
def tiny_source(tmp_path, monkeypatch):
    native = TinyPipeline()
    source = tmp_path / "source"
    native.save_model(source)
    tokenizer = source / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")
    spec = get_pipeline("flux2-klein-4b")
    (source / "omlx-mflux.json").write_text(
        json.dumps(
            {
                "version": 2,
                "backend": "mflux",
                "base_model": spec.base_model,
                "pipeline_id": spec.id,
                "format": "mflux",
                "quantization_bits": None,
                "components": list(spec.components),
            }
        )
    )
    assert detect_checkpoint(source) is not None

    def load(backend, checkpoint, pipeline_id):
        backend._models[id(native)] = (spec, source)
        return native

    monkeypatch.setattr(MFluxBackend, "load", load)
    # This toy has no FLUX prediction factory. Numerical and native disk behavior
    # remain real; the real FLUX calibration is a separate opt-in verification.
    from contextlib import nullcontext

    monkeypatch.setattr(
        MFluxBackend, "calibration_context", lambda self, model: nullcontext()
    )
    monkeypatch.setattr(preparation, "_admit", lambda provenance: {})
    return source, native


def calibration_for(native):
    collector = ActivationCollector()
    with collector.capture(native):
        mx.eval(native.transformer(mx.ones((4, 64))))
    report = collector.report()
    report["base_model"] = "flux2-klein-4b"
    return report


def test_local_calibration_records_resolved_tasks_and_rejects_overwrite(
    tiny_source, tmp_path
):
    source, native = tiny_source
    output = tmp_path / "calibration.json"
    report = preparation.calibrate_diffusion(
        source, output, [ImageTask(prompt="A teapot", width=256, height=256)]
    )
    assert report["transformer_calls"] == 4
    assert report["resolved_tasks"][0]["num_inference_steps"] == 4
    assert report["source"]["stored_quantization_bits"] is None
    assert len(report["outputs"][0]["pixel_sha256"]) == 64
    assert isinstance(native.transformer.linear, nn.Linear)
    original = output.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        preparation.calibrate_diffusion(source, output, [ImageTask(prompt="Other")])
    assert output.read_bytes() == original


def test_complete_workflow_preserves_source_and_other_components(tiny_source, tmp_path):
    source, native = tiny_source
    original_files = {
        str(p.relative_to(source)): p.read_bytes()
        for p in source.rglob("*")
        if p.is_file()
    }
    text = dict(tree_flatten(native.text_encoder.parameters()))
    vae = dict(tree_flatten(native.vae.parameters()))
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(calibration_for(native)))
    output = tmp_path / "saved"
    plan = preparation.quantize_diffusion(source, calibration, output, budget_ratio=1)
    checkpoint = detect_checkpoint(output)
    assert checkpoint.quantization == 4
    assert plan["actual_transformer_bytes"] == plan["budget_bytes"]
    manifest = json.loads((output / "omlx-mflux.json").read_text())
    assert manifest["quantization_policy"]["method"] == "diffusion-oQe"
    assert (output / "diffusion-quantization.json").is_file()
    assert (output / "tokenizer" / "tokenizer.json").read_text() == "{}"
    assert all(
        (source / name).read_bytes() == content
        for name, content in original_files.items()
    )
    for component, parameters in (("text_encoder", text), ("vae", vae)):
        saved = mx.load(str(output / component / "0.safetensors"))
        assert all(mx.array_equal(saved[k], v).item() for k, v in parameters.items())


def test_float_restoration_snapshot_released_before_cache_clear(
    tiny_source, tmp_path, monkeypatch
):
    import weakref

    source, native = tiny_source
    floating_layer = weakref.ref(native.transformer.linear)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(calibration_for(native)))
    checked = []

    def clear():
        checked.append(True)
        assert floating_layer() is None

    monkeypatch.setattr(mx, "clear_cache", clear)
    preparation.quantize_diffusion(
        source, calibration, tmp_path / "saved", budget_ratio=1
    )
    assert checked


def test_save_failure_preserves_output_and_removes_staging(
    tiny_source, tmp_path, monkeypatch
):
    source, native = tiny_source
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(calibration_for(native)))
    output = tmp_path / "saved"
    output.mkdir()

    def fail(*args, **kwargs):
        raise RuntimeError("save failed")

    monkeypatch.setattr(MFluxBackend, "save", fail)
    with pytest.raises(RuntimeError, match="save failed"):
        preparation.quantize_diffusion(source, calibration, output)
    assert output.is_dir() and not list(output.iterdir())
    assert not list(tmp_path.glob(".saved-*"))


@pytest.mark.parametrize(
    "source", ["Qwen/Qwen-Image-2.1", "flux2-klein-4b", "/missing/checkpoint"]
)
def test_no_acquisition_for_nonlocal_sources(source, tmp_path, monkeypatch):
    import huggingface_hub

    def forbidden(*args, **kwargs):
        pytest.fail("Attempted download")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden)
    with pytest.raises(ValueError, match="local checkpoint"):
        preparation.calibrate_diffusion(
            source, tmp_path / "report.json", [ImageTask(prompt="A teapot")]
        )
    with pytest.raises(ValueError, match="local checkpoint"):
        preparation.quantize_diffusion(
            source, tmp_path / "calibration.json", tmp_path / "saved"
        )


def test_packed_source_rejected_before_loading(tiny_source, tmp_path, monkeypatch):
    source, native = tiny_source
    nn.quantize(native.transformer, bits=4, group_size=64)
    native.bits = 4
    native.save_model(source)
    manifest = source / "omlx-mflux.json"
    data = json.loads(manifest.read_text())
    data["quantization_bits"] = 4
    manifest.write_text(json.dumps(data))
    monkeypatch.setattr(
        MFluxBackend, "load", lambda *args: pytest.fail("Should reject before loading")
    )
    with pytest.raises(ValueError, match="floating-point source"):
        preparation.quantize_diffusion(
            source, tmp_path / "missing-report", tmp_path / "saved"
        )
    assert not (tmp_path / "saved").exists()


def test_bad_task_rejected_before_loading(tiny_source, tmp_path, monkeypatch):
    source, _ = tiny_source
    monkeypatch.setattr(
        MFluxBackend, "load", lambda *args: pytest.fail("Should reject before loading")
    )
    with pytest.raises(ValueError, match="negative_prompt"):
        preparation.calibrate_diffusion(
            source,
            tmp_path / "report.json",
            [ImageTask(prompt="A teapot", negative_prompt="Blurry")],
        )


def test_output_cannot_modify_source(tiny_source):
    source, _ = tiny_source
    with pytest.raises(ValueError, match="separate"):
        preparation.calibrate_diffusion(
            source, source / "report.json", [ImageTask(prompt="A teapot")]
        )


def test_negative_cli_seed_is_not_silently_wrapped():
    args = SimpleNamespace(command="diffusion-calibrate", seed=-1)
    with pytest.raises(ValueError, match="seed"):
        preparation.cli_command(args)


def test_native_flux_prediction_is_eager_counts_cfg_and_restores_on_failure(
    monkeypatch,
):
    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
    from mflux.utils.apple_silicon import AppleSiliconUtil

    class KeywordTransformer(TinyTransformer):
        def __call__(self, hidden_states, **kwargs):
            return self.linear(hidden_states)

    class NativeFactoryPipeline(nn.Module):
        _predict = staticmethod(Flux2Klein._predict)

        def __init__(self):
            super().__init__()
            self.transformer = KeywordTransformer()

    native = NativeFactoryPipeline()
    original = native.transformer
    backend = MFluxBackend()
    backend._models[id(native)] = (get_pipeline("flux2-klein-4b"), None)
    collector = ActivationCollector()
    monkeypatch.setattr(AppleSiliconUtil, "is_m1_or_m2", lambda: False)
    monkeypatch.setattr(
        mx, "compile", lambda *a, **kw: pytest.fail("Calibration must stay eager")
    )
    with (
        pytest.raises(RuntimeError, match="interrupted"),
        collector.capture(native),
        backend.calibration_context(native),
    ):
        predict = native._predict(native.transformer)
        for _ in range(4):
            mx.eval(
                predict(
                    latents=mx.ones((4, 64)),
                    latent_ids=mx.zeros((4, 3)),
                    prompt_embeds=mx.ones((4, 64)),
                    text_ids=mx.zeros((4, 3)),
                    negative_prompt_embeds=mx.zeros((4, 64)),
                    negative_text_ids=mx.zeros((4, 3)),
                    guidance=2,
                    timestep=mx.array([0.5]),
                )
            )
        raise RuntimeError("interrupted")
    assert native.transformer is original
    assert "_predict" not in vars(native)
    assert not AppleSiliconUtil.is_m1_or_m2()
    assert collector.report()["transformer_calls"] == 8
    assert collector.report()["layers"]["linear"]["observations"] == 8


class CancelledError(Exception):
    pass


def source_files(source):
    return {
        str(path.relative_to(source)): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }


def test_calibration_cancellation_between_forwards_restores_model(
    tiny_source, tmp_path, monkeypatch
):
    source, native = tiny_source
    original = native.transformer
    linear = original.linear
    before = source_files(source)
    released = []
    monkeypatch.setattr(
        MFluxBackend, "release", lambda self, model: released.append(model)
    )

    def observe(phase, progress, detail):
        if phase == "calibrating" and detail == "Transformer forward 1":
            raise CancelledError("cancel calibration")

    output = tmp_path / "cancelled-report.json"
    with pytest.raises(CancelledError):
        preparation.calibrate_diffusion(
            source, output, [ImageTask(prompt="A teapot")], observer=observe
        )
    assert native.transformer is original and original.linear is linear
    assert released == [native]
    assert not output.exists()
    assert source_files(source) == before


def test_calibration_cancel_at_report_publication_removes_temporary(
    tiny_source, tmp_path
):
    source, native = tiny_source
    original = native.transformer.linear
    output = tmp_path / "report.json"

    def observe(phase, progress, detail):
        if detail == "Publishing calibration report":
            assert not output.exists()
            raise CancelledError()

    with pytest.raises(CancelledError):
        preparation.calibrate_diffusion(
            source, output, [ImageTask(prompt="A teapot")], observer=observe
        )
    assert not output.exists() and not list(tmp_path.glob(".report.json-*"))
    assert native.transformer.linear is original


@pytest.mark.parametrize(
    "boundary",
    [
        "Transformer quantization finished",
        "Native checkpoint saved",
        "Publishing complete checkpoint",
    ],
)
def test_quantization_cancellation_restores_native_and_cleans_staging(
    tiny_source, tmp_path, monkeypatch, boundary
):
    source, native = tiny_source
    original = native.transformer.linear
    before = source_files(source)
    released = []
    monkeypatch.setattr(
        MFluxBackend, "release", lambda self, model: released.append(model)
    )
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(calibration_for(native)))
    output = tmp_path / "saved"
    output.mkdir()

    def observe(phase, progress, detail):
        if detail == boundary:
            raise CancelledError(boundary)

    with pytest.raises(CancelledError):
        preparation.quantize_diffusion(source, calibration, output, observer=observe)
    assert native.transformer.linear is original and native.bits is None
    assert released == [native]
    assert output.is_dir() and not list(output.iterdir())
    assert not list(tmp_path.glob(".saved-*"))
    assert source_files(source) == before


def test_workflow_progress_is_monotonic_and_never_notifies_after_publication(
    tiny_source, tmp_path
):
    source, native = tiny_source
    calibration = tmp_path / "calibration.json"
    calibration_events = []

    def observe_calibration(*event):
        assert not calibration.exists()
        calibration_events.append(event)

    preparation.calibrate_diffusion(
        source,
        calibration,
        [ImageTask(prompt="A teapot")],
        observer=observe_calibration,
    )
    output = tmp_path / "saved"
    quantization_events = []

    def observe_quantization(*event):
        assert not output.exists()
        quantization_events.append(event)

    preparation.quantize_diffusion(
        source, calibration, output, observer=observe_quantization
    )
    for events, operation in (
        (calibration_events, "calibrating"),
        (quantization_events, "quantizing"),
    ):
        values = [event[1] for event in events]
        assert values == sorted(values) and all(0 <= value < 1 for value in values)
        assert set(event[0] for event in events) == {"loading", operation, "saving"}
        assert values[-1] == 0.99


def test_memory_admission_respects_pool_remaining_capacity(monkeypatch):
    import psutil

    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(available=1000)
    )
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 900}
    )
    monkeypatch.setattr(mx, "get_active_memory", lambda: 100)
    admission = preparation._admit({"weight_file_bytes": 60}, memory_limit_bytes=100)
    assert admission["available_capacity_bytes"] == 100
    assert admission["memory_limit_bytes"] == 100
    assert admission["model_fraction_limit"] == 0.60
    with pytest.raises(ValueError, match="memory allowance"):
        preparation._admit({"weight_file_bytes": 61}, memory_limit_bytes=100)
    assert (
        preparation._admit({"weight_file_bytes": 100})["available_capacity_bytes"]
        == 800
    )


def test_workflows_forward_memory_ceiling(tiny_source, tmp_path, monkeypatch):
    source, native = tiny_source
    calls = []

    def admit(provenance, *, memory_limit_bytes):
        calls.append(memory_limit_bytes)
        return {"memory_limit_bytes": memory_limit_bytes}

    monkeypatch.setattr(preparation, "_admit", admit)

    def cancel(phase, progress, detail):
        if detail == "Loading native checkpoint":
            raise CancelledError()

    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps(calibration_for(native)))
    with pytest.raises(CancelledError):
        preparation.calibrate_diffusion(
            source,
            tmp_path / "report.json",
            [ImageTask(prompt="Test")],
            observer=cancel,
            memory_limit_bytes=100,
        )
    with pytest.raises(CancelledError):
        preparation.quantize_diffusion(
            source,
            calibration,
            tmp_path / "saved",
            observer=cancel,
            memory_limit_bytes=200,
        )
    assert calls == [100, 200]
