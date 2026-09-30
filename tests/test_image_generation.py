import argparse
import base64
import json
import types
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import pytest
from fastapi import HTTPException
from PIL import Image

from omlx.api import image_routes
from omlx.api.image_models import ImageGenerationRequest
from omlx.cli import mflux_save_command
from omlx.engine import image_generation
from omlx.engine.image_generation import MFluxImageEngine
from omlx.engine_pool import EnginePool
from omlx.model_discovery import discover_models, is_mflux_image_model_dir


def _write_mflux_layout(path):
    for name in ("vae", "transformer", "text_encoder"):
        component = path / name
        component.mkdir(parents=True, exist_ok=True)
        (component / "0.safetensors").write_bytes(b"weights")
    (path / "tokenizer").mkdir(exist_ok=True)
    (path / "tokenizer" / "tokenizer.json").write_text("{}")
    (path / "omlx-mflux.json").write_text(
        json.dumps(
            {
                "version": 1,
                "backend": "mflux",
                "model_family": "z-image-turbo",
                "quantize": None,
            }
        )
    )


def test_discovers_mflux_saved_z_image_checkpoint(tmp_path):
    model_path = tmp_path / "z-image-turbo-8bit"
    _write_mflux_layout(model_path)

    assert is_mflux_image_model_dir(model_path)
    discovered = discover_models(tmp_path)
    model = discovered["z-image-turbo-8bit"]
    assert model.model_type == "image_generation"
    assert model.engine_type == "image_generation"
    assert model.config_model_type == "z_image_turbo"
    assert model.estimated_size > 0


def test_engine_pool_path_guard_accepts_mflux_checkpoint_without_root_config(tmp_path):
    model_path = tmp_path / "z-image-turbo-4bit"
    _write_mflux_layout(model_path)
    pool = EnginePool()
    pool._get_final_ceiling = lambda: 0
    pool.discover_models(str(tmp_path))

    entry = pool.get_entry("z-image-turbo-4bit")

    assert entry is not None
    pool._raise_if_model_path_missing_locked("z-image-turbo-4bit", entry)
    assert pool.get_entry("z-image-turbo-4bit") is entry


def test_manifest_allows_a_custom_mflux_directory_name(tmp_path):
    model_path = tmp_path / "my-image-model"
    _write_mflux_layout(model_path)
    (model_path / "omlx-mflux.json").write_text(
        json.dumps(
            {
                "version": 1,
                "backend": "mflux",
                "model_family": "z-image-turbo",
                "quantize": None,
            }
        )
    )

    assert is_mflux_image_model_dir(model_path)
    assert discover_models(tmp_path)["my-image-model"].model_type == "image_generation"


def test_discovers_mflux_checkpoint_in_huggingface_cache(tmp_path):
    cache_entry = tmp_path / "models--filipstrand--Z-Image-Turbo-mflux-4bit"
    model_path = cache_entry / "snapshots" / "commit-hash"
    _write_mflux_layout(model_path)

    discovered = discover_models(tmp_path)

    model = discovered["filipstrand--Z-Image-Turbo-mflux-4bit"]
    assert model.model_type == "image_generation"
    assert model.source_repo_id == "filipstrand/Z-Image-Turbo-mflux-4bit"


@pytest.mark.asyncio
async def test_image_endpoint_returns_base64_png_and_sequential_seeds(monkeypatch):
    engine = MFluxImageEngine("/tmp/model")
    calls = []

    async def generate_image(**kwargs):
        calls.append(kwargs)
        return b"png"

    engine.generate_image = generate_image

    monkeypatch.setattr(
        image_routes,
        "detect_checkpoint",
        lambda path: types.SimpleNamespace(
            base_model="z-image-turbo", default_pipeline="z-image-turbo"
        ),
    )

    class Pool:
        def get_entry(self, model_id):
            return types.SimpleNamespace(
                model_path="/tmp/model", engine_type="image_generation"
            )

        @asynccontextmanager
        async def acquire(self, model_id, *, image_pipeline=None):
            assert model_id == "image-model"
            yield engine

    monkeypatch.setattr(image_routes, "_get_engine_pool", lambda: Pool())
    monkeypatch.setattr(image_routes, "_resolve_model", lambda model_id: model_id)

    response = await image_routes.create_image(
        ImageGenerationRequest(
            model="image-model",
            prompt="A puffin",
            n=2,
            size="512x768",
            seed=41,
            steps=4,
        )
    )

    assert [item.seed for item in response.data] == [41, 42]
    assert [item.b64_json for item in response.data] == [
        base64.b64encode(b"png").decode("ascii"),
        base64.b64encode(b"png").decode("ascii"),
    ]
    assert [(call["width"], call["height"]) for call in calls] == [
        (512, 768),
        (512, 768),
    ]


@pytest.mark.asyncio
async def test_image_engine_serializes_mflux_generated_image_wrapper(monkeypatch):
    class Model:
        def generate_image(self, **kwargs):
            return types.SimpleNamespace(image=Image.new("RGB", (16, 16), "red"))

    engine = MFluxImageEngine("/tmp/model")
    engine._model = Model()
    engine._started = True
    engine._checkpoint = types.SimpleNamespace(
        base_model="z-image-turbo", default_pipeline="z-image-turbo"
    )
    engine._pipeline = image_generation.get_pipeline("z-image-turbo")

    async def finish_activity(activity_id):
        engine._end_activity(activity_id)

    monkeypatch.setattr(engine, "_finish_activity", finish_activity)
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(image_generation, "get_mlx_executor", lambda: executor)
        png = await engine.generate_image(
            prompt="A red square",
            seed=1,
            width=256,
            height=256,
            steps=1,
        )

    assert png.startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("size", ["1024", "255x512", "513x512", "4096x4096"])
def test_image_endpoint_rejects_invalid_sizes(size):
    with pytest.raises(HTTPException) as exc_info:
        image_routes._parse_size(size)
    assert exc_info.value.status_code == 400


def test_mflux_save_writes_discovery_manifest(tmp_path, monkeypatch):
    captured = {}

    from omlx import diffusion

    source = tmp_path / "source"
    _write_mflux_layout(source)

    class Backend:
        def instantiate(self, base_model, **kwargs):
            captured.update(base_model=base_model, **kwargs)
            return object()

        def save(self, model, output, **kwargs):
            _write_mflux_layout(output)
            (output / "transformer" / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"quantization_level": 4},
                        "weight_map": {"tensor": "0.safetensors"},
                    }
                )
            )
            (output / "omlx-mflux.json").write_text(
                json.dumps(
                    {
                        "version": 2,
                        "backend": "mflux",
                        "base_model": "z-image-turbo",
                        "pipeline_id": "z-image-turbo",
                        "quantization_bits": 4,
                        "format": "mflux",
                        "components": ["vae", "transformer", "text_encoder"],
                    }
                )
            )

    monkeypatch.setattr(diffusion, "MFluxBackend", Backend)
    output = tmp_path / "converted"
    args = argparse.Namespace(output=str(output), model=str(source), quantize=4)
    assert mflux_save_command(args) == 0
    assert captured == {
        "base_model": "z-image-turbo",
        "model_path": str(source.resolve()),
        "quantization": 4,
        "pipeline_id": "z-image-turbo",
    }
    manifest = json.loads((output / "omlx-mflux.json").read_text())
    assert manifest["backend"] == "mflux"
    assert manifest["base_model"] == "z-image-turbo"
    assert manifest["quantization_bits"] == 4


@pytest.mark.asyncio
async def test_stopping_image_engine_releases_model_and_clears_mlx_cache(monkeypatch):
    calls = []
    engine = MFluxImageEngine("/tmp/model")
    engine._model = object()

    monkeypatch.setattr(
        image_generation.mx, "synchronize", lambda: calls.append("sync")
    )
    monkeypatch.setattr(
        image_generation.mx, "clear_cache", lambda: calls.append("clear")
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(image_generation, "get_mlx_executor", lambda: executor)
        await engine.stop()

    assert engine._model is None
    assert calls == ["sync", "clear"]
