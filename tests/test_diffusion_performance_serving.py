"""Batch admission, request ordering, and cache lifecycle without model weights."""

import asyncio
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from PIL import Image

from omlx.api import image_routes
from omlx.engine import base, image_generation
from omlx.engine.image_generation import DiffusionImageEngine
from omlx.exceptions import InsufficientMemoryError


def _png(seed=0):
    output = io.BytesIO()
    Image.new("RGB", (32, 32), (seed % 256, 0, 0)).save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    checkpoint = SimpleNamespace(
        base_model="flux2-klein-4b",
        default_pipeline="flux2-klein-4b",
        path=tmp_path,
    )
    monkeypatch.setattr(image_generation, "detect_checkpoint", lambda _: checkpoint)
    monkeypatch.setattr(image_routes, "detect_checkpoint", lambda _: checkpoint)
    return checkpoint


@pytest.fixture
def executor(monkeypatch):
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(image_generation, "get_mlx_executor", lambda: executor)
        monkeypatch.setattr(base, "get_mlx_executor", lambda: executor)
        monkeypatch.setattr(image_generation.mx, "synchronize", lambda: None)
        monkeypatch.setattr(image_generation.mx, "clear_cache", lambda: None)
        yield executor


@pytest.fixture
def api(checkpoint, monkeypatch):
    engine = DiffusionImageEngine("/checkpoint")
    calls = []
    state = {"acquires": 0}

    async def one(**kwargs):
        calls.append(("single", kwargs))
        return _png(kwargs["seed"])

    async def batch(**kwargs):
        calls.append(("batch", kwargs))
        return [_png(seed) for seed in kwargs["seeds"]]

    engine.generate_image = one
    engine.generate_images = batch

    class Pool:
        def get_entry(self, _):
            return SimpleNamespace(
                model_path="/checkpoint", engine_type="image_generation"
            )

        @asynccontextmanager
        async def acquire(self, _, *, image_pipeline=None):
            state["acquires"] += 1
            yield engine

    monkeypatch.setattr(image_routes, "_get_engine_pool", lambda: Pool())
    monkeypatch.setattr(image_routes, "_resolve_model", lambda value: value)
    app = FastAPI()
    app.include_router(image_routes.router)
    return app, engine, calls, state


@pytest.mark.asyncio
async def test_opt_in_batch_seed_order_and_singleton_tail(api):
    app, _, calls, state = api
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proof"
    ) as client:
        response = await client.post(
            "/v1/images/generations",
            json={
                "model": "image",
                "prompt": "A bird",
                "n": 3,
                "batch_size": 2,
                "seed": 2**32 - 1,
            },
        )
    assert response.status_code == 200, response.text
    assert [item["seed"] for item in response.json()["data"]] == [2**32 - 1, 0, 1]
    assert [kind for kind, _ in calls] == ["batch", "single"]
    assert calls[0][1]["seeds"] == (2**32 - 1, 0)
    assert calls[1][1]["seed"] == 1
    assert state["acquires"] == 1


@pytest.mark.asyncio
async def test_default_n_remains_serial(api):
    app, _, calls, _ = api
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proof"
    ) as client:
        response = await client.post(
            "/v1/images/generations",
            json={"model": "image", "prompt": "A bird", "n": 2},
        )
    assert response.status_code == 200
    assert [kind for kind, _ in calls] == ["single", "single"]


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size,n,status", [(2, 1, 400), (5, 4, 422), (0, 2, 422)])
async def test_invalid_batch_refused_before_acquire(api, batch_size, n, status):
    app, _, _, state = api
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proof"
    ) as client:
        response = await client.post(
            "/v1/images/generations",
            json={
                "model": "image",
                "prompt": "A bird",
                "batch_size": batch_size,
                "n": n,
            },
        )
    assert response.status_code == status
    assert state["acquires"] == 0


@pytest.mark.asyncio
async def test_unsupported_pipeline_batch_refused_before_acquire(api, checkpoint):
    checkpoint.base_model = checkpoint.default_pipeline = "z-image-turbo"
    app, _, _, state = api
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proof"
    ) as client:
        response = await client.post(
            "/v1/images/generations",
            json={"model": "image", "prompt": "A bird", "batch_size": 2, "n": 2},
        )
    assert response.status_code == 400
    assert "at most 1" in response.text
    assert state["acquires"] == 0


@pytest.mark.asyncio
async def test_incomplete_batch_is_error(api):
    app, engine, _, _ = api

    async def incomplete(**_):
        return [_png()]

    engine.generate_images = incomplete
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://proof",
    ) as client:
        response = await client.post(
            "/v1/images/generations",
            json={"model": "image", "prompt": "A bird", "batch_size": 2, "n": 2},
        )
    assert response.status_code == 500


class Backend:
    def __init__(self):
        self.calls = []
        self.entered = threading.Event()
        self.proceed = threading.Event()
        self.block = False

    def load(self, checkpoint, pipeline):
        self.calls.append(("load", threading.get_ident()))
        return object()

    def generate_batch(self, model, tasks, pipeline):
        self.calls.append(
            ("batch", threading.get_ident(), tuple(t.seed for t in tasks))
        )
        if self.block:
            self.entered.set()
            assert self.proceed.wait(5)
        return [SimpleNamespace(image=Image.new("RGB", (32, 32))) for _ in tasks]

    def clear_cache(self, model):
        self.calls.append(("clear", threading.get_ident()))
        return 2

    def get_cache_stats(self, model):
        return {"entries": 2, "bytes": 24}

    def release(self, model):
        self.calls.append(("release", threading.get_ident()))


def _engine():
    engine = DiffusionImageEngine("/checkpoint")
    engine._backend = Backend()
    return engine


@pytest.mark.asyncio
async def test_batch_clear_stop_serialized_and_cancellation_drained(
    checkpoint, executor, monkeypatch
):
    engine = _engine()
    monkeypatch.setattr(engine, "_admit_batch", lambda *_: None)
    await engine.start()
    backend = engine._backend
    backend.block = True
    generation = asyncio.create_task(
        engine.generate_images(
            prompt="A bird",
            seeds=(1, 2),
            width=256,
            height=256,
        )
    )
    assert await asyncio.to_thread(backend.entered.wait, 3)
    generation.cancel()
    generation.cancel()
    clear = asyncio.create_task(engine.clear_prompt_caches(hot=True))
    await asyncio.sleep(0)
    assert not generation.done() and not clear.done()
    assert engine.get_activity_snapshot()["active_requests"] == 1
    assert engine.get_runtime_cache_stats()["entries"] == 2
    backend.proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await generation
    assert await clear == {"hot_cleared": 2, "ssd_deleted": 0}
    await engine.stop()
    assert engine.get_activity_snapshot()["active_requests"] == 0
    assert engine.get_runtime_cache_stats() is None
    assert [c[0] for c in backend.calls] == ["load", "batch", "clear", "release"]
    assert len({c[1] for c in backend.calls}) == 1


@pytest.mark.asyncio
async def test_batch_memory_refusal_precedes_native_generation(
    checkpoint, executor, monkeypatch
):
    engine = _engine()
    monkeypatch.setattr(
        "psutil.virtual_memory", lambda: SimpleNamespace(available=100 * 1024**3)
    )
    monkeypatch.setattr(image_generation.mx, "get_active_memory", lambda: 0)
    monkeypatch.setattr(
        image_generation.mx,
        "device_info",
        lambda: {"max_recommended_working_set_size": 100 * 1024**3},
    )
    engine.set_memory_soft_limit(1024**3)
    await engine.start()
    with pytest.raises(InsufficientMemoryError):
        await engine.generate_images(
            prompt="A bird", seeds=(1, 2), width=256, height=256
        )
    assert [c[0] for c in engine._backend.calls] == ["load"]
    assert engine.get_activity_snapshot()["active_requests"] == 0
    await engine.stop()
