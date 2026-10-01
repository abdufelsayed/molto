"""Serving proofs use tiny Pillow images and fake backends, never model weights."""

import asyncio
import base64
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from omlx_runtime.engine import base, image_generation
from omlx_runtime.engine.image_generation import DiffusionImageEngine, MFluxImageEngine
from omlx_server.api import image_routes
from PIL import Image


def png_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def checkpoint(monkeypatch):
    cp = SimpleNamespace(base_model="z-image-turbo", default_pipeline="z-image-turbo")
    monkeypatch.setattr(image_generation, "detect_checkpoint", lambda path: cp)
    monkeypatch.setattr(image_routes, "detect_checkpoint", lambda path: cp)
    return cp


@pytest.fixture
def executor(monkeypatch):
    with ThreadPoolExecutor(max_workers=1) as thread:
        monkeypatch.setattr(image_generation, "get_mlx_executor", lambda: thread)
        monkeypatch.setattr(base, "get_mlx_executor", lambda: thread)
        monkeypatch.setattr(image_generation.mx, "synchronize", lambda: None)
        monkeypatch.setattr(image_generation.mx, "clear_cache", lambda: None)
        yield thread


class Backend:
    def __init__(self):
        self.calls = []
        self.live = set()
        self.entered = threading.Event()
        self.proceed = threading.Event()
        self.block_load = False
        self.block_generate = False

    def load(self, checkpoint, pipeline):
        self.calls.append(("load", pipeline))
        assert not self.live, "Previous pipeline is still resident"
        if self.block_load:
            self.entered.set()
            assert self.proceed.wait(5)
        model = object()
        self.live.add(model)
        return model

    def generate(self, model, task, pipeline):
        assert model in self.live
        self.calls.append(("generate", pipeline))
        if self.block_generate:
            self.entered.set()
            assert self.proceed.wait(5)
        return SimpleNamespace(image=Image.new("RGB", (32, 32), "red"))

    def release(self, model):
        self.calls.append(("release", model))
        self.live.remove(model)


async def entered(backend):
    assert await asyncio.to_thread(backend.entered.wait, 3)


def engine_with_backend():
    engine = DiffusionImageEngine("/checkpoint")
    backend = Backend()
    engine._backend = backend
    return engine, backend


@pytest.mark.asyncio
async def test_generation_cancellation_keeps_activity_and_stop_locked(
    checkpoint, executor
):
    engine, backend = engine_with_backend()
    await engine.start()
    backend.block_generate = True
    generation = asyncio.create_task(
        engine.generate_image(prompt="A bird", seed=1, width=512, height=512)
    )
    await entered(backend)
    generation.cancel()
    await asyncio.sleep(0)
    generation.cancel()  # A second cancellation must not release the lease early either.
    stopping = asyncio.create_task(engine.stop())
    await asyncio.sleep(0)
    assert not generation.done()
    assert not stopping.done()
    assert engine.get_activity_snapshot()["active_requests"] == 1
    assert len(backend.live) == 1
    backend.proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await generation
    await stopping
    assert not backend.live
    assert engine.get_activity_snapshot()["active_requests"] == 0
    assert [call[0] for call in backend.calls] == ["load", "generate", "release"]


@pytest.mark.asyncio
async def test_startup_cancellation_does_not_orphan_loaded_model(checkpoint, executor):
    engine, backend = engine_with_backend()
    backend.block_load = True
    starting = asyncio.create_task(engine.start())
    await entered(backend)
    starting.cancel()
    stopping = asyncio.create_task(engine.stop())
    await asyncio.sleep(0)
    assert not starting.done()
    assert not stopping.done()
    backend.proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await starting
    await stopping
    assert engine._model is None
    assert not backend.live
    assert [call[0] for call in backend.calls] == ["load", "release"]


@pytest.mark.asyncio
async def test_concurrent_start_only_loads_once_and_switch_releases_first(
    checkpoint, executor
):
    engine, backend = engine_with_backend()
    await asyncio.gather(engine.start(), engine.start())
    output = await engine.generate_image(
        prompt="A bird",
        seed=1,
        width=512,
        height=512,
        pipeline="z-image-turbo/img2img",
        image_paths=("normalized.png",),
    )
    assert output.startswith(b"\x89PNG")
    assert [call[0] for call in backend.calls] == [
        "load",
        "release",
        "load",
        "generate",
    ]
    assert engine.get_stats()["pipeline"] == "z-image-turbo/img2img"
    await engine.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"guidance": 2},
        {"negative_prompt": ""},
        {"options": {"ignored": 1}},
        {"pipeline": "unknown"},
    ],
)
async def test_engine_invalid_tasks_rejected_before_load(checkpoint, executor, kwargs):
    engine, backend = engine_with_backend()
    with pytest.raises(ValueError):
        await engine.generate_image(
            prompt="A bird", seed=1, width=512, height=512, **kwargs
        )
    assert backend.calls == []


@pytest.fixture
def api(monkeypatch, checkpoint):
    engine = MFluxImageEngine("/checkpoint")
    calls = []
    state = {"acquires": 0, "leased": False}

    async def generate(**kwargs):
        calls.append(kwargs)
        for value in kwargs["image_paths"]:
            assert Path(value).is_file()
        return png_bytes()

    engine.generate_image = generate

    class Pool:
        def get_entry(self, model):
            if model == "missing":
                return None
            return SimpleNamespace(
                model_path="/checkpoint",
                engine_type="text" if model == "text" else "image_generation",
            )

        @asynccontextmanager
        async def acquire(self, model, *, image_pipeline=None):
            state["acquires"] += 1
            state["leased"] = True
            try:
                yield engine
            finally:
                state["leased"] = False

    monkeypatch.setattr(image_routes, "_get_engine_pool", lambda request: Pool())
    monkeypatch.setattr(
        image_routes,
        "_resolve_model",
        lambda value, request: "image" if value == "alias" else value,
    )
    app = FastAPI()
    from omlx_server.state import ServerState

    app.state.server_state = ServerState()
    app.include_router(image_routes.router)
    return app, calls, state, engine


@pytest.mark.asyncio
async def test_actual_generation_schema_compatibility_and_seed_wrap(api):
    app, calls, state, _ = api
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/images/generations",
            json={"model": "alias", "prompt": "A bird", "n": 2, "seed": 2**32 - 1},
        )
        assert response.status_code == 200
        assert [item["seed"] for item in response.json()["data"]] == [2**32 - 1, 0]
        assert base64.b64decode(response.json()["data"][0]["b64_json"]).startswith(
            b"\x89PNG"
        )
        assert calls[0]["width"] == calls[0]["height"] == 1024
        assert (
            calls[0]["steps"] is None
        )  # The descriptor, not the HTTP schema, owns defaults.
        extra = await client.post(
            "/v1/images/generations",
            json={"model": "image", "prompt": "A bird", "ignored": True},
        )
        assert extra.status_code == 422
    assert state["acquires"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [
        {"guidance": 1},
        {"negative_prompt": ""},
        {"pipeline": "unknown"},
        {"options": {"unknown": 1}},
        {"pipeline": "dev"},
    ],
)
async def test_route_capability_preflight_precedes_pool_acquire(api, extra):
    app, _, state, _ = api
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/images/generations",
            json={"model": "image", "prompt": "A bird", **extra},
        )
    assert response.status_code == 400
    assert state["acquires"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("model,status", [("missing", 404), ("text", 400)])
async def test_wrong_model_is_rejected_before_acquire(api, model, status):
    app, _, state, _ = api
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/images/generations", json={"model": model, "prompt": "A bird"}
        )
    assert response.status_code == status
    assert not state["acquires"]


@pytest.mark.asyncio
@pytest.mark.parametrize("multipart", [False, True])
async def test_edit_inputs_are_normalized_and_temporary_files_removed(api, multipart):
    app, calls, _, _ = api
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        if multipart:
            response = await client.post(
                "/v1/images/edits",
                data={"model": "image", "prompt": "Edit a bird"},
                files={"image": ("../../ignored.png", png_bytes(), "image/png")},
            )
        else:
            response = await client.post(
                "/v1/images/edits",
                json={
                    "model": "image",
                    "prompt": "Edit a bird",
                    "image": base64.b64encode(png_bytes()).decode(),
                },
            )
    assert response.status_code == 200, response.text
    assert calls[0]["pipeline"] == "z-image-turbo/img2img"
    assert not Path(calls[0]["image_paths"][0]).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "image",
    ["http://localhost/private", "/etc/passwd", base64.b64encode(b"invalid").decode()],
)
async def test_edit_rejects_urls_paths_and_invalid_media(api, image):
    app, _, state, _ = api
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/images/edits",
            json={"model": "image", "prompt": "Edit", "image": image},
        )
    assert response.status_code == 400
    assert not state["acquires"]


@pytest.mark.asyncio
async def test_mask_and_unknown_operation_rejected_without_loading(api):
    app, _, state, _ = api
    image = base64.b64encode(png_bytes()).decode()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        mask = await client.post(
            "/v1/images/edits",
            json={"model": "image", "prompt": "Edit", "image": image, "mask": image},
        )
        operation = await client.post(
            "/v1/images/operations",
            json={"model": "image", "operation": "bogus", "images": [image]},
        )
    assert mask.status_code == 400
    assert operation.status_code == 422
    assert not state["acquires"]


@pytest.mark.asyncio
async def test_media_body_bound_and_strict_multipart_fields(api, monkeypatch):
    app, _, state, _ = api
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        extra = await client.post(
            "/v1/images/edits",
            data={"model": "image", "prompt": "Edit", "unsupported": "x"},
            files={"image": ("input.png", png_bytes())},
        )
        assert extra.status_code == 422
        monkeypatch.setattr(image_routes, "_MAX_BODY_BYTES", 10)
        large = await client.post(
            "/v1/images/edits",
            json={"model": "image", "prompt": "Edit", "image": "large"},
        )
        assert large.status_code == 413
    assert not state["acquires"]


@pytest.mark.asyncio
async def test_capabilities_without_loading(api):
    app, _, state, _ = api
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(
            "/v1/images/capabilities", params={"model": "alias"}
        )
    assert response.status_code == 200
    assert {p["operation"] for p in response.json()["pipelines"]} == {
        "txt2img",
        "img2img",
    }
    assert not state["acquires"]


@pytest.mark.asyncio
async def test_cancelled_edit_retains_media_and_pool_lease_until_executor_finishes(
    api, checkpoint, executor
):
    app, calls, state, engine = api
    backend = Backend()
    engine._backend = backend
    # Restore the implementation replaced by the API fixture.
    engine.generate_image = DiffusionImageEngine.generate_image.__get__(engine)
    await engine.start()
    backend.block_generate = True
    paths = []
    original_generate = backend.generate

    def generate(model, task, pipeline):
        paths.extend(task.image_paths)
        result = original_generate(model, task, pipeline)
        assert all(Path(path).exists() for path in paths)
        return result

    backend.generate = generate
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        pending = asyncio.create_task(
            client.post(
                "/v1/images/edits",
                json={
                    "model": "image",
                    "prompt": "Edit",
                    "image": base64.b64encode(png_bytes()).decode(),
                },
            )
        )
        await entered(backend)
        pending.cancel()
        await asyncio.sleep(0)
        assert state["leased"]
        assert all(Path(path).exists() for path in paths)
        backend.proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
    assert not state["leased"]
    assert all(not Path(path).exists() for path in paths)
    await engine.stop()


@pytest.mark.asyncio
async def test_validation_failure_cleans_already_normalized_media(api, monkeypatch):
    app, _, state, _ = api
    paths = []
    original = image_routes._normalize_image

    def normalize(value, path):
        paths.append(path)
        return original(value, path)

    monkeypatch.setattr(image_routes, "_normalize_image", normalize)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/images/edits",
            json={
                "model": "image",
                "prompt": "Edit",
                "image": [base64.b64encode(png_bytes()).decode(), "bad base64"],
            },
        )
    assert response.status_code == 400
    assert not state["acquires"]
    assert paths and all(
        not path.exists() and not path.parent.exists() for path in paths
    )


@pytest.mark.asyncio
async def test_multipart_duplicate_fields_and_json_extras_fail_closed(api):
    app, _, state, _ = api
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        duplicate = await client.post(
            "/v1/images/edits",
            files=[
                ("model", (None, "image")),
                ("model", (None, "other")),
                ("prompt", (None, "Edit")),
                ("image", ("input.png", png_bytes())),
            ],
        )
        extra = await client.post(
            "/v1/images/operations",
            json={
                "model": "image",
                "operation": "img2img",
                "prompt": "Edit",
                "images": [base64.b64encode(png_bytes()).decode()],
                "extra": True,
            },
        )
    assert duplicate.status_code == 400
    assert extra.status_code == 422
    assert not state["acquires"]


@pytest.mark.asyncio
async def test_inpaint_and_upscale_use_registry_contract(api, checkpoint):
    app, calls, _, _ = api
    image = base64.b64encode(png_bytes()).decode()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        checkpoint.base_model = checkpoint.default_pipeline = "dev-fill"
        inpaint = await client.post(
            "/v1/images/operations",
            json={
                "model": "image",
                "operation": "inpaint",
                "prompt": "Edit",
                "images": [image],
                "mask": image,
            },
        )
        assert inpaint.status_code == 200, inpaint.text
        assert calls[-1]["pipeline"] == "dev-fill"
        assert calls[-1]["mask_path"]
        checkpoint.base_model = checkpoint.default_pipeline = "seedvr2-3b"
        upscale = await client.post(
            "/v1/images/operations",
            json={
                "model": "image",
                "operation": "upscale",
                "images": [image],
                "options": {"resolution": 512},
            },
        )
        assert upscale.status_code == 200, upscale.text
        assert calls[-1]["prompt"] is None
        assert calls[-1]["width"] is None
        assert calls[-1]["steps"] is None
        ignored = await client.post(
            "/v1/images/operations",
            json={
                "model": "image",
                "operation": "upscale",
                "images": [image],
                "prompt": "ignored",
            },
        )
        assert ignored.status_code == 400


@pytest.mark.asyncio
async def test_generation_failure_keeps_exception_mapping_and_cleans_media(api):
    from omlx_runtime.exceptions import ModelBusyError

    app, calls, state, engine = api
    paths = []

    async def fail(**kwargs):
        paths.extend(kwargs["image_paths"])
        raise ModelBusyError("image", "generate")

    engine.generate_image = fail
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/images/edits",
            json={
                "model": "image",
                "prompt": "Edit",
                "image": base64.b64encode(png_bytes()).decode(),
            },
        )
    assert response.status_code == 409
    assert not state["leased"]
    assert all(not Path(path).exists() for path in paths)


@pytest.mark.asyncio
async def test_upscale_aspect_ratio_cannot_exceed_output_budget(api, checkpoint):
    app, _, state, _ = api
    checkpoint.base_model = checkpoint.default_pipeline = "seedvr2-3b"
    stream = io.BytesIO()
    Image.new("RGB", (1, 4096), "red").save(stream, format="PNG")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/images/operations",
            json={
                "model": "image",
                "operation": "upscale",
                "images": [base64.b64encode(stream.getvalue()).decode()],
                "options": {"resolution": 512},
            },
        )
    assert response.status_code == 400
    assert not state["acquires"]


@pytest.mark.asyncio
async def test_cancelled_multipart_parse_closes_spooled_upload():
    from omlx_contracts.api.image_models import ImageEditRequest
    from starlette.datastructures import FormData, UploadFile

    entered_form = asyncio.Event()
    finish_form = asyncio.Event()
    upload = UploadFile(io.BytesIO(png_bytes()), filename="image.png")

    class FormContext:
        async def __aenter__(self):
            entered_form.set()
            await finish_form.wait()
            return FormData([("model", "image"), ("prompt", "Edit"), ("image", upload)])

        async def __aexit__(self, *args):
            await upload.close()

    class Request:
        headers = {"content-type": "multipart/form-data; boundary=unused"}

        async def stream(self):
            yield b"bounded body"

        def form(self, **kwargs):
            return FormContext()

    parsing = asyncio.create_task(
        image_routes._media_request(Request(), ImageEditRequest, multipart=True)
    )
    await entered_form.wait()
    parsing.cancel()
    await asyncio.sleep(0)
    assert not parsing.done()
    assert not upload.file.closed
    finish_form.set()
    with pytest.raises(asyncio.CancelledError):
        await parsing
    assert upload.file.closed


def test_openapi_describes_strict_media_request_schemas(api):
    app, _, _, _ = api
    paths = app.openapi()["paths"]
    edits = paths["/v1/images/edits"]["post"]["requestBody"]["content"]
    assert set(edits) == {"application/json", "multipart/form-data"}
    assert edits["application/json"]["schema"]["additionalProperties"] is False
    assert (
        edits["multipart/form-data"]["schema"]["properties"]["mask"]["format"]
        == "binary"
    )
    operations = paths["/v1/images/operations"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]
    assert "upscale" in operations["properties"]["operation"]["enum"]


class RetainingPool:
    """Model the pool fast path that retains a started engine after request errors."""

    def __init__(self, engine):
        self.engine = engine
        self.entry = SimpleNamespace(engine=None)

    @asynccontextmanager
    async def acquire(self):
        if self.entry.engine is None:
            await self.engine.start()
            self.entry.engine = self.engine
        yield self.entry.engine


@pytest.mark.asyncio
async def test_pool_retained_engine_recovers_after_failed_pipeline_switch(
    checkpoint, executor
):
    engine, backend = engine_with_backend()
    pool = RetainingPool(engine)
    original_load = backend.load

    def failing_load(cp, pipeline):
        if pipeline == "z-image-turbo/img2img":
            backend.calls.append(("failed_load", pipeline))
            raise ValueError("Variant load failed")
        return original_load(cp, pipeline)

    backend.load = failing_load
    async with pool.acquire() as acquired:
        with pytest.raises(ValueError, match="Variant load failed"):
            await acquired.generate_image(
                prompt="Edit",
                seed=1,
                width=512,
                height=512,
                pipeline="z-image-turbo/img2img",
                image_paths=("normalized.png",),
            )
    assert pool.entry.engine is engine
    assert not engine.get_stats()["loaded"]
    assert not backend.live
    async with pool.acquire() as acquired:
        output = await acquired.generate_image(
            prompt="A bird", seed=2, width=512, height=512
        )
    assert output.startswith(b"\x89PNG")
    assert engine.get_stats()["loaded"]
    assert engine.get_stats()["pipeline"] == "z-image-turbo"
    assert [call[0] for call in backend.calls] == [
        "load",
        "release",
        "failed_load",
        "load",
        "generate",
    ]
    await engine.stop()


@pytest.mark.asyncio
async def test_pool_retained_engine_recovers_after_cancelled_pipeline_switch(
    checkpoint, executor
):
    engine, backend = engine_with_backend()
    pool = RetainingPool(engine)
    async with pool.acquire():
        pass
    backend.block_load = True

    async def switch():
        async with pool.acquire() as acquired:
            await acquired.generate_image(
                prompt="Edit",
                seed=1,
                width=512,
                height=512,
                pipeline="z-image-turbo/img2img",
                image_paths=("normalized.png",),
            )

    pending = asyncio.create_task(switch())
    await entered(backend)
    pending.cancel()
    await asyncio.sleep(0)
    pending.cancel()
    assert not pending.done()
    backend.proceed.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert pool.entry.engine is engine
    assert not engine.get_stats()["loaded"]
    assert not backend.live
    backend.block_load = False
    async with pool.acquire() as acquired:
        output = await acquired.generate_image(
            prompt="A bird", seed=2, width=512, height=512
        )
    assert output.startswith(b"\x89PNG")
    assert engine.get_stats()["loaded"]
    assert [call[0] for call in backend.calls] == [
        "load",
        "release",
        "load",
        "release",
        "load",
        "generate",
    ]
    await engine.stop()


@pytest.mark.asyncio
async def test_unstarted_and_explicitly_stopped_engine_do_not_reload(
    checkpoint, executor
):
    engine, backend = engine_with_backend()
    with pytest.raises(RuntimeError, match="Engine not started"):
        await engine.generate_image(prompt="A bird", seed=1, width=512, height=512)
    assert not backend.calls
    await engine.start()
    await engine.stop()
    calls = list(backend.calls)
    with pytest.raises(RuntimeError, match="Engine not started"):
        await engine.generate_image(prompt="A bird", seed=2, width=512, height=512)
    assert backend.calls == calls
    assert not engine.get_stats()["loaded"]


@pytest.fixture
def pooled_api(api, checkpoint, executor, tmp_path, monkeypatch):
    """Run actual pool construction/start/acquire paths with a fake native backend."""
    import json

    from omlx_runtime import engine_pool

    app, _, _, _ = api
    path = tmp_path / "image"
    for component in ("vae", "transformer", "text_encoder"):
        (path / component).mkdir(parents=True)
        (path / component / "0.safetensors").write_bytes(b"weights")
    (path / "tokenizer").mkdir()
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
    pool = engine_pool.EnginePool()
    pool.discover_models(str(tmp_path))
    monkeypatch.setattr(pool, "_ensure_gpu_keep_warm_task", lambda: None)
    monkeypatch.setattr(engine_pool, "get_mlx_executor", lambda: executor)
    monkeypatch.setattr(engine_pool, "get_phys_footprint", lambda: 0)
    monkeypatch.setattr(engine_pool.mx, "get_active_memory", lambda: 0)
    monkeypatch.setattr(engine_pool.mx, "get_cache_memory", lambda: 0)
    backend = Backend()
    engines = []

    def construct(model_name, **kwargs):
        engine = DiffusionImageEngine(model_name, **kwargs)
        engine._backend = backend
        engines.append(engine)
        return engine

    original_generate = backend.generate

    def generate(model, task, pipeline):
        assert pool.get_entry("image").in_use == 1
        return original_generate(model, task, pipeline)

    backend.generate = generate
    monkeypatch.setattr(engine_pool, "MFluxImageEngine", construct)
    monkeypatch.setattr(image_routes, "_get_engine_pool", lambda request: pool)
    return app, pool, backend, engines


@pytest.mark.asyncio
async def test_cold_edit_loads_selected_pipeline_once_then_reuses_and_switches(
    pooled_api,
):
    app, pool, backend, engines = pooled_api
    image = base64.b64encode(png_bytes()).decode()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        edit = {"model": "image", "prompt": "Edit", "image": image}
        cold = await client.post("/v1/images/edits", json=edit)
        assert cold.status_code == 200, cold.text
        assert [call for call in backend.calls if call[0] == "load"] == [
            ("load", "z-image-turbo/img2img")
        ]
        warm = await client.post("/v1/images/edits", json=edit)
        assert warm.status_code == 200, warm.text
        assert len([call for call in backend.calls if call[0] == "load"]) == 1
        generation = await client.post(
            "/v1/images/generations", json={"model": "image", "prompt": "A bird"}
        )
        assert generation.status_code == 200, generation.text
        switched = await client.post("/v1/images/edits", json=edit)
        assert switched.status_code == 200, switched.text
    assert len(engines) == 1
    assert pool.get_entry("image").engine is engines[0]
    assert pool.get_entry("image").in_use == 0
    assert [call for call in backend.calls if call[0] == "load"] == [
        ("load", "z-image-turbo/img2img"),
        ("load", "z-image-turbo"),
        ("load", "z-image-turbo/img2img"),
    ]
    assert [call[0] for call in backend.calls] == [
        "load",
        "generate",
        "generate",
        "release",
        "load",
        "generate",
        "release",
        "load",
        "generate",
    ]
    assert len(backend.live) == 1
    await engines[0].stop()


@pytest.mark.asyncio
async def test_cancelled_cold_edit_drains_selected_load_and_can_retry(pooled_api):
    app, pool, backend, engines = pooled_api
    backend.block_load = True
    edit = {
        "model": "image",
        "prompt": "Edit",
        "image": base64.b64encode(png_bytes()).decode(),
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        pending = asyncio.create_task(client.post("/v1/images/edits", json=edit))
        await entered(backend)
        pending.cancel()
        await asyncio.sleep(0)
        assert not pending.done()
        assert pool.get_entry("image").is_loading
        assert [call for call in backend.calls if call[0] == "load"] == [
            ("load", "z-image-turbo/img2img")
        ]
        backend.proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert not backend.live
        assert pool.get_entry("image").engine is None
        assert not pool.get_entry("image").is_loading
        assert not engines[0]._started
        backend.block_load = False
        retry = await client.post("/v1/images/edits", json=edit)
        assert retry.status_code == 200, retry.text
    assert [call for call in backend.calls if call[0] == "load"] == [
        ("load", "z-image-turbo/img2img")
    ] * 2
    assert pool.get_entry("image").in_use == 0
    await engines[-1].stop()


@pytest.mark.asyncio
async def test_pool_text_acquisition_keeps_original_call_shape(monkeypatch):
    from omlx_runtime.engine_pool import EnginePool

    pool = EnginePool()
    calls = []
    engine = object()

    async def get_engine(model_id, *, force_lm, _lease):
        calls.append((model_id, force_lm, _lease))
        return engine

    async def release(model_id):
        calls.append(("release", model_id))

    monkeypatch.setattr(pool, "get_engine", get_engine)
    monkeypatch.setattr(pool, "release_engine", release)
    async with pool.acquire("text", force_lm=True) as acquired:
        assert acquired is engine
    assert calls == [("text", True, True), ("release", "text")]
