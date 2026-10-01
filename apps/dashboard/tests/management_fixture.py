"""Real Molto management routes and persistence with synthetic engines, never weights."""

import asyncio
import json
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import uvicorn
from fastapi import Depends, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse
from molto_config.model_settings import ModelSettings, ModelSettingsManager
from molto_config.settings import GlobalSettings, SubKeyEntry
from molto_management import management
from molto_management.management import ManagementContext, ManagementError
from molto_management.management_runtime import ManagementRuntime
from molto_runtime.engine_pool import EngineEntry, EnginePool
from molto_runtime.server_metrics import ServerMetrics
from molto_runtime.usage_history import UsageHistory
from molto_server.api.dashboard_access_routes import router as dashboard_access_router
from molto_server.api.management_routes import router
from molto_server.api.management_setup_routes import router as setup_router
from molto_server.auth import AuthContext, require_model_load_key

MODEL = "mlx-community/test-model"
GIB = 1024**3


class Engine:
    def get_runtime_cache_stats(self):
        return {
            "ssd_cache": {
                "num_files": 3,
                "total_size_bytes": 128 * 1024**2,
                "max_size_bytes": GIB,
                "hot_cache_size_bytes": 64 * 1024**2,
                "hot_cache_max_bytes": 256 * 1024**2,
            },
            "cache_rates": {"prefix_hits": 18, "prefix_misses": 6},
        }

    async def clear_prompt_caches(self, **tiers):
        return {
            "hot_cleared": 2 if tiers.get("hot") else 0,
            "ssd_deleted": 3 if tiers.get("ssd") else 0,
        }


class Pool(EnginePool):
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        self.tasks = set()
        self._current_ceiling = lambda: 16 * GIB
        self._model_dirs = [directory / "models"]
        self._scheduler_config = SimpleNamespace(hot_cache_budget=None)
        self.add(MODEL, loaded=True)
        self.add("local/busy-model", loaded=True)
        self.add("local/unloaded-model")
        self.add("local/embedding", kind="embedding")

    def add(self, model_id, loaded=False, kind="llm"):
        path = self.directory / "models" / model_id
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "llama",
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "vocab_size": 16,
                }
            )
        )
        (path / "generation_config.json").write_text(
            json.dumps({"temperature": 0.6, "top_p": 0.95})
        )
        self._entries[model_id] = EngineEntry(
            model_id=model_id,
            model_path=str(path),
            model_type=kind,
            engine_type="batched" if kind == "llm" else "embedding",
            estimated_size=GIB,
            config_model_type="llama",
        )
        self._entries[model_id].engine = Engine() if loaded else None
        self.account()

    def account(self):
        self._current_model_memory = sum(
            e.estimated_size for e in self._entries.values() if e.engine
        )

    async def get_engine(self, model_id):
        entry = self._entries[model_id]
        entry.is_loading = True
        await asyncio.sleep(0.15)
        entry.engine = Engine()
        entry.is_loading = False
        entry.load_failed = False
        self.account()
        return entry.engine

    async def request_unload(self, model_id, *, reason, abort_active=False):
        entry = self._entries[model_id]
        if "busy" in model_id:
            entry.pending_unload_reason = reason

            async def later():
                await asyncio.sleep(1)
                entry.engine = None
                entry.pending_unload_reason = None
                self.account()

            task = asyncio.create_task(later())
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
            return False
        entry.engine = None
        self.account()
        return True

    def discover_models(self, *args):
        self.add("local/new-checkpoint")

    def _entry_is_busy(self, entry):
        return "busy" in entry.model_id and entry.engine is not None

    async def apply_embedding_batch_size(self, value):
        return None


@asynccontextmanager
async def lifespan(app):
    await reset()
    yield
    await cleanup()


app = FastAPI(lifespan=lifespan)
app.include_router(router)
app.include_router(setup_router)
app.include_router(dashboard_access_router)

stats = dict(
    total_tokens_served=2048,
    total_cached_tokens=512,
    cache_efficiency=25.0,
    total_prompt_tokens=1024,
    total_completion_tokens=1024,
    total_requests=8,
    avg_prefill_tps=120.0,
    avg_generation_tps=45.0,
    uptime_seconds=600.0,
)
management.get_server_metrics = lambda: SimpleNamespace(
    get_snapshot=lambda **kwargs: {
        **stats,
        "total_requests": 80 if kwargs.get("scope") == "alltime" else 8,
    }
)


@app.exception_handler(ManagementError)
async def error_handler(request, error):
    return JSONResponse(
        {"detail": error.detail},
        status_code={
            "not_found": 404,
            "busy": 409,
            "conflict": 409,
            "invalid_configuration": 400,
            "unavailable": 503,
            "persistence_failed": 503,
            "rollback_failed": 503,
            "runtime_failed": 503,
        }.get(error.code, 400),
    )


async def cleanup():
    runtime = getattr(app.state, "management_runtime", None)
    if runtime:
        await runtime.shutdown()
    context = getattr(app.state, "context", None)
    if context:
        for task in list(context.engine_pool.tasks):
            task.cancel()
        await asyncio.gather(*context.engine_pool.tasks, return_exceptions=True)
        context.runtime_state.server_metrics.usage_history.close()
    temporary = getattr(app.state, "temporary", None)
    if temporary:
        temporary.cleanup()


@app.post("/__test__/reset")
async def reset(setup: bool = False):
    await cleanup()
    app.state.proxy_metrics = {
        "streams_started": 0,
        "streams_cancelled": 0,
        "ws_closed": [],
        "requests": [],
    }
    app.state.temporary = tempfile.TemporaryDirectory(prefix="molto-dashboard-test-")
    directory = Path(app.state.temporary.name)
    pool = Pool(directory)
    settings = GlobalSettings(base_path=directory)
    settings.model.model_dirs = [str(directory / "models")]
    settings.huggingface.hf_cache_enabled = False
    settings.auth.api_key = None if setup else "dashboard-test-key"
    settings.auth.sub_keys = [
        SubKeyEntry(key="inference-sub-key", name="Fixture inference")
    ]
    settings.save()
    logs = settings.logging.get_log_dir(directory)
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "server.log").write_text(
        "2026-10-01 10:00:00,001 - molto_runtime.fixture - INFO - fixture ready\nERROR fixture info continuation\n2026-10-01 10:00:00,002 - molto_runtime.fixture - WARNING - fixture warning\n2026-10-01 10:00:00,003 - molto_runtime.fixture - ERROR - fixture failure\n  traceback continuation\n"
    )
    (logs / "server.log.1").write_text(
        "2026-10-01 09:00:00,001 - molto_runtime.fixture - INFO - rotated fixture record\n"
    )
    history = UsageHistory(directory / "usage.sqlite3")
    for model_id, timestamp in [
        (MODEL, time.time()),
        (
            "local/embedding",
            (
                datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
                - timedelta(days=1)
            ).timestamp(),
        ),
    ]:
        history.record(
            model_id=model_id,
            prompt_tokens=1024,
            completion_tokens=512,
            cached_tokens=256,
            prefill_duration=2,
            generation_duration=10,
            request_duration=12,
            timestamp=timestamp,
        )
    history.flush()
    metrics = ServerMetrics(directory / "stats.json")
    metrics.usage_history = history
    metrics.get_snapshot = lambda **kwargs: {
        **stats,
        "total_requests": 80 if kwargs.get("scope") == "alltime" else 8,
    }
    manager = ModelSettingsManager(directory)
    manager.set_settings(
        MODEL, ModelSettings(temperature=0.7, top_p=0.9, max_tokens=1024)
    )
    pointer = {"default": MODEL, "key": settings.auth.api_key, "restarts": 0}

    def request_restart():
        pointer["restarts"] += 1

    context = ManagementContext(
        engine_pool=pool,
        settings_manager=manager,
        global_settings=settings,
        get_default_model=lambda: pointer["default"],
        set_default_model=lambda model_id: pointer.update(default=model_id),
        apply_sampling=lambda: None,
        get_api_key=lambda: pointer["key"],
        set_api_key=lambda key: pointer.update(key=key),
        get_bind_host=lambda: "127.0.0.1",
        get_server_info=lambda: {
            "version": "fixture",
            "host": "127.0.0.1",
            "port": 8765,
        },
        runtime_state=SimpleNamespace(
            server_metrics=metrics, request_restart=request_restart
        ),
    )
    app.state.context = context
    app.state.management_context_provider = lambda: context
    app.state.management_auth_provider = lambda: AuthContext(
        pointer["key"], settings.auth.sub_keys, "127.0.0.1"
    )
    app.state.management_runtime = ManagementRuntime(context)

    async def completed_record(operation_id):
        return {"verified": True, "weights_loaded": False}

    operation = app.state.management_runtime.control.start_operation(
        "fixture_verify", completed_record, model_id=MODEL, cancellable=False
    )
    await app.state.management_runtime.control._tasks[operation["id"]]
    return {"ok": True}


@app.get("/__test__/persisted-auth")
async def persisted_auth():
    # This state exists only under this fixture's TemporaryDirectory.
    path = app.state.context.global_settings.base_path / "settings.json"
    return {"api_key": json.loads(path.read_text())["auth"].get("api_key")}


@app.post("/__test__/fail-save")
async def fail_save():
    def failed_save(*args, **kwargs):
        raise OSError("Fixture settings storage unavailable")

    app.state.context.global_settings._save_data = failed_save
    return {"ok": True}


# Synthetic protocol endpoints exercise the public transport, never inference.
@app.get("/__test__/proxy-metrics")
async def proxy_metrics():
    return app.state.proxy_metrics


@app.get("/v1/models", dependencies=[Depends(require_model_load_key)])
async def inference_models(request: Request):
    return {
        "object": "list",
        "data": [{"id": MODEL}],
        "authorization": request.headers.get("authorization"),
        "cookie": request.headers.get("cookie"),
        "query": str(request.url.query),
    }


@app.post("/v1/audio/transcriptions", dependencies=[Depends(require_model_load_key)])
async def multipart_echo(request: Request):
    body = await request.body()
    app.state.proxy_metrics["requests"].append(
        {
            "path": request.url.path,
            "authorization": request.headers.get("authorization"),
            "content_type": request.headers.get("content-type"),
            "bytes": list(body),
        }
    )
    return Response(
        body,
        media_type="application/octet-stream",
        headers={"X-Fixture-Upstream": "multipart"},
    )


@app.post("/v1/audio/speech", dependencies=[Depends(require_model_load_key)])
async def binary_audio():
    return Response(
        bytes([0, 255, 1, 128]),
        media_type="audio/wav",
        headers={"X-Fixture-Upstream": "binary"},
    )


@app.post("/v1/embeddings", dependencies=[Depends(require_model_load_key)])
async def inference_error():
    return JSONResponse(
        {"error": {"message": "Synthetic model unavailable", "type": "fixture_error"}},
        status_code=422,
        headers={"X-Fixture-Upstream": "error"},
    )


@app.post("/v1/chat/completions", dependencies=[Depends(require_model_load_key)])
async def inference_stream():
    async def stream():
        app.state.proxy_metrics["streams_started"] += 1
        try:
            yield 'data: {"fixture":"first"}\n\n'
            # A buffered proxy cannot finish this response. The client must see
            # the first chunk and abort, propagating cancellation to this wait.
            await asyncio.Event().wait()
        finally:
            app.state.proxy_metrics["streams_cancelled"] += 1

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"X-Fixture-Upstream": "stream"},
    )


@app.websocket("/v1/audio/transcriptions/realtime")
async def synthetic_realtime(websocket: WebSocket):
    from molto_server.api.audio_routes import _verify_ws_api_key

    await websocket.accept()
    try:
        first = await websocket.receive_text()
        start = json.loads(first)
        if start.get("type") != "start" or not _verify_ws_api_key(
            start.get("api_key"), websocket
        ):
            await websocket.close(1008, "Fixture authentication rejected")
            return
        # Echo the exact first frame, including its credential and field order.
        await websocket.send_text(first)
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                app.state.proxy_metrics["ws_closed"].append(message.get("code"))
                return
            if message.get("bytes") is not None:
                await websocket.send_bytes(message["bytes"])
            elif message.get("text") == "upstream-close":
                await websocket.close(1008, "Fixture upstream close")
                return
            else:
                await websocket.send_text(message.get("text", ""))
    except WebSocketDisconnect as exc:
        app.state.proxy_metrics["ws_closed"].append(exc.code)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        app, host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    )
