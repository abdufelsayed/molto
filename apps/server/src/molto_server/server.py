"""Composition root for the private Molto FastAPI application."""

import argparse
import os

from fastapi import Depends, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from molto_config._version import __version__
from molto_config.config import DEFAULT_SERVER_PORT
from molto_management.management import ManagementError
from molto_runtime.engine.distributed import DistributedInferenceError
from molto_runtime.exceptions import (
    InvalidRequestError,
    PrefillMemoryExceededError,
    SchedulerQueueFullError,
)

from .anthropic import AnthropicController
from .composition import CompositionController
from .dependencies import verify_inference_api_key
from .engine_requests import EngineRequestsController
from .errors import ErrorsController
from .inventory import InventoryController
from .openai import OpenaiController
from .openai_streaming import OpenaiStreamingController
from .responses import ResponsesController
from .shared import ClientDisconnectTrackingMiddleware, DebugRequestLoggingMiddleware
from .state import ServerState
from .transport import TransportController


class ServerApplication(
    CompositionController,
    ErrorsController,
    EngineRequestsController,
    TransportController,
    InventoryController,
    OpenaiController,
    OpenaiStreamingController,
    AnthropicController,
    ResponsesController,
):
    """Instance-owned protocol controllers and application resources."""

    def __init__(self, state: ServerState):
        self.state = state
        self.cluster_routes_registered = False


def create_app(state: ServerState | None = None) -> FastAPI:
    controller = ServerApplication(state if state is not None else ServerState())
    app = FastAPI(
        title="Molto API",
        description="LLM inference, optimized for your Mac",
        version=__version__,
        lifespan=controller.lifespan,
    )
    controller.app = app
    app.state.server_state = controller.state
    app.state.controller = controller
    app.state.management_context_provider = controller._management_context
    app.state.management_auth_provider = controller._management_auth_context
    app.state.diffusion_jobs_provider = controller._diffusion_jobs_provider
    from .api.image_routes import router as image_router
    from .api.management_routes import router as management_router
    from .api.management_setup_routes import router as setup_router
    from .api.mcp_routes import router as mcp_router
    from .api.websearch_routes import router as websearch_router

    for router in (mcp_router, websearch_router, image_router):
        app.include_router(router, dependencies=[Depends(verify_inference_api_key)])
    app.include_router(management_router)
    app.include_router(setup_router)
    try:
        import mlx_audio  # noqa: F401
    except ImportError:
        pass
    else:
        from .api.audio_routes import realtime_router
        from .api.audio_routes import router as audio_router

        app.include_router(
            audio_router, dependencies=[Depends(verify_inference_api_key)]
        )
        app.include_router(realtime_router)
    app.add_middleware(ClientDisconnectTrackingMiddleware)
    app.add_middleware(DebugRequestLoggingMiddleware)
    app.add_exception_handler(ManagementError, controller.management_error_handler)
    app.add_exception_handler(HTTPException, controller.http_exception_handler)
    app.add_exception_handler(
        RequestValidationError, controller.validation_exception_handler
    )
    app.add_exception_handler(
        InvalidRequestError, controller.invalid_request_error_handler
    )
    app.add_exception_handler(
        SchedulerQueueFullError, controller.scheduler_queue_full_handler
    )
    app.add_exception_handler(
        DistributedInferenceError, controller.distributed_unavailable_handler
    )
    app.add_exception_handler(
        PrefillMemoryExceededError, controller.prefill_memory_exceeded_handler
    )
    app.add_exception_handler(Exception, controller.unhandled_exception_handler)
    app.add_api_route("/health", controller.health, methods=["GET"])
    app.add_api_route("/api/status", controller.server_status, methods=["GET"])
    app.add_api_route("/v1/models", controller.list_models, methods=["GET"])
    app.add_api_route(
        "/v1/models/status", controller.list_models_status, methods=["GET"]
    )
    app.add_api_route(
        "/v1/models/{model_id}/unload", controller.unload_model, methods=["POST"]
    )
    app.add_api_route(
        "/v1/models/{model_id}/load", controller.load_model_public, methods=["POST"]
    )
    app.add_api_route("/v1/embeddings", controller.create_embeddings, methods=["POST"])
    app.add_api_route("/v1/rerank", controller.create_rerank, methods=["POST"])
    app.add_api_route("/v1/completions", controller.create_completion, methods=["POST"])
    app.add_api_route(
        "/v1/chat/completions", controller.create_chat_completion, methods=["POST"]
    )
    app.add_api_route(
        "/v1/messages", controller.create_anthropic_message, methods=["POST"]
    )
    app.add_api_route(
        "/v1/messages/count_tokens", controller.count_anthropic_tokens, methods=["POST"]
    )
    app.add_api_route("/v1/responses", controller.create_response, methods=["POST"])
    app.add_api_route(
        "/v1/responses/{response_id}", controller.get_response, methods=["GET"]
    )
    app.add_api_route(
        "/v1/responses/{response_id}", controller.delete_response, methods=["DELETE"]
    )
    return app


app = create_app()


def main():
    """Run the server (use molto CLI instead)."""
    parser = argparse.ArgumentParser(
        description="Molto multi-model serving for Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Multi-model serving
    python -m molto_server.server --model-dir /path/to/models

    # With MCP tools
    python -m molto_server.server --model-dir /path/to/models --mcp-config mcp.json

Note: Use the molto CLI for full feature support. Pinned models, default
model and sampling defaults are managed via the management API.
        """,
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing model subdirectories",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind to (default: settings or 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help="Port to bind to",
    )
    parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP configuration file (JSON/YAML)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (required for non-loopback binds)",
    )

    args = parser.parse_args()

    # Set MCP config for lifespan
    if args.mcp_config:
        os.environ["MOLTO_MCP_CONFIG"] = args.mcp_config

    # Pass loaded settings and the API key to the same server initializer
    # used by the CLI. This entry point keeps scheduler/cache wiring minimal.
    from molto_config.settings import init_settings

    settings = init_settings()
    if args.host is not None:
        settings.server.host = args.host
    settings.server.port = args.port
    if args.api_key is not None:
        settings.auth.api_key = args.api_key
    errors = settings.validate()
    if errors:
        for error in errors:
            print(f"Configuration error: {error}")
        raise SystemExit(1)
    settings.ensure_directories()

    # Match the cli.py launcher: keep freed GPU buffers in the pool so
    # allocator::free() never releases a buffer the GPU may still be
    # using (kernel panics on M4 otherwise; see cli.py and issue #300).
    # EnginePool eviction also assumes this cache limit is in place.
    import mlx.core as mx

    total_mem = mx.device_info().get("memory_size", 0)
    if total_mem > 0:
        mx.set_cache_limit(total_mem)

    # Initialize server
    app = create_app()
    app.state.controller.initialize(
        model_dirs=args.model_dir,
        api_key=settings.auth.api_key,
        global_settings=settings,
    )

    # Start server
    import uvicorn

    uvicorn.run(app, host=settings.server.host, port=args.port)


if __name__ == "__main__":
    main()
