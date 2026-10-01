# SPDX-License-Identifier: Apache-2.0
"""State for the private inference application."""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from omlx_runtime.engine_pool import EnginePool
from omlx_runtime.server_metrics import ServerMetrics

# Import from new modular API
from omlx_server.api.responses_utils import ResponseStore

logger = logging.getLogger(__name__)


class EngineType(Enum):
    """Type of engine to retrieve."""

    LLM = "llm"
    EMBEDDING = "embedding"
    RERANKER = "reranker"


@dataclass
class SamplingDefaults:
    """Default sampling parameters."""

    # Fallback context length used by ``get_max_context_window`` only
    # when neither a per-model override nor a model-config-discovered
    # native context length is available. Setting this does NOT cap
    # models that declare their own context — use
    # ``max_context_window_policy`` for the operator-policy cap.
    max_context_window: int = 32768
    # Optional operator policy cap. When set, models whose native
    # context length is discovered get ``min(native, policy)``. Per-model
    # overrides and the fallback default above are not affected — those
    # represent explicit choices that the policy cannot override
    # without surprising migration semantics for existing
    # ``settings.json`` files.
    max_context_window_policy: int | None = None
    max_tokens: int = 32768
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 0
    repetition_penalty: float = 1.0
    force_sampling: bool = False


@dataclass
class ServerState:
    """Resources and configuration owned by one FastAPI application."""

    metrics: ServerMetrics = field(default_factory=ServerMetrics)
    engine_pool: EnginePool | None = None
    default_model: str | None = None
    cluster_services: object | None = None
    pairing_manager: object | None = None
    mcp_manager: object | None = None
    mcp_executor: object | None = None
    sampling: SamplingDefaults = field(default_factory=SamplingDefaults)
    api_key: str | None = None
    # Bind address snapshot for security checks. Unlike GlobalSettings.server.host,
    # this remains unchanged until the process restarts on the new address.
    bind_host: str | None = None
    bind_port: int | None = None
    settings_manager: object | None = None  # ModelSettingsManager
    global_settings: object | None = None  # GlobalSettings
    process_memory_enforcer: object | None = None  # ProcessMemoryEnforcer
    responses_store: ResponseStore = field(default_factory=ResponseStore)
    oq_manager: object | None = None  # No quantizer is started by the server.
    diffusion_jobs: object | None = None
    # False while the startup pinned-model preload is still running.
    # /health returns 503 with status "loading" until it flips to True so
    # port watchdogs see liveness instead of a closed port (#2184).
    pinned_preload_complete: bool = True
    # Snapshot at controller.initialize(). Settings may be edited while this process is
    # running, but routes, navigation, and Bonjour switch together on restart.
    distributed_inference_enabled: bool = False

    def request_restart(self) -> bool:
        """Request graceful termination only when a supervisor can respawn us."""
        import signal

        if not os.environ.get("OMLX_SUPERVISED"):
            return False

        def terminate():
            marker = os.environ.get("OMLX_RESTART_MARKER")
            if os.environ.get("OMLX_SUPERVISED") == "application" and marker:
                try:
                    settings = self.global_settings
                    Path(marker).write_text(
                        json.dumps(
                            {
                                "host": (
                                    settings.server.host if settings else self.bind_host
                                ),
                                "port": (
                                    settings.server.port if settings else self.bind_port
                                ),
                            }
                        )
                    )
                except OSError:
                    logger.exception("Could not record requested application restart")
            os.kill(os.getpid(), signal.SIGTERM)

        asyncio.get_running_loop().call_later(0.5, terminate)
        return True
