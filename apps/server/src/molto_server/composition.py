# SPDX-License-Identifier: Apache-2.0
"""Composition for the private inference application."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress
from functools import partial
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi import Request as FastAPIRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from molto_config._version import __version__
from molto_management.management import ManagementContext, ManagementService
from molto_runtime.engine_pool import EnginePool
from molto_runtime.server_metrics import ServerMetrics

# Import from new modular API
from molto_server.api.responses_utils import ResponseStore
from molto_server.auth import (
    AuthContext,
    fingerprint_key,
    require_management_key,
    verify_any_api_key,
)

from .state import SamplingDefaults

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


class CompositionController:
    def get_server_metrics(self):
        return self.state.metrics

    def _apply_global_sampling(self) -> None:
        """Apply persisted sampling defaults to the active request state."""
        settings = self.state.global_settings
        if settings is None:
            self.state.sampling = SamplingDefaults()
            return
        sampling = settings.sampling
        self.state.sampling = SamplingDefaults(
            max_context_window=sampling.max_context_window,
            max_context_window_policy=sampling.max_context_window_policy,
            max_tokens=sampling.max_tokens,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            repetition_penalty=sampling.repetition_penalty,
        )

    def get_engine_pool(self) -> EnginePool:
        """Get the engine pool, raising error if not initialized."""
        if self.state.engine_pool is None:
            raise HTTPException(status_code=503, detail="Server not initialized")
        return self.state.engine_pool

    def get_mcp_manager(self):
        """Get the MCP manager instance (may be None)."""
        return self.state.mcp_manager

    def mcp_tools_exposed(self) -> bool:
        """Whether backend MCP tools are exposed to clients.

        Controlled by mcp.expose_tools in the global settings file.
        Defaults to True (backward compatible) when global settings are
        unavailable, e.g. when MCP was started via env var/CLI without a
        settings file.
        """
        gs = self.state.global_settings
        if gs is None:
            return True
        return bool(getattr(gs.mcp, "expose_tools", True))

    async def verify_api_key(
        self,
        request: FastAPIRequest,
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> bool:
        """Verify API key unless an explicitly loopback-only server allows no auth.

        Checks the provided Bearer token against the main API key and all sub keys.
        Also accepts the x-api-key header as a fallback (Anthropic SDK compatibility).
        """
        from molto_config.utils.network import is_loopback_bind

        global_settings = self.state.global_settings
        configured_host = getattr(
            getattr(global_settings, "server", None), "host", None
        )
        if not isinstance(configured_host, str):
            configured_host = None
        bind_host = getattr(self.state, "bind_host", None)
        active_host = bind_host if isinstance(bind_host, str) else configured_host
        loopback_only = active_host is None or is_loopback_bind(active_host)

        # A missing key is accepted only when the configured bind is loopback-only.
        if self.state.api_key is None:
            if loopback_only:
                return True
            raise HTTPException(status_code=401, detail="API key required")

        # Skip verification if enabled
        if (
            global_settings is not None
            and global_settings.auth.skip_api_key_verification
            and loopback_only
        ):
            return True

        # Extract API key from Bearer token or x-api-key header
        if credentials is not None:
            api_key_value = credentials.credentials
        else:
            # Fallback: check x-api-key header (Anthropic SDK compatibility)
            api_key_value = request.headers.get("x-api-key")
            if api_key_value is None:
                raise HTTPException(status_code=401, detail="API key required")

        # Check main key and sub keys
        sub_keys = global_settings.auth.sub_keys if global_settings is not None else []
        if not verify_any_api_key(api_key_value, self.state.api_key, sub_keys):
            logger.warning("Rejected API key (fp=%s)", fingerprint_key(api_key_value))
            raise HTTPException(status_code=401, detail="Invalid API key")

        return True

    def allows_unauthenticated_inference(self) -> bool:
        settings = self.state.global_settings
        return (
            settings is not None
            and settings.auth.allow_unauthenticated_inference is True
        )

    async def verify_inference_api_key(
        self,
        request: FastAPIRequest,
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ) -> bool:
        """Allow the manual inference opt-in without changing management auth."""
        if self.allows_unauthenticated_inference():
            return True
        return await self.verify_api_key(request, credentials)

    def distributed_inference_enabled(self) -> bool:
        """Whether the experimental distributed surface is exposed this run."""

        return self.state.distributed_inference_enabled

    async def require_distributed_inference_enabled(self) -> bool:
        """Hide the experimental cluster surface until explicitly enabled."""

        if not self.distributed_inference_enabled():
            raise HTTPException(status_code=404, detail="Not found")
        return True

    def _reset_boundary_snapshots_for_server(self) -> None:
        """Reset ephemeral boundary snapshots at server lifecycle boundaries."""
        engine_pool = self.state.engine_pool
        if engine_pool is None:
            return

        scheduler_config = getattr(engine_pool, "_scheduler_config", None)
        cache_dir = getattr(scheduler_config, "paged_ssd_cache_dir", None)
        if not cache_dir:
            return

        try:
            from molto_runtime.cache.boundary_snapshot_store import (
                reset_boundary_snapshot_root,
            )

            reset_boundary_snapshot_root(Path(cache_dir))
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            logger.warning("Failed to reset boundary snapshot directory: %s", exc)

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """FastAPI lifespan for startup/shutdown events."""
        from molto_runtime.cluster.discovery import BonjourPublisher

        bonjour_publisher = None
        bonjour_task = None
        self._reset_boundary_snapshots_for_server()

        # Reap distributed ranks orphaned by a crashed previous coordinator
        # (G8): all teardown used to live in-process, so a SIGKILL/panic of
        # molto-server stranded loaded ranks with no owner. The launch manifest
        # written at spawn lets this new coordinator finish the teardown.
        # Best effort: a reaping failure must never block server startup.
        try:
            from molto_runtime.cluster.launch import reap_orphaned_launches

            orphan_report = await asyncio.to_thread(reap_orphaned_launches)
            if orphan_report["reaped"] or orphan_report["failures"]:
                logger.warning(
                    "Reaped orphaned distributed launches from a previous "
                    "coordinator: %s",
                    orphan_report,
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Orphaned-launch reaper failed at startup: %s", exc)

        # Publish the interpreter another Mac's coordinator discovers over SSH.
        # Without it a packaged-app peer fails every discovery candidate and gets
        # reported as "worker runtime is not installed" (#2680). Best effort: a
        # read-only home must never keep this node from serving inference.
        if self.distributed_inference_enabled():
            try:
                from molto_runtime.cluster.worker_shim import ensure_cluster_python_shim

                ensure_cluster_python_shim()
            except (ImportError, OSError, RuntimeError) as exc:
                # RuntimeError: Path.home() cannot resolve a home directory.
                logger.warning(
                    "Could not publish the cluster interpreter shim; a peer "
                    "coordinator may not discover this node over SSH: %r",
                    exc,
                )

        # Advertise this Molto instance so another Mac can identify it by hostname
        # and API port without asking the user to type an SSH target. Publication
        # is best-effort: inference remains available if Bonjour is disabled.
        if (
            self.state.global_settings is not None
            and self.distributed_inference_enabled()
            and os.environ.get("MOLTO_BONJOUR", "1").strip().lower()
            not in {"0", "false", "no", "off"}
        ):
            bonjour_publisher = BonjourPublisher(
                port=self.state.global_settings.server.port,
                version=__version__,
            )
            bonjour_publisher.start()

            async def _bonjour_supervisor() -> None:
                while True:
                    try:
                        bonjour_publisher.ensure_running()
                        await asyncio.sleep(5.0)
                    except asyncio.CancelledError:
                        break

            bonjour_task = asyncio.create_task(_bonjour_supervisor())

        # Cluster v2: always-on peer discovery (mDNS + IPv6 multicast fallback +
        # manual + Tailscale). Best-effort: discovery failures must never block
        # serving. MOLTO_DISCOVERY=0 disables it for hostile networks.
        discovery_service = None
        if self.distributed_inference_enabled() and os.environ.get(
            "MOLTO_DISCOVERY", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}:
            try:
                from molto_runtime.cluster.discovery import (
                    DiscoveryConfig,
                    DiscoveryService,
                    load_cluster_name,
                )

                cluster_base = (
                    Path(self.state.global_settings.base_path)
                    if self.state.global_settings is not None
                    else Path.home() / ".molto"
                )
                discovery_service = DiscoveryService(
                    self.state.cluster_services.identity,
                    self.state.cluster_services.devices,
                    DiscoveryConfig(
                        cluster_name=load_cluster_name(cluster_base),
                        http_port=(
                            self.state.global_settings.server.port
                            if self.state.global_settings is not None
                            else 8000
                        ),
                    ),
                )
                self.state.cluster_services.discovery = discovery_service
                await asyncio.to_thread(discovery_service.start)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Cluster discovery service failed to start: %s", exc)
                discovery_service = None

        # Start process memory enforcer if configured
        if (
            self.state.global_settings is not None
            and self.state.engine_pool is not None
        ):
            from molto_runtime.process_memory_enforcer import ProcessMemoryEnforcer

            memory_settings = self.state.global_settings.memory
            enforcer = ProcessMemoryEnforcer(
                engine_pool=self.state.engine_pool,
                memory_guard_tier=memory_settings.memory_guard_tier,
                memory_guard_custom_ceiling_gb=memory_settings.memory_guard_custom_ceiling_gb,
                settings_manager=self.state.settings_manager,
                prefill_memory_guard=memory_settings.prefill_memory_guard,
                global_settings=self.state.global_settings,
                soft_threshold=memory_settings.soft_threshold,
                hard_threshold=memory_settings.hard_threshold,
                prefill_safe_zone_ratio=memory_settings.prefill_safe_zone_ratio,
                prefill_min_chunk_tokens=memory_settings.prefill_min_chunk_tokens,
            )
            self.state.process_memory_enforcer = enforcer
            self.state.engine_pool.configure_memory_enforcer(enforcer)
            enforcer.start()

        # Startup: Preload pinned models in the background so uvicorn binds the
        # HTTP port immediately after this lifespan yields. A large pinned
        # preload (hundreds of GB) otherwise keeps the port closed for minutes,
        # and anything watchdogging the port hard-kills the process mid-load —
        # the worst moment for the kernel wired-memory stranding in #2184.
        # /health answers 503 with status "loading" until the preload finishes.
        # Runs after the enforcer wiring above so the preload sees the final
        # memory ceiling.
        preload_task = None
        if self.state.engine_pool is not None:
            self.state.pinned_preload_complete = False

            async def _preload_pinned() -> None:
                try:
                    await self.state.engine_pool.preload_pinned_models()
                finally:
                    self.state.pinned_preload_complete = True

            preload_task = asyncio.create_task(_preload_pinned())

        # Start TTL-only checker if process memory enforcer is not running
        # (enforcer already includes TTL checks in its polling loop)
        ttl_task = None
        if (
            self.state.process_memory_enforcer is None
            and self.state.engine_pool is not None
        ):

            async def _ttl_check_loop():
                while True:
                    try:
                        if self.state.settings_manager is not None:
                            await self.state.engine_pool.check_ttl_expirations(
                                self.state.settings_manager,
                                global_idle_timeout_seconds=(
                                    self.state.global_settings.idle_timeout.idle_timeout_seconds
                                    if self.state.global_settings
                                    else None
                                ),
                            )
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.error(f"TTL check error: {e}")
                    await asyncio.sleep(1.0)

            ttl_task = asyncio.create_task(_ttl_check_loop())

        # Initialize MCP if config provided
        # Priority: env var > settings.json
        mcp_config = os.environ.get("MOLTO_MCP_CONFIG")
        if not mcp_config and self.state.global_settings:
            mcp_config = self.state.global_settings.mcp.config_path
        if mcp_config:
            await self.init_mcp(mcp_config)

        yield

        # Shutdown: Save all-time stats, stop TTL task, process memory enforcer, etc.
        if bonjour_task is not None:
            bonjour_task.cancel()
            with suppress(asyncio.CancelledError):
                await bonjour_task
        if bonjour_publisher is not None:
            bonjour_publisher.stop()
        if discovery_service is not None:
            try:
                await asyncio.to_thread(discovery_service.stop)
                self.state.cluster_services.discovery = None
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Cluster discovery service failed to stop: %s", exc)
        if preload_task is not None and not preload_task.done():
            # SIGTERM arrived while pinned models were still loading. Cancel the
            # await; engine_pool.shutdown() below unloads whatever finished.
            preload_task.cancel()
            with suppress(asyncio.CancelledError):
                await preload_task
        management_runtime = getattr(self.app.state, "management_runtime", None)
        if management_runtime is not None:
            await management_runtime.shutdown()
            self.app.state.management_runtime = None
        self.get_server_metrics().close()
        if ttl_task is not None:
            ttl_task.cancel()
            try:
                await ttl_task
            except asyncio.CancelledError:
                pass
        if self.state.diffusion_jobs is not None:
            await self.state.diffusion_jobs.shutdown()
            self.state.diffusion_jobs = None
        if self.state.process_memory_enforcer is not None:
            await self.state.process_memory_enforcer.stop()
            if self.state.engine_pool is not None:
                self.state.engine_pool.configure_memory_enforcer(None)
            logger.info("Process memory enforcer stopped")
        if self.state.mcp_manager is not None:
            await self.state.mcp_manager.stop()
            logger.info("MCP manager stopped")
        if self.state.engine_pool is not None:
            await self.state.engine_pool.shutdown()
            self._reset_boundary_snapshots_for_server()
            logger.info("Engine pool shutdown")

    def _management_context(self) -> ManagementContext:
        pool = self.state.engine_pool
        manager = self.state.settings_manager
        if pool is None or manager is None:
            raise HTTPException(status_code=503, detail="Server not initialized")
        return ManagementContext(
            engine_pool=pool,
            settings_manager=manager,
            global_settings=self.state.global_settings,
            get_default_model=lambda: self.state.default_model,
            set_default_model=lambda model_id: setattr(
                self.state, "default_model", model_id
            ),
            apply_sampling=self._apply_global_sampling,
            get_api_key=lambda: self.state.api_key,
            set_api_key=lambda key: setattr(self.state, "api_key", key),
            get_bind_host=lambda: self.state.bind_host or "127.0.0.1",
            get_server_info=lambda: {
                "version": __version__,
                "bind_host": self.state.bind_host,
                "port": self.state.bind_port,
                "restart_supported": bool(os.environ.get("MOLTO_SUPERVISED")),
                "distributed_inference_active": self.state.distributed_inference_enabled,
                "supervisor": os.environ.get("MOLTO_SUPERVISED"),
                "uptime_seconds": self.get_server_metrics()
                .get_snapshot()
                .get("uptime_seconds", 0),
            },
            runtime_state=self.state,
        )

    def _management_auth_context(self) -> AuthContext:
        settings = self.state.global_settings
        return AuthContext(
            main_key=self.state.api_key,
            sub_keys=settings.auth.sub_keys if settings is not None else [],
            bind_host=self.state.bind_host,
            skip_api_key_verification=(
                settings.auth.skip_api_key_verification
                if settings is not None
                else False
            ),
        )

    def _diffusion_jobs_provider(self):
        jobs = self.state.diffusion_jobs
        if jobs is None:
            raise HTTPException(status_code=503, detail="Server not initialized")
        return jobs

    def _register_cluster_routes(self) -> None:
        """Register experimental routes only for an opted-in server process."""

        if self.cluster_routes_registered:
            return
        from molto_server.cluster.routes import join_router as cluster_join_router
        from molto_server.cluster.routes import router as cluster_router

        self.app.include_router(
            cluster_router,
            dependencies=[
                Depends(require_management_key),
                Depends(self.require_distributed_inference_enabled),
            ],
        )
        # Cluster v2 model-sync manifest: admin-gated like the rest of the
        # cluster surface; peers use it to compare model contents before sync.
        from molto_server.cluster.modelsync_routes import (
            manifest_router as cluster_manifest_router,
        )

        self.app.include_router(
            cluster_manifest_router,
            dependencies=[
                Depends(require_management_key),
                Depends(self.require_distributed_inference_enabled),
            ],
        )
        # The bootstrap bytes are public but pinned by SHA-256 in an admin-created
        # command. Claim/source/complete authenticate with one-time enrollment
        # credentials, not the browser's admin cookie.
        self.app.include_router(
            cluster_join_router,
            dependencies=[Depends(self.require_distributed_inference_enabled)],
        )
        # Cluster v2: /api/cluster/node_id is a deliberately unauthenticated,
        # rate-limited probe peers use to verify announced addresses before any
        # pairing trust exists; /api/cluster/devices requires admin per-route.
        from molto_server.cluster.discovery_routes import discovery_router

        self.app.include_router(
            discovery_router,
            dependencies=[Depends(self.require_distributed_inference_enabled)],
        )
        # Cluster v2 pairing (Module B): pair/request carries only a salted PBKDF2
        # verifier bound to the node and SSH identities; pair/status returns a
        # code-encrypted cluster key only after admin approval. Approve, deny, and
        # unpair are admin-only like every other cluster mutation.
        from molto_server.cluster.pairing_routes import pair_admin_router, pair_router

        self.app.include_router(
            pair_router,
            dependencies=[Depends(self.require_distributed_inference_enabled)],
        )
        self.app.include_router(
            pair_admin_router,
            dependencies=[
                Depends(require_management_key),
                Depends(self.require_distributed_inference_enabled),
            ],
        )
        self.cluster_routes_registered = True

    def initialize(
        self,
        model_dirs: str | list[str],
        scheduler_config=None,
        api_key: str | None = None,
        global_settings: object | None = None,
    ):
        """
        Initialize server with model directories for multi-model serving.

        Args:
            model_dirs: Path or list of paths to directories containing model subdirectories
            scheduler_config: Scheduler config for BatchedEngine
            api_key: API key for authentication (optional)
            global_settings: GlobalSettings instance (optional)

        Note:
            - Pinned models and default model are managed in model_settings.json
            - Sampling parameters (max_tokens, temperature, etc.) are per-model settings

        Raises:
            ValueError: If network authentication is unsafe, the model directory
                doesn't exist, or no models are found.
        """
        from pathlib import Path

        from molto_config.model_settings import ModelSettingsManager
        from molto_config.utils.network import network_auth_error

        if global_settings is not None:
            auth_error = network_auth_error(
                global_settings.server.host,
                api_key,
                global_settings.auth.skip_api_key_verification,
            )
            if auth_error:
                raise ValueError(auth_error)

        if global_settings is not None:
            global_settings.ensure_inference_auth_setting()

        # Store API key
        self.state.api_key = api_key
        self.state.global_settings = global_settings
        self.state.bind_host = (
            global_settings.server.host if global_settings is not None else None
        )
        self.state.bind_port = (
            global_settings.server.port if global_settings is not None else 8000
        )
        if self.allows_unauthenticated_inference():
            logger.warning(
                "Unauthenticated inference is enabled on %s. Anyone who can connect "
                "can use inference, stored Responses, audio, MCP tools, and web "
                "search/fetch. Management endpoints still require authentication "
                "on non-loopback binds.",
                self.state.bind_host,
            )
        from molto_runtime.cluster.exposure import (
            distributed_inference_enabled as is_enabled,
        )

        self.state.distributed_inference_enabled = is_enabled(global_settings)
        if self.state.distributed_inference_enabled:
            self._register_cluster_routes()
        response_state_dir = None
        if global_settings:
            response_state_dir = (
                global_settings.cache.get_ssd_cache_dir(global_settings.base_path)
                / "response-state"
            )
        self.state.responses_store = ResponseStore(state_dir=response_state_dir)

        # Configure CORS middleware from settings
        cors_origins = global_settings.server.cors_origins if global_settings else ["*"]
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info(f"CORS origins: {cors_origins}")

        # Initialize model settings manager
        base_path = (
            Path(global_settings.base_path)
            if global_settings
            else Path.home() / ".molto"
        )
        self.state.settings_manager = ModelSettingsManager(base_path)

        # Get pinned models from persisted settings.
        pinned_models = self.state.settings_manager.get_pinned_model_ids()

        # Get default model from persisted settings.
        settings_default = self.state.settings_manager.get_default_model_id()

        # Load default sampling values from global settings
        # Per-model settings will override these via get_sampling_params()
        self._apply_global_sampling()

        # Normalize model_dirs to list
        if isinstance(model_dirs, str):
            dir_list = [model_dirs]
        else:
            dir_list = list(model_dirs)
        if global_settings and hasattr(global_settings, "get_effective_model_dirs"):
            dir_list = [str(d) for d in global_settings.get_effective_model_dirs()]

        # Create directories if needed
        for md in dir_list:
            model_path = Path(md)
            if not model_path.exists():
                model_path.mkdir(parents=True, exist_ok=True)
                logger.warning(f"Model directory created (empty): {md}")

        # Create engine pool. The pool consults enforcer.get_final_ceiling()
        # for pre-load admission — wired up later in startup once the enforcer
        # is constructed.
        self.state.engine_pool = EnginePool(
            scheduler_config=scheduler_config,
        )
        self.state.engine_pool.configure_gpu_keep_warm(
            global_settings.server.gpu_keep_warm_interval if global_settings else 0.5
        )
        from molto_management.diffusion_jobs import DiffusionJobs

        async def refresh_prepared_models():
            await ManagementService(self._management_context()).refresh()

        self.state.diffusion_jobs = None
        try:
            self.state.diffusion_jobs = DiffusionJobs(
                self.state.engine_pool,
                artifacts_dir=base_path / "preparation" / "diffusion",
                output_dir=Path(dir_list[0]),
                on_complete=refresh_prepared_models,
            )
        except (OSError, ValueError) as exc:
            logger.warning("Diffusion preparation jobs unavailable: %s", exc)
        from molto_runtime.cluster.discovery import announced_addrs
        from molto_runtime.cluster.launch import attach_rdma_stage_links
        from molto_runtime.cluster.pairing import PairingManager
        from molto_runtime.cluster.rdma.launch_links import enrolled_node_addresses

        from molto_server.cluster.services import ClusterServices

        services = ClusterServices.create(base_path)
        self.state.cluster_services = services
        self.state.engine_pool.configure_cluster_registry(services.registry)
        self.state.engine_pool.configure_rdma_stage_links(
            partial(
                attach_rdma_stage_links,
                nodes=lambda: enrolled_node_addresses(services.enrollment),
                store=lambda: services.rdma_links,
            )
        )
        self.state.pairing_manager = PairingManager(
            services.devices,
            services.enrollment,
            base_path=base_path,
            caps_provider=services.announced_caps,
            address_provider=announced_addrs,
            http_port=global_settings.server.port if global_settings else 8000,
        )

        # Discover models (use pinned models from settings file)
        self.state.engine_pool.configure_settings_manager(self.state.settings_manager)
        self.state.engine_pool.discover_models(dir_list, pinned_models)
        self.state.engine_pool.apply_settings_overrides(self.state.settings_manager)

        if self.state.engine_pool.model_count == 0:
            logger.warning(
                f"No models found in {', '.join(dir_list)}. Add models to serve them."
            )

        # Set default model (from settings file, fallback to first model)
        available_models = self.state.engine_pool.get_model_ids()
        if available_models:
            if settings_default:
                if settings_default in available_models:
                    self.state.default_model = settings_default
                else:
                    logger.warning(
                        f"Default model '{settings_default}' not found, using first model"
                    )
                    self.state.default_model = available_models[0]
            else:
                self.state.default_model = available_models[0]
        else:
            self.state.default_model = None

        # Reset server metrics for fresh start (with all-time persistence)
        stats_path = base_path / "stats.json"
        self.state.metrics.close()
        self.state.metrics = ServerMetrics(stats_path=stats_path)
        from molto_runtime.usage_history import UsageHistory

        try:
            self.state.metrics.usage_history = UsageHistory(
                stats_path.parent / "usage.sqlite3",
                enabled=getattr(
                    getattr(global_settings, "usage", None), "usage_history", True
                ),
            )
        except Exception:
            logger.warning("Usage history initialization failed; serving continues")

        logger.info(
            f"Server initialized with {self.state.engine_pool.model_count} models"
        )
        if self.state.default_model:
            logger.info(f"Default model: {self.state.default_model}")
        else:
            logger.info("No default model (no models available)")
        if global_settings and getattr(global_settings, "memory", None):
            logger.info(
                f"Memory guard tier: {global_settings.memory.memory_guard_tier} "
                f"(guard {'on' if global_settings.memory.prefill_memory_guard else 'off'})"
            )
        logger.info(f"Default max tokens: {self.state.sampling.max_tokens}")
        if api_key:
            logger.info("API key authentication: enabled")

    async def init_mcp(self, config_path: str):
        """Initialize MCP manager from config file."""
        try:
            from molto_server.mcp import MCPClientManager, ToolExecutor, load_mcp_config

            config = load_mcp_config(config_path)
            self.state.mcp_manager = MCPClientManager(config)
            await self.state.mcp_manager.start()

            self.state.mcp_executor = ToolExecutor(self.state.mcp_manager)

            logger.info(
                f"MCP initialized with {len(self.state.mcp_manager.get_all_tools())} tools"
            )

        except ImportError:
            logger.warning(
                "MCP SDK not installed. MCP features disabled. "
                "Install with: pip install mcp"
            )
            return
        except Exception as e:
            logger.error(
                f"Failed to initialize MCP: {e}. "
                "MCP features disabled. Fix your MCP config and restart."
            )
            return
