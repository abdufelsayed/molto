"""Private inference child startup, invoked only by the application supervisor."""

import faulthandler
import sys

from molto_config.startup import _has_cli_overrides, _migrate_saved_network_auth


def serve_private(args):
    import logging
    import os
    import socket

    from molto_server.server import create_app

    application = create_app()

    import uvicorn
    from molto_config import process_title
    from molto_config._version import __version__
    from molto_config.logging_config import (
        ManagementAccessFilter,
        configure_file_logging,
    )
    from molto_config.settings import burst_decode_env, init_settings

    process_title.set_process_title()

    try:
        from molto_runtime._build_info import build_number
    except ImportError:
        build_number = None

    # Redirected startup logs and NO_COLOR stay plain text.
    colored = (
        sys.stdout.isatty()
        and "NO_COLOR" not in os.environ
        and not getattr(args, "no_color", False)
    )
    prefix, suffix = ("\033[33m", "\033[0m") if colored else ("", "")
    print(f"{prefix}Molto - LLM inference, optimized for your Mac{suffix}")
    print(f"{prefix}├─ https://github.com/abdufelsayed/molto{suffix}")
    if build_number:
        print(f"{prefix}├─ Version: {__version__}{suffix}")
        print(f"{prefix}└─ Build: {build_number}{suffix}")
    else:
        print(f"{prefix}└─ Version: {__version__}{suffix}")
    print()

    # Initialize global settings first (to get log_level from file if not specified)
    settings = init_settings(base_path=args.base_path, cli_args=args)

    # The native ANE compile-cache gate reads this env var once, at the first
    # compile, so it must be exported before any engine loads. setdefault
    # keeps an explicit env override authoritative.
    if settings.cache.ane_compile_cache:
        os.environ.setdefault("MOLTO_QWEN35_ANE_COMPILE_CACHE", "1")

    # Register TRACE level (5) — includes full message content
    TRACE = 5
    logging.addLevelName(TRACE, "TRACE")

    # Configure logging (use settings value which has proper priority)
    level_name = settings.server.log_level.upper()
    log_level = (
        TRACE if level_name == "TRACE" else getattr(logging, level_name, logging.INFO)
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Set molto loggers
    for name in [
        "molto",
        "molto_runtime.scheduler",
        "molto_runtime.paged_ssd_cache",
        "molto_runtime.memory_monitor",
        "molto_runtime.paged_cache",
        "molto_runtime.prefix_cache",
        "molto_runtime.engine_pool",
        "molto_runtime.model_discovery",
    ]:
        logging.getLogger(name).setLevel(log_level)

    # Suppress repetitive successful management polling access logs
    logging.getLogger("uvicorn.access").addFilter(ManagementAccessFilter())

    # Suppress noisy third-party loggers unless trace level
    if log_level > TRACE:
        logging.getLogger("httpcore").setLevel(logging.INFO)
        logging.getLogger("httpx").setLevel(logging.INFO)

    # Ensure required directories exist
    settings.ensure_directories()

    # Apply HuggingFace endpoint if configured
    if settings.huggingface.endpoint:
        os.environ["HF_ENDPOINT"] = settings.huggingface.endpoint

    # Apply ModelScope endpoint if configured
    if settings.modelscope.endpoint:
        os.environ["MODELSCOPE_DOMAIN"] = settings.modelscope.endpoint

    # Apply proxy/TLS settings if configured
    if settings.network.http_proxy:
        os.environ["HTTP_PROXY"] = settings.network.http_proxy
        os.environ["http_proxy"] = settings.network.http_proxy
    if settings.network.https_proxy:
        os.environ["HTTPS_PROXY"] = settings.network.https_proxy
        os.environ["https_proxy"] = settings.network.https_proxy
    if settings.network.no_proxy:
        os.environ["NO_PROXY"] = settings.network.no_proxy
        os.environ["no_proxy"] = settings.network.no_proxy
    if settings.network.ca_bundle:
        os.environ["REQUESTS_CA_BUNDLE"] = settings.network.ca_bundle
        os.environ["SSL_CERT_FILE"] = settings.network.ca_bundle

    # Seed Burst Decode env vars so EngineConfig picks up the saved mode at
    # engine construction (no restart needed when the mode changes later).
    for _key, _value in burst_decode_env(settings.server.burst_decode_mode).items():
        os.environ[_key] = _value

    # Validate before persisting CLI overrides, so invalid flags never poison
    # settings.json.
    try:
        _migrate_saved_network_auth(settings, args)
    except (OSError, ValueError) as error:
        print(f"Configuration error: {error}")
        sys.exit(1)
    errors = settings.validate()
    if errors:
        for error in errors:
            print(f"Configuration error: {error}")
        sys.exit(1)

    # Persist explicit flags once. Supervised reloads retain their effective
    # precedence without overwriting settings edited through the dashboard.
    if _has_cli_overrides(args) and os.environ.get("MOLTO_BACKEND_RESTART") != "1":
        try:
            settings.save_cli_overrides(args)
            print("Saved CLI arguments to settings.json")
        except Exception as e:
            print(f"Warning: Failed to save settings: {e}")

    # Configure file logging (writes to {base_path}/logs/server.log)
    log_dir = settings.logging.get_log_dir(settings.base_path)
    configure_file_logging(
        log_dir=log_dir,
        level=settings.server.log_level,
        include_request_id=True,
        retention_days=settings.logging.retention_days,
    )
    print(f"Log directory: {log_dir}")

    # Enable native crash diagnostics (SIGABRT, SIGSEGV, SIGFPE, SIGBUS).
    # On Metal/MLX crashes (#511, #520), this dumps all Python thread
    # tracebacks to the server log before the process terminates.
    crash_log_path = log_dir / "crash.log"
    _crash_file = open(crash_log_path, "a")
    faulthandler.enable(file=_crash_file, all_threads=True)

    # Bind the socket before importing/initializing the server. Uvicorn's
    # normal startup runs ASGI lifespan before binding host/port, which means
    # pinned models can be preloaded before a port conflict is detected.
    bind_hosts = [h.strip() for h in settings.server.host.split(",") if h.strip()]
    print("Starting private inference listener for the Molto dashboard")
    # uvicorn does not support "trace" — map to "debug" for its internal logging
    uvicorn_level = (
        "debug" if settings.server.log_level == "trace" else settings.server.log_level
    )
    # Only show access logs at trace level
    show_access_log = settings.server.log_level == "trace"
    uvicorn_config = uvicorn.Config(
        application,
        host=bind_hosts[0],
        port=settings.server.port,
        log_level=uvicorn_level,
        access_log=show_access_log,
        proxy_headers=True,
        forwarded_allow_ips="127.0.0.1,::1",
    )
    # The parent owns the private listener. Keep settings.host/port public:
    # authentication and advertised client URLs describe the external boundary.
    inherited_fd = int(os.environ["MOLTO_INTERNAL_FD"])
    serve_sockets = [socket.socket(fileno=inherited_fd)]

    try:
        # Import server and config after the port is known to be available.
        from molto_config.config import parse_size
        from molto_runtime.settings_adapter import (
            scheduler_config as make_scheduler_config,
        )

        model_dirs = settings.get_effective_model_dirs()
        print(f"Base path: {settings.base_path}")
        print(f"Model directories: {', '.join(str(d) for d in model_dirs)}")
        # State first: a bare tier line reads as "this is enforced" even when
        # the guard is off, and with it off the tier governs nothing.
        if settings.memory.prefill_memory_guard:
            print(f"Memory guard: on (tier: {settings.memory.memory_guard_tier})")
        else:
            print("Memory guard: off")

        # Store MCP config path for FastAPI startup
        # Priority: CLI arg > settings.json
        mcp_config = args.mcp_config or settings.mcp.config_path
        if mcp_config:
            print(f"MCP config: {mcp_config}")
            os.environ["MOLTO_MCP_CONFIG"] = mcp_config

        # Determine paged SSD cache directory
        # Priority: --no-cache > CLI arg > settings file
        if args.no_cache:
            paged_ssd_cache_dir = None
        elif args.paged_ssd_cache_dir:
            # CLI argument takes precedence
            paged_ssd_cache_dir = args.paged_ssd_cache_dir
        elif settings.cache.enabled:
            # Use settings file value (resolved path or default)
            paged_ssd_cache_dir = str(
                settings.cache.get_ssd_cache_dir(settings.base_path)
            )
        else:
            # Cache explicitly disabled in settings
            paged_ssd_cache_dir = None

        # Build scheduler config for BatchedEngine
        scheduler_config = make_scheduler_config(settings)
        # Set paged SSD cache options
        scheduler_config.paged_ssd_cache_dir = paged_ssd_cache_dir
        # Determine cache max size: CLI arg > settings (with auto resolution)
        if paged_ssd_cache_dir:
            if (
                args.paged_ssd_cache_max_size
                and args.paged_ssd_cache_max_size.lower() != "auto"
            ):
                # CLI argument specified explicitly
                cache_max_size_bytes = parse_size(args.paged_ssd_cache_max_size)
            else:
                # Resolve the initial automatic budget from disk space and existing cache.
                cache_max_size_bytes = settings.cache.get_ssd_cache_max_size_bytes(
                    settings.base_path
                )
            scheduler_config.paged_ssd_cache_max_size = cache_max_size_bytes
            scheduler_config.paged_ssd_cache_auto_size = (
                args.paged_ssd_cache_max_size or settings.cache.ssd_cache_max_size
            ).lower() == "auto"
        else:
            scheduler_config.paged_ssd_cache_max_size = 0
            cache_max_size_bytes = 0

        # Hot cache: CLI arg > settings
        if paged_ssd_cache_dir:
            if args.hot_cache_max_size:
                hot_cache_max_bytes = parse_size(args.hot_cache_max_size)
            else:
                hot_cache_max_bytes = settings.cache.get_hot_cache_max_size_bytes()
            scheduler_config.hot_cache_max_size = hot_cache_max_bytes
        else:
            scheduler_config.hot_cache_max_size = 0

        # Write-through: explicit CLI flag > settings file (already mapped by
        # settings.to_scheduler_config()).
        if getattr(args, "hot_cache_write_through", None) is not None:
            scheduler_config.hot_cache_write_through = bool(
                args.hot_cache_write_through
            )

        if args.no_cache:
            print(
                "Mode: Multi-model serving (no Molto cache, mlx-lm BatchGenerator only)"
            )
        elif paged_ssd_cache_dir:
            print("Mode: Multi-model serving (continuous batching + paged SSD cache)")
            # Format cache size for display
            cache_max_size_display = f"{cache_max_size_bytes / (1024**3):.1f}GB"
            if scheduler_config.paged_ssd_cache_auto_size:
                cache_max_size_display = f"auto, current limit {cache_max_size_display}"
            print(
                f"paged SSD cache: {paged_ssd_cache_dir} (max: {cache_max_size_display})"
            )
            if scheduler_config.hot_cache_max_size > 0:
                hot_display = f"{scheduler_config.hot_cache_max_size / (1024**3):.1f}GB"
                print(f"Hot cache: {hot_display} (in-memory)")
        else:
            print("Mode: Multi-model serving (continuous batching, no cache)")

        # Set MLX buffer cache limit high to prevent the allocator from
        # immediately releasing Metal buffers when the cache is full.
        # Without this, allocator::free() can call buf->release() while the
        # GPU is still using the buffer, causing kernel panics on M4.
        # With a large cache limit, freed buffers always stay in the pool
        # and are only released via mx.clear_cache() (which we protect
        # with mx.synchronize()). See issue #300.
        import mlx.core as mx

        total_mem = mx.device_info().get("memory_size", 0)
        if total_mem > 0:
            mx.set_cache_limit(total_mem)

        # Initialize server
        # pinned_models and default_model are managed through model settings.
        # Sampling parameters (max_tokens, temperature, etc.) are per-model settings
        application.state.controller.initialize(
            model_dirs=[str(d) for d in model_dirs],
            scheduler_config=scheduler_config,
            api_key=settings.auth.api_key,
            global_settings=settings,
        )

        print("Private inference server initialized")
        try:
            uvicorn.Server(uvicorn_config).run(sockets=serve_sockets)
        except KeyboardInterrupt:
            pass
    finally:
        # Uvicorn closes sockets during normal shutdown; this covers failures
        # after bind succeeds but before the server takes ownership.
        for sock in serve_sockets:
            sock.close()
