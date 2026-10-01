#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
CLI for oMLX.

Commands:
    omlx serve --model-dir /path/to/models    Start multi-model server

Usage:
    # Multi-model serving
    omlx serve --model-dir /path/to/models

    # With pinned models
    omlx serve --model-dir /path/to/models --pin llama-3b,qwen-7b
"""

import argparse
import math
import sys

from omlx_config._version import __version__ as __version__
from omlx_config.startup import _migrate_saved_network_auth


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than 0")
    return parsed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be an integer greater than 0")
    return parsed


def mflux_save_command(args) -> int:
    """Convert and save a supported diffusion checkpoint for oMLX discovery."""
    try:
        from omlx_runtime.mflux_conversion import convert_mflux_model

        result = convert_mflux_model(
            args.model,
            args.output,
            args.quantize,
            base_model=getattr(args, "base_model", None),
            pipeline=getattr(args, "pipeline", None),
            revision=getattr(args, "revision", None),
        )
    except (ImportError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Saved oMLX mflux model to {result['output']}")
    return 0


def diffusion_prepare_command(args) -> int:
    """Local-only calibrated diffusion preparation."""
    try:
        from omlx_runtime.diffusion.preparation import cli_command

        cli_command(args)
    except (ImportError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Saved {args.command} output to {args.output}")
    return 0


def serve_command(args):
    """Start the OpenAI-compatible multi-model server."""
    import os

    if "OMLX_INTERNAL_FD" not in os.environ:
        from omlx_config.settings import init_settings

        from omlx_cli.application import run_application

        settings = init_settings(base_path=args.base_path, cli_args=args)
        try:
            _migrate_saved_network_auth(settings, args)
        except (OSError, ValueError) as error:
            print(f"Configuration error: {error}")
            raise SystemExit(1) from error
        errors = settings.validate()
        if errors:
            for error in errors:
                print(f"Configuration error: {error}")
            raise SystemExit(1)
        raise SystemExit(run_application(settings, cli_args=args))

    from omlx_server.bootstrap import serve_private

    return serve_private(args)


def launch_command(args, extra_args: list[str] | None = None):
    """Launch an external tool integrated with oMLX.

    extra_args are unknown CLI tokens forwarded to the underlying tool binary
    (e.g. ``-r`` / ``--resume <id>`` for Claude Code).
    """
    import requests
    from omlx_config.settings import GlobalSettings

    from omlx_cli.integrations import (
        IntegrationContext,
        get_integration,
        list_integrations,
    )

    def _optional_str(value) -> str | None:
        return value if isinstance(value, str) and value else None

    tool_name = args.tool

    if tool_name == "list":
        print("Available integrations:")
        for integ in list_integrations():
            installed = "installed" if integ.is_installed() else "not installed"
            print(f"  {integ.name:12s} {integ.display_name} ({installed})")
        return

    integration = get_integration(tool_name)
    if integration is None:
        print(f"Unknown integration: {tool_name}")
        print("Available: " + ", ".join(i.name for i in list_integrations()))
        sys.exit(1)

    # Resolve host/port: CLI args > env vars > settings.json > defaults
    settings = GlobalSettings.load()
    host = args.host or settings.server.host
    port = args.port or settings.server.port

    # host may be a comma-separated list of bind addresses; pick the first one
    # for connecting. Wildcard addresses (0.0.0.0, ::) are valid bind targets
    # but not connectable — fall back to localhost in that case.
    first_bind = [h.strip() for h in host.split(",") if h.strip()][0] if host else ""
    connect_host = (
        "::1"
        if first_bind == "::"
        else first_bind
        if first_bind not in ("", "0.0.0.0")
        else "127.0.0.1"
    )

    # Check if oMLX server is running
    from omlx_cli.client import server_url

    base_url = server_url(connect_host, port)
    try:
        resp = requests.get(f"{base_url}/health", timeout=3)
        resp.raise_for_status()
    except Exception:
        print(f"oMLX server is not running at {base_url}")
        print("Start the server first: omlx serve")
        sys.exit(1)

    # Get API key: CLI args > settings.json > empty
    api_key = getattr(args, "api_key", None) or settings.auth.api_key or ""

    claude_settings = getattr(settings, "claude_code", None)
    cli_opus_model = _optional_str(getattr(args, "opus_model", None))
    cli_sonnet_model = _optional_str(getattr(args, "sonnet_model", None))
    cli_haiku_model = _optional_str(getattr(args, "haiku_model", None))
    settings_opus_model = _optional_str(getattr(claude_settings, "opus_model", None))
    settings_sonnet_model = _optional_str(
        getattr(claude_settings, "sonnet_model", None)
    )
    settings_haiku_model = _optional_str(getattr(claude_settings, "haiku_model", None))
    opus_model = cli_opus_model or settings_opus_model
    sonnet_model = cli_sonnet_model or settings_sonnet_model
    haiku_model = cli_haiku_model or settings_haiku_model

    # Build headers for authenticated requests
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Pre-fetch model status (context_window, max_tokens, model_type per model)
    models_status_map: dict[str, dict] = {}
    try:
        resp = requests.get(f"{base_url}/v1/models/status", headers=headers, timeout=5)
        if resp.ok:
            for m in resp.json().get("models", []):
                if m_id := m.get("id"):
                    models_status_map[m_id] = m
                if model_alias := m.get("model_alias"):
                    models_status_map[model_alias] = m
    except Exception:
        pass

    # Determine model. Explicit CLI tier flags bypass the picker; otherwise always
    # prompt interactively so the user's selection is honoured.
    model = args.model
    if not model and (cli_opus_model or cli_sonnet_model or cli_haiku_model):
        model = cli_sonnet_model or cli_opus_model or cli_haiku_model or ""
    elif not model:
        # Fetch available models from server
        try:
            resp = requests.get(f"{base_url}/v1/models", headers=headers, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            models = [
                m["id"]
                for m in data.get("data", [])
                if m.get("model_type") in ("llm", "vlm", None)
            ]
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in (401, 403):
                print("Model access denied. Check your API key.", file=sys.stderr)
            else:
                print(
                    "The server could not list models. Check its logs.", file=sys.stderr
                )
            sys.exit(1)
        except (requests.RequestException, ValueError, KeyError, TypeError):
            print(
                "Could not retrieve the model list. Check the server connection.",
                file=sys.stderr,
            )
            sys.exit(1)

        if not models:
            print("No models available. Load a model first.")
            sys.exit(1)

        if len(models) == 1:
            model = models[0]
            print(f"Using model: {model}")
        else:
            models_info_list = [
                {"id": m_id, **models_status_map.get(m_id, {})} for m_id in models
            ]
            model = integration.select_model(models_info_list, integration.display_name)

    # Check if tool is installed
    if not integration.is_installed():
        print(f"{integration.display_name} is not installed.")
        print(f"Install: {integration.install_hint}")
        sys.exit(1)

    # Tier precedence: explicit tier flag > saved claude_code tier setting >
    # the model picked (or auto-selected) above. The picker only chooses the
    # default model; tiers configured on the Claude Code settings page keep
    # their role, otherwise the three persisted selections would be silently
    # replaced by one model on every interactive launch (#3543). Roles without
    # a saved model fall back to the picked model in the integration.

    # Enforce Claude Code's model requirements after all interactive,
    # automatic, and explicit model paths have resolved. The picker also marks
    # disabled models, but this central check prevents --model and tier flags
    # from bypassing the same restriction.
    if tool_name == "claude":
        from omlx_cli.integrations.claude import claude_code_model_disabled_reason

        models_to_validate = [
            ("", model),
            ("Opus tier ", opus_model),
            ("Sonnet tier ", sonnet_model),
            ("Haiku tier ", haiku_model),
        ]
        validated_models: set[str] = set()
        for role, model_id in models_to_validate:
            if not model_id or model_id in validated_models:
                continue
            validated_models.add(model_id)
            disabled_reason = claude_code_model_disabled_reason(
                {"id": model_id, **models_status_map.get(model_id, {})}
            )
            if disabled_reason:
                print(
                    f"Cannot launch {integration.display_name} with "
                    f"{role}model '{model_id}'."
                )
                print(disabled_reason)
                print(
                    "Choose a model with at least 48K context or increase its "
                    "configured max_context_window."
                )
                sys.exit(1)

    # Resolve model limits from pre-fetched status
    model_info = models_status_map.get(model, {})
    context_window = model_info.get("max_context_window")
    if tool_name == "claude":
        # Claude's context overrides are process-wide, including tier switches
        # and subagents. Do not advertise more than any configured model allows.
        context_windows = [
            info["max_context_window"]
            for model_id in (model, opus_model, sonnet_model, haiku_model)
            if (info := models_status_map.get(model_id, {}))
            and isinstance(info.get("max_context_window"), int)
            and info["max_context_window"] > 0
        ]
        context_window = min(context_windows) if context_windows else None
    ctx = IntegrationContext(
        host=connect_host,
        port=port,
        api_key=api_key,
        model=model,
        opus_model=opus_model if tool_name == "claude" else None,
        sonnet_model=sonnet_model if tool_name == "claude" else None,
        haiku_model=haiku_model if tool_name == "claude" else None,
        context_window=context_window,
        max_tokens=model_info.get("max_tokens"),
        model_type=model_info.get("model_type"),
        reasoning=model_info.get("enable_thinking"),
        tools_profile=getattr(args, "tools_profile", "coding"),
        extra_args=tuple(extra_args or ()),
        cross_session=getattr(args, "cross_session", False),
    )

    # Launch
    print(f"Launching {integration.display_name} with model {model}...")
    integration.launch(ctx)


def lifecycle_command(args) -> int:
    """Manage the bundled application without importing the inference runtime."""
    from omlx_cli.cli_lifecycle import run
    from omlx_cli.cli_output import Output

    output = Output(args)
    with output.watch(f"{args.command.capitalize()} oMLX server…"):
        result = run(args)
    output.emit(result, title="oMLX server")
    return 0


def init_command(args) -> int:
    """Use the existing guarded setup endpoint for a fresh local server."""
    import getpass
    import os

    from omlx_config.auth import validate_api_key

    from omlx_cli.cli_output import Output
    from omlx_cli.client import (
        CLIError,
        ManagementClient,
        loopback_origin,
        resolve_connection,
    )

    output = Output(args)
    url, key, _ = resolve_connection(args, require_key=False)
    if not loopback_origin(url):
        raise CLIError(
            "Initial setup requires a loopback URL and a locally bound server.", 2
        )
    client = ManagementClient(url, "", getattr(args, "timeout", 30))
    try:
        status = client.public_request("GET", "/api/setup")
        if not status.get("allowed"):
            raise CLIError(status.get("reason") or "Initial setup is unavailable.", 3)
        if not status.get("setup_required"):
            output.emit(
                {
                    "configured": True,
                    "message": "A main key already exists. Use it for management commands.",
                }
            )
            return 0
        explicit = (
            getattr(args, "api_key", None)
            or getattr(args, "api_key_file", None)
            or os.environ.get("OMLX_API_KEY")
        )
        if not explicit:
            if not sys.stdin.isatty() or output.json:
                raise CLIError(
                    "Provide the new key through OMLX_API_KEY or --api-key-file for unattended setup.",
                    2,
                )
            try:
                key = getpass.getpass("New main API key: ")
                confirmation = getpass.getpass("Confirm main API key: ")
            except EOFError as exc:
                raise CLIError("Operation cancelled.", 130) from exc
            if key != confirmation:
                raise CLIError("The keys do not match.", 2)
        valid, message = validate_api_key(key)
        if not valid or len(key) > 4096:
            raise CLIError(message or "The key must be at most 4096 characters.", 2)
        result = client.public_request(
            "POST",
            "/api/setup",
            json={"key": key, "confirmation": key},
            headers={"Origin": url},
        )
        output.emit(result)
        return 0
    finally:
        client.close()


def cluster_command(args) -> int:
    """Run cluster diagnostics, collective checks, and shard planning."""
    import json

    action = getattr(args, "cluster_action", None)
    if action == "status":
        from omlx_runtime.cluster.probe import (
            collect_cluster_status,
            format_cluster_status,
        )

        try:
            status = collect_cluster_status(route_to=args.route_to)
        except ValueError as exc:
            print(f"Cluster status error: {exc}", file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(status.to_dict(), indent=2, sort_keys=True))
        else:
            print(format_cluster_status(status))
        return 0

    if action == "worker-smoke":
        from omlx_runtime.cluster.supervisor import run_worker_smoke

        try:
            result = run_worker_smoke(timeout=args.timeout)
        except (OSError, RuntimeError, TimeoutError) as exc:
            print(f"Cluster worker smoke failed: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print("oMLX cluster worker smoke passed")
            print(f"Worker PID:  {result['worker_pid']}")
            print(f"Protocol:    {result['protocol_version']}")
            print(f"Round trip:  {result['elapsed_seconds']:.3f}s")
        return 0

    if action == "collective-smoke":
        from omlx_runtime.cluster.collective import (
            CollectiveSmokeError,
            run_local_collective_smoke,
        )

        try:
            result = run_local_collective_smoke(timeout=args.timeout)
        except (CollectiveSmokeError, OSError, RuntimeError, ValueError) as exc:
            print(f"Cluster collective smoke failed: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print("oMLX local MLX collective smoke passed")
            print(f"Backend:     {result['backend']} (loopback only)")
            print(f"Ranks:       {result['rank_count']}")
            print(f"All-sum:     {result['expected_sum']}")
            print(f"MLX:         {result['mlx_version']}")
            print(f"Elapsed:     {result['elapsed_seconds']:.3f}s")
        return 0

    if action == "pipeline-smoke":
        from omlx_runtime.cluster.collective import (
            CollectiveSmokeError,
            run_local_pipeline_smoke,
        )

        try:
            result = run_local_pipeline_smoke(timeout=args.timeout)
        except (CollectiveSmokeError, OSError, RuntimeError, ValueError) as exc:
            print(f"Cluster pipeline smoke failed: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print("oMLX unequal Nemotron-H pipeline smoke passed")
            print(f"Backend:     {result['backend']} (loopback only)")
            print(f"Ranks:       {result['rank_count']}")
            print(f"Checksum:    {result['ranks'][0]['checksum']}")
            print(f"Elapsed:     {result['elapsed_seconds']:.3f}s")
        return 0

    if action == "plan":
        import socket

        from omlx_config.config import parse_size
        from omlx_runtime.cluster.planner import (
            NodeBudget,
            PlanningError,
            format_shard_plan,
            locate_model_layout,
            plan_unequal_pipeline,
            synthetic_model_layout,
        )
        from omlx_runtime.utils import hardware

        def parse_cluster_size(value: str) -> int:
            normalized = (
                value.strip()
                .upper()
                .replace("KIB", "KB")
                .replace("MIB", "MB")
                .replace("GIB", "GB")
                .replace("TIB", "TB")
            )
            size = parse_size(normalized)
            if size < 0:
                raise ValueError("sizes must be non-negative")
            return size

        try:
            reserve_bytes = parse_cluster_size(args.reserve)
            nodes = []
            for rank, definition in enumerate(args.node or []):
                node_id, separator, raw_size = definition.rpartition("=")
                if not separator or not node_id.strip() or not raw_size.strip():
                    raise ValueError(
                        "--node must use NAME=SIZE (for example studio=256GB)"
                    )
                nodes.append(
                    NodeBudget(
                        node_id=node_id.strip(),
                        capacity_bytes=parse_cluster_size(raw_size),
                        reserve_bytes=reserve_bytes,
                        rank=rank,
                    )
                )
            if not nodes:
                detected = hardware.detect_hardware()
                nodes.append(
                    NodeBudget(
                        node_id=socket.gethostname(),
                        capacity_bytes=detected.max_working_set_bytes,
                        reserve_bytes=reserve_bytes,
                        rank=0,
                    )
                )

            holder = None
            if args.model:
                # The Mac being planned for is often the one holding a single
                # stage, so ask each peer rather than assume this node can
                # read the whole model.
                holder = locate_model_layout(args.model, args.peer or [])
                model = holder.layout
            else:
                model = synthetic_model_layout(
                    total_weight_bytes=parse_cluster_size(args.model_size),
                    layer_count=args.layers,
                )
            plan = plan_unequal_pipeline(model, nodes)
        except (OSError, PlanningError, ValueError) as exc:
            print(f"Cluster planning failed: {exc}", file=sys.stderr)
            return 2

        if args.json:
            print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        else:
            print(format_shard_plan(plan))
            if holder is not None and not holder.is_local:
                print(f"Measured:    {holder.node} (the node holding the model)")
        return 0

    print(
        "Unknown cluster action. Available: status, worker-smoke, "
        "collective-smoke, pipeline-smoke, plan",
        file=sys.stderr,
    )
    return 2


def main():
    from omlx_cli.cli_output import CLIParser, Output
    from omlx_cli.client import CLIError
    from omlx_cli.connection_options import add_connection_options

    parser = CLIParser(
        description="oMLX: run and manage your local inference application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  omlx start
  omlx status
  omlx models list
  omlx models load qwen3.5
  omlx keys create --name coding
  omlx --url https://your-server.example models list --json
  omlx serve --model-dir ~/.omlx/models --port 8000
  omlx launch codex --model qwen3.5
        """,
    )
    add_connection_options(parser)
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print the oMLX version and exit",
    )
    subparsers = parser.add_subparsers(
        dest="command", title="Commands", metavar="COMMAND"
    )

    for name, help_text in (
        ("start", "Start the application in the background"),
        ("stop", "Stop the locally managed application"),
        ("restart", "Restart the locally managed application"),
        ("status", "Show local lifecycle and public server health"),
    ):
        lifecycle_parser = subparsers.add_parser(
            name,
            help=help_text,
            description=help_text,
        )
        add_connection_options(lifecycle_parser)
        if name in {"start", "restart"}:
            lifecycle_parser.add_argument(
                "--no-wait",
                action="store_true",
                help="Return after spawning; use status to check readiness",
            )
            lifecycle_parser.add_argument("--host", help="Public bind address")
            lifecycle_parser.add_argument(
                "--port", type=_positive_int, help="Public port"
            )
            lifecycle_parser.add_argument("--model-dir", help="Model directory")
            lifecycle_parser.add_argument(
                "--dashboard-dev",
                action="store_true",
                help="Use the source dashboard development server",
            )

    from omlx_cli.cli_commands import register_commands

    register_commands(subparsers)
    init_parser = subparsers.add_parser(
        "init", help="Create a main API key on a new local server"
    )
    add_connection_options(init_parser)
    open_parser = subparsers.add_parser(
        "open", help="Open the dashboard in your browser"
    )
    add_connection_options(open_parser)

    # Serve command (multi-model)
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start multi-model OpenAI-compatible server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Start a multi-model inference server with LRU-based memory management.

Models are discovered from subdirectories of --model-dir. Each subdirectory
should contain a valid model with config.json and *.safetensors files.

Example directory structure:
  /path/to/models/
  ├── llama-3b/           → model_id: "llama-3b"
  │   ├── config.json
  │   └── model.safetensors
  ├── qwen-7b/            → model_id: "qwen-7b"
  └── mistral-7b/         → model_id: "mistral-7b"
""",
    )
    serve_parser.add_argument(
        "--no-color",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Disable startup banner colors",
    )

    # Required arguments
    serve_parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory containing model subdirectories (default: ~/.omlx/models)",
    )
    # Server options
    serve_parser.add_argument(
        "--host", type=str, default=None, help="Host to bind (default: 127.0.0.1)"
    )
    serve_parser.add_argument(
        "--port", type=int, default=None, help="Port to bind (default: 8000)"
    )
    serve_parser.add_argument(
        "--log-level",
        type=str,
        choices=["trace", "debug", "info", "warning", "error"],
        default=None,
        help="Log level (default: info). trace includes full message content",
    )
    serve_parser.add_argument(
        "--sse-keepalive-mode",
        type=str,
        choices=["chunk", "comment", "off"],
        default=None,
        help="SSE keepalive emission mode (default: chunk). 'chunk' emits "
        "protocol-aware no-op events compatible with strict clients like "
        "OpenClaw / WorkBuddy; 'comment' emits the legacy ': keep-alive' SSE "
        "comment; 'off' disables keepalive entirely",
    )
    serve_parser.add_argument(
        "--max-audio-upload-size",
        type=str,
        default=None,
        help="Maximum audio upload size for /v1/audio/transcriptions and "
        "/v1/audio/process (e.g. '100MB', '500MB'). Overrides the value "
        "in settings.json (built-in default: 100MB). Uploads are buffered "
        "in memory, so this is also a per-request RAM cap",
    )
    serve_parser.add_argument(
        "--max-image-upload-size",
        type=str,
        default=None,
        help="Maximum image payload size for VLM inputs (e.g. '50MB', '100MB'). "
        "Overrides the value in settings.json (built-in default: 50MB).",
    )
    serve_parser.add_argument(
        "--max-image-side-length",
        type=int,
        default=None,
        help="Maximum side length in pixels for VLM input images. Images exceeding "
        "this limit are downscaled preserving aspect ratio (built-in default: 2048, "
        "0 to disable).",
    )

    # Scheduler options (for BatchedEngine)
    serve_parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=None,
        help="Max requests processed simultaneously. Higher values increase throughput but use more memory. (default: 8)",
    )
    serve_parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=None,
        help="Max embedding inputs processed in one forward pass. Higher values increase throughput but use more memory. (default: 32)",
    )

    # Memory guard options
    serve_parser.add_argument(
        "--memory-guard",
        type=str,
        choices=["off", "safe", "balanced", "aggressive"],
        default=None,
        help="Memory guard tier, or 'off' to disable the guard. The tier sets how much memory stays free for other apps: safe keeps about 20%% of RAM (6-16 GB), balanced about 8%% (3-8 GB), aggressive 2%% (1.5-4 GB) and may compress other apps' memory. Passing a tier also turns the guard on. (default: balanced)",
    )
    serve_parser.add_argument(
        "--memory-guard-gb",
        type=_positive_float,
        default=None,
        help="Custom memory guard ceiling in GB. Sets memory guard tier to custom and turns the guard on.",
    )

    # paged SSD cache options
    serve_parser.add_argument(
        "--paged-ssd-cache-dir",
        type=str,
        default=None,
        help="Directory for paged SSD cache storage (enables oMLX prefix cache)",
    )
    serve_parser.add_argument(
        "--paged-ssd-cache-max-size",
        type=str,
        default=None,
        help="Maximum paged SSD cache size (e.g., '100GB', '50GB'). Default: 100GB",
    )
    serve_parser.add_argument(
        "--hot-cache-max-size",
        type=str,
        default=None,
        help="Maximum in-memory hot cache size (e.g., '8GB', '4GB'). Default: 0 (disabled)",
    )
    serve_parser.add_argument(
        "--hot-cache-write-through",
        action="store_true",
        default=None,
        help="Persist every hot-cache block to SSD immediately (write-through). "
        "Keeps RAM-speed resume while retaining SSD durability for all sessions.",
    )
    serve_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable oMLX paged SSD cache. mlx-lm BatchGenerator still manages KV states internally.",
    )
    serve_parser.add_argument(
        "--initial-cache-blocks",
        type=int,
        default=None,
        help="Number of cache blocks to pre-allocate at startup (default: 256). "
        "Higher values reduce dynamic allocation overhead for large contexts.",
    )

    # MCP options
    serve_parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP configuration file (JSON/YAML) for tool integration",
    )

    # HuggingFace options
    serve_parser.add_argument(
        "--hf-endpoint",
        type=str,
        default=None,
        help="Custom HuggingFace Hub endpoint URL (e.g., https://hf-mirror.com)",
    )
    serve_parser.add_argument(
        "--hf-cache",
        dest="hf_cache_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Discover models from the standard HuggingFace Hub local cache (default: enabled)",
    )

    # ModelScope options
    serve_parser.add_argument(
        "--ms-endpoint",
        type=str,
        default=None,
        help="Custom ModelScope Hub endpoint URL",
    )

    # Network options
    serve_parser.add_argument(
        "--http-proxy",
        type=str,
        default=None,
        help="HTTP proxy URL (e.g., http://proxy.company.com:8080)",
    )
    serve_parser.add_argument(
        "--https-proxy",
        type=str,
        default=None,
        help="HTTPS proxy URL (e.g., http://proxy.company.com:8080)",
    )
    serve_parser.add_argument(
        "--no-proxy",
        type=str,
        default=None,
        help="Comma-separated hosts/IPs to bypass proxy (e.g., localhost,127.0.0.1)",
    )
    serve_parser.add_argument(
        "--ca-bundle",
        type=str,
        default=None,
        help="Path to CA bundle PEM file for TLS interception environments",
    )

    serve_parser.add_argument(
        "--dashboard-dev",
        action="store_true",
        help="Run dashboard source with Vite on the configured public port (source checkout only)",
    )

    # Base path and auth
    serve_parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="Base directory for oMLX data (default: ~/.omlx)",
    )
    serve_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (required for non-loopback binds)",
    )

    from omlx_runtime.diffusion.registry import PIPELINES

    mflux_save_parser = subparsers.add_parser(
        "mflux-save",
        help="Convert and quantize a supported diffusion model for oMLX",
        description=(
            "Load a supported image model through mflux, optionally quantize it, and "
            "save a local checkpoint with the metadata oMLX uses for discovery."
        ),
    )
    mflux_save_parser.add_argument(
        "--model",
        default="z-image-turbo",
        help=(
            "mflux model alias, Hugging Face repo, or local checkpoint "
            "(default: z-image-turbo)"
        ),
    )
    mflux_save_parser.add_argument(
        "--base-model",
        help="mflux registry key or alias identifying a custom checkpoint's base model",
    )
    mflux_save_parser.add_argument(
        "--revision",
        help="Hugging Face branch, tag, or commit to acquire",
    )
    mflux_save_parser.add_argument(
        "--pipeline",
        help="Pipeline ID; defaults to the base model pipeline. Available: "
        + ", ".join(PIPELINES),
    )
    mflux_save_parser.add_argument(
        "--output",
        required=True,
        help="Empty output directory to create",
    )
    mflux_save_parser.add_argument(
        "--quantize",
        type=int,
        choices=[3, 4, 5, 6, 8],
        default=8,
        help="Quantization bits for the saved checkpoint (default: 8)",
    )
    mflux_save_parser.add_argument(
        "--no-quantize",
        dest="quantize",
        action="store_const",
        const=None,
        help="Preserve source precision without applying new MLX quantization",
    )

    calibration_parser = subparsers.add_parser(
        "diffusion-calibrate",
        help="Collect transformer activation energy from a local checkpoint",
    )
    calibration_parser.add_argument(
        "--model",
        required=True,
        help="Complete local FLUX.2 Klein 4B or Qwen-Image-2.1 directory; no downloads",
    )
    calibration_parser.add_argument(
        "--output", required=True, help="New calibration JSON file"
    )
    calibration_parser.add_argument(
        "--prompt",
        required=True,
        action="append",
        help="Calibration prompt; repeat for more samples",
    )
    calibration_parser.add_argument("--width", type=_positive_int, default=256)
    calibration_parser.add_argument("--height", type=_positive_int, default=256)
    calibration_parser.add_argument("--steps", type=_positive_int)
    calibration_parser.add_argument("--seed", type=int, default=17)
    calibration_parser.add_argument("--guidance", type=float)
    calibration_parser.add_argument("--negative-prompt")
    calibration_parser.add_argument("--max-rows", type=_positive_int, default=256)

    quantization_parser = subparsers.add_parser(
        "diffusion-quantize",
        help="Quantize a local float transformer using diffusion calibration",
    )
    quantization_parser.add_argument(
        "--model",
        required=True,
        help="Complete local floating-point checkpoint; no downloads",
    )
    quantization_parser.add_argument(
        "--calibration", required=True, help="diffusion-calibrate JSON report"
    )
    quantization_parser.add_argument(
        "--output", required=True, help="New or empty checkpoint directory"
    )
    quantization_parser.add_argument(
        "--bits", type=int, choices=[3, 4, 5, 6, 8], default=4
    )
    quantization_parser.add_argument(
        "--group-size", type=int, choices=[32, 64, 128], default=64
    )
    budget = quantization_parser.add_mutually_exclusive_group()
    budget.add_argument(
        "--budget-bytes",
        type=_positive_int,
        help="Transformer parameter byte ceiling, including retained float weights",
    )
    budget.add_argument(
        "--budget-ratio",
        type=_positive_float,
        default=1.10,
        help="Budget relative to base-bit transformer allocation (default: 1.10)",
    )
    quantization_parser.add_argument(
        "--protect",
        action="append",
        default=[],
        help="Transformer-relative linear module glob to retain in float; repeatable",
    )

    # Launch command
    launch_parser = subparsers.add_parser(
        "launch",
        help="Launch an external tool with oMLX integration",
        description=(
            "Configure and launch external coding tools (Claude Code, Copilot, "
            "Codex, Codex App, OpenCode, OpenClaw, Hermes Agent, Pi) to use "
            "the running oMLX server."
        ),
    )
    launch_parser.add_argument(
        "tool",
        type=str,
        help=(
            "Tool to launch: claude, copilot, codex, codex_app, opencode, "
            "openclaw, hermes, pi, or 'list' to show available"
        ),
    )
    launch_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (interactive selection if not specified)",
    )
    launch_parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="oMLX server host (default: from settings or 127.0.0.1)",
    )
    launch_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="oMLX server port (default: from settings or 8000)",
    )
    launch_parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for oMLX server authentication",
    )
    launch_parser.add_argument(
        "--tools-profile",
        type=str,
        default="coding",
        choices=["minimal", "coding", "messaging", "full"],
        help="OpenClaw tools profile (default: coding)",
    )
    launch_parser.add_argument(
        "--opus",
        dest="opus_model",
        type=str,
        default=None,
        help="Claude Code Opus tier model (Claude integration only)",
    )
    launch_parser.add_argument(
        "--sonnet",
        dest="sonnet_model",
        type=str,
        default=None,
        help="Claude Code Sonnet tier model (Claude integration only)",
    )
    launch_parser.add_argument(
        "--haiku",
        dest="haiku_model",
        type=str,
        default=None,
        help="Claude Code Haiku tier model (Claude integration only)",
    )
    launch_parser.add_argument(
        "--cross-session",
        action="store_true",
        default=False,
        help=(
            "Allow the launched session to be reachable via Claude Code's "
            "cross-session messaging (ListAgents/SendMessage). This requires "
            "enabling telemetry and feature-flag traffic to Anthropic that is "
            "otherwise kept disabled by default (Claude integration only)."
        ),
    )

    # Cluster diagnostics and planning for the first implementation slice.
    cluster_parser = subparsers.add_parser(
        "cluster",
        help="Inspect distributed-node readiness and exercise a local worker",
        description=(
            "Distributed-cluster diagnostics and unequal-memory planning. "
            "This command does not configure interfaces or initialize JACCL."
        ),
    )
    cluster_subparsers = cluster_parser.add_subparsers(
        dest="cluster_action",
        required=True,
        help="Cluster diagnostic command",
    )
    cluster_status_parser = cluster_subparsers.add_parser(
        "status",
        help="Report local memory, runtime, RDMA, and Thunderbolt readiness",
    )
    cluster_status_parser.add_argument(
        "--route-to",
        metavar="IP",
        default=None,
        help="Also inspect the active route to an IPv4 or IPv6 peer address",
    )
    cluster_status_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )
    cluster_smoke_parser = cluster_subparsers.add_parser(
        "worker-smoke",
        help="Run a real isolated worker ready/ping/shutdown round trip",
    )
    cluster_smoke_parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=5.0,
        help="Per-operation worker deadline in seconds (default: 5)",
    )
    cluster_smoke_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )
    cluster_collective_parser = cluster_subparsers.add_parser(
        "collective-smoke",
        help="Run two local MLX ranks and verify a ring all-sum",
    )
    cluster_collective_parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=20.0,
        help="Overall collective deadline in seconds (default: 20)",
    )
    cluster_collective_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )
    cluster_pipeline_parser = cluster_subparsers.add_parser(
        "pipeline-smoke",
        help="Run an unequal two-rank hybrid Nemotron-H graph",
    )
    cluster_pipeline_parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=30.0,
        help="Overall pipeline deadline in seconds (default: 30)",
    )
    cluster_pipeline_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )
    cluster_plan_parser = cluster_subparsers.add_parser(
        "plan",
        help="Plan contiguous layers across unequal node memory budgets",
    )
    cluster_plan_source = cluster_plan_parser.add_mutually_exclusive_group(
        required=True
    )
    cluster_plan_source.add_argument(
        "--model",
        metavar="PATH",
        help="Inspect safetensors headers from a downloaded model directory",
    )
    cluster_plan_source.add_argument(
        "--model-size",
        metavar="SIZE",
        help="Plan an estimated model before download (for example 300GB)",
    )
    cluster_plan_parser.add_argument(
        "--layers",
        type=_positive_int,
        default=80,
        help="Layer count used with --model-size (default: 80)",
    )
    cluster_plan_parser.add_argument(
        "--node",
        action="append",
        metavar="NAME=SIZE",
        help=(
            "Node memory budget in rank order; repeat for each node. "
            "Defaults to this Mac's recommended working set."
        ),
    )
    cluster_plan_parser.add_argument(
        "--reserve",
        default="0",
        metavar="SIZE",
        help="Memory to reserve on every node for KV/activations (default: 0)",
    )
    cluster_plan_parser.add_argument(
        "--peer",
        action="append",
        metavar="SSH_HOST",
        help=(
            "SSH host that may hold the model; repeat for each. Used with "
            "--model when this Mac holds only its own stage: the first peer "
            "that can read a complete model is the one that measures it."
        ),
    )
    cluster_plan_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )

    # Split launch's forwarding separator before argparse. parse_known_args()
    # inconsistently retains it when known options precede it, and stripping it
    # afterward cannot distinguish it from a separator intended for the tool.
    argv = sys.argv[1:]
    if argv[:1] == ["launch"] and "--" in argv[2:]:
        separator_index = argv.index("--", 2)
        args, extra_args = parser.parse_known_args(argv[:separator_index])
        extra_args.extend(argv[separator_index + 1 :])
    else:
        args, extra_args = parser.parse_known_args(argv)

    try:
        if args.command == "launch":
            launch_command(args, extra_args=extra_args)
        else:
            if extra_args:
                parser.error(f"unrecognized arguments: {' '.join(extra_args)}")
            if args.command == "serve":
                if (
                    getattr(args, "memory_guard", None) == "off"
                    and getattr(args, "memory_guard_gb", None) is not None
                ):
                    parser.error(
                        "--memory-guard off cannot be combined with "
                        "--memory-guard-gb (a custom ceiling needs the guard on)"
                    )
                serve_command(args)
            elif getattr(args, "management_command", False):
                from omlx_cli.cli_commands import run

                sys.exit(run(args))
            elif args.command in {"start", "stop", "restart", "status"}:
                sys.exit(lifecycle_command(args))
            elif args.command == "init":
                sys.exit(init_command(args))
            elif args.command == "open":
                import webbrowser

                from omlx_cli.client import resolve_connection

                url, _, _ = resolve_connection(args, require_key=False)
                if not webbrowser.open(url):
                    raise CLIError(f"Could not open a browser. Open {url} manually.")
                Output(args).emit({"url": url, "opened": True})
            elif args.command == "mflux-save":
                sys.exit(mflux_save_command(args))
            elif args.command in {"diffusion-calibrate", "diffusion-quantize"}:
                sys.exit(diffusion_prepare_command(args))
            elif args.command == "cluster":
                sys.exit(cluster_command(args))
            else:
                parser.print_help()
                sys.exit(1)
    except CLIError as error:
        Output(args).error(error)
        sys.exit(error.exit_code)
    except KeyboardInterrupt:
        Output(args).error(CLIError("Operation cancelled.", 130))
        sys.exit(130)
    except BrokenPipeError:
        sys.exit(0)


if __name__ == "__main__":
    main()
