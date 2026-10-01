"""Configuration precedence and saved authentication migrations at startup."""


def _has_cli_overrides(args) -> bool:
    """Check if CLI args contain non-default values that should be saved.

    All argparse defaults are None, so `is not None` means the user
    explicitly passed the flag on the command line.
    """
    persisted_fields = (
        "model_dir",
        "port",
        "host",
        "log_level",
        "sse_keepalive_mode",
        "max_audio_upload_size",
        "max_image_upload_size",
        "max_image_side_length",
        "max_concurrent_requests",
        "embedding_batch_size",
        "memory_guard",
        "memory_guard_gb",
        "paged_ssd_cache_dir",
        "paged_ssd_cache_max_size",
        "hot_cache_max_size",
        "hot_cache_write_through",
        "initial_cache_blocks",
        "mcp_config",
        "hf_endpoint",
        "hf_cache_enabled",
        "ms_endpoint",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "ca_bundle",
    )
    if any(getattr(args, field, None) is not None for field in persisted_fields):
        return True

    # --no-cache is the only persistable boolean flag with a False default.
    return bool(getattr(args, "no_cache", False))


def _migrate_saved_network_auth(settings, args) -> None:
    import json
    import os
    import tempfile
    from pathlib import Path

    from omlx_config.utils.network import is_valid_bind_host, network_auth_error

    if getattr(args, "host", None) is not None or os.environ.get("OMLX_HOST"):
        return
    path = settings.base_path / "settings.json"
    if not path.exists():
        return
    host = settings.server.host
    if not isinstance(host, str) or not all(
        is_valid_bind_host(part.strip()) for part in host.split(",")
    ):
        return
    if not network_auth_error(
        host, settings.auth.api_key, settings.auth.skip_api_key_verification
    ):
        return

    settings.server.host = "127.0.0.1"
    if settings.validate():
        settings.server.host = host
        return

    message = (
        f"The saved server address ({host}) was changed to 127.0.0.1 because "
        "API key authentication is required for access from other devices. "
        "The server is now limited to this Mac. Your other settings and models "
        "have been preserved. To allow access from other devices, set an API "
        "key with --api-key or OMLX_API_KEY, keep API-key verification enabled "
        "in settings.json, then restore server.host."
    )
    notice_path = os.environ.get("OMLX_STARTUP_NOTICE_PATH")
    if not notice_path:
        warning = message.replace("was changed", "will be changed").replace(
            "is now limited", "will be limited"
        )
        print(f"Warning: {warning}", flush=True)
        try:
            input("Press Enter to continue, or Ctrl+C to cancel. ")
        except (EOFError, KeyboardInterrupt):
            print("\nStartup canceled. Settings have not been changed.", flush=True)
            raise SystemExit(1) from None

    # Preserve unknown settings and avoid persisting environment overrides.
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("server", {})["host"] = settings.server.host
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            json.dump(data, output, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    if notice_path:
        print(f"Warning: {message}", flush=True)

        notice = Path(notice_path)
        temporary = notice.with_suffix(".tmp")
        try:
            temporary.write_text(message, encoding="utf-8")
            os.replace(temporary, notice)
        finally:
            temporary.unlink(missing_ok=True)
