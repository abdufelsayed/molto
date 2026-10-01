"""Management command grammar and thin, authenticated API dispatch."""

from __future__ import annotations

import json
import math
import secrets
import time
from pathlib import Path
from urllib.parse import quote, urlsplit

from omlx_cli.cli_output import Output
from omlx_cli.client import CLIError, ManagementClient, read_key_file, validate_path
from omlx_cli.connection_options import add_connection_options


def _leaf(subparsers, name, help_text):
    parser = subparsers.add_parser(name, help=help_text, description=help_text)
    parser.set_defaults(management_command=True)
    add_connection_options(parser)
    return parser


def _group(subparsers, name, dest, help_text):
    parser = subparsers.add_parser(name, help=help_text)
    add_connection_options(parser)
    return parser.add_subparsers(dest=dest, required=True)


def _assignments(parser):
    parser.add_argument(
        "assignments",
        nargs="+",
        metavar="FIELD=JSON",
        help='Typed values, e.g. temperature=0.7 or model_alias="my-model"',
    )


def _poll_options(parser):
    parser.add_argument(
        "--interval", type=float, default=1, help="Polling interval in seconds"
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=300,
        help="Maximum watch duration in seconds",
    )


def register_commands(subparsers):
    models = _group(subparsers, "models", "model_action", "Manage the model library")
    for action in ("list", "refresh"):
        _leaf(models, action, f"{action.capitalize()} the model inventory")
    for action in ("show", "load", "unload", "move", "remove"):
        parser = _leaf(models, action, f"{action.capitalize()} a model")
        parser.add_argument("model_id")
        if action in {"move", "remove"}:
            parser.add_argument(
                "--drain",
                action="store_true",
                help="Wait for active model requests to drain",
            )
        if action == "move":
            parser.add_argument(
                "--root-id", required=True, help="Destination storage root ID"
            )
    parser = _leaf(models, "download", "Download a model from a provider")
    parser.add_argument("repo_id")
    parser.add_argument("--provider", choices=("hf", "ms"), default="hf")
    parser.add_argument("--token-file", help="Provider credential file")
    settings = _group(
        models, "settings", "model_settings_action", "Read or change model settings"
    )
    for action in ("get", "set", "reset"):
        parser = _leaf(settings, action, f"{action.capitalize()} model settings")
        parser.add_argument("model_id")
        if action == "set":
            _assignments(parser)
    profiles = _group(models, "profiles", "profile_action", "Manage per-model profiles")
    for action in ("list", "create", "apply", "delete"):
        parser = _leaf(profiles, action, f"{action.capitalize()} a model profile")
        parser.add_argument("model_id")
        if action != "list":
            parser.add_argument("name")
        if action == "create":
            parser.add_argument("--settings", default="{}", metavar="JSON_OR_@FILE")
            parser.add_argument("--description")
            parser.add_argument("--display-name", default="")
            parser.add_argument("--expose-as-model", action="store_true")
    keys = _group(subparsers, "keys", "key_action", "Manage API credentials")
    for action in ("list", "create", "edit", "revoke", "rotate"):
        parser = _leaf(keys, action, f"{action.capitalize()} API keys")
        if action == "list":
            parser.add_argument(
                "--reveal",
                action="store_true",
                help="Explicitly show credential values",
            )
        if action in {"edit", "revoke"}:
            parser.add_argument("key_id")
        if action in {"create", "edit"}:
            parser.add_argument("--name", default="" if action == "create" else None)
            parser.add_argument("--key-file", help="Use a replacement key from a file")
        if action == "rotate":
            target = parser.add_mutually_exclusive_group(required=True)
            target.add_argument(
                "--main",
                action="store_true",
                help="Rotate the main management credential",
            )
            target.add_argument("--key-id", help="Rotate this subkey")
            parser.add_argument(
                "--key-file",
                help="Read replacement credential instead of generating one",
            )
    settings = _group(
        subparsers, "settings", "settings_action", "Manage persisted server settings"
    )
    for action in ("get", "set", "defaults"):
        parser = _leaf(settings, action, f"{action.capitalize()} server settings")
        if action == "set":
            _assignments(parser)
        parser.add_argument(
            "--reveal", action="store_true", help="Explicitly show secret settings"
        )
    jobs = _group(
        subparsers, "jobs", "job_action", "Inspect and control background operations"
    )
    for action in ("list", "show", "watch", "cancel", "retry"):
        parser = _leaf(jobs, action, f"{action.capitalize()} background jobs")
        parser.add_argument(
            "--kind",
            choices=("all", "acquisition", "workspace", "diffusion")
            if action == "list"
            else ("acquisition", "workspace", "diffusion"),
            default="all" if action == "list" else "acquisition",
        )
        if action != "list":
            parser.add_argument("job_id")
        if action == "watch":
            _poll_options(parser)
        if action == "retry":
            parser.add_argument("--token-file")
    parser = _leaf(subparsers, "logs", "Read or follow server logs")
    parser.set_defaults(log_command=True)
    parser.add_argument("--lines", type=int, default=100)
    parser.add_argument("--file")
    parser.add_argument(
        "--level", choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    )
    parser.add_argument("--follow", action="store_true")
    _poll_options(parser)
    cache = _group(
        subparsers, "cache", "cache_action", "Inspect or clear inference caches"
    )
    _leaf(cache, "show", "Inspect cache usage")
    parser = _leaf(cache, "clear", "Clear a cache")
    parser.add_argument("kind", choices=("hot", "ssd"))
    diagnostics = _group(
        subparsers, "diagnostics", "diagnostic_action", "Inspect and run diagnostics"
    )
    for action in ("status", "list", "show", "results", "start", "cancel"):
        parser = _leaf(diagnostics, action, f"{action.capitalize()} diagnostics")
        if action in {"show", "results", "cancel"}:
            parser.add_argument("run_id")
        if action == "start":
            parser.add_argument(
                "kind", choices=("throughput", "accuracy", "context", "ane")
            )
            parser.add_argument("--options", default="{}", metavar="JSON_OR_@FILE")
    monitoring = _group(
        subparsers, "monitoring", "monitoring_action", "Inspect activity and usage"
    )
    _leaf(monitoring, "activity", "Inspect live server activity")
    parser = _leaf(monitoring, "usage", "Inspect persisted usage")
    parser.add_argument(
        "--range",
        choices=("today", "yesterday", "7d", "30d", "90d", "month"),
        default="today",
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--details", action="store_true")
    parser = _leaf(subparsers, "api", "Call a management endpoint directly")
    parser.set_defaults(raw_api=True)
    parser.add_argument(
        "method", choices=("GET", "POST", "PATCH", "PUT", "DELETE"), type=str.upper
    )
    parser.add_argument(
        "path", help="Management-relative path, e.g. server/info or /server/info"
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Query parameter; repeat for multiple values",
    )
    parser.add_argument("--body", metavar="JSON_OR_@FILE")
    parser.add_argument(
        "--reveal", action="store_true", help="Explicitly show secrets in API responses"
    )


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite JSON numbers are not supported")
    return number


def _json(value):
    try:
        text = Path(value[1:]).read_text() if value.startswith("@") else value
        return json.loads(
            text,
            parse_float=_finite_float,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError("Non-finite JSON numbers are not supported")
            ),
        )
    except (OSError, ValueError) as exc:
        raise CLIError(f"Invalid JSON input: {exc}", exit_code=2) from exc


def _object(value):
    data = _json(value)
    if not isinstance(data, dict):
        raise CLIError("Expected a JSON object")
    return data


def _fields(assignments, nested=False):
    result = {}
    for assignment in assignments:
        field, sep, raw = assignment.partition("=")
        parts = field.split(".")
        if not sep or any(not part or not part.isidentifier() for part in parts):
            raise CLIError("Expected FIELD=JSON with a valid field name")
        if nested and len(parts) != 2:
            raise CLIError(
                "Server fields require SECTION.FIELD=JSON, e.g. scheduler.max_num_seqs=4"
            )
        if not nested and len(parts) != 1:
            raise CLIError("Model fields must be top-level names")
        target = result.setdefault(parts[0], {}) if nested else result
        name = parts[-1]
        if name in target:
            raise CLIError(f"Repeated setting: {field}")
        target[name] = _json(raw)
    return result


def _secret_file(path):
    return read_key_file(path)


def _mask(data):
    if isinstance(data, list):
        return [_mask(item) for item in data]
    if isinstance(data, dict):
        if "section" in data and "secret" in data and "key" in data:
            return {
                key: ("••••••••" if value else value)
                if key == "default" and data.get("secret")
                else _mask(value)
                for key, value in data.items()
            }
        return {
            key: ("••••••••" if value else value)
            if key.lower()
            in {
                "key",
                "main_key",
                "new_key",
                "api_key",
                "token",
                "brave_api_key",
                "hf_token",
                "ms_token",
            }
            or "api_key" in key.lower()
            or key.lower().endswith(("_secret", "_token", "password"))
            else _mask(value)
            for key, value in data.items()
        }
    return data


def _id(value):
    return quote(value, safe="")


def _log_suffix(old, text):
    """Find rolling tail overlap in linear time, including repeated lines."""
    if not old:
        return text
    # Prefix-function over text followed by a unique sentinel and the old tail.
    sentinel = object()
    sequence = [*text, sentinel, *old[-len(text) :]] if text else []
    prefix = [0] * len(sequence)
    for index in range(1, len(sequence)):
        length = prefix[index - 1]
        while length and sequence[index] != sequence[length]:
            length = prefix[length - 1]
        if sequence[index] == sequence[length]:
            length += 1
        prefix[index] = length
    return text[prefix[-1] :] if prefix else ""


def _watch(client, output, args, path, params=None, logs=False):
    with output.watch("Following server logs" if logs else "Watching job"):
        return _watch_loop(client, output, args, path, params, logs)


def _watch_loop(client, output, args, path, params=None, logs=False):
    if (
        not math.isfinite(args.interval)
        or not math.isfinite(args.wait_timeout)
        or args.interval <= 0
        or args.wait_timeout <= 0
    ):
        raise CLIError("Polling interval and wait timeout must be positive")
    deadline = time.monotonic() + args.wait_timeout
    previous = None
    while True:
        data = client.request("GET", path, params=params)
        if not getattr(args, "json", False) and data != previous:
            if logs:
                text = data.get("logs", "")
                old = previous.get("logs", "") if previous else ""
                # Tail windows shift; print only the suffix following their overlap.
                suffix = _log_suffix(old, text)
                if suffix:
                    output.emit(suffix)
            else:
                output.emit(
                    _mask(
                        {
                            key: data[key]
                            for key in (
                                "id",
                                "status",
                                "state",
                                "progress",
                                "stage",
                                "phase",
                                "error",
                            )
                            if key in data
                        }
                    )
                )
        previous = data
        state = data.get("status", data.get("state"))
        if not logs and state in {
            "completed",
            "succeeded",
            "failed",
            "error",
            "cancelled",
            "canceled",
        }:
            if getattr(args, "json", False):
                output.emit(_mask(data))
            return 1 if state in {"failed", "error"} else 0
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if getattr(args, "json", False):
                output.emit(_mask(data))
            if logs:
                return 0
            raise CLIError("Job is still active; watch timed out", exit_code=1)
        time.sleep(min(args.interval, remaining))


def run(args):
    output = Output(args)
    client = ManagementClient.from_args(args)
    try:
        if getattr(args, "model_action", None):
            result = _models(client, output, args)
        elif getattr(args, "key_action", None):
            result = _keys(client, output, args)
        elif getattr(args, "settings_action", None):
            action = args.settings_action
            result = client.request(
                "PATCH" if action == "set" else "GET",
                "/server/defaults" if action == "defaults" else "/server/settings",
                json=_fields(args.assignments, nested=True)
                if action == "set"
                else None,
            )
        elif getattr(args, "job_action", None):
            base = "/diffusion/jobs" if args.kind == "diffusion" else "/operations"
            action = args.job_action
            path = base if action == "list" else f"{base}/{_id(args.job_id)}"
            if action == "retry" and args.kind == "diffusion":
                raise CLIError(
                    "Diffusion jobs do not support retry; start a new calibration or quantization operation"
                )
            if action == "watch":
                return _watch(client, output, args, path)
            body = (
                {"token": _secret_file(args.token_file) if args.token_file else ""}
                if action == "retry"
                else None
            )
            result = client.request(
                "POST" if action in {"cancel", "retry"} else "GET",
                path + (f"/{action}" if action in {"cancel", "retry"} else ""),
                json=body,
            )
            if action == "list" and args.kind == "all":
                availability = {"available": True}
                try:
                    diffusion = client.request("GET", "/diffusion/jobs")
                except CLIError as exc:
                    status = (exc.details or {}).get("status")
                    if status not in {404, 503}:
                        raise
                    diffusion = {"jobs": []}
                    availability = {
                        "available": False,
                        "status": status,
                        "message": str(exc),
                    }
                    if not getattr(args, "json", False):
                        output.emit(
                            "Diffusion jobs are unavailable on this server; showing management operations."
                        )
                result = {
                    "operations": result.get("operations", []),
                    "diffusion_jobs": diffusion.get("jobs", []),
                    "availability": {"diffusion": availability},
                }
            elif action == "list" and args.kind in {"acquisition", "workspace"}:
                result = {
                    "operations": [
                        op
                        for op in result.get("operations", [])
                        if _operation_group(op) == args.kind
                    ]
                }
        elif getattr(args, "log_command", False):
            if not 1 <= args.lines <= 10000:
                raise CLIError("--lines must be between 1 and 10000")
            params = {
                k: v
                for k, v in {
                    "lines": args.lines,
                    "file": args.file,
                    "level": args.level,
                }.items()
                if v is not None
            }
            if args.follow:
                return _watch(
                    client, output, args, "/monitoring/logs", params, logs=True
                )
            result = client.request("GET", "/monitoring/logs", params=params)
            if not getattr(args, "json", False):
                output.emit(result.get("logs", ""))
                return 0
        elif getattr(args, "cache_action", None):
            if args.cache_action == "clear":
                output.confirm(f"Clear the {args.kind} inference cache?")
                result = client.request("POST", f"/cache/{args.kind}/clear")
            else:
                result = client.request("GET", "/cache")
        elif getattr(args, "diagnostic_action", None):
            action = args.diagnostic_action
            path = (
                "/diagnostics/capabilities"
                if action == "status"
                else "/diagnostics/runs"
            )
            if action in {"show", "results", "cancel"}:
                path += f"/{_id(args.run_id)}" + (
                    f"/{action}" if action in {"cancel", "results"} else ""
                )
            result = client.request(
                "POST" if action in {"start", "cancel"} else "GET",
                path,
                json={"kind": args.kind, "options": _object(args.options)}
                if action == "start"
                else None,
            )
        elif getattr(args, "monitoring_action", None):
            params = (
                {
                    "range": args.range,
                    "model": args.model,
                    "include_details": args.details,
                }
                if args.monitoring_action == "usage"
                else None
            )
            result = client.request(
                "GET", f"/monitoring/{args.monitoring_action}", params=params
            )
        elif getattr(args, "raw_api", False):
            parsed = urlsplit(args.path)
            if parsed.scheme or parsed.netloc:
                raise CLIError("API path must be management-relative")
            path = "/" + validate_path(args.path)
            params = []
            for query in args.query:
                name, separator, value = query.partition("=")
                if (
                    not separator
                    or not name
                    or any(ord(char) <= 32 or ord(char) == 127 for char in name)
                ):
                    raise CLIError("Query parameters require NAME=VALUE")
                params.append((name, value))
            if args.method == "DELETE":
                output.confirm(f"DELETE {path}?")
            kwargs = {"json": _json(args.body) if args.body else None}
            if params:
                kwargs["params"] = params
            result = client.request(args.method, path, **kwargs)
        else:
            raise CLIError("Unknown management command")
        reveal = getattr(args, "reveal", False) or getattr(
            args, "key_action", None
        ) in {"create", "rotate"}
        data = result if reveal else _mask(result)
        if (
            not reveal
            and isinstance(data, dict)
            and isinstance(data.get("fields"), list)
        ):
            for field in data["fields"]:
                if field.get("secret"):
                    section = data.get("sections", {}).get(field.get("section"), {})
                    if section.get(field.get("key")):
                        section[field["key"]] = "••••••••"
        if not getattr(args, "json", False):
            data = _human_summary(data, args)
        output.emit(data)
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        client.close()


def _operation_group(operation):
    kind = operation.get("kind", "")
    return (
        "workspace"
        if kind.startswith("workspace")
        or kind in {"move", "verify", "revision", "update_check", "stage_update"}
        else "acquisition"
    )


def _human_summary(data, args):
    if not isinstance(data, dict):
        return data
    if getattr(args, "model_action", None) == "list":
        return [
            {
                "model": item.get("id"),
                "state": "loading"
                if item.get("is_loading")
                else "unloading"
                if item.get("is_unloading")
                else "loaded"
                if item.get("loaded")
                else "unloaded",
                "type": item.get("model_type"),
                "engine": item.get("engine_type"),
            }
            for item in data.get("models", [])
        ]
    if getattr(args, "job_action", None) == "list":
        rows = [
            dict(item, source=_operation_group(item))
            for item in data.get("operations", [])
        ]
        rows += [
            dict(item, source="diffusion")
            for item in data.get("diffusion_jobs", data.get("jobs", []))
        ]
        return [
            {
                key: item.get(key)
                for key in (
                    "id",
                    "source",
                    "kind",
                    "model_id",
                    "status",
                    "progress",
                    "stage",
                )
            }
            for item in rows
        ]
    if getattr(args, "key_action", None) == "list":
        return {
            "main_key": data.get("main_key"),
            "policy": data.get("policy", {}),
            "sub_keys": [
                {key: item.get(key) for key in ("id", "name", "key", "created_at")}
                for item in data.get("sub_keys", [])
            ],
        }
    if getattr(args, "diagnostic_action", None) == "list":
        rows = data.get("runs", [])
        return [
            {
                key: item.get(key)
                for key in ("id", "kind", "status", "created_at", "error")
            }
            for item in rows
        ]
    if getattr(args, "settings_action", None):
        return {
            "base_path": data.get("base_path"),
            "sections": data.get("sections", {}),
            **{
                key: value
                for key, value in data.items()
                if key
                in {
                    "restart_required",
                    "changed",
                    "live_applied",
                    "effective_model_dirs",
                }
            },
        }
    return data


def _models(client, output, args):
    action = args.model_action
    if action == "list":
        return client.request("GET", "/models")
    if action == "refresh":
        return client.request("POST", "/models/refresh")
    if action == "download":
        return client.request(
            "POST",
            f"/acquisition/{args.provider}/downloads",
            json={
                "repo_id": args.repo_id,
                "token": _secret_file(args.token_file) if args.token_file else "",
            },
        )
    path = f"/models/{_id(args.model_id)}"
    if action == "show":
        inventory = client.request("GET", "/models")
        match = next(
            (
                model
                for model in inventory.get("models", [])
                if model.get("id", model.get("model_id")) == args.model_id
            ),
            None,
        )
        if match is None:
            raise CLIError(f"Model not found: {args.model_id}")
        return match
    if action in {"load", "unload"}:
        return client.request("POST", f"{path}/{action}")
    if action == "settings":
        sub = args.model_settings_action
        return client.request(
            "PATCH" if sub == "set" else "POST" if sub == "reset" else "GET",
            f"{path}/settings" + ("/reset" if sub == "reset" else ""),
            json=_fields(args.assignments) if sub == "set" else None,
        )
    if action == "profiles":
        sub = args.profile_action
        endpoint = f"{path}/profiles"
        if sub in {"apply", "delete"}:
            endpoint += f"/{_id(args.name)}" + ("/apply" if sub == "apply" else "")
        if sub == "delete":
            output.confirm(f"Delete profile {args.name!r} from {args.model_id!r}?")
        body = (
            {
                "name": args.name,
                "settings": _object(args.settings),
                "description": args.description,
                "display_name": args.display_name,
                "expose_as_model": args.expose_as_model,
            }
            if sub == "create"
            else None
        )
        return client.request(
            "DELETE"
            if sub == "delete"
            else "POST"
            if sub in {"create", "apply"}
            else "GET",
            endpoint,
            json=body,
        )
    workspace = f"/workspace/models/{_id(args.model_id)}"
    if action == "move":
        output.confirm(f"Move {args.model_id!r} to storage root {args.root_id!r}?")
        return client.request(
            "POST",
            workspace + "/move",
            json={"root_id": args.root_id, "drain": args.drain},
        )
    if action == "remove":
        plan = client.request("GET", workspace + "/delete-plan")
        if getattr(args, "json", False):
            # Preserve the single JSON result on stdout; preview uses stderr.
            import sys

            print(json.dumps(_mask(plan), indent=2), file=sys.stderr)
        else:
            output.emit(plan, title="Files and settings to remove")
        output.confirm(
            f"Permanently remove {args.model_id!r} and the data in this plan?"
        )
        return client.request(
            "DELETE",
            workspace + "/delete",
            json={"plan_token": plan["plan_token"], "drain": args.drain},
        )
    raise CLIError("Unknown model command")


def _keys(client, output, args):
    action = args.key_action
    if action == "list":
        return client.request("GET", "/auth/keys")
    if action in {"create", "edit"}:
        body = {"name": args.name} if args.name is not None else {}
        if args.key_file:
            body["key"] = _secret_file(args.key_file)
        if action == "edit" and not body:
            raise CLIError("Provide --name or --key-file to edit a key")
        return client.request(
            "POST" if action == "create" else "PATCH",
            "/auth/subkeys" + (f"/{_id(args.key_id)}" if action == "edit" else ""),
            json=body,
        )
    if action == "revoke":
        output.confirm(f"Revoke API subkey {args.key_id!r}?")
        return client.request("DELETE", f"/auth/subkeys/{_id(args.key_id)}")
    output.confirm(
        "Rotate the main key? Existing management credentials will stop working."
        if args.main
        else f"Rotate API subkey {args.key_id!r}?"
    )
    key = _secret_file(args.key_file) if args.key_file else secrets.token_hex(32)
    result = client.request(
        "PATCH",
        "/auth/main-key" if args.main else f"/auth/subkeys/{_id(args.key_id)}",
        json={"key": key},
    )
    # Main-key response intentionally omits the secret. Show the new key once.
    return (
        {
            **result,
            "new_key": key,
            "next_step": "Use this replacement key for future management commands; reconnect the dashboard.",
        }
        if args.main
        else result
    )
