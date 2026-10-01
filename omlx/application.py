# SPDX-License-Identifier: Apache-2.0
"""Own the public dashboard and private inference processes for ``omlx serve``.

This module deliberately imports no inference or MLX modules. Both transports are
owned by this launcher; inference starts only after the public server is ready.
"""

from __future__ import annotations

import collections
import contextlib
import http.client
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import psutil


class ApplicationError(RuntimeError):
    """An actionable bundled-application startup or runtime failure."""


class ApplicationRestartError(Exception):
    """An explicit restart changed the effective public binding."""


def restart_settings(marker: Path, current, cli_args):
    from .settings import init_settings

    try:
        intent = json.loads(marker.read_text())
    except (OSError, ValueError) as exc:
        raise ApplicationError(
            "Invalid application restart request; restart omlx serve manually."
        ) from exc
    if (
        not isinstance(intent, dict)
        or not isinstance(intent.get("host"), str)
        or not isinstance(intent.get("port"), int)
    ):
        raise ApplicationError(
            "Invalid application restart request; restart omlx serve manually."
        )
    updated = init_settings(base_path=cli_args.base_path, cli_args=cli_args)
    errors = updated.validate()
    if errors:
        raise ApplicationError(
            "Cannot restart with saved configuration: " + "; ".join(errors)
        )
    if (updated.server.host, updated.server.port) != (
        current.server.host,
        current.server.port,
    ):
        return updated
    return None


def dashboard_command() -> list[str]:
    package = Path(__file__).resolve().parent
    bundled = package / "_dashboard"
    entry = bundled / "server" / "index.mjs"
    source = package.parent / "dashboard" / ".output" / "server" / "index.mjs"
    if not entry.is_file():
        entry = source
    if not entry.is_file():
        raise ApplicationError(
            "Dashboard assets are missing. Install a complete oMLX distribution, "
            "or build dashboard/ with pnpm build before running omlx serve."
        )
    node = bundled / "runtime" / "node"
    if not (node.is_file() and os.access(node, os.X_OK)):
        # A wheel must contain its runtime; PATH fallback is source-checkout only.
        if entry != source:
            raise ApplicationError(
                "Bundled Node runtime is missing or not executable; reinstall oMLX."
            )
        override = os.environ.get("OMLX_NODE")
        resolved = shutil.which(override or "node")
        if not resolved:
            raise ApplicationError(
                "Node is required for this source checkout. Install Node or set OMLX_NODE."
            )
        node = Path(resolved)
    return [str(node), str(entry)]


def development_command(host: str, port: int) -> list[str]:
    """Use native Vite only from a source checkout, on the application's port."""
    root = Path(__file__).resolve().parent.parent
    dashboard = root / "dashboard"
    if (
        not (root / "pyproject.toml").is_file()
        or not (dashboard / "package.json").is_file()
    ):
        raise ApplicationError(
            "--dashboard-dev requires an oMLX source checkout with dashboard/. Installed distributions use the bundled dashboard."
        )
    pnpm = shutil.which("pnpm")
    if pnpm is None:
        raise ApplicationError(
            "--dashboard-dev requires pnpm. Install pnpm, then run pnpm --dir dashboard install in the source checkout."
        )
    return [
        pnpm,
        "--dir",
        str(dashboard),
        "exec",
        "vite",
        "dev",
        "--host",
        host,
        "--port",
        str(port),
        "--strictPort",
    ]


def bind_public(host: str, port: int) -> list[socket.socket]:
    sockets: list[socket.socket] = []
    try:
        for name in dict.fromkeys(
            part.strip() for part in host.split(",") if part.strip()
        ):
            address = socket.getaddrinfo(name, port, type=socket.SOCK_STREAM)[0]
            family, kind, protocol, _, endpoint = address
            sock = socket.socket(family, kind, protocol)
            sockets.append(sock)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind(endpoint)
            sock.listen(128)
        if not sockets:
            raise ApplicationError("No public listening address was configured.")
        return sockets
    except OSError as exc:
        for sock in sockets:
            sock.close()
        raise ApplicationError(
            f"Cannot bind public dashboard address {host}:{port}: {exc}"
        ) from exc


def http_health(host: str, port: int, path: str, statuses=(200,)) -> bool:
    connection = http.client.HTTPConnection(host, port, timeout=0.3)
    try:
        connection.request("GET", path)
        return connection.getresponse().status in statuses
    except (OSError, http.client.HTTPException):
        return False
    finally:
        connection.close()


def public_ready(host: str, port: int, instance: str) -> bool:
    """Identify this launched Nitro process without trusting a foreign listener."""
    connection = http.client.HTTPConnection(host, port, timeout=0.3)
    try:
        connection.request("GET", "/_omlx/ready")
        response = connection.getresponse()
        if response.status != 200:
            return False
        payload = response.read(4097)
        if len(payload) > 4096:
            return False
        data = json.loads(payload)
        return isinstance(data, dict) and data.get("instance") == instance
    except (OSError, ValueError, http.client.HTTPException):
        return False
    finally:
        connection.close()


def child_alive(child: subprocess.Popen) -> bool:
    """Inspect without reaping: the original PID still anchors its process group."""
    if child.returncode is not None:
        return False
    try:
        return psutil.Process(child.pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def group_members(child: subprocess.Popen) -> list[psutil.Process]:
    """Snapshot owned identities before reaping makes the group ID reusable."""
    if child.returncode is not None:
        return []
    members = []
    for process in psutil.process_iter():
        try:
            if os.getpgid(process.pid) == child.pid:
                member = psutil.Process(process.pid)
                member.create_time()  # Cache identity for PID-reuse checks.
                members.append(member)
        except (ProcessLookupError, psutil.NoSuchProcess):
            continue
        except PermissionError:
            # Other users' process groups cannot contain our owned children.
            continue
    return members


def signal_members(members: list[psutil.Process], force: bool = False) -> None:
    # psutil validates each recorded creation time before signaling. Never send
    # another group signal after the leader has been reaped: macOS can return
    # EPERM for an empty group, and a reused group could belong to another app.
    for process in members:
        with contextlib.suppress(psutil.NoSuchProcess):
            process.kill() if force else process.terminate()


def stop_children(children: dict[str, subprocess.Popen], grace: float = 10) -> None:
    """Keep leaders unreaped until their final owned descendants are captured."""
    members = [
        process for child in children.values() for process in group_members(child)
    ]
    signal_members(members)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and any(
        child_alive(child) for child in children.values()
    ):
        time.sleep(min(0.02, max(0, deadline - time.monotonic())))

    # Stop any remaining leader before its last snapshot, so a TERM handler
    # cannot keep creating children while we discover the final group members.
    leaders = [
        psutil.Process(child.pid) for child in children.values() if child_alive(child)
    ]
    signal_members(leaders, force=True)
    kill_deadline = time.monotonic() + 1
    while time.monotonic() < kill_deadline and any(
        child_alive(child) for child in children.values()
    ):
        time.sleep(0.01)

    # Even a zombie leader still reserves its PID/group. Capture children born
    # during shutdown before wait() releases that ownership anchor.
    members.extend(
        process for child in children.values() for process in group_members(child)
    )
    signal_members(members, force=True)
    for child in children.values():
        child.wait(timeout=max(0, kill_deadline - time.monotonic()))
    _, alive = psutil.wait_procs(
        members, timeout=max(0, kill_deadline - time.monotonic())
    )
    signal_members(alive, force=True)


def supervise(
    commands,
    environments,
    descriptors,
    ready,
    restart_marker: Path,
    stop: threading.Event,
    *,
    startup_timeout: float = 300,
    shutdown_timeout: float = 10,
    retry_window: float = 60,
    retry_limit: int = 3,
    restart_requested=None,
) -> None:
    """Keep both children alive, retaining dashboard sessions on backend restart."""
    children: dict[str, subprocess.Popen] = {}
    starts: dict[str, float] = {}
    failures = {name: collections.deque() for name in commands}
    pending: dict[str, float] = {}
    live: set[str] = set()
    launched: set[str] = set()
    try:
        while not stop.is_set():
            now = time.monotonic()
            for name, command in commands.items():
                # Public readiness must precede any inference process startup.
                if (
                    name == "backend"
                    and "dashboard" not in live
                    and name not in children
                ):
                    continue
                child = children.get(name)
                if child is not None and not child_alive(child):
                    # Capture descendants while the leader is still unreaped.
                    abandoned = group_members(child)
                    child.wait()
                    signal_members(abandoned, force=True)
                    intentional = name == "backend" and restart_marker.exists()
                    if intentional:
                        if restart_requested is not None and restart_requested():
                            raise ApplicationRestartError()
                        restart_marker.unlink(missing_ok=True)
                    else:
                        history = failures[name]
                        while history and now - history[0] > retry_window:
                            history.popleft()
                        history.append(now)
                        if len(history) > retry_limit:
                            raise ApplicationError(
                                f"The {name} repeatedly exited (status {child.returncode}); check the server logs."
                            )
                    live.discard(name)
                    children.pop(name)
                    pending[name] = now + (
                        0 if intentional else min(2 ** (len(failures[name]) - 1), 5)
                    )
                    child = None
                if child is None:
                    if now < pending.get(name, 0):
                        continue
                    child_env = environments[name].copy()
                    if name == "dashboard":
                        # Readiness identifies this process, including crash recovery.
                        instance = secrets.token_urlsafe(32)
                        child_env["OMLX_INSTANCE_ID"] = instance
                        environments[name]["OMLX_INSTANCE_ID"] = instance
                    if name == "backend" and name in launched:
                        child_env["OMLX_BACKEND_RESTART"] = "1"
                    children[name] = subprocess.Popen(
                        command,
                        env=child_env,
                        pass_fds=descriptors[name],
                        start_new_session=True,
                    )
                    starts[name] = now
                    launched.add(name)
                if name not in live:
                    if (
                        now - starts[name] >= 0.2
                        and child_alive(children[name])
                        and ready[name]()
                        and child_alive(children[name])
                    ):
                        live.add(name)
                    elif now - starts[name] > startup_timeout:
                        raise ApplicationError(
                            f"The {name} did not become ready within {startup_timeout:g} seconds; check the server logs."
                        )
            stop.wait(0.1)
    finally:
        stop_children(children, shutdown_timeout)


def run_application(settings, argv: list[str] | None = None, *, cli_args=None) -> int:
    """Start standard Nitro first, then the private inference child."""
    stop = threading.Event()
    previous = {}
    application_restarted = False
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, lambda signum, frame: stop.set())
        while not stop.is_set():
            hosts = list(
                dict.fromkeys(
                    h.strip() for h in settings.server.host.split(",") if h.strip()
                )
            )
            if len(hosts) != 1:
                raise ApplicationError(
                    "The bundled dashboard supports one public bind address. Set server.host to a single address."
                )
            host = hosts[0]
            node_command = (
                development_command(host, settings.server.port)
                if getattr(cli_args, "dashboard_dev", False)
                else dashboard_command()
            )
            # Fail immediately on an occupied public port, before either child starts.
            # Nitro performs its own bind after this preflight and remains authoritative.
            for listener in bind_public(host, settings.server.port):
                listener.close()
            from .cli_lifecycle import update_binding

            update_binding(settings.base_path, host, settings.server.port)
            probe_host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
            with tempfile.TemporaryDirectory(prefix="omlx-", dir="/tmp") as directory:
                os.chmod(directory, 0o700)
                marker = Path(directory) / "restart"
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as private:
                    private.bind(("127.0.0.1", 0))
                    private.listen(128)
                    backend_port = private.getsockname()[1]
                    backend_env = os.environ.copy()
                    backend_env.update(
                        OMLX_INTERNAL_FD=str(private.fileno()),
                        OMLX_SUPERVISED="application",
                        OMLX_RESTART_MARKER=str(marker),
                    )
                    if application_restarted:
                        backend_env["OMLX_BACKEND_RESTART"] = "1"
                    else:
                        backend_env.pop("OMLX_BACKEND_RESTART", None)
                    dashboard_env = os.environ.copy()
                    for key in (
                        "OMLX_INTERNAL_FD",
                        "OMLX_API_KEY",
                        "OMLX_RESTART_MARKER",
                        "OMLX_BACKEND_RESTART",
                    ):
                        dashboard_env.pop(key, None)
                    dashboard_env.pop("OMLX_INSTANCE_ID", None)
                    dashboard_env.update(
                        OMLX_API_URL=f"http://127.0.0.1:{backend_port}",
                        NITRO_HOST=host,
                        NITRO_PORT=str(settings.server.port),
                        HOST=host,
                        PORT=str(settings.server.port),
                    )
                    command_args = list(sys.argv[1:] if argv is None else argv)
                    commands = {
                        "dashboard": node_command,
                        "backend": [sys.executable, "-m", "omlx.cli", *command_args],
                    }
                    address = f"[{host}]" if ":" in host else host
                    print(
                        f"Starting oMLX dashboard at http://{address}:{settings.server.port}",
                        flush=True,
                    )
                    updated_settings = []

                    def restart_requested(
                        marker=marker, current=settings, results=updated_settings
                    ):
                        if cli_args is None:
                            return False
                        updated = restart_settings(marker, current, cli_args)
                        if updated is not None:
                            results.append(updated)
                            return True
                        return False

                    try:
                        supervise(
                            commands,
                            {"backend": backend_env, "dashboard": dashboard_env},
                            {"backend": (private.fileno(),), "dashboard": ()},
                            {
                                "backend": lambda backend_port=backend_port: (
                                    http_health(
                                        "127.0.0.1", backend_port, "/health", (200, 503)
                                    )
                                ),
                                "dashboard": lambda probe_host=probe_host, port=settings.server.port, dashboard_env=dashboard_env: (
                                    public_ready(
                                        probe_host,
                                        port,
                                        dashboard_env["OMLX_INSTANCE_ID"],
                                    )
                                ),
                            },
                            marker,
                            stop,
                            restart_requested=restart_requested,
                        )
                    except ApplicationRestartError:
                        settings = updated_settings[0]
                        application_restarted = True
                        continue
                break
        return 0
    except (ApplicationError, OSError) as exc:
        print(f"oMLX startup error: {exc}", file=sys.stderr)
        return 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
