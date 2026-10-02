# SPDX-License-Identifier: Apache-2.0
"""Owned local application lifecycle; inference settings stay the child's concern."""

from __future__ import annotations

import contextlib
import copy
import fcntl
import http.client
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import psutil
from molto_config.settings import GlobalSettings, resolve_default_base_path
from molto_config.utils.install import is_homebrew

from molto_cli.client import CLIError, server_url, validate_origin


def failure(message: str, code: int = 1):
    from molto_cli.client import CLIError

    return CLIError(message, exit_code=code)


def serve_command(args, base: Path) -> list[str]:
    command = [sys.executable, "-m", "molto_cli.cli", "serve", "--base-path", str(base)]
    for name in ("host", "port", "model_dir"):
        value = getattr(args, name, None)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    if getattr(args, "dashboard_dev", False):
        command.append("--dashboard-dev")
    return command


@contextlib.contextmanager
def locked(base: Path, timeout: float):
    directory = base / "run"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if directory.is_symlink():
        raise failure(
            f"Lifecycle state directory must not be a symlink: {directory}", 2
        )
    directory.chmod(0o700)
    descriptor = os.open(
        directory / "application.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
    )
    try:
        os.fchmod(descriptor, 0o600)
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise failure(
                        "Another Molto lifecycle command is still running; retry after it completes."
                    ) from None
                time.sleep(0.05)
        yield directory / "application.json"
    finally:
        os.close(descriptor)


def read_record(path: Path) -> dict | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_size > 65536
    ):
        raise failure(f"Unsafe lifecycle record {path}; inspect it before retrying.", 2)
    try:
        record = json.loads(path.read_text())
        if not isinstance(record, dict) or record.get("version") != 1:
            raise ValueError("unsupported record")
        pid, created = record["pid"], record["created"]
        if (
            type(pid) is not int
            or pid <= 1
            or type(created) not in (int, float)
            or not math.isfinite(created)
            or created <= 0
        ):
            raise ValueError("invalid process identity")
        if not isinstance(record.get("argv"), list) or not all(
            isinstance(arg, str) for arg in record["argv"]
        ):
            raise ValueError("invalid command")
        if (
            not isinstance(record.get("host"), str)
            or type(record.get("port")) is not int
        ):
            raise ValueError("invalid address")
        return record
    except (ValueError, KeyError, OSError) as exc:
        raise failure(
            f"Invalid lifecycle record {path}; inspect it before retrying. No process was signaled.",
            2,
        ) from exc


def write_record(path: Path, record: dict) -> None:
    descriptor, name = tempfile.mkstemp(prefix="application-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(record, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def matching_command(expected: list[str], actual: list[str]) -> bool:
    if not actual or actual[1:] != expected[1:]:
        return False
    if actual[0] == expected[0]:
        return True
    try:
        executable = Path(expected[0]).resolve(strict=True)
        observed = Path(actual[0]).resolve(strict=True)
        if executable == observed:
            return True
        # python.org's macOS launcher re-execs its framework's app executable.
        framework = executable.parent.parent
        return (
            sys.platform == "darwin"
            and executable.parent.name == "bin"
            and framework.parent.name == "Versions"
            and framework.parent.parent.name == "Python.framework"
            and executable.name in ("python", "python3", f"python{framework.name}")
            and observed
            == (framework / "Resources/Python.app/Contents/MacOS/Python").resolve(
                strict=True
            )
        )
    except (OSError, RuntimeError):
        return False


def owned_process(record: dict | None, base: Path) -> psutil.Process | None:
    if record is None:
        return None
    expected = record["argv"]
    if len(expected) < 6 or expected[1:4] != ["-m", "molto_cli.cli", "serve"]:
        raise failure(
            "Lifecycle record does not identify a Molto application; no process was signaled.",
            2,
        )
    try:
        base_index = expected.index("--base-path") + 1
        if Path(expected[base_index]).resolve() != base:
            raise ValueError("different base path")
    except (ValueError, IndexError) as exc:
        raise failure(
            "Lifecycle record belongs to another base path; no process was signaled.", 2
        ) from exc
    try:
        process = psutil.Process(record["pid"])
        if not math.isclose(
            process.create_time(), record["created"], rel_tol=0, abs_tol=0.001
        ):
            return None
        if process.status() == psutil.STATUS_ZOMBIE:
            return None
        if process.uids().real != os.getuid() or not matching_command(
            expected, process.cmdline()
        ):
            return None
        return process
    except psutil.NoSuchProcess:
        return None
    except psutil.AccessDenied as exc:
        raise failure(
            "Cannot verify the recorded Molto process owner; no process was signaled."
        ) from exc


def public_status(record: dict, process: psutil.Process) -> int | None:
    try:
        # A foreign listener must not make our still-starting application ready.
        processes = [process, *process.children(recursive=True)]
        if not any(
            connection.status == psutil.CONN_LISTEN
            and connection.laddr.port == record["port"]
            for member in processes
            for connection in member.net_connections(kind="tcp")
        ):
            return None
        return health_status(server_url(record["host"], record["port"]))
    except psutil.Error:
        return None


def public_health(record: dict, process: psutil.Process) -> bool:
    return public_status(record, process) == 200


def result(
    command: str, base: Path, state: str, record: dict | None = None, **extra
) -> dict:
    data = {
        "command": command,
        "manager": "local",
        "state": state,
        "base_path": str(base),
    }
    if record:
        data.update(
            pid=record["pid"],
            url=server_url(record["host"], record["port"]),
            log_path=str(base / "logs" / "application.log"),
        )
    data.update(extra)
    return data


def health_status(url: str, timeout: float = 0.3) -> int | None:
    parsed = urlsplit(url)
    connection_type = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = connection_type(parsed.hostname, parsed.port, timeout=timeout)
    try:
        connection.request("GET", "/health")
        return connection.getresponse().status
    except (OSError, http.client.HTTPException):
        return None
    finally:
        connection.close()


def probe_url(url: str, timeout: float = 0.3) -> bool:
    return health_status(url, timeout) == 200


def unmanaged_url(base: Path, args=None) -> str | None:
    settings = GlobalSettings.load(base_path=str(base), cli_args=args)
    url = server_url(settings.server.host, settings.server.port)
    return url if health_status(url) in (200, 503) else None


def update_binding(base, host: str, port: int) -> bool:
    """Refresh this application's own record without contending with CLI start."""
    base = Path(base).expanduser().resolve()
    path = base / "run" / "application.json"
    try:
        record = read_record(path)
        if record is None or record["pid"] != os.getpid():
            return False
        process = owned_process(record, base)
        if process is None or process.pid != os.getpid():
            return False
        record.update(host=host, port=port)
        write_record(path, record)
        return True
    except (CLIError, OSError):
        return False


def stop_local(path: Path, base: Path, timeout: float) -> dict:
    record = read_record(path)
    process = owned_process(record, base)
    if process is None:
        url = unmanaged_url(base)
        if url is not None:
            raise failure(
                f"A healthy server at {url} is unmanaged by this CLI. Stop its foreground command or service manager; no process was signaled."
            )
        path.unlink(missing_ok=True)
        return result("stop", base, "stopped", already_stopped=True)
    try:
        process.terminate()  # psutil checks the recorded creation-time identity again.
        process.wait(timeout=timeout)
    except psutil.NoSuchProcess:
        pass
    except psutil.TimeoutExpired as exc:
        raise failure(
            f"Molto did not stop within {timeout:g}s. It remains recorded; inspect {base / 'logs' / 'application.log'} before retrying."
        ) from exc
    except psutil.AccessDenied as exc:
        raise failure(
            "Permission denied stopping the owned Molto application; its record was preserved."
        ) from exc
    path.unlink(missing_ok=True)
    return result("stop", base, "stopped")


def start_local(args, path: Path, base: Path, timeout: float) -> dict:
    record = read_record(path)
    process = owned_process(record, base)
    already = process is not None
    child = None
    if process is None:
        startup_args = SimpleNamespace(
            **{
                name: getattr(args, name, None)
                for name in ("host", "port", "model_dir")
            }
        )
        settings = GlobalSettings.load(base_path=str(base), cli_args=startup_args)
        if (
            not isinstance(settings.server.host, str)
            or type(settings.server.port) is not int
        ):
            raise failure(
                "Invalid server configuration: host must be text and port must be an integer.",
                2,
            )
        errors = settings.validate()
        if errors:
            raise failure("Invalid server configuration: " + "; ".join(errors), 2)
        if "," in settings.server.host:
            raise failure(
                "The bundled application requires one public host address.", 2
            )
        url = server_url(settings.server.host, settings.server.port)
        if health_status(url) in (200, 503):
            raise failure(
                f"A healthy server at {url} is unmanaged by this CLI. Use its existing command or service manager instead of starting another application."
            )
        argv = serve_command(args, base)
        log_path = base / "logs" / "application.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            log_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW, 0o600
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "ab") as log:
                child = subprocess.Popen(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
            process = psutil.Process(child.pid)
            record = {
                "version": 1,
                "pid": child.pid,
                "created": process.create_time(),
                "argv": argv,
                "host": settings.server.host,
                "port": settings.server.port,
            }
            write_record(path, record)
        except BaseException:
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait(timeout=timeout)
            raise
    if getattr(args, "no_wait", False):
        return result("start", base, "starting", record, already_running=already)
    deadline = time.monotonic() + timeout
    while True:
        record = read_record(path) or record
        process = owned_process(record, base)
        if process is None:
            exit_code = child.poll() if child is not None else None
            path.unlink(missing_ok=True)
            raise failure(
                f"Molto exited before becoming ready (exit {exit_code}). Inspect {base / 'logs' / 'application.log'}."
            )
        if public_health(record, process):
            return result("start", base, "running", record, already_running=already)
        if time.monotonic() >= deadline:
            raise failure(
                f"Molto is still starting after {timeout:g}s and remains managed. Run molto status --base-path {base}; inspect {base / 'logs' / 'application.log'}."
            )
        time.sleep(0.1)


def homebrew(args, base: Path, timeout: float) -> dict:
    brew = shutil.which("brew")
    if not brew:
        raise failure("Homebrew is not available on PATH.")
    command = args.command
    deadline = time.monotonic() + timeout
    argv = (
        [brew, "services", "info", "molto", "--json"]
        if command == "status"
        else [brew, "services", command, "molto"]
    )
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise failure(
            f"Homebrew {command} timed out; inspect brew services info molto."
        ) from exc
    if completed.returncode:
        raise failure(f"Homebrew {command} failed; inspect brew services info molto.")
    if command == "status":
        try:
            details = json.loads(completed.stdout)
            if isinstance(details, list):
                info = next(item for item in details if item.get("name") == "molto")
            elif isinstance(details, dict):
                info = details.get("molto", details)
            else:
                raise ValueError("invalid service response")
            state = "running" if info.get("status") == "started" else "stopped"
        except (ValueError, StopIteration, AttributeError) as exc:
            raise failure(
                "Cannot read Homebrew service status; inspect brew services info molto."
            ) from exc
    else:
        state = "stopped" if command == "stop" else "starting"
        if command in ("start", "restart") and not getattr(args, "no_wait", False):
            settings = GlobalSettings.load(base_path=str(base), cli_args=args)
            url = server_url(settings.server.host, settings.server.port)
            while not probe_url(url):
                if time.monotonic() >= deadline:
                    raise failure(
                        f"Homebrew Molto is not ready after {timeout:g}s; inspect brew services info molto and {base / 'logs'}."
                    )
                time.sleep(0.1)
            state = "running"
    data = {
        "command": command,
        "manager": "homebrew",
        "state": state,
        "base_path": str(base),
    }
    if command == "status" and state == "running":
        settings = GlobalSettings.load(base_path=str(base), cli_args=args)
        url = server_url(settings.server.host, settings.server.port)
        code = health_status(url)
        data.update(
            url=url,
            state="running"
            if code == 200
            else "loading"
            if code == 503
            else "unhealthy",
            healthy=code == 200,
            alive=code in (200, 503),
        )
    return data


def requested_origin(args, base: Path):
    raw = getattr(args, "url", None) or os.environ.get("MOLTO_URL")
    if not raw:
        return None
    target = validate_origin(raw)
    settings = GlobalSettings.load(base_path=str(base), cli_args=args)
    local = server_url(settings.server.host, settings.server.port)

    def canonical(url):
        parsed = urlsplit(url)
        host = "127.0.0.1" if parsed.hostname == "localhost" else parsed.hostname
        return (
            parsed.scheme,
            host,
            parsed.port or (443 if parsed.scheme == "https" else 80),
        )

    return None if canonical(target) == canonical(local) else target


def remote_status(origin: str, timeout: float) -> dict:
    code = health_status(origin, min(timeout, 5))
    return {
        "command": "status",
        "manager": "remote",
        "state": "running"
        if code == 200
        else "loading"
        if code == 503
        else "unreachable",
        "url": origin,
        "healthy": code == 200,
        "alive": code in (200, 503),
    }


def _run(args) -> dict:
    try:
        timeout = float(getattr(args, "timeout", 60))
    except (TypeError, ValueError) as exc:
        raise failure(
            "--timeout must be a finite number greater than zero.", 2
        ) from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise failure("--timeout must be a finite number greater than zero.", 2)
    default = resolve_default_base_path().expanduser().resolve()
    base = Path(getattr(args, "base_path", None) or default).expanduser().resolve()
    origin = requested_origin(args, base)
    if origin is not None:
        if args.command == "status":
            return remote_status(origin, timeout)
        raise failure(
            "Local lifecycle commands cannot control --url/MOLTO_URL pointing to another server. Remove that target to manage this installation; use molto api POST server/restart for remote API restart.",
            2,
        )
    explicit = (
        base != default
        or bool(os.environ.get("MOLTO_BASE_PATH"))
        or any(
            getattr(args, name, None) is not None
            for name in ("host", "port", "model_dir")
        )
        or bool(getattr(args, "dashboard_dev", False))
    )
    if (
        is_homebrew()
        and not explicit
        and not (base / "run" / "application.json").exists()
    ):
        return homebrew(args, base, timeout)
    with locked(base, timeout) as path:
        if args.command == "status":
            record = read_record(path)
            process = owned_process(record, base)
            if process is None:
                url = unmanaged_url(base, args)
                if url is not None:
                    code = health_status(url)
                    return {
                        "command": "status",
                        "manager": "unmanaged",
                        "state": "running" if code == 200 else "loading",
                        "healthy": code == 200,
                        "alive": code in (200, 503),
                        "url": url,
                        "base_path": str(base),
                    }
                return result(
                    "status", base, "stopped", stale_record=record is not None
                )
            code = public_status(record, process)
            return result(
                "status",
                base,
                "running" if code == 200 else "loading" if code == 503 else "starting",
                record,
                healthy=code == 200,
                alive=True,
            )
        if args.command == "stop":
            return stop_local(path, base, timeout)
        if args.command == "restart":
            previous = read_record(path)
            restart_args = copy.copy(args)
            if (
                previous is not None
                and owned_process(previous, base) is not None
                and "--dashboard-dev" in previous["argv"]
            ):
                restart_args.dashboard_dev = True
            stop_local(path, base, timeout)
            data = start_local(restart_args, path, base, timeout)
            data["command"] = "restart"
            return data
        if args.command == "start":
            return start_local(args, path, base, timeout)
        raise failure(f"Unknown lifecycle command: {args.command}", 2)


def run(args) -> dict:
    try:
        return _run(args)
    except (OSError, psutil.Error) as exc:
        base = getattr(args, "base_path", None) or resolve_default_base_path()
        raise failure(
            f"Cannot manage the Molto application ({type(exc).__name__}). Inspect {Path(base).expanduser() / 'logs' / 'application.log'}."
        ) from exc
