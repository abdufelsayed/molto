"""Harmless detached-process proofs for CLI ownership and lifecycle behavior."""

import argparse
import concurrent.futures
import fcntl
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
from molto_cli import cli_lifecycle as lifecycle
from molto_cli.client import CLIError as LifecycleError


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for key in list(os.environ):
        if key.startswith("MOLTO_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("MOLTO_PORT", str(free_port()))
    monkeypatch.setattr(lifecycle, "is_homebrew", lambda: False)


def args(base, command="start", **overrides):
    values = dict(
        command=command,
        base_path=str(base),
        timeout=2,
        no_wait=False,
        host=None,
        port=None,
        model_dir=None,
        dashboard_dev=False,
        url=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


@pytest.fixture
def harmless_cli(monkeypatch, tmp_path):
    root = tmp_path / "fake-package"
    package = root / "molto_cli"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    (package / "cli.py").write_text("""
import argparse, http.server, os, signal, socketserver, sys
parser = argparse.ArgumentParser()
parser.add_argument('command')
parser.add_argument('--base-path')
parser.add_argument('--port', type=int)
parser.add_argument('--host', default='127.0.0.1')
args, _ = parser.parse_known_args()
if os.environ.get('FAKE_EXIT'):
    print('Synthetic startup failure', flush=True)
    sys.exit(7)
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{}')
    def log_message(self, *args):
        pass
class Server(socketserver.TCPServer):
    allow_reuse_address = True
signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))
if os.environ.get('FAKE_IGNORE_TERM'):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
# HTTPServer performs reverse DNS during bind; this loopback fixture needs no DNS.
server = Server((args.host, args.port), Handler)
server.serve_forever()
""")
    monkeypatch.chdir(root)
    children = []
    spawn = subprocess.Popen

    def track(*args, **kwargs):
        child = spawn(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", track)
    try:
        yield root
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=2)
        log_path = tmp_path / "base" / "logs" / "application.log"
        if log_path.exists():
            print("Synthetic server log:", log_path.read_text()[-4096:])


def test_real_start_status_stop_idempotent(harmless_cli, tmp_path):
    base = tmp_path / "base"
    started = lifecycle.run(args(base, port=free_port()))
    try:
        assert started["state"] == "running"
        assert started["manager"] == "local"
        record = base / "run" / "application.json"
        assert record.stat().st_mode & 0o777 == 0o600
        assert record.parent.stat().st_mode & 0o777 == 0o700
        assert not (base / "settings.json").exists()
        assert lifecycle.run(args(base, "status"))["pid"] == started["pid"]
        again = lifecycle.run(args(base, port=free_port()))
        assert again["pid"] == started["pid"]
        assert again["already_running"] is True
    finally:
        stopped = lifecycle.run(args(base, "stop"))
    assert stopped["state"] == "stopped"
    assert not record.exists()
    assert lifecycle.run(args(base, "stop"))["already_stopped"] is True


def test_detached_lifecycle_fixture_does_not_require_dns(harmless_cli, tmp_path):
    script = harmless_cli / "molto_cli" / "cli.py"
    script.write_text(
        "import socket\n"
        "def forbidden_lookup(host):\n"
        "    raise AssertionError('Loopback fixture attempted reverse DNS')\n"
        "socket.getfqdn = forbidden_lookup\n" + script.read_text()
    )
    base = tmp_path / "base"
    started = lifecycle.run(args(base, port=free_port()))
    try:
        assert started["state"] == "running"
        assert lifecycle.run(args(base, "status"))["healthy"] is True
    finally:
        assert lifecycle.run(args(base, "stop"))["state"] == "stopped"


def test_real_restart_changes_owned_pid(harmless_cli, tmp_path):
    base = tmp_path / "base"
    port = free_port()
    first = lifecycle.run(args(base, port=port))
    try:
        second = lifecycle.run(args(base, "restart", port=port))
        assert second["command"] == "restart"
        assert second["pid"] != first["pid"]
        assert second["state"] == "running"
    finally:
        lifecycle.run(args(base, "stop"))


def test_concurrent_starts_share_one_process(harmless_cli, tmp_path):
    base = tmp_path / "base"
    port = free_port()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(lifecycle.run, args(base, port=port)) for _ in range(2)
            ]
            results = [future.result() for future in futures]
        assert results[0]["pid"] == results[1]["pid"]
    finally:
        lifecycle.run(args(base, "stop"))


def test_child_early_exit_removes_record_and_points_to_log(
    harmless_cli, monkeypatch, tmp_path
):
    monkeypatch.setenv("FAKE_EXIT", "1")
    base = tmp_path / "base"
    with pytest.raises(LifecycleError, match="application.log"):
        lifecycle.run(args(base, port=free_port()))
    assert not (base / "run" / "application.json").exists()
    assert (
        "Synthetic startup failure" in (base / "logs" / "application.log").read_text()
    )


def test_no_wait_returns_durable_owned_process(harmless_cli, tmp_path):
    base = tmp_path / "base"
    started = lifecycle.run(args(base, port=free_port(), no_wait=True))
    try:
        assert started["state"] == "starting"
        assert (
            started["pid"]
            == json.loads((base / "run" / "application.json").read_text())["pid"]
        )
    finally:
        lifecycle.run(args(base, "stop"))


def test_corrupt_record_never_signals_any_process(monkeypatch, tmp_path):
    base = tmp_path / "base"
    (base / "run").mkdir(parents=True)
    (base / "run" / "application.json").write_text('{"pid":1}')
    monkeypatch.setattr(
        psutil.Process, "terminate", lambda *args: pytest.fail("Process signaled")
    )
    with pytest.raises(LifecycleError, match="Invalid lifecycle record"):
        lifecycle.run(args(base, "stop"))


def test_pid_reuse_record_never_kills_unrelated_process(tmp_path):
    base = tmp_path / "base"
    (base / "run").mkdir(parents=True)
    child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    try:
        record = {
            "version": 1,
            "pid": child.pid,
            "created": psutil.Process(child.pid).create_time() - 60,
            "argv": lifecycle.serve_command(args(base), base.resolve()),
            "host": "127.0.0.1",
            "port": 8000,
        }
        (base / "run" / "application.json").write_text(json.dumps(record))
        assert lifecycle.run(args(base, "stop"))["state"] == "stopped"
        assert child.poll() is None
    finally:
        child.terminate()
        child.wait()


def test_matching_pid_time_with_foreign_command_is_not_owned(tmp_path):
    base = tmp_path / "base"
    process = psutil.Process()
    record = {
        "pid": process.pid,
        "created": process.create_time(),
        "argv": lifecycle.serve_command(args(base), base.resolve()),
    }
    assert lifecycle.owned_process(record, base.resolve()) is None


@pytest.fixture
def framework_process(monkeypatch, tmp_path):
    framework = tmp_path / "Python.framework" / "Versions" / "3.13"
    launcher = framework / "bin" / "python3.13"
    application = (
        framework / "Resources" / "Python.app" / "Contents" / "MacOS" / "Python"
    )
    for executable in (launcher, application):
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.touch()
    alias = tmp_path / "venv" / "bin" / "python"
    alias.parent.mkdir(parents=True)
    alias.symlink_to(launcher)
    base = tmp_path.resolve()
    process = psutil.Process()
    record = {
        "pid": process.pid,
        "created": process.create_time(),
        "argv": [str(alias), *lifecycle.serve_command(args(base), base)[1:]],
    }
    actual = [str(application), *record["argv"][1:]]
    monkeypatch.setattr(psutil.Process, "cmdline", lambda self: actual)
    monkeypatch.setattr(sys, "platform", "darwin")
    return base, record, actual, launcher


@pytest.mark.parametrize("executable", ["framework", "symlink"])
def test_owned_process_accepts_equivalent_python_executables(
    framework_process, executable
):
    base, record, actual, launcher = framework_process
    if executable == "symlink":
        actual[0] = str(launcher)
    assert lifecycle.owned_process(record, base).pid == record["pid"]


@pytest.mark.parametrize(
    "mismatch",
    [
        "root",
        "version",
        "arguments",
        "owner",
        "created",
        "zombie",
        "platform",
        "missing",
        "launcher",
    ],
)
def test_framework_process_still_requires_matching_identity(
    framework_process, monkeypatch, tmp_path, mismatch
):
    base, record, actual, launcher = framework_process
    if mismatch in ("root", "version"):
        actual[0] = actual[0].replace(
            str(tmp_path) if mismatch == "root" else "/Versions/3.13/",
            str(tmp_path / "other") if mismatch == "root" else "/Versions/3.12/",
        )
        path = Path(actual[0])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    elif mismatch == "arguments":
        actual.extend(["--port", "9999"])
    elif mismatch == "owner":
        monkeypatch.setattr(
            psutil.Process, "uids", lambda self: SimpleNamespace(real=os.getuid() + 1)
        )
    elif mismatch == "created":
        record["created"] -= 60
    elif mismatch == "zombie":
        monkeypatch.setattr(psutil.Process, "status", lambda self: psutil.STATUS_ZOMBIE)
    elif mismatch == "platform":
        monkeypatch.setattr(sys, "platform", "linux")
    elif mismatch == "missing":
        launcher.unlink()
    elif mismatch == "launcher":
        foreign = launcher.with_name("pip3")
        foreign.touch()
        record["argv"][0] = str(foreign)
    assert lifecycle.owned_process(record, base) is None


def test_lock_timeout_is_bounded(tmp_path):
    base = tmp_path / "base"
    (base / "run").mkdir(parents=True)
    with (base / "run" / "application.lock").open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        start = time.monotonic()
        with pytest.raises(LifecycleError, match="Another Molto"):
            lifecycle.run(args(base, "status", timeout=0.1))
        assert time.monotonic() - start < 0.5


def test_stop_timeout_preserves_identity_record(harmless_cli, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_IGNORE_TERM", "1")
    base = tmp_path / "base"
    started = lifecycle.run(args(base, port=free_port()))
    try:
        with pytest.raises(LifecycleError, match="remains recorded"):
            lifecycle.run(args(base, "stop", timeout=0.1))
        assert (base / "run" / "application.json").exists()
    finally:
        process = psutil.Process(started["pid"])
        process.kill()
        process.wait(timeout=2)


def test_remote_stop_rejected_without_local_state(monkeypatch, tmp_path):
    base = tmp_path / "base"
    monkeypatch.setenv("MOLTO_URL", "http://192.0.2.1:8000")
    with pytest.raises(LifecycleError, match="another server"):
        lifecycle.run(args(base, "stop"))
    assert not base.exists()


def test_remote_status_does_not_claim_owned_pid(monkeypatch, tmp_path):
    observed = []

    def status(origin, timeout):
        observed.append(origin)
        return {"manager": "remote", "state": "running", "healthy": True}

    monkeypatch.setattr(lifecycle, "remote_status", status)
    result = lifecycle.run(args(tmp_path / "base", "status", url="https://example.com"))
    assert result["manager"] == "remote"
    assert "pid" not in result
    assert observed == ["https://example.com"]


def test_homebrew_default_preserved_with_explicit_options_local(monkeypatch, tmp_path):
    monkeypatch.setattr(lifecycle, "is_homebrew", lambda: True)
    monkeypatch.setattr(
        lifecycle, "resolve_default_base_path", lambda: tmp_path / "default"
    )
    monkeypatch.setattr(
        lifecycle, "homebrew", lambda args, base, timeout: {"manager": "homebrew"}
    )
    assert lifecycle.run(args(tmp_path / "default", "status"))["manager"] == "homebrew"
    assert lifecycle.run(args(tmp_path / "custom", "status"))["manager"] == "local"
    assert (
        lifecycle.run(args(tmp_path / "default", "status", port=8123))["manager"]
        == "local"
    )


def test_only_explicit_serve_options_forwarded(tmp_path):
    base = tmp_path.resolve()
    command = lifecycle.serve_command(
        args(
            base, host="127.0.0.1", port=8123, model_dir="/models", dashboard_dev=True
        ),
        base,
    )
    assert command[1:4] == ["-m", "molto_cli.cli", "serve"]
    assert command[-7:] == [
        "--host",
        "127.0.0.1",
        "--port",
        "8123",
        "--model-dir",
        "/models",
        "--dashboard-dev",
    ]


def test_update_binding_requires_current_owned_identity(
    harmless_cli, monkeypatch, tmp_path
):
    base = tmp_path / "base"
    started = lifecycle.run(args(base, port=free_port()))
    path = base / "run" / "application.json"
    try:
        original = path.read_bytes()
        assert lifecycle.update_binding(base, "127.0.0.1", 9000) is False
        assert path.read_bytes() == original
        with monkeypatch.context() as patch:
            patch.setattr(os, "getpid", lambda: started["pid"])
            with lifecycle.locked(base, 0.2):
                assert lifecycle.update_binding(base, "127.0.0.1", 9000) is True
        record = json.loads(path.read_text())
        assert record["port"] == 9000
        assert record["pid"] == started["pid"]
        assert path.stat().st_mode & 0o777 == 0o600
        record["created"] -= 60
        lifecycle.write_record(path, record)
        with monkeypatch.context() as patch:
            patch.setattr(os, "getpid", lambda: started["pid"])
            assert lifecycle.update_binding(base, "127.0.0.1", 9999) is False
        assert json.loads(path.read_text())["port"] == 9000
        record["created"] += 60
        lifecycle.write_record(path, record)
    finally:
        lifecycle.run(args(base, "stop"))


def test_update_binding_missing_record_does_not_create_state(tmp_path):
    base = tmp_path / "base"
    assert lifecycle.update_binding(base, "127.0.0.1", 8000) is False
    assert not base.exists()


def test_invalid_bind_configuration_never_spawns(monkeypatch, tmp_path):
    monkeypatch.setattr(
        subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Process spawned")
    )
    for overrides in ({"port": -1}, {"host": "bad host"}):
        with pytest.raises(LifecycleError, match="Invalid server configuration"):
            lifecycle.run(args(tmp_path / "base", **overrides))
    assert not (tmp_path / "base" / "models").exists()
    assert not (tmp_path / "base" / "settings.json").exists()


def test_healthy_unmanaged_server_is_reported_and_never_stopped(monkeypatch, tmp_path):
    monkeypatch.setattr(lifecycle, "health_status", lambda url, timeout=0.3: 200)
    monkeypatch.setattr(
        psutil.Process,
        "terminate",
        lambda *args: pytest.fail("Unowned process signaled"),
    )
    base = tmp_path / "base"
    status = lifecycle.run(args(base, "status"))
    assert status["manager"] == "unmanaged"
    assert status["healthy"] is True
    assert "pid" not in status
    for command in ("stop", "restart", "start"):
        with pytest.raises(LifecycleError, match="unmanaged"):
            lifecycle.run(args(base, command))


def test_homebrew_waits_for_health_unless_no_wait(monkeypatch, tmp_path):
    monkeypatch.setattr(lifecycle.shutil, "which", lambda command: "/usr/bin/brew")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: argparse.Namespace(returncode=0, stdout=""),
    )
    seen = []

    def health(url, timeout=0.3):
        seen.append(url)
        return len(seen) > 1

    monkeypatch.setattr(lifecycle, "probe_url", health)
    waited = lifecycle.homebrew(args(tmp_path, "start"), tmp_path, 1)
    assert waited["state"] == "running"
    assert len(seen) == 2
    seen.clear()
    detached = lifecycle.homebrew(args(tmp_path, "restart", no_wait=True), tmp_path, 1)
    assert detached["state"] == "starting"
    assert seen == []


def test_homebrew_readiness_timeout_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setattr(lifecycle.shutil, "which", lambda command: "/usr/bin/brew")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: argparse.Namespace(returncode=0, stdout=""),
    )
    monkeypatch.setattr(lifecycle, "probe_url", lambda url: False)
    with pytest.raises(LifecycleError, match="brew services info"):
        lifecycle.homebrew(args(tmp_path, "start"), tmp_path, 0.05)


def test_wildcard_results_use_connectable_loopback(tmp_path):
    record = {"pid": 123, "host": "0.0.0.0", "port": 8000}
    assert (
        lifecycle.result("status", tmp_path, "running", record)["url"]
        == "http://127.0.0.1:8000"
    )


def test_record_write_failure_stops_new_child(harmless_cli, monkeypatch, tmp_path):
    original = subprocess.Popen
    children = []

    def spawn(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", spawn)

    def broken(*args):
        raise OSError("synthetic persistence failure")

    monkeypatch.setattr(lifecycle, "write_record", broken)
    with pytest.raises(LifecycleError, match="Cannot manage"):
        lifecycle.run(args(tmp_path / "base", port=free_port()))
    assert children[0].poll() is not None


def test_lifecycle_record_does_not_capture_runtime_keys(
    harmless_cli, monkeypatch, tmp_path
):
    base = tmp_path / "base"
    monkeypatch.setenv("MOLTO_API_KEY", "runtime-secret")
    lifecycle.run(args(base, port=free_port()))
    try:
        record = (base / "run" / "application.json").read_text()
        assert "runtime-secret" not in record
        assert not (base / "settings.json").exists()
    finally:
        lifecycle.run(args(base, "stop"))


def test_loading_503_is_alive_but_start_waits_for_ready(
    harmless_cli, monkeypatch, tmp_path
):
    base = tmp_path / "base"
    started = lifecycle.run(args(base, port=free_port(), no_wait=True))
    try:
        monkeypatch.setattr(lifecycle, "public_status", lambda record, process: 503)
        status = lifecycle.run(args(base, "status"))
        assert status["state"] == "loading"
        assert status["alive"] is True
        assert status["healthy"] is False
        with pytest.raises(LifecycleError, match="remains managed"):
            lifecycle.run(args(base, timeout=0.1))
        assert (
            json.loads((base / "run" / "application.json").read_text())["pid"]
            == started["pid"]
        )
        assert psutil.Process(started["pid"]).is_running()
        monkeypatch.setattr(lifecycle, "public_status", lambda record, process: 200)
        assert lifecycle.run(args(base))["state"] == "running"
    finally:
        lifecycle.run(args(base, "stop"))


def test_restart_preserves_nonpersisted_dashboard_dev(harmless_cli, tmp_path):
    base = tmp_path / "base"
    port = free_port()
    lifecycle.run(args(base, port=port, dashboard_dev=True))
    try:
        restarted = lifecycle.run(args(base, "restart", port=port))
        assert restarted["state"] == "running"
        record = json.loads((base / "run" / "application.json").read_text())
        assert "--dashboard-dev" in record["argv"]
    finally:
        lifecycle.run(args(base, "stop"))


def test_remote_loading_is_reported_without_ownership(monkeypatch):
    monkeypatch.setattr(lifecycle, "health_status", lambda url, timeout: 503)
    data = lifecycle.remote_status("http://example.com:8000", 1)
    assert data["state"] == "loading"
    assert data["alive"] is True
    assert data["healthy"] is False
    assert "pid" not in data


def test_unmanaged_loading_never_claims_ready(monkeypatch, tmp_path):
    monkeypatch.setattr(lifecycle, "health_status", lambda url, timeout=0.3: 503)
    data = lifecycle.run(args(tmp_path / "base", "status"))
    assert data["manager"] == "unmanaged"
    assert data["state"] == "loading"
    assert data["healthy"] is False
    assert data["alive"] is True


def test_homebrew_status_reports_loading_and_readiness(monkeypatch, tmp_path):
    monkeypatch.setattr(lifecycle.shutil, "which", lambda name: "/usr/bin/brew")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: argparse.Namespace(
            returncode=0, stdout='[{"name":"molto","status":"started"}]'
        ),
    )
    monkeypatch.setattr(lifecycle, "health_status", lambda url: 503)
    loading = lifecycle.homebrew(args(tmp_path, "status"), tmp_path, 1)
    assert loading["state"] == "loading"
    assert loading["healthy"] is False
    monkeypatch.setattr(lifecycle, "health_status", lambda url: 200)
    assert (
        lifecycle.homebrew(args(tmp_path, "status"), tmp_path, 1)["state"] == "running"
    )


@pytest.mark.parametrize("port,host", [("bogus", "127.0.0.1"), (8123, 3)])
def test_malformed_saved_server_types_never_spawn(monkeypatch, tmp_path, port, host):
    base = tmp_path / "base"
    base.mkdir()
    saved = base / "settings.json"
    saved.write_text(json.dumps({"server": {"port": port, "host": host}}))
    monkeypatch.delenv("MOLTO_PORT")
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("Spawned with malformed configuration"),
    )
    with pytest.raises(LifecycleError, match="Invalid server configuration"):
        lifecycle.run(args(base))
