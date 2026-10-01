"""The application boundary is verified with lightweight synthetic children."""

import contextlib
import os
import signal
import socket
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from molto_cli import application


def settings(port, host="127.0.0.1"):
    return SimpleNamespace(server=SimpleNamespace(host=host, port=port))


def test_public_conflict_never_starts_child(monkeypatch):
    monkeypatch.setattr(application, "dashboard_command", lambda: ["node", "index.mjs"])
    monkeypatch.setattr(
        application.subprocess, "Popen", lambda *a, **kw: pytest.fail("child started")
    )
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        assert (
            application.run_application(settings(occupied.getsockname()[1]), ["serve"])
            == 1
        )


def test_multiple_hosts_fail_without_widening(monkeypatch):
    monkeypatch.setattr(application, "dashboard_command", lambda: ["node", "index.mjs"])
    monkeypatch.setattr(
        application.subprocess, "Popen", lambda *a, **kw: pytest.fail("child started")
    )
    assert application.run_application(settings(1234, "127.0.0.1,::1")) == 1


def test_missing_dashboard_actionable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        application, "__file__", str(tmp_path / "molto" / "application.py")
    )
    with pytest.raises(application.ApplicationError, match="build apps/dashboard/"):
        application.dashboard_command()


def test_installed_assets_require_bundled_node(monkeypatch, tmp_path):
    package = tmp_path / "molto"
    entry = package / "_dashboard" / "server" / "index.mjs"
    entry.parent.mkdir(parents=True)
    entry.touch()
    monkeypatch.setattr(application, "__file__", str(package / "application.py"))
    with pytest.raises(application.ApplicationError, match="Bundled Node runtime"):
        application.dashboard_command()


def test_source_runtime_override(monkeypatch, tmp_path):
    (tmp_path / "pnpm-workspace.yaml").touch()
    (tmp_path / "apps/dashboard").mkdir(parents=True)
    (tmp_path / "apps/dashboard/package.json").touch()
    entry = tmp_path / "apps/dashboard" / ".output" / "server" / "index.mjs"
    entry.parent.mkdir(parents=True)
    entry.touch()
    monkeypatch.setattr(
        application, "__file__", str(tmp_path / "molto" / "application.py")
    )
    monkeypatch.setenv("MOLTO_NODE", sys.executable)
    assert application.dashboard_command() == [sys.executable, str(entry)]


def child_command(body):
    return [sys.executable, "-c", body]


def run_supervisor(commands, ready, marker, stop, **kwargs):
    application.supervise(
        commands,
        {n: os.environ.copy() for n in commands},
        {n: () for n in commands},
        ready,
        marker,
        stop,
        shutdown_timeout=0.3,
        **kwargs,
    )


def test_dashboard_start_failure_never_starts_backend(tmp_path):
    backend = tmp_path / "backend-started"
    commands = {
        "dashboard": child_command("raise SystemExit(7)"),
        "backend": child_command(f"open({str(backend)!r}, 'w').close()"),
    }
    with pytest.raises(
        application.ApplicationError, match="dashboard repeatedly exited"
    ):
        run_supervisor(
            commands,
            {"dashboard": lambda: False, "backend": lambda: True},
            tmp_path / "restart",
            threading.Event(),
            retry_limit=0,
        )
    assert not backend.exists()


def test_backend_start_failure_reaps_dashboard(tmp_path):
    pid = tmp_path / "pid"
    commands = {
        "dashboard": child_command(
            f"import os,time; open({str(pid)!r}, 'w').write(str(os.getpid())); time.sleep(30)"
        ),
        "backend": child_command("raise SystemExit(9)"),
    }
    with pytest.raises(application.ApplicationError, match="backend repeatedly exited"):
        run_supervisor(
            commands,
            {"dashboard": lambda: pid.exists(), "backend": lambda: False},
            tmp_path / "restart",
            threading.Event(),
            retry_limit=0,
        )
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid.read_text()), 0)


def test_intentional_restart_keeps_dashboard_alive_and_shutdown_reaps(tmp_path):
    dashboard_pid = tmp_path / "dashboard-pid"
    backend_pids = tmp_path / "backend-pids"
    restart_flags = tmp_path / "restart-flags"
    marker = tmp_path / "restart"
    stop = threading.Event()
    commands = {
        "dashboard": child_command(
            f"import os,time; open({str(dashboard_pid)!r}, 'w').write(str(os.getpid())); time.sleep(30)"
        ),
        "backend": child_command(
            f"import os,time; open({str(backend_pids)!r}, 'a').write(str(os.getpid())+'\\n'); open({str(restart_flags)!r}, 'a').write(os.environ.get('MOLTO_BACKEND_RESTART', 'initial')+'\\n'); time.sleep(30)"
        ),
    }
    errors = []

    def run():
        try:
            run_supervisor(
                commands,
                {
                    "dashboard": lambda: dashboard_pid.exists(),
                    "backend": lambda: backend_pids.exists(),
                },
                marker,
                stop,
                retry_limit=0,
            )
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not backend_pids.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        first = int(backend_pids.read_text().splitlines()[0])
        dashboard = int(dashboard_pid.read_text())
        marker.touch()
        os.kill(first, signal.SIGTERM)
        while (
            len(backend_pids.read_text().splitlines()) < 2
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert len(backend_pids.read_text().splitlines()) == 2
        assert restart_flags.read_text().splitlines() == ["initial", "1"]
        assert int(dashboard_pid.read_text()) == dashboard
        os.kill(dashboard, 0)
    finally:
        stop.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert not errors
    for pid in [dashboard, *map(int, backend_pids.read_text().splitlines())]:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_readiness_timeout_reaps_partial_start(tmp_path):
    pid = tmp_path / "pid"
    commands = {
        "dashboard": child_command(
            f"import os,time; open({str(pid)!r}, 'w').write(str(os.getpid())); time.sleep(30)"
        )
    }
    with pytest.raises(application.ApplicationError, match="did not become ready"):
        run_supervisor(
            commands,
            {"dashboard": lambda: False},
            tmp_path / "restart",
            threading.Event(),
            startup_timeout=0.3,
        )
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid.read_text()), 0)


def test_runtime_contract_preserves_args_and_public_settings(monkeypatch, tmp_path):
    monkeypatch.setattr(application, "dashboard_command", lambda: ["node", "index.mjs"])
    monkeypatch.setenv("MOLTO_API_KEY", "never-log-this")
    observed = {}

    def capture(commands, environments, descriptors, ready, marker, stop, **kwargs):
        observed.update(
            commands=commands,
            environments=environments,
            descriptors=descriptors,
            marker=marker,
            mode=marker.parent.stat().st_mode & 0o777,
        )
        fd = descriptors["backend"][0]
        with socket.socket(fileno=os.dup(fd)) as listener:
            assert listener.getsockname()[0] == "127.0.0.1"
            assert (
                environments["dashboard"]["MOLTO_API_URL"]
                == f"http://127.0.0.1:{listener.getsockname()[1]}"
            )

    monkeypatch.setattr(application, "supervise", capture)
    configured = settings(0)
    configured.base_path = tmp_path
    assert (
        application.run_application(configured, ["serve", "--model-dir", "/example"])
        == 0
    )
    assert observed["commands"]["backend"][-3:] == ["serve", "--model-dir", "/example"]
    assert observed["environments"]["backend"]["MOLTO_SUPERVISED"] == "application"
    assert "MOLTO_API_KEY" not in observed["environments"]["dashboard"]
    assert observed["environments"]["dashboard"]["NITRO_HOST"] == configured.server.host
    assert observed["environments"]["dashboard"]["NITRO_PORT"] == "0"
    assert observed["descriptors"]["dashboard"] == ()
    assert observed["mode"] == 0o700
    assert not observed["marker"].parent.exists()


def test_preload_503_is_alive(monkeypatch):
    class Connection:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, method, path):
            assert (method, path) == ("GET", "/health")

        def getresponse(self):
            return SimpleNamespace(status=503)

        def close(self):
            pass

    monkeypatch.setattr(application.http.client, "HTTPConnection", Connection)
    assert application.http_health("127.0.0.1", 1234, "/health", (200, 503))
    assert not application.http_health("127.0.0.1", 1234, "/health")


def test_cli_parent_dispatches_before_inference_imports(monkeypatch):
    import builtins

    from molto_cli import cli
    from molto_config import settings as settings_module

    monkeypatch.delenv("MOLTO_INTERNAL_FD", raising=False)
    configured = SimpleNamespace(validate=lambda: [])
    monkeypatch.setattr(settings_module, "init_settings", lambda **kw: configured)
    monkeypatch.setattr(cli, "_migrate_saved_network_auth", lambda *args: None)
    observed = []
    monkeypatch.setattr(
        application,
        "run_application",
        lambda value, **kwargs: observed.append(value) or 0,
    )
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "uvicorn" or name.startswith("mlx") or name == "server":
            pytest.fail(f"Parent imported inference dependency: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(SystemExit) as exit:
        cli.serve_command(SimpleNamespace(base_path="/example"))
    assert exit.value.code == 0
    assert observed == [configured]


def test_restart_settings_honors_original_overrides(monkeypatch, tmp_path):
    from molto_config import settings as settings_module

    marker = tmp_path / "restart"
    marker.write_text('{"host":"0.0.0.0","port":9000}')
    current = settings(8000)
    effective = settings(8000)
    effective.validate = lambda: []
    args = SimpleNamespace(base_path=tmp_path, host="127.0.0.1", port=8000)
    observed = []
    monkeypatch.setattr(
        settings_module,
        "init_settings",
        lambda **kwargs: observed.append(kwargs) or effective,
    )
    assert application.restart_settings(marker, current, args) is None
    assert observed == [{"base_path": tmp_path, "cli_args": args}]


def test_changed_public_binding_restarts_application(monkeypatch, tmp_path):
    import json

    from molto_cli import cli_lifecycle

    monkeypatch.setattr(application, "dashboard_command", lambda: ["node", "index.mjs"])
    current = settings(0)
    changed = settings(0, "localhost")
    current.base_path = changed.base_path = tmp_path
    binding_updates = []
    monkeypatch.setattr(
        cli_lifecycle,
        "update_binding",
        lambda base, host, port: binding_updates.append((base, host, port)),
    )
    commands_seen = []
    monkeypatch.setattr(
        application, "restart_settings", lambda marker, active, args: changed
    )

    def capture(
        commands, environments, descriptors, ready, marker, stop, *, restart_requested
    ):
        commands_seen.append(environments["dashboard"]["HOST"])
        if len(commands_seen) == 1:
            assert "MOLTO_BACKEND_RESTART" not in environments["backend"]
            marker.write_text(json.dumps({"host": "localhost", "port": 0}))
            assert restart_requested()
            raise application.ApplicationRestartError()
        assert environments["backend"]["MOLTO_BACKEND_RESTART"] == "1"
        stop.set()

    monkeypatch.setattr(application, "supervise", capture)
    assert (
        application.run_application(
            current, ["serve"], cli_args=SimpleNamespace(base_path="/example")
        )
        == 0
    )
    assert commands_seen == ["127.0.0.1", "localhost"]
    assert binding_updates == [(tmp_path, "127.0.0.1", 0), (tmp_path, "localhost", 0)]


def test_invalid_restart_marker_is_actionable(tmp_path):
    marker = tmp_path / "restart"
    marker.write_text("broken")
    with pytest.raises(
        application.ApplicationError, match="restart molto serve manually"
    ):
        application.restart_settings(
            marker, settings(1234), SimpleNamespace(base_path=tmp_path)
        )


def test_real_saved_binding_reload_does_not_rewrite_settings(monkeypatch, tmp_path):
    import argparse
    import json

    for name in ("MOLTO_HOST", "MOLTO_PORT", "MOLTO_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    saved = tmp_path / "settings.json"
    saved.write_text(
        json.dumps(
            {"server": {"host": "127.0.0.1", "port": 9000}, "custom": {"keep": True}}
        )
    )
    original = saved.read_bytes()
    marker = tmp_path / "restart"
    marker.write_text('{"host":"127.0.0.1","port":9000}')
    args = argparse.Namespace(base_path=str(tmp_path), host=None, port=None)
    updated = application.restart_settings(marker, settings(8000), args)
    assert updated.server.port == 9000
    assert saved.read_bytes() == original
    args.port = 8000
    assert application.restart_settings(marker, settings(8000), args) is None
    assert saved.read_bytes() == original


@pytest.fixture
def instance_server():
    """A real listener used to distinguish foreign and expected readiness."""
    import http.server
    import json

    observations = []
    payload = {"instance": "foreign"}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            observations.append((self.path, dict(self.headers)))
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}
    )
    thread.start()
    try:
        yield server.server_address[1], payload, observations
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_public_readiness_identifies_instance_without_disclosing_nonce(instance_server):
    port, payload, observations = instance_server
    assert not application.public_ready("127.0.0.1", port, "expected-nonce")
    payload["instance"] = "expected-nonce"
    assert application.public_ready("127.0.0.1", port, "expected-nonce")
    for path, headers in observations:
        assert path == "/_molto/ready"
        assert "expected-nonce" not in repr(headers)
        assert "expected-nonce" not in path


def test_foreign_success_listener_never_starts_backend(instance_server, tmp_path):
    port, _, _ = instance_server
    started = tmp_path / "backend-started"
    commands = {
        "dashboard": child_command("import time; time.sleep(30)"),
        "backend": child_command(f"open({str(started)!r}, 'w').close()"),
    }
    with pytest.raises(
        application.ApplicationError, match="dashboard did not become ready"
    ):
        run_supervisor(
            commands,
            {
                "dashboard": lambda: application.public_ready(
                    "127.0.0.1", port, "own-instance"
                ),
                "backend": lambda: True,
            },
            tmp_path / "restart",
            threading.Event(),
            startup_timeout=0.4,
        )
    assert not started.exists()


def test_matching_instance_allows_backend_start(instance_server, tmp_path):
    port, payload, _ = instance_server
    payload["instance"] = "own-instance"
    started = tmp_path / "backend-started"
    stop = threading.Event()
    commands = {
        "dashboard": child_command("import time; time.sleep(30)"),
        "backend": child_command(
            f"import time; open({str(started)!r}, 'w').close(); time.sleep(30)"
        ),
    }

    def backend_ready():
        if started.exists():
            stop.set()
            return True
        return False

    run_supervisor(
        commands,
        {
            "dashboard": lambda: application.public_ready(
                "127.0.0.1", port, "own-instance"
            ),
            "backend": backend_ready,
        },
        tmp_path / "restart",
        stop,
        startup_timeout=2,
    )
    assert started.exists()


def test_stale_instance_after_dashboard_respawn_never_starts_backend(
    monkeypatch, instance_server, tmp_path
):
    import subprocess

    port, payload, _ = instance_server
    started = tmp_path / "backend-started"
    commands = {
        "dashboard": child_command("import time; time.sleep(30)"),
        "backend": child_command(f"open({str(started)!r}, 'w').close()"),
    }
    environments = {name: os.environ.copy() for name in commands}
    original = subprocess.Popen
    identities = []

    def spawn(command, **kwargs):
        assert command == commands["dashboard"], "Backend started from stale readiness"
        identities.append(kwargs["env"]["MOLTO_INSTANCE_ID"])
        # The public listener keeps reporting the identity of the first process.
        payload["instance"] = identities[0]
        if len(identities) == 1:
            command = child_command("import time; time.sleep(0.05)")
        return original(command, **kwargs)

    monkeypatch.setattr(application.subprocess, "Popen", spawn)
    with pytest.raises(
        application.ApplicationError, match="dashboard did not become ready"
    ):
        application.supervise(
            commands,
            environments,
            {name: () for name in commands},
            {
                "dashboard": lambda: application.public_ready(
                    "127.0.0.1", port, environments["dashboard"]["MOLTO_INSTANCE_ID"]
                ),
                "backend": lambda: True,
            },
            tmp_path / "restart",
            threading.Event(),
            startup_timeout=0.4,
            shutdown_timeout=0.3,
        )
    assert len(identities) == 2
    assert identities[0] != identities[1]
    assert not started.exists()


def test_development_command_uses_native_vite_on_public_address(monkeypatch, tmp_path):
    (tmp_path / "apps/dashboard").mkdir(parents=True)
    (tmp_path / "apps/dashboard" / "package.json").write_text("{}")
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "pnpm-workspace.yaml").touch()
    monkeypatch.setattr(
        application, "__file__", str(tmp_path / "molto" / "application.py")
    )
    monkeypatch.setattr(
        application.shutil,
        "which",
        lambda name: "/bin/pnpm" if name == "pnpm" else None,
    )
    assert application.development_command("127.0.0.1", 8123) == [
        "/bin/pnpm",
        "--dir",
        str(tmp_path / "apps/dashboard"),
        "exec",
        "vite",
        "dev",
        "--host",
        "127.0.0.1",
        "--port",
        "8123",
        "--strictPort",
    ]


def test_development_mode_rejects_installed_distribution(monkeypatch, tmp_path):
    monkeypatch.setattr(
        application, "__file__", str(tmp_path / "molto" / "application.py")
    )
    with pytest.raises(
        application.ApplicationError, match="requires a Molto source checkout"
    ):
        application.development_command("127.0.0.1", 8000)


def test_development_mode_missing_pnpm_is_actionable(monkeypatch, tmp_path):
    (tmp_path / "apps/dashboard").mkdir(parents=True)
    (tmp_path / "apps/dashboard" / "package.json").write_text("{}")
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "pnpm-workspace.yaml").touch()
    monkeypatch.setattr(
        application, "__file__", str(tmp_path / "molto" / "application.py")
    )
    monkeypatch.setattr(application.shutil, "which", lambda name: None)
    with pytest.raises(application.ApplicationError, match="requires pnpm"):
        application.development_command("127.0.0.1", 8000)


def test_development_launch_needs_no_production_assets(monkeypatch, tmp_path):
    monkeypatch.setattr(
        application,
        "dashboard_command",
        lambda: pytest.fail("production assets requested"),
    )
    observed = []

    def development(host, port):
        observed.append((host, port))
        return ["pnpm", "exec", "vite", "dev"]

    monkeypatch.setattr(application, "development_command", development)

    def capture(commands, environments, descriptors, ready, marker, stop, **kwargs):
        assert commands["dashboard"] == ["pnpm", "exec", "vite", "dev"]
        assert environments["dashboard"]["MOLTO_API_URL"].startswith(
            "http://127.0.0.1:"
        )
        assert environments["dashboard"]["NITRO_PORT"] == "0"
        assert commands["backend"][-2:] == ["serve", "--dashboard-dev"]
        stop.set()

    monkeypatch.setattr(application, "supervise", capture)
    configured = settings(0)
    configured.base_path = tmp_path
    assert (
        application.run_application(
            configured,
            ["serve", "--dashboard-dev"],
            cli_args=SimpleNamespace(dashboard_dev=True),
        )
        == 0
    )
    assert observed == [("127.0.0.1", 0)]


def test_shutdown_kills_stubborn_descendant_without_signaling_reaped_group(
    monkeypatch, tmp_path
):
    import subprocess

    import psutil

    pid_file = tmp_path / "descendant"
    descendant_code = (
        "import os,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(30)"
    )
    leader_code = f"import subprocess,sys,time; subprocess.Popen([sys.executable, '-c', {descendant_code!r}]); time.sleep(30)"
    leader = subprocess.Popen(child_command(leader_code), start_new_session=True)
    descendant = None
    try:
        deadline = time.monotonic() + 3
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        descendant = psutil.Process(int(pid_file.read_text()))
        monkeypatch.setattr(
            os, "killpg", lambda *args: pytest.fail("Unsafe group signal")
        )
        application.stop_children({"dashboard": leader}, grace=0.2)
        assert leader.returncode is not None
        _, alive = psutil.wait_procs([descendant], timeout=2)
        assert not alive
    finally:
        if leader.poll() is None:
            leader.kill()
            leader.wait()
        if descendant is not None:
            with contextlib.suppress(psutil.NoSuchProcess):
                descendant.kill()


def test_already_reaped_group_is_never_rediscovered(monkeypatch):
    import psutil

    monkeypatch.setattr(
        psutil, "process_iter", lambda: pytest.fail("Reused process group inspected")
    )
    assert application.group_members(SimpleNamespace(pid=12345, returncode=0)) == []


def test_owned_process_permission_failure_is_not_hidden():
    import psutil

    class Protected:
        def terminate(self):
            raise psutil.AccessDenied(pid=12345)

    with pytest.raises(psutil.AccessDenied):
        application.signal_members([Protected()])


def test_shutdown_captures_child_born_in_term_handler(monkeypatch, tmp_path):
    import subprocess

    import psutil

    ready = tmp_path / "ready"
    pid_file = tmp_path / "late-descendant"
    code = f"""
import os, signal, subprocess, sys, time

def shutdown(signum, frame):
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    open({str(pid_file)!r}, 'w').write(str(child.pid))
    sys.exit(0)

signal.signal(signal.SIGTERM, shutdown)
open({str(ready)!r}, 'w').close()
time.sleep(30)
"""
    leader = subprocess.Popen(child_command(code), start_new_session=True)
    descendant = None
    try:
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        monkeypatch.setattr(
            os, "killpg", lambda *args: pytest.fail("Unsafe group signal")
        )
        start = time.monotonic()
        application.stop_children({"dashboard": leader}, grace=0.5)
        assert time.monotonic() - start < 2
        assert leader.returncode == 0
        pid = int(pid_file.read_text())
        try:
            descendant = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        _, alive = psutil.wait_procs([descendant], timeout=1)
        assert not alive
    finally:
        if leader.poll() is None:
            leader.kill()
            leader.wait()
        if descendant is not None:
            with contextlib.suppress(psutil.NoSuchProcess):
                descendant.kill()
