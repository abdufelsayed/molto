"""CLI subprocesses through built Nitro and real management routes, without weights."""

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil
import pytest

ROOT = Path(__file__).parents[1]
MODEL = "mlx-community/test-model"


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def ready(url, process):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Isolated gateway fixture exited during startup")
        try:
            if httpx.get(url, timeout=0.5).status_code == 200:
                return
        except httpx.RequestError:
            pass
        time.sleep(0.1)
    raise RuntimeError("Isolated gateway fixture did not become ready")


@pytest.fixture
def gateway(tmp_path):
    node = shutil.which("node")
    output = ROOT / "dashboard/.output/server/index.mjs"
    if not node or not output.exists():
        pytest.skip("Run pnpm build in dashboard and install Node for public CLI proof")
    backend_port, public_port = free_port(), free_port()
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("OMLX_")
    }
    env.update(HOME=str(tmp_path), PYTHONPATH=str(ROOT), NO_COLOR="1")
    url = f"http://127.0.0.1:{public_port}"
    processes = []
    try:
        with (
            (tmp_path / "backend.log").open("w") as backend_log,
            (tmp_path / "gateway.log").open("w") as gateway_log,
        ):
            backend = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "dashboard/tests/omlx_server.py"),
                    str(backend_port),
                ],
                env=env,
                cwd=tmp_path,
                stdout=backend_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(backend)
            ready(f"http://127.0.0.1:{backend_port}/health", backend)
            public_env = {
                **env,
                "OMLX_API_URL": f"http://127.0.0.1:{backend_port}",
                "HOST": "127.0.0.1",
                "PORT": str(public_port),
            }
            public = subprocess.Popen(
                [node, str(output)],
                env=public_env,
                cwd=tmp_path,
                stdout=gateway_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(public)
            ready(url + "/health", public)

            def command(*args, key="dashboard-test-key", json_output=True, expected=0):
                command_env = {**env, "OMLX_API_KEY": key}
                argv = [
                    sys.executable,
                    "-m",
                    "omlx.cli",
                    "--url",
                    url,
                    "--base-path",
                    str(tmp_path / "client"),
                ]
                if json_output:
                    argv.append("--json")
                result = subprocess.run(
                    [*argv, *args],
                    env=command_env,
                    cwd=tmp_path,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                assert result.returncode == expected, result.stderr
                if expected:
                    assert not result.stdout
                    return json.loads(result.stderr)
                assert not result.stderr, result.stderr
                return json.loads(result.stdout) if json_output else result.stdout

            yield command, url, backend_port
    finally:
        for process in reversed(processes):
            try:
                leader = psutil.Process(process.pid)
                children = leader.children(recursive=True)
                leader.terminate()
                process.wait(timeout=10)
                for child in children:
                    if child.is_running():
                        child.kill()
                psutil.wait_procs(children, timeout=3)
            except psutil.NoSuchProcess:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)


def test_public_inventory_and_model_settings(gateway):
    command, _, _ = gateway
    inventory = command("models", "list")
    assert MODEL in [item["id"] for item in inventory["models"]]
    shown = command("models", "show", MODEL)
    assert shown["id"] == MODEL
    changed = command("models", "settings", "set", MODEL, "temperature=0.25")
    assert changed["settings"]["temperature"] == 0.25
    assert (
        command("models", "settings", "get", MODEL)["settings"]["temperature"] == 0.25
    )
    human = command("models", "list", json_output=False)
    assert "Model" in human and MODEL in human
    assert "\x1b[" not in human


def test_public_key_controls_and_subkey_permissions(gateway):
    command, _, _ = gateway
    masked = command("keys", "list")
    assert masked["main_key"] != "dashboard-test-key"
    created = command("keys", "create", "--name", "CLI proof")
    subkey = created["sub_key"]
    assert subkey["name"] == "CLI proof" and len(subkey["key"]) >= 4
    denied = command("models", "list", key=subkey["key"], expected=3)
    assert "main API key" in denied["error"]["message"]
    renamed = command("keys", "edit", subkey["id"], "--name", "Renamed")
    assert renamed["sub_key"]["name"] == "Renamed"
    assert renamed["sub_key"]["key"] != subkey["key"]  # masked output
    assert command("keys", "revoke", subkey["id"], "--yes")["deleted"]


def test_public_operations_settings_and_filtered_logs(gateway):
    command, _, _ = gateway
    changed = command("settings", "set", "sampling.temperature=0.35")
    assert "sampling.temperature" in changed["changed"]
    assert command("settings", "get")["sections"]["sampling"]["temperature"] == 0.35
    jobs = command("jobs", "list")
    assert jobs["operations"]
    operation_id = jobs["operations"][0]["id"]
    watched = command("jobs", "watch", operation_id, "--wait-timeout", "2")
    assert watched["status"] == "succeeded"
    logs = command("logs", "--level", "ERROR")
    assert "fixture failure" in logs["logs"]
    assert "fixture ready" not in logs["logs"]
    stats = command("api", "GET", "stats", "--query", "scope=alltime")
    assert stats["total_requests"] == 80


def test_main_key_required_even_when_local_bypass_enabled(gateway):
    command, url, _ = gateway
    command(
        "api", "PATCH", "auth/policy", "--body", '{"skip_api_key_verification":true}'
    )
    command("models", "list", key="inference-sub-key", expected=3)
    command("models", "list", key="wrong-key", expected=3)
    assert httpx.get(url + "/api/management/v1/models").status_code == 401
    assert (
        httpx.get(
            url + "/management/v1/models",
            headers={"Authorization": "Bearer dashboard-test-key"},
        ).status_code
        == 404
    )


def test_guarded_setup_from_cli(gateway):
    command, _, backend_port = gateway
    assert (
        httpx.post(
            f"http://127.0.0.1:{backend_port}/__test__/reset", params={"setup": "true"}
        ).status_code
        == 200
    )
    result = command("init", key="new-local-main-key")
    assert result == {"configured": True}
    assert command("models", "list", key="new-local-main-key")["models"]
