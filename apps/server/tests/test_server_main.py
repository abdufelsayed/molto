"""Standalone module startup must wire management and inference to one runtime."""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from unittest.mock import patch

import pytest


@pytest.fixture
def module_entry(monkeypatch, tmp_path, request):
    """Run server.main() as ``python -m omlx_server.server --model-dir <tmp>``."""
    from omlx_config.settings import reset_settings
    from omlx_server import server

    # main() loads GlobalSettings. OMLX_BASE_PATH is first in its base
    # path resolution order, so setting it (plus HOME for any other ~
    # expansion) keeps the test away from the real user configuration;
    # HOME alone is not enough when a macOS app bootstrap file exists.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OMLX_BASE_PATH", str(tmp_path / "omlx-base"))
    monkeypatch.delenv("OMLX_API_KEY", raising=False)
    reset_settings()

    # Reset the middleware stack so init_server's add_middleware works
    # even if another test in this process already started the app.
    server.app.middleware_stack = None

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    argv = [
        "omlx_server.server",
        "--model-dir",
        str(model_dir),
        "--api-key",
        "test-key",
    ]
    if hasattr(request, "param"):
        argv.extend(["--port", str(request.param)])
    with (
        patch.object(sys, "argv", argv),
        # Keep the process-wide allocator setting out of the test run.
        patch("mlx.core.set_cache_limit"),
        patch("uvicorn.run") as uvicorn_run,
    ):
        server.main()

    from types import SimpleNamespace

    yield SimpleNamespace(app=uvicorn_run.call_args.args[0]), uvicorn_run
    reset_settings()


def test_main_reaches_uvicorn_with_model_dir(module_entry):
    _, uvicorn_run = module_entry
    uvicorn_run.assert_called_once()


def test_main_defaults_to_loopback(module_entry):
    _, uvicorn_run = module_entry
    assert uvicorn_run.call_args.kwargs["host"] == "127.0.0.1"


@pytest.mark.parametrize("module_entry", [9123], indirect=True)
def test_module_port_matches_management_listener(module_entry):
    from fastapi.testclient import TestClient

    server, uvicorn_run = module_entry
    response = TestClient(server.app).get(
        "/management/v1/server/info",
        headers={"Authorization": "Bearer test-key"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["port"] == uvicorn_run.call_args.kwargs["port"] == 9123
    server.app.state.controller.state.global_settings.server.port = 9234
    pending = (
        TestClient(server.app)
        .get(
            "/management/v1/server/info",
            headers={"Authorization": "Bearer test-key"},
        )
        .json()
    )
    assert pending["port"] == 9123
    assert pending["configured_port"] == 9234


def test_main_rejects_network_bind_without_api_key(monkeypatch, tmp_path, capsys):
    from omlx_config.settings import reset_settings
    from omlx_server import server

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OMLX_BASE_PATH", str(tmp_path / "omlx-base"))
    monkeypatch.delenv("OMLX_API_KEY", raising=False)
    reset_settings()
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    argv = [
        "omlx_server.server",
        "--model-dir",
        str(model_dir),
        "--host",
        "0.0.0.0",
    ]

    try:
        with (
            patch.object(sys, "argv", argv),
            patch.object(server.ServerApplication, "initialize") as init_server,
            patch("uvicorn.run") as uvicorn_run,
            pytest.raises(SystemExit) as exc_info,
        ):
            server.main()

        assert exc_info.value.code == 1
        assert "API key is required" in capsys.readouterr().out
        init_server.assert_not_called()
        uvicorn_run.assert_not_called()
    finally:
        reset_settings()


def test_main_accepts_network_bind_with_api_key(monkeypatch, tmp_path):
    from omlx_config.settings import reset_settings
    from omlx_server import server

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OMLX_BASE_PATH", str(tmp_path / "omlx-base"))
    monkeypatch.delenv("OMLX_API_KEY", raising=False)
    reset_settings()
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    argv = [
        "omlx_server.server",
        "--model-dir",
        str(model_dir),
        "--host",
        "0.0.0.0",
        "--api-key",
        "test-key",
    ]

    try:
        with (
            patch.object(sys, "argv", argv),
            patch.object(server.ServerApplication, "initialize") as init_server,
            patch("mlx.core.set_cache_limit"),
            patch("uvicorn.run") as uvicorn_run,
        ):
            server.main()

        init_server.assert_called_once()
        assert init_server.call_args.kwargs["api_key"] == "test-key"
        assert uvicorn_run.call_args.kwargs["host"] == "0.0.0.0"
    finally:
        reset_settings()


def test_init_server_rejects_network_bind_without_api_key(tmp_path):
    from omlx_config.settings import GlobalSettings
    from omlx_server import server

    settings = GlobalSettings(base_path=tmp_path)
    settings.server.host = "0.0.0.0"

    with pytest.raises(ValueError, match="API key is required"):
        server.app.state.controller.initialize(
            model_dirs=str(tmp_path),
            api_key=None,
            global_settings=settings,
        )


def test_main_wires_global_settings(module_entry):
    # Module startup supplies the same persisted settings to management.
    server, _ = module_entry
    assert server.app.state.controller.state.global_settings is not None


def test_management_reads_initialized_runtime(module_entry):
    from fastapi.testclient import TestClient

    server, _ = module_entry
    client = TestClient(server.app, client=("127.0.0.1", 50000))
    assert client.get("/management/v1/state").status_code == 401
    response = client.get(
        "/management/v1/state", headers={"Authorization": "Bearer test-key"}
    )
    assert response.status_code == 200, response.text
    assert response.json()["model_count"] == 0
    assert client.get("/admin").status_code == 404
    jobs = client.get(
        "/management/v1/diffusion/jobs",
        headers={"Authorization": "Bearer test-key"},
    )
    assert jobs.status_code == 200, jobs.text
    assert jobs.json() == {"jobs": []}
    assert (
        server.app.state.controller.state.diffusion_jobs.pool
        is server.app.state.controller.state.engine_pool
    )


def test_bad_preparation_history_preserves_inference_startup(monkeypatch, tmp_path):
    from omlx_config.settings import GlobalSettings
    from omlx_server import server

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    history = tmp_path / "preparation" / "diffusion" / "jobs.json"
    history.parent.mkdir(parents=True)
    history.write_text("invalid json")
    settings = GlobalSettings(base_path=tmp_path)
    settings.server.gpu_keep_warm_interval = 0
    app = server.create_app()
    with patch("mlx.core.set_cache_limit"):
        app.state.controller.initialize(
            model_dirs=str(model_dir), api_key="test-key", global_settings=settings
        )
    assert app.state.controller.state.engine_pool is not None
    assert app.state.controller.state.diffusion_jobs is None


def test_module_entry_management_end_to_end(tmp_path):
    # A fresh interpreter verifies module startup and HTTP wiring without
    # inheriting the test process's application state or monkeypatches.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    env["OMLX_BASE_PATH"] = str(tmp_path / "base")
    env["OMLX_API_KEY"] = "e2e-key-1234"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "omlx_server.server",
            "--model-dir",
            str(model_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                pytest.fail(f"server exited early:\n{proc.stdout.read()}")
            try:
                urllib.request.urlopen(f"{base}/health", timeout=1)
                break
            except (urllib.error.URLError, OSError):
                time.sleep(0.25)
        else:
            pytest.fail("server did not become healthy within 60s")

        with pytest.raises(urllib.error.HTTPError) as denied:
            urllib.request.urlopen(f"{base}/management/v1/state", timeout=10)
        assert denied.value.code == 401
        headers = {"Authorization": "Bearer e2e-key-1234"}
        for path, expected in [
            ("/management/v1/state", "model_count"),
            ("/management/v1/models", "models"),
            ("/v1/models", "data"),
        ]:
            request = urllib.request.Request(f"{base}{path}", headers=headers)
            with urllib.request.urlopen(request, timeout=10) as response:
                assert response.status == 200
                payload = json.load(response)
                assert expected in payload, payload
        for path in ("/admin", "/admin/dashboard", "/admin/api/models"):
            with pytest.raises(urllib.error.HTTPError) as removed:
                urllib.request.urlopen(
                    urllib.request.Request(f"{base}{path}", headers=headers), timeout=10
                )
            assert removed.value.code == 404
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)
