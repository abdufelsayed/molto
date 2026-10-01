"""Ordinary application startup must not install optional cluster helpers."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from molto_server import server


@pytest.mark.asyncio
async def test_disabled_cluster_does_not_publish_an_interpreter(monkeypatch):
    publish = Mock()
    app = server.create_app()
    monkeypatch.setattr(
        app.state.controller, "_reset_boundary_snapshots_for_server", lambda: None
    )
    monkeypatch.setattr(
        "molto_runtime.cluster.launch.reap_orphaned_launches",
        lambda: {"reaped": [], "failures": []},
    )
    monkeypatch.setattr(
        "molto_runtime.cluster.worker_shim.ensure_cluster_python_shim", publish
    )
    app.state.server_state.metrics = SimpleNamespace(close=lambda: None)
    monkeypatch.delenv("MOLTO_MCP_CONFIG", raising=False)

    async with app.state.controller.lifespan(app):
        publish.assert_not_called()
    publish.assert_not_called()
