"""Ordinary application startup must not install optional cluster helpers."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI

from omlx import server


@pytest.mark.asyncio
async def test_disabled_cluster_does_not_publish_an_interpreter(monkeypatch):
    publish = Mock()
    monkeypatch.setattr(server, "_server_state", server.ServerState())
    monkeypatch.setattr(server, "_reset_boundary_snapshots_for_server", lambda: None)
    monkeypatch.setattr(
        "omlx.cluster.launch.reap_orphaned_launches",
        lambda: {"reaped": [], "failures": []},
    )
    monkeypatch.setattr("omlx.cluster.worker_shim.ensure_cluster_python_shim", publish)
    monkeypatch.setattr(
        server, "get_server_metrics", lambda: SimpleNamespace(close=lambda: None)
    )
    monkeypatch.delenv("OMLX_MCP_CONFIG", raising=False)

    async with server.lifespan(FastAPI()):
        publish.assert_not_called()
    publish.assert_not_called()
