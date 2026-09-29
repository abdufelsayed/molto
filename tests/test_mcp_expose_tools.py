# SPDX-License-Identifier: Apache-2.0
"""Tests for the MCP tool exposure setting used by the inference server."""

from types import SimpleNamespace

import omlx.server as server
from omlx.settings import GlobalSettings, MCPSettings

class TestMcpToolsExposedHelper:
    """Unit tests for ``omlx.server.mcp_tools_exposed``."""

    def test_true_when_global_settings_unavailable(self, monkeypatch):
        """No global settings (e.g. MCP via env var) -> keep exposing."""
        monkeypatch.setattr(server._server_state, "global_settings", None)
        assert server.mcp_tools_exposed() is True

    def test_true_when_expose_tools_enabled(self, monkeypatch):
        monkeypatch.setattr(
            server._server_state,
            "global_settings",
            GlobalSettings(mcp=MCPSettings(expose_tools=True)),
        )
        assert server.mcp_tools_exposed() is True

    def test_false_when_expose_tools_disabled(self, monkeypatch):
        monkeypatch.setattr(
            server._server_state,
            "global_settings",
            GlobalSettings(mcp=MCPSettings(expose_tools=False)),
        )
        assert server.mcp_tools_exposed() is False

    def test_true_for_legacy_settings_without_flag(self, monkeypatch):
        """A settings object without the attribute must default to True."""
        monkeypatch.setattr(
            server._server_state,
            "global_settings",
            SimpleNamespace(mcp=SimpleNamespace()),
        )
        assert server.mcp_tools_exposed() is True
