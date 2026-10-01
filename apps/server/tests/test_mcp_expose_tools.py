# SPDX-License-Identifier: Apache-2.0
"""Tests for the MCP tool exposure setting used by the inference server."""

from types import SimpleNamespace

import pytest
from omlx_config.settings import GlobalSettings, MCPSettings
from omlx_server.server import create_app


@pytest.fixture
def controller():
    return create_app().state.controller


class TestMcpToolsExposedHelper:
    """Unit tests for ``omlx_server.server.mcp_tools_exposed``."""

    def test_true_when_global_settings_unavailable(self, monkeypatch, controller):
        """No global settings (e.g. MCP via env var) -> keep exposing."""
        monkeypatch.setattr(controller.state, "global_settings", None)
        assert controller.mcp_tools_exposed() is True

    def test_true_when_expose_tools_enabled(self, monkeypatch, controller):
        monkeypatch.setattr(
            controller.state,
            "global_settings",
            GlobalSettings(mcp=MCPSettings(expose_tools=True)),
        )
        assert controller.mcp_tools_exposed() is True

    def test_false_when_expose_tools_disabled(self, monkeypatch, controller):
        monkeypatch.setattr(
            controller.state,
            "global_settings",
            GlobalSettings(mcp=MCPSettings(expose_tools=False)),
        )
        assert controller.mcp_tools_exposed() is False

    def test_true_for_legacy_settings_without_flag(self, monkeypatch, controller):
        """A settings object without the attribute must default to True."""
        monkeypatch.setattr(
            controller.state,
            "global_settings",
            SimpleNamespace(mcp=SimpleNamespace()),
        )
        assert controller.mcp_tools_exposed() is True
