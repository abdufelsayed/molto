"""Integration registry for external coding tools."""

from omlx_cli.integrations.base import Integration, IntegrationContext
from omlx_cli.integrations.claude import ClaudeCodeIntegration
from omlx_cli.integrations.codex import CodexIntegration
from omlx_cli.integrations.codex_app import CodexAppIntegration
from omlx_cli.integrations.copilot import CopilotIntegration
from omlx_cli.integrations.hermes import HermesIntegration
from omlx_cli.integrations.openclaw import OpenClawIntegration
from omlx_cli.integrations.opencode import OpenCodeIntegration
from omlx_cli.integrations.pi import PiIntegration

INTEGRATIONS: dict[str, Integration] = {
    "claude": ClaudeCodeIntegration(),
    "codex": CodexIntegration(),
    "codex_app": CodexAppIntegration(),
    "opencode": OpenCodeIntegration(),
    "openclaw": OpenClawIntegration(),
    "hermes": HermesIntegration(),
    "pi": PiIntegration(),
    "copilot": CopilotIntegration(),
}


def get_integration(name: str) -> Integration | None:
    """Get an integration by name."""
    return INTEGRATIONS.get(name)


def list_integrations() -> list[Integration]:
    """List all available integrations."""
    return list(INTEGRATIONS.values())


__all__ = [
    "Integration",
    "IntegrationContext",
    "ClaudeCodeIntegration",
    "CodexAppIntegration",
    "CopilotIntegration",
    "HermesIntegration",
    "INTEGRATIONS",
    "get_integration",
    "list_integrations",
]
