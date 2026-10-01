"""Display metadata shared by management and command launchers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class IntegrationDescriptor:
    display_name: str


INTEGRATIONS = {
    name: IntegrationDescriptor(label)
    for name, label in (
        ("claude", "Claude Code"),
        ("codex", "Codex"),
        ("codex_app", "Codex App"),
        ("opencode", "OpenCode"),
        ("openclaw", "OpenClaw"),
        ("hermes", "Hermes Agent"),
        ("pi", "Pi"),
        ("copilot", "Copilot CLI"),
    )
}
