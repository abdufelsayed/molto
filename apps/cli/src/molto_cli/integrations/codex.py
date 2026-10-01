# SPDX-License-Identifier: Apache-2.0
"""Codex (OpenAI Codex CLI) integration."""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

from molto_config.utils.install import get_cli_command_prefix

from molto_cli.integrations.base import Integration, IntegrationContext

CODEX_CONFIG_PATH = Path.home() / ".codex" / "config.toml"


def write_codex_config(config_path: Path, ctx: IntegrationContext) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)

    existing_content = ""
    if config_path.exists():
        # Create backup
        timestamp = int(time.time())
        backup = config_path.with_suffix(f".{timestamp}.bak")
        try:
            shutil.copy2(config_path, backup)
            existing_content = config_path.read_text(encoding="utf-8")
            print(f"Backup: {backup}")
        except OSError as e:
            print(f"Warning: could not create backup or read config: {e}")

    # Parse existing config lines to preserve other settings
    lines = existing_content.splitlines()
    new_lines = []
    in_any_section = False
    in_molto_section = False

    # Keys to override at the top level
    top_level_overrides = {
        "model": f'"{ctx.model or "select-a-model"}"',
        "model_provider": '"molto"',
    }

    # If it is a reasoning model, add reasoning effort
    is_reasoning = (
        bool(ctx.reasoning)
        if ctx.reasoning is not None
        else bool(re.search(r"\b(thinking|o1|o3|r1)\b", ctx.model.lower()))
    )
    if is_reasoning:
        top_level_overrides["model_reasoning_effort"] = '"high"'

    # Keys managed by Molto that should be removed when not applicable
    managed_keys = {"model_reasoning_effort"} - set(top_level_overrides.keys())

    seen_keys = set()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_any_section = True
            in_molto_section = stripped == "[model_providers.molto]"

        # Handle top-level keys
        if not in_any_section and "=" in stripped:
            key = stripped.split("=")[0].strip()
            if key in top_level_overrides:
                new_lines.append(f"{key} = {top_level_overrides[key]}")
                seen_keys.add(key)
                continue
            if key in managed_keys:
                continue

        # Skip old Molto section
        if in_molto_section:
            continue

        new_lines.append(line)

    # Add missing top-level keys
    for key, val in top_level_overrides.items():
        if key not in seen_keys:
            new_lines.insert(0, f"{key} = {val}")

    # Append new Molto provider section
    new_lines.append("\n[model_providers.molto]")
    new_lines.append('name = "Molto"')
    new_lines.append(f'base_url = "{ctx.openai_base_url}"')
    new_lines.append('env_key = "MOLTO_API_KEY"')

    config_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"Config updated: {config_path}")


def codex_config_args(ctx: IntegrationContext) -> list[str]:
    """Build process-scoped Codex config overrides for a Molto launch."""
    overrides: list[tuple[str, str]] = [
        ("model_provider", json.dumps("molto")),
        ("model_providers.molto.name", json.dumps("Molto")),
        ("model_providers.molto.base_url", json.dumps(ctx.openai_base_url)),
        ("model_providers.molto.env_key", json.dumps("MOLTO_API_KEY")),
    ]
    if ctx.context_window is not None and ctx.context_window > 0:
        overrides.append(("model_context_window", str(ctx.context_window)))

    is_reasoning = (
        bool(ctx.reasoning)
        if ctx.reasoning is not None
        else bool(re.search(r"\b(thinking|o1|o3|r1)\b", ctx.model.lower()))
    )
    if is_reasoning:
        overrides.append(("model_reasoning_effort", json.dumps("high")))

    return [arg for key, value in overrides for arg in ("-c", f"{key}={value}")]


class CodexIntegration(Integration):
    """Codex integration using process-scoped configuration for Molto."""

    def __init__(self):
        super().__init__(
            name="codex",
            display_name="Codex",
            type="env_var",
            install_check="codex",
            install_hint="npm install -g @openai/codex",
        )

    def get_command(self, ctx: IntegrationContext) -> str:
        return (
            f"{get_cli_command_prefix()} "
            f"launch codex --model {ctx.model or 'select-a-model'}"
        )

    def configure(self, ctx: IntegrationContext) -> None:
        # Launch-time arguments carry the Molto settings. Keeping this a no-op
        # ensures normal Codex sessions continue to use the user's config.
        return None

    def launch(self, ctx: IntegrationContext) -> None:
        self.configure(ctx)

        env = self._scrubbed_env()
        env["MOLTO_API_KEY"] = ctx.auth_token

        args = ["codex", *codex_config_args(ctx)]
        if ctx.model:
            args.extend(["-m", ctx.model])
        args.extend(ctx.extra_args)

        os.execvpe("codex", args, env)
