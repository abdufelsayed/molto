# SPDX-License-Identifier: Apache-2.0
"""
CLI tests for Molto.

Tests CLI argument parsing, command setup, and help text.
Note: Configuration validation tests are in test_config.py.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from molto_config._version import __version__
from molto_config.settings import GlobalSettings


class TestCLIModule:
    """Tests for CLI module existence and basic functionality."""

    def test_cli_module_importable(self):
        """Test that CLI module can be imported."""
        from molto_cli import cli

        assert hasattr(cli, "main")

    def test_cli_has_serve_command(self):
        """Test that CLI has serve command setup."""
        from molto_cli import cli

        # The module should have the main entry point
        assert callable(cli.main)


class TestCLIHelp:
    """Tests for CLI help functionality."""

    def test_main_help(self):
        """Test main CLI help output."""
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should succeed with help
        assert result.returncode == 0
        # Should show available commands
        assert "serve" in result.stdout.lower()

    def test_main_version(self):
        """Test main CLI version output."""
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert result.stdout.strip() == __version__
        assert result.stderr == ""

    def test_serve_help(self):
        """Test serve command help output."""
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should succeed with help
        assert result.returncode == 0
        # Should show serve options
        stdout_lower = result.stdout.lower()
        assert "host" in stdout_lower
        assert "port" in stdout_lower
        assert "model-dir" in stdout_lower

    def test_lifecycle_commands_in_main_help(self):
        """Homebrew service commands remain available."""
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        stdout_lower = result.stdout.lower()
        assert "start" in stdout_lower
        assert "stop" in stdout_lower
        assert "restart" in stdout_lower
        assert "diagnose" not in stdout_lower

    def test_start_help_has_readiness_and_local_options(self):
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli", "start", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "background" in result.stdout
        assert "--timeout" in result.stdout
        assert "--no-wait" in result.stdout
        assert "--base-path" in result.stdout


class TestLifecycleCommand:
    """CLI presentation is independent of the lifecycle manager."""

    @staticmethod
    def _args(command, **overrides):
        values = {"command": command}
        values.update(overrides)
        return SimpleNamespace(**values)

    @pytest.mark.parametrize("command", ["start", "stop", "restart", "status"])
    def test_lifecycle_delegates_to_owned_manager(self, command, monkeypatch, capsys):
        from molto_cli import cli, cli_lifecycle

        manager = MagicMock(return_value={"manager": "local", "state": "running"})
        monkeypatch.setattr(cli_lifecycle, "run", manager)
        args = self._args(command, json=True)
        assert cli.lifecycle_command(args) == 0
        manager.assert_called_once_with(args)
        output = capsys.readouterr().out
        assert json.loads(output)["state"] == "running"


class TestCLIEntryPoint:
    """Tests for CLI entry point functionality."""

    def test_module_runnable(self):
        """Test that CLI module is runnable."""
        # Should not crash when running with --help
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0

    def test_invalid_command_error(self):
        """Test error handling for invalid command."""
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli", "invalid_command"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should fail with non-zero exit code
        assert result.returncode != 0

    def test_no_command_shows_help(self):
        """Test that no command shows help."""
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should exit with non-zero (no command provided)
        assert result.returncode != 0


class TestServeCommandOptions:
    """Tests for serve command options via help output."""

    def test_serve_has_model_dir_option(self):
        """Test that serve command has --model-dir option."""
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--model-dir" in result.stdout

    def test_serve_has_no_max_memory_options(self):
        """The --max-model-memory and --max-process-memory CLI flags are removed."""
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--max-model-memory" not in result.stdout
        assert "--max-process-memory" not in result.stdout

    def test_serve_has_memory_guard_options(self):
        """Test that serve command exposes memory guard controls."""
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--memory-guard" in result.stdout
        assert "--memory-guard-gb" in result.stdout
        assert "safe" in result.stdout
        assert "balanced" in result.stdout
        assert "aggressive" in result.stdout

    def test_serve_no_model_specific_options(self):
        """Test that serve command does not have model-specific options (managed via admin page)."""
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # These options are now managed via admin page, not CLI
        assert "--pin" not in result.stdout
        assert "--default-model" not in result.stdout
        assert "--max-tokens" not in result.stdout
        assert "--temperature" not in result.stdout
        assert "--top-p" not in result.stdout
        assert "--top-k" not in result.stdout
        assert "--force-sampling" not in result.stdout

    def test_serve_has_host_port_options(self):
        """Test that serve command has --host and --port options."""
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--host" in result.stdout
        assert "--port" in result.stdout

    def test_serve_has_scheduler_options(self):
        """Test that serve command has scheduler options."""
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--max-concurrent-requests" in result.stdout
        assert "--embedding-batch-size" in result.stdout
        assert "--max-audio-upload-size" in result.stdout
        assert "settings.json" in result.stdout
        assert "Default: 100MB" not in result.stdout

    def test_serve_has_cache_options(self):
        """Test that serve command has cache options."""
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--paged-ssd-cache-dir" in result.stdout
        assert "--paged-ssd-cache-max-size" in result.stdout
        assert "--no-cache" in result.stdout

    def test_serve_has_mcp_option(self):
        """Test that serve command has --mcp-config option."""
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--mcp-config" in result.stdout

    def test_serve_has_base_path_option(self):
        """Test that serve command has --base-path option."""
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--base-path" in result.stdout

    def test_serve_has_api_key_option(self):
        """Test that serve command has --api-key option."""
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--api-key" in result.stdout


class TestLaunchCommandOptions:
    """Tests for launch command options via help output."""

    def test_launch_has_host_port_options(self):
        """Test that launch command has --host and --port options."""
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli", "launch", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "--host" in result.stdout
        assert "--port" in result.stdout

    def test_launch_has_model_option(self):
        """Test that launch command has --model option."""
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli", "launch", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--model" in result.stdout

    def test_launch_has_claude_tier_options(self):
        """Claude tier options should remain accepted for copied app commands."""
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli", "launch", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--opus" in result.stdout
        assert "--sonnet" in result.stdout
        assert "--haiku" in result.stdout

    def test_launch_has_cross_session_option(self):
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli", "launch", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert "--cross-session" in result.stdout

    def test_launch_lists_hermes(self):
        """Test that launch help lists Hermes as an available integration."""
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli", "launch", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "hermes" in result.stdout
        assert "Hermes Agent" in result.stdout

    def test_launch_lists_codex_app(self):
        """Test that launch help lists the Codex Desktop App target."""
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli", "launch", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "codex_app" in result.stdout
        assert "Codex App" in " ".join(result.stdout.split())


class TestLaunchCommandFunction:
    """Tests for launch command runtime behavior."""

    def test_launch_command_passes_model_type_to_integration(self):
        """VLM model metadata should be forwarded to integrations."""
        from molto_cli.cli import launch_command

        integration = MagicMock()
        integration.display_name = "OpenCode"
        integration.is_installed.return_value = True

        health_response = MagicMock()
        health_response.raise_for_status.return_value = None

        status_response = MagicMock()
        status_response.ok = True
        status_response.json.return_value = {
            "models": [
                {
                    "id": "qwen2.5-vl",
                    "model_type": "vlm",
                    "max_context_window": 32768,
                    "max_tokens": 8192,
                }
            ]
        }

        settings = MagicMock()
        settings.server.host = "127.0.0.1"
        settings.server.port = 8000

        args = argparse.Namespace(
            tool="opencode",
            host=None,
            port=None,
            api_key="test-key",
            model="qwen2.5-vl",
            tools_profile="coding",
        )

        with (
            patch("requests.get", side_effect=[health_response, status_response]),
            patch("molto_cli.integrations.get_integration", return_value=integration),
            patch("molto_config.settings.GlobalSettings.load", return_value=settings),
        ):
            launch_command(args)

        integration.launch.assert_called_once()
        ctx = integration.launch.call_args.args[0]
        assert ctx.host == "127.0.0.1"
        assert ctx.port == 8000
        assert ctx.api_key == "test-key"
        assert ctx.model == "qwen2.5-vl"
        assert ctx.tools_profile == "coding"
        assert ctx.context_window == 32768
        assert ctx.max_tokens == 8192
        assert ctx.cross_session is False
        assert ctx.model_type == "vlm"
        assert ctx.extra_args == ()

    def test_launch_command_passes_cross_session_flag_to_integration(self):
        from molto_cli.cli import launch_command

        integration = MagicMock()
        integration.display_name = "Claude Code"
        integration.is_installed.return_value = True

        health_response = MagicMock()
        health_response.raise_for_status.return_value = None

        status_response = MagicMock()
        status_response.ok = True
        status_response.json.return_value = {"models": []}

        settings = MagicMock()
        settings.server.host = "127.0.0.1"
        settings.server.port = 8000
        settings.claude_code = None

        args = argparse.Namespace(
            tool="claude",
            host=None,
            port=None,
            api_key="test-key",
            model="qwen3.5",
            tools_profile="coding",
            opus_model=None,
            sonnet_model=None,
            haiku_model=None,
            cross_session=True,
        )

        with (
            patch("requests.get", side_effect=[health_response, status_response]),
            patch("molto_cli.integrations.get_integration", return_value=integration),
            patch("molto_config.settings.GlobalSettings.load", return_value=settings),
        ):
            launch_command(args)

        ctx = integration.launch.call_args.args[0]
        assert ctx.cross_session is True

    def test_launch_command_resolves_alias_status_metadata(self):
        """Alias model IDs should keep status metadata from the real model."""
        from molto_cli.cli import launch_command

        integration = MagicMock()
        integration.display_name = "OpenCode"
        integration.is_installed.return_value = True

        health_response = MagicMock()
        health_response.raise_for_status.return_value = None

        status_response = MagicMock()
        status_response.ok = True
        status_response.json.return_value = {
            "models": [
                {
                    "id": "qwen2.5-vl-raw",
                    "model_alias": "gpt-4o",
                    "model_type": "vlm",
                    "max_context_window": 32768,
                    "max_tokens": 8192,
                    "enable_thinking": False,
                }
            ]
        }

        settings = MagicMock()
        settings.server.host = "127.0.0.1"
        settings.server.port = 8000

        args = argparse.Namespace(
            tool="opencode",
            host=None,
            port=None,
            api_key="test-key",
            model="gpt-4o",
            tools_profile="coding",
        )

        with (
            patch("requests.get", side_effect=[health_response, status_response]),
            patch("molto_cli.integrations.get_integration", return_value=integration),
            patch("molto_config.settings.GlobalSettings.load", return_value=settings),
        ):
            launch_command(args)

        ctx = integration.launch.call_args.args[0]
        assert ctx.model == "gpt-4o"
        assert ctx.context_window == 32768
        assert ctx.max_tokens == 8192
        assert ctx.model_type == "vlm"
        assert ctx.reasoning is False

    def test_launch_command_forwards_extra_args(self):
        """Unknown CLI tokens (e.g. --resume <id>) should reach integration.launch."""
        from molto_cli.cli import launch_command

        integration = MagicMock()
        integration.display_name = "Claude Code"
        integration.is_installed.return_value = True

        health_response = MagicMock()
        health_response.raise_for_status.return_value = None

        status_response = MagicMock()
        status_response.ok = True
        status_response.json.return_value = {
            "models": [
                {
                    "id": "qwen2.5-vl",
                    "model_type": "llm",
                    "max_context_window": 65536,
                    "max_tokens": 8192,
                }
            ]
        }

        settings = MagicMock()
        settings.server.host = "127.0.0.1"
        settings.server.port = 8000

        args = argparse.Namespace(
            tool="claude",
            host=None,
            port=None,
            api_key="test-key",
            model="qwen2.5-vl",
            tools_profile="coding",
        )

        with (
            patch("requests.get", side_effect=[health_response, status_response]),
            patch("molto_cli.integrations.get_integration", return_value=integration),
            patch("molto_config.settings.GlobalSettings.load", return_value=settings),
        ):
            launch_command(args, extra_args=["--resume", "abc123"])

        ctx = integration.launch.call_args.args[0]
        assert ctx.extra_args == ("--resume", "abc123")

    def test_launch_command_rejects_small_explicit_claude_model(self, capsys):
        """--model must not bypass Claude Code's minimum context check."""
        from molto_cli.cli import launch_command

        integration = MagicMock()
        integration.display_name = "Claude Code"
        integration.is_installed.return_value = True

        health_response = MagicMock()
        health_response.raise_for_status.return_value = None

        status_response = MagicMock()
        status_response.ok = True
        status_response.json.return_value = {
            "models": [
                {
                    "id": "qwen-32k",
                    "model_type": "llm",
                    "max_context_window": 32768,
                    "max_tokens": 8192,
                }
            ]
        }

        settings = MagicMock()
        settings.server.host = "127.0.0.1"
        settings.server.port = 8000

        args = argparse.Namespace(
            tool="claude",
            host=None,
            port=None,
            api_key="test-key",
            model="qwen-32k",
            tools_profile="coding",
            opus_model=None,
            sonnet_model=None,
            haiku_model=None,
        )

        with (
            patch("requests.get", side_effect=[health_response, status_response]),
            patch("molto_cli.integrations.get_integration", return_value=integration),
            patch("molto_config.settings.GlobalSettings.load", return_value=settings),
            pytest.raises(SystemExit) as exc,
        ):
            launch_command(args)

        assert exc.value.code == 1
        integration.launch.assert_not_called()
        output = capsys.readouterr().out
        assert "Cannot launch Claude Code with model 'qwen-32k'" in output
        assert "at least 48K" in output

    def test_launch_command_rejects_small_claude_tier_model(self, capsys):
        """Explicit tier flags must all satisfy the same context requirement."""
        from molto_cli.cli import launch_command

        integration = MagicMock()
        integration.display_name = "Claude Code"
        integration.is_installed.return_value = True

        health_response = MagicMock()
        health_response.raise_for_status.return_value = None

        status_response = MagicMock()
        status_response.ok = True
        status_response.json.return_value = {
            "models": [
                {"id": "opus-32k", "max_context_window": 32768},
                {"id": "sonnet-64k", "max_context_window": 65536},
                {"id": "haiku-64k", "max_context_window": 65536},
            ]
        }

        settings = SimpleNamespace(
            server=SimpleNamespace(host="127.0.0.1", port=8000),
            auth=SimpleNamespace(api_key="saved-key"),
            claude_code=SimpleNamespace(
                opus_model=None,
                sonnet_model=None,
                haiku_model=None,
            ),
        )

        args = argparse.Namespace(
            tool="claude",
            host=None,
            port=None,
            api_key=None,
            model=None,
            tools_profile="coding",
            opus_model="opus-32k",
            sonnet_model="sonnet-64k",
            haiku_model="haiku-64k",
        )

        with (
            patch("requests.get", side_effect=[health_response, status_response]),
            patch("molto_cli.integrations.get_integration", return_value=integration),
            patch("molto_config.settings.GlobalSettings.load", return_value=settings),
            pytest.raises(SystemExit) as exc,
        ):
            launch_command(args)

        assert exc.value.code == 1
        integration.launch.assert_not_called()
        output = capsys.readouterr().out
        assert "Opus tier model 'opus-32k'" in output
        assert "at least 48K" in output

    def test_launch_command_rejects_small_auto_selected_claude_model(self, capsys):
        """A single available model must not bypass the minimum context check."""
        from molto_cli.cli import launch_command

        integration = MagicMock()
        integration.display_name = "Claude Code"
        integration.is_installed.return_value = True

        health_response = MagicMock()
        health_response.raise_for_status.return_value = None

        status_response = MagicMock()
        status_response.ok = True
        status_response.json.return_value = {
            "models": [{"id": "only-32k", "max_context_window": 32768}]
        }

        models_response = MagicMock()
        models_response.raise_for_status.return_value = None
        models_response.json.return_value = {
            "data": [{"id": "only-32k", "model_type": "llm"}]
        }

        settings = SimpleNamespace(
            server=SimpleNamespace(host="127.0.0.1", port=8000),
            auth=SimpleNamespace(api_key="saved-key"),
            claude_code=SimpleNamespace(
                opus_model=None,
                sonnet_model=None,
                haiku_model=None,
            ),
        )

        args = argparse.Namespace(
            tool="claude",
            host=None,
            port=None,
            api_key=None,
            model=None,
            tools_profile="coding",
            opus_model=None,
            sonnet_model=None,
            haiku_model=None,
        )

        with (
            patch(
                "requests.get",
                side_effect=[health_response, status_response, models_response],
            ),
            patch("molto_cli.integrations.get_integration", return_value=integration),
            patch("molto_config.settings.GlobalSettings.load", return_value=settings),
            pytest.raises(SystemExit) as exc,
        ):
            launch_command(args)

        assert exc.value.code == 1
        integration.launch.assert_not_called()
        output = capsys.readouterr().out
        assert "Using model: only-32k" in output
        assert "Cannot launch Claude Code with model 'only-32k'" in output

    def test_launch_command_shows_picker_and_keeps_saved_tiers(self):
        """Bare `molto launch claude` shows the picker for the default model and keeps the saved tier models (#3543)."""
        from molto_cli.cli import launch_command

        integration = MagicMock()
        integration.display_name = "Claude Code"
        integration.is_installed.return_value = True
        integration.select_model.return_value = "sonnet-local"

        health_response = MagicMock()
        health_response.raise_for_status.return_value = None

        status_map_response = MagicMock()
        status_map_response.ok = True
        status_map_response.json.return_value = {"models": []}

        models_response = MagicMock()
        models_response.raise_for_status.return_value = None
        models_response.json.return_value = {
            "data": [
                {"id": "sonnet-local", "model_type": "llm"},
                {"id": "opus-local", "model_type": "llm"},
            ]
        }

        settings = SimpleNamespace(
            server=SimpleNamespace(host="127.0.0.1", port=8000),
            auth=SimpleNamespace(api_key="saved-key"),
            claude_code=SimpleNamespace(
                opus_model="opus-local",
                sonnet_model="sonnet-local",
                haiku_model="haiku-local",
            ),
        )

        args = argparse.Namespace(
            tool="claude",
            host=None,
            port=None,
            api_key=None,
            model=None,
            tools_profile="coding",
            opus_model=None,
            sonnet_model=None,
            haiku_model=None,
        )

        with (
            patch(
                "requests.get",
                side_effect=[health_response, status_map_response, models_response],
            ),
            patch("molto_cli.integrations.get_integration", return_value=integration),
            patch("molto_config.settings.GlobalSettings.load", return_value=settings),
        ):
            launch_command(args)

        integration.select_model.assert_called_once()
        ctx = integration.launch.call_args.args[0]
        assert ctx.model == "sonnet-local"
        assert ctx.opus_model == "opus-local"
        assert ctx.sonnet_model == "sonnet-local"
        assert ctx.haiku_model == "haiku-local"
        assert ctx.api_key == "saved-key"

    def test_launch_command_claude_cli_tiers_override_saved_settings(self):
        """Explicit --opus/--sonnet/--haiku should win over saved settings."""
        from molto_cli.cli import launch_command

        integration = MagicMock()
        integration.display_name = "Claude Code"
        integration.is_installed.return_value = True

        health_response = MagicMock()
        health_response.raise_for_status.return_value = None

        status_response = MagicMock()
        status_response.ok = True
        status_response.json.return_value = {"models": []}

        settings = SimpleNamespace(
            server=SimpleNamespace(host="127.0.0.1", port=8000),
            auth=SimpleNamespace(api_key="saved-key"),
            claude_code=SimpleNamespace(
                opus_model="saved-opus",
                sonnet_model="saved-sonnet",
                haiku_model="saved-haiku",
            ),
        )

        args = argparse.Namespace(
            tool="claude",
            host=None,
            port=None,
            api_key=None,
            model=None,
            tools_profile="coding",
            opus_model="cli-opus",
            sonnet_model="cli-sonnet",
            haiku_model="cli-haiku",
        )

        with (
            patch("requests.get", side_effect=[health_response, status_response]),
            patch("molto_cli.integrations.get_integration", return_value=integration),
            patch("molto_config.settings.GlobalSettings.load", return_value=settings),
        ):
            launch_command(args)

        ctx = integration.launch.call_args.args[0]
        assert ctx.model == "cli-sonnet"
        assert ctx.opus_model == "cli-opus"
        assert ctx.sonnet_model == "cli-sonnet"
        assert ctx.haiku_model == "cli-haiku"


class TestLaunchArgvParsing:
    """Tests for top-level argv parsing of `molto launch ...`."""

    def test_launch_removes_forwarding_separator_after_known_option(self, monkeypatch):
        """The Molto separator must not reach the launched tool."""
        from molto_cli import cli

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "molto",
                "launch",
                "claude",
                "--cross-session",
                "--",
                "--allow-dangerously-skip-permissions",
            ],
        )
        with patch.object(cli, "launch_command") as launch:
            cli.main()

        args = launch.call_args.args[0]
        assert args.cross_session is True
        assert launch.call_args.kwargs["extra_args"] == [
            "--allow-dangerously-skip-permissions"
        ]

    def test_launch_preserves_separator_intended_for_tool(self, monkeypatch):
        """A second separator belongs to the launched tool's argv."""
        from molto_cli import cli

        monkeypatch.setattr(
            sys,
            "argv",
            ["molto", "launch", "claude", "--", "--", "--literal-prompt"],
        )
        with patch.object(cli, "launch_command") as launch:
            cli.main()

        assert launch.call_args.kwargs["extra_args"] == ["--", "--literal-prompt"]

    def test_serve_still_rejects_unknown_args(self):
        """Non-launch commands must keep strict argparse rejection."""
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli", "serve", "--bogus-flag"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0
        assert (
            "unrecognized arguments" in result.stderr or "--bogus-flag" in result.stderr
        )


class TestServeCommandFunctions:
    """Tests for serve command function."""

    @staticmethod
    def _make_serve_args(tmp_path, host="127.0.0.1", port=8000, **overrides):
        defaults = {
            "model_dir": None,
            "host": host,
            "port": port,
            "log_level": None,
            "sse_keepalive_mode": None,
            "max_audio_upload_size": None,
            "max_concurrent_requests": None,
            "embedding_batch_size": None,
            "memory_guard": None,
            "memory_guard_gb": None,
            "paged_ssd_cache_dir": None,
            "paged_ssd_cache_max_size": None,
            "hot_cache_max_size": None,
            "no_cache": True,
            "initial_cache_blocks": None,
            "mcp_config": None,
            "hf_endpoint": None,
            "hf_cache_enabled": None,
            "ms_endpoint": None,
            "http_proxy": None,
            "https_proxy": None,
            "no_proxy": None,
            "ca_bundle": None,
            "base_path": str(tmp_path),
            "api_key": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    @staticmethod
    def _make_settings(tmp_path, host="127.0.0.1", port=8000):
        log_dir = tmp_path / "logs"
        settings = SimpleNamespace()
        settings.base_path = tmp_path
        settings.server = SimpleNamespace(
            host=host, port=port, log_level="info", burst_decode_mode="balanced"
        )
        settings.huggingface = SimpleNamespace(endpoint=None, hf_cache_enabled=True)
        settings.modelscope = SimpleNamespace(endpoint=None)
        settings.network = SimpleNamespace(
            http_proxy=None,
            https_proxy=None,
            no_proxy=None,
            ca_bundle=None,
        )
        settings.logging = SimpleNamespace(
            retention_days=7,
            get_log_dir=lambda base_path: log_dir,
        )
        settings.model = SimpleNamespace(
            get_model_dirs=lambda base_path: [tmp_path / "models"],
        )
        settings.get_effective_model_dirs = lambda: [tmp_path / "models"]
        settings.memory = SimpleNamespace(
            memory_guard_tier="balanced", prefill_memory_guard=True
        )
        settings.mcp = SimpleNamespace(config_path=None)
        settings.cache = SimpleNamespace(
            enabled=False,
            ane_compile_cache=False,
            get_ssd_cache_dir=lambda base_path: tmp_path / "cache",
            get_ssd_cache_max_size_bytes=lambda base_path: 0,
            get_hot_cache_max_size_bytes=lambda: 0,
        )
        settings.auth = SimpleNamespace(api_key=None)
        settings.ensure_directories = lambda: log_dir.mkdir(parents=True, exist_ok=True)
        settings.validate = lambda: []
        settings.save = MagicMock()
        settings.save_cli_overrides = MagicMock()
        settings.to_scheduler_config = lambda: SimpleNamespace(
            paged_ssd_cache_dir=None,
            paged_ssd_cache_max_size=0,
            hot_cache_max_size=0,
        )
        return settings

    @staticmethod
    def _reserve_port(host="127.0.0.1"):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, 0))
        sock.listen(1)
        return sock

    def test_serve_command_exists(self):
        """Test that serve_command function exists."""
        from molto_cli.cli import serve_command

        assert callable(serve_command)

    def test_serve_model_dir_optional_with_default(self):
        """Test that serve --model-dir is optional with default ~/.molto/models."""
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should show that model-dir has a default
        assert "default" in result.stdout.lower()
        # Help text should mention ~/.molto/models or similar
        assert ".molto" in result.stdout or "model" in result.stdout.lower()

    def test_invalid_embedding_batch_size_is_not_persisted(self, tmp_path):
        """Invalid CLI scheduler values should fail before saving settings.json."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "molto_cli.cli",
                "serve",
                "--base-path",
                str(tmp_path),
                "--embedding-batch-size",
                "0",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode != 0
        assert "embedding_batch_size" in result.stdout
        assert not (tmp_path / "settings.json").exists()

    def test_network_bind_without_api_key_exits_before_persisting(self, tmp_path):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "molto_cli.cli",
                "serve",
                "--base-path",
                str(tmp_path),
                "--host",
                "0.0.0.0",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode != 0
        assert "API key is required" in result.stdout
        assert not (tmp_path / "settings.json").exists()

    def test_invalid_memory_guard_gb_is_not_persisted(self, tmp_path):
        """Invalid custom memory guard values should fail before saving settings.json."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "molto_cli.cli",
                "serve",
                "--base-path",
                str(tmp_path),
                "--memory-guard-gb",
                "0",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode != 0
        assert "--memory-guard-gb" in result.stderr
        assert not (tmp_path / "settings.json").exists()

    def test_non_finite_memory_guard_gb_is_rejected(self, tmp_path):
        """NaN and infinity must not be accepted as custom memory ceilings."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "molto_cli.cli",
                "serve",
                "--base-path",
                str(tmp_path),
                "--memory-guard-gb",
                "nan",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode != 0
        assert "finite number" in result.stderr
        assert not (tmp_path / "settings.json").exists()

    def test_serve_exits_on_port_conflict_before_importing_server(
        self, tmp_path, monkeypatch
    ):
        """Port conflicts should fail before server import can preload pinned models."""
        from molto_cli.cli import serve_command

        listener = self._reserve_port()
        host, port = listener.getsockname()
        settings = self._make_settings(tmp_path, host=host, port=port)
        args = self._make_serve_args(tmp_path, host=host, port=port)
        previous_server = sys.modules.pop("molto_server.server", None)
        events = []

        from molto_cli import application

        original_bind = application.bind_public

        def tracking_bind(host, port):
            events.append("bind")
            return original_bind(host, port)

        monkeypatch.delenv("MOLTO_INTERNAL_FD", raising=False)
        monkeypatch.setattr(
            "molto_config.settings.init_settings", lambda **kwargs: settings
        )
        monkeypatch.setattr(
            application, "dashboard_command", lambda: ["node", "index.mjs"]
        )
        monkeypatch.setattr(application, "bind_public", tracking_bind)
        try:
            with pytest.raises(SystemExit) as exc:
                serve_command(args)

            assert exc.value.code != 0
            assert events == ["bind"]
            settings.save_cli_overrides.assert_not_called()
            settings.save.assert_not_called()
            assert "molto_server.server" not in sys.modules
        finally:
            listener.close()
            if previous_server is not None:
                sys.modules["molto_server.server"] = previous_server

    @pytest.mark.parametrize("restarted", [False, True])
    def test_serve_hands_prebound_socket_to_uvicorn(
        self, tmp_path, monkeypatch, restarted
    ):
        """Successful serve startup should pass the pre-bound socket into uvicorn."""
        import molto
        from molto_cli.cli import serve_command

        host, port = "127.0.0.1", 0
        settings = self._make_settings(tmp_path, host=host, port=port)
        args = self._make_serve_args(tmp_path, host=host, port=8000)
        saved = tmp_path / "settings.json"
        saved.write_text(
            json.dumps(
                {
                    "server": {"host": host, "port": 9000},
                    "auth": {"api_key": "saved-key"},
                }
            )
        )
        persistent = GlobalSettings.load(base_path=str(tmp_path), cli_args=args)
        persistent.auth.api_key = "runtime-secret"
        settings.save_cli_overrides = MagicMock(wraps=persistent.save_cli_overrides)
        if restarted:
            monkeypatch.setenv("MOLTO_BACKEND_RESTART", "1")
        else:
            monkeypatch.delenv("MOLTO_BACKEND_RESTART", raising=False)
        monkeypatch.setenv("FORWARDED_ALLOW_IPS", "*")
        events = []

        fake_server = ModuleType("molto_server.server")

        async def app(scope, receive, send):
            return None

        def fake_init_server(**kwargs):
            events.append("init")

        fake_server.app = app
        initialize = MagicMock(side_effect=fake_init_server)
        app.state = SimpleNamespace(controller=SimpleNamespace(initialize=initialize))
        fake_server.create_app = lambda: app
        adapter = ModuleType("molto_runtime.settings_adapter")
        adapter.scheduler_config = lambda value: value.to_scheduler_config()
        monkeypatch.setitem(sys.modules, "molto_runtime.settings_adapter", adapter)
        monkeypatch.setitem(sys.modules, "molto_server.server", fake_server)
        monkeypatch.setattr(molto, "server", fake_server, raising=False)

        fake_mlx = ModuleType("mlx")
        fake_mlx_core = ModuleType("mlx.core")
        fake_mlx_core.device_info = lambda: {"memory_size": 0}
        fake_mlx_core.set_cache_limit = MagicMock()
        fake_mlx.core = fake_mlx_core
        monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
        monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx_core)

        monkeypatch.setattr(
            "molto_config.settings.init_settings", lambda **kwargs: settings
        )
        monkeypatch.setattr(
            "molto_config.logging_config.configure_file_logging",
            lambda **kwargs: None,
        )
        monkeypatch.setattr("faulthandler.enable", lambda *args, **kwargs: None)
        captured = {}
        listener = socket.socket()
        listener.bind((host, port))
        listener.listen(128)
        monkeypatch.setenv("MOLTO_INTERNAL_FD", str(os.dup(listener.fileno())))

        def fake_run(self, sockets=None):
            self.config.load()
            events.append("run")
            captured["socket_name"] = sockets[0].getsockname()
            captured["socket_count"] = len(sockets)
            captured["proxy_headers"] = self.config.proxy_headers
            captured["forwarded_allow_ips"] = self.config.forwarded_allow_ips

        monkeypatch.setattr("uvicorn.Server.run", fake_run)

        try:
            serve_command(args)
        finally:
            listener.close()

        initialize.assert_called_once()
        assert events == ["init", "run"]
        assert captured["socket_count"] == 1
        assert captured["socket_name"][0] == host
        assert captured["socket_name"][1] > 0
        assert captured["proxy_headers"] is True
        assert captured["forwarded_allow_ips"] == "127.0.0.1,::1"
        if restarted:
            settings.save_cli_overrides.assert_not_called()
            assert json.loads(saved.read_text())["server"]["port"] == 9000
        else:
            settings.save_cli_overrides.assert_called_once_with(args)
            assert json.loads(saved.read_text())["server"]["port"] == 8000
        assert json.loads(saved.read_text())["auth"]["api_key"] == "saved-key"


class TestCLIDocstrings:
    """Tests for CLI module docstrings and descriptions."""

    def test_main_has_description(self):
        """Test that main help has description."""
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should have some description
        assert "molto" in result.stdout.lower() or "llm" in result.stdout.lower()

    def test_serve_has_description(self):
        """Test that serve command has description."""
        result = subprocess.run(
            [sys.executable, "-m", "molto_cli.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Should describe multi-model serving
        assert (
            "multi-model" in result.stdout.lower() or "server" in result.stdout.lower()
        )


class TestLaunchClaudeTierPrecedence:
    def _run(
        self, *, args_model, settings_tiers, cli_tiers=None, picked="picked-model"
    ):
        from molto_cli.cli import launch_command

        integration = MagicMock()
        integration.display_name = "Claude Code"
        integration.is_installed.return_value = True
        integration.select_model.return_value = picked

        health_response = MagicMock()
        health_response.raise_for_status.return_value = None
        status_response = MagicMock()
        status_response.ok = True
        status_response.json.return_value = {
            "models": [
                {"id": m, "max_context_window": 131072}
                for m in (
                    "picked-model",
                    "other-model",
                    "opus-cfg",
                    "sonnet-cfg",
                    "haiku-cfg",
                    "opus-flag",
                )
            ]
        }

        # Third request: the interactive path lists /v1/models before the picker.
        models_response = MagicMock()
        models_response.raise_for_status.return_value = None
        models_response.json.return_value = {
            "data": [
                {"id": m["id"], "model_type": "llm"}
                for m in status_response.json.return_value["models"]
            ]
        }

        settings = SimpleNamespace(
            server=SimpleNamespace(host="127.0.0.1", port=8000),
            auth=SimpleNamespace(api_key="saved-key"),
            claude_code=SimpleNamespace(**settings_tiers),
        )
        cli_tiers = cli_tiers or {}
        args = argparse.Namespace(
            tool="claude",
            host=None,
            port=None,
            api_key=None,
            model=args_model,
            tools_profile="coding",
            opus_model=cli_tiers.get("opus_model"),
            sonnet_model=cli_tiers.get("sonnet_model"),
            haiku_model=cli_tiers.get("haiku_model"),
        )
        with (
            patch(
                "requests.get",
                side_effect=[health_response, status_response, models_response],
            ),
            patch("molto_cli.integrations.get_integration", return_value=integration),
            patch("molto_config.settings.GlobalSettings.load", return_value=settings),
        ):
            launch_command(args)
        integration.launch.assert_called_once()
        return integration.launch.call_args.args[0]

    def test_interactive_pick_keeps_saved_tier_models(self):
        """The picker chooses the default model; the persisted tiers keep their roles (#3543)."""
        ctx = self._run(
            args_model=None,
            settings_tiers={
                "opus_model": "opus-cfg",
                "sonnet_model": "sonnet-cfg",
                "haiku_model": "haiku-cfg",
            },
        )
        assert ctx.model == "picked-model"
        assert (ctx.opus_model, ctx.sonnet_model, ctx.haiku_model) == (
            "opus-cfg",
            "sonnet-cfg",
            "haiku-cfg",
        )

    def test_explicit_tier_flag_overrides_saved_setting(self):
        ctx = self._run(
            args_model="picked-model",
            settings_tiers={
                "opus_model": "opus-cfg",
                "sonnet_model": "sonnet-cfg",
                "haiku_model": "haiku-cfg",
            },
            cli_tiers={"opus_model": "opus-flag"},
        )
        assert ctx.opus_model == "opus-flag"
        assert (ctx.sonnet_model, ctx.haiku_model) == ("sonnet-cfg", "haiku-cfg")

    def test_without_saved_tiers_the_picked_model_is_used(self):
        ctx = self._run(
            args_model=None,
            settings_tiers={
                "opus_model": None,
                "sonnet_model": None,
                "haiku_model": None,
            },
        )
        assert ctx.model == "picked-model"
        assert (ctx.opus_model, ctx.sonnet_model, ctx.haiku_model) == (None, None, None)

    @pytest.mark.parametrize(
        "windows, expected_window",
        [
            ((131072, 49152, 65536, 65536), "49152"),
            ((131072, 65536, 49152, 65536), "49152"),
            ((131072, 65536, 65536, 49152), "49152"),
            ((49152, 131072, 131072, 131072), "49152"),
            ((131072, 131072, 131072, 131072), "131072"),
            ((131072, None, None, None), "131072"),
            ((None, 49152, 65536, 65536), "49152"),
            ((None, None, None, None), None),
        ],
    )
    def test_launch_passes_initial_model_and_shared_context_limit(
        self, windows, expected_window
    ):
        from molto_cli.cli import launch_command
        from molto_cli.integrations.claude import ClaudeCodeIntegration

        integration = ClaudeCodeIntegration()
        model_ids = ("picked-model", "opus-cfg", "sonnet-cfg", "haiku-cfg")
        models = [
            {"id": model_id, "max_context_window": window, "model_type": "llm"}
            for model_id, window in zip(model_ids, windows)
        ]
        responses = [MagicMock(), MagicMock(), MagicMock()]
        responses[1].json.return_value = {"models": models}
        responses[2].json.return_value = {"data": models}
        settings = SimpleNamespace(
            server=SimpleNamespace(host="127.0.0.1", port=8000),
            auth=SimpleNamespace(api_key="saved-key"),
            claude_code=SimpleNamespace(
                opus_model="opus-cfg",
                sonnet_model="sonnet-cfg",
                haiku_model="haiku-cfg",
            ),
        )
        args = argparse.Namespace(
            tool="claude", host=None, port=None, api_key=None, model=None
        )
        with (
            patch("requests.get", side_effect=responses),
            patch("molto_config.settings.GlobalSettings.load", return_value=settings),
            patch("molto_cli.integrations.get_integration", return_value=integration),
            patch.object(integration, "is_installed", return_value=True),
            patch.object(integration, "select_model", return_value="picked-model"),
            patch.object(integration, "_find_claude_binary", return_value="claude"),
            patch.dict("os.environ", {"ANTHROPIC_MODEL": "old-model"}, clear=True),
            patch("molto_cli.integrations.claude.os.execvpe") as execute,
        ):
            launch_command(args, extra_args=["--resume", "session-id"])

        execute.assert_called_once()
        binary, argv, env = execute.call_args.args
        assert binary == "claude"
        assert argv == [
            "claude",
            "--disallowedTools",
            "LSP",
            "--settings",
            '{"useAutoModeDuringPlan":false}',
            "--resume",
            "session-id",
        ]
        assert env["ANTHROPIC_MODEL"] == "picked-model"
        assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "opus-cfg"
        assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "sonnet-cfg"
        assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "haiku-cfg"
        assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "haiku-cfg"
        assert env.get("CLAUDE_CODE_MAX_CONTEXT_TOKENS") == expected_window
        assert env.get("CLAUDE_CODE_AUTO_COMPACT_WINDOW") == expected_window


def test_dashboard_dev_is_parsed_and_not_persisted(monkeypatch):
    from molto_cli import cli

    observed = []
    monkeypatch.setattr(sys, "argv", ["molto", "serve", "--dashboard-dev"])
    monkeypatch.setattr(cli, "serve_command", lambda args: observed.append(args))
    cli.main()
    assert observed[0].dashboard_dev is True
    from molto_config.startup import _has_cli_overrides

    assert _has_cli_overrides(observed[0]) is False
