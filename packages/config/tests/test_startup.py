"""Persisted startup overrides and saved network authentication migrations."""

import argparse
import json
from unittest.mock import patch

import pytest
from molto_config.settings import GlobalSettings
from molto_config.startup import _has_cli_overrides, _migrate_saved_network_auth


class TestHasCliOverrides:
    """Tests for _has_cli_overrides() — detects explicitly passed CLI args."""

    @staticmethod
    def _make_args(**kwargs):
        """Namespace with all serve defaults (None), then apply overrides."""
        defaults = {
            "model_dir": None,
            "port": None,
            "host": None,
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
            "no_cache": False,
            "initial_cache_blocks": None,
            "mcp_config": None,
            "hf_endpoint": None,
            "hf_cache_enabled": None,
            "ms_endpoint": None,
            "http_proxy": None,
            "https_proxy": None,
            "no_proxy": None,
            "ca_bundle": None,
            "api_key": None,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_no_overrides_returns_false(self):

        assert _has_cli_overrides(self._make_args()) is False

    def test_host_explicit(self):

        assert _has_cli_overrides(self._make_args(host="0.0.0.0")) is True
        # Even the default value, when explicitly passed, counts as override
        assert _has_cli_overrides(self._make_args(host="127.0.0.1")) is True

    def test_port_explicit(self):

        assert _has_cli_overrides(self._make_args(port=9000)) is True
        assert _has_cli_overrides(self._make_args(port=8000)) is True

    def test_model_dir_explicit(self):

        assert _has_cli_overrides(self._make_args(model_dir="/tmp/models")) is True

    def test_log_level_explicit(self):

        assert _has_cli_overrides(self._make_args(log_level="info")) is True
        assert _has_cli_overrides(self._make_args(log_level="debug")) is True

    def test_embedding_batch_size_explicit(self):

        assert _has_cli_overrides(self._make_args(embedding_batch_size=4)) is True

    def test_memory_guard_explicit(self):

        assert _has_cli_overrides(self._make_args(memory_guard="safe")) is True

    def test_memory_guard_gb_explicit(self):

        assert _has_cli_overrides(self._make_args(memory_guard_gb=48.0)) is True

    def test_hf_cache_explicit(self):

        assert _has_cli_overrides(self._make_args(hf_cache_enabled=False)) is True
        assert _has_cli_overrides(self._make_args(hf_cache_enabled=True)) is True

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("sse_keepalive_mode", "off"),
            ("max_audio_upload_size", "250MB"),
            ("max_concurrent_requests", 2),
            ("paged_ssd_cache_dir", "/tmp/cache"),
            ("paged_ssd_cache_max_size", "2GB"),
            ("hot_cache_max_size", "1GB"),
            ("no_cache", True),
            ("initial_cache_blocks", 64),
        ],
    )
    def test_all_persisted_serve_flags_count_as_overrides(self, field, value):

        assert _has_cli_overrides(self._make_args(**{field: value})) is True

    def test_api_key_alone_is_not_persisted(self):
        """A command-line secret must not be written to settings.json."""

        assert _has_cli_overrides(self._make_args(api_key="test-key")) is False

    def test_multiple_overrides(self):

        assert _has_cli_overrides(self._make_args(host="0.0.0.0", port=9000)) is True

    def test_empty_namespace(self):

        assert _has_cli_overrides(argparse.Namespace()) is False


class TestSavedNetworkAuthMigration:
    @pytest.fixture(autouse=True)
    def setup_migration(self, tmp_path, monkeypatch):
        for name in ("MOLTO_HOST", "MOLTO_API_KEY", "MOLTO_STARTUP_NOTICE_PATH"):
            monkeypatch.delenv(name, raising=False)
        self.path = tmp_path / "settings.json"
        self.data = {
            "server": {"host": "0.0.0.0"},
            "auth": {"api_key": "existing-key", "skip_api_key_verification": True},
            "custom": {"preserve": True},
        }
        self.args = argparse.Namespace(host=None)
        with patch("builtins.input", return_value="") as self.prompt:
            yield

    def load(self):
        return GlobalSettings.load(base_path=str(self.path.parent))

    def write_settings(self):
        self.path.write_text(json.dumps(self.data))
        return self.path.read_bytes()

    @pytest.mark.parametrize("api_key,skip", [("existing-key", True), (None, False)])
    def test_saved_unsafe_host_is_migrated_once(self, capsys, api_key, skip):
        self.data["auth"].update(api_key=api_key, skip_api_key_verification=skip)
        self.write_settings()
        settings = self.load()
        _migrate_saved_network_auth(settings, self.args)
        _migrate_saved_network_auth(self.load(), self.args)
        self.prompt.assert_called_once()
        assert "Enter" in self.prompt.call_args.args[0]
        self.data["server"]["host"] = "127.0.0.1"
        assert json.loads(self.path.read_text()) == self.data
        assert settings.server.host == "127.0.0.1"
        assert (
            "API-key verification enabled in settings.json" in capsys.readouterr().out
        )

    @pytest.mark.parametrize(
        "case", ["cli", "env", "authenticated", "loopback", "invalid"]
    )
    def test_non_migration_cases_preserve_settings(self, monkeypatch, case):
        if case == "cli":
            self.args.host = "0.0.0.0"
        elif case == "env":
            monkeypatch.setenv("MOLTO_HOST", "0.0.0.0")
        elif case == "authenticated":
            self.data["auth"]["skip_api_key_verification"] = False
        elif case == "loopback":
            self.data["server"]["host"] = "127.0.0.1,::1"
        else:
            self.data["server"]["port"] = -1
        before = self.write_settings()
        settings = self.load()
        _migrate_saved_network_auth(settings, self.args)
        self.prompt.assert_not_called()
        assert self.path.read_bytes() == before
        assert settings.server.host == self.data["server"]["host"]

    def test_app_receives_notice_without_cli_prompt(self, tmp_path, monkeypatch):
        notice = tmp_path / "notice.txt"
        monkeypatch.setenv("MOLTO_STARTUP_NOTICE_PATH", str(notice))
        self.write_settings()
        _migrate_saved_network_auth(self.load(), self.args)
        self.prompt.assert_not_called()
        assert "127.0.0.1" in notice.read_text()

    @pytest.mark.parametrize("interruption", [EOFError, KeyboardInterrupt])
    def test_canceled_migration_preserves_settings(self, interruption):
        before = self.write_settings()
        self.prompt.side_effect = interruption
        with pytest.raises(SystemExit):
            _migrate_saved_network_auth(self.load(), self.args)
        assert self.path.read_bytes() == before

    def test_inference_opt_in_keeps_authenticated_network_bind(self):
        self.data["auth"].update(
            skip_api_key_verification=False, allow_unauthenticated_inference=True
        )
        before = self.write_settings()
        settings = self.load()
        _migrate_saved_network_auth(settings, self.args)
        self.prompt.assert_not_called()
        assert self.path.read_bytes() == before
        assert settings.server.host == "0.0.0.0"
        assert settings.validate() == []

    def test_migration_does_not_persist_runtime_secrets_or_cli_overrides(
        self, monkeypatch
    ):
        self.write_settings()
        monkeypatch.setenv("MOLTO_API_KEY", "runtime-secret")
        self.args.port = 9001
        settings = GlobalSettings.load(
            base_path=str(self.path.parent), cli_args=self.args
        )
        assert settings.auth.api_key == "runtime-secret"
        assert settings.server.port == 9001
        _migrate_saved_network_auth(settings, self.args)
        expected = dict(self.data)
        expected["server"] = {"host": "127.0.0.1"}
        assert json.loads(self.path.read_text()) == expected
        assert "runtime-secret" not in self.path.read_text()
