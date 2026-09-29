"""Tests for installation method detection."""

from unittest.mock import patch

from omlx.utils.install import (
    get_cli_command_prefix,
    get_cli_prefix,
    get_install_method,
    is_homebrew,
)


def test_cli_prefix_is_installed_command():
    assert get_cli_prefix() == "omlx"
    assert get_cli_command_prefix() == "omlx"


class TestIsHomebrew:
    def test_not_homebrew_in_dev(self):
        assert not is_homebrew()

    def test_cellar_prefix(self):
        with patch("omlx.utils.install.sys") as mock_sys:
            mock_sys.prefix = "/opt/homebrew/Cellar/omlx/0.3.0/libexec"
            assert is_homebrew()

    def test_homebrew_prefix(self):
        with patch("omlx.utils.install.sys") as mock_sys:
            mock_sys.prefix = "/usr/local/homebrew/opt/omlx/libexec"
            assert is_homebrew()

    def test_non_homebrew_prefix(self):
        with patch("omlx.utils.install.sys") as mock_sys:
            mock_sys.prefix = "/Users/me/.venv"
            assert not is_homebrew()


class TestGetInstallMethod:
    def test_homebrew_detected(self):
        with patch("omlx.utils.install.sys") as mock_sys:
            mock_sys.prefix = "/opt/homebrew/Cellar/omlx/0.3.0/libexec"
            assert get_install_method() == "homebrew"

    def test_pip_default(self):
        assert get_install_method() == "pip"
