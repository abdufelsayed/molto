"""Installation method detection and CLI command display."""

import shlex
import sys


def is_homebrew() -> bool:
    """Return True if running inside a Homebrew-installed virtualenv."""
    prefix = sys.prefix
    return "/Cellar/" in prefix or "/homebrew/" in prefix


def get_install_method() -> str:
    """Return the installation method: 'homebrew' or 'pip'."""
    return "homebrew" if is_homebrew() else "pip"


def get_cli_prefix() -> str:
    """Return the installed CLI command for display."""
    return "omlx"


def get_cli_command_prefix() -> str:
    """Return a shell-safe CLI command prefix for display/copy-paste."""
    return shlex.quote(get_cli_prefix())
