"""Locate checkout resources independently of an owning test directory."""

from pathlib import Path


def repository_root(source: str) -> Path:
    for directory in Path(source).resolve().parents:
        if (directory / "pnpm-workspace.yaml").is_file() and (
            directory / "pyproject.toml"
        ).is_file():
            return directory
    raise RuntimeError(f"Cannot locate Molto workspace from {source}")
