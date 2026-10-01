"""Declared engine provenance for workspace and bundled installations."""

from __future__ import annotations

import importlib.metadata
import json
import re
import tomllib
from importlib.resources import files
from pathlib import Path
from typing import Any

_GIT_REQUIREMENT = re.compile(
    r"^([\w.-]+)\s*@\s*git\+(https://[^@\s]+)@([0-9a-f]{7,40})(?:\s*;.*)?$"
)


def _declared(requirements: list[str]) -> dict[str, dict[str, str]]:
    result = {}
    for requirement in requirements:
        match = _GIT_REQUIREMENT.match(requirement)
        if match:
            name, url, commit = match.groups()
            result[name.lower().replace("_", "-")] = {"url": url, "commit": commit}
    return result


def _workspace_requirements(module_path: Path) -> list[str]:
    """Use only this package's pyproject inside the identified uv workspace."""
    package = module_path.parent.parent.parent
    workspace = package.parent.parent
    try:
        project = tomllib.loads((package / "pyproject.toml").read_text())
        root = tomllib.loads((workspace / "pyproject.toml").read_text())
        if project.get("project", {}).get("name") != "omlx-runtime":
            return []
        if not root.get("tool", {}).get("uv", {}).get("workspace"):
            return []
        if not (workspace / "pnpm-workspace.yaml").is_file():
            return []
        return project["project"].get("dependencies", [])
    except (OSError, ValueError, KeyError):
        return []


def version_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    """Return bundled commit records and declared Git dependency pins."""
    commits: dict[str, Any] = {}
    declared: dict[str, Any] = {}
    resource_root = files("omlx_runtime")
    for name in ("_engine_commits.json", "_version_sources.json"):
        try:
            payload = json.loads(resource_root.joinpath(name).read_text())
            if not isinstance(payload, dict):
                continue
            if name == "_engine_commits.json":
                commits.update(payload)
            else:
                commits.update(payload.get("commits", {}))
                declared.update(payload.get("declared_dependencies", {}))
        except (OSError, ValueError, TypeError):
            continue
    for distribution in ("omlx-runtime", "omlx"):
        try:
            requirements = importlib.metadata.requires(distribution) or []
            declared.update(_declared(requirements))
        except importlib.metadata.PackageNotFoundError:
            continue
    # Editable metadata can omit dependency declarations; source ownership is
    # validated before looking at checkout files. Installed wheels have no such marker.
    for name, source in _declared(_workspace_requirements(Path(__file__))).items():
        declared.setdefault(name, source)
    return commits, declared
