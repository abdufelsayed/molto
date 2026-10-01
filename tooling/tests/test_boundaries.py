"""Architectural checks reject violations even in lazy imports and metadata."""

import pytest

from tooling.check_boundaries import PROJECTS, ROOT, violations


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.0"\n')
    for namespace, project in PROJECTS.items():
        member = tmp_path / project
        (member / "src" / namespace).mkdir(parents=True)
        (member / "pyproject.toml").write_text(
            f'[project]\nname = "{namespace.replace("_", "-")}"\n'
            'version = "1.0"\ndependencies = []\n'
        )
    return tmp_path


def source(workspace, namespace, contents):
    path = workspace / PROJECTS[namespace] / "src" / namespace / "example.py"
    path.write_text(contents)


def test_empty_workspace_passes(workspace):
    assert violations(workspace) == []


def test_current_workspace_passes():
    assert violations(ROOT) == []


def test_lazy_config_import_cannot_reach_runtime(workspace):
    source(
        workspace, "molto_config", "def read():\n    import molto_runtime.engine_pool\n"
    )
    assert any(
        "molto_config cannot import molto_runtime" in error
        for error in violations(workspace)
    )


def test_allowed_import_requires_declared_member_dependency(workspace):
    source(
        workspace, "molto_runtime", "from molto_contracts.runtime import ModelView\n"
    )
    assert any(
        "undeclared dependency molto-contracts" in error
        for error in violations(workspace)
    )


def test_forbidden_dependency_is_rejected_without_an_import(workspace):
    path = workspace / PROJECTS["molto_config"] / "pyproject.toml"
    path.write_text(
        path.read_text().replace(
            "dependencies = []", 'dependencies = ["molto-server>=1"]'
        )
    )
    assert any(
        "forbidden dependency molto-server" in error for error in violations(workspace)
    )


def test_runtime_cannot_own_http_framework(workspace):
    source(workspace, "molto_runtime", "from fastapi import HTTPException\n")
    assert any(
        "runtime imports HTTP framework" in error for error in violations(workspace)
    )


def test_member_versions_cannot_drift(workspace):
    path = workspace / PROJECTS["molto_cli"] / "pyproject.toml"
    path.write_text(path.read_text().replace('version = "1.0"', 'version = "2.0"'))
    assert any(
        "version differs from workspace" in error for error in violations(workspace)
    )
