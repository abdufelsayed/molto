"""Release staging tests use tiny synthetic archives, never a model or Node download."""

import hashlib
import importlib.util
import io
import os
import subprocess
import sys
import tarfile
import zipfile

import pytest
from repo_paths import repository_root

ROOT = repository_root(__file__)
spec = importlib.util.spec_from_file_location(
    "dashboard_bundle", ROOT / "tooling/release/build_dashboard_bundle.py"
)
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


def fixture_bundle(tmp_path):
    output = tmp_path / "output"
    (output / "server").mkdir(parents=True)
    (output / "server/index.mjs").write_text("console.log('dashboard')")
    (output / "public/assets").mkdir(parents=True)
    (output / "public/assets/app.js").write_text("asset")
    archive = tmp_path / "node.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, value in (
            ("bin/node", b"#!/bin/sh\necho v24.21.0\n"),
            ("LICENSE", b"Node license"),
        ):
            member = tarfile.TarInfo(f"node-v24.21.0-darwin-arm64/{name}")
            member.size = len(value)
            tar.addfile(member, io.BytesIO(value))
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    return output, archive, checksum


def test_stage_includes_assets_runtime_and_license(tmp_path):
    output, archive, checksum = fixture_bundle(tmp_path)
    destination = tmp_path / "omlx/_dashboard"
    bundle.stage_bundle(output, archive, checksum, "24.21.0", destination)
    assert (destination / "public/assets/app.js").read_text() == "asset"
    assert (destination / "runtime/LICENSE").read_text() == "Node license"
    assert os.access(destination / "runtime/node", os.X_OK)


def test_checksum_failure_preserves_previous_bundle(tmp_path):
    output, archive, _ = fixture_bundle(tmp_path)
    destination = tmp_path / "bundle"
    destination.mkdir()
    (destination / "keep").write_text("old")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        bundle.stage_bundle(output, archive, "0" * 64, "24.21.0", destination)
    assert (destination / "keep").read_text() == "old"


def test_minimal_wheel_includes_bundle_with_executable_mode(tmp_path):
    # Build an isolated tiny package with the production packaging declarations.
    # This deliberately does not build or install the active oMLX checkout.
    output, archive, checksum = fixture_bundle(tmp_path)
    project = tmp_path / "project"
    package = project / "omlx"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    bundle.stage_bundle(output, archive, checksum, "24.21.0", package / "_dashboard")
    build_spec = importlib.util.spec_from_file_location(
        "release_build", ROOT / "tooling/release/build.py"
    )
    release = importlib.util.module_from_spec(build_spec)
    build_spec.loader.exec_module(release)
    # Use the real release assembler with only one synthetic workspace member.
    workspace = tmp_path / "workspace"
    (workspace / "apps/cli/src").mkdir(parents=True)
    (workspace / "apps/server/src").mkdir(parents=True)
    runtime = workspace / "packages/runtime/src/omlx_runtime"
    runtime.mkdir(parents=True)
    (runtime / "__init__.py").write_text("")
    (workspace / "packages/runtime/pyproject.toml").write_text(
        '[project]\nname="omlx-runtime"\nversion="1"\n'
        'dependencies=["mlx-lm @ git+https://github.com/ml-explore/mlx-lm@872ae88d1fac77350db23c8c04fe8dd372a9e3e8"]\n'
    )
    (workspace / "pyproject.toml").write_text(
        '[project]\nversion="1"\nrequires-python=">=3.11"\n'
        'description="Product description"\nreadme="README.md"\n'
        'authors=[{name="Product authors"}]\nkeywords=["mlx"]\n'
        'classifiers=["Operating System :: MacOS"]\n'
        '[project.urls]\nHomepage="https://example.org/omlx"\n'
    )
    (workspace / "apps/cli/pyproject.toml").write_text(
        '[project]\nname="omlx-cli"\nversion="1"\ndependencies=["omlx-cli>=1", "httpx>=0.27"]\n'
    )
    (workspace / "LICENSE").write_text("license")
    (workspace / "README.md").write_text("# Product readme")
    import shutil

    shutil.copytree(package, workspace / "apps/cli/src/omlx_cli")
    shutil.copy2(
        ROOT / "packages/runtime/setup.py", workspace / "packages/runtime/setup.py"
    )
    release.ROOT = workspace
    shutil.rmtree(project)
    project.mkdir()
    release.stage(project)
    subprocess.run(
        [sys.executable, "setup.py", "bdist_wheel"],
        cwd=project,
        check=True,
        capture_output=True,
    )
    wheel = next((project / "dist").glob("*.whl"))
    assert wheel.name.endswith("py3-none-macosx_15_0_arm64.whl")
    with zipfile.ZipFile(wheel) as zipped:
        names = zipped.namelist()
        metadata = zipped.read(
            next(name for name in names if name.endswith("METADATA"))
        ).decode()
        assert "Name: omlx\n" in metadata
        assert "Summary: Product description" in metadata
        assert "Author: Product authors" in metadata
        assert "Keywords: mlx" in metadata
        assert "Classifier: Operating System :: MacOS" in metadata
        assert "Project-URL: Homepage, https://example.org/omlx" in metadata
        assert "Description-Content-Type: text/markdown" in metadata
        assert "# Product readme" in metadata
        assert "Requires-Dist: omlx-" not in metadata
        assert "Requires-Dist: httpx>=0.27" in metadata
        assert any(name.endswith("entry_points.txt") for name in names)
        import json

        provenance = json.loads(
            zipped.read(
                next(
                    name
                    for name in names
                    if name.endswith("omlx_runtime/_version_sources.json")
                )
            )
        )
        pin = provenance["declared_dependencies"]["mlx-lm"]
        assert pin["url"] == "https://github.com/ml-explore/mlx-lm"
        assert pin["commit"] == "872ae88d1fac77350db23c8c04fe8dd372a9e3e8"
        assert provenance["commits"]["mlx-lm"] == pin
        node = next(name for name in names if name.endswith("_dashboard/runtime/node"))
        assert (zipped.getinfo(node).external_attr >> 16) & 0o111
        assert any(name.endswith("_dashboard/server/index.mjs") for name in names)
        assert any(name.endswith("_dashboard/public/assets/app.js") for name in names)
        installed = tmp_path / "installed"
        zipped.extractall(installed)
    # pip preserves the ZIP execute flags; ordinary ZipFile extraction does not.
    restored = installed / node
    restored.chmod(zipped.getinfo(node).external_attr >> 16)
    assert (
        subprocess.check_output([str(restored)], cwd=tmp_path, text=True).strip()
        == "v24.21.0"
    )


def test_download_rejects_other_sources(tmp_path):
    with pytest.raises(ValueError, match="official HTTPS"):
        bundle.download("http://nodejs.org/dist/archive", tmp_path / "archive")
