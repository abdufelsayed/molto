"""Release staging tests use tiny synthetic archives, never a model or Node download."""

import hashlib
import importlib.util
import io
import os
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "dashboard_bundle", ROOT / "scripts/build_dashboard_bundle.py"
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
    setup_source = (ROOT / "setup.py").read_text()
    declarations = setup_source[
        setup_source.index("class DashboardWheel") : setup_source.index(
            'if __name__ == "__main__":'
        )
    ]
    patterns = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"][
        "setuptools"
    ]["package-data"]["omlx"]
    (project / "setup.py").write_text(
        "from pathlib import Path\nfrom setuptools import setup\nfrom wheel.bdist_wheel import bdist_wheel\n"
        + declarations
        + f"\nsetup(name='fixture-dashboard', version='1', packages=['omlx'], package_data={{'omlx': {patterns!r}}}, cmdclass={{'bdist_wheel': DashboardWheel}})\n"
    )
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
