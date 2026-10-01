"""Assemble the workspace into the single installable Molto distribution."""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def stage(destination: Path) -> None:
    workspace = tomllib.loads((ROOT / "pyproject.toml").read_text())
    paths = [
        ROOT / "apps/cli",
        ROOT / "apps/server",
        *sorted((ROOT / "packages").iterdir()),
    ]
    projects = [
        tomllib.loads((path / "pyproject.toml").read_text())
        for path in paths
        if (path / "pyproject.toml").is_file()
    ]
    names = {project["project"]["name"] for project in projects}
    dependencies = set()
    extras = {}
    for path in paths:
        if not (path / "pyproject.toml").is_file():
            continue
        project = tomllib.loads((path / "pyproject.toml").read_text())["project"]
        if project["version"] != workspace["project"]["version"]:
            raise ValueError(f"Workspace version mismatch: {path}")
        for dependency in project.get("dependencies", []):
            if re.match(r"[A-Za-z0-9_.-]+", dependency).group(0) not in names:
                dependencies.add(dependency)
        for extra, requirements in project.get("optional-dependencies", {}).items():
            extras.setdefault(extra, set()).update(requirements)
        for package in (path / "src").iterdir():
            if package.is_dir() and not package.name.endswith(".egg-info"):
                shutil.copytree(
                    package,
                    destination / package.name,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
    runtime_project = next(
        project["project"]
        for project in projects
        if project["project"]["name"] == "molto-runtime"
    )
    declared = {}
    for requirement in runtime_project.get("dependencies", []):
        match = re.fullmatch(
            r"([\w.-]+)\s*@\s*git\+(https://[^@\s]+)@([0-9a-f]{7,40})",
            requirement,
        )
        if match:
            name, url, commit = match.groups()
            declared[name.lower().replace("_", "-")] = {"url": url, "commit": commit}
    # The standalone distribution installs these exact VCS requirements; record
    # their immutable pins so provenance survives missing dependency metadata.
    provenance = {"declared_dependencies": declared, "commits": declared}
    (destination / "molto_runtime/_version_sources.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    shutil.copy2(ROOT / "LICENSE", destination / "LICENSE")
    runtime_setup = (
        (ROOT / "packages/runtime/setup.py")
        .read_text()
        .replace('sourcedir="src/', 'sourcedir="')
    )
    runtime_setup = runtime_setup[: runtime_setup.index('if __name__ == "__main__":')]
    product = workspace["project"]
    readme = product.get("readme")
    metadata = dict(
        name="molto",
        version=workspace["project"]["version"],
        python_requires=workspace["project"]["requires-python"],
        install_requires=sorted(dependencies),
        extras_require={k: sorted(v) for k, v in extras.items()},
        entry_points={"console_scripts": ["molto=molto_cli.cli:main"]},
        license=product.get("license", "Apache-2.0"),
        description=product.get("description", ""),
        author=", ".join(
            author["name"] for author in product.get("authors", []) if "name" in author
        ),
        keywords=product.get("keywords", []),
        classifiers=product.get("classifiers", []),
        project_urls=product.get("urls", {}),
        long_description=(ROOT / readme).read_text() if isinstance(readme, str) else "",
        long_description_content_type="text/markdown"
        if isinstance(readme, str) and readme.endswith(".md")
        else "text/plain",
    )
    # Include every owned resource, including vendor licenses, corpora, compiled
    # kernels, and the verified standalone Node bundle. Native source is retained
    # for source diagnostics; extension compilation runs against this staging tree.
    setup = (
        runtime_setup
        + """
from setuptools import find_packages
from wheel.bdist_wheel import bdist_wheel

class BundledWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False
        self.plat_name = "macosx_15_0_arm64"
        self.plat_name_supplied = True

    def get_tag(self):
        python, abi, platform = super().get_tag()
        if self.distribution.has_ext_modules():
            return python, abi, platform
        return "py3", "none", platform
"""
    )
    setup += f"\nmetadata = {metadata!r}\n"
    setup += """kwargs = _custom_kernel_build_kwargs()
kwargs.setdefault("cmdclass", {})["bdist_wheel"] = BundledWheel
setup(packages=find_packages(), include_package_data=True, **metadata, **kwargs)
"""
    (destination / "setup.py").write_text(setup)
    (destination / "MANIFEST.in").write_text(
        "recursive-include molto_* *\nglobal-exclude *.pyc\nglobal-exclude __pycache__\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-dashboard",
        action="store_true",
        help="Reuse the existing verified dashboard bundle",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    if not args.skip_dashboard:
        subprocess.run(
            [sys.executable, str(ROOT / "tooling/release/build_dashboard_bundle.py")],
            check=True,
        )
    bundle = ROOT / "apps/cli/src/molto_cli/_dashboard"
    for resource in (
        "server/index.mjs",
        "runtime/node",
        "runtime/LICENSE",
        "runtime/manifest.json",
    ):
        if not (bundle / resource).is_file():
            parser.error(f"Missing dashboard bundle resource: {resource}")
    manifest = json.loads((bundle / "runtime/manifest.json").read_text())
    if manifest.get("platform") != "darwin-arm64":
        parser.error("The release bundle must contain the Darwin arm64 runtime")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workspace = tomllib.loads((ROOT / "pyproject.toml").read_text())
    (args.output_dir / "molto-overrides.txt").write_text(
        "\n".join(workspace["tool"]["uv"].get("override-dependencies", [])) + "\n"
    )
    with tempfile.TemporaryDirectory(prefix="molto-release-") as temporary:
        destination = Path(temporary)
        stage(destination)
        subprocess.run(
            [
                sys.executable,
                "setup.py",
                "bdist_wheel",
                "--dist-dir",
                str(args.output_dir.resolve()),
            ],
            cwd=destination,
            check=True,
        )


if __name__ == "__main__":
    main()
