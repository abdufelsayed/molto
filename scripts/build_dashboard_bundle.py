"""Stage the Nitro server and a verified standalone Node runtime for release wheels."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

DEFAULT_NODE_VERSION = "24.21.0"
ROOT = Path(__file__).resolve().parents[1]


def download(url: str, destination: Path) -> None:
    if not url.startswith("https://nodejs.org/dist/"):
        raise ValueError("Node downloads must use the official HTTPS distribution")
    with urllib.request.urlopen(url, timeout=120) as response:
        if not response.url.startswith("https://nodejs.org/dist/"):
            raise ValueError(
                "Node download redirected outside the official distribution"
            )
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)


def stage_bundle(
    output: Path, archive: Path, sha256: str, version: str, destination: Path
) -> None:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", sha256):
        raise ValueError("A SHA256 checksum is required")
    with archive.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual.lower() != sha256.lower():
        raise ValueError("Node archive SHA256 mismatch")
    if not (output / "server/index.mjs").is_file() or not (output / "public").is_dir():
        raise ValueError("Build the dashboard before staging its Nitro output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".dashboard-stage-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary) / "bundle"
        staged.mkdir()
        for component in ("server", "public"):
            shutil.copytree(output / component, staged / component)
        runtime = staged / "runtime"
        runtime.mkdir()
        prefix = f"node-v{version}-darwin-arm64"
        with tarfile.open(archive, "r:gz") as source:
            for name, target in (
                (f"{prefix}/bin/node", "node"),
                (f"{prefix}/LICENSE", "LICENSE"),
            ):
                member = source.getmember(name)
                if not member.isfile():
                    raise ValueError(
                        f"Node archive member is not a regular file: {name}"
                    )
                stream = source.extractfile(member)
                if stream is None:
                    raise ValueError(f"Missing Node archive member: {name}")
                with stream, (runtime / target).open("wb") as handle:
                    shutil.copyfileobj(stream, handle)
        (runtime / "node").chmod(0o755)
        (runtime / "LICENSE").chmod(0o644)
        (runtime / "manifest.json").write_text(
            json.dumps(
                {
                    "node_version": version,
                    "platform": "darwin-arm64",
                    "archive_sha256": actual,
                },
                indent=2,
            )
            + "\n"
        )
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(staged, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-version", default=DEFAULT_NODE_VERSION)
    parser.add_argument("--node-archive", type=Path)
    parser.add_argument("--node-sha256")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "dashboard/.output")
    parser.add_argument("--destination", type=Path, default=ROOT / "omlx/_dashboard")
    args = parser.parse_args()
    version = args.node_version.removeprefix("v")
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        parser.error("--node-version must be an exact version")
    if args.node_archive and not args.node_sha256:
        parser.error("--node-archive requires --node-sha256")
    if not args.skip_build:
        subprocess.run(
            ["pnpm", "install", "--frozen-lockfile"], cwd=ROOT / "dashboard", check=True
        )
        subprocess.run(["pnpm", "build"], cwd=ROOT / "dashboard", check=True)
    with tempfile.TemporaryDirectory(prefix="omlx-node-") as temporary:
        temp = Path(temporary)
        archive = args.node_archive
        checksum = args.node_sha256
        filename = f"node-v{version}-darwin-arm64.tar.gz"
        base = f"https://nodejs.org/dist/v{version}"
        if archive is None:
            archive = temp / filename
            sums = temp / "SHASUMS256.txt"
            download(f"{base}/SHASUMS256.txt", sums)
            published = {
                line.split()[1]: line.split()[0]
                for line in sums.read_text().splitlines()
                if len(line.split()) == 2
            }
            checksum = published.get(filename)
            if checksum is None:
                raise ValueError(
                    "Official checksum listing lacks the requested Node archive"
                )
            if args.node_sha256 and args.node_sha256.lower() != checksum.lower():
                raise ValueError(
                    "Explicit checksum disagrees with the official release"
                )
            download(f"{base}/{filename}", archive)
        stage_bundle(args.output, archive, checksum, version, args.destination)
    print(f"Dashboard bundle staged at {args.destination}")


if __name__ == "__main__":
    main()
