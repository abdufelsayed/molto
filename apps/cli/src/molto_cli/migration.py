"""Move an existing oMLX data directory to Molto without copying checkpoints."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import stat
import sys
from pathlib import Path

import psutil

from molto_cli.client import CLIError


def _check_stopped() -> None:
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if process.pid == os.getpid():
            continue
        try:
            command = process.info["cmdline"] or []
            name = (process.info["name"] or "").lower()
            modules = (
                "omlx_cli.",
                "omlx_server.",
                "omlx_runtime.",
                "molto_cli.",
                "molto_server.",
                "molto_runtime.",
            )
            owned = name.startswith(("omlx", "molto")) or any(
                arg.startswith(modules) for arg in command
            )
            if owned and "migrate" not in command:
                raise CLIError(
                    "Stop the running oMLX/Molto application and cluster workers before migrating.",
                    2,
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def _rewrite_paths(value, source: Path, target: Path):
    if isinstance(value, dict):
        return {
            key: _rewrite_paths(item, source, target) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_paths(item, source, target) for item in value]
    if isinstance(value, str):
        for old, new in ((str(source), str(target)), ("~/.omlx", "~/.molto")):
            if value == old or value.startswith(old + "/"):
                return new + value[len(old) :]
    return value


def migrate_home(
    source: Path, target: Path, *, home: Path | None = None, dry_run: bool = False
) -> dict:
    home = home or Path.home()
    source = source.expanduser().absolute()
    target = target.expanduser().absolute()
    if source.is_symlink() or target.is_symlink():
        raise CLIError("Migration roots must be real directories, not symlinks.", 2)
    if not source.is_dir():
        raise CLIError(f"Source data directory does not exist: {source}", 2)
    if target.exists():
        raise CLIError(
            f"Destination already exists: {target}. Migration will not merge or overwrite it.",
            2,
        )
    if source == target or source in target.parents or target in source.parents:
        raise CLIError("Source and destination must be separate directories.", 2)
    if (
        not target.parent.is_dir()
        or source.stat().st_dev != target.parent.stat().st_dev
    ):
        raise CLIError(
            "Migration requires an existing destination parent on the same filesystem.",
            2,
        )
    _check_stopped()
    if (source / "migration-backup").exists():
        raise CLIError(
            "A migration backup already exists in the source. Preserve it separately before migrating again.",
            2,
        )
    for name in ("molto", "molto-cluster-python"):
        if (source / "bin" / name).exists() or (source / "bin" / name).is_symlink():
            raise CLIError(
                f"A {name} launcher already exists in the source; resolve the conflict first.",
                2,
            )
    old_support = home / "Library/Application Support/oMLX"
    new_support = home / "Library/Application Support/Molto"
    move_support = source == home / ".omlx" and old_support.is_dir()
    if move_support and (old_support.is_symlink() or new_support.exists()):
        raise CLIError(
            "Molto application support already exists or legacy support is a symlink; resolve that conflict first.",
            2,
        )
    files = list(source.rglob("*"))
    rewrites = {}
    links = {}
    for path in files:
        if path.is_symlink():
            original = os.readlink(path)
            updated = _rewrite_paths(original, source, target)
            if updated != original:
                links[path] = (original, updated)
        elif (
            path.is_file()
            and path.suffix == ".json"
            and "models" not in path.relative_to(source).parts
        ):
            original = path.read_bytes()
            try:
                value = json.loads(original)
            except (ValueError, UnicodeError):
                continue
            updated = _rewrite_paths(value, source, target)
            if value != updated:
                rewrites[path] = (
                    original,
                    (json.dumps(updated, indent=2) + "\n").encode(),
                )
    hub = home / ".cache/huggingface/hub"
    if hub.is_symlink():
        original = os.readlink(hub)
        updated = _rewrite_paths(original, source, target)
        if original != updated:
            links[hub] = (original, updated)
    command_links = []
    for path in (
        home / ".local/bin/omlx",
        Path("/opt/homebrew/bin/omlx"),
        Path("/usr/local/bin/omlx"),
    ):
        if path.is_symlink() and os.readlink(path) == str(source / "bin/omlx"):
            if not os.access(path.parent, os.W_OK):
                raise CLIError(f"Cannot replace legacy command link: {path}", 2)
            command_links.append(path)
    report = {
        "source": str(source),
        "target": str(target),
        "dry_run": dry_run,
        "files": sum(path.is_file() and not path.is_symlink() for path in files),
        "bytes": sum(
            path.stat().st_size
            for path in files
            if path.is_file() and not path.is_symlink()
        ),
        "settings_files_updated": len(rewrites),
        "symlinks_updated": len(links),
        "application_support_moved": move_support,
        "legacy_command_links_removed": len(command_links),
    }
    if dry_run:
        return report
    moved_support = False
    changed_links = []
    changed_files = []
    removed_command_links = []
    source.rename(target)
    try:
        backup = target / "migration-backup"
        backup.mkdir(mode=0o700)
        for old, (original, updated) in rewrites.items():
            relative = old.relative_to(source)
            path = target / relative
            saved = backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            saved.write_bytes(original)
            saved.chmod(stat.S_IMODE(path.stat().st_mode))
            changed_files.append((path, original))
            path.write_bytes(updated)
        for old, (original, updated) in links.items():
            path = (
                target / old.relative_to(source) if old.is_relative_to(source) else old
            )
            temporary = path.with_name(path.name + ".molto-migration")
            temporary.symlink_to(updated)
            os.replace(temporary, path)
            changed_links.append((path, original))
        if move_support:
            old_support.rename(new_support)
            moved_support = True
            pointer = new_support / "base-path"
            if pointer.is_file() and not pointer.is_symlink():
                original = pointer.read_bytes()
                changed_files.append((pointer, original))
                pointer.write_text(str(target) + "\n")
        binary = target / "bin"
        binary.mkdir(exist_ok=True)
        for name in ("omlx", "omlx-cluster-python", "omlx-source-python"):
            old = binary / name
            if old.exists() or old.is_symlink():
                shutil.copy2(old, backup / name, follow_symlinks=False)
        launcher = binary / "molto"
        launcher.write_text(
            f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -m molto_cli.cli "$@"\n'
        )
        launcher.chmod(0o755)
        worker = binary / "molto-cluster-python"
        worker.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
        worker.chmod(0o755)
        (backup / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        stale_record = target / "run/application.json"
        if stale_record.is_file() and not stale_record.is_symlink():
            saved_record = backup / "run/application.json"
            if not saved_record.exists():
                saved_record.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                shutil.copy2(stale_record, saved_record)
            changed_files.append((stale_record, stale_record.read_bytes()))
            stale_record.unlink()
        for path in command_links:
            original = os.readlink(path)
            path.unlink()
            removed_command_links.append((path, original))
        for name in ("omlx", "omlx-cluster-python", "omlx-source-python"):
            (binary / name).unlink(missing_ok=True)
    except Exception:
        for path, original in removed_command_links:
            path.symlink_to(original)
        for path, original in reversed(changed_files):
            path.write_bytes(original)
        for path, original in reversed(changed_links):
            path.unlink()
            path.symlink_to(original)
        if moved_support:
            new_support.rename(old_support)
        for name in ("molto", "molto-cluster-python"):
            (target / "bin" / name).unlink(missing_ok=True)
        if (target / "migration-backup").exists():
            for name in ("omlx", "omlx-cluster-python", "omlx-source-python"):
                saved = target / "migration-backup" / name
                if saved.exists() or saved.is_symlink():
                    shutil.copy2(saved, target / "bin" / name, follow_symlinks=False)
            shutil.rmtree(target / "migration-backup")
        target.rename(source)
        raise
    return report


def run(args) -> int:
    from molto_cli.cli_output import Output

    output = Output(args)
    source, target = Path(args.source), Path(args.target)
    if not args.dry_run:
        output.confirm(
            f"Move {source} to {target}, preserving models and updating stored paths?"
        )
    output.emit(migrate_home(source, target, dry_run=args.dry_run))
    return 0
