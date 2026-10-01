import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from molto_cli.client import CLIError
from molto_cli.migration import migrate_home


@pytest.fixture(autouse=True)
def stopped(monkeypatch):
    monkeypatch.setattr("molto_cli.migration.psutil.process_iter", lambda _: [])


def test_migration_preserves_checkpoint_identity_permissions_and_paths(tmp_path):
    source, target = tmp_path / ".omlx", tmp_path / ".molto"
    models = source / "models"
    models.mkdir(parents=True)
    checkpoint = models / "weights.safetensors"
    checkpoint.write_bytes(b"unchanged checkpoint")
    identity = checkpoint.stat().st_ino
    (models / "relative").symlink_to("weights.safetensors")
    (models / "absolute").symlink_to(checkpoint)
    settings = source / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "model": {"model_dirs": [str(models)]},
                "api_key": "preserved",
                "other": str(source) + "-unrelated",
            }
        )
    )
    settings.chmod(0o600)
    hub = tmp_path / ".cache/huggingface/hub"
    hub.parent.mkdir(parents=True)
    hub.symlink_to(models)
    support = tmp_path / "Library/Application Support/oMLX"
    support.mkdir(parents=True)
    (support / "base-path").write_text(str(source))

    report = migrate_home(source, target, home=tmp_path)

    assert not source.exists()
    assert (target / "models/weights.safetensors").stat().st_ino == identity
    assert (target / "models/relative").read_bytes() == b"unchanged checkpoint"
    assert os.readlink(target / "models/absolute") == str(
        target / "models/weights.safetensors"
    )
    assert hub.resolve() == target / "models"
    updated = json.loads((target / "settings.json").read_text())
    assert updated["model"]["model_dirs"] == [str(target / "models")]
    assert updated["api_key"] == "preserved"
    assert updated["other"] == str(source) + "-unrelated"
    assert (target / "settings.json").stat().st_mode & 0o777 == 0o600
    assert (target / "migration-backup/settings.json").stat().st_mode & 0o777 == 0o600
    assert (target / "bin/molto").stat().st_mode & 0o111
    assert (
        tmp_path / "Library/Application Support/Molto/base-path"
    ).read_text().strip() == str(target)
    assert report["symlinks_updated"] == 2


def test_dry_run_does_not_create_or_modify_anything(tmp_path):
    source, target = tmp_path / ".omlx", tmp_path / ".molto"
    source.mkdir()
    (source / "settings.json").write_text('{"model_dir":"~/.omlx/models"}')
    before = (source / "settings.json").read_bytes()
    report = migrate_home(source, target, home=tmp_path, dry_run=True)
    assert report["settings_files_updated"] == 1
    assert (source / "settings.json").read_bytes() == before
    assert not target.exists()


def test_diffusion_manifest_migrates_without_rewriting_weights_or_vendor_config(
    tmp_path,
):
    source, target = tmp_path / ".omlx", tmp_path / ".molto"
    model = source / "models/diffusion"
    model.mkdir(parents=True)
    weights = model / "model.safetensors"
    weights.write_bytes(b"preserved checkpoint")
    identity = weights.stat().st_ino
    manifest = model / "omlx-mflux.json"
    manifest.write_text(json.dumps({"source_path": str(model), "backend": "mflux"}))
    vendor_config = model / "config.json"
    vendor_config.write_text('{"upstream_path":"~/.omlx/unchanged"}')
    vendor_original = vendor_config.read_bytes()
    original = manifest.read_bytes()

    preview = migrate_home(source, target, home=tmp_path, dry_run=True)
    assert preview["model_manifests_renamed"] == 1
    assert manifest.read_bytes() == original
    assert not target.exists()

    report = migrate_home(source, target, home=tmp_path)
    migrated = target / "models/diffusion"
    assert not (migrated / manifest.name).exists()
    assert json.loads((migrated / "molto-mflux.json").read_text()) == {
        "source_path": str(migrated),
        "backend": "mflux",
    }
    assert (migrated / weights.name).stat().st_ino == identity
    assert (migrated / weights.name).read_bytes() == b"preserved checkpoint"
    assert (migrated / vendor_config.name).read_bytes() == vendor_original
    assert report["model_manifests_renamed"] == 1


def test_conflicting_model_manifests_block_migration_before_moving_data(tmp_path):
    source, target = tmp_path / ".omlx", tmp_path / ".molto"
    source.mkdir()
    for name in ("omlx-mflux.json", "molto-mflux.json"):
        (source / name).write_text("{}")
    with pytest.raises(CLIError, match="Both legacy and Molto"):
        migrate_home(source, target, home=tmp_path)
    assert source.is_dir()
    assert not target.exists()


def test_model_manifest_name_and_contents_roll_back_on_launcher_failure(
    tmp_path, monkeypatch
):
    source, target = tmp_path / ".omlx", tmp_path / ".molto"
    source.mkdir()
    manifest = source / "omlx-mflux.json"
    manifest.write_text('{"source_path":"~/.omlx/models/diffusion"}')
    original = manifest.read_bytes()
    write = Path.write_text

    def fail_launcher(path, data, *args, **kwargs):
        if path == target / "bin/molto":
            raise OSError("launcher failed")
        return write(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_launcher)
    with pytest.raises(OSError, match="launcher failed"):
        migrate_home(source, target, home=tmp_path)
    assert manifest.read_bytes() == original
    assert not (source / "molto-mflux.json").exists()
    assert not target.exists()


def test_migration_retires_legacy_launcher_and_stale_lifecycle_record(tmp_path):
    source, target = tmp_path / ".omlx", tmp_path / ".molto"
    (source / "bin").mkdir(parents=True)
    old_launcher = source / "bin/omlx"
    old_launcher.write_text("legacy app launcher")
    (source / "run").mkdir()
    record = source / "run/application.json"
    record.write_text('{"argv":["python","-m","omlx_cli.cli","serve"]}')
    public = tmp_path / ".local/bin/omlx"
    public.parent.mkdir(parents=True)
    public.symlink_to(old_launcher)
    report = migrate_home(source, target, home=tmp_path)
    assert not public.is_symlink()
    assert not (target / "bin/omlx").exists()
    assert not (target / "run/application.json").exists()
    assert (target / "migration-backup/omlx").read_text() == "legacy app launcher"
    assert (
        json.loads((target / "migration-backup/run/application.json").read_text())[
            "argv"
        ][2]
        == "omlx_cli.cli"
    )
    assert report["legacy_command_links_removed"] == 1


def test_existing_destination_is_never_merged_or_overwritten(tmp_path):
    source, target = tmp_path / ".omlx", tmp_path / ".molto"
    source.mkdir()
    target.mkdir()
    (target / "data").write_text("keep")
    with pytest.raises(CLIError, match="already exists"):
        migrate_home(source, target, home=tmp_path)
    assert source.exists()
    assert (target / "data").read_text() == "keep"


def test_symlinked_source_is_rejected(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    source = tmp_path / ".omlx"
    source.symlink_to(actual, target_is_directory=True)
    with pytest.raises(CLIError, match="symlinks"):
        migrate_home(source, tmp_path / ".molto", home=tmp_path)
    assert source.is_symlink()


def test_running_legacy_backend_blocks_migration(tmp_path, monkeypatch):
    source = tmp_path / ".omlx"
    source.mkdir()
    monkeypatch.setattr(
        "molto_cli.migration.psutil.process_iter",
        lambda _: [
            SimpleNamespace(
                pid=999999,
                info={
                    "name": "python",
                    "cmdline": ["python", "-m", "omlx_server.server"],
                },
            )
        ],
    )
    with pytest.raises(CLIError, match="Stop the running"):
        migrate_home(source, tmp_path / ".molto", home=tmp_path)
    assert source.exists()


def test_failed_settings_write_rolls_back_root_and_original_configuration(
    tmp_path, monkeypatch
):
    source, target = tmp_path / ".omlx", tmp_path / ".molto"
    source.mkdir()
    (source / "settings.json").write_text('{"model_dir":"~/.omlx/models"}')
    original = (source / "settings.json").read_bytes()
    write = Path.write_bytes
    failed = False

    def fail_once(path, data):
        nonlocal failed
        if path == target / "settings.json" and not failed:
            failed = True
            raise OSError("disk error")
        return write(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_once)
    with pytest.raises(OSError, match="disk error"):
        migrate_home(source, target, home=tmp_path)
    assert source.is_dir()
    assert not target.exists()
    assert (source / "settings.json").read_bytes() == original
