"""MTPLX file transactions with synthetic bytes and mocked tensor loading."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from molto_runtime import oq


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    root = tmp_path / "checkpoint"
    root.mkdir()
    config = {"model_type": "qwen3_5", "quantization": {"bits": 4, "group_size": 64}}
    (root / "config.json").write_text(json.dumps(config))
    (root / "mtplx_runtime.json").write_text(json.dumps({"arch_id": "qwen3-next-mtp"}))
    (root / "mtp.safetensors").write_bytes(b"original sidecar")
    (root / "model.safetensors").write_bytes(b"original backbone")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 32},
                "weight_map": {"backbone.weight": "model.safetensors"},
            }
        )
    )
    arrays = {
        "mtp.fc.weight": SimpleNamespace(nbytes=16),
        "mtp.fc.scales": SimpleNamespace(nbytes=8),
    }

    def load(path):
        if Path(path).read_bytes() == b"remapped shard":
            return {"language_model." + key: value for key, value in arrays.items()}
        return arrays

    def save(path, weights, **_):
        Path(path).write_bytes(b"remapped shard")

    monkeypatch.setattr(oq.mx, "load", MagicMock(side_effect=load))
    monkeypatch.setattr(oq.mx, "eval", MagicMock())
    monkeypatch.setattr(oq.mx, "save_safetensors", MagicMock(side_effect=save))
    return root


def snapshot(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_malformed_quantization_fails_before_file_mutation(checkpoint):
    root = checkpoint
    config = json.loads((root / "config.json").read_text())
    config["quantization"] = {}
    (root / "config.json").write_text(json.dumps(config))
    before = snapshot(root)
    for _ in range(2):
        with pytest.raises(ValueError, match="no global quantization"):
            oq.import_mtplx_sidecar(root)
        assert snapshot(root) == before
    oq.mx.save_safetensors.assert_not_called()


@pytest.mark.parametrize("vlm", [False, True])
@pytest.mark.parametrize(
    "destination",
    ["model-mtp.safetensors", "model.safetensors.index.json", "config.json"],
)
def test_publish_io_failure_restores_all_original_files_and_retry_succeeds(
    checkpoint, monkeypatch, vlm, destination
):
    root = checkpoint
    if vlm:
        index_path = root / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        index["weight_map"] = {"language_model.backbone.weight": "model.safetensors"}
        index_path.write_text(json.dumps(index))
    (root / "model-mtp.safetensors").write_bytes(b"prior shard must survive rollback")
    before = snapshot(root)
    replace = Path.replace
    failed = False

    def fail_once(source, target):
        nonlocal failed
        if Path(target) == root / destination and not failed:
            failed = True
            raise OSError("injected publish failure")
        return replace(source, target)

    with monkeypatch.context() as patcher:
        patcher.setattr(Path, "replace", fail_once)
        with pytest.raises(OSError, match="injected publish"):
            oq.import_mtplx_sidecar(root)
    assert snapshot(root) == before
    assert not list(root.glob(".mtplx-import-*"))
    result = oq.import_mtplx_sidecar(root)
    assert result["merge_mode"] == ("remap" if vlm else "rename")
    assert oq.import_mtplx_sidecar(root)["merge_mode"] == "noop"
    config = json.loads((root / "config.json").read_text())
    assert config["mtp_num_hidden_layers"] == 1
    prefix = "language_model." if vlm else ""
    assert config["quantization"][prefix + "mtp.fc"]["bits"] == 4


def test_remap_sidecar_rename_failure_restores_existing_orig(checkpoint, monkeypatch):
    root = checkpoint
    index_path = root / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {"weight_map": {"language_model.backbone.weight": "model.safetensors"}}
        )
    )
    (root / "mtp.safetensors.orig").write_bytes(b"previous archive")
    before = snapshot(root)
    replace = Path.replace
    failed = False

    def fail_once(source, target):
        nonlocal failed
        if Path(target) == root / "mtp.safetensors.orig" and not failed:
            failed = True
            raise OSError("sidecar archive failed")
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_once)
    with pytest.raises(OSError, match="archive failed"):
        oq.import_mtplx_sidecar(root)
    assert snapshot(root) == before


def test_shard_staging_failure_has_no_checkpoint_side_effects(checkpoint, monkeypatch):
    root = checkpoint
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {"language_model.backbone.weight": "model.safetensors"}}
        )
    )
    before = snapshot(root)
    monkeypatch.setattr(
        oq.mx, "save_safetensors", MagicMock(side_effect=OSError("disk full"))
    )
    with pytest.raises(OSError, match="disk full"):
        oq.import_mtplx_sidecar(root)
    assert snapshot(root) == before
    assert not list(root.glob(".mtplx-import-*"))


def test_legacy_partial_index_does_not_skip_missing_configuration_repair(checkpoint):
    root = checkpoint
    (root / "mtp.safetensors").replace(root / "model-mtp.safetensors")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 56},
                "weight_map": {
                    "backbone.weight": "model.safetensors",
                    "mtp.fc.weight": "model-mtp.safetensors",
                    "mtp.fc.scales": "model-mtp.safetensors",
                },
            }
        )
    )
    result = oq.import_mtplx_sidecar(root)
    assert result["merge_mode"] != "noop"
    config = json.loads((root / "config.json").read_text())
    assert config["mtp_num_hidden_layers"] == 1
    assert config["quantization"]["mtp.fc"]["bits"] == 4
    assert (
        json.loads((root / "model.safetensors.index.json").read_text())["metadata"][
            "total_size"
        ]
        == 56
    )


def test_rollback_io_failure_retains_named_original_backups(checkpoint, monkeypatch):
    root = checkpoint
    original_config = (root / "config.json").read_bytes()
    replace = Path.replace

    def deny_config_publish_and_restore(source, target):
        if Path(target) == root / "config.json":
            raise OSError("config filesystem unavailable")
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", deny_config_publish_and_restore)
    with pytest.raises(RuntimeError, match="recovery files retained"):
        oq.import_mtplx_sidecar(root)
    directories = list(root.glob(".mtplx-import-*"))
    assert len(directories) == 1
    backups = list(directories[0].glob("backup-*-config.json"))
    assert len(backups) == 1 and backups[0].read_bytes() == original_config
    assert (root / "mtp.safetensors").read_bytes() == b"original sidecar"
    assert (
        "mtp.fc.weight"
        not in json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
    )
