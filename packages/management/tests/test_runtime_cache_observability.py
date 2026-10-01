# SPDX-License-Identifier: Apache-2.0
"""Management cache inspection keeps engine observability without UI transforms."""

from types import SimpleNamespace

from molto_management.management import ManagementContext, ManagementService


class _Pool:
    def __init__(self, engine):
        self.entry = SimpleNamespace(engine=engine)

    def get_loaded_model_ids(self):
        return ["model-a"]

    def cache_status(self, directory=None):
        from molto_runtime.cache_operations import cache_status

        return cache_status(self, directory)

    def get_entry(self, model_id):
        assert model_id == "model-a"
        return self.entry


def _inspect(engine, tmp_path):
    settings = SimpleNamespace(
        base_path=tmp_path,
        cache=SimpleNamespace(get_ssd_cache_dir=lambda base: tmp_path / "ssd_cache"),
    )
    context = ManagementContext(
        engine_pool=_Pool(engine),
        settings_manager=None,
        global_settings=settings,
        get_default_model=lambda: None,
        set_default_model=lambda value: None,
        apply_sampling=lambda: None,
    )
    return ManagementService(context).cache()


def test_dflash_runtime_cache_stats_are_reported(tmp_path):
    stats = {
        "ssd_cache": {
            "num_files": 3,
            "total_size_bytes": 300,
            "hot_cache_size_bytes": 512,
            "hot_cache_entries": 2,
        },
        "cache_rates": {"cumulative": {"prefix_hits": 5}},
    }
    engine = SimpleNamespace(get_runtime_cache_stats=lambda: stats)

    payload = _inspect(engine, tmp_path)

    assert payload["models"] == [{"model_id": "model-a", "stats": stats}]
    assert payload["ssd_cache_dir"] == str(tmp_path / "ssd_cache")


def test_scheduler_stats_are_used_when_engine_has_no_cache_method(tmp_path):
    stats = {"ssd_cache": {"num_files": 1, "total_size_bytes": 100}}
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(get_ssd_cache_stats=lambda: stats)
    )

    assert _inspect(engine, tmp_path)["models"][0]["stats"] == stats


def test_gdn_and_boundary_snapshot_stats_remain_available(tmp_path):
    stats = {
        "prefix_cache": {
            "gdn_checkpoint_loads": 3,
            "gdn_checkpoint_walkbacks": 2,
            "gdn_last_restore": {"chosen_endpoint_tokens": 169984},
        },
        "gdn_staging": {"sidecar_count": 84, "state_dtype": "rht_int8"},
        "boundary_snapshots": {
            "capture_attempts": 5,
            "reasons": {"cache_offset_mismatch": 1},
        },
        "last_prefix_lookup": {
            "reused_kv_tokens": 16384,
            "unreused_common_prefix_tokens": 1616,
        },
        "ssd_cache": {"saves_persisted": 8, "loads": 6, "errors": 1},
    }
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(get_ssd_cache_stats=lambda: stats)
    )

    reported = _inspect(engine, tmp_path)["models"][0]["stats"]

    assert reported["prefix_cache"]["gdn_checkpoint_loads"] == 3
    assert reported["gdn_staging"]["sidecar_count"] == 84
    assert reported["boundary_snapshots"]["reasons"] == {"cache_offset_mismatch": 1}
    assert reported["last_prefix_lookup"]["reused_kv_tokens"] == 16384
    assert reported["ssd_cache"]["saves_persisted"] == 8


def test_engine_without_cache_stats_is_not_reported(tmp_path):
    engine = SimpleNamespace(get_runtime_cache_stats=lambda: None)

    assert _inspect(engine, tmp_path)["models"] == []
