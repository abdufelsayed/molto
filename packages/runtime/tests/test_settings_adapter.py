"""Runtime conversion of persisted settings."""

from unittest.mock import patch

from omlx_config.settings import GlobalSettings
from omlx_runtime.settings_adapter import scheduler_config as build_scheduler_config


class TestSettingsAdapter:
    def test_to_scheduler_config(self):
        """Test conversion to SchedulerConfig."""
        settings = GlobalSettings()
        settings.scheduler.max_concurrent_requests = 128
        settings.scheduler.embedding_batch_size = 12

        scheduler_config = build_scheduler_config(settings)
        assert scheduler_config.max_num_seqs == 128
        assert scheduler_config.completion_batch_size == 128
        assert scheduler_config.embedding_batch_size == 12
        assert scheduler_config.initial_cache_blocks == 256  # default
        assert scheduler_config.gdn_ssd_split_enabled is True
        assert scheduler_config.gdn_ssd_pending_max_bytes == 512 * 1024**2
        assert scheduler_config.gdn_sidecar_state_dtype == "fp32"

    def test_to_scheduler_config_initial_cache_blocks(self):
        """Test that initial_cache_blocks passes through to SchedulerConfig."""
        settings = GlobalSettings()
        settings.cache.initial_cache_blocks = 8192

        scheduler_config = build_scheduler_config(settings)
        assert scheduler_config.initial_cache_blocks == 8192

    def test_to_scheduler_config_gdn_split_settings(self):
        """GDN settings are attached to the scheduler config for later runtime use."""
        settings = GlobalSettings()
        settings.cache.gdn_ssd_split_enabled = True
        settings.cache.gdn_ssd_pending_max_size = "1GB"
        settings.cache.gdn_sidecar_state_dtype = "int8"

        scheduler_config = build_scheduler_config(settings)
        assert scheduler_config.gdn_ssd_split_enabled is True
        assert scheduler_config.gdn_ssd_pending_max_bytes == 1024**3
        assert scheduler_config.gdn_sidecar_state_dtype == "int8"

    def test_to_scheduler_config_rht_int8_gdn_state_dtype(self):
        settings = GlobalSettings()
        settings.cache.gdn_ssd_split_enabled = True
        settings.cache.gdn_sidecar_state_dtype = "rht_int8"

        scheduler_config = build_scheduler_config(settings)
        assert scheduler_config.gdn_ssd_split_enabled is True
        assert scheduler_config.gdn_sidecar_state_dtype == "rht_int8"


class TestInitSettings:
    """Tests for init_settings and get_settings."""

    def test_qwen4_decode_setting(self):
        settings = GlobalSettings()
        settings.server.qwen4_gdn_decode_wide_proj = True
        assert build_scheduler_config(settings).qwen4_gdn_decode_wide_proj is True

    def test_auto_cache_capacity(self, tmp_path):
        settings = GlobalSettings(base_path=tmp_path)
        cache_dir = settings.cache.get_ssd_cache_dir(tmp_path)
        (cache_dir / "a").mkdir(parents=True)
        (cache_dir / "a" / "block.safetensors").write_bytes(b"x" * 60)
        sidecars = cache_dir / "_gdn_sidecars" / ("b" * 64)
        sidecars.mkdir(parents=True)
        (sidecars / "state.safetensors").write_bytes(b"x" * 40)
        with patch("omlx_config.settings.shutil.disk_usage") as usage:
            usage.return_value.free = 200
            config = build_scheduler_config(settings)
            assert config.paged_ssd_cache_auto_size is True
            assert config.paged_ssd_cache_max_size == 150
