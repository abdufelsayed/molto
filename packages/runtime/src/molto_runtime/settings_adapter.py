"""Translate persisted configuration into inference runtime settings."""

from molto_config.config import parse_size
from molto_config.settings import GlobalSettings

from molto_runtime.scheduler import SchedulerConfig


def scheduler_config(settings: GlobalSettings) -> SchedulerConfig:
    """Build execution settings without coupling configuration to the runtime."""
    # Always resolve ssd_dir so the scheduler can initialize PagedSSDCacheManager.
    # When hot_cache_only=True, PagedSSDCacheManager skips directory init and
    # the writer thread internally — the dir is not used for disk I/O.
    ssd_dir = (
        settings.cache.get_ssd_cache_dir(settings.base_path)
        if settings.cache.enabled
        else None
    )

    return SchedulerConfig(
        qwen4_gdn_decode_wide_proj=settings.server.qwen4_gdn_decode_wide_proj,
        max_num_seqs=settings.scheduler.max_concurrent_requests,
        completion_batch_size=settings.scheduler.max_concurrent_requests,
        embedding_batch_size=settings.scheduler.embedding_batch_size,
        chunked_prefill=settings.scheduler.chunked_prefill,
        prefill_speed_priority=(settings.scheduler.prefill_priority == "speed"),
        decode_fairness=settings.scheduler.decode_fairness,
        initial_cache_blocks=settings.cache.initial_cache_blocks,
        paged_ssd_cache_dir=str(ssd_dir) if ssd_dir else None,
        hot_cache_only=settings.cache.hot_cache_only,
        paged_ssd_cache_max_size=settings.cache.get_ssd_cache_max_size_bytes(
            settings.base_path
        ),
        paged_ssd_cache_auto_size=settings.cache.ssd_cache_max_size.lower() == "auto",
        hot_cache_max_size=settings.cache.get_hot_cache_max_size_bytes(),
        hot_cache_write_through=settings.cache.hot_cache_write_through,
        gdn_ssd_split_enabled=settings.cache.get_gdn_ssd_split_enabled(),
        gdn_ssd_pending_max_bytes=parse_size(settings.cache.gdn_ssd_pending_max_size),
        gdn_sidecar_state_dtype=settings.cache.gdn_sidecar_state_dtype,
    )
