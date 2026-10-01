# SPDX-License-Identifier: Apache-2.0
"""
molto: LLM inference server, optimized for your Mac

This package provides native Apple Silicon GPU acceleration using
Apple's MLX framework and mlx-lm for LLMs.

Features:
- Continuous batching via vLLM-style scheduler
- OpenAI-compatible API server
- Paged KV cache with prefix sharing
- Tiered cache (GPU + paged SSD offloading)
"""

from molto_config._version import __version__

_LAZY = {
    "Request": "molto_runtime.request",
    "RequestOutput": "molto_runtime.request",
    "RequestStatus": "molto_runtime.request",
    "SamplingParams": "molto_runtime.request",
    "Scheduler": "molto_runtime.scheduler",
    "SchedulerConfig": "molto_runtime.scheduler",
    "SchedulerOutput": "molto_runtime.scheduler",
    "EngineCore": "molto_runtime.engine_core",
    "AsyncEngineCore": "molto_runtime.engine_core",
    "EngineConfig": "molto_runtime.engine_core",
    "BlockAwarePrefixCache": "molto_runtime.cache.prefix_cache",
    "PagedCacheManager": "molto_runtime.cache.paged_cache",
    "CacheBlock": "molto_runtime.cache.paged_cache",
    "BlockTable": "molto_runtime.cache.paged_cache",
    "PrefixCacheStats": "molto_runtime.cache.stats",
    "PagedCacheStats": "molto_runtime.cache.stats",
    "CacheStats": "molto_runtime.cache.stats",
    "get_registry": "molto_runtime.model_registry",
    "ModelOwnershipError": "molto_runtime.model_registry",
}


def __getattr__(name: str):
    import importlib

    if name in _LAZY:
        mod = importlib.import_module(_LAZY[name])
        attr = "PagedCacheStats" if name == "CacheStats" else name
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Request management
    "Request",
    "RequestOutput",
    "RequestStatus",
    "SamplingParams",
    # Scheduler
    "Scheduler",
    "SchedulerConfig",
    "SchedulerOutput",
    # Engine
    "EngineCore",
    "AsyncEngineCore",
    "EngineConfig",
    # Model registry
    "get_registry",
    "ModelOwnershipError",
    # Prefix cache (paged SSD-only)
    "BlockAwarePrefixCache",
    # Paged cache (memory efficiency)
    "PagedCacheManager",
    "CacheBlock",
    "BlockTable",
    "PagedCacheStats",
    "CacheStats",  # Backward compatibility alias
    # Version
    "__version__",
]
