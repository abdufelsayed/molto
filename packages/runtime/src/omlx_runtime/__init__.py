# SPDX-License-Identifier: Apache-2.0
"""
omlx: LLM inference server, optimized for your Mac

This package provides native Apple Silicon GPU acceleration using
Apple's MLX framework and mlx-lm for LLMs.

Features:
- Continuous batching via vLLM-style scheduler
- OpenAI-compatible API server
- Paged KV cache with prefix sharing
- Tiered cache (GPU + paged SSD offloading)
"""

from omlx_config._version import __version__

_LAZY = {
    "Request": "omlx_runtime.request",
    "RequestOutput": "omlx_runtime.request",
    "RequestStatus": "omlx_runtime.request",
    "SamplingParams": "omlx_runtime.request",
    "Scheduler": "omlx_runtime.scheduler",
    "SchedulerConfig": "omlx_runtime.scheduler",
    "SchedulerOutput": "omlx_runtime.scheduler",
    "EngineCore": "omlx_runtime.engine_core",
    "AsyncEngineCore": "omlx_runtime.engine_core",
    "EngineConfig": "omlx_runtime.engine_core",
    "BlockAwarePrefixCache": "omlx_runtime.cache.prefix_cache",
    "PagedCacheManager": "omlx_runtime.cache.paged_cache",
    "CacheBlock": "omlx_runtime.cache.paged_cache",
    "BlockTable": "omlx_runtime.cache.paged_cache",
    "PrefixCacheStats": "omlx_runtime.cache.stats",
    "PagedCacheStats": "omlx_runtime.cache.stats",
    "CacheStats": "omlx_runtime.cache.stats",
    "get_registry": "omlx_runtime.model_registry",
    "ModelOwnershipError": "omlx_runtime.model_registry",
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
