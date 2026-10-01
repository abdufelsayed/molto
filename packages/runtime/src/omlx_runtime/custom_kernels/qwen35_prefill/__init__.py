"""Qwen3.5/3.6 prefill kernels used by oMLX runtime patches."""

from omlx_runtime.custom_kernels.qwen35_prefill import fast
from omlx_runtime.custom_kernels.qwen35_prefill.gdn import (
    gated_delta_blocked_seq,
    gated_delta_chunked_metal,
    gated_delta_pipelined,
)

__all__ = [
    "fast",
    "gated_delta_blocked_seq",
    "gated_delta_chunked_metal",
    "gated_delta_pipelined",
]
