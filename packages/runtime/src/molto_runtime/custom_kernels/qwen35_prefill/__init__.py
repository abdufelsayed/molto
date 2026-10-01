"""Qwen3.5/3.6 prefill kernels used by Molto runtime patches."""

from molto_runtime.custom_kernels.qwen35_prefill import fast
from molto_runtime.custom_kernels.qwen35_prefill.gdn import (
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
