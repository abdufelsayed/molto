# SPDX-License-Identifier: Apache-2.0
"""Optional exact decode SDPA kernel used by Qwen4 QSA."""

from omlx_runtime.custom_kernels.decode_fast.fast import NATIVE_AVAILABLE, sdpa_decode

__all__ = ["NATIVE_AVAILABLE", "sdpa_decode"]
