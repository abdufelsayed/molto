"""Native GLM kernel extensions used by Molto monkey patches."""

from molto_runtime.custom_kernels.glm_moe_dsa import fast

__all__ = ["fast"]
