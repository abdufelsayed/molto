# SPDX-License-Identifier: Apache-2.0
"""Model-format checks used before enabling load-time engine features."""

from __future__ import annotations

import json
from pathlib import Path


def mtp_compatibility(model_path: str | Path) -> tuple[bool, str]:
    """Check the same config and weight evidence as the MTP runtime loader."""
    from molto_runtime.utils.model_loading import (
        _checkpoint_has_mtp_weights,
        _has_mtp_heads,
        _is_mtp_compatible,
    )

    path = Path(model_path)
    config_path = path / "config.json"
    if not config_path.exists():
        return False, "config.json not found"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, ValueError) as exc:
        return False, f"failed to read config: {exc}"
    quant = config.get("quantization_config") or {}
    if str(quant.get("quant_method") or "").lower() == "paroquant":
        return False, "Not supported on paroquant models yet"
    model_type = config.get("model_type")
    if not _has_mtp_heads(config):
        return False, "model has no MTP heads in config"
    if model_type != "qwen4_exp" and not _is_mtp_compatible(config, model_type):
        return False, f"model_type={model_type!r} is not on the MTP whitelist"
    if not _checkpoint_has_mtp_weights(str(path)):
        from molto_runtime.oq import _resolve_mtplx_sidecar

        if _resolve_mtplx_sidecar(path, config) is not None:
            return False, "MTPLX side-car detected but not imported"
        if model_type == "qwen4_exp":
            return False, (
                "Qwen4-Exp Lightning MTP requires embedded mtp.* tensors; "
                "native nextn layers are not supported by its dedicated runtime."
            )
        return False, "Config declares MTP layers but the weight files lack MTP tensors"
    return True, ""
