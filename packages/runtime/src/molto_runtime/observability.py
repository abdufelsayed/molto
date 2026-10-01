"""Public monitoring operations over runtime-owned metrics and memory state."""

from molto_contracts.runtime import RuntimeOperationError


def reset_metrics(metrics, scope: str) -> dict:
    if scope == "alltime":
        path = getattr(metrics, "_stats_path", None)
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                raise RuntimeOperationError(
                    "unavailable", "Persisted stats could not be reset"
                ) from exc
        metrics.clear_alltime_metrics()
    else:
        metrics.clear_metrics()
    return {"status": "ok", "scope": scope, "history_preserved": True}


def memory_policy_snapshot(enforcer) -> dict:
    breakdown = enforcer.get_ceiling_breakdown()
    return {
        "available": True,
        "tier": enforcer._memory_guard_tier,
        "custom_ceiling_gb": enforcer._memory_guard_custom_ceiling_bytes / 1024**3,
        "guard_enabled": enforcer._prefill_memory_guard,
        "ceiling_bytes": int(breakdown["hard_limit"]),
        "breakdown": breakdown,
        "wired_limit_request_bytes": int(
            getattr(enforcer, "_metal_wired_limit_request", 0) or 0
        ),
    }
