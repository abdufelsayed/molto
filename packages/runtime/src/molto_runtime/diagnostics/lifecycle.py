"""Cancellation controls for runtime diagnostic workloads."""


def cancellation_requested(run) -> bool:
    return bool(getattr(run, "_cancel_requested", False))


def request_cancellation(run) -> None:
    run._cancel_requested = True
    event = getattr(run, "_cancel_calibration", None)
    if event is not None:
        event.set()
