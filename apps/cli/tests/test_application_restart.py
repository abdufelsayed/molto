"""Explicit restart must record public intent before supervised termination."""

import json
import os
import signal
from types import SimpleNamespace

from molto_server import state as server


def scheduled_restart(monkeypatch, tmp_path):
    marker = tmp_path / "restart"
    monkeypatch.setenv("MOLTO_SUPERVISED", "application")
    monkeypatch.setenv("MOLTO_RESTART_MARKER", str(marker))
    callbacks = []
    monkeypatch.setattr(
        server.asyncio,
        "get_running_loop",
        lambda: SimpleNamespace(call_later=lambda delay, fn: callbacks.append(fn)),
    )
    signals = []
    monkeypatch.setattr(server.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    state = server.ServerState(
        bind_host="127.0.0.1",
        bind_port=8000,
        global_settings=SimpleNamespace(
            server=SimpleNamespace(host="0.0.0.0", port=9000)
        ),
    )
    return marker, callbacks, signals, state


def test_restart_records_pending_public_binding_before_signal(monkeypatch, tmp_path):
    marker, callbacks, signals, state = scheduled_restart(monkeypatch, tmp_path)
    assert state.request_restart() is True
    assert not marker.exists() and not signals
    callbacks[0]()
    assert json.loads(marker.read_text()) == {"host": "0.0.0.0", "port": 9000}
    assert signals == [(os.getpid(), signal.SIGTERM)]


def test_marker_failure_does_not_prevent_graceful_termination(monkeypatch, tmp_path):
    _, callbacks, signals, state = scheduled_restart(monkeypatch, tmp_path)
    monkeypatch.setenv("MOLTO_RESTART_MARKER", str(tmp_path / "absent" / "restart"))
    assert state.request_restart() is True
    callbacks[0]()
    assert signals == [(os.getpid(), signal.SIGTERM)]


def test_unsupervised_process_refuses_restart(monkeypatch, tmp_path):
    _, callbacks, signals, state = scheduled_restart(monkeypatch, tmp_path)
    monkeypatch.delenv("MOLTO_SUPERVISED")
    assert state.request_restart() is False
    assert not callbacks and not signals
