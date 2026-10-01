import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from molto_management.management import ManagementError
from molto_management.management_monitoring import MonitoringService
from molto_runtime.usage_history import UsageHistory
from molto_server.api.management_monitoring_routes import router
from molto_server.auth import AuthContext


@pytest.fixture
def setup(tmp_path):
    metrics = SimpleNamespace(
        usage_history=None,
        get_snapshot=lambda: {"uptime_seconds": 12},
        clear_metrics=Mock(),
        clear_alltime_metrics=Mock(),
    )
    entries = {}
    pool = SimpleNamespace(
        get_entry=entries.get,
        get_status=lambda: {
            "models": [],
            "current_model_memory": 32,
            "final_ceiling": 64,
        },
    )
    from molto_runtime.cache_operations import activity, probe

    pool.probe_cache = lambda request, settings: probe(pool, request, settings)
    pool.activity_snapshot = lambda supplied_metrics: activity(pool, supplied_metrics)
    context = SimpleNamespace(
        engine_pool=pool,
        runtime_state=SimpleNamespace(server_metrics=metrics),
        global_settings=SimpleNamespace(
            base_path=tmp_path, logging=SimpleNamespace(get_log_dir=lambda base: base)
        ),
        settings_manager=SimpleNamespace(get_settings=lambda model: None),
    )
    app = FastAPI()

    @app.exception_handler(ManagementError)
    async def management_error(request, exc):
        code = {
            "invalid_configuration": 400,
            "not_found": 404,
            "busy": 409,
            "unavailable": 503,
        }[exc.code]
        return JSONResponse(status_code=code, content={"detail": exc.detail})

    app.state.management_context_provider = lambda: context
    app.state.management_auth_provider = lambda: AuthContext(
        "main-key", [SimpleNamespace(key="sub-key")], "127.0.0.1"
    )
    app.include_router(router, prefix="/management/v1")
    client = TestClient(app)
    client.headers["Authorization"] = "Bearer main-key"
    return client, context, entries, tmp_path


def test_auth_and_validation(setup):
    client, *_ = setup
    for path in ("activity", "usage", "logs", "versions"):
        assert (
            client.get(
                "/management/v1/monitoring/" + path,
                headers={"Authorization": "Bearer sub-key"},
            ).status_code
            == 401
        )
    assert (
        client.post(
            "/management/v1/monitoring/cache/probe",
            json={"model_id": "x", "messages": [{}]},
            headers={"Authorization": "Bearer sub-key"},
        ).status_code
        == 401
    )
    assert (
        client.get("/management/v1/monitoring/usage?range=forever").status_code == 422
    )
    assert client.get("/management/v1/monitoring/logs?lines=10001").status_code == 422
    assert (
        client.get("/management/v1/monitoring/usage?model=../secret").status_code == 400
    )


def test_usage_disabled_error_filters_and_details(setup):
    client, context, _, path = setup
    assert (
        client.get("/management/v1/monitoring/usage").json()["state"] == "unavailable"
    )
    history = UsageHistory(path / "usage.sqlite3", enabled=False)
    context.runtime_state.server_metrics.usage_history = history
    try:
        assert (
            client.get("/management/v1/monitoring/usage").json()["state"] == "disabled"
        )
        history.set_enabled(True)
        history.record(
            model_id="canonical/model",
            prompt_tokens=10,
            completion_tokens=5,
            cached_tokens=2,
            prefill_duration=1,
            generation_duration=1,
        )
        history.record(
            model_id="other",
            prompt_tokens=100,
            completion_tokens=1,
            cached_tokens=0,
            prefill_duration=1,
            generation_duration=1,
        )
        history.flush()
        result = client.get(
            "/management/v1/monitoring/usage?model=canonical/model&include_details=true"
        ).json()
        assert result["totals"]["total_tokens"] == 15
        assert result["models"][0]["model_id"] == "canonical/model"
        assert "hourly" in result and "daily" in result
        history.path.unlink()
        assert (
            client.get("/management/v1/monitoring/usage").json()["state"]
            == "unavailable"
        )
    finally:
        history.close()


def test_rotated_logs_filter_and_traversal(setup):
    client, _, _, path = setup
    (path / "server.log").write_text(
        "2026-10-01 10:00:00,001 - molto_server.server - INFO - first\n2026-10-01 10:00:00,002 - molto_server.server - ERROR - second\n2026-10-01 10:00:00,003 - molto_server.server - INFO - third\n2026-10-01 10:00:00,004 - molto_server.server - ERROR - fourth\n"
    )
    (path / "server.log.1").write_text("WARNING rotated\n")
    result = client.get("/management/v1/monitoring/logs?lines=1&level=ERROR").json()
    assert (
        result["logs"]
        == "2026-10-01 10:00:00,004 - molto_server.server - ERROR - fourth\n"
    )
    assert result["matched_lines"] == 2 and result["total_lines"] == 4
    assert "server.log.1" in result["available_files"]
    assert (
        "rotated"
        in client.get("/management/v1/monitoring/logs?file=server.log.1").json()["logs"]
    )
    assert (
        client.get("/management/v1/monitoring/logs?file=../server.log").status_code
        == 400
    )
    (path / "server.log.2").symlink_to(path / "server.log")
    assert (
        client.get("/management/v1/monitoring/logs?file=server.log.2").status_code
        == 400
    )
    assert (
        client.get("/management/v1/monitoring/logs?file=server.log.9").status_code
        == 404
    )


def test_activity_all_engine_shapes(setup):
    _, context, entries, _ = setup
    now = time.monotonic()
    request = SimpleNamespace(
        request_id="run", num_output_tokens=9, generation_started_at=now - 3
    )
    waiting = SimpleNamespace(
        request_id="wait", arrival_time=now - 2, num_prompt_tokens=11
    )
    scheduler = SimpleNamespace(
        snapshot_for_admin=lambda: {
            "running_by_id": {"run": request},
            "waiting": [waiting],
        }
    )
    entries.update(
        llm=SimpleNamespace(
            engine=SimpleNamespace(
                _engine=SimpleNamespace(engine=SimpleNamespace(scheduler=scheduler))
            )
        ),
        embedding=SimpleNamespace(
            engine=SimpleNamespace(
                get_activity_snapshot=lambda: {
                    "active_requests": 2,
                    "activities": [{"kind": "embed"}],
                }
            )
        ),
        diffusion=SimpleNamespace(
            engine=SimpleNamespace(
                get_activity_snapshot=lambda: {
                    "active_requests": 1,
                    "activities": [{"kind": "image"}],
                }
            )
        ),
        loading=SimpleNamespace(engine=None),
        cluster=SimpleNamespace(
            engine=SimpleNamespace(
                get_live_metrics=lambda: {
                    "stale": False,
                    "metrics": {"active_requests": 3},
                }
            )
        ),
    )
    context.engine_pool.get_status = lambda: {
        "models": [
            {
                "id": key,
                "loaded": key != "loading",
                "is_loading": key == "loading",
                "loading_started_at": now - 2 if key == "loading" else None,
            }
            for key in entries
        ],
        "current_model_memory": 32,
        "final_ceiling": 64,
    }
    result = MonitoringService(context).activity()
    assert result["total_active_requests"] == 7
    assert result["total_waiting_requests"] == 1
    assert result["uptime_seconds"] == 12
    assert (
        next(m for m in result["models"] if m["id"] == "llm")["generating"][0][
            "generated_tokens"
        ]
        == 9
    )


def test_probe_boundaries_and_reset(setup):
    client, context, entries, _ = setup
    payload = {"model_id": "x", "messages": [{"role": "user", "content": "hello"}]}
    assert (
        client.post("/management/v1/monitoring/cache/probe", json=payload).status_code
        == 404
    )
    entries["x"] = SimpleNamespace(is_loading=True, engine=None)
    assert (
        client.post("/management/v1/monitoring/cache/probe", json=payload).status_code
        == 409
    )
    entries["x"].is_loading = False
    assert (
        client.post("/management/v1/monitoring/cache/probe", json=payload).json()[
            "model_loaded"
        ]
        is False
    )
    entries["x"].engine = SimpleNamespace()
    assert (
        client.post("/management/v1/monitoring/cache/probe", json=payload).status_code
        == 400
    )
    from molto_runtime.cache.paged_cache import compute_block_hash

    block_hash = compute_block_hash(b"", [1, 2], extra_keys=None, model_name="x")
    scheduler = SimpleNamespace(
        config=SimpleNamespace(paged_cache_block_size=2),
        paged_cache_manager=SimpleNamespace(model_name="x"),
        paged_ssd_cache_manager=SimpleNamespace(
            _hot_cache={block_hash: True},
            _index=SimpleNamespace(contains=lambda block: False),
        ),
    )
    engine = SimpleNamespace(
        _tokenizer=SimpleNamespace(
            apply_chat_template=lambda *a, **kw: "prompt", encode=lambda text: [1, 2, 3]
        ),
        scheduler=scheduler,
    )
    entries["x"].engine = engine
    result = client.post("/management/v1/monitoring/cache/probe", json=payload).json()
    assert (
        result["blocks_ssd_hot"] == 1
        and result["blocks_cold"] == 1
        and result["cold_tokens"] == 1
    )
    assert client.post(
        "/management/v1/monitoring/stats/reset", json={"scope": "session"}
    ).json()["history_preserved"]
    context.runtime_state.server_metrics.clear_metrics.assert_called_once()
    assert (
        client.get("/management/v1/monitoring/versions").json()["engines"]["mflux"][
            "name"
        ]
        == "mflux"
    )


def test_date_log_and_session_scope_validation(setup):
    client, context, _, path = setup
    (path / "server.log.2026-10-01").write_text("INFO dated\n")
    assert (
        "dated"
        in client.get(
            "/management/v1/monitoring/logs?file=server.log.2026-10-01"
        ).json()["logs"]
    )
    assert (
        client.post(
            "/management/v1/monitoring/stats/reset", json={"scope": "history"}
        ).status_code
        == 422
    )
    stats = path / "cannot-unlink-directory"
    stats.mkdir()
    context.runtime_state.server_metrics._stats_path = stats
    assert (
        client.post(
            "/management/v1/monitoring/stats/reset", json={"scope": "alltime"}
        ).status_code
        == 503
    )
    context.runtime_state.server_metrics.clear_alltime_metrics.assert_not_called()


def test_actual_scheduler_containers_and_stale_cluster(setup):
    _, context, entries, _ = setup
    from collections import deque

    scheduler = SimpleNamespace(
        running={"run": SimpleNamespace(num_output_tokens=3)},
        waiting=deque(
            [SimpleNamespace(request_id="wait", arrival_time=time.monotonic())]
        ),
    )
    entries["local"] = SimpleNamespace(
        engine=SimpleNamespace(
            _engine=SimpleNamespace(engine=SimpleNamespace(scheduler=scheduler))
        )
    )
    entries["cluster"] = SimpleNamespace(
        engine=SimpleNamespace(
            get_live_metrics=lambda: {"stale": True, "metrics": {"active_requests": 99}}
        )
    )
    context.engine_pool.get_status = lambda: {
        "models": [{"id": key, "loaded": True} for key in entries]
    }
    result = MonitoringService(context).activity()
    assert result["total_active_requests"] == 1
    assert result["total_waiting_requests"] == 1


def test_unknown_mutation_fields_rejected(setup):
    client, context, *_ = setup
    assert (
        client.post(
            "/management/v1/monitoring/stats/reset", json={"scpoe": "alltime"}
        ).status_code
        == 422
    )
    context.runtime_state.server_metrics.clear_metrics.assert_not_called()
    assert (
        client.post(
            "/management/v1/monitoring/cache/probe",
            json={"model_id": "x", "messages": [{}], "typo": True},
        ).status_code
        == 422
    )


def test_log_severity_field_minimum_and_continuations(setup):
    client, _, _, path = setup
    (path / "server.log").write_text(
        "2026-10-01 10:00:00,001 - molto_server.server - INFO - message containing ERROR\n2026-10-01 10:00:00,002 - molto_server.server - ERROR - actual error\nTraceback continuation\n2026-10-01 10:00:00,003 - molto_server.server - CRITICAL - worse\n"
    )
    result = client.get("/management/v1/monitoring/logs?level=ERROR").json()
    assert "containing ERROR" not in result["logs"]
    assert (
        "actual error" in result["logs"]
        and "Traceback continuation" in result["logs"]
        and "worse" in result["logs"]
    )
    assert result["matched_lines"] == 3


def test_log_scan_large_history_and_long_physical_line(setup):
    client, _, _, path = setup
    (path / "server.log").write_bytes(
        b"ERROR older unsearched\n"
        + b"x" * (2 * 1024 * 1024)
        + b"\n2026-10-01 10:00:00,001 - molto_server.server - INFO - ERROR in message\n2026-10-01 10:00:00,002 - molto_server.server - ERROR - newest\n"
    )
    result = client.get("/management/v1/monitoring/logs?level=ERROR").json()
    assert (
        result["logs"]
        == "2026-10-01 10:00:00,002 - molto_server.server - ERROR - newest\n"
    )
    assert result["total_lines"] is None
    assert result["scan_truncated"] and result["scan_message"]
    assert result["scanned_bytes"] == 1024 * 1024
    assert result["scanned_lines"] == 2
    (path / "server.log").write_bytes(b"ERROR " + b"x" * (2 * 1024 * 1024))
    result = client.get("/management/v1/monitoring/logs").json()
    assert result["logs"] == "" and result["scan_truncated"]


def test_spoofed_severity_continuation_keeps_record_level(setup):
    client, _, _, path = setup
    (path / "server.log").write_text(
        "2026-10-01 10:00:00,001 - molto_server.server - INFO - ordinary record\n"
        "ERROR this is message content\n"
        "still info content\n"
        "2026-10-01 10:00:00,002 - molto_server.server - ERROR - real failure\n"
        "real failure continuation\n"
        '{"level":"INFO","message":"JSON INFO mentioning ERROR"}\n'
        '{"level":"CRITICAL","message":"JSON critical"}\n'
    )
    result = client.get("/management/v1/monitoring/logs?level=ERROR").json()
    assert "this is message content" not in result["logs"]
    assert "still info content" not in result["logs"]
    assert "JSON INFO" not in result["logs"]
    assert "real failure continuation" in result["logs"]
    assert "JSON critical" in result["logs"]
    assert result["matched_lines"] == 3


def test_versions_read_runtime_workspace_declarations(setup):
    client, *_ = setup
    engines = client.get("/management/v1/monitoring/versions").json()["engines"]
    declared = engines["mlx-lm"]["declared_dependency"]
    assert declared["url"] == "https://github.com/ml-explore/mlx-lm"
    assert declared["commit"] == "872ae88d1fac77350db23c8c04fe8dd372a9e3e8"


def test_versions_bundle_provenance_without_internal_distribution(
    setup, tmp_path, monkeypatch
):
    import importlib.metadata

    import molto_runtime.version_sources as sources

    client, *_ = setup
    resources = tmp_path / "runtime"
    resources.mkdir()
    (resources / "_version_sources.json").write_text(
        '{"commits":{"mlx-lm":{"commit":"abcdef0123456789"}},'
        '"declared_dependencies":{"mlx-lm":{"commit":"abcdef0123456789","url":"https://example.org/mlx-lm"}}}'
    )
    monkeypatch.setattr(sources, "files", lambda name: resources)
    monkeypatch.setattr(sources, "__file__", str(resources / "version_sources.py"))

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "requires", missing)
    monkeypatch.setattr(importlib.metadata, "distribution", missing)
    info = client.get("/management/v1/monitoring/versions").json()["engines"]["mlx-lm"]
    assert info["commit"] == "abcdef0123456789"
    assert info["source"] == "bundle"
    assert info["declared_dependency"]["url"] == "https://example.org/mlx-lm"
