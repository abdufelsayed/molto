# SPDX-License-Identifier: Apache-2.0
"""Cluster deployment status, live metrics, and cache reporting.

Distributed engines own no local scheduler, so management status adapts their
rows from rank zero's telemetry marker instead.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from molto_runtime.cluster.deployment import ClusterDeployment, ClusterHost
from molto_runtime.cluster.planner import PipelineAssignment
from molto_runtime.engine.distributed import DistributedBatchedEngine
from molto_runtime.engine_pool import EngineEntry, EnginePool
from repo_paths import repository_root

ROOT = repository_root(__file__)


def _deployment(
    model_path: str = "/models/nemotron",
    *,
    host_count: int = 2,
    tensor_parallel_size: int = 1,
) -> ClusterDeployment:
    node_ids = ["local"] + [f"peer-{i}" for i in range(1, host_count)]
    return ClusterDeployment(
        deployment_id="status-test",
        model=model_path,
        backend="ring",
        hosts=tuple(
            ClusterHost(
                node_id,
                "127.0.0.1" if rank == 0 else f"{node_id}.local",
                (f"10.0.0.{rank + 1}",),
            )
            for rank, node_id in enumerate(node_ids)
        ),
        assignments=tuple(
            PipelineAssignment(node_id, rank, rank * 2, rank * 2 + 2, 10, 10, 8, 128)
            for rank, node_id in enumerate(node_ids)
        ),
        plan_hash="d" * 64,
        tensor_parallel_size=tensor_parallel_size,
    )


def _write_marker(state_dir: Path, deployment_id: str, payload: dict) -> None:
    (state_dir / f"{deployment_id}-rank-0.json").write_text(json.dumps(payload))


def _engine_with_marker(tmp_path: Path) -> DistributedBatchedEngine:
    engine = DistributedBatchedEngine(_deployment())
    engine._supervisor = SimpleNamespace(state_dir=str(tmp_path))
    return engine


def _metrics_payload(**overrides) -> dict:
    metrics = {
        "scope": "end_to_end_pipeline",
        "active_requests": 0,
        "requests_completed": 3,
        "requests_failed": 0,
        "requests_cancelled": 0,
        "prompt_tokens_total": 900,
        "completion_tokens_total": 120,
        "cached_tokens_total": 300,
        "aggregate_decode_tps": 11.5,
        "cache": {
            "affinity": "deployment",
            "lookups": 4,
            "hits": 3,
            "misses": 1,
            "hit_rate": 0.75,
            "tokens_reused": 300,
            "entries": 2,
            "bytes": 4096,
        },
        "pipeline": {
            "batch_steps": 9,
            "busy_seconds": 4.0,
            "idle_seconds": 6.0,
            "utilization": 0.4,
            "microbatch_target": 1,
            "async_overlap": False,
            "last_batch": None,
        },
        "last_request": None,
    }
    metrics.update(overrides)
    return {
        "updated_at": datetime.now(UTC).isoformat(),
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# DistributedBatchedEngine.get_live_metrics
# ---------------------------------------------------------------------------


def test_get_live_metrics_reads_rank_zero_marker(tmp_path):
    engine = _engine_with_marker(tmp_path)
    _write_marker(tmp_path, "status-test", _metrics_payload())

    live = engine.get_live_metrics()

    assert live is not None
    assert live["metrics"]["cache"]["hit_rate"] == 0.75
    assert isinstance(live["updated_at"], str)
    assert live["age_seconds"] is not None and live["age_seconds"] < 5
    assert live["stale"] is False


def test_get_live_metrics_marks_old_heartbeat_stale(tmp_path):
    engine = _engine_with_marker(tmp_path)
    payload = _metrics_payload()
    payload["updated_at"] = (datetime.now(UTC) - timedelta(seconds=120)).isoformat()
    _write_marker(tmp_path, "status-test", payload)

    live = engine.get_live_metrics()

    assert live is not None
    assert live["stale"] is True
    assert live["age_seconds"] >= 120


def test_get_live_metrics_returns_none_without_marker_or_metrics(tmp_path):
    engine = _engine_with_marker(tmp_path)

    assert engine.get_live_metrics() is None

    _write_marker(tmp_path, "status-test", {"updated_at": "now"})
    assert engine.get_live_metrics() is None

    (tmp_path / "status-test-rank-0.json").write_text("not json")
    assert engine.get_live_metrics() is None


# ---------------------------------------------------------------------------
# EnginePool.get_status cluster badge payload
# ---------------------------------------------------------------------------


def _pool_with_cluster_entry(tmp_path, deployment) -> EnginePool:
    pool = EnginePool()
    pool._cluster_registry = SimpleNamespace(
        get_for_model=lambda model: deployment if model == deployment.model else None
    )
    pool._entries["nemotron"] = EngineEntry(
        model_id="nemotron",
        model_path=deployment.model,
        model_type="llm",
        engine_type="batched",
        estimated_size=300,
    )
    return pool


def test_pool_status_cluster_payload_pipeline_strategy(tmp_path):
    deployment = _deployment(host_count=2, tensor_parallel_size=1)
    model = _pool_with_cluster_entry(tmp_path, deployment).get_status()["models"][0]

    assert model["distributed"] is True
    assert model["cluster"] == {
        "deployment_id": "status-test",
        "world_size": 2,
        "tensor_parallel_size": 1,
        "pipeline_stages": 2,
        "strategy": "pipeline",
        "backend": "ring",
        "target_context_tokens": 8192,
        "profile": "balanced",
    }


def test_pool_status_cluster_payload_tensor_strategy(tmp_path):
    deployment = _deployment(host_count=2, tensor_parallel_size=2)
    cluster = _pool_with_cluster_entry(tmp_path, deployment).get_status()["models"][0][
        "cluster"
    ]

    assert cluster["strategy"] == "tensor"
    assert cluster["tensor_parallel_size"] == 2
    assert cluster["pipeline_stages"] == 1


def test_pool_status_cluster_payload_hybrid_strategy(tmp_path):
    deployment = _deployment(host_count=4, tensor_parallel_size=2)
    cluster = _pool_with_cluster_entry(tmp_path, deployment).get_status()["models"][0][
        "cluster"
    ]

    assert cluster["strategy"] == "hybrid"
    assert cluster["world_size"] == 4
    assert cluster["tensor_parallel_size"] == 2
    assert cluster["pipeline_stages"] == 2


def test_pool_status_local_model_has_no_cluster_payload(tmp_path):
    pool = EnginePool()
    pool._cluster_registry = SimpleNamespace(get_for_model=lambda model: None)
    pool._entries["local-model"] = EngineEntry(
        model_id="local-model",
        model_path="/models/local",
        model_type="llm",
        engine_type="batched",
        estimated_size=100,
    )

    model = pool.get_status()["models"][0]

    assert model["distributed"] is False
    assert model["cluster"] is None
