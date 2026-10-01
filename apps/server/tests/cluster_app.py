"""Construct isolated HTTP applications around explicitly selected cluster stores."""

from types import SimpleNamespace

from fastapi import FastAPI
from molto_runtime.cluster import (
    discovery,
    enrollment,
    identity,
    incidents,
    registry,
    strategy_benchmarks,
)
from molto_runtime.cluster.rdma import store
from molto_server.cluster.services import ClusterServices
from molto_server.state import ServerState


def _configured(getter):
    try:
        return getter()
    except RuntimeError:
        return None


def cluster_app():
    """Snapshot stores chosen by runtime fixtures into this app's own context."""
    services = ClusterServices(
        registry=_configured(registry.get_cluster_registry),
        enrollment=_configured(enrollment.get_cluster_enrollment),
        incidents=_configured(incidents.get_cluster_incidents),
        strategy_benchmarks=_configured(
            strategy_benchmarks.get_strategy_benchmark_store
        ),
        identity=_configured(identity.get_node_identity),
        devices=_configured(registry.get_device_registry),
        rdma_links=_configured(store.get_rdma_link_store),
        discovery=_configured(discovery.get_discovery_service),
    )
    app = FastAPI()
    app.state.server_state = ServerState(cluster_services=services)
    app.state.controller = SimpleNamespace(get_engine_pool=lambda: None)
    return app
