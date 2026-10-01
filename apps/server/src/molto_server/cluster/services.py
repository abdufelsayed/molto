"""Cluster stores and discovery resources owned by a single server application."""

import threading
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Request
from molto_runtime.cluster.enrollment import ClusterEnrollmentStore
from molto_runtime.cluster.identity import load_or_create
from molto_runtime.cluster.incidents import IncidentStore
from molto_runtime.cluster.rdma.store import RdmaLinkStore
from molto_runtime.cluster.registry import ClusterRegistry, DeviceRegistry
from molto_runtime.cluster.strategy_benchmarks import StrategyBenchmarkStore

from .rate_limit import ProbeRateLimiter


@dataclass
class ClusterServices:
    staging_jobs: dict = field(default_factory=dict)
    staging_lock: object = field(default_factory=threading.Lock)
    peer_health: dict = field(default_factory=dict)
    peer_health_lock: object = field(default_factory=threading.Lock)
    probe_limiter: ProbeRateLimiter = field(default_factory=ProbeRateLimiter)
    pair_request_limiter: ProbeRateLimiter = field(
        default_factory=lambda: ProbeRateLimiter(0.5, 8)
    )
    pair_status_limiter: ProbeRateLimiter = field(
        default_factory=lambda: ProbeRateLimiter(5.0, 20)
    )
    registry: object | None = None
    enrollment: object | None = None
    incidents: object | None = None
    strategy_benchmarks: object | None = None
    identity: object | None = None
    devices: object | None = None
    rdma_links: object | None = None
    discovery: object | None = None

    @classmethod
    def create(cls, base_path: Path):
        services = cls(
            registry=ClusterRegistry(base_path),
            enrollment=ClusterEnrollmentStore(base_path),
            incidents=IncidentStore(base_path),
            strategy_benchmarks=StrategyBenchmarkStore(base_path),
            rdma_links=RdmaLinkStore(base_path),
        )
        try:
            services.identity = load_or_create(base_path / "cluster" / "identity.json")
            services.devices = DeviceRegistry(base_path / "cluster" / "devices.json")
        except (OSError, ValueError):
            # Identity discovery is optional; serving and persistent model jobs
            # remain available when its files cannot be read or created.
            pass
        return services

    def announced_caps(self):
        from molto_runtime.cluster.discovery import local_caps

        if self.discovery is not None:
            caps = self.discovery.config.caps.to_dict()
            if caps:
                return caps
        try:
            return local_caps().to_dict()
        except Exception:
            return {}


def get_cluster_services(request: Request) -> ClusterServices:
    services = request.app.state.server_state.cluster_services
    if services is None:
        raise RuntimeError("cluster services are not configured")
    return services


def _required(value, name):
    if value is None:
        raise RuntimeError(f"{name} is not configured")
    return value


def get_cluster_registry(request: Request):
    return _required(get_cluster_services(request).registry, "cluster registry")


def get_cluster_enrollment(request: Request):
    return _required(get_cluster_services(request).enrollment, "cluster enrollment")


def get_cluster_incidents(request: Request):
    return _required(get_cluster_services(request).incidents, "cluster incidents")


def get_strategy_benchmark_store(request: Request):
    return _required(
        get_cluster_services(request).strategy_benchmarks, "strategy benchmarks"
    )


def get_node_identity(request: Request):
    return _required(get_cluster_services(request).identity, "node identity")


def get_device_registry(request: Request):
    return _required(get_cluster_services(request).devices, "device registry")


def get_discovery_service(request: Request):
    return _required(get_cluster_services(request).discovery, "cluster discovery")


def get_rdma_link_store(request: Request):
    return _required(get_cluster_services(request).rdma_links, "RDMA link store")


def get_pairing_manager(request: Request):
    return _required(request.app.state.server_state.pairing_manager, "pairing manager")
