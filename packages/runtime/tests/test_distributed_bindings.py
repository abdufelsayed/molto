"""App-owned RDMA dependencies reach the detached runtime supervisor."""

from types import SimpleNamespace

from molto_runtime.cluster.deployment import ClusterDeployment, ClusterHost
from molto_runtime.cluster.planner import PipelineAssignment
from molto_runtime.cluster.rdma.launch_links import enrolled_node_addresses
from molto_runtime.engine.distributed import DistributedBatchedEngine


def test_engine_forwards_launch_binding_without_global_configuration():
    deployment = ClusterDeployment(
        deployment_id="bindings-test",
        model="/models/fixture",
        backend="ring",
        hosts=(
            ClusterHost("local", "127.0.0.1", ("10.0.0.1",)),
            ClusterHost("peer", "peer", ("10.0.0.2",)),
        ),
        assignments=(
            PipelineAssignment("local", 0, 0, 2, 10, 10, 8, 128),
            PipelineAssignment("peer", 1, 2, 4, 10, 10, 8, 128),
        ),
        plan_hash="d" * 64,
    )
    callback = lambda item: (item, {"active": False})
    engine = DistributedBatchedEngine(deployment, attach_stage_links=callback)
    assert engine._supervisor._attach_stage_links is callback


def test_enrolled_addresses_use_explicit_application_enrollment():
    node = SimpleNamespace(
        node_id="peer",
        ssh="user@peer",
        addresses=("10.0.0.2",),
        hostname="peer",
        python_executable="/opt/python",
    )
    enrollment = SimpleNamespace(list_nodes=lambda: [node])
    addresses = enrolled_node_addresses(enrollment)
    assert addresses[0].node_id == "peer"
    assert addresses[0].addresses == ("10.0.0.2",)
