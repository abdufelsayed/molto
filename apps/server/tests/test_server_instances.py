"""The private application factory isolates resources between server instances."""

from types import SimpleNamespace

from fastapi.testclient import TestClient
from omlx_server.server import create_app
from omlx_server.state import ServerState


def _tool(name):
    return SimpleNamespace(
        full_name=name, description=name, server_name="test", input_schema={}
    )


def test_factory_owns_distinct_state_and_metrics():
    first, second = create_app(), create_app()
    assert first.state.server_state is first.state.controller.state
    assert first.state.server_state is not second.state.server_state
    assert (
        first.state.server_state.responses_store
        is not second.state.server_state.responses_store
    )
    first.state.server_state.metrics.record_request_complete(
        prompt_tokens=3, completion_tokens=5, cached_tokens=0
    )
    assert (
        first.state.controller.get_server_metrics().get_snapshot()["total_requests"]
        == 1
    )
    assert (
        second.state.controller.get_server_metrics().get_snapshot()["total_requests"]
        == 0
    )


def test_authentication_and_mcp_resources_are_bound_to_request_app():
    first_state = ServerState(api_key="first-secret", bind_host="127.0.0.1")
    second_state = ServerState(api_key="second-secret", bind_host="127.0.0.1")
    first_state.mcp_manager = SimpleNamespace(
        get_all_tools=lambda: [_tool("first-tool")]
    )
    second_state.mcp_manager = SimpleNamespace(
        get_all_tools=lambda: [_tool("second-tool")]
    )
    first, second = (
        TestClient(create_app(first_state)),
        TestClient(create_app(second_state)),
    )
    assert (
        first.get(
            "/v1/mcp/tools", headers={"Authorization": "Bearer second-secret"}
        ).status_code
        == 401
    )
    assert (
        second.get(
            "/v1/mcp/tools", headers={"Authorization": "Bearer first-secret"}
        ).status_code
        == 401
    )
    assert (
        first.get(
            "/v1/mcp/tools", headers={"Authorization": "Bearer first-secret"}
        ).json()["tools"][0]["name"]
        == "first-tool"
    )
    assert (
        second.get(
            "/v1/mcp/tools", headers={"Authorization": "Bearer second-secret"}
        ).json()["tools"][0]["name"]
        == "second-tool"
    )


def test_health_readiness_and_stored_responses_do_not_cross_instances():
    first_state, second_state = ServerState(), ServerState()
    first_state.pinned_preload_complete = False
    first_state.responses_store.put(
        "resp_local", {"id": "resp_local", "object": "response"}
    )
    first, second = (
        TestClient(create_app(first_state)),
        TestClient(create_app(second_state)),
    )
    assert first.get("/health").status_code == 503
    assert second.get("/health").status_code == 200
    assert first.get("/v1/responses/resp_local").status_code == 200
    assert second.get("/v1/responses/resp_local").status_code == 404
    assert second.delete("/v1/responses/resp_local").status_code == 404
    assert first.get("/v1/responses/resp_local").status_code == 200


def test_cluster_registry_and_pairing_manager_are_application_owned():
    from omlx_server.cluster.services import ClusterServices

    def application(name):
        state = ServerState(
            api_key=f"{name}-key",
            bind_host="127.0.0.1",
            distributed_inference_enabled=True,
        )
        state.cluster_services = ClusterServices(
            registry=SimpleNamespace(to_dict=lambda: {"application": name})
        )
        state.pairing_manager = SimpleNamespace(
            ui_session=SimpleNamespace(poll=lambda: {"application": name})
        )
        app = create_app(state)
        app.state.controller._register_cluster_routes()
        return TestClient(app, headers={"Authorization": f"Bearer {name}-key"})

    first, second = application("first"), application("second")
    assert first.get("/admin/api/cluster/deployments").json() == {
        "application": "first"
    }
    assert second.get("/admin/api/cluster/deployments").json() == {
        "application": "second"
    }
    assert first.get("/api/cluster/pair/join").json() == {"application": "first"}
    assert second.get("/api/cluster/pair/join").json() == {"application": "second"}


def test_initialized_pools_forward_their_owned_rdma_stores(monkeypatch, tmp_path):
    from omlx_config.settings import GlobalSettings
    from omlx_runtime.cluster import launch

    def inspect_owned_services(deployment, *, nodes, store):
        return deployment, (nodes(), store())

    monkeypatch.setattr(launch, "attach_rdma_stage_links", inspect_owned_services)
    applications = []
    for name in ("first", "second"):
        base = tmp_path / name
        settings = GlobalSettings(base_path=base)
        models = base / "models"
        models.mkdir(parents=True)
        app = create_app()
        app.state.controller.initialize(str(models), global_settings=settings)
        services = app.state.server_state.cluster_services
        node = SimpleNamespace(
            node_id=name,
            ssh="localhost",
            addresses=("127.0.0.1",),
            hostname=name,
            python_executable="python",
        )
        services.enrollment = SimpleNamespace(list_nodes=lambda node=node: [node])
        services.rdma_links = object()
        applications.append(app)

    for name, app in zip(("first", "second"), applications, strict=True):
        deployment = object()
        returned, (nodes, store) = (
            app.state.server_state.engine_pool._attach_rdma_stage_links(deployment)
        )
        assert returned is deployment
        assert [node.node_id for node in nodes] == [name]
        assert store is app.state.server_state.cluster_services.rdma_links
    assert (
        applications[0].state.server_state.cluster_services.rdma_links
        is not applications[1].state.server_state.cluster_services.rdma_links
    )
    for app in applications:
        app.state.server_state.metrics.close()
