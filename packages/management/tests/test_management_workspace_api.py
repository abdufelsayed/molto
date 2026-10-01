"""Workspace routing inherits main-key authority and rejects path payloads."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from molto_server.api.management_workspace_routes import router, workspace
from molto_server.auth import AuthContext, require_management_key


@pytest.fixture
def client():
    app = FastAPI()
    app.state.management_auth_provider = lambda: AuthContext(
        main_key="main-key",
        sub_keys=[SimpleNamespace(key="sub-key")],
        skip_api_key_verification=False,
        bind_host="127.0.0.1",
    )
    app.include_router(
        router, prefix="/management/v1", dependencies=[Depends(require_management_key)]
    )
    service = SimpleNamespace(
        registry=AsyncMock(return_value={"models": []}),
        move=AsyncMock(return_value={"moved": True}),
        delete=AsyncMock(return_value={"deleted": True}),
    )
    app.dependency_overrides[workspace] = lambda: service
    return TestClient(app), service


@pytest.mark.parametrize("token", [None, "sub-key", "invalid-key"])
def test_workspace_requires_main_key(client, token):
    request, service = client
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    assert (
        request.get("/management/v1/workspace/registry", headers=headers).status_code
        == 401
    )
    service.registry.assert_not_awaited()


def test_authorized_registry_and_canonical_slash_model_route(client):
    request, service = client
    headers = {"Authorization": "Bearer main-key"}
    assert request.get("/management/v1/workspace/registry", headers=headers).json() == {
        "models": []
    }
    response = request.post(
        "/management/v1/workspace/models/org/model/move",
        headers=headers,
        json={"root_id": "derived-root-id"},
    )
    assert response.status_code == 200
    service.move.assert_awaited_once_with(
        "org/model", destination_root_id="derived-root-id", drain=False
    )


def test_arbitrary_output_and_unplanned_deletion_are_rejected(client):
    request, service = client
    headers = {"Authorization": "Bearer main-key"}
    assert (
        request.post(
            "/management/v1/workspace/models/model/move",
            headers=headers,
            json={"root_id": "derived-root-id", "output_path": "/arbitrary"},
        ).status_code
        == 422
    )
    assert (
        request.request(
            "DELETE",
            "/management/v1/workspace/models/model/delete",
            headers=headers,
            json={"drain": True},
        ).status_code
        == 422
    )
    service.move.assert_not_awaited()
    service.delete.assert_not_awaited()
