# SPDX-License-Identifier: Apache-2.0
"""Management unload keeps active requests safe while reporting progress."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from molto_management.management import (
    ManagementContext,
    ManagementError,
    ManagementService,
)
from molto_runtime.engine_pool import EngineEntry, EnginePool
from molto_server.api.management_routes import get_management_service, router
from molto_server.auth import require_management_key


def _service(pool):
    return ManagementService(
        ManagementContext(
            engine_pool=pool,
            settings_manager=MagicMock(),
            global_settings=None,
            get_default_model=lambda: None,
            set_default_model=lambda _: None,
            apply_sampling=lambda: None,
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("unloaded, status", [(False, "unloading"), (True, "ok")])
async def test_unload_reports_whether_teardown_has_finished(unloaded, status):
    entry = MagicMock(loaded=True, is_loading=False)
    pool = MagicMock()
    pool.get_model_view.return_value = entry
    pool.request_unload = AsyncMock(return_value=unloaded)

    assert await _service(pool).unload("model-a") == {
        "status": status,
        "model_id": "model-a",
    }
    pool.request_unload.assert_awaited_once_with(
        "model-a", reason="manual management unload"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entry, code",
    [(None, "not_found"), (MagicMock(loaded=False), "invalid_configuration")],
)
async def test_unload_rejects_missing_or_unloaded_model(entry, code):
    pool = MagicMock()
    pool.get_model_view.return_value = entry
    with pytest.raises(ManagementError) as exc_info:
        await _service(pool).unload("model-a")
    assert exc_info.value.code == code
    pool.request_unload.assert_not_called()


@pytest.mark.asyncio
async def test_active_lease_is_aborted_then_unloaded_after_release():
    pool = EnginePool()
    engine = SimpleNamespace(
        has_active_requests=lambda: False,
        abort_all_requests=AsyncMock(),
    )
    entry = EngineEntry(
        model_id="model-a",
        model_path="/unused",
        model_type="llm",
        engine_type="batched",
        estimated_size=1,
    )
    entry.engine = engine
    entry.in_use = 1
    pool._entries["model-a"] = entry
    pool._unload_engine = AsyncMock()

    result = await _service(pool).unload("model-a")

    assert result["status"] == "unloading"
    assert entry.abort_requested is True
    pool._unload_engine.assert_not_awaited()
    engine.abort_all_requests.assert_awaited_once()

    pending = pool._pending_unload_tasks["model-a"]
    await pool.release_engine("model-a")
    await asyncio.wait_for(pending, timeout=1)
    pool._unload_engine.assert_awaited_once_with("model-a")


def test_unload_route_returns_accepted_while_requests_drain():
    service = MagicMock()
    service.unload = AsyncMock(
        return_value={"status": "unloading", "model_id": "model-a"}
    )
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_management_key] = lambda: True
    app.dependency_overrides[get_management_service] = lambda: service

    response = TestClient(app).post("/management/v1/models/model-a/unload")

    assert response.status_code == 202
    assert response.json() == {
        "status": "unloading",
        "model_id": "model-a",
        "message": None,
    }


@pytest.mark.asyncio
async def test_lease_rejected_during_manual_unload_uses_unload_error():
    pool = MagicMock()
    pool.get_abort_requested_reason.return_value = "manual management unload"
    from molto_server.engine_requests import _raise_if_llm_lease_abort_requested
    from molto_server.shared import _LLMEngineLease

    lease = _LLMEngineLease(pool=pool, model_id="model-a")
    with pytest.raises(HTTPException) as exc_info:
        await _raise_if_llm_lease_abort_requested(lease)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == (
        "Request aborted because this model is being unloaded."
    )
