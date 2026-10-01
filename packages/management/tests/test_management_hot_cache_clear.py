# SPDX-License-Identifier: Apache-2.0
"""Management cache clearing respects active requests and clears retained data."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from molto_management.management import (
    ManagementContext,
    ManagementError,
    ManagementService,
)
from molto_runtime.engine_pool import EngineEntry, EnginePool


def _service(tmp_path, pool=None):
    pool = pool or EnginePool()
    settings = SimpleNamespace(
        base_path=tmp_path,
        cache=SimpleNamespace(get_ssd_cache_dir=lambda _: tmp_path / "ssd"),
    )
    return ManagementService(
        ManagementContext(
            engine_pool=pool,
            settings_manager=MagicMock(),
            global_settings=settings,
            get_default_model=lambda: None,
            set_default_model=lambda _: None,
            apply_sampling=lambda: None,
        )
    )


def _loaded(pool, *, active=False, clear_hot_cache=None):
    manager = SimpleNamespace(
        clear_hot_cache=clear_hot_cache or MagicMock(return_value=3),
        clear=MagicMock(return_value=2),
    )
    scheduler = SimpleNamespace(
        paged_ssd_cache_manager=manager,
        _cache_rate_tracker=None,
        _stream="test-stream",
    )
    core = SimpleNamespace(scheduler=scheduler, _mlx_executor=None)
    engine = SimpleNamespace(
        engine=core,
        has_active_requests=lambda: active,
        get_cache_stats=lambda: {"blocks": 1},
    )
    entry = EngineEntry(
        model_id="model-a",
        model_path="/unused",
        model_type="llm",
        engine_type="batched",
        estimated_size=1,
    )
    entry.engine = engine
    pool._entries["model-a"] = entry
    return entry, manager


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["hot", "ssd"])
async def test_clear_refuses_busy_engine_without_touching_cache(tmp_path, kind):
    pool = EnginePool()
    entry, manager = _loaded(pool)
    entry.in_use = 1

    with pytest.raises(ManagementError) as exc_info:
        await _service(tmp_path, pool).clear_cache(kind)

    assert exc_info.value.code == "busy"
    manager.clear_hot_cache.assert_not_called()
    manager.clear.assert_not_called()


@pytest.mark.asyncio
async def test_hot_clear_reclaims_even_when_no_model_is_loaded(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "molto_runtime.scheduler._sync_and_clear_cache",
        lambda stream: calls.append(stream),
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(
            "molto_runtime.engine_core.get_mlx_executor", lambda: executor
        )
        result = await _service(tmp_path).clear_cache("hot")

    assert result == {"status": "ok", "kind": "hot", "total_cleared": 0}
    assert calls == [None]


@pytest.mark.asyncio
async def test_hot_clear_reaches_loaded_engine_and_orphan_budget(tmp_path, monkeypatch):
    pool = EnginePool()
    _, manager = _loaded(pool)
    budget = SimpleNamespace(clear_all_owners=MagicMock(return_value=5))
    pool._scheduler_config = SimpleNamespace(hot_cache_budget=budget)
    calls = []
    monkeypatch.setattr(
        "molto_runtime.scheduler._sync_and_clear_cache",
        lambda stream: calls.append(stream),
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(
            "molto_runtime.engine_core.get_mlx_executor", lambda: executor
        )
        result = await _service(tmp_path, pool).clear_cache("hot")

    assert result["total_cleared"] == 8
    manager.clear_hot_cache.assert_called_once_with()
    budget.clear_all_owners.assert_called_once_with()
    assert calls == [None]


@pytest.mark.asyncio
async def test_ssd_clear_removes_persisted_cache_of_unloaded_model(tmp_path):
    cache_file = tmp_path / "ssd" / "0" / "item.safetensors"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"cache")

    result = await _service(tmp_path).clear_cache("ssd")

    assert result == {"status": "ok", "kind": "ssd", "total_cleared": 1}
    assert not cache_file.exists()


@pytest.mark.asyncio
async def test_distributed_ssd_clear_aggregates_rank_report(tmp_path):
    pool = EnginePool()
    entry, manager = _loaded(pool)
    entry.engine.engine.clear_prompt_caches = AsyncMock(return_value={"ssd_deleted": 4})

    result = await _service(tmp_path, pool).clear_cache("ssd")

    assert result == {"status": "ok", "kind": "ssd", "total_cleared": 4}
    entry.engine.engine.clear_prompt_caches.assert_awaited_once_with(ssd=True)
    manager.clear.assert_not_called()


@pytest.mark.asyncio
async def test_distributed_ssd_clear_reports_rank_failure(tmp_path):
    pool = EnginePool()
    entry, manager = _loaded(pool)
    entry.engine.engine.clear_prompt_caches = AsyncMock(
        side_effect=RuntimeError("rank failed")
    )

    with pytest.raises(ManagementError) as exc_info:
        await _service(tmp_path, pool).clear_cache("ssd")

    assert exc_info.value.code == "unavailable"
    assert "model-a: rank failed" in exc_info.value.detail
    manager.clear.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_cache_kind_is_rejected(tmp_path):
    with pytest.raises(ManagementError) as exc_info:
        await _service(tmp_path).clear_cache("probe")
    assert exc_info.value.code == "not_found"
