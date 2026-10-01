"""Preparation admission drains inference without evicting resident models."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from molto_runtime.engine_pool import EngineEntry, EnginePool
from molto_runtime.exceptions import ModelBusyError


def make_pool():
    pool = EnginePool()
    engine = SimpleNamespace()
    entry = EngineEntry(
        model_id="local",
        model_path="/local",
        engine_type="image_generation",
        model_type="image_generation",
        estimated_size=100,
    )
    entry.engine = engine
    entry.is_pinned = True
    pool._entries["local"] = entry
    pool._get_final_ceiling = lambda: 0
    return pool, entry


async def wait_active(pool):
    async with asyncio.timeout(2):
        while not pool._preparation_active:
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_preparation_waits_for_lease_blocks_fast_path_and_preserves_pinned():
    pool, entry = make_pool()
    entered, finish = asyncio.Event(), asyncio.Event()

    async def prepare():
        async with pool.exclusive_preparation(lambda: None) as allowance:
            assert allowance is None
            entered.set()
            await finish.wait()

    async with pool.acquire("local") as engine:
        task = asyncio.create_task(prepare())
        await wait_active(pool)
        assert not entered.is_set()
        with pytest.raises(ModelBusyError):
            await pool.get_engine("local")
        assert entry.engine is engine
        assert entry.in_use == 1
    await asyncio.wait_for(entered.wait(), 2)
    with pytest.raises(ModelBusyError):
        await pool.get_engine("unknown")
    finish.set()
    await task
    assert not pool._lock.locked()
    assert not pool._preparation_active
    assert await pool.get_engine("local") is engine
    assert entry.is_pinned


@pytest.mark.asyncio
async def test_cancel_waiting_gate_does_not_hold_lock_or_abort_inference():
    pool, entry = make_pool()
    entry.in_use = 1
    cancelled = False

    def check():
        if cancelled:
            raise ValueError("cancelled")

    async def prepare():
        async with pool.exclusive_preparation(check):
            pytest.fail("active lease should prevent admission")

    task = asyncio.create_task(prepare())
    await wait_active(pool)
    cancelled = True
    with pytest.raises(ValueError, match="cancelled"):
        await task
    assert entry.in_use == 1
    assert not pool._lock.locked()
    assert not pool._preparation_active


@pytest.mark.asyncio
async def test_scheduler_work_and_load_waiters_cannot_bypass_gate(monkeypatch):
    pool, entry = make_pool()
    active = True
    monkeypatch.setattr(pool, "_entry_has_scheduler_work", lambda _: active)
    await pool._lock.acquire()
    entry.engine = None
    pool._load_engine = AsyncMock()
    load = asyncio.create_task(pool.get_engine("local"))
    await asyncio.sleep(0)
    entered = asyncio.Event()

    async def prepare():
        async with pool.exclusive_preparation(lambda: None):
            entered.set()

    task = asyncio.create_task(prepare())
    await wait_active(pool)
    pool._lock.release()
    with pytest.raises(ModelBusyError):
        await load
    await asyncio.sleep(0.06)
    assert not entered.is_set()
    active = False
    await asyncio.wait_for(task, 2)
    pool._load_engine.assert_not_awaited()


@pytest.mark.asyncio
async def test_preparation_honors_pool_ceiling_and_releases_on_failure(monkeypatch):
    pool, _ = make_pool()
    pool._get_final_ceiling = lambda: 1000
    pool._current_model_memory = 400
    monkeypatch.setattr("molto_runtime.engine_pool.mx.get_active_memory", lambda: 200)
    monkeypatch.setattr(
        "molto_runtime.engine_pool._settled_phys_footprint", lambda: 500
    )
    with pytest.raises(RuntimeError, match="worker failed"):
        async with pool.exclusive_preparation(lambda: None) as allowance:
            assert allowance == 500
            assert pool._lock.locked()
            raise RuntimeError("worker failed")
    assert not pool._lock.locked()
    assert not pool._preparation_active
