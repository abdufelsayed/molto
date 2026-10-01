"""Management runners exclusively use the pool without holding its load lock."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from molto_runtime.engine_pool import EngineEntry, EnginePool
from molto_runtime.exceptions import ModelBusyError


def make_pool():
    pool = EnginePool()
    entry = EngineEntry(
        model_id="local",
        model_path="/local",
        engine_type="image_generation",
        model_type="image_generation",
        estimated_size=100,
    )
    entry.engine = SimpleNamespace()
    pool._entries["local"] = entry
    pool._get_final_ceiling = lambda: 0
    return pool, entry


async def wait_active(pool):
    async with asyncio.timeout(2):
        while not pool._preparation_active:
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_lease_drains_before_owner_and_outsider_stays_blocked():
    pool, entry = make_pool()
    entered, finish = asyncio.Event(), asyncio.Event()

    async def run():
        async with pool.exclusive_management():
            assert pool.management_operation_allowed()
            assert not pool._lock.locked()
            assert await asyncio.create_task(pool.get_engine("local")) is entry.engine
            entered.set()
            await finish.wait()

    async with pool.acquire("local"):
        task = asyncio.create_task(run())
        await wait_active(pool)
        assert not entered.is_set()
        assert entry.in_use == 1
        assert not entry.abort_requested
        assert not pool.management_operation_allowed()
        with pytest.raises(ModelBusyError):
            await pool.get_engine("local")
    await asyncio.wait_for(entered.wait(), 2)
    with pytest.raises(ModelBusyError):
        await pool.get_engine("local")
    finish.set()
    await task
    assert pool.management_operation_allowed()
    assert pool._management_owner.get() is None


@pytest.mark.asyncio
async def test_owner_cold_load_and_unload_do_not_deadlock(monkeypatch, tmp_path):
    pool, entry = make_pool()
    engine = entry.engine
    entry.engine = None
    entry.model_path = str(tmp_path)
    (tmp_path / "config.json").write_text("{}")

    async def load(*args, **kwargs):
        entry.engine = engine

    async def unload(*args, **kwargs):
        entry.engine = None

    monkeypatch.setattr(pool, "_load_engine", AsyncMock(side_effect=load))
    monkeypatch.setattr(pool, "_unload_engine", AsyncMock(side_effect=unload))
    async with asyncio.timeout(2):
        async with pool.exclusive_management():
            assert await pool.get_engine("local") is engine
            assert await pool.unload_if_idle_unpinned("local")
    pool._load_engine.assert_awaited_once()
    pool._unload_engine.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["cancel_drain", "error_runner", "cancel_runner", "shutdown"]
)
async def test_failure_clears_gate_owner_and_locks(failure):
    pool, entry = make_pool()
    entered = asyncio.Event()
    if failure in {"cancel_drain", "shutdown"}:
        entry.in_use = 1

    async def run():
        async with pool.exclusive_management():
            entered.set()
            if failure == "error_runner":
                raise ValueError("runner failed")
            await asyncio.Event().wait()

    task = asyncio.create_task(run())
    if failure == "error_runner":
        await entered.wait()
    else:
        await wait_active(pool)
    if failure == "error_runner":
        with pytest.raises(ValueError, match="runner failed"):
            await task
    elif failure == "shutdown":
        pool._shutting_down = True
        with pytest.raises(RuntimeError, match="shutting down"):
            await task
    else:
        if failure == "cancel_runner":
            await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not pool._preparation_active
    assert pool._management_token is None
    assert pool._management_owner.get() is None
    assert not pool._preparation_lock.locked()
    assert not pool._lock.locked()
    assert not entry.abort_requested


@pytest.mark.asyncio
async def test_stale_child_token_cannot_bypass_next_gate_and_fifo_exclusion():
    pool, _ = make_pool()
    test_child, child_done = asyncio.Event(), asyncio.Event()
    order = []

    async def child():
        await test_child.wait()
        assert not pool.management_operation_allowed()
        with pytest.raises(ModelBusyError):
            await pool.get_engine("local")
        child_done.set()

    async def native():
        async with pool.exclusive_preparation(lambda: None):
            order.append("native")
            test_child.set()
            await child_done.wait()

    async def next_management():
        async with pool.exclusive_management():
            order.append("management")
            assert pool.management_operation_allowed()

    async with asyncio.timeout(2):
        async with pool.exclusive_management():
            stale_child = asyncio.create_task(child())
            native_task = asyncio.create_task(native())
            await asyncio.sleep(0)
            next_task = asyncio.create_task(next_management())
            await asyncio.sleep(0)
            assert order == []
        await asyncio.gather(stale_child, native_task, next_task)
    assert order == ["native", "management"]


@pytest.mark.asyncio
async def test_drain_load_scheduler_and_callback_cancel(monkeypatch):
    pool, entry = make_pool()
    entry.is_loading = True
    scheduler_busy = True
    cancelled = False
    entered = asyncio.Event()

    def check_cancel():
        if cancelled:
            raise asyncio.CancelledError()

    monkeypatch.setattr(pool, "_entry_has_scheduler_work", lambda _: scheduler_busy)

    async def run():
        async with pool.exclusive_management(check_cancel):
            entered.set()

    task = asyncio.create_task(run())
    await wait_active(pool)
    await asyncio.sleep(0.06)
    assert not entered.is_set()
    entry.is_loading = False
    await asyncio.sleep(0.06)
    assert not entered.is_set()
    cancelled = True
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not pool._preparation_active
    assert pool._management_token is None
    assert not pool._preparation_lock.locked()
    scheduler_busy = False
    async with pool.exclusive_management():
        assert pool.management_operation_allowed()


def test_management_views_preserve_activity_and_unload_signals():
    pool, entry = make_pool()
    assert not pool.is_model_busy("local")
    entry.in_use = 1
    assert pool.has_active_requests()
    assert pool.is_model_busy("local")
    entry.in_use = 0
    pool._unloading_models.add("local")
    assert pool.is_model_unloading("local")
    assert pool.is_model_busy("local")
    pool.set_model_pinned("local", True)
    pool.set_model_type("local", "embedding")
    assert entry.is_pinned and entry.engine_type == "embedding"
    previous = pool.set_prefill_speed_priority(True)
    assert pool._scheduler_config.prefill_speed_priority
    pool.set_prefill_speed_priority(previous)


def test_model_view_is_immutable_and_does_not_expose_engine():
    from dataclasses import FrozenInstanceError

    pool, entry = make_pool()
    view = pool.get_model_view("local")
    assert view.loaded and not hasattr(view, "engine")
    with pytest.raises(FrozenInstanceError):
        view.is_pinned = True
    pool.set_model_pinned("local", True)
    assert not view.is_pinned
    assert pool.get_model_view("local").is_pinned


def test_application_composition_binds_admission_and_clears_on_shutdown():
    pool, _ = make_pool()
    enforcer = SimpleNamespace(
        get_final_ceiling=lambda: 100,
        get_admission_ceiling=lambda: 90,
        get_admission_soft_target=lambda: 80,
        get_residency_ceiling=lambda: 70,
    )
    pool.configure_memory_enforcer(enforcer)
    assert pool.process_memory_enforcer is enforcer
    assert pool._current_ceiling() == 100
    pool.configure_memory_enforcer(None)
    assert pool.process_memory_enforcer is None
    assert pool._get_final_ceiling is None
    assert pool._get_admission_ceiling is None
    assert pool._get_admission_soft_target is None
    assert pool._get_residency_ceiling is None
