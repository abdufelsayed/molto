"""Public serving telemetry keeps engine implementation details in the runtime."""

from types import SimpleNamespace

from omlx_runtime.engine_pool import EnginePool


def test_request_counts_sum_loaded_engines_without_counting_other_engine_types():
    pool = EnginePool()
    first = SimpleNamespace(
        _engine=SimpleNamespace(
            engine=SimpleNamespace(
                _output_collectors={"a": None, "b": None},
                scheduler=SimpleNamespace(waiting=[1, 2, 3]),
            )
        )
    )
    second = SimpleNamespace(
        _engine=SimpleNamespace(
            engine=SimpleNamespace(
                _output_collectors={"c": None}, scheduler=SimpleNamespace(waiting=[4])
            )
        )
    )
    pool._entries = {
        "first": SimpleNamespace(engine=first),
        "second": SimpleNamespace(engine=second),
        "unloaded": SimpleNamespace(engine=None),
        "embedding": SimpleNamespace(engine=SimpleNamespace()),
    }
    assert pool.get_request_counts() == (3, 4)
    del first._engine.engine._output_collectors["a"]
    assert pool.get_request_counts() == (2, 4)


def test_ane_snapshot_includes_attempted_models_without_leaking_patch_state(
    monkeypatch,
):
    from omlx_runtime.patches import qwen35_ane_prefill

    pool = EnginePool()
    attempted = object()
    untouched = object()
    patch_state = {"attempted": True, "configured": True}
    monkeypatch.setattr(
        qwen35_ane_prefill,
        "qwen35_ane_prefill_status",
        lambda model: (
            patch_state
            if model is attempted
            else {"attempted": False, "configured": False}
        ),
    )
    pool._entries = {
        "configured": SimpleNamespace(engine=SimpleNamespace(_model=attempted)),
        "untouched": SimpleNamespace(engine=SimpleNamespace(_vlm_model=untouched)),
        "unloaded": SimpleNamespace(engine=None),
    }
    snapshot = pool.get_ane_prefill_status()
    assert snapshot == {
        "patch_available": True,
        "configured_models": 1,
        "models": [{"attempted": True, "configured": True, "model_id": "configured"}],
    }
    snapshot["models"][0]["configured"] = False
    assert patch_state == {"attempted": True, "configured": True}
