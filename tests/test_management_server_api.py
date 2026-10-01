"""Server management with disposable state and no native work."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient

from omlx.api.management_dependencies import get_runtime
from omlx.api.management_server_routes import router
from omlx.auth import AuthContext, require_management_key
from omlx.services.management import ManagementContext
from omlx.services.management_server import ServerManagementService
from omlx.settings import GlobalSettings


@pytest.fixture
def setup(tmp_path):
    settings = GlobalSettings(base_path=tmp_path)
    settings.auth.api_key = "main-original"
    active = {"key": "main-original", "host": "127.0.0.1"}
    state = SimpleNamespace()
    context = ManagementContext(
        engine_pool=Mock(),
        settings_manager=Mock(),
        global_settings=settings,
        get_default_model=lambda: None,
        set_default_model=lambda _: None,
        apply_sampling=Mock(),
        get_api_key=lambda: active["key"],
        set_api_key=lambda key: active.update(key=key),
        get_bind_host=lambda: active["host"],
        runtime_state=state,
    )
    runtime = SimpleNamespace(
        mutation_lock=asyncio.Lock(), operation_lock=asyncio.Lock()
    )
    app = FastAPI()
    app.state.management_context_provider = lambda: context
    app.state.management_auth_provider = lambda: AuthContext(
        active["key"],
        settings.auth.sub_keys,
        active["host"],
        settings.auth.skip_api_key_verification,
    )
    authenticated = APIRouter(
        prefix="/management/v1", dependencies=[Depends(require_management_key)]
    )
    authenticated.include_router(router)
    app.include_router(authenticated)
    app.dependency_overrides[get_runtime] = lambda: runtime
    client = TestClient(app)
    client.headers["Authorization"] = "Bearer main-original"
    return SimpleNamespace(
        settings=settings,
        active=active,
        state=state,
        context=context,
        svc=ServerManagementService(context),
        client=client,
        runtime=runtime,
    )


def test_all_settings_have_metadata_and_defaults(setup):
    data = setup.client.get("/management/v1/server/settings").json()
    assert set(data["sections"]) >= {
        "server",
        "model",
        "memory",
        "scheduler",
        "cache",
        "network",
        "logging",
        "integrations",
        "sampling",
        "mcp",
    }
    assert "auth" not in data["sections"]
    pairs = {(f["section"], f["key"]) for f in data["fields"]}
    assert pairs == {(s, k) for s, values in data["sections"].items() for k in values}
    assert all("restart_required" in f and "description" in f for f in data["fields"])
    assert (
        setup.client.get("/management/v1/server/defaults").json()["sections"]["server"][
            "port"
        ]
        == 8000
    )
    assert not (setup.settings.base_path / "settings.json").exists()


@pytest.mark.parametrize(
    "patch",
    [
        {"server": {"port": "8001"}},
        {"server": {"port": True}},
        {"server": {"port": 0}},
        {"sampling": {"top_p": 2}},
        {"cache": {"enabled": "yes"}},
        {"server": {"unknown": 1}},
        {"auth": {"api_key": "bypass"}},
        {"integrations": {"web_search_max_results": 11}},
        {"memory": {"soft_threshold": 0.99, "hard_threshold": 0.95}},
        {"model": {"model_dirs": [""]}},
        {"network": {"ca_bundle": "/does-not-exist"}},
        {"integrations": {"web_search_provider": "bogus"}},
        {"integrations": {"web_search_searxng_url": "file:///tmp/test"}},
        {"integrations": {"web_search_ddgs_backends": "unknown"}},
    ],
)
def test_invalid_candidate_never_mutates_or_saves(setup, patch):
    before = setup.settings.to_dict()
    response = setup.client.patch("/management/v1/server/settings", json=patch)
    assert response.status_code == 422
    assert setup.settings.to_dict() == before
    assert not (setup.settings.base_path / "settings.json").exists()


def test_patch_reports_live_and_restart_and_preserves_auth(setup):
    setup.settings.save()
    response = setup.client.patch(
        "/management/v1/server/settings",
        json={
            "sections": {
                "server": {"port": 8001},
                "sampling": {"temperature": 0.2},
                "model": {"model_dirs": [str(setup.settings.base_path / "new-models")]},
                "network": {"https_proxy": "https://proxy.example:443"},
            }
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["live_applied"] == ["sampling.temperature"]
    assert "server.port" in response.json()["restart_required"]
    setup.context.apply_sampling.assert_called_once()
    data = json.loads((setup.settings.base_path / "settings.json").read_text())
    assert data["auth"]["api_key"] == "main-original"
    assert data["model"]["model_dir"] == data["model"]["model_dirs"][0]
    assert not (setup.settings.base_path / "new-models").exists()


def test_failed_save_leaves_memory_and_runtime_unchanged(setup, monkeypatch):
    monkeypatch.setattr(
        GlobalSettings, "_save_data", Mock(side_effect=OSError("disk full"))
    )
    response = setup.client.patch(
        "/management/v1/server/settings", json={"sampling": {"temperature": 0.2}}
    )
    assert response.status_code == 500
    assert setup.settings.sampling.temperature == 1.0
    setup.context.apply_sampling.assert_not_called()


def test_runtime_failure_rolls_back_durable_candidate(setup):
    setup.settings.save()
    setup.context.apply_sampling.side_effect = [RuntimeError("failure"), None]
    response = setup.client.patch(
        "/management/v1/server/settings", json={"sampling": {"temperature": 0.2}}
    )
    assert response.status_code == 500
    assert setup.settings.sampling.temperature == 1.0
    assert (
        json.loads((setup.settings.base_path / "settings.json").read_text())[
            "sampling"
        ]["temperature"]
        == 1.0
    )


def test_restart_requires_supervisor_and_rejects_busy(setup):
    assert (
        setup.client.get("/management/v1/server/info").json()["restart_supported"]
        is False
    )
    assert setup.client.post("/management/v1/server/restart").status_code == 503
    setup.state.request_restart = Mock(return_value=True)
    assert setup.client.post("/management/v1/server/restart").status_code == 202
    setup.state.request_restart.assert_called_once()
    asyncio.run(setup.runtime.operation_lock.acquire())
    assert setup.client.post("/management/v1/server/restart").status_code == 409
    setup.state.request_restart.assert_called_once()
    setup.runtime.operation_lock.release()


def test_copyable_integration_commands_quote_models(setup):
    setup.settings.integrations.codex_model = "model with spaces; $(danger)"
    response = setup.client.get("/management/v1/server/integrations")
    command = next(
        x["command"] for x in response.json()["integrations"] if x["id"] == "codex"
    )
    import shlex

    assert shlex.split(command) == [
        "omlx",
        "launch",
        "codex",
        "--model",
        "model with spaces; $(danger)",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]


def test_websearch_test_uses_pending_values_without_save(setup, monkeypatch):
    from unittest.mock import AsyncMock

    mock = AsyncMock(return_value={"ok": False, "error": {"code": "request_failed"}})
    monkeypatch.setattr("omlx.websearch.run_web_search_test", mock)
    response = setup.client.post(
        "/management/v1/server/web-search/test",
        json={"provider": "brave", "brave_api_key": "pending"},
    )
    assert response.status_code == 200 and response.json()["ok"] is False
    assert mock.call_args.kwargs["brave_api_key"] == "pending"
    assert setup.settings.integrations.web_search_brave_api_key == ""
    assert not (setup.settings.base_path / "settings.json").exists()


def test_update_distinguishes_network_failure_from_no_update(setup, monkeypatch):
    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, *args, **kwargs):
            raise httpx.ConnectError("offline")

    monkeypatch.setattr("omlx.services.management_server.httpx.AsyncClient", Client)
    data = setup.client.get("/management/v1/server/update").json()
    assert data["status"] == "failed" and data["update_available"] is None


def test_update_filters_prereleases_even_with_false_flag(setup, monkeypatch):
    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, *args, **kwargs):
            return httpx.Response(
                200,
                request=httpx.Request("GET", "https://example.com"),
                json=[
                    {"tag_name": "v999.0.0rc1", "prerelease": False},
                    {"tag_name": "v998.0.0", "html_url": "https://example.com/release"},
                ],
            )

    monkeypatch.setattr("omlx.services.management_server.httpx.AsyncClient", Client)
    assert (
        setup.client.get("/management/v1/server/update").json()["latest_version"]
        == "998.0.0"
    )
    assert (
        setup.client.get("/management/v1/server/update?channel=beta").json()[
            "latest_version"
        ]
        == "999.0.0rc1"
    )


def test_routes_require_main_key(setup):
    setup.client.headers.clear()
    assert setup.client.get("/management/v1/server/settings").status_code == 401
    assert setup.client.post("/management/v1/server/restart").status_code == 401


def test_cache_raw_dataclass_fields_survive_persisted_compatibility_names(setup):
    response = setup.client.patch(
        "/management/v1/server/settings",
        json={
            "cache": {"gdn_ssd_split_enabled": None, "gdn_sidecar_state_dtype": "bf16"}
        },
    )
    assert response.status_code == 200
    data = response.json()["sections"]["cache"]
    assert data["gdn_ssd_split_enabled"] is None
    assert data["gdn_sidecar_state_dtype"] == "bf16"
    reloaded = GlobalSettings.load(base_path=setup.settings.base_path)
    assert reloaded.cache.gdn_ssd_split_enabled is None
    assert reloaded.cache.gdn_sidecar_state_dtype == "bf16"


@pytest.mark.parametrize(
    "patch",
    [
        {"network": {"https_proxy": "https://"}},
        {"huggingface": {"endpoint": "https://example.com:invalid"}},
        {"cache": {"ssd_cache_dir": "bad\x00path"}},
    ],
)
def test_endpoint_and_path_validation_is_atomic(setup, patch):
    response = setup.client.patch("/management/v1/server/settings", json=patch)
    assert response.status_code == 422
    assert not (setup.settings.base_path / "settings.json").exists()


def test_unrelated_edit_preserves_persisted_values_under_runtime_overrides(setup):
    setup.settings.save()
    path = setup.settings.base_path / "settings.json"
    disk = json.loads(path.read_text())
    disk["server"]["port"] = 8000
    disk["network"]["https_proxy"] = "https://saved.example:443"
    disk["future_extension"] = {"preserve": True}
    path.write_text(json.dumps(disk))
    setup.settings.server.port = 9123
    setup.settings.network.https_proxy = "https://runtime.example:443"
    response = setup.client.patch(
        "/management/v1/server/settings", json={"sampling": {"temperature": 0.2}}
    )
    assert response.status_code == 200, response.text
    saved = json.loads(path.read_text())
    assert saved["server"]["port"] == 8000
    assert saved["network"]["https_proxy"] == "https://saved.example:443"
    assert saved["future_extension"] == {"preserve": True}
    assert saved["sampling"]["temperature"] == 0.2
    assert setup.settings.server.port == 9123
    assert setup.settings.network.https_proxy == "https://runtime.example:443"


def test_explicit_value_matching_runtime_override_is_persisted(setup):
    setup.settings.save()
    setup.settings.server.port = 9123
    response = setup.client.patch(
        "/management/v1/server/settings", json={"server": {"port": 9123}}
    )
    assert response.status_code == 200
    assert "server.port" in response.json()["changed"]
    assert (
        json.loads((setup.settings.base_path / "settings.json").read_text())["server"][
            "port"
        ]
        == 9123
    )


def test_auth_edits_do_not_freeze_unrelated_runtime_overrides(setup):
    setup.settings.save()
    setup.settings.server.port = 9123
    setup.svc.subkey({"key": "separate-sub"})
    setup.svc.main_key("separate-main")
    setup.svc.policy({"allow_unauthenticated_inference": True})
    saved = json.loads((setup.settings.base_path / "settings.json").read_text())
    assert saved["server"]["port"] == 8000
    assert saved["auth"]["api_key"] == "separate-main"
    assert saved["auth"]["sub_keys"][0]["key"] == "separate-sub"
    assert saved["auth"]["allow_unauthenticated_inference"] is True


def test_failed_live_apply_restores_exact_persisted_values_not_overrides(setup):
    setup.settings.save()
    path = setup.settings.base_path / "settings.json"
    before = json.loads(path.read_text())
    setup.settings.server.port = 9123
    setup.context.apply_sampling.side_effect = [RuntimeError("failed"), None]
    response = setup.client.patch(
        "/management/v1/server/settings", json={"sampling": {"temperature": 0.2}}
    )
    assert response.status_code == 500
    assert json.loads(path.read_text()) == before
    assert setup.settings.server.port == 9123


def test_info_and_launch_commands_keep_active_port_until_restart(setup):
    setup.state.bind_port = 9123
    assert (
        setup.client.patch(
            "/management/v1/server/settings", json={"server": {"port": 8123}}
        ).status_code
        == 200
    )
    data = setup.client.get("/management/v1/server/info").json()
    assert data["port"] == 9123
    assert data["configured_port"] == 8123
    commands = setup.client.get("/management/v1/server/integrations").json()[
        "integrations"
    ]
    assert all("--port 9123" in command["command"] for command in commands)


def test_actual_info_callback_port_overrides_configured_and_state_fallback(setup):
    from dataclasses import replace

    setup.state.bind_port = 9123
    setup.settings.server.port = 8123
    svc = ServerManagementService(
        replace(setup.context, get_server_info=lambda: {"port": 9234})
    )
    assert svc.info()["port"] == 9234
    assert svc.info()["configured_port"] == 8123


def test_unsupervised_callable_restart_is_unavailable_and_never_invoked(setup):
    from dataclasses import replace

    from omlx.api.management_server_routes import service

    setup.state.request_restart = Mock(return_value=False)
    context = replace(
        setup.context, get_server_info=lambda: {"restart_supported": False}
    )
    setup.client.app.dependency_overrides[service] = lambda: ServerManagementService(
        context
    )
    assert (
        setup.client.get("/management/v1/server/info").json()["restart_supported"]
        is False
    )
    response = setup.client.post("/management/v1/server/restart")
    assert response.status_code == 503
    setup.state.request_restart.assert_not_called()


def test_supervised_restart_rejection_remains_busy(setup):
    setup.state.request_restart = Mock(return_value=False)
    assert setup.client.post("/management/v1/server/restart").status_code == 409


def test_ca_bundle_is_normalized_before_persistence(setup, monkeypatch):
    from pathlib import Path

    certificate = setup.settings.base_path / "certificate.pem"
    certificate.write_text("test CA")
    original_expanduser = Path.expanduser
    monkeypatch.setattr(
        Path,
        "expanduser",
        lambda path: (
            certificate
            if str(path) == "~/certificate.pem"
            else original_expanduser(path)
        ),
    )
    response = setup.client.patch(
        "/management/v1/server/settings",
        json={"network": {"ca_bundle": "~/certificate.pem"}},
    )
    assert response.status_code == 200, response.text
    expected = str(certificate.resolve())
    assert setup.settings.network.ca_bundle == expected
    assert response.json()["sections"]["network"]["ca_bundle"] == expected
    assert (
        json.loads((setup.settings.base_path / "settings.json").read_text())["network"][
            "ca_bundle"
        ]
        == expected
    )


def test_repeated_sampling_callback_failure_still_restores_disk(setup):
    setup.settings.save()
    path = setup.settings.base_path / "settings.json"
    before = json.loads(path.read_text())
    setup.context.apply_sampling.side_effect = RuntimeError("always fails")
    response = setup.client.patch(
        "/management/v1/server/settings", json={"sampling": {"temperature": 0.2}}
    )
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "rollback_failed"
    assert "sampling callback" in response.json()["detail"]["message"]
    assert "persisted settings restored" in response.json()["detail"]["message"]
    assert setup.context.apply_sampling.call_count == 2
    assert setup.settings.sampling.temperature == 1.0
    assert json.loads(path.read_text()) == before


def test_disk_rollback_failure_reports_runtime_restoration_separately(
    setup, monkeypatch
):
    setup.settings.save()
    original_save = GlobalSettings._save_data
    saves = []

    def failing_restore(settings, data):
        saves.append(data)
        if len(saves) == 2:
            raise OSError("rollback disk failure")
        return original_save(settings, data)

    monkeypatch.setattr(GlobalSettings, "_save_data", failing_restore)
    setup.context.apply_sampling.side_effect = [RuntimeError("apply failed"), None]
    response = setup.client.patch(
        "/management/v1/server/settings", json={"sampling": {"temperature": 0.2}}
    )
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "rollback_failed"
    assert (
        "persisted settings restoration failed" in response.json()["detail"]["message"]
    )
    assert setup.context.apply_sampling.call_count == 2
    assert setup.settings.sampling.temperature == 1.0
    assert (
        json.loads((setup.settings.base_path / "settings.json").read_text())[
            "sampling"
        ]["temperature"]
        == 0.2
    )


@pytest.fixture
def resource_reads(setup, monkeypatch):
    from omlx import process_memory_enforcer as memory

    gib = 1024**3
    monkeypatch.setattr("omlx.settings.get_system_memory", lambda: 64 * gib)
    monkeypatch.setattr("omlx.settings.detect_system_memory", lambda: 64 * gib)
    monkeypatch.setattr(
        "omlx.utils.psutil_compat.virtual_memory",
        lambda: SimpleNamespace(available=20 * gib),
    )
    monkeypatch.setattr(memory, "get_effective_metal_cap_bytes", lambda: 48 * gib)
    previews = {
        tier: {
            "reserve_bytes": 4 * gib,
            "free_bytes": 8 * gib,
            "inactive_bytes": 12 * gib,
            "other_apps_bytes": 10 * gib,
            "static_bytes": 60 * gib,
            "dynamic_bytes": 40 * gib if tier != "custom" else 0,
            "metal_cap_bytes": 49 * gib,
            "ceiling_bytes": 40 * gib if tier != "custom" else 49 * gib,
            "binding": "dynamic" if tier != "custom" else "metal_cap",
        }
        for tier in ("safe", "balanced", "aggressive", "custom")
    }
    monkeypatch.setattr(memory, "preview_tier_ceilings", Mock(return_value=previews))
    forbidden = Mock(
        side_effect=AssertionError(
            "Resource preview must never execute commands or change MLX limits"
        )
    )
    monkeypatch.setattr(memory.subprocess, "run", forbidden)
    monkeypatch.setattr(memory.mx, "set_wired_limit", forbidden)
    monkeypatch.setattr(memory.ProcessMemoryEnforcer, "start", forbidden)
    setup.context.engine_pool._process_memory_enforcer = None
    return SimpleNamespace(gib=gib, memory=memory, forbidden=forbidden)


def test_resources_distinguishes_saved_active_and_unsaved_draft(setup, resource_reads):
    gib = resource_reads.gib
    setup.settings.save()
    setup.settings.memory.memory_guard_tier = "aggressive"
    setup.state.process_memory_enforcer = SimpleNamespace(
        _memory_guard_tier="safe",
        _memory_guard_custom_ceiling_bytes=0,
        _prefill_memory_guard=True,
        _metal_wired_limit_request=60 * gib,
        get_ceiling_breakdown=lambda: {
            "static": 58 * gib,
            "dynamic": 35 * gib,
            "metal_cap": 49 * gib,
            "hard_limit": 35 * gib,
        },
    )
    before = (setup.settings.base_path / "settings.json").read_text()
    response = setup.client.get(
        "/management/v1/server/resources?tier=custom&custom_ceiling_gb=32"
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["hardware"] == {
        "physical_memory_bytes": 64 * gib,
        "available_memory_bytes": 20 * gib,
        "metal_cap_bytes": 48 * gib,
    }
    assert data["saved"]["tier"] == "balanced"
    assert data["runtime"]["tier"] == "safe"
    assert data["runtime"]["ceiling_bytes"] == 35 * gib
    assert data["draft"]["tier"] == "custom"
    assert data["draft"]["preview"]["dynamic_bytes"] == 32 * gib
    assert data["draft"]["preview"]["ceiling_bytes"] == 32 * gib
    assert data["draft"]["preview"]["binding"] == "dynamic"
    assert setup.settings.memory.memory_guard_tier == "aggressive"
    assert (setup.settings.base_path / "settings.json").read_text() == before
    resource_reads.forbidden.assert_not_called()


def test_custom_resource_preview_clamps_to_real_static_and_metal_terms(
    setup, resource_reads
):
    response = setup.client.get(
        "/management/v1/server/resources?tier=custom&custom_ceiling_gb=999"
    )
    assert response.status_code == 200
    preview = response.json()["draft"]["preview"]
    assert preview["ceiling_bytes"] == 49 * resource_reads.gib
    assert preview["binding"] == "metal_cap"
    assert response.json()["runtime"]["available"] is False
    assert response.json()["wired_limit"]["command"] is None
    resource_reads.forbidden.assert_not_called()


def test_wired_command_is_copy_only_ram_clamped_and_mib_aligned(setup, resource_reads):
    gib = resource_reads.gib
    setup.state.process_memory_enforcer = SimpleNamespace(
        _memory_guard_tier="custom",
        _memory_guard_custom_ceiling_bytes=63 * gib,
        _prefill_memory_guard=True,
        _metal_wired_limit_request=63 * gib + 12345,
        get_ceiling_breakdown=lambda: {
            "static": 62 * gib,
            "dynamic": 63 * gib,
            "metal_cap": 49 * gib,
            "hard_limit": 49 * gib,
        },
    )
    response = setup.client.get("/management/v1/server/resources")
    assert response.status_code == 200, response.text
    wired = response.json()["wired_limit"]
    expected = (64 * gib - (64 * gib) // 20) // 1024**2 * 1024**2
    assert wired["limited"] is True
    assert wired["recommended_bytes"] == expected
    assert wired["recommended_bytes"] % 1024**2 == 0
    assert wired["recommended_mib"] == expected // 1024**2
    assert wired["command"] == f"sudo sysctl iogpu.wired_limit_mb={expected//1024**2}"
    assert wired["copy_only"] is True
    assert response.json()["warnings"]
    resource_reads.forbidden.assert_not_called()


@pytest.mark.parametrize(
    "query",
    [
        "tier=unknown",
        "tier=custom&custom_ceiling_gb=0",
        "custom_ceiling_gb=-1",
        "custom_ceiling_gb=nan",
        "custom_ceiling_gb=inf",
        "tier=custom",
    ],
)
def test_resource_preview_rejects_invalid_drafts_without_hardware_work(
    setup, resource_reads, query
):
    response = setup.client.get("/management/v1/server/resources?" + query)
    assert response.status_code == 422
    resource_reads.memory.preview_tier_ceilings.assert_not_called()
    resource_reads.forbidden.assert_not_called()


def test_resources_unavailable_hardware_never_fakes_a_limit_or_command(
    setup, resource_reads, monkeypatch
):
    monkeypatch.setattr(
        "omlx.settings.detect_system_memory", Mock(side_effect=OSError("unavailable"))
    )
    monkeypatch.setattr(
        resource_reads.memory, "get_effective_metal_cap_bytes", lambda: 0
    )
    monkeypatch.setattr(
        resource_reads.memory,
        "preview_tier_ceilings",
        Mock(side_effect=RuntimeError("unavailable")),
    )
    response = setup.client.get("/management/v1/server/resources")
    assert response.status_code == 200
    data = response.json()
    assert data["hardware"]["physical_memory_bytes"] is None
    assert data["hardware"]["metal_cap_bytes"] is None
    assert data["draft"]["preview"] is None
    assert data["wired_limit"]["limited"] is None
    assert data["wired_limit"]["command"] is None
    assert data["warnings"]


def test_resources_requires_main_key(setup, resource_reads):
    setup.svc.subkey({"key": "resources-sub"})
    setup.client.headers["Authorization"] = "Bearer resources-sub"
    assert setup.client.get("/management/v1/server/resources").status_code == 401
    resource_reads.memory.preview_tier_ceilings.assert_not_called()


def test_actual_ram_detection_failure_is_unavailable_without_runtime_fallback(
    setup, monkeypatch
):
    from omlx import process_memory_enforcer as memory
    from omlx.settings import detect_system_memory, get_system_memory

    # Exercise the actual detector's failure path, rather than replacing it.
    monkeypatch.setattr("omlx.settings.os.sysconf", lambda _: 0)
    monkeypatch.setattr("omlx.utils.psutil_compat.get_total_memory", lambda: 0)
    monkeypatch.setattr(
        "omlx.utils.psutil_compat.virtual_memory",
        lambda: SimpleNamespace(available=2 * 1024**3),
    )
    monkeypatch.setattr(memory, "get_effective_metal_cap_bytes", lambda: 8 * 1024**3)
    previews = Mock(side_effect=AssertionError("Guessed RAM cannot power previews"))
    monkeypatch.setattr(memory, "preview_tier_ceilings", previews)
    setup.state.process_memory_enforcer = SimpleNamespace(
        _memory_guard_tier="balanced",
        _memory_guard_custom_ceiling_bytes=0,
        _prefill_memory_guard=True,
        _metal_wired_limit_request=15 * 1024**3,
        get_ceiling_breakdown=lambda: {
            "static": 13 * 1024**3,
            "dynamic": 12 * 1024**3,
            "metal_cap": 8 * 1024**3,
            "hard_limit": 8 * 1024**3,
        },
    )
    assert detect_system_memory() is None
    assert get_system_memory() == 16 * 1024**3
    response = setup.client.get("/management/v1/server/resources")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["hardware"]["physical_memory_bytes"] is None
    assert data["tier_previews"] == {}
    assert data["saved"]["preview"] is None
    assert data["draft"]["preview"] is None
    assert data["wired_limit"]["recommended_bytes"] is None
    assert data["wired_limit"]["command"] is None
    assert "Physical memory unavailable" in data["warnings"]
    previews.assert_not_called()


def test_resource_draft_guard_query_does_not_change_saved_or_active_guard(
    setup, resource_reads
):
    gib = resource_reads.gib
    setup.settings.memory.prefill_memory_guard = False
    setup.settings.save()
    setup.state.process_memory_enforcer = SimpleNamespace(
        _memory_guard_tier="balanced",
        _memory_guard_custom_ceiling_bytes=0,
        _prefill_memory_guard=False,
        _metal_wired_limit_request=0,
        get_ceiling_breakdown=lambda: {
            "static": 0,
            "dynamic": 0,
            "metal_cap": 0,
            "hard_limit": 0,
        },
    )
    before = (setup.settings.base_path / "settings.json").read_text()
    response = setup.client.get("/management/v1/server/resources?guard_enabled=true")
    assert response.status_code == 200
    data = response.json()
    assert data["saved"]["guard_enabled"] is False
    assert data["saved"]["preview"]["ceiling_bytes"] == 0
    assert data["runtime"]["guard_enabled"] is False
    assert data["runtime"]["ceiling_bytes"] == 0
    assert data["draft"]["guard_enabled"] is True
    assert data["draft"]["preview"]["ceiling_bytes"] == 40 * gib
    assert setup.settings.memory.prefill_memory_guard is False
    assert (setup.settings.base_path / "settings.json").read_text() == before
    assert (
        setup.client.get(
            "/management/v1/server/resources?guard_enabled=invalid"
        ).status_code
        == 422
    )
    resource_reads.forbidden.assert_not_called()
