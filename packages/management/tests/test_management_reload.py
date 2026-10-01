# SPDX-License-Identifier: Apache-2.0
"""Management refresh re-discovers models from persisted configuration."""

from unittest.mock import MagicMock

import pytest
from omlx_management.management import (
    ManagementContext,
    ManagementError,
    ManagementService,
)


def _service(pool, manager, global_settings, default=None):
    current = {"model": default}
    service = ManagementService(
        ManagementContext(
            engine_pool=pool,
            settings_manager=manager,
            global_settings=global_settings,
            get_default_model=lambda: current["model"],
            set_default_model=lambda model_id: current.__setitem__("model", model_id),
            apply_sampling=lambda: None,
        )
    )
    return service, current


@pytest.mark.asyncio
async def test_refresh_loads_saved_settings_and_discovers_configured_directories():
    pool = MagicMock()
    pool.get_model_ids.return_value = ["model-a", "model-b"]
    pool.model_count = 2
    manager = MagicMock()
    manager.get_pinned_model_ids.return_value = ["model-b"]
    manager.get_default_model_id.return_value = "model-b"
    settings = MagicMock()
    settings.get_effective_model_dirs.return_value = ["/models"]

    service, current = _service(pool, manager, settings, "model-a")
    assert await service.refresh() == {"status": "ok", "model_count": 2}

    manager.reload.assert_called_once_with()
    pool.discover_models.assert_called_once_with(["/models"], ["model-b"])
    pool.apply_settings_overrides.assert_called_once_with(manager)
    assert current["model"] == "model-b"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "preferred, current, available, expected",
    [
        ("missing", "model-b", ["model-a", "model-b"], "model-b"),
        ("missing", "removed", ["model-a"], "model-a"),
        ("missing", "removed", [], None),
    ],
)
async def test_refresh_repairs_default_after_discovery(
    preferred, current, available, expected
):
    pool = MagicMock()
    pool.get_model_ids.return_value = available
    pool.model_count = len(available)
    manager = MagicMock()
    manager.get_pinned_model_ids.return_value = []
    manager.get_default_model_id.return_value = preferred
    settings = MagicMock()
    settings.get_effective_model_dirs.return_value = ["/models"]

    service, pointer = _service(pool, manager, settings, current)
    await service.refresh()

    assert pointer["model"] == expected


@pytest.mark.asyncio
async def test_refresh_refuses_empty_model_directory_configuration():
    pool = MagicMock()
    manager = MagicMock()
    settings = MagicMock()
    settings.get_effective_model_dirs.return_value = []
    service, _ = _service(pool, manager, settings)

    with pytest.raises(ManagementError) as exc_info:
        await service.refresh()

    assert exc_info.value.code == "unavailable"
    pool.discover_models.assert_not_called()


@pytest.mark.asyncio
async def test_refresh_waits_until_preparation_releases_admission():
    pool, manager, settings = MagicMock(), MagicMock(), MagicMock()
    pool._preparation_active = True
    pool.management_operation_allowed.return_value = False
    service, _ = _service(pool, manager, settings)
    with pytest.raises(ManagementError) as error:
        await service.refresh()
    assert error.value.code == "busy"
    manager.reload.assert_not_called()
    pool.discover_models.assert_not_called()
