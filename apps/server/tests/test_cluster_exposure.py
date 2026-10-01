# SPDX-License-Identifier: Apache-2.0
"""The distributed surface remains dark until explicitly enabled."""

from __future__ import annotations

from types import SimpleNamespace

from omlx_runtime.cluster.exposure import distributed_inference_enabled
from repo_paths import repository_root

ROOT = repository_root(__file__)


def _settings(enabled: bool):
    return SimpleNamespace(
        server=SimpleNamespace(distributed_inference_enabled=enabled)
    )


def test_distributed_inference_is_disabled_without_settings():
    assert distributed_inference_enabled(None) is False


def test_distributed_inference_requires_explicit_opt_in():
    assert distributed_inference_enabled(_settings(False)) is False
    assert distributed_inference_enabled(_settings(True)) is True


def test_server_uses_one_startup_snapshot_for_routes_and_bonjour():
    source = (ROOT / "apps/server/src/omlx_server/composition.py").read_text()

    assert (
        "self.state.distributed_inference_enabled = is_enabled(global_settings)"
        in source
    )
    assert "if self.distributed_inference_enabled():" in source
    assert "self._register_cluster_routes()" in source
    assert "Depends(self.require_distributed_inference_enabled)" in source
    assert (
        "self.state.global_settings is not None\n"
        "            and self.distributed_inference_enabled()"
    ) in source


def test_worker_join_routes_use_enrollment_auth_not_the_admin_cookie():
    source = (ROOT / "apps/server/src/omlx_server/composition.py").read_text()

    assert (
        "from omlx_server.cluster.routes import join_router as cluster_join_router"
        in source
    )
    assert (
        "cluster_join_router,\n"
        "            dependencies=[Depends(self.require_distributed_inference_enabled)]"
    ) in source
    assert "ClusterServices.create(base_path)" in source
