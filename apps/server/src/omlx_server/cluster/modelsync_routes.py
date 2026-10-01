# SPDX-License-Identifier: Apache-2.0
"""Server-owned model manifest HTTP routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from omlx_runtime.cluster.modelsync import ModelSyncManager, build_manifest
from omlx_runtime.exceptions import ModelNotFoundError

# -- manifest endpoint ------------------------------------------------------

manifest_router = APIRouter(prefix="/api/cluster", tags=["cluster-modelsync"])


def _manager(http_request: Request) -> ModelSyncManager:
    controller = http_request.app.state.controller
    return ModelSyncManager(
        pool_getter=controller.get_engine_pool,
        settings_loader=lambda: controller.state.global_settings,
    )


def _allowed_model_roots(http_request: Request) -> tuple[Path, ...]:
    try:
        settings = http_request.app.state.controller.state.global_settings
        return tuple(
            Path(entry).expanduser().resolve()
            for entry in settings.get_effective_model_dirs()
        )
    except Exception:  # noqa: BLE001
        return ()


def _guard_manifest_path(path: Path, http_request: Request) -> None:
    """Path-typed model IDs stay inside the configured model directories.

    Repo-style IDs are resolved through the engine pool / discovery, which
    already only surfaces registered models. A raw path must not let an admin
    token enumerate arbitrary directories.
    """

    roots = _allowed_model_roots(http_request)
    if not roots:
        return
    if not any(root == path or root in path.parents for root in roots):
        raise HTTPException(
            status_code=403,
            detail="model path is outside the configured model directories",
        )


@manifest_router.get("/models/{model_id:path}/manifest")
async def cluster_model_manifest(
    model_id: str, http_request: Request
) -> dict[str, Any]:
    """Safetensors index hash + file list + total bytes for a local model.

    Peers call this to decide whether they must sync before activation; the
    payload is names and sizes only — never weights, never credentials.
    """

    import asyncio

    manager = _manager(http_request)
    try:
        path = manager.resolve_local_model_path(model_id)
    except ModelNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"unknown model: {model_id}"
        ) from exc
    _guard_manifest_path(path, http_request)
    try:
        manifest = await asyncio.to_thread(build_manifest, path, model_id=model_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return manifest.to_dict()
