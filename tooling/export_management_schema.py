"""Export wire contracts from routers without creating the inference application."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from omlx_server.api.management_routes import router
from omlx_server.api.management_setup_routes import router as setup_router
from omlx_server.cluster.discovery_routes import discovery_router
from omlx_server.cluster.pairing_routes import pair_admin_router, pair_router
from omlx_server.cluster.routes import router as cluster_router


def export() -> dict:
    app = FastAPI(title="oMLX management API", version="1")
    for routes in (
        router,
        setup_router,
        cluster_router,
        discovery_router,
        pair_router,
        pair_admin_router,
    ):
        app.include_router(routes)
    return app.openapi()


if __name__ == "__main__":
    destination = (
        Path(__file__).resolve().parents[1]
        / "packages/contracts/generated/openapi.json"
    )
    destination.write_text(json.dumps(export(), indent=2) + "\n")
