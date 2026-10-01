# SPDX-License-Identifier: Apache-2.0
"""Admin handlers for the coordinator's RDMA links, registered on the cluster router."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from fastapi import HTTPException, Request
from omlx_runtime.cluster.rdma.daemon import read_status
from omlx_runtime.cluster.rdma.launch_links import (
    VERIFYING,
    claim_link,
    claimed_links,
    describe_owner,
    release_link,
)
from omlx_runtime.cluster.rdma.link_probe import FULL, verify_link
from omlx_runtime.cluster.rdma.links import discover_links
from omlx_runtime.cluster.rdma.store import RdmaLinkStore
from omlx_runtime.cluster.rdma.verification import (
    cached_driver_identity,
    link_identity,
    read_driver_identity,
)
from omlx_runtime.cluster.rdma.words import load_word_ops
from pydantic import BaseModel, ConfigDict, Field

from omlx_server.cluster.services import get_cluster_enrollment, get_rdma_link_store

# claim_link lets an owner claim again, and every dashboard verification is VERIFYING.
_verify_claim_lock = threading.Lock()


class RdmaLinkVerifyRequest(BaseModel):
    """Which link to verify."""

    model_config = ConfigDict(extra="forbid")

    link: str = Field(min_length=1, max_length=20, pattern=r"^[A-Za-z0-9_-]+$")


def _store(http_request: Request) -> RdmaLinkStore | None:
    try:
        return get_rdma_link_store(http_request)
    except RuntimeError:
        return None


def _inventory(http_request: Request) -> dict[str, Any]:
    status = read_status()
    links = discover_links(status, enrolled_node_addresses(http_request))
    store = _store(http_request)
    records = store.all() if store is not None else {}
    driver = cached_driver_identity()
    ops, ops_reason = load_word_ops()
    claims = claimed_links()
    now = time.time()
    rows = []
    for link in links:
        record = records.get(link.name)
        stale = (
            record.stale_reason(link_identity(link, status, driver), now)
            if record
            else "never verified"
        )
        holder = claims.get(link.name)
        rows.append(
            {
                **link.to_dict(),
                "in_use_by": holder if holder != VERIFYING else None,
                "verifying": holder == VERIFYING,
                "verified": stale is None,
                "stale_reason": stale,
                "verification": record.to_dict() if record else None,
            }
        )
    return {
        "daemon": status.to_dict(),
        "helper": {"available": ops is not None, "reason": ops_reason},
        "driver": {"version": driver.version, "uuid": driver.uuid} if driver else None,
        "store_error": store.load_error
        if store is not None
        else "RDMA link store is not configured",
        "links": rows,
    }


def _verify(http_request: Request, name: str) -> dict[str, Any]:
    status = read_status()
    if not status.reachable:
        raise HTTPException(status_code=409, detail=status.reason)
    nodes = enrolled_node_addresses(http_request)
    link = next(
        (item for item in discover_links(status, nodes) if item.name == name), None
    )
    if link is None:
        raise HTTPException(
            status_code=404, detail=f"mcdma-rpcd has no link named {name}"
        )
    node = next((item for item in nodes if item.node_id == link.peer_node_id), None)
    if node is None:
        raise HTTPException(
            status_code=409,
            detail=link.reason or f"link {name} does not reach an enrolled worker",
        )
    with _verify_claim_lock:
        owner = claimed_links().get(name) or claim_link(name, VERIFYING)
    if owner is not None:
        raise HTTPException(
            status_code=409, detail=f"link {name} is in use by {describe_owner(owner)}"
        )
    try:
        ops, ops_reason = load_word_ops()
        verification = verify_link(
            link,
            node,
            status=status,
            driver=read_driver_identity(),
            ops=ops,
            ops_reason=ops_reason,
            settings=FULL,
        )
    finally:
        release_link(name, VERIFYING)
    store = _store(http_request)
    if store is not None:
        store.record(verification)
    return verification.to_dict()


async def cluster_rdma_links(http_request: Request) -> dict[str, Any]:
    """The coordinator's RDMA links, each with its latest evidence and whether it still holds."""
    return await asyncio.to_thread(_inventory, http_request)


async def cluster_rdma_link_verify(
    http_request: Request, request: RdmaLinkVerifyRequest
) -> dict[str, Any]:
    """Probe one link live in both directions and record the result."""
    return await asyncio.to_thread(_verify, http_request, request.link)


def enrolled_node_addresses(http_request: Request):
    from omlx_runtime.cluster.rdma.links import NodeAddress

    try:
        nodes = get_cluster_enrollment(http_request).list_nodes()
    except RuntimeError:
        return ()
    return tuple(
        NodeAddress(
            node_id=node.node_id,
            ssh=node.ssh,
            addresses=tuple(node.addresses),
            hostname=node.hostname,
            python_executable=node.python_executable,
        )
        for node in nodes
    )
