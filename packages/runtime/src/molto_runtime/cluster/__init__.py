# SPDX-License-Identifier: Apache-2.0
"""Distributed-cluster control-plane primitives."""

from molto_runtime.cluster.models import (
    CLUSTER_PROTOCOL_VERSION,
    WORKER_PROTOCOL_VERSION,
)

__all__ = ["CLUSTER_PROTOCOL_VERSION", "WORKER_PROTOCOL_VERSION"]
