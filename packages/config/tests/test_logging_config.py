# SPDX-License-Identifier: Apache-2.0
"""Polling logs must not hide failed authentication or state mutations."""

import logging

import pytest
from molto_config.logging_config import ManagementAccessFilter


@pytest.mark.parametrize("path", ["state", "stats", "stats?scope=alltime", "cache"])
def test_suppresses_successful_management_polling(path: str) -> None:
    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        "",
        0,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1", "GET", f"/management/v1/{path}", "1.1", 200),
        None,
    )
    assert ManagementAccessFilter().filter(record) is False


@pytest.mark.parametrize(
    "method,path,status",
    [
        ("GET", "/management/v1/stats", 401),
        ("GET", "/management/v1/stats", 500),
        ("POST", "/management/v1/cache/hot/clear", 200),
        ("GET", "/v1/models", 200),
        ("GET", "/health", 200),
        ("POST", "/v1/chat/completions", 200),
    ],
)
def test_preserves_failures_mutations_and_inference(
    method: str, path: str, status: int
) -> None:
    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        "",
        0,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1", method, path, "1.1", status),
        None,
    )
    assert ManagementAccessFilter().filter(record) is True


def test_preserves_unrecognized_log_format() -> None:
    record = logging.LogRecord(
        "uvicorn.access", logging.INFO, "", 0, "message", (), None
    )
    assert ManagementAccessFilter().filter(record) is True
