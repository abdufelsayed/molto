# SPDX-License-Identifier: Apache-2.0
"""API-key authentication shared by inference and management.

Management does not accept browser sessions. Its main-key privilege is
deliberately distinct from inference access granted by subkeys.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import HTTPException, Request
from omlx_config.utils.network import is_loopback_bind


def compare_keys(provided_key: str, expected_key: str) -> bool:
    """Compare UTF-8 bytes so malformed Unicode remains an auth failure."""
    return secrets.compare_digest(
        provided_key.encode("utf-8", "surrogatepass"),
        expected_key.encode("utf-8", "surrogatepass"),
    )


def fingerprint_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8", "surrogatepass")).hexdigest()[:8]


def verify_api_key(api_key: str, server_api_key: str) -> bool:
    return bool(api_key and server_api_key and compare_keys(api_key, server_api_key))


def verify_any_api_key(api_key: str, main_key: str | None, sub_keys: list) -> bool:
    if not api_key:
        return False
    if main_key and compare_keys(api_key, main_key):
        return True
    return any(
        isinstance(getattr(entry, "key", None), str)
        and bool(entry.key)
        and compare_keys(api_key, entry.key)
        for entry in sub_keys
    )


def validate_api_key(api_key: str) -> tuple[bool, str]:
    if len(api_key) < 4:
        return False, "API key must be at least 4 characters"
    if any(char.isspace() for char in api_key):
        return False, "API key must not contain whitespace"
    if not api_key.isprintable():
        return False, "API key must contain only printable characters"
    if not api_key.isascii():
        return False, "API key must contain only ASCII characters"
    return True, ""


@dataclass(frozen=True)
class AuthContext:
    main_key: str | None
    sub_keys: list
    bind_host: str | None
    skip_api_key_verification: bool = False


def _auth_context(request: Request) -> AuthContext:
    provider: Callable[[], AuthContext] | None = getattr(
        request.app.state, "management_auth_provider", None
    )
    if provider is None:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return provider()


def _bearer_token(request: Request) -> str | None:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    return token if scheme.lower() == "bearer" and token else None


def _require_key(request: Request, *, allow_subkeys: bool) -> bool:
    context = _auth_context(request)
    token = _bearer_token(request)
    # A management client presenting a key must authenticate that key, even
    # when direct local management without a key is explicitly enabled.
    if (
        context.skip_api_key_verification
        and is_loopback_bind(context.bind_host)
        and (allow_subkeys or token is None)
    ):
        return True
    if token is not None and (
        verify_any_api_key(token, context.main_key, context.sub_keys)
        if allow_subkeys
        else verify_api_key(token, context.main_key or "")
    ):
        return True
    raise HTTPException(
        status_code=401,
        detail="Management API key required",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def require_management_key(request: Request) -> bool:
    """Require the main API key for management reads and mutations."""
    return _require_key(request, allow_subkeys=False)


async def require_model_load_key(request: Request) -> bool:
    """Preserve CLI model-load access for inference subkeys."""
    return _require_key(request, allow_subkeys=True)
