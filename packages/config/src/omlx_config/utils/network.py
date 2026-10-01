# SPDX-License-Identifier: Apache-2.0
"""Server bind-host validation and network authentication checks."""

from __future__ import annotations

import ipaddress
import re

# RFC 1123 hostname label: letters, digits, hyphens; 1-63 chars per label.
# Allows trailing dot. Total length capped at 253.
_HOSTNAME_LABEL = re.compile(r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)$")
# The rightmost label must contain at least one letter.  IANA has never
# delegated a purely numeric TLD, so an all-digit final label (e.g. the
# "999" in "999.999.999.999") indicates an IP-shaped string masquerading
# as a hostname, not a real DNS name.  This mirrors the approach used by
# the ``validators`` library (python-validators on PyPI).
_HOSTNAME_LAST_LABEL_HAS_LETTER = re.compile(r"[A-Za-z]")


def is_valid_hostname(value: str) -> bool:
    """Return True if ``value`` looks like a valid DNS hostname."""
    if not value or len(value) > 253:
        return False
    candidate = value[:-1] if value.endswith(".") else value
    labels = candidate.split(".")
    # For dotted names only: the rightmost label must contain at least one
    # letter.  IANA has never delegated a purely numeric TLD, so an all-digit
    # final label (e.g. "999" in "999.999.999.999") signals an IP-shaped
    # string masquerading as a hostname.  Single-label names (no dots) are
    # local hostnames and are not subject to this TLD constraint.
    if len(labels) > 1 and not _HOSTNAME_LAST_LABEL_HAS_LETTER.search(labels[-1]):
        return False
    return all(_HOSTNAME_LABEL.match(label) for label in labels)


def is_valid_bind_host(value: str) -> bool:
    """Return True if ``value`` is a valid host to bind a server socket to.

    Accepts any parseable IP address (including ``0.0.0.0`` and ``::``) and
    valid DNS hostnames.
    """
    if not isinstance(value, str):
        return False
    value = value.strip()
    if not value:
        return False
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        pass
    # Fast-path rejection for IP-shaped strings that failed IP parsing
    # (e.g. "999.999.999.999").  is_valid_hostname() would also reject them
    # via the last-label letter requirement, but this regex short-circuits
    # before the per-label loop.
    if re.match(r"^\d+(\.\d+)+$", value):
        return False
    return is_valid_hostname(value)


def is_loopback_bind_host(value: str) -> bool:
    """Return whether a bind host is explicitly limited to loopback."""

    if not isinstance(value, str):
        return False
    candidate = value.strip()
    if not candidate:
        return False
    if candidate.rstrip(".").lower() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    return bool(
        isinstance(address, ipaddress.IPv6Address)
        and address.ipv4_mapped is not None
        and address.ipv4_mapped.is_loopback
    )


def is_loopback_bind(value: str) -> bool:
    """Return whether every configured bind host is loopback-only."""

    if not isinstance(value, str):
        return False
    bind_hosts = [part.strip() for part in value.split(",") if part.strip()]
    return bool(bind_hosts) and all(is_loopback_bind_host(part) for part in bind_hosts)


def network_auth_error(
    host: str,
    api_key: str | None,
    skip_api_key_verification: bool,
) -> str | None:
    """Reject unauthenticated servers that bind beyond loopback."""

    if not isinstance(host, str):
        return "Server host must be a string."
    if is_loopback_bind(host):
        return None
    if skip_api_key_verification:
        return (
            "API key verification cannot be skipped when binding to a "
            f"non-loopback host ({host})."
        )
    if not isinstance(api_key, str) or not api_key.strip():
        return (
            f"An API key is required when binding to a non-loopback host ({host}). "
            "Set --api-key or OMLX_API_KEY, or bind to 127.0.0.1."
        )
    return None
