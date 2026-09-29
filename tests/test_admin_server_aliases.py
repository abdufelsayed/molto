# SPDX-License-Identifier: Apache-2.0
"""Tests for server alias support: /admin/api/server-info endpoint and
``server_aliases`` save/validate path in /admin/api/global-settings."""

import asyncio
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import omlx.utils.network as network
from omlx.settings import GlobalSettings
from omlx.utils.network import (
    detect_server_aliases,
    is_loopback_bind,
    is_loopback_bind_host,
    is_valid_alias,
    is_valid_bind_host,
    is_valid_hostname,
    is_valid_ip,
    network_auth_error,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_global_settings(
    server_aliases: list[str] | None = None, host: str = "127.0.0.1"
):
    """Build a MagicMock GlobalSettings with the fields the alias paths touch."""
    gs = MagicMock()
    gs.server.host = host
    gs.server.port = 8000
    gs.server.log_level = "info"
    gs.server.server_aliases = list(server_aliases or [])
    gs.server.preserve_mid_system_cache = True
    gs.auth.api_key = None
    gs.auth.skip_api_key_verification = False
    # Validation is invoked at the end of update_global_settings; return no errors.
    gs.validate.return_value = []
    gs.save.return_value = None
    return gs


@contextmanager
def _patched_global_settings(gs):
    """Patch the module-level _get_global_settings getter without disturbing others."""
    if isinstance(gs, MagicMock):
        if not isinstance(gs.server.host, str):
            gs.server.host = "127.0.0.1"
        if not isinstance(gs.auth.api_key, (str, type(None))):
            gs.auth.api_key = None
        if not isinstance(gs.auth.skip_api_key_verification, bool):
            gs.auth.skip_api_key_verification = False
    original = admin_routes._get_global_settings
    admin_routes._get_global_settings = lambda: gs
    try:
        yield
    finally:
        admin_routes._get_global_settings = original


# =============================================================================
# Unit tests for omlx.utils.network
# =============================================================================


class TestNetworkValidation:
    """Validation primitives used by the alias save path."""

    def test_valid_ipv4(self):
        assert is_valid_ip("192.168.1.10")
        assert is_valid_ip("127.0.0.1")

    def test_valid_ipv6(self):
        assert is_valid_ip("::1")
        assert is_valid_ip("fe80::1")

    @pytest.mark.parametrize(
        "host",
        [
            "localhost",
            "LOCALHOST.",
            "127.0.0.1",
            "127.42.0.9",
            "::1",
            "::ffff:127.0.0.1",
        ],
    )
    def test_recognizes_loopback_bind_hosts(self, host):
        assert is_loopback_bind_host(host)

    @pytest.mark.parametrize(
        "host",
        ["0.0.0.0", "::", "192.168.1.10", "host.local", "example.com", ""],
    )
    def test_rejects_non_loopback_bind_hosts(self, host):
        assert not is_loopback_bind_host(host)

    def test_network_bind_requires_api_key(self):
        error = network_auth_error("0.0.0.0", None, False)

        assert error is not None
        assert "API key is required" in error

    def test_mixed_bind_list_requires_api_key(self):
        error = network_auth_error("127.0.0.1,192.168.1.10", None, False)

        assert error is not None
        assert "API key is required" in error

    def test_authenticated_network_bind_is_allowed(self):
        assert network_auth_error("0.0.0.0", "secret-key", False) is None

    def test_loopback_bind_requires_every_configured_host_to_be_loopback(self):
        assert is_loopback_bind("127.0.0.1, ::1")
        assert not is_loopback_bind("127.0.0.1, 192.168.1.10")
        assert not is_loopback_bind("")

    def test_auth_bypass_is_loopback_only(self):
        assert network_auth_error("127.0.0.1,::1", None, True) is None
        error = network_auth_error("0.0.0.0", "secret-key", True)

        assert error is not None
        assert "cannot be skipped" in error

    def test_rejects_unspecified_ipv4(self):
        """0.0.0.0 parses as a valid IP but is not routable as an alias."""
        assert not is_valid_ip("0.0.0.0")

    def test_rejects_unspecified_ipv6(self):
        """:: is the IPv6 unspecified address — also not usable as an alias."""
        assert not is_valid_ip("::")

    def test_rejects_garbage(self):
        assert not is_valid_ip("not-an-ip")
        assert not is_valid_ip("999.999.999.999")

    def test_valid_hostname(self):
        assert is_valid_hostname("example.local")
        assert is_valid_hostname("my-mac")
        assert is_valid_hostname("a.b.c.d")
        assert is_valid_hostname("web1.local")

    def test_rejects_invalid_hostname(self):
        assert not is_valid_hostname("")
        assert not is_valid_hostname("with space")
        assert not is_valid_hostname("-leading-dash")
        assert not is_valid_hostname("a" * 300)

    def test_rejects_all_numeric_last_label_in_dotted_names(self):
        # For dotted (multi-label) names: IANA never delegates numeric TLDs,
        # so an all-digit rightmost label signals an IP-shaped string.
        # Mirrors the approach used by the ``validators`` PyPI library.
        assert not is_valid_hostname("999.999.999.999")
        assert not is_valid_hostname("1.2.3.4")
        assert not is_valid_hostname("host.123")

    def test_accepts_single_label_without_letters(self):
        # Single-label names (no dots) are local hostnames — no TLD constraint.
        assert is_valid_hostname("192-168-1-1")
        assert is_valid_hostname("web1")

    def test_alias_accepts_either(self):
        assert is_valid_alias("localhost")
        assert is_valid_alias("192.168.1.10")
        assert is_valid_alias("foo.local")
        assert is_valid_alias("::1")

    def test_alias_rejects_unspecified(self):
        assert not is_valid_alias("0.0.0.0")
        assert not is_valid_alias("::")

    def test_alias_rejects_non_string(self):
        assert not is_valid_alias(None)  # type: ignore[arg-type]
        assert not is_valid_alias(123)  # type: ignore[arg-type]


class TestIsValidBindHost:
    """is_valid_bind_host() accepts IPs (including unspecified) and hostnames,
    but must reject IP-shaped values that fail IP parsing."""

    # ------------------------------------------------------------------
    # Valid IPv4 — including unspecified/wildcard addresses that are
    # rejected by is_valid_alias() but are legitimate bind targets.
    # ------------------------------------------------------------------

    def test_accepts_regular_ipv4(self):
        assert is_valid_bind_host("127.0.0.1")
        assert is_valid_bind_host("192.168.1.10")
        assert is_valid_bind_host("10.0.0.255")
        assert is_valid_bind_host("255.255.255.255")

    def test_accepts_wildcard_ipv4(self):
        assert is_valid_bind_host("0.0.0.0")

    # ------------------------------------------------------------------
    # Valid IPv6 — the ip-shaped guard uses a digit+dot regex so it
    # never interferes with colon-containing IPv6 addresses.
    # ------------------------------------------------------------------

    def test_accepts_wildcard_ipv6(self):
        assert is_valid_bind_host("::")

    def test_accepts_loopback_ipv6(self):
        assert is_valid_bind_host("::1")

    def test_accepts_link_local_ipv6(self):
        assert is_valid_bind_host("fe80::1")

    def test_accepts_full_ipv6(self):
        assert is_valid_bind_host("2001:db8::1")

    def test_accepts_ipv4_mapped_ipv6(self):
        # ::ffff:192.168.1.1 is valid IPv6 and contains dots, but the
        # colon means it reaches ipaddress.ip_address() first and parses fine.
        assert is_valid_bind_host("::ffff:192.168.1.1")

    # ------------------------------------------------------------------
    # Valid hostnames
    # ------------------------------------------------------------------

    def test_accepts_simple_hostname(self):
        assert is_valid_bind_host("localhost")
        assert is_valid_bind_host("my-host")

    def test_accepts_fqdn(self):
        assert is_valid_bind_host("my-host.local")
        assert is_valid_bind_host("example.com")

    def test_accepts_hostname_with_leading_digit_label(self):
        # Numeric-prefixed labels are valid hostnames (e.g. "web1.local")
        assert is_valid_bind_host("web1.local")

    def test_accepts_hostname_with_dashes_instead_of_dots(self):
        # "192-168-1-1" looks IP-like but uses dashes — valid hostname,
        # does not match the digit-dot regex.
        assert is_valid_bind_host("192-168-1-1")

    def test_accepts_all_letter_dotted_hostname(self):
        # Labels a.b.c.d contain letters so they don't match the ip-shaped guard.
        assert is_valid_bind_host("a.b.c.d")

    # ------------------------------------------------------------------
    # Rejected: IP-shaped strings that fail IP parsing
    # The bug: ipaddress.ip_address() raises ValueError, and digit-only
    # dotted labels also match the hostname regex — so without the guard
    # they would be accepted silently.
    # ------------------------------------------------------------------

    def test_rejects_ipv4_all_octets_out_of_range(self):
        assert not is_valid_bind_host("999.999.999.999")

    def test_rejects_ipv4_first_octet_out_of_range(self):
        assert not is_valid_bind_host("256.0.0.1")

    def test_rejects_ipv4_last_octet_out_of_range(self):
        assert not is_valid_bind_host("1.2.3.999")

    def test_rejects_ip_shaped_too_few_octets(self):
        # 3-part and 2-part dotted numeric strings look IP-shaped but are
        # not valid IPs and must not slip through as hostnames.
        assert not is_valid_bind_host("1.2.3")
        assert not is_valid_bind_host("1.2")

    def test_rejects_ip_shaped_too_many_octets(self):
        assert not is_valid_bind_host("1.2.3.4.5")

    # ------------------------------------------------------------------
    # Rejected: malformed hostnames
    # ------------------------------------------------------------------

    def test_rejects_leading_dash(self):
        assert not is_valid_bind_host("-bad-host")

    def test_rejects_hostname_with_space(self):
        assert not is_valid_bind_host("with space")

    def test_rejects_label_too_long(self):
        assert not is_valid_bind_host("a" * 64 + ".local")

    def test_rejects_value_too_long(self):
        assert not is_valid_bind_host("a." * 127 + "b")

    def test_rejects_invalid_ipv6_form(self):
        # Colon-containing but not a valid IP — falls through to hostname,
        # which rejects colons.
        assert not is_valid_bind_host(":invalid:")
        assert not is_valid_bind_host("[::1]")

    # ------------------------------------------------------------------
    # Rejected: empty / wrong types
    # ------------------------------------------------------------------

    def test_rejects_empty_string(self):
        assert not is_valid_bind_host("")

    def test_rejects_whitespace_only(self):
        assert not is_valid_bind_host("   ")

    def test_rejects_non_string(self):
        assert not is_valid_bind_host(None)  # type: ignore[arg-type]
        assert not is_valid_bind_host(8080)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Whitespace stripping
    # ------------------------------------------------------------------

    def test_strips_surrounding_whitespace(self):
        assert is_valid_bind_host("  127.0.0.1  ")
        assert is_valid_bind_host("  localhost  ")


class TestDetectServerAliases:
    """Auto-detection should always return at least loopback when bound to localhost."""

    def test_localhost_includes_loopback(self):
        aliases = detect_server_aliases(host="127.0.0.1")
        assert "localhost" in aliases
        assert "127.0.0.1" in aliases

    def test_no_unspecified_in_output(self):
        """Even when bound to 0.0.0.0, detection should not return 0.0.0.0 itself."""
        aliases = detect_server_aliases(host="0.0.0.0")
        assert "0.0.0.0" not in aliases
        assert "::" not in aliases

    def test_returns_unique_values(self):
        aliases = detect_server_aliases()
        assert len(aliases) == len(set(aliases))

    def test_comma_separated_host_includes_loopback(self):
        """Comma-separated bind hosts containing a loopback must not drop localhost aliases."""
        aliases = detect_server_aliases(host="127.0.0.1, ::1")
        assert "localhost" in aliases
        assert "127.0.0.1" in aliases

    def test_comma_separated_wildcard_includes_loopback(self):
        aliases = detect_server_aliases(host="0.0.0.0, ::1")
        assert "localhost" in aliases
        assert "127.0.0.1" in aliases

    def test_comma_separated_non_loopback_skips_loopback(self):
        """If no part of the comma-separated host is a loopback/wildcard, no loopback aliases."""
        aliases = detect_server_aliases(host="192.168.1.10, 10.0.0.1")
        assert "localhost" not in aliases

    def test_slow_reverse_lookup_does_not_block(self, monkeypatch):
        """A resolver that never answers costs the FQDN alias, not server startup."""
        release = threading.Event()
        monkeypatch.setattr(network, "_FQDN_TIMEOUT_S", 0.05)
        monkeypatch.setattr(
            network.socket, "getfqdn", lambda: release.wait(5) and "slow.example"
        )
        try:
            start = time.monotonic()
            aliases = detect_server_aliases(host="127.0.0.1")
            assert time.monotonic() - start < 1.0
            assert "localhost" in aliases
            assert "slow.example" not in aliases
        finally:
            release.set()


# =============================================================================
# /admin/api/server-info endpoint
# =============================================================================
