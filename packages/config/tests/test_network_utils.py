# SPDX-License-Identifier: Apache-2.0
"""Tests for server bind-host validation and network authentication."""

import pytest
from molto_config.utils.network import (
    is_loopback_bind,
    is_loopback_bind_host,
    is_valid_bind_host,
    is_valid_hostname,
    network_auth_error,
)

# =============================================================================
# Unit tests for molto_config.utils.network
# =============================================================================


class TestNetworkValidation:
    """Validation primitives used by server startup."""

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


class TestIsValidBindHost:
    """is_valid_bind_host() accepts IPs (including unspecified) and hostnames,
    but must reject IP-shaped values that fail IP parsing."""

    # ------------------------------------------------------------------
    # Valid IPv4 — including unspecified/wildcard bind addresses.
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
