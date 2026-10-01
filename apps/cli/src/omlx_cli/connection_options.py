"""Command-line connection and output flags."""

import argparse
import math


def positive_timeout(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def add_connection_options(parser):
    """Suppress defaults so options before and after a command compose."""
    options = parser.add_argument_group("Connection and output")
    options.add_argument(
        "--url", default=argparse.SUPPRESS, help="Public dashboard origin, or OMLX_URL"
    )
    credentials = options.add_mutually_exclusive_group()
    credentials.add_argument(
        "--api-key",
        default=argparse.SUPPRESS,
        help="Main management key; prefer OMLX_API_KEY or --api-key-file",
    )
    credentials.add_argument(
        "--api-key-file",
        default=argparse.SUPPRESS,
        metavar="PATH",
        help="Read the main key from a private file",
    )
    options.add_argument(
        "--base-path",
        default=argparse.SUPPRESS,
        metavar="PATH",
        help="Local settings directory, or OMLX_BASE_PATH",
    )
    options.add_argument(
        "--timeout",
        type=positive_timeout,
        default=argparse.SUPPRESS,
        help="HTTP timeout in seconds (default: 30)",
    )
    options.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Write machine-readable JSON",
    )
    options.add_argument(
        "--no-color",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Disable terminal colors",
    )
    options.add_argument(
        "--yes",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Confirm a destructive operation without prompting",
    )
