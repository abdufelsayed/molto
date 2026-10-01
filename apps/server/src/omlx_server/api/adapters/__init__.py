# SPDX-License-Identifier: Apache-2.0
"""
API Adapters for oMLX.

This package provides adapters for different API formats (OpenAI, Anthropic),
enabling clean separation between API-specific logic and core inference.
"""

from omlx_server.api.adapters.anthropic import AnthropicAdapter
from omlx_server.api.adapters.base import (
    BaseAdapter,
    InternalMessage,
    InternalRequest,
    InternalResponse,
    StreamChunk,
)
from omlx_server.api.adapters.openai import OpenAIAdapter
from omlx_server.api.adapters.sse_formatter import (
    AnthropicSSEFormatter,
    OpenAISSEFormatter,
    SSEFormatter,
)

__all__ = [
    # Base classes and types
    "BaseAdapter",
    "InternalMessage",
    "InternalRequest",
    "InternalResponse",
    "StreamChunk",
    # Adapters
    "OpenAIAdapter",
    "AnthropicAdapter",
    # SSE Formatters
    "SSEFormatter",
    "OpenAISSEFormatter",
    "AnthropicSSEFormatter",
]
