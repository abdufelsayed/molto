# SPDX-License-Identifier: Apache-2.0
"""
MCP (Model Context Protocol) client support for oMLX.

This module provides integration with MCP servers, allowing the omlx server
to discover and execute tools from external MCP servers.

Example usage:
    from omlx_server.mcp import MCPClientManager, load_mcp_config

    config = load_mcp_config("./mcp.json")
    manager = MCPClientManager(config)
    await manager.start()

    # Get all available tools in OpenAI format
    tools = manager.get_all_tools()

    # Execute a tool call
    result = await manager.execute_tool("filesystem__read_file", {"path": "/tmp/test.txt"})
"""

from omlx_server.mcp.client import MCPClient
from omlx_server.mcp.config import load_mcp_config, validate_config
from omlx_server.mcp.executor import ToolExecutor
from omlx_server.mcp.manager import MCPClientManager
from omlx_server.mcp.tools import (
    format_tool_result,
    mcp_tool_to_openai,
    openai_call_to_mcp,
)
from omlx_server.mcp.types import (
    MCPConfig,
    MCPServerConfig,
    MCPServerStatus,
    MCPTool,
    MCPToolResult,
)

__all__ = [
    # Types
    "MCPServerConfig",
    "MCPConfig",
    "MCPTool",
    "MCPToolResult",
    "MCPServerStatus",
    # Config
    "load_mcp_config",
    "validate_config",
    # Client
    "MCPClient",
    "MCPClientManager",
    # Tools
    "mcp_tool_to_openai",
    "openai_call_to_mcp",
    "format_tool_result",
    # Executor
    "ToolExecutor",
]
