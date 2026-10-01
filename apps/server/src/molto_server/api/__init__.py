# SPDX-License-Identifier: Apache-2.0
"""
API models, utilities, and tool calling support for Molto.

This module provides shared components used by the server:
- Pydantic models for OpenAI-compatible API
- Pydantic models for Anthropic Messages API
- Utility functions for text processing
- Tool calling parsing and conversion
"""

from molto_contracts.api.anthropic_models import (
    AnthropicErrorDetail,
    AnthropicErrorResponse,
    AnthropicMessage,
    AnthropicTool,
    AnthropicUsage,
    ContentBlockDeltaEvent,
    ContentBlockImage,
    ContentBlockInputAudio,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    ContentBlockText,
    ContentBlockToolResult,
    ContentBlockToolUse,
    ErrorEvent,
    InputJsonDelta,
    MessageDeltaEvent,
    MessageStartEvent,
    MessageStopEvent,
    PingEvent,
    SystemContent,
    TextDelta,
    ThinkingConfig,
    TokenCountRequest,
    TokenCountResponse,
    ToolChoice,
)
from molto_contracts.api.anthropic_models import (
    MessagesRequest as AnthropicMessagesRequest,
)
from molto_contracts.api.anthropic_models import (
    MessagesResponse as AnthropicMessagesResponse,
)
from molto_contracts.api.embedding_models import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)
from molto_contracts.api.openai_models import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    ContentPart,
    FileContent,
    FunctionCall,
    MCPExecuteRequest,
    MCPExecuteResponse,
    MCPServerInfo,
    MCPServersResponse,
    MCPToolInfo,
    MCPToolsResponse,
    Message,
    ModelInfo,
    ModelsResponse,
    ResponseFormat,
    ResponseFormatJsonSchema,
    ToolCall,
    ToolDefinition,
    Usage,
)
from molto_runtime.generation.anthropic_utils import (
    convert_anthropic_to_internal,
    convert_anthropic_tools_to_internal,
    convert_internal_to_anthropic_response,
    create_content_block_start_event,
    create_content_block_stop_event,
    create_error_event,
    create_input_json_delta_event,
    create_message_delta_event,
    create_message_start_event,
    create_message_stop_event,
    create_ping_event,
    create_text_delta_event,
    format_sse_event,
    map_finish_reason_to_stop_reason,
)
from molto_runtime.generation.thinking import ThinkingParser, extract_thinking
from molto_runtime.generation.tool_calling import (
    build_json_system_prompt,
    convert_tools_for_template,
    extract_json_from_text,
    parse_json_output,
    parse_tool_calls,
    validate_json_schema,
)
from molto_runtime.generation.utils import (
    SPECIAL_TOKENS_PATTERN,
    clean_output_text,
    clean_special_tokens,
    extract_text_content,
)

from molto_server.api.embedding_utils import (
    count_tokens,
    encode_embedding_base64,
    normalize_input,
    truncate_embedding,
)

__all__ = [
    "ContentBlockInputAudio",
    # Models
    "ContentPart",
    "FileContent",
    "Message",
    "FunctionCall",
    "ToolCall",
    "ToolDefinition",
    "ResponseFormat",
    "ResponseFormatJsonSchema",
    "ChatCompletionRequest",
    "ChatCompletionChoice",
    "ChatCompletionResponse",
    "AssistantMessage",
    "CompletionRequest",
    "CompletionChoice",
    "CompletionResponse",
    "Usage",
    "ModelInfo",
    "ModelsResponse",
    "MCPToolInfo",
    "MCPToolsResponse",
    "MCPServerInfo",
    "MCPServersResponse",
    "MCPExecuteRequest",
    "MCPExecuteResponse",
    # Utils
    "clean_output_text",
    "clean_special_tokens",
    "extract_text_content",
    "SPECIAL_TOKENS_PATTERN",
    # Thinking
    "ThinkingParser",
    "extract_thinking",
    # Tool calling
    "parse_tool_calls",
    "convert_tools_for_template",
    # Structured output
    "parse_json_output",
    "validate_json_schema",
    "extract_json_from_text",
    "build_json_system_prompt",
    # Anthropic models
    "ContentBlockText",
    "ContentBlockImage",
    "ContentBlockToolUse",
    "ContentBlockToolResult",
    "SystemContent",
    "AnthropicMessage",
    "AnthropicTool",
    "ToolChoice",
    "ThinkingConfig",
    "AnthropicMessagesRequest",
    "AnthropicMessagesResponse",
    "AnthropicUsage",
    "TokenCountRequest",
    "TokenCountResponse",
    "MessageStartEvent",
    "ContentBlockStartEvent",
    "ContentBlockDeltaEvent",
    "ContentBlockStopEvent",
    "MessageDeltaEvent",
    "MessageStopEvent",
    "PingEvent",
    "ErrorEvent",
    "TextDelta",
    "InputJsonDelta",
    "AnthropicErrorResponse",
    "AnthropicErrorDetail",
    # Anthropic utils
    "convert_anthropic_to_internal",
    "convert_anthropic_tools_to_internal",
    "convert_internal_to_anthropic_response",
    "map_finish_reason_to_stop_reason",
    "format_sse_event",
    "create_message_start_event",
    "create_content_block_start_event",
    "create_text_delta_event",
    "create_input_json_delta_event",
    "create_content_block_stop_event",
    "create_message_delta_event",
    "create_message_stop_event",
    "create_ping_event",
    "create_error_event",
    # Embedding models
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingData",
    "EmbeddingUsage",
    # Embedding utils
    "encode_embedding_base64",
    "truncate_embedding",
    "count_tokens",
    "normalize_input",
    # MCP routes
]
