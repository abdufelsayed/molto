# SPDX-License-Identifier: Apache-2.0
"""
Web search / page fetch API routes for the admin chat UI.

- POST /v1/web/search - Search with the configured provider
- POST /v1/web/fetch - Fetch a public page as markdown

Both endpoints always answer HTTP 200 with an {"ok": bool, ...} payload;
failures ride in the payload so the chat model can read and relay them.
The chat page is the only intended caller — these tools are never merged
into /v1/chat/completions tool lists.
"""

from typing import Any

from fastapi import APIRouter, Request
from omlx_management.websearch import (
    failure_payload,
    fetch_content_chars,
    run_fetch_url,
    run_web_search,
)
from pydantic import BaseModel

router = APIRouter(prefix="/v1/web", tags=["web"])


class WebSearchRequest(BaseModel):
    # Defaults keep blank input as a readable payload error instead of a
    # 422 the model cannot interpret.
    query: str = ""


class WebFetchRequest(BaseModel):
    url: str = ""


@router.post("/search")
async def web_search(
    request: WebSearchRequest, http_request: Request
) -> dict[str, Any]:
    """Run a web search and return an {"ok": bool, ...} payload."""
    settings = http_request.app.state.server_state.global_settings
    if settings is None:
        return failure_payload("unexpected_failure", "Server settings are unavailable.")
    return await run_web_search(request.query, settings.integrations)


@router.post("/fetch")
async def web_fetch(request: WebFetchRequest, http_request: Request) -> dict[str, Any]:
    """Fetch a public page and return an {"ok": bool, ...} payload."""
    settings = http_request.app.state.server_state.global_settings
    integrations = settings.integrations if settings is not None else None
    if integrations is not None:
        return await run_fetch_url(
            request.url,
            max_chars=fetch_content_chars(integrations),
            truncate=bool(getattr(integrations, "web_search_content_truncate", True)),
        )
    return await run_fetch_url(request.url)
