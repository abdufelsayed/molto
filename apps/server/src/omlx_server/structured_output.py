# SPDX-License-Identifier: Apache-2.0
"""Structured output for the private inference application."""

import logging

from fastapi import HTTPException

# Import from new modular API
from omlx_runtime.engine import BaseEngine
from omlx_runtime.exceptions import InvalidRequestError

logger = logging.getLogger(__name__)


def _inject_json_instruction(messages: list, instruction: str) -> list:
    """
    Inject JSON instruction into messages.

    If a system message exists, append to it. Otherwise, prepend a new system message.
    """
    messages = list(messages)  # Make a copy

    # Only attach to a leading system message. A mid-conversation system
    # message may be intentionally placed there to preserve prefix cache hits.
    system_idx = None
    if messages:
        first = messages[0]
        role = (
            first.get("role")
            if isinstance(first, dict)
            else getattr(first, "role", None)
        )
        if role == "system":
            system_idx = 0

    if system_idx is not None:
        # Append to existing system message
        msg = messages[system_idx]
        if isinstance(msg, dict):
            existing = msg.get("content", "")
            msg["content"] = f"{existing}\n\n{instruction}"
        else:
            existing = getattr(msg, "content", "") or ""
            msg.content = f"{existing}\n\n{instruction}"
    else:
        # Prepend new system message
        messages.insert(0, {"role": "system", "content": instruction})

    return messages


def _normalize_structured_outputs(
    structured_outputs=None, guided_grammar: str | None = None
):
    """Fold guided_grammar into the existing structured_outputs grammar shape."""
    if structured_outputs is not None:
        return structured_outputs
    if guided_grammar:
        return {"grammar": guided_grammar}
    return None


def _reject_diffusion_structured_outputs(
    engine: BaseEngine,
    *,
    response_format=None,
    structured_outputs=None,
    guided_grammar: str | None = None,
) -> None:
    if not getattr(engine, "is_diffusion_model", False):
        return
    # ``response_format`` (json_object / json_schema) is NOT rejected here:
    # it degrades to prompt-injected JSON with a Warning header, the same
    # fallback used when xgrammar is unavailable (#1241).  Only explicit
    # grammar requests — ``structured_outputs`` and ``guided_grammar`` —
    # are rejected, because logit-mask enforcement has no equivalent in
    # the parallel denoising loop.
    if structured_outputs is None and not guided_grammar:
        return
    raise InvalidRequestError(
        "structured_outputs and guided grammar are not supported "
        "with diffusion models (response_format degrades to "
        "prompt-injected JSON).",
        field="response_format",
    )


def _settings_guided_grammar(settings) -> str | None:
    """Return a non-empty enabled model-level guided grammar."""
    if not settings:
        return None
    if not getattr(settings, "guided_grammar_enabled", False):
        return None
    grammar = getattr(settings, "guided_grammar", None)
    if not grammar:
        return None
    grammar = grammar.strip()
    return grammar or None


def _effective_guided_grammar(
    structured_outputs=None,
    response_format=None,
    request_guided_grammar: str | None = None,
    settings_guided_grammar: str | None = None,
) -> str | None:
    """Choose the request grammar alias or eligible model default."""
    if request_guided_grammar:
        return request_guided_grammar
    if structured_outputs is None and response_format is None:
        return settings_guided_grammar
    return None


def _build_format_element(structured_outputs=None, response_format=None):
    """Build an xgrammar structural-tag format element from the request.

    Returns a format dict (e.g. ``{"type": "json_schema", ...}``) suitable
    for embedding in a structural tag, or ``None`` if no grammar is needed.
    Also returns ``"bare"`` compilation hint when the grammar should be
    compiled directly (EBNF / regex / choice) rather than via structural tag.
    """
    import json as _json

    from omlx_contracts.api.openai_models import StructuredOutputOptions

    if structured_outputs is not None:
        if isinstance(structured_outputs, dict):
            structured_outputs = StructuredOutputOptions(**structured_outputs)

        if structured_outputs.json_schema is not None:
            schema = structured_outputs.json_schema
            if isinstance(schema, str):
                schema = _json.loads(schema)
            return {"type": "json_schema", "json_schema": schema}
        if structured_outputs.grammar is not None:
            return {"type": "grammar", "grammar": structured_outputs.grammar}
        if structured_outputs.regex is not None:
            return {"type": "regex", "pattern": structured_outputs.regex}
        if structured_outputs.choice is not None:
            ebnf = "root ::= " + " | ".join(
                _json.dumps(c) for c in structured_outputs.choice
            )
            return {"type": "grammar", "grammar": ebnf}

    if response_format is not None:
        rf = response_format
        rf_type = rf.get("type") if isinstance(rf, dict) else getattr(rf, "type", None)
        if rf_type == "json_schema":
            js = (
                rf.get("json_schema")
                if isinstance(rf, dict)
                else getattr(rf, "json_schema", None)
            )
            if js is not None:
                schema = (
                    js.get("schema")
                    if isinstance(js, dict)
                    else getattr(js, "schema_", None)
                )
                if schema is not None:
                    return {"type": "json_schema", "json_schema": schema}
        elif rf_type == "json_object":
            return {"type": "json_schema", "json_schema": {}}

    return None


def _patch_output_format(tag_dict: dict, user_grammar: dict) -> bool:
    """Replace the output ``any_text`` slot in a builtin structural tag.

    Walks the structural tag dict produced by
    ``xgrammar.get_builtin_structural_tag`` and swaps the ``any_text``
    element that represents the model's output with ``user_grammar``.

    Returns ``True`` if a replacement was made.
    """
    fmt = tag_dict.get("format", tag_dict)

    if fmt.get("type") == "any_text":
        tag_dict["format"] = user_grammar
        return True

    if fmt.get("type") == "sequence":
        for i in range(len(fmt["elements"]) - 1, -1, -1):
            if fmt["elements"][i].get("type") == "any_text":
                fmt["elements"][i] = user_grammar
                return True

    if fmt.get("type") == "tags_with_separator":
        for tag in reversed(fmt["tags"]):
            if tag.get("type") == "tag" and "final" in tag.get("begin", ""):
                tag["content"] = user_grammar
                return True
        if fmt["tags"]:
            fmt["tags"][-1]["content"] = user_grammar
            return True

    return False


def _compile_with_structural_tag(
    compiler, fmt: dict, reasoning_parser: str, chat_template_kwargs: dict | None
):
    """Compile a grammar wrapped in an xgrammar builtin structural tag.

    Uses ``xgrammar.get_builtin_structural_tag`` to obtain the model's
    protocol structure (thinking tags, channel markers, etc.) and patches
    the user's grammar into the output slot.
    """
    from omlx_runtime._torch_stub import install as _install_torch_stub

    _install_torch_stub()
    import xgrammar as xgr

    reasoning = not (
        chat_template_kwargs and chat_template_kwargs.get("enable_thinking") is False
    )
    tag = xgr.get_builtin_structural_tag(reasoning_parser, reasoning=reasoning)
    tag_dict = tag.model_dump()
    if not _patch_output_format(tag_dict, fmt):
        logger.warning(
            "Could not patch output format for reasoning_parser=%s, "
            "compiling structural tag as-is",
            reasoning_parser,
        )
    return compiler.compile_structural_tag(tag_dict)


def _compile_bare_grammar(compiler, fmt: dict):
    """Compile a grammar without any structural tag wrapping."""
    if fmt["type"] == "json_schema":
        import json as _json

        schema = fmt["json_schema"]
        if not schema:
            return compiler.compile_builtin_json_grammar()
        schema_str = _json.dumps(schema) if isinstance(schema, dict) else schema
        return compiler.compile_json_schema(schema_str)
    elif fmt["type"] == "grammar":
        return compiler.compile_grammar(fmt["grammar"])
    elif fmt["type"] == "regex":
        return compiler.compile_regex(fmt["pattern"])
    return None


def _response_format_requests_strict(response_format) -> bool:
    """True when an OpenAI ``response_format`` demands strict json_schema output.

    A ``json_schema`` response_format with ``strict: true`` signals that the
    caller expects schema-conformant output, not best-effort.  When
    grammar-constrained decoding is unavailable the request still falls back to
    prompt injection, but the downgrade is logged at a level that names the
    unhonored ``strict`` intent so it is not silent (issue #1241).
    """
    if response_format is None:
        return False
    rf = response_format
    rf_type = rf.get("type") if isinstance(rf, dict) else getattr(rf, "type", None)
    if rf_type != "json_schema":
        return False
    js = (
        rf.get("json_schema")
        if isinstance(rf, dict)
        else getattr(rf, "json_schema", None)
    )
    if js is None:
        return False
    strict = js.get("strict") if isinstance(js, dict) else getattr(js, "strict", None)
    return bool(strict)


def _response_format_requests_grammar(response_format) -> bool:
    """True when an OpenAI ``response_format`` maps to grammar-constrained JSON.

    Delegates to :func:`_build_format_element` so the unenforced-degrade signal
    stays in sync with what actually gets compiled: a format earns the
    Warning header / prompt-injection fallback only when a grammar element would
    have been built for it.  That is non-``None`` exactly for ``json_object``
    and a ``json_schema`` carrying a schema; a plain ``{"type": "text"}`` (or a
    json_schema with no schema) maps to nothing and must not warn.  Sharing the
    one source of truth keeps the header consistent with the server-side warn
    log and avoids claiming "grammar-constrained decoding unavailable" for a
    request that never described an enforceable grammar (#1241 review).
    """
    if response_format is None:
        return False
    return _build_format_element(response_format=response_format) is not None


def _compile_grammar_for_request(
    engine: BaseEngine,
    structured_outputs=None,
    response_format=None,
    chat_template_kwargs=None,
    reasoning_parser=None,
):
    """Compile a grammar from structured_outputs or response_format.

    When ``reasoning_parser`` is set (e.g. ``"qwen"``, ``"harmony"``),
    the user's grammar is wrapped in an xgrammar builtin structural tag
    so that protocol tokens (thinking tags, channel markers) are handled
    automatically.  When not set, the grammar is compiled bare.

    Returns a compiled grammar object or ``None``.  ``structured_outputs``
    raises :class:`HTTPException` when grammar is unavailable or fails to
    compile.  A ``response_format`` degrades to ``None`` so the caller can fall
    back to prompt injection; the downgrade is logged (and named as an
    unhonored strict request when ``strict: true`` was set) rather than being
    silent (#1241).
    """
    compiler = getattr(engine, "grammar_compiler", None)

    fmt = _build_format_element(structured_outputs, response_format)
    if fmt is None:
        return None

    if compiler is None:
        if structured_outputs is not None:
            from omlx_config.utils.install import get_install_method

            method = get_install_method()
            if method == "homebrew":
                detail = (
                    "Structured output requires xgrammar. "
                    "Reinstall with: brew reinstall omlx --with-grammar"
                )
            else:
                detail = (
                    "Structured output requires xgrammar. "
                    "Install with: pip install 'omlx[grammar]'"
                )
            raise HTTPException(status_code=400, detail=detail)
        if response_format is not None:
            _warn_response_format_not_enforced(response_format)
        return None

    try:
        if reasoning_parser:
            return _compile_with_structural_tag(
                compiler,
                fmt,
                reasoning_parser,
                chat_template_kwargs,
            )
        return _compile_bare_grammar(compiler, fmt)
    except Exception as e:
        if structured_outputs is not None:
            raise HTTPException(
                status_code=400,
                detail=f"Grammar compilation error: {e}",
            )
        _warn_response_format_not_enforced(response_format, error=e)
    return None


def _warn_response_format_not_enforced(response_format, error=None):
    """Log that a ``response_format`` request fell back to prompt injection.

    Previously a ``response_format`` that could not be grammar-constrained
    (no compiler available, or a compilation error) degraded to best-effort
    prompt injection silently, giving the client no signal that the schema was
    not enforced (#1241).  A ``strict: true`` request gets a message that names
    the unhonored strict intent.
    """
    reason = f" ({error})" if error is not None else ""
    if _response_format_requests_strict(response_format):
        logger.warning(
            "response_format requested strict json_schema output but "
            "grammar-constrained decoding is unavailable; strict enforcement "
            "cannot be honored, falling back to best-effort prompt injection "
            "(output is NOT schema-enforced)%s.",
            reason,
        )
    else:
        logger.warning(
            "response_format requested but grammar-constrained decoding is "
            "unavailable; output will not be schema-enforced (falling back to "
            "prompt injection)%s.",
            reason,
        )


def _response_format_warning_header(response_format) -> str:
    """Build an RFC 7234 ``Warning`` header for an unenforced response_format.

    The server already logs the downgrade (see
    :func:`_warn_response_format_not_enforced`), but that signal is only
    visible to the operator.  This header surfaces the same fact to the API
    caller so a client can tell that ``response_format`` fell back to
    best-effort prompt injection rather than schema-enforced output (#1241).
    Header values must be single-line ASCII, so the text is terse.
    """
    if _response_format_requests_strict(response_format):
        text = (
            "response_format strict json_schema not enforced; "
            "grammar-constrained decoding unavailable, output is "
            "best-effort and NOT schema-enforced"
        )
    else:
        text = (
            "response_format not enforced; grammar-constrained decoding "
            "unavailable, output is best-effort"
        )
    return f'199 omlx "{text}"'
