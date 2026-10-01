"""Operations offered by inference to management, without an engine dependency."""

from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


def engine_type_for(model_type: str | None, default: str = "batched") -> str:
    return {
        "llm": "batched",
        "vlm": "vlm",
        "embedding": "embedding",
        "reranker": "reranker",
        "audio_stt": "audio_stt",
        "audio_tts": "audio_tts",
        "audio_sts": "audio_sts",
        "image_generation": "image_generation",
    }.get(model_type, default)


@dataclass(frozen=True)
class ModelView:
    model_id: str
    model_path: str
    model_type: str
    engine_type: str
    loaded: bool
    config_model_type: str | None = None
    is_loading: bool = False
    in_use: int = 0
    pending_unload_reason: str | None = None
    is_pinned: bool = False
    preserve_thinking_default: bool | None = None
    source_type: str = "local"
    estimated_size: int = 0
    text_only_size: int = 0


class RuntimeOperationError(Exception):
    """Stable runtime capability error translated by management orchestration."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(detail)


class ManagementEngine(Protocol):
    """Explicit pool interface consumed by management services and diagnostics."""

    model_count: int
    model_directories: Sequence[Path]
    preparation_active: bool
    preparation_pending: bool
    ssd_cache_enabled: bool
    process_memory_enforcer: Any

    def cache_status(self, directory: Path | None = None) -> dict[str, Any]: ...
    async def clear_cache(
        self, kind: str, directory: Path | None = None
    ) -> dict[str, Any]: ...
    def probe_cache(self, request: Any, settings: Any) -> dict[str, Any]: ...
    def activity_snapshot(self, metrics: Any) -> dict[str, Any]: ...
    def get_model_view(self, model_id: str) -> ModelView | None: ...
    async def load_model(self, model_id: str) -> None: ...
    async def smoke_model(self, model_id: str, model_type: str) -> dict[str, Any]: ...
    def get_status(self) -> dict[str, Any]: ...
    def get_model_ids(self) -> list[str]: ...
    def management_operation_allowed(self) -> bool: ...
    def is_model_unloading(self, model_id: str) -> bool: ...
    def is_model_busy(self, model_id: str) -> bool: ...
    def has_active_requests(self) -> bool: ...
    def runtime_signature(self, model_id: str, settings: Any) -> Any: ...
    def set_model_pinned(self, model_id: str, pinned: bool) -> None: ...
    def set_model_type(self, model_id: str, model_type: str | None) -> None: ...
    def apply_settings_overrides(self, manager: Any) -> None: ...
    def discover_models(self, model_dirs: Any, pinned_ids: Any) -> None: ...
    def exclusive_management(
        self, check_cancel: Callable = ...
    ) -> AbstractAsyncContextManager: ...
    def exclusive_preparation(
        self, check_cancel: Callable
    ) -> AbstractAsyncContextManager: ...
    async def request_unload(self, model_id: str, **kwargs: Any) -> bool: ...
    async def apply_embedding_batch_size(self, batch_size: int) -> None: ...
