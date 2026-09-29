# SPDX-License-Identifier: Apache-2.0
"""Typed wire contracts for the small engine management API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelSettingsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_alias: str | None = None
    model_type_override: (
        Literal[
            "llm",
            "vlm",
            "embedding",
            "reranker",
            "audio_stt",
            "audio_tts",
            "audio_sts",
            "image_generation",
        ]
        | None
    ) = None
    max_context_window: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    repetition_penalty: float | None = Field(default=None, gt=0)
    min_p: float | None = Field(default=None, ge=0, le=1)
    presence_penalty: float | None = None
    force_sampling: bool | None = None
    max_tool_result_tokens: int | None = Field(default=None, ge=1)
    chat_template_kwargs: dict[str, Any] | None = None
    forced_ct_kwargs: list[str] | None = None
    ttl_seconds: int | None = Field(default=None, ge=0)
    index_cache_freq: int | None = Field(default=None, ge=1)
    enable_thinking: bool | None = None
    preserve_thinking: bool | None = None
    thinking_budget_enabled: bool | None = None
    thinking_budget_tokens: int | None = Field(default=None, ge=0)
    cache_reasoning_output: bool | None = None
    guided_grammar_enabled: bool | None = None
    guided_grammar: str | None = None
    qwen4_ple_ssd_offload: bool | None = None
    deepseek_v41_engram_ssd_offload: bool | None = None
    deepseek_v41_ced_prefill_enabled: bool | None = None
    turboquant_kv_enabled: bool | None = None
    turboquant_kv_bits: float | None = None
    moe_expert_offload_enabled: bool | None = None
    moe_expert_offload_resident_fraction: float | None = Field(default=None, gt=0, le=1)
    specprefill_enabled: bool | None = None
    specprefill_draft_model: str | None = None
    dflash_enabled: bool | None = None
    dflash_draft_model: str | None = None
    mtp_enabled: bool | None = None
    mtp_adaptive_max_depth: int | None = None
    mtp_fixed_depth: int | None = None
    vlm_mtp_enabled: bool | None = None
    vlm_mtp_draft_model: str | None = None
    is_pinned: bool | None = None
    is_default: bool | None = None
    is_hidden: bool | None = None


class GlobalSettingsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_context_window: int | None = Field(default=None, ge=1)
    max_context_window_policy: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    repetition_penalty: float | None = Field(default=None, gt=0)
    max_concurrent_requests: int | None = Field(default=None, ge=1)
    embedding_batch_size: int | None = Field(default=None, ge=1)
    chunked_prefill: bool | None = None
    prefill_priority: Literal["context", "speed"] | None = None
    decode_fairness: bool | None = None


class ProfileWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    display_name: str = ""
    description: str | None = None
    settings: ModelSettingsPatch
    expose_as_model: bool = False
    api_name: str | None = None


class ProfileUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    new_name: str | None = None
    display_name: str | None = None
    description: str | None = None
    settings: ModelSettingsPatch | None = None
    expose_as_model: bool | None = None
    api_name: str | None = None


class ModelSettingsView(BaseModel):
    """Describe persisted settings without revalidating legacy saved values."""

    model_config = ConfigDict(extra="allow")

    model_alias: str | None = None
    model_type_override: str | None = None
    max_context_window: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    enable_thinking: bool | None = None
    is_pinned: bool = False
    is_default: bool = False


class ModelInventoryItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    loaded: bool
    is_loading: bool
    is_unloading: bool
    engine_type: str
    model_type: str
    settings: ModelSettingsView


class InventoryResponse(BaseModel):
    models: list[ModelInventoryItem]
    model_count: int


class ModelStateItem(BaseModel):
    id: str
    loaded: bool
    is_loading: bool
    is_unloading: bool
    load_failed: bool


class StateResponse(BaseModel):
    default_model: str | None
    model_count: int
    loaded_count: int
    current_model_memory: int
    final_ceiling: int
    models: list[ModelStateItem]


class ModelOperationResponse(BaseModel):
    status: Literal["ok", "unloading"]
    model_id: str
    message: str | None = None


class RefreshResponse(BaseModel):
    status: Literal["ok"]
    model_count: int


class SamplingSettingsView(BaseModel):
    max_context_window: int
    max_context_window_policy: int | None
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float


class SchedulerSettingsView(BaseModel):
    max_concurrent_requests: int
    embedding_batch_size: int
    chunked_prefill: bool
    prefill_priority: Literal["context", "speed"]
    decode_fairness: bool


class GlobalSettingsView(BaseModel):
    sampling: SamplingSettingsView
    scheduler: SchedulerSettingsView


class GlobalSettingsUpdateResponse(GlobalSettingsView):
    requires_restart: bool


class ModelSettingsResponse(BaseModel):
    model_id: str
    settings: ModelSettingsView


class ModelSettingsUpdateResponse(ModelSettingsResponse):
    requires_reload: bool
    auto_unloaded: bool
    auto_reloaded: bool
    reload_deferred: bool
    reload_error: str | None


class ProfileView(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    display_name: str
    settings: dict[str, Any]
    expose_as_model: bool = False


class ProfilesResponse(BaseModel):
    profiles: list[ProfileView]


class ProfileResponse(BaseModel):
    profile: ProfileView


class ProfileDeleteResponse(BaseModel):
    deleted: bool
    name: str


class StatsResponse(BaseModel):
    total_tokens_served: int
    total_cached_tokens: int
    cache_efficiency: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_requests: int
    avg_prefill_tps: float
    avg_generation_tps: float
    uptime_seconds: float


class CacheModelStatus(BaseModel):
    model_id: str
    stats: dict[str, Any] | None


class CacheResponse(BaseModel):
    models: list[CacheModelStatus]
    ssd_cache_dir: str | None


class CacheClearResponse(BaseModel):
    status: Literal["ok"]
    kind: Literal["hot", "ssd"]
    total_cleared: int
