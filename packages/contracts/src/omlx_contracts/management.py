# SPDX-License-Identifier: Apache-2.0
"""Typed wire contracts for the small engine management API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelSettingsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

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
    qwen35_ane_prefill_enabled: bool | None = None
    qwen35_ane_prefill_fraction: float | None = Field(default=None, gt=0, le=1)
    qwen35_ane_prefill_shared_fraction: float | None = Field(default=None, ge=0, le=1)
    qwen35_ane_prefill_sequence_length: int | None = Field(default=None, ge=32)
    qwen35_ane_prefill_cpu_enabled: bool | None = None
    qwen35_ane_prefill_gdn: bool | None = None
    turboquant_kv_enabled: bool | None = None
    turboquant_kv_bits: float | None = None
    moe_expert_offload_enabled: bool | None = None
    moe_expert_offload_resident_fraction: float | None = Field(default=None, gt=0, le=1)
    specprefill_enabled: bool | None = None
    specprefill_draft_model: str | None = None
    dflash_enabled: bool | None = None
    dflash_draft_model: str | None = None
    mtp_enabled: bool | None = None
    mtp_adaptive_max_depth: int | None = Field(default=None, ge=1, le=8)
    mtp_fixed_depth: int | None = Field(default=None, ge=1, le=8)
    vlm_mtp_enabled: bool | None = None
    vlm_mtp_draft_model: str | None = None
    is_pinned: bool | None = None
    is_default: bool | None = None
    is_hidden: bool | None = None

    turboquant_skip_last: bool | None = None
    qwen35_ane_prefill_tail_padding_min_tokens: int | None = Field(default=None, ge=0)
    qwen35_ane_prefill_fused_down: bool | None = None
    qwen35_ane_prefill_max_layers: int | None = Field(default=None, ge=1)
    qwen35_ane_prefill_dual_ane: bool | None = None
    qwen35_ane_prefill_gdn_fraction: float | None = Field(default=None, ge=0, le=1)
    qwen35_ane_prefill_gdn_max_layers: int | None = Field(default=None, ge=1)
    qwen35_ane_prefill_cpu_fraction: float | None = Field(default=None, ge=0, le=1)
    qwen35_ane_prefill_cpu_down_fraction: float | None = Field(default=None, ge=0, le=1)
    qwen35_ane_prefill_cpu_gdn_fraction: float | None = Field(default=None, ge=0, le=1)
    qwen35_ane_prefill_cpu_threads: int | None = Field(default=None, ge=0)
    qwen35_ane_prefill_cpu_shared_resource: bool | None = None
    qwen35_oq_a8_enabled: bool | None = None
    qwen35_oq_a8_min_tokens: int | None = Field(default=None, ge=1)
    specprefill_keep_pct: float | None = Field(default=None, ge=0, le=1)
    specprefill_threshold: int | None = Field(default=None, ge=1)
    dflash_draft_quant_enabled: bool | None = None
    dflash_draft_quant_weight_bits: int | None = Field(default=None, ge=1)
    dflash_draft_quant_activation_bits: int | None = Field(default=None, ge=1)
    dflash_draft_quant_group_size: int | None = Field(default=None, ge=1)
    dflash_max_ctx: int | None = Field(default=None, ge=1)
    dflash_in_memory_cache: bool | None = None
    dflash_in_memory_cache_max_entries: int | None = Field(default=None, ge=0)
    dflash_in_memory_cache_max_bytes: int | None = Field(default=None, ge=0)
    dflash_ssd_cache: bool | None = None
    dflash_ssd_cache_max_bytes: int | None = Field(default=None, ge=0)
    dflash_draft_window_size: int | None = Field(default=None, ge=1)
    dflash_draft_sink_size: int | None = Field(default=None, ge=0)
    dflash_block_size: int | None = Field(default=None, ge=1)
    dflash_verify_mode: str | None = None
    vlm_mtp_draft_block_size: int | None = Field(default=None, ge=1)
    reasoning_parser: str | None = None
    is_favorite: bool | None = None
    trust_remote_code: bool | None = None
    display_name: str | None = None
    description: str | None = None

    @model_validator(mode="after")
    def validate_choices(self):
        choices = {
            "turboquant_kv_bits": (2, 2.5, 3, 3.5, 4, 6, 8),
            "dflash_verify_mode": ("dflash", "adaptive", "ddtree", "off"),
            "dflash_draft_quant_weight_bits": (2, 4, 8),
            "dflash_draft_quant_activation_bits": (16, 32),
            "dflash_draft_quant_group_size": (32, 64, 128),
        }
        for key, allowed in choices.items():
            value = getattr(self, key)
            if value is not None and value not in allowed:
                raise ValueError(f"{key} must be one of {allowed}")
        for key in ("mtp_fixed_depth", "mtp_adaptive_max_depth"):
            value = getattr(self, key)
            if value is not None and not 1 <= value <= 8:
                raise ValueError(f"{key} must be between 1 and 8")
        return self


class TemplateWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    display_name: str = ""
    description: str | None = None
    settings: ModelSettingsPatch


class TemplateUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    new_name: str | None = None
    display_name: str | None = None
    description: str | None = None
    settings: ModelSettingsPatch | None = None


class RecipeApply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recipe: str = Field(max_length=9000)


class OptimalApply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    benchmark_id: str = Field(pattern=r"^[a-z0-9]{1,16}$")


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
    preparation_active: bool = False
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


class DiffusionCalibrationTask(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    prompt: str = Field(min_length=1, max_length=32768)
    seed: int = Field(default=0, ge=0, le=4294967295)
    width: int = Field(default=256, ge=64, le=2048)
    height: int = Field(default=256, ge=64, le=2048)
    steps: int | None = Field(default=None, ge=1, le=1000)
    guidance: float | None = Field(default=None, ge=0)
    negative_prompt: str | None = Field(default=None, max_length=32768)


class DiffusionCalibrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1)
    tasks: list[DiffusionCalibrationTask] = Field(min_length=1, max_length=32)
    max_rows: int = Field(default=256, ge=1, le=4096)


class DiffusionQuantizationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    model_id: str = Field(min_length=1)
    calibration_job_id: str = Field(min_length=1)
    bits: Literal[3, 4, 5, 6, 8] = 4
    group_size: Literal[32, 64, 128] = 64
    budget_bytes: int | None = Field(default=None, gt=0)
    budget_ratio: float = Field(default=1.10, ge=1)
    protected: list[str] = Field(default_factory=list, max_length=128)

    @model_validator(mode="after")
    def _exclusive_budget(self):
        if self.budget_bytes is not None and "budget_ratio" in self.model_fields_set:
            raise ValueError("Specify budget_bytes or budget_ratio, not both")
        if any(not pattern or len(pattern) > 512 for pattern in self.protected):
            raise ValueError(
                "Protected layer patterns must contain 1 to 512 characters"
            )
        return self


class DiffusionJobView(BaseModel):
    id: str
    kind: Literal["calibration", "quantization"]
    model_id: str
    base_model: str
    status: Literal[
        "queued", "waiting", "running", "cancelling", "completed", "failed", "cancelled"
    ]
    phase: str
    progress: float = Field(ge=0, le=1)
    detail: str
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class DiffusionJobsResponse(BaseModel):
    jobs: list[DiffusionJobView]
