"""Model-agnostic native diffusion boundary; optional mflux imports are lazy."""

from molto_runtime.diffusion.backend import MFluxBackend, download_patterns
from molto_runtime.diffusion.checkpoint import (
    Checkpoint,
    detect_checkpoint,
    model_capabilities,
    read_quantization,
    validate_checkpoint,
)
from molto_runtime.diffusion.registry import (
    MODEL_IDENTITIES,
    PIPELINES,
    ImageTask,
    PipelineSpec,
    get_pipeline,
    pipelines_for_model,
    resolve_base_model,
    validate_task,
)


def list_pipelines():
    return tuple(PIPELINES.values())


__all__ = [
    "ImageTask",
    "PipelineSpec",
    "Checkpoint",
    "MFluxBackend",
    "PIPELINES",
    "MODEL_IDENTITIES",
    "get_pipeline",
    "pipelines_for_model",
    "resolve_base_model",
    "validate_task",
    "detect_checkpoint",
    "validate_checkpoint",
    "read_quantization",
    "model_capabilities",
    "download_patterns",
    "list_pipelines",
]
