# SPDX-License-Identifier: Apache-2.0
"""
Engine abstraction for Molto inference.

Provides multiple engine implementations:
- BatchedEngine: Continuous batching for multiple concurrent users
- VLMBatchedEngine: Vision-language model engine with image support
- EmbeddingEngine: Batch embedding generation using mlx-embeddings
- RerankerEngine: Document reranking using SequenceClassification models

Also re-exports core engine components for backwards compatibility.
"""

# Re-export from parent engine.py for backwards compatibility
from molto_runtime.engine.base import (
    BaseEngine,
    BaseNonStreamingEngine,
    GenerationOutput,
)
from molto_runtime.engine.batched import BatchedEngine
from molto_runtime.engine.dflash import DFlashEngine
from molto_runtime.engine.embedding import EmbeddingEngine
from molto_runtime.engine.image_generation import MFluxImageEngine
from molto_runtime.engine.reranker import RerankerEngine
from molto_runtime.engine.sts import STSEngine
from molto_runtime.engine.stt import STTEngine
from molto_runtime.engine.tts import TTSEngine
from molto_runtime.engine.vlm import VLMBatchedEngine
from molto_runtime.engine_core import AsyncEngineCore, EngineConfig, EngineCore

__all__ = [
    "BaseEngine",
    "BaseNonStreamingEngine",
    "GenerationOutput",
    "BatchedEngine",
    "DFlashEngine",
    "VLMBatchedEngine",
    "EmbeddingEngine",
    "MFluxImageEngine",
    "RerankerEngine",
    "STTEngine",
    "STSEngine",
    "TTSEngine",
    # Core engine components
    "EngineCore",
    "AsyncEngineCore",
    "EngineConfig",
]
