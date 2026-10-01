# SPDX-License-Identifier: Apache-2.0
"""
MLX Model wrappers for oMLX.

This module provides wrappers around mlx-lm and mlx-embeddings
for integration with oMLX's model execution system.
"""

from omlx_runtime.models.embedding import EmbeddingOutput, MLXEmbeddingModel
from omlx_runtime.models.llm import MLXLanguageModel

__all__ = ["MLXLanguageModel", "MLXEmbeddingModel", "EmbeddingOutput"]
