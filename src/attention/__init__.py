"""Attention mechanisms for S-GraphLLM."""

from .graph_aware_attention import (
    GraphAwareAttention,
    GraphAwareAttentionConfig,
    MultiHeadGraphAwareAttention,
    compute_structural_similarity_matrix,
)

__all__ = [
    "GraphAwareAttention",
    "GraphAwareAttentionConfig",
    "MultiHeadGraphAwareAttention",
    "compute_structural_similarity_matrix",
]
