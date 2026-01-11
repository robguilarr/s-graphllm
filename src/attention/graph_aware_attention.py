"""
Graph-Aware Attention mechanism for modulating LLM attention based on graph topology.

The attention weight between tokens i and j is modified as:
α'_ij = exp(q_i^T k_j / sqrt(d_k) + β * S(n_i, n_j)) / sum_l exp(q_i^T k_l / sqrt(d_k) + β * S(n_i, n_l))

where S(n_i, n_j) is a structural similarity score between graph nodes n_i and n_j,
and β is a learnable parameter.
"""

from typing import Optional, Dict, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class GraphAwareAttentionConfig:
    """Configuration for graph-aware attention."""
    hidden_dim: int = 768
    num_heads: int = 8
    beta_init: float = 0.5
    similarity_metric: str = "shortest_path"  # or "common_neighbors"
    dropout: float = 0.1
    max_distance: int = 5


class GraphAwareAttention(nn.Module):
    """
    Graph-Aware Attention module that modulates attention scores based on graph structure.
    """
    
    def __init__(self, config: GraphAwareAttentionConfig):
        """
        Initialize the graph-aware attention module.
        
        Args:
            config: Configuration for the attention mechanism
        """
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Learnable parameter β for graph structure modulation
        self.beta = nn.Parameter(torch.tensor(config.beta_init))
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Structural similarity scores (will be set dynamically)
        self.structural_similarity_matrix: Optional[torch.Tensor] = None
    
    def set_structural_similarity_matrix(
        self,
        similarity_matrix: np.ndarray
    ) -> None:
        """
        Set the structural similarity matrix for the current batch.
        
        Args:
            similarity_matrix: (seq_len, seq_len) matrix of structural similarities
        """
        self.structural_similarity_matrix = torch.from_numpy(
            similarity_matrix
        ).float().to(self.query.weight.device)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        structural_similarity: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of graph-aware attention.
        
        Args:
            query: (batch_size, seq_len, hidden_dim)
            key: (batch_size, seq_len, hidden_dim)
            value: (batch_size, seq_len, hidden_dim)
            attention_mask: (batch_size, seq_len) or (batch_size, seq_len, seq_len)
            structural_similarity: (batch_size, seq_len, seq_len) or None
        
        Returns:
            output: (batch_size, seq_len, hidden_dim)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = query.shape
        
        # Project Q, K, V
        Q = self.query(query)  # (batch_size, seq_len, hidden_dim)
        K = self.key(key)      # (batch_size, seq_len, hidden_dim)
        V = self.value(value)  # (batch_size, seq_len, hidden_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Add graph structure modulation if available
        if structural_similarity is not None:
            # Expand structural similarity for multi-head attention
            if len(structural_similarity.shape) == 2:
                # (seq_len, seq_len) -> (1, 1, seq_len, seq_len)
                structural_similarity = structural_similarity.unsqueeze(0).unsqueeze(0)
            elif len(structural_similarity.shape) == 3:
                # (batch_size, seq_len, seq_len) -> (batch_size, 1, seq_len, seq_len)
                structural_similarity = structural_similarity.unsqueeze(1)
            
            # Modulate scores with graph structure
            scores = scores + self.beta * structural_similarity
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                # (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            elif len(attention_mask.shape) == 3:
                # (batch_size, seq_len, seq_len) -> (batch_size, 1, seq_len, seq_len)
                attention_mask = attention_mask.unsqueeze(1)
            
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.hidden_dim)
        
        # Final output projection
        output = self.output_projection(output)
        
        return output, attention_weights
    
    def get_attention_focus(self) -> Dict[str, float]:
        """
        Get statistics about attention focus on graph-related tokens.
        
        Returns:
            Dictionary with attention focus statistics
        """
        if self.structural_similarity_matrix is None:
            return {"error": "No structural similarity matrix set"}
        
        # This would be computed from the last attention weights
        # For now, return placeholder
        return {
            "beta_value": self.beta.item(),
            "similarity_matrix_mean": self.structural_similarity_matrix.mean().item(),
            "similarity_matrix_std": self.structural_similarity_matrix.std().item(),
        }


class MultiHeadGraphAwareAttention(nn.Module):
    """
    Multi-head graph-aware attention that can be integrated into transformer layers.
    """
    
    def __init__(self, config: GraphAwareAttentionConfig):
        """Initialize multi-head graph-aware attention."""
        super().__init__()
        self.config = config
        self.attention_heads = nn.ModuleList([
            GraphAwareAttention(config) for _ in range(config.num_heads)
        ])
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        structural_similarity: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with multiple attention heads.
        
        Args:
            query: (batch_size, seq_len, hidden_dim)
            key: (batch_size, seq_len, hidden_dim)
            value: (batch_size, seq_len, hidden_dim)
            attention_mask: Optional attention mask
            structural_similarity: Optional structural similarity matrix
        
        Returns:
            output: Concatenated output from all heads
            attention_weights: List of attention weights from each head
        """
        outputs = []
        all_attention_weights = []
        
        for head in self.attention_heads:
            output, attention_weights = head(
                query, key, value,
                attention_mask=attention_mask,
                structural_similarity=structural_similarity
            )
            outputs.append(output)
            all_attention_weights.append(attention_weights)
        
        # Concatenate outputs from all heads
        final_output = torch.cat(outputs, dim=-1)
        
        return final_output, all_attention_weights


def compute_structural_similarity_matrix(
    adj_matrix: np.ndarray,
    seq_len: int,
    metric: str = "shortest_path",
    max_distance: int = 5
) -> np.ndarray:
    """
    Compute structural similarity matrix for a sequence of graph nodes.
    
    Args:
        adj_matrix: Adjacency matrix of the graph
        seq_len: Length of the sequence (number of nodes to consider)
        metric: Similarity metric ("shortest_path" or "common_neighbors")
        max_distance: Maximum distance for shortest path
    
    Returns:
        (seq_len, seq_len) similarity matrix
    """
    similarity_matrix = np.zeros((seq_len, seq_len))
    
    for i in range(seq_len):
        for j in range(seq_len):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                if metric == "shortest_path":
                    # Compute shortest path distance
                    distance = _compute_shortest_path(adj_matrix, i, j, max_distance)
                    similarity_matrix[i, j] = 1.0 / (1.0 + distance) if distance < float('inf') else 0.0
                elif metric == "common_neighbors":
                    # Compute Jaccard similarity of neighborhoods
                    neighbors_i = set(np.where(adj_matrix[i] > 0)[0])
                    neighbors_j = set(np.where(adj_matrix[j] > 0)[0])
                    common = len(neighbors_i & neighbors_j)
                    union = len(neighbors_i | neighbors_j)
                    similarity_matrix[i, j] = common / union if union > 0 else 0.0
    
    return similarity_matrix


def _compute_shortest_path(
    adj_matrix: np.ndarray,
    start: int,
    end: int,
    max_distance: int
) -> float:
    """
    Compute shortest path distance using BFS.
    
    Args:
        adj_matrix: Adjacency matrix
        start: Start node
        end: End node
        max_distance: Maximum distance to search
    
    Returns:
        Shortest path distance (or inf if not found)
    """
    from collections import deque
    
    n = adj_matrix.shape[0]
    visited = [False] * n
    queue = deque([(start, 0)])
    visited[start] = True
    
    while queue:
        node, dist = queue.popleft()
        
        if node == end:
            return float(dist)
        
        if dist < max_distance:
            neighbors = np.where(adj_matrix[node] > 0)[0]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append((neighbor, dist + 1))
    
    return float('inf')
