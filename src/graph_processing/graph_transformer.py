"""
Graph Transformer (GRIT) implementation based on GraphLLM paper.

This module implements the graph transformer component that learns graph structure
representations using relative random walk positional encoding (RRWP) and sparse attention.

Reference: Chai, Z., et al. (2025). GraphLLM: Boosting Graph Reasoning Ability of Large Language Model.
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter, scatter_max, scatter_add
import logging

logger = logging.getLogger(__name__)


def pyg_softmax(src: torch.Tensor, index: torch.Tensor, num_nodes: Optional[int] = None) -> torch.Tensor:
    """
    Compute sparsely evaluated softmax for graph attention.
    
    Args:
        src: Source tensor with attention scores
        index: Indices for grouping
        num_nodes: Number of nodes
    
    Returns:
        Softmax-normalized tensor
    """
    if num_nodes is None:
        num_nodes = index.max().item() + 1
    
    # Subtract max for numerical stability
    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    
    # Normalize
    out = out / (scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)
    
    return out


def compute_rrwp_encoding(
    adj_matrix: torch.Tensor,
    max_steps: int = 8
) -> torch.Tensor:
    """
    Compute Relative Random Walk Positional Encoding (RRWP).
    
    Formula from paper (Equation 6):
    R_i,j = [I_i,j, M_i,j, M²_i,j, ..., M^(C-1)_i,j] ∈ ℝ^C
    
    where M = D^(-1)A is the random walk matrix.
    
    Args:
        adj_matrix: Adjacency matrix (N x N)
        max_steps: Maximum random walk steps (C in the paper)
    
    Returns:
        RRWP encoding tensor (N x N x max_steps)
    """
    device = adj_matrix.device
    n = adj_matrix.shape[0]
    
    # Compute degree matrix
    degree = adj_matrix.sum(dim=1)
    degree_inv = torch.where(degree > 0, 1.0 / degree, torch.zeros_like(degree))
    D_inv = torch.diag(degree_inv)
    
    # Random walk matrix M = D^(-1)A
    M = torch.matmul(D_inv, adj_matrix)
    
    # Compute powers of M
    rrwp = []
    M_power = torch.eye(n, device=device)  # M^0 = I
    
    for step in range(max_steps):
        rrwp.append(M_power.unsqueeze(-1))
        M_power = torch.matmul(M_power, M)
    
    # Stack along last dimension: (N, N, max_steps)
    rrwp_encoding = torch.cat(rrwp, dim=-1)
    
    return rrwp_encoding


class MultiHeadGraphAttention(nn.Module):
    """
    Multi-head sparse graph attention mechanism from GraphLLM paper.
    
    This implements the attention mechanism described in Section 3.3 of the paper,
    which uses edge features and positional encodings for structure understanding.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        rrwp_dim: int = 8,
        clamp: float = 5.0,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head graph attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            rrwp_dim: Dimension of RRWP encoding
            clamp: Clamping value for attention scores
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.rrwp_dim = rrwp_dim
        self.clamp = clamp
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.wq = nn.Linear(embed_dim, embed_dim, bias=True)
        self.wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.wv = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Edge feature projections (for positional encoding)
        self.w_eb = nn.Linear(rrwp_dim, embed_dim, bias=True)  # Edge bias
        self.w_ew = nn.Linear(rrwp_dim, embed_dim, bias=True)  # Edge weight
        
        # Output projections
        self.wo = nn.Linear(embed_dim, embed_dim, bias=False)
        self.weo = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Attention weight parameter
        self.Aw = nn.Parameter(torch.zeros(self.head_dim, num_heads, 1), requires_grad=True)
        nn.init.xavier_normal_(self.Aw)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer norms
        self.node_attn_norm = nn.LayerNorm(embed_dim)
        self.edge_attn_norm = nn.LayerNorm(rrwp_dim)
    
    def propagate_attention(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Propagate attention over graph edges.
        
        Args:
            xq: Query tensor (N x num_heads x head_dim)
            xk: Key tensor (N x num_heads x head_dim)
            xv: Value tensor (N x num_heads x head_dim)
            edge_index: Edge indices (2 x E)
            edge_attr: Edge attributes/RRWP encoding (E x rrwp_dim)
        
        Returns:
            Tuple of (node_output, edge_output)
        """
        # Get source and destination nodes
        src_idx = edge_index[0]  # Source nodes
        dst_idx = edge_index[1]  # Destination nodes
        
        # Compute attention scores
        src = xk[src_idx]  # E x num_heads x head_dim
        dst = xq[dst_idx]  # E x num_heads x head_dim
        score = src + dst  # Element-wise addition
        
        # Edge feature modulation
        eb = self.w_eb(edge_attr).view(-1, self.num_heads, self.head_dim)  # Edge bias
        ew = self.w_ew(edge_attr).view(-1, self.num_heads, self.head_dim)  # Edge weight
        
        # Apply edge weight modulation
        score = score * ew
        
        # Signed square root (from paper)
        score = torch.sqrt(torch.relu(score)) - torch.sqrt(torch.relu(-score))
        
        # Add edge bias
        score = score + eb
        score = F.relu(score)
        
        # Edge output
        e_out = score.flatten(1)
        
        # Final attention computation using Einstein summation
        # score: E x num_heads x head_dim
        # Aw: head_dim x num_heads x 1
        # result: E x num_heads x 1
        score = torch.einsum("ehd,dhc->ehc", score, self.Aw)
        
        # Clamp for stability
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)
        
        # Softmax over edges
        score = pyg_softmax(score, dst_idx)  # E x num_heads x 1
        
        # Aggregate messages
        msg = xv[src_idx] * score  # E x num_heads x head_dim
        x_out = torch.zeros_like(xv)  # N x num_heads x head_dim
        scatter(msg, dst_idx, dim=0, out=x_out, reduce='add')
        
        return x_out, e_out
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of graph attention.
        
        Args:
            x: Node features (N x embed_dim)
            edge_index: Edge indices (2 x E)
            edge_attr: Edge attributes (E x rrwp_dim)
        
        Returns:
            Tuple of (updated_nodes, updated_edges)
        """
        # Project to Q, K, V
        xq = self.wq(x).view(-1, self.num_heads, self.head_dim)
        xk = self.wk(x).view(-1, self.num_heads, self.head_dim)
        xv = self.wv(x).view(-1, self.num_heads, self.head_dim)
        
        # Propagate attention
        x_out, e_out = self.propagate_attention(xq, xk, xv, edge_index, edge_attr)
        
        # Reshape and project
        h = x_out.view(x_out.shape[0], -1)
        h = self.wo(h)
        e_out = self.weo(e_out)
        
        # Residual connections
        h = h + x
        e_out = e_out + edge_attr
        
        # Layer normalization
        h = self.node_attn_norm(h)
        e_out = self.edge_attn_norm(e_out)
        
        return h, e_out


class GraphTransformerLayer(nn.Module):
    """
    Single layer of the graph transformer with attention and feed-forward.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        rrwp_dim: int = 8,
        dropout: float = 0.1
    ):
        """Initialize graph transformer layer."""
        super().__init__()
        
        self.attention = MultiHeadGraphAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            rrwp_dim=rrwp_dim,
            dropout=dropout
        )
        
        # Feed-forward networks
        self.node_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        self.edge_ffn = nn.Sequential(
            nn.Linear(rrwp_dim, rrwp_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rrwp_dim * 2, rrwp_dim)
        )
        
        # Layer norms
        self.node_ffn_norm = nn.LayerNorm(embed_dim)
        self.edge_ffn_norm = nn.LayerNorm(rrwp_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through transformer layer."""
        # Attention
        x_attn, e_attn = self.attention(x, edge_index, edge_attr)
        
        # Feed-forward with residual
        x_out = x_attn + self.node_ffn(x_attn)
        e_out = e_attn + self.edge_ffn(e_attn)
        
        # Layer norm
        x_out = self.node_ffn_norm(x_out)
        e_out = self.edge_ffn_norm(e_out)
        
        return x_out, e_out


class GraphTransformer(nn.Module):
    """
    Graph Transformer (GRIT) for structure understanding in GraphLLM.
    
    This implements the graph transformer described in Section 3.3 of the GraphLLM paper,
    which learns graph structure representations using RRWP and sparse attention.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        rrwp_dim: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize graph transformer.
        
        Args:
            embed_dim: Embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            rrwp_dim: Dimension of RRWP encoding
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.rrwp_dim = rrwp_dim
        
        # Transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                rrwp_dim=rrwp_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # RRWP projection
        self.rrwp_projection = nn.Linear(rrwp_dim, rrwp_dim)
        
        logger.info(f"Graph Transformer initialized with {num_layers} layers")
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        adj_matrix: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through graph transformer.
        
        Args:
            x: Node features (N x embed_dim)
            edge_index: Edge indices (2 x E)
            adj_matrix: Optional adjacency matrix for RRWP computation
        
        Returns:
            Graph representation tensor
        """
        # Compute RRWP if adjacency matrix provided
        if adj_matrix is not None:
            rrwp = compute_rrwp_encoding(adj_matrix, max_steps=self.rrwp_dim)
            # Extract edge attributes from RRWP
            edge_attr = rrwp[edge_index[0], edge_index[1], :]
            edge_attr = self.rrwp_projection(edge_attr)
        else:
            # Use zero edge attributes if no adjacency matrix
            num_edges = edge_index.shape[1]
            edge_attr = torch.zeros(num_edges, self.rrwp_dim, device=x.device)
        
        # Apply transformer layers
        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr)
        
        return x
