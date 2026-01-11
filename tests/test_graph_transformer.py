"""
Unit tests for Graph Transformer (GRIT) implementation.
"""

import pytest
import torch
import networkx as nx
from src.graph_processing.graph_transformer import (
    GraphTransformer,
    MultiHeadGraphAttention,
    compute_rrwp_encoding,
    pyg_softmax
)


class TestRRWPEncoding:
    """Test Relative Random Walk Positional Encoding."""
    
    def test_rrwp_shape(self):
        """Test RRWP output shape."""
        # Create simple adjacency matrix
        adj = torch.tensor([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0]
        ], dtype=torch.float32)
        
        max_steps = 8
        rrwp = compute_rrwp_encoding(adj, max_steps=max_steps)
        
        assert rrwp.shape == (4, 4, max_steps)
    
    def test_rrwp_identity_at_zero(self):
        """Test that RRWP starts with identity matrix."""
        adj = torch.tensor([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ], dtype=torch.float32)
        
        rrwp = compute_rrwp_encoding(adj, max_steps=3)
        
        # First step should be identity
        identity = torch.eye(3)
        assert torch.allclose(rrwp[:, :, 0], identity, atol=1e-6)
    
    def test_rrwp_isolated_node(self):
        """Test RRWP with isolated node."""
        adj = torch.tensor([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0]  # Isolated node
        ], dtype=torch.float32)
        
        rrwp = compute_rrwp_encoding(adj, max_steps=3)
        
        # Should not produce NaN
        assert not torch.isnan(rrwp).any()


class TestPygSoftmax:
    """Test sparse softmax function."""
    
    def test_pyg_softmax_basic(self):
        """Test basic softmax computation."""
        src = torch.tensor([1.0, 2.0, 3.0, 4.0])
        index = torch.tensor([0, 0, 1, 1])
        
        result = pyg_softmax(src, index, num_nodes=2)
        
        # Check shape
        assert result.shape == src.shape
        
        # Check sum to 1 for each group
        group0_sum = result[index == 0].sum()
        group1_sum = result[index == 1].sum()
        
        assert torch.allclose(group0_sum, torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(group1_sum, torch.tensor(1.0), atol=1e-6)
    
    def test_pyg_softmax_single_element(self):
        """Test softmax with single element per group."""
        src = torch.tensor([1.0, 2.0, 3.0])
        index = torch.tensor([0, 1, 2])
        
        result = pyg_softmax(src, index, num_nodes=3)
        
        # Each element should be 1.0
        assert torch.allclose(result, torch.ones_like(result), atol=1e-6)


class TestMultiHeadGraphAttention:
    """Test multi-head graph attention mechanism."""
    
    def test_attention_initialization(self):
        """Test attention layer initialization."""
        attention = MultiHeadGraphAttention(
            embed_dim=512,
            num_heads=8,
            rrwp_dim=8
        )
        
        assert attention.embed_dim == 512
        assert attention.num_heads == 8
        assert attention.head_dim == 64
    
    def test_attention_forward_shape(self):
        """Test attention forward pass output shape."""
        embed_dim = 64
        num_nodes = 10
        num_edges = 20
        rrwp_dim = 8
        
        attention = MultiHeadGraphAttention(
            embed_dim=embed_dim,
            num_heads=4,
            rrwp_dim=rrwp_dim
        )
        
        # Create dummy inputs
        x = torch.randn(num_nodes, embed_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, rrwp_dim)
        
        # Forward pass
        node_out, edge_out = attention(x, edge_index, edge_attr)
        
        assert node_out.shape == (num_nodes, embed_dim)
        assert edge_out.shape == (num_edges, rrwp_dim)
    
    def test_attention_residual_connection(self):
        """Test that residual connections are working."""
        embed_dim = 64
        num_nodes = 5
        num_edges = 8
        
        attention = MultiHeadGraphAttention(
            embed_dim=embed_dim,
            num_heads=4,
            rrwp_dim=8
        )
        
        x = torch.randn(num_nodes, embed_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, 8)
        
        node_out, edge_out = attention(x, edge_index, edge_attr)
        
        # Output should be different from input (not just identity)
        assert not torch.allclose(node_out, x)


class TestGraphTransformer:
    """Test complete graph transformer."""
    
    def test_transformer_initialization(self):
        """Test transformer initialization."""
        transformer = GraphTransformer(
            embed_dim=512,
            num_layers=2,
            num_heads=8,
            rrwp_dim=8
        )
        
        assert len(transformer.layers) == 2
        assert transformer.embed_dim == 512
    
    def test_transformer_forward_shape(self):
        """Test transformer forward pass output shape."""
        embed_dim = 128
        num_nodes = 15
        
        transformer = GraphTransformer(
            embed_dim=embed_dim,
            num_layers=2,
            num_heads=4,
            rrwp_dim=8
        )
        
        # Create graph
        graph = nx.karate_club_graph()
        adj_matrix = torch.tensor(nx.to_numpy_array(graph), dtype=torch.float32)
        edge_index = torch.tensor(list(graph.edges())).t()
        
        # Node features
        x = torch.randn(len(graph.nodes()), embed_dim)
        
        # Forward pass
        output = transformer(x, edge_index, adj_matrix)
        
        assert output.shape == x.shape
    
    def test_transformer_without_adj_matrix(self):
        """Test transformer without adjacency matrix."""
        embed_dim = 64
        num_nodes = 10
        num_edges = 15
        
        transformer = GraphTransformer(
            embed_dim=embed_dim,
            num_layers=1,
            num_heads=4,
            rrwp_dim=8
        )
        
        x = torch.randn(num_nodes, embed_dim)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        # Forward pass without adjacency matrix
        output = transformer(x, edge_index, adj_matrix=None)
        
        assert output.shape == x.shape
    
    def test_transformer_on_real_graph(self):
        """Test transformer on real graph (Karate Club)."""
        graph = nx.karate_club_graph()
        num_nodes = len(graph.nodes())
        embed_dim = 128
        
        transformer = GraphTransformer(
            embed_dim=embed_dim,
            num_layers=2,
            num_heads=8,
            rrwp_dim=8
        )
        
        # Prepare graph data
        adj_matrix = torch.tensor(nx.to_numpy_array(graph), dtype=torch.float32)
        edge_index = torch.tensor(list(graph.edges())).t()
        x = torch.randn(num_nodes, embed_dim)
        
        # Forward pass
        output = transformer(x, edge_index, adj_matrix)
        
        assert output.shape == (num_nodes, embed_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_transformer_gradient_flow(self):
        """Test that gradients flow through transformer."""
        embed_dim = 64
        num_nodes = 8
        num_edges = 12
        
        transformer = GraphTransformer(
            embed_dim=embed_dim,
            num_layers=1,
            num_heads=4,
            rrwp_dim=8
        )
        
        x = torch.randn(num_nodes, embed_dim, requires_grad=True)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        output = transformer(x, edge_index, adj_matrix=None)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
