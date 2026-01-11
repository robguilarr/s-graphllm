"""
Unit tests for Node Encoder-Decoder implementation.
"""

import pytest
import torch
from src.agents.node_encoder_decoder import (
    NodeEncoder,
    NodeDecoder,
    NodeEncoderDecoder
)


class TestNodeEncoder:
    """Test node encoder component."""
    
    def test_encoder_initialization(self):
        """Test encoder initialization."""
        encoder = NodeEncoder(
            input_dim=768,
            hidden_dim=512,
            num_layers=2,
            num_heads=4
        )
        
        assert encoder.input_dim == 768
        assert encoder.hidden_dim == 512
    
    def test_encoder_forward_shape(self):
        """Test encoder forward pass output shape."""
        batch_size = 4
        seq_len = 32
        input_dim = 768
        hidden_dim = 512
        
        encoder = NodeEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4
        )
        
        # Create dummy input
        node_embeddings = torch.randn(batch_size, seq_len, input_dim)
        
        # Forward pass
        context = encoder(node_embeddings)
        
        assert context.shape == (batch_size, hidden_dim)
    
    def test_encoder_with_attention_mask(self):
        """Test encoder with attention mask."""
        batch_size = 4
        seq_len = 32
        input_dim = 768
        hidden_dim = 512
        
        encoder = NodeEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4
        )
        
        node_embeddings = torch.randn(batch_size, seq_len, input_dim)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, 20:] = 0  # Mask last 12 tokens
        
        # Forward pass
        context = encoder(node_embeddings, attention_mask)
        
        assert context.shape == (batch_size, hidden_dim)
        assert not torch.isnan(context).any()
    
    def test_encoder_gradient_flow(self):
        """Test gradient flow through encoder."""
        encoder = NodeEncoder(
            input_dim=128,
            hidden_dim=64,
            num_layers=1,
            num_heads=2
        )
        
        node_embeddings = torch.randn(2, 16, 128, requires_grad=True)
        context = encoder(node_embeddings)
        loss = context.sum()
        loss.backward()
        
        assert node_embeddings.grad is not None
        assert not torch.isnan(node_embeddings.grad).any()


class TestNodeDecoder:
    """Test node decoder component."""
    
    def test_decoder_initialization(self):
        """Test decoder initialization."""
        decoder = NodeDecoder(
            hidden_dim=512,
            output_dim=512,
            num_layers=2,
            num_heads=4
        )
        
        assert decoder.hidden_dim == 512
        assert decoder.output_dim == 512
        assert decoder.query_embedding.shape == (1, 1, 512)
    
    def test_decoder_forward_shape(self):
        """Test decoder forward pass output shape."""
        batch_size = 4
        hidden_dim = 512
        output_dim = 512
        
        decoder = NodeDecoder(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=2,
            num_heads=4
        )
        
        # Create dummy context
        context = torch.randn(batch_size, hidden_dim)
        
        # Forward pass
        node_repr = decoder(context)
        
        assert node_repr.shape == (batch_size, output_dim)
    
    def test_decoder_different_output_dim(self):
        """Test decoder with different output dimension."""
        batch_size = 4
        hidden_dim = 512
        output_dim = 256
        
        decoder = NodeDecoder(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=2,
            num_heads=4
        )
        
        context = torch.randn(batch_size, hidden_dim)
        node_repr = decoder(context)
        
        assert node_repr.shape == (batch_size, output_dim)
    
    def test_decoder_gradient_flow(self):
        """Test gradient flow through decoder."""
        decoder = NodeDecoder(
            hidden_dim=64,
            output_dim=64,
            num_layers=1,
            num_heads=2
        )
        
        context = torch.randn(2, 64, requires_grad=True)
        node_repr = decoder(context)
        loss = node_repr.sum()
        loss.backward()
        
        assert context.grad is not None
        assert not torch.isnan(context.grad).any()


class TestNodeEncoderDecoder:
    """Test complete encoder-decoder module."""
    
    def test_encoder_decoder_initialization(self):
        """Test encoder-decoder initialization."""
        enc_dec = NodeEncoderDecoder(
            input_dim=768,
            hidden_dim=512,
            output_dim=512,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        
        assert enc_dec.encoder.input_dim == 768
        assert enc_dec.encoder.hidden_dim == 512
        assert enc_dec.decoder.output_dim == 512
    
    def test_encoder_decoder_forward_shape(self):
        """Test encoder-decoder forward pass output shape."""
        batch_size = 4
        seq_len = 32
        input_dim = 768
        output_dim = 512
        
        enc_dec = NodeEncoderDecoder(
            input_dim=input_dim,
            hidden_dim=512,
            output_dim=output_dim,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        
        node_embeddings = torch.randn(batch_size, seq_len, input_dim)
        node_repr = enc_dec(node_embeddings)
        
        assert node_repr.shape == (batch_size, output_dim)
    
    def test_encoder_decoder_with_mask(self):
        """Test encoder-decoder with attention mask."""
        batch_size = 4
        seq_len = 32
        
        enc_dec = NodeEncoderDecoder(
            input_dim=768,
            hidden_dim=512,
            output_dim=512
        )
        
        node_embeddings = torch.randn(batch_size, seq_len, 768)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, 20:] = 0
        
        node_repr = enc_dec(node_embeddings, attention_mask)
        
        assert node_repr.shape == (batch_size, 512)
        assert not torch.isnan(node_repr).any()
    
    def test_encoder_decoder_end_to_end(self):
        """Test end-to-end encoder-decoder processing."""
        enc_dec = NodeEncoderDecoder(
            input_dim=128,
            hidden_dim=64,
            output_dim=64,
            num_encoder_layers=1,
            num_decoder_layers=1
        )
        
        # Simulate processing multiple nodes
        num_nodes = 10
        seq_len = 16
        
        node_embeddings = torch.randn(num_nodes, seq_len, 128)
        node_repr = enc_dec(node_embeddings)
        
        assert node_repr.shape == (num_nodes, 64)
        assert not torch.isnan(node_repr).any()
        assert not torch.isinf(node_repr).any()
    
    def test_encoder_decoder_gradient_flow(self):
        """Test gradient flow through entire encoder-decoder."""
        enc_dec = NodeEncoderDecoder(
            input_dim=128,
            hidden_dim=64,
            output_dim=64,
            num_encoder_layers=1,
            num_decoder_layers=1
        )
        
        node_embeddings = torch.randn(2, 16, 128, requires_grad=True)
        node_repr = enc_dec(node_embeddings)
        loss = node_repr.sum()
        loss.backward()
        
        assert node_embeddings.grad is not None
        assert not torch.isnan(node_embeddings.grad).any()
    
    def test_encoder_decoder_batch_consistency(self):
        """Test that batch processing is consistent."""
        enc_dec = NodeEncoderDecoder(
            input_dim=128,
            hidden_dim=64,
            output_dim=64
        )
        
        # Process single item
        single_embedding = torch.randn(1, 16, 128)
        single_repr = enc_dec(single_embedding)
        
        # Process batch with same item
        batch_embedding = single_embedding.repeat(3, 1, 1)
        batch_repr = enc_dec(batch_embedding)
        
        # All outputs should be similar (not exactly equal due to layer norm)
        assert batch_repr.shape == (3, 64)
        
        # Check that outputs are similar
        for i in range(3):
            similarity = F.cosine_similarity(
                single_repr[0:1],
                batch_repr[i:i+1],
                dim=1
            )
            assert similarity > 0.95  # High similarity expected


class TestIntegration:
    """Integration tests for encoder-decoder."""
    
    def test_variable_sequence_lengths(self):
        """Test handling of variable sequence lengths with masking."""
        enc_dec = NodeEncoderDecoder(
            input_dim=128,
            hidden_dim=64,
            output_dim=64
        )
        
        batch_size = 4
        max_seq_len = 32
        
        # Create variable length sequences
        node_embeddings = torch.randn(batch_size, max_seq_len, 128)
        attention_mask = torch.zeros(batch_size, max_seq_len)
        
        # Different lengths for each sequence
        lengths = [10, 15, 20, 25]
        for i, length in enumerate(lengths):
            attention_mask[i, :length] = 1
        
        node_repr = enc_dec(node_embeddings, attention_mask)
        
        assert node_repr.shape == (batch_size, 64)
        assert not torch.isnan(node_repr).any()
    
    def test_large_batch(self):
        """Test with large batch size."""
        enc_dec = NodeEncoderDecoder(
            input_dim=128,
            hidden_dim=64,
            output_dim=64,
            num_encoder_layers=1,
            num_decoder_layers=1
        )
        
        batch_size = 64
        seq_len = 16
        
        node_embeddings = torch.randn(batch_size, seq_len, 128)
        node_repr = enc_dec(node_embeddings)
        
        assert node_repr.shape == (batch_size, 64)
        assert not torch.isnan(node_repr).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
