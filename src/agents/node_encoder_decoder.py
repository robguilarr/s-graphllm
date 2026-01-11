"""
Node Encoder-Decoder for node understanding in GraphLLM.

This module implements the textual transformer encoder-decoder described in Section 3.2
of the GraphLLM paper, which extracts semantic information from node descriptions.

Reference: Chai, Z., et al. (2025). GraphLLM: Boosting Graph Reasoning Ability of Large Language Model.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class NodeEncoder(nn.Module):
    """
    Textual transformer encoder for extracting node semantic information.
    
    Formula from paper (Equation 5a):
    c_i = TransformerEncoder(d_i, W_D)
    
    where:
    - d_i is the textual description embedding of node i
    - W_D is a down-projection matrix
    - c_i is the context vector
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize node encoder.
        
        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden dimension for encoder
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Down-projection matrix W_D
        self.down_projection = nn.Linear(input_dim, hidden_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        logger.info(f"Node Encoder initialized: {input_dim} -> {hidden_dim}")
    
    def forward(
        self,
        node_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode node textual descriptions.
        
        Args:
            node_embeddings: Node description embeddings (batch_size x seq_len x input_dim)
            attention_mask: Optional attention mask
        
        Returns:
            Context vectors (batch_size x hidden_dim)
        """
        # Down-project
        x = self.down_projection(node_embeddings)
        
        # Apply transformer encoder
        # Convert attention mask to transformer format if provided
        if attention_mask is not None:
            # Transformer expects: (batch_size x seq_len) -> inverted boolean mask
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        encoded = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Mean pooling over sequence length to get context vector
        if attention_mask is not None:
            # Masked mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(encoded.size())
            sum_encoded = torch.sum(encoded * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            context = sum_encoded / sum_mask
        else:
            # Simple mean pooling
            context = encoded.mean(dim=1)
        
        # Layer norm
        context = self.layer_norm(context)
        
        return context


class NodeDecoder(nn.Module):
    """
    Textual transformer decoder for producing node representations.
    
    Formula from paper (Equation 5b):
    H_i = TransformerDecoder(Q, c_i)
    
    where:
    - Q is a learnable query embedding
    - c_i is the context vector from encoder
    - H_i is the final node representation
    """
    
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize node decoder.
        
        Args:
            hidden_dim: Hidden dimension
            output_dim: Output dimension for node representations
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Learnable query embedding Q
        self.query_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        nn.init.xavier_normal_(self.query_embedding)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(output_dim)
        
        logger.info(f"Node Decoder initialized: {hidden_dim} -> {output_dim}")
    
    def forward(
        self,
        context: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode node representations from context vectors.
        
        Args:
            context: Context vectors from encoder (batch_size x hidden_dim)
            memory_mask: Optional memory mask
        
        Returns:
            Node representations (batch_size x output_dim)
        """
        batch_size = context.shape[0]
        
        # Expand query embedding for batch
        query = self.query_embedding.expand(batch_size, -1, -1)
        
        # Expand context for cross-attention (add sequence dimension)
        memory = context.unsqueeze(1)  # (batch_size x 1 x hidden_dim)
        
        # Apply transformer decoder
        decoded = self.transformer_decoder(
            tgt=query,
            memory=memory
        )
        
        # Remove sequence dimension and project to output
        decoded = decoded.squeeze(1)  # (batch_size x hidden_dim)
        output = self.output_projection(decoded)
        
        # Layer norm
        output = self.layer_norm(output)
        
        return output


class NodeEncoderDecoder(nn.Module):
    """
    Complete encoder-decoder module for node understanding.
    
    This combines the encoder and decoder to implement the full node understanding
    component described in Section 3.2 of the GraphLLM paper.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 512,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize encoder-decoder.
        
        Args:
            input_dim: Input embedding dimension (from LLM tokenizer)
            hidden_dim: Hidden dimension for encoder/decoder
            output_dim: Output dimension for node representations
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.encoder = NodeEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.decoder = NodeDecoder(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        logger.info("Node Encoder-Decoder initialized")
    
    def forward(
        self,
        node_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process node descriptions to produce node representations.
        
        Args:
            node_embeddings: Node description embeddings (batch_size x seq_len x input_dim)
            attention_mask: Optional attention mask
        
        Returns:
            Node representations (batch_size x output_dim)
        """
        # Encode: extract semantic information
        context = self.encoder(node_embeddings, attention_mask)
        
        # Decode: produce node representations
        node_repr = self.decoder(context)
        
        return node_repr
    
    def encode_batch(
        self,
        node_texts: List[str],
        tokenizer,
        max_length: int = 128,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Encode a batch of node text descriptions.
        
        Args:
            node_texts: List of node text descriptions
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length
            device: Device for computation
        
        Returns:
            Node representations
        """
        # Tokenize texts
        encoded = tokenizer(
            node_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Get embeddings (assuming tokenizer has an embedding layer)
        # In practice, this would use the LLM's embedding layer
        embeddings = torch.randn(
            input_ids.shape[0],
            input_ids.shape[1],
            self.encoder.input_dim,
            device=device
        )  # Placeholder
        
        # Forward pass
        node_repr = self.forward(embeddings, attention_mask)
        
        return node_repr
