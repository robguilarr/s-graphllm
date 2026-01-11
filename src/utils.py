"""
Utility functions for S-GraphLLM system.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel


# Configure logging
def setup_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with the given name and level."""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


logger = setup_logger(__name__)


class Config(BaseModel):
    """Configuration model for S-GraphLLM."""
    
    # Model configuration
    model_name: str = "gpt-4.1-mini"
    embedding_dim: int = 768
    hidden_dim: int = 1024
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1
    
    # Graph configuration
    max_nodes_per_partition: int = 10000
    max_edges_per_partition: int = 50000
    num_partitions: Optional[int] = None
    
    # Reasoning configuration
    max_hops: int = 3
    context_window: int = 4096
    temperature: float = 0.7
    top_k: int = 50
    
    # Attention configuration
    graph_aware_attention_beta: float = 0.5
    similarity_metric: str = "shortest_path"  # or "common_neighbors"
    
    # Training configuration
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 1000
    
    # Data configuration
    dataset_name: str = "ogbn-mag"
    data_dir: str = "./data"
    cache_dir: str = "./cache"
    
    class Config:
        """Pydantic config."""
        extra = "allow"


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)


def save_config(config: Config, config_path: str) -> None:
    """Save configuration to YAML file."""
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config.dict(), f, default_flow_style=False)


def compute_structural_similarity(
    adj_matrix: np.ndarray,
    node_i: int,
    node_j: int,
    metric: str = "shortest_path",
    max_distance: int = 5
) -> float:
    """
    Compute structural similarity between two nodes.
    
    Args:
        adj_matrix: Adjacency matrix of the graph
        node_i: First node index
        node_j: Second node index
        metric: Similarity metric ("shortest_path" or "common_neighbors")
        max_distance: Maximum distance to consider for shortest path
    
    Returns:
        Structural similarity score in [0, 1]
    """
    if metric == "shortest_path":
        # Use BFS to compute shortest path
        from collections import deque
        
        n = adj_matrix.shape[0]
        visited = [False] * n
        queue = deque([(node_i, 0)])
        visited[node_i] = True
        
        while queue:
            node, dist = queue.popleft()
            if node == node_j:
                # Normalize distance to [0, 1]
                return 1.0 / (1.0 + dist)
            
            if dist < max_distance:
                neighbors = np.where(adj_matrix[node] > 0)[0]
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append((neighbor, dist + 1))
        
        # No path found
        return 0.0
    
    elif metric == "common_neighbors":
        # Count common neighbors
        neighbors_i = set(np.where(adj_matrix[node_i] > 0)[0])
        neighbors_j = set(np.where(adj_matrix[node_j] > 0)[0])
        common = len(neighbors_i & neighbors_j)
        union = len(neighbors_i | neighbors_j)
        
        if union == 0:
            return 0.0
        return common / union
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def batch_structural_similarity(
    adj_matrix: np.ndarray,
    node_pairs: List[Tuple[int, int]],
    metric: str = "shortest_path"
) -> np.ndarray:
    """
    Compute structural similarity for multiple node pairs.
    
    Args:
        adj_matrix: Adjacency matrix
        node_pairs: List of (node_i, node_j) tuples
        metric: Similarity metric
    
    Returns:
        Array of similarity scores
    """
    similarities = []
    for node_i, node_j in node_pairs:
        sim = compute_structural_similarity(adj_matrix, node_i, node_j, metric)
        similarities.append(sim)
    return np.array(similarities)


def normalize_text(text: str) -> str:
    """Normalize text for processing."""
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Convert to lowercase
    text = text.lower()
    return text


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def format_graph_context(
    nodes: List[Dict[str, Any]],
    edges: List[Tuple[int, int, str]],
    max_length: int = 2000
) -> str:
    """
    Format graph data as text context for LLM.
    
    Args:
        nodes: List of node dictionaries with 'id' and 'description'
        edges: List of (source, target, relation) tuples
        max_length: Maximum length of formatted context
    
    Returns:
        Formatted text context
    """
    context = "Graph Context:\n"
    context += "Nodes:\n"
    
    for node in nodes:
        node_text = f"- Node {node.get('id', '?')}: {node.get('description', 'N/A')}\n"
        if len(context) + len(node_text) > max_length:
            context += "...\n"
            break
        context += node_text
    
    context += "\nRelationships:\n"
    for source, target, relation in edges:
        edge_text = f"- {source} --[{relation}]--> {target}\n"
        if len(context) + len(edge_text) > max_length:
            context += "...\n"
            break
        context += edge_text
    
    return context


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save results to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


def load_results(output_path: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(output_path, 'r') as f:
        results = json.load(f)
    return results
