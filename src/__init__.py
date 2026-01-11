"""S-GraphLLM: Scalable Graph-Augmented Language Model."""

from .agents import HierarchicalReasoningOrchestrator, LLMAgent, LLMConfig
from .graph_processing import GraphPartitioner, GraphCoarsener
from .attention import GraphAwareAttention, GraphAwareAttentionConfig
from .utils import Config, setup_logger

__version__ = "0.1.0"

__all__ = [
    "HierarchicalReasoningOrchestrator",
    "LLMAgent",
    "LLMConfig",
    "GraphPartitioner",
    "GraphCoarsener",
    "GraphAwareAttention",
    "GraphAwareAttentionConfig",
    "Config",
    "setup_logger",
]
