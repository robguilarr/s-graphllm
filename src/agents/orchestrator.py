"""
Hierarchical Reasoning Orchestrator for S-GraphLLM.

Operates in two stages:
1. Coarse-Grained Reasoning: Identify relevant subgraphs
2. Fine-Grained Reasoning: Perform detailed reasoning within selected subgraphs
"""

from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from dataclasses import dataclass, asdict
import networkx as nx

from ..graph_processing import GraphPartitioner, GraphCoarsener
from ..utils import Config, format_graph_context

logger = logging.getLogger(__name__)


@dataclass
class ReasoningResult:
    """Result of hierarchical reasoning."""
    query: str
    coarse_reasoning: str
    selected_partitions: List[int]
    fine_grained_reasoning: str
    final_answer: str
    confidence: float
    reasoning_steps: List[str]


class HierarchicalReasoningOrchestrator:
    """
    Orchestrates the hierarchical reasoning process for S-GraphLLM.
    """
    
    def __init__(
        self,
        graph: nx.Graph,
        config: Config,
        node_features: Optional[Dict[int, Dict]] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            graph: The knowledge graph
            config: Configuration object
            node_features: Optional node feature dictionary
        """
        self.graph = graph
        self.config = config
        self.node_features = node_features or {}
        
        # Initialize partitioner and coarsener
        self.partitioner = GraphPartitioner(
            max_nodes_per_partition=config.max_nodes_per_partition
        )
        self.coarsener = GraphCoarsener()
        
        # State
        self.partitions: Optional[Dict[int, set]] = None
        self.coarse_graph: Optional[nx.Graph] = None
        self.reasoning_history: List[ReasoningResult] = []
    
    def setup(self) -> None:
        """Set up the orchestrator by partitioning and coarsening the graph."""
        logger.info("Setting up hierarchical reasoning orchestrator...")
        
        # Partition the graph
        num_partitions = self.config.num_partitions
        if num_partitions is None:
            num_partitions = max(
                1,
                len(self.graph.nodes()) // self.config.max_nodes_per_partition
            )
        
        self.partitions = self.partitioner.partition_graph(
            self.graph,
            num_partitions=num_partitions,
            method="metis_like"
        )
        
        logger.info(f"Graph partitioned into {len(self.partitions)} partitions")
        
        # Create coarse graph
        self.coarse_graph = self.coarsener.coarsen_graph(
            self.graph,
            self.partitions,
            self.node_features
        )
        
        logger.info("Coarse graph created for hierarchical reasoning")
        
        # Log partition statistics
        stats = self.partitioner.get_partition_stats()
        logger.info(f"Partition statistics: {stats}")
    
    def reason(
        self,
        query: str,
        llm_agent: 'LLMAgent'
    ) -> ReasoningResult:
        """
        Perform hierarchical reasoning on the query.
        
        Args:
            query: User query
            llm_agent: LLM agent for reasoning
        
        Returns:
            ReasoningResult with the final answer
        """
        logger.info(f"Starting hierarchical reasoning for query: {query}")
        
        if self.coarse_graph is None:
            raise ValueError("Orchestrator not set up. Call setup() first.")
        
        reasoning_steps = []
        
        # Stage 1: Coarse-Grained Reasoning
        logger.info("Stage 1: Coarse-grained reasoning")
        coarse_context = self.coarsener.get_coarse_graph_description()
        reasoning_steps.append(f"Generated coarse graph context with {len(self.coarse_graph.nodes())} partitions")
        
        # Extract keywords from query
        query_keywords = query.lower().split()
        
        # Find relevant partitions
        relevant_partitions = self.coarsener.find_relevant_partitions(
            query_keywords,
            self.node_features,
            top_k=min(5, len(self.partitions))
        )
        selected_partition_ids = [pid for pid, _ in relevant_partitions]
        
        reasoning_steps.append(f"Selected {len(selected_partition_ids)} relevant partitions")
        
        # Perform coarse-grained reasoning with LLM
        coarse_prompt = self._create_coarse_reasoning_prompt(
            query,
            coarse_context,
            selected_partition_ids
        )
        
        coarse_reasoning = llm_agent.reason(coarse_prompt)
        reasoning_steps.append("Completed coarse-grained reasoning")
        
        # Stage 2: Fine-Grained Reasoning
        logger.info("Stage 2: Fine-grained reasoning")
        
        # Get subgraph for selected partitions
        fine_graph = self.coarsener.get_subgraph_for_partitions(
            self.graph,
            selected_partition_ids,
            include_neighbors=True
        )
        
        reasoning_steps.append(f"Extracted fine-grained subgraph with {len(fine_graph.nodes())} nodes")
        
        # Format fine-grained context
        nodes_list = [
            {
                "id": node,
                "description": self.node_features.get(node, {}).get("description", f"Node {node}")
            }
            for node in list(fine_graph.nodes())[:50]  # Limit to 50 nodes for context
        ]
        edges_list = [
            (src, dst, "related")
            for src, dst in list(fine_graph.edges())[:100]  # Limit to 100 edges
        ]
        
        fine_context = format_graph_context(nodes_list, edges_list)
        reasoning_steps.append("Formatted fine-grained graph context")
        
        # Perform fine-grained reasoning with LLM
        fine_prompt = self._create_fine_reasoning_prompt(
            query,
            coarse_reasoning,
            fine_context
        )
        
        fine_reasoning = llm_agent.reason(fine_prompt)
        reasoning_steps.append("Completed fine-grained reasoning")
        
        # Stage 3: Answer Synthesis
        logger.info("Stage 3: Answer synthesis")
        
        synthesis_prompt = self._create_synthesis_prompt(
            query,
            coarse_reasoning,
            fine_reasoning
        )
        
        final_answer = llm_agent.reason(synthesis_prompt)
        reasoning_steps.append("Synthesized final answer")
        
        # Create result
        result = ReasoningResult(
            query=query,
            coarse_reasoning=coarse_reasoning,
            selected_partitions=selected_partition_ids,
            fine_grained_reasoning=fine_reasoning,
            final_answer=final_answer,
            confidence=0.8,  # Placeholder
            reasoning_steps=reasoning_steps
        )
        
        self.reasoning_history.append(result)
        logger.info("Hierarchical reasoning completed")
        
        return result
    
    def _create_coarse_reasoning_prompt(
        self,
        query: str,
        coarse_context: str,
        selected_partitions: List[int]
    ) -> str:
        """Create prompt for coarse-grained reasoning."""
        prompt = f"""Given the following query and coarse-grained graph structure, identify which partitions are most relevant for answering the query.

Query: {query}

{coarse_context}

Selected partitions for detailed analysis: {selected_partitions}

Please analyze the query and explain which partitions contain the most relevant information for answering it. Provide your reasoning step by step."""
        
        return prompt
    
    def _create_fine_reasoning_prompt(
        self,
        query: str,
        coarse_reasoning: str,
        fine_context: str
    ) -> str:
        """Create prompt for fine-grained reasoning."""
        prompt = f"""Based on the coarse-grained analysis and the detailed graph structure below, perform multi-hop reasoning to answer the query.

Query: {query}

Coarse-grained Analysis:
{coarse_reasoning}

Detailed Graph Structure:
{fine_context}

Please perform detailed reasoning over the graph to answer the query. Consider all relevant paths and relationships. Provide step-by-step reasoning."""
        
        return prompt
    
    def _create_synthesis_prompt(
        self,
        query: str,
        coarse_reasoning: str,
        fine_reasoning: str
    ) -> str:
        """Create prompt for answer synthesis."""
        prompt = f"""Based on the hierarchical reasoning process, synthesize a final answer to the query.

Query: {query}

Coarse-grained Reasoning:
{coarse_reasoning}

Fine-grained Reasoning:
{fine_reasoning}

Please provide a clear, concise final answer to the query, synthesizing the insights from both the coarse and fine-grained reasoning stages. Also provide a confidence score (0-1) for your answer."""
        
        return prompt
    
    def get_reasoning_trace(self, index: int = -1) -> Dict[str, Any]:
        """
        Get the reasoning trace for a specific query.
        
        Args:
            index: Index of the reasoning result (-1 for most recent)
        
        Returns:
            Dictionary with reasoning trace
        """
        if not self.reasoning_history:
            raise ValueError("No reasoning history available")
        
        result = self.reasoning_history[index]
        return asdict(result)
    
    def save_reasoning_history(self, output_path: str) -> None:
        """Save reasoning history to file."""
        history = [asdict(result) for result in self.reasoning_history]
        with open(output_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Reasoning history saved to {output_path}")
    
    def get_partition_info(self, partition_id: int) -> Dict[str, Any]:
        """Get information about a specific partition."""
        if self.partitions is None:
            raise ValueError("Orchestrator not set up")
        
        if partition_id not in self.partitions:
            raise ValueError(f"Partition {partition_id} not found")
        
        nodes = self.partitions[partition_id]
        subgraph = self.graph.subgraph(nodes)
        
        return {
            "partition_id": partition_id,
            "num_nodes": len(nodes),
            "num_edges": len(subgraph.edges()),
            "avg_degree": sum(dict(subgraph.degree()).values()) / len(nodes) if nodes else 0,
            "sample_nodes": list(nodes)[:10],
        }
