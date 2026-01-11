"""
Graph coarsening module for creating summarized graphs for hierarchical reasoning.
"""

from typing import Dict, Set, List, Tuple, Optional
import numpy as np
import networkx as nx
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class GraphCoarsener:
    """
    Creates coarsened (summarized) graphs where each node represents a subgraph.
    Used for coarse-grained reasoning in the hierarchical reasoning orchestrator.
    """
    
    def __init__(self):
        """Initialize the graph coarsener."""
        self.coarse_graph: Optional[nx.Graph] = None
        self.partition_mapping: Dict[int, Set[int]] = {}  # coarse_node -> fine_nodes
        self.node_summaries: Dict[int, Dict] = {}  # coarse_node -> summary info
    
    def coarsen_graph(
        self,
        graph: nx.Graph,
        partitions: Dict[int, Set[int]],
        node_features: Optional[Dict[int, Dict]] = None
    ) -> nx.Graph:
        """
        Create a coarsened graph from partitions.
        
        Args:
            graph: Original fine-grained graph
            partitions: Dictionary mapping partition ID to set of nodes
            node_features: Optional node feature dictionary
        
        Returns:
            Coarsened graph where nodes represent partitions
        """
        self.partition_mapping = partitions
        self.coarse_graph = nx.Graph()
        
        # Add coarse nodes (one for each partition)
        for partition_id in partitions.keys():
            self.coarse_graph.add_node(partition_id)
            self._create_partition_summary(
                partition_id,
                partitions[partition_id],
                graph,
                node_features
            )
        
        # Add coarse edges (between partitions with cut edges)
        cut_edges_set = set()
        for src, dst in graph.edges():
            src_partition = self._find_node_partition(src, partitions)
            dst_partition = self._find_node_partition(dst, partitions)
            
            if src_partition != dst_partition:
                edge_key = tuple(sorted([src_partition, dst_partition]))
                cut_edges_set.add(edge_key)
        
        for src_partition, dst_partition in cut_edges_set:
            self.coarse_graph.add_edge(src_partition, dst_partition)
        
        logger.info(f"Coarsened graph created: {len(self.coarse_graph.nodes())} nodes, "
                   f"{len(self.coarse_graph.edges())} edges")
        
        return self.coarse_graph
    
    def _find_node_partition(self, node: int, partitions: Dict[int, Set[int]]) -> Optional[int]:
        """Find which partition a node belongs to."""
        for partition_id, nodes in partitions.items():
            if node in nodes:
                return partition_id
        return None
    
    def _create_partition_summary(
        self,
        partition_id: int,
        nodes: Set[int],
        graph: nx.Graph,
        node_features: Optional[Dict[int, Dict]] = None
    ) -> None:
        """
        Create a summary for a partition.
        
        Args:
            partition_id: ID of the partition
            nodes: Set of nodes in the partition
            graph: Original graph
            node_features: Optional node features
        """
        summary = {
            "partition_id": partition_id,
            "num_nodes": len(nodes),
            "node_ids": list(nodes),
        }
        
        # Compute partition statistics
        subgraph = graph.subgraph(nodes)
        summary["num_edges"] = len(subgraph.edges())
        
        # Compute average degree
        if len(nodes) > 0:
            degrees = [graph.degree(node) for node in nodes]
            summary["avg_degree"] = np.mean(degrees)
            summary["max_degree"] = max(degrees)
        else:
            summary["avg_degree"] = 0
            summary["max_degree"] = 0
        
        # Get node descriptions if available
        if node_features:
            descriptions = []
            for node in list(nodes)[:5]:  # Get first 5 node descriptions
                if node in node_features:
                    descriptions.append(node_features[node].get("description", f"Node {node}"))
            summary["sample_descriptions"] = descriptions
        
        self.node_summaries[partition_id] = summary
    
    def get_coarse_node_summary(self, partition_id: int) -> Dict:
        """
        Get summary information for a coarse node (partition).
        
        Args:
            partition_id: ID of the partition
        
        Returns:
            Dictionary with partition summary
        """
        if partition_id not in self.node_summaries:
            raise ValueError(f"Partition {partition_id} not found")
        return self.node_summaries[partition_id]
    
    def get_fine_nodes_for_coarse_node(self, partition_id: int) -> Set[int]:
        """
        Get all fine-grained nodes represented by a coarse node.
        
        Args:
            partition_id: ID of the coarse node (partition)
        
        Returns:
            Set of fine-grained node IDs
        """
        if partition_id not in self.partition_mapping:
            raise ValueError(f"Partition {partition_id} not found")
        return self.partition_mapping[partition_id]
    
    def get_neighboring_partitions(self, partition_id: int) -> List[int]:
        """
        Get neighboring partitions in the coarse graph.
        
        Args:
            partition_id: ID of the partition
        
        Returns:
            List of neighboring partition IDs
        """
        if self.coarse_graph is None:
            raise ValueError("Coarse graph not created yet")
        
        if partition_id not in self.coarse_graph.nodes():
            raise ValueError(f"Partition {partition_id} not found in coarse graph")
        
        return list(self.coarse_graph.neighbors(partition_id))
    
    def get_coarse_graph_description(self) -> str:
        """
        Get a text description of the coarse graph for LLM reasoning.
        
        Returns:
            Text description of the coarse graph
        """
        if self.coarse_graph is None:
            raise ValueError("Coarse graph not created yet")
        
        description = "Coarse-Grained Graph Structure:\n"
        description += f"Number of partitions: {len(self.coarse_graph.nodes())}\n"
        description += f"Number of inter-partition connections: {len(self.coarse_graph.edges())}\n\n"
        
        description += "Partition Details:\n"
        for partition_id in sorted(self.coarse_graph.nodes()):
            summary = self.node_summaries[partition_id]
            description += f"\nPartition {partition_id}:\n"
            description += f"  - Number of nodes: {summary['num_nodes']}\n"
            description += f"  - Number of edges: {summary['num_edges']}\n"
            description += f"  - Average degree: {summary['avg_degree']:.2f}\n"
            
            neighbors = self.get_neighboring_partitions(partition_id)
            if neighbors:
                description += f"  - Connected to partitions: {neighbors}\n"
        
        return description
    
    def find_relevant_partitions(
        self,
        query_keywords: List[str],
        node_features: Dict[int, Dict],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Find the most relevant partitions for a query.
        
        Args:
            query_keywords: List of keywords from the query
            node_features: Dictionary of node features
            top_k: Number of top partitions to return
        
        Returns:
            List of (partition_id, relevance_score) tuples
        """
        partition_scores = {}
        
        for partition_id, nodes in self.partition_mapping.items():
            score = 0.0
            
            # Count keyword matches in partition nodes
            for node in nodes:
                if node in node_features:
                    node_desc = node_features[node].get("description", "").lower()
                    for keyword in query_keywords:
                        if keyword.lower() in node_desc:
                            score += 1.0
            
            # Normalize by partition size
            if len(nodes) > 0:
                score /= len(nodes)
            
            partition_scores[partition_id] = score
        
        # Sort by score and return top-k
        sorted_partitions = sorted(
            partition_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_partitions[:top_k]
    
    def get_subgraph_for_partitions(
        self,
        graph: nx.Graph,
        partition_ids: List[int],
        include_neighbors: bool = True
    ) -> nx.Graph:
        """
        Get a subgraph containing specified partitions and optionally their neighbors.
        
        Args:
            graph: Original fine-grained graph
            partition_ids: List of partition IDs to include
            include_neighbors: Whether to include neighboring partitions
        
        Returns:
            Subgraph containing nodes from specified partitions
        """
        nodes_to_include = set()
        
        # Add nodes from specified partitions
        for partition_id in partition_ids:
            if partition_id in self.partition_mapping:
                nodes_to_include.update(self.partition_mapping[partition_id])
        
        # Optionally add nodes from neighboring partitions
        if include_neighbors:
            neighbor_partitions = set()
            for partition_id in partition_ids:
                neighbors = self.get_neighboring_partitions(partition_id)
                neighbor_partitions.update(neighbors)
            
            for partition_id in neighbor_partitions:
                if partition_id in self.partition_mapping:
                    nodes_to_include.update(self.partition_mapping[partition_id])
        
        # Extract subgraph
        subgraph = graph.subgraph(nodes_to_include).copy()
        return subgraph
