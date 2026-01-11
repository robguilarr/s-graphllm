"""
Graph partitioning module for dividing large graphs into manageable subgraphs.
Uses METIS algorithm to minimize cut edges.
"""

from typing import List, Tuple, Dict, Optional, Set
import numpy as np
import networkx as nx
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class GraphPartitioner:
    """
    Partitions a large graph into smaller subgraphs using various strategies.
    
    The objective is to minimize the number of cut edges:
    minimize cut(V_1, V_2, ..., V_k) = (1/2) * sum_i |E(V_i, V_bar_i)|
    """
    
    def __init__(self, max_nodes_per_partition: int = 10000):
        """
        Initialize the graph partitioner.
        
        Args:
            max_nodes_per_partition: Maximum number of nodes per partition
        """
        self.max_nodes_per_partition = max_nodes_per_partition
        self.partitions: Dict[int, Set[int]] = {}
        self.partition_edges: Dict[int, List[Tuple[int, int]]] = {}
        self.cut_edges: List[Tuple[int, int, int, int]] = []  # (src, dst, src_part, dst_part)
    
    def partition_graph(
        self,
        graph: nx.Graph,
        num_partitions: Optional[int] = None,
        method: str = "metis_like"
    ) -> Dict[int, Set[int]]:
        """
        Partition the graph into subgraphs.
        
        Args:
            graph: NetworkX graph object
            num_partitions: Number of partitions (if None, computed from max_nodes_per_partition)
            method: Partitioning method ("metis_like", "community", or "balanced")
        
        Returns:
            Dictionary mapping partition ID to set of node IDs
        """
        if num_partitions is None:
            num_partitions = max(1, len(graph.nodes()) // self.max_nodes_per_partition)
        
        logger.info(f"Partitioning graph with {len(graph.nodes())} nodes into {num_partitions} partitions")
        
        if method == "metis_like":
            self.partitions = self._metis_like_partition(graph, num_partitions)
        elif method == "community":
            self.partitions = self._community_partition(graph, num_partitions)
        elif method == "balanced":
            self.partitions = self._balanced_partition(graph, num_partitions)
        else:
            raise ValueError(f"Unknown partitioning method: {method}")
        
        # Compute cut edges
        self._compute_cut_edges(graph)
        
        logger.info(f"Partitioning complete. Cut edges: {len(self.cut_edges)}")
        return self.partitions
    
    def _metis_like_partition(
        self,
        graph: nx.Graph,
        num_partitions: int
    ) -> Dict[int, Set[int]]:
        """
        Implement a METIS-like partitioning strategy using recursive bisection.
        
        This is a simplified version that uses graph structure to guide partitioning.
        """
        partitions = {}
        
        if num_partitions == 1:
            partitions[0] = set(graph.nodes())
            return partitions
        
        # Use spectral clustering as a heuristic for METIS-like behavior
        try:
            # Try to use spectral clustering
            from sklearn.cluster import SpectralClustering
            
            # Get adjacency matrix
            adj_matrix = nx.to_scipy_sparse_array(graph)
            
            # Apply spectral clustering
            clustering = SpectralClustering(
                n_clusters=num_partitions,
                affinity='precomputed',
                random_state=42,
                assign_labels='kmeans'
            )
            labels = clustering.fit_predict(adj_matrix)
            
            # Group nodes by cluster
            for node_idx, node in enumerate(graph.nodes()):
                partition_id = labels[node_idx]
                if partition_id not in partitions:
                    partitions[partition_id] = set()
                partitions[partition_id].add(node)
        
        except ImportError:
            # Fallback to simple balanced partition
            logger.warning("scikit-learn not available, using balanced partition")
            partitions = self._balanced_partition(graph, num_partitions)
        
        return partitions
    
    def _community_partition(
        self,
        graph: nx.Graph,
        num_partitions: int
    ) -> Dict[int, Set[int]]:
        """
        Use community detection for partitioning.
        """
        try:
            import networkx.algorithms.community as nx_community
            
            # Use Louvain method for community detection
            communities = nx_community.greedy_modularity_communities(graph)
            
            partitions = {}
            for i, community in enumerate(communities[:num_partitions]):
                partitions[i] = set(community)
            
            # If fewer communities than requested, add remaining nodes to largest partition
            if len(communities) < num_partitions:
                remaining_nodes = set(graph.nodes()) - set().union(*[partitions[i] for i in partitions])
                if remaining_nodes:
                    largest_partition = max(partitions.keys(), key=lambda k: len(partitions[k]))
                    partitions[largest_partition].update(remaining_nodes)
        
        except Exception as e:
            logger.warning(f"Community detection failed: {e}, using balanced partition")
            partitions = self._balanced_partition(graph, num_partitions)
        
        return partitions
    
    def _balanced_partition(
        self,
        graph: nx.Graph,
        num_partitions: int
    ) -> Dict[int, Set[int]]:
        """
        Simple balanced partition based on node ordering.
        """
        nodes = list(graph.nodes())
        partition_size = len(nodes) // num_partitions
        
        partitions = {}
        for i in range(num_partitions):
            start_idx = i * partition_size
            end_idx = (i + 1) * partition_size if i < num_partitions - 1 else len(nodes)
            partitions[i] = set(nodes[start_idx:end_idx])
        
        return partitions
    
    def _compute_cut_edges(self, graph: nx.Graph) -> None:
        """
        Compute edges that cross partition boundaries (cut edges).
        """
        self.cut_edges = []
        
        # Create reverse mapping from node to partition
        node_to_partition = {}
        for partition_id, nodes in self.partitions.items():
            for node in nodes:
                node_to_partition[node] = partition_id
        
        # Find cut edges
        for src, dst in graph.edges():
            src_partition = node_to_partition.get(src)
            dst_partition = node_to_partition.get(dst)
            
            if src_partition != dst_partition:
                self.cut_edges.append((src, dst, src_partition, dst_partition))
    
    def get_partition_subgraph(
        self,
        graph: nx.Graph,
        partition_id: int
    ) -> nx.Graph:
        """
        Extract subgraph for a specific partition.
        
        Args:
            graph: Original graph
            partition_id: Partition ID
        
        Returns:
            Subgraph containing only nodes in the partition
        """
        if partition_id not in self.partitions:
            raise ValueError(f"Partition {partition_id} not found")
        
        nodes = self.partitions[partition_id]
        subgraph = graph.subgraph(nodes).copy()
        return subgraph
    
    def get_cut_edge_ratio(self) -> float:
        """
        Get the ratio of cut edges to total edges.
        
        Returns:
            Ratio of cut edges (lower is better)
        """
        total_edges = sum(len(list(self.partitions[p])) for p in self.partitions)
        if total_edges == 0:
            return 0.0
        return len(self.cut_edges) / total_edges
    
    def get_partition_stats(self) -> Dict[str, any]:
        """
        Get statistics about the partitioning.
        
        Returns:
            Dictionary with partition statistics
        """
        partition_sizes = {pid: len(nodes) for pid, nodes in self.partitions.items()}
        
        return {
            "num_partitions": len(self.partitions),
            "partition_sizes": partition_sizes,
            "min_partition_size": min(partition_sizes.values()) if partition_sizes else 0,
            "max_partition_size": max(partition_sizes.values()) if partition_sizes else 0,
            "avg_partition_size": sum(partition_sizes.values()) / len(partition_sizes) if partition_sizes else 0,
            "num_cut_edges": len(self.cut_edges),
            "cut_edge_ratio": self.get_cut_edge_ratio(),
        }
    
    def visualize_partitions(self, output_path: str = "partitions.png") -> None:
        """
        Visualize the partitions (for small graphs).
        
        Args:
            output_path: Path to save visualization
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            
            # Create a simple visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            
            colors = plt.cm.tab20(np.linspace(0, 1, len(self.partitions)))
            
            for partition_id, nodes in self.partitions.items():
                ax.scatter(
                    range(len(nodes)),
                    [partition_id] * len(nodes),
                    c=[colors[partition_id]],
                    label=f"Partition {partition_id}",
                    s=100
                )
            
            ax.set_xlabel("Node Index")
            ax.set_ylabel("Partition ID")
            ax.set_title("Graph Partitions")
            ax.legend()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Partition visualization saved to {output_path}")
        
        except ImportError:
            logger.warning("matplotlib not available for visualization")
