"""
Tests for graph partitioning module.
"""

import unittest
import networkx as nx
from src.graph_processing import GraphPartitioner


class TestGraphPartitioner(unittest.TestCase):
    """Test cases for GraphPartitioner."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a sample graph
        self.graph = nx.karate_club_graph()
        self.partitioner = GraphPartitioner(max_nodes_per_partition=10)
    
    def test_partition_graph(self):
        """Test basic graph partitioning."""
        partitions = self.partitioner.partition_graph(self.graph, num_partitions=2)
        
        # Check that all nodes are assigned to a partition
        all_nodes = set()
        for partition_nodes in partitions.values():
            all_nodes.update(partition_nodes)
        
        self.assertEqual(all_nodes, set(self.graph.nodes()))
    
    def test_partition_count(self):
        """Test that correct number of partitions are created."""
        num_partitions = 3
        partitions = self.partitioner.partition_graph(
            self.graph,
            num_partitions=num_partitions
        )
        
        self.assertEqual(len(partitions), num_partitions)
    
    def test_cut_edges(self):
        """Test cut edge computation."""
        self.partitioner.partition_graph(self.graph, num_partitions=2)
        
        # Cut edges should be less than total edges
        self.assertLess(len(self.partitioner.cut_edges), len(self.graph.edges()))
    
    def test_partition_stats(self):
        """Test partition statistics."""
        self.partitioner.partition_graph(self.graph, num_partitions=2)
        stats = self.partitioner.get_partition_stats()
        
        self.assertIn("num_partitions", stats)
        self.assertIn("partition_sizes", stats)
        self.assertIn("num_cut_edges", stats)
        self.assertEqual(stats["num_partitions"], 2)
    
    def test_get_partition_subgraph(self):
        """Test extracting subgraph for a partition."""
        self.partitioner.partition_graph(self.graph, num_partitions=2)
        
        # Get subgraph for first partition
        subgraph = self.partitioner.get_partition_subgraph(self.graph, 0)
        
        # Subgraph should have nodes from partition 0
        self.assertTrue(len(subgraph.nodes()) > 0)
        self.assertLessEqual(len(subgraph.nodes()), len(self.graph.nodes()))


class TestPartitioningMethods(unittest.TestCase):
    """Test different partitioning methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.graph = nx.barabasi_albert_graph(50, 3)
        self.partitioner = GraphPartitioner(max_nodes_per_partition=10)
    
    def test_metis_like_partitioning(self):
        """Test METIS-like partitioning."""
        partitions = self.partitioner.partition_graph(
            self.graph,
            num_partitions=3,
            method="metis_like"
        )
        
        self.assertEqual(len(partitions), 3)
    
    def test_balanced_partitioning(self):
        """Test balanced partitioning."""
        partitions = self.partitioner.partition_graph(
            self.graph,
            num_partitions=3,
            method="balanced"
        )
        
        self.assertEqual(len(partitions), 3)
        
        # Check that partitions are roughly balanced
        sizes = [len(p) for p in partitions.values()]
        max_size = max(sizes)
        min_size = min(sizes)
        
        # Sizes should not differ by more than 2 nodes
        self.assertLessEqual(max_size - min_size, 2)
    
    def test_community_partitioning(self):
        """Test community-based partitioning."""
        partitions = self.partitioner.partition_graph(
            self.graph,
            num_partitions=3,
            method="community"
        )
        
        self.assertGreater(len(partitions), 0)


if __name__ == "__main__":
    unittest.main()
