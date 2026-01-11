"""
Tests for hierarchical reasoning module.
"""

import unittest
import networkx as nx
from unittest.mock import Mock, patch
from src.agents import HierarchicalReasoningOrchestrator, LLMAgent, LLMConfig
from src.utils import Config


class TestHierarchicalReasoningOrchestrator(unittest.TestCase):
    """Test cases for HierarchicalReasoningOrchestrator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a sample graph
        self.graph = nx.karate_club_graph()
        
        # Create node features
        self.node_features = {
            node: {"description": f"Node {node}"}
            for node in self.graph.nodes()
        }
        
        # Create configuration
        self.config = Config(
            max_nodes_per_partition=10,
            max_hops=3,
            context_window=2048
        )
        
        # Create orchestrator
        self.orchestrator = HierarchicalReasoningOrchestrator(
            graph=self.graph,
            config=self.config,
            node_features=self.node_features
        )
    
    def test_orchestrator_setup(self):
        """Test orchestrator setup."""
        self.orchestrator.setup()
        
        # Check that partitions are created
        self.assertIsNotNone(self.orchestrator.partitions)
        self.assertGreater(len(self.orchestrator.partitions), 0)
        
        # Check that coarse graph is created
        self.assertIsNotNone(self.orchestrator.coarse_graph)
    
    def test_partition_coverage(self):
        """Test that all nodes are covered by partitions."""
        self.orchestrator.setup()
        
        all_nodes = set()
        for partition_nodes in self.orchestrator.partitions.values():
            all_nodes.update(partition_nodes)
        
        self.assertEqual(all_nodes, set(self.graph.nodes()))
    
    def test_get_partition_info(self):
        """Test getting partition information."""
        self.orchestrator.setup()
        
        # Get info for first partition
        partition_id = list(self.orchestrator.partitions.keys())[0]
        info = self.orchestrator.get_partition_info(partition_id)
        
        self.assertIn("partition_id", info)
        self.assertIn("num_nodes", info)
        self.assertIn("num_edges", info)
        self.assertEqual(info["partition_id"], partition_id)
    
    @patch('src.agents.llm_agent.OpenAI')
    def test_reasoning_with_mock_llm(self, mock_openai):
        """Test reasoning with mocked LLM."""
        # Set up mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices[0].message.content = "Test reasoning output"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Set up orchestrator
        self.orchestrator.setup()
        
        # Create LLM agent
        llm_config = LLMConfig(model="gpt-4.1-mini")
        llm_agent = LLMAgent(llm_config)
        
        # Perform reasoning
        query = "What is the structure of this graph?"
        result = self.orchestrator.reason(query, llm_agent)
        
        # Check result
        self.assertEqual(result.query, query)
        self.assertIsNotNone(result.coarse_reasoning)
        self.assertGreater(len(result.selected_partitions), 0)
        self.assertIsNotNone(result.final_answer)
    
    def test_reasoning_history(self):
        """Test reasoning history tracking."""
        self.orchestrator.setup()
        
        # Create mock LLM agent
        mock_llm = Mock()
        mock_llm.reason.return_value = "Mock reasoning output"
        
        # Perform reasoning
        query = "Test query"
        result = self.orchestrator.reason(query, mock_llm)
        
        # Check history
        self.assertEqual(len(self.orchestrator.reasoning_history), 1)
        self.assertEqual(self.orchestrator.reasoning_history[0].query, query)


class TestLLMAgent(unittest.TestCase):
    """Test cases for LLMAgent."""
    
    @patch('src.agents.llm_agent.OpenAI')
    def setUp(self, mock_openai):
        """Set up test fixtures."""
        # Set up mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Create LLM agent
        self.config = LLMConfig(model="gpt-4.1-mini")
        self.agent = LLMAgent(self.config)
    
    @patch('src.agents.llm_agent.OpenAI')
    def test_agent_initialization(self, mock_openai):
        """Test agent initialization."""
        config = LLMConfig(model="gpt-4.1-mini")
        agent = LLMAgent(config)
        
        self.assertEqual(agent.config.model, "gpt-4.1-mini")
        self.assertEqual(len(agent.conversation_history), 0)
    
    @patch('src.agents.llm_agent.OpenAI')
    def test_reason(self, mock_openai):
        """Test reasoning."""
        # Set up mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices[0].message.content = "Test reasoning"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        # Create agent
        config = LLMConfig(model="gpt-4.1-mini")
        agent = LLMAgent(config)
        
        # Perform reasoning
        prompt = "What is 2+2?"
        response = agent.reason(prompt)
        
        self.assertEqual(response, "Test reasoning")
        self.assertEqual(len(agent.conversation_history), 2)


if __name__ == "__main__":
    unittest.main()
