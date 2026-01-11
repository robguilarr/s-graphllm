"""
Main entry point for S-GraphLLM system.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import networkx as nx

from agents import HierarchicalReasoningOrchestrator, LLMAgent, LLMConfig
from utils import Config, load_config, setup_logger

# Set up logging
logger = setup_logger(__name__, logging.INFO)


def create_sample_graph(num_nodes: int = 100) -> nx.Graph:
    """
    Create a sample knowledge graph for testing.
    
    Args:
        num_nodes: Number of nodes in the graph
    
    Returns:
        NetworkX graph object
    """
    logger.info(f"Creating sample graph with {num_nodes} nodes...")
    
    # Create a scale-free graph (similar to real knowledge graphs)
    graph = nx.barabasi_albert_graph(num_nodes, 3)
    
    # Add node features
    node_features = {}
    for node in graph.nodes():
        node_features[node] = {
            "description": f"Entity {node}: A knowledge graph node representing a concept or entity",
            "type": "entity"
        }
    
    logger.info(f"Sample graph created with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
    
    return graph, node_features


def load_graph_from_file(graph_path: str) -> nx.Graph:
    """
    Load a graph from file (GraphML or other formats).
    
    Args:
        graph_path: Path to the graph file
    
    Returns:
        NetworkX graph object
    """
    logger.info(f"Loading graph from {graph_path}...")
    
    if graph_path.endswith(".graphml"):
        graph = nx.read_graphml(graph_path)
    elif graph_path.endswith(".gml"):
        graph = nx.read_gml(graph_path)
    elif graph_path.endswith(".edgelist"):
        graph = nx.read_edgelist(graph_path)
    else:
        raise ValueError(f"Unsupported graph format: {graph_path}")
    
    logger.info(f"Graph loaded: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
    
    return graph


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="S-GraphLLM: Scalable Graph-Augmented Language Model"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--graph",
        type=str,
        default=None,
        help="Path to graph file (if not provided, a sample graph is created)"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What are the key entities in this graph?",
        help="Query to answer"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/result.json",
        help="Path to save output"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    logger.setLevel(log_level)
    
    try:
        # Load configuration
        if Path(args.config).exists():
            config = load_config(args.config)
            logger.info(f"Configuration loaded from {args.config}")
        else:
            config = Config()
            logger.info("Using default configuration")
        
        # Load or create graph
        if args.graph:
            graph = load_graph_from_file(args.graph)
            node_features = {}
        else:
            graph, node_features = create_sample_graph(num_nodes=100)
        
        # Initialize orchestrator
        logger.info("Initializing hierarchical reasoning orchestrator...")
        orchestrator = HierarchicalReasoningOrchestrator(
            graph=graph,
            config=config,
            node_features=node_features
        )
        
        # Set up orchestrator
        orchestrator.setup()
        
        # Initialize LLM agent
        logger.info("Initializing LLM agent...")
        llm_config = LLMConfig(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.context_window
        )
        llm_agent = LLMAgent(llm_config)
        
        # Perform reasoning
        logger.info(f"Performing hierarchical reasoning for query: {args.query}")
        result = orchestrator.reason(args.query, llm_agent)
        
        # Print results
        print("\n" + "="*80)
        print("HIERARCHICAL REASONING RESULT")
        print("="*80)
        print(f"\nQuery: {result.query}")
        print(f"\nCoarse-Grained Reasoning:\n{result.coarse_reasoning}")
        print(f"\nSelected Partitions: {result.selected_partitions}")
        print(f"\nFine-Grained Reasoning:\n{result.fine_grained_reasoning}")
        print(f"\nFinal Answer:\n{result.final_answer}")
        print(f"\nConfidence: {result.confidence:.2f}")
        print("\n" + "="*80)
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        orchestrator.save_reasoning_history(str(output_path))
        logger.info(f"Results saved to {output_path}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
