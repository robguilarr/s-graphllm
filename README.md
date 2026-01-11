# S-GraphLLM: Scalable Graph-Augmented Language Model

A cutting-edge system for performing efficient reasoning over large-scale knowledge graphs using hierarchical reasoning and graph-aware attention mechanisms.

## Overview

S-GraphLLM addresses the limitations of existing graph reasoning systems by introducing:

1. **Hierarchical Reasoning Architecture**: A two-stage reasoning process combining coarse-grained and fine-grained analysis
2. **Graph Partitioning**: Efficient division of billion-node graphs into manageable subgraphs using METIS-like algorithms
3. **Graph-Aware Attention**: Specialized attention mechanism that modulates LLM attention based on graph topology
4. **Scalable Design**: Distributed processing capabilities for handling massive knowledge graphs

## Key Features

- **Large-Scale Graph Support**: Handles graphs with billions of nodes and edges
- **Hierarchical Reasoning**: Two-stage reasoning process for improved accuracy and efficiency
- **Graph-Aware Attention**: Attention mechanism modulated by structural similarity
- **Multi-Hop Reasoning**: Support for complex, multi-entity reasoning tasks
- **Benchmark Evaluation**: Built-in support for HotpotQA and WebQSP datasets
- **Modular Architecture**: Clean separation of concerns for easy extension

## System Architecture

```
Input Query + Large Knowledge Graph
         ↓
    Graph Partitioning Agent
    (METIS-like algorithm)
         ↓
    Hierarchical Reasoning Orchestrator
    ├─ Coarse-Grained Reasoning
    │  └─ Identify relevant subgraphs
    ├─ Fine-Grained Reasoning
    │  └─ Detailed multi-hop reasoning
    └─ Answer Synthesis
         ↓
    Graph-Aware LLM Agent
    (with attention modulation)
         ↓
    Final Answer
```

## Installation

### Prerequisites

- Python 3.11+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/s-graphllm/s-graphllm.git
cd s-graphllm
```

2. Install dependencies:
```bash
pip install -e .
```

3. Install optional dependencies for distributed processing:
```bash
pip install -e ".[distributed]"
```

4. Set up environment variables:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

### Basic Usage

```python
import networkx as nx
from src.agents import HierarchicalReasoningOrchestrator, LLMAgent, LLMConfig
from src.utils import Config

# Create or load a knowledge graph
graph = nx.karate_club_graph()

# Initialize configuration
config = Config(
    max_nodes_per_partition=10000,
    max_hops=3,
    model_name="gpt-4.1-mini"
)

# Create orchestrator
orchestrator = HierarchicalReasoningOrchestrator(
    graph=graph,
    config=config
)
orchestrator.setup()

# Initialize LLM agent
llm_config = LLMConfig(model="gpt-4.1-mini")
llm_agent = LLMAgent(llm_config)

# Perform reasoning
query = "What are the key entities and their relationships?"
result = orchestrator.reason(query, llm_agent)

print(result.final_answer)
```

### Command Line Usage

```bash
python -m src.main \
    --config configs/model_config.yaml \
    --query "Your question here" \
    --output output/result.json
```

## Configuration

### Model Configuration (`configs/model_config.yaml`)

```yaml
# Model parameters
model_name: "gpt-4.1-mini"
embedding_dim: 768
hidden_dim: 1024
num_layers: 3
num_heads: 8

# Graph parameters
max_nodes_per_partition: 10000
max_edges_per_partition: 50000
num_partitions: null  # Auto-compute

# Reasoning parameters
max_hops: 3
context_window: 4096
temperature: 0.7
```

## Components

### 1. Graph Partitioner (`src/graph_processing/partitioner.py`)

Divides large graphs into manageable subgraphs using various strategies:

- **METIS-like**: Uses spectral clustering for balanced partitioning
- **Community Detection**: Identifies natural communities in the graph
- **Balanced**: Simple balanced partitioning

```python
from src.graph_processing import GraphPartitioner

partitioner = GraphPartitioner(max_nodes_per_partition=10000)
partitions = partitioner.partition_graph(graph, num_partitions=100)
```

### 2. Graph Coarsener (`src/graph_processing/coarsener.py`)

Creates summarized graphs for hierarchical reasoning:

```python
from src.graph_processing import GraphCoarsener

coarsener = GraphCoarsener()
coarse_graph = coarsener.coarsen_graph(graph, partitions)
relevant_partitions = coarsener.find_relevant_partitions(
    query_keywords, node_features, top_k=5
)
```

### 3. Graph-Aware Attention (`src/attention/graph_aware_attention.py`)

Modulates attention based on graph structure:

```python
from src.attention import GraphAwareAttention, GraphAwareAttentionConfig

config = GraphAwareAttentionConfig(
    hidden_dim=768,
    num_heads=8,
    beta_init=0.5,
    similarity_metric="shortest_path"
)
attention = GraphAwareAttention(config)
```

### 4. Hierarchical Reasoning Orchestrator (`src/agents/orchestrator.py`)

Coordinates the two-stage reasoning process:

```python
from src.agents import HierarchicalReasoningOrchestrator

orchestrator = HierarchicalReasoningOrchestrator(graph, config)
orchestrator.setup()
result = orchestrator.reason(query, llm_agent)
```

### 5. LLM Agent (`src/agents/llm_agent.py`)

Interfaces with OpenAI API for reasoning:

```python
from src.agents import LLMAgent, LLMConfig

llm_config = LLMConfig(model="gpt-4.1-mini", temperature=0.7)
llm_agent = LLMAgent(llm_config)
response = llm_agent.reason(prompt)
```

## Benchmarks

S-GraphLLM is evaluated on standard benchmarks:

### Datasets

- **OGBN-MAG**: Heterogeneous academic graph with 1M+ nodes
- **Wikidata**: Large-scale knowledge graph with 100M+ entities
- **HotpotQA**: Multi-hop question answering dataset
- **WebQSP**: Web question answering with SPARQL annotations

### Metrics

- **Accuracy**: Correctness of answers
- **F1-Score**: Harmonic mean of precision and recall
- **Latency**: End-to-end reasoning time
- **Memory Usage**: Peak memory consumption

## Performance Targets

- **Accuracy**: ≥95% on HotpotQA dev set
- **Latency**: <10 seconds for 3-hop reasoning
- **Scalability**: Support for graphs with 1B+ nodes
- **Memory Efficiency**: Run on 64GB RAM clusters

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run specific test:

```bash
pytest tests/test_partitioning.py -v
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## Experiments

Run the OGBN-MAG experiment:

```bash
bash experiments/run_ogbn_mag.sh
```

## Project Structure

```
s-graphllm/
├── src/
│   ├── agents/
│   │   ├── orchestrator.py      # Hierarchical reasoning orchestrator
│   │   └── llm_agent.py          # LLM agent interface
│   ├── graph_processing/
│   │   ├── partitioner.py        # Graph partitioning
│   │   └── coarsener.py          # Graph coarsening
│   ├── attention/
│   │   └── graph_aware_attention.py  # Graph-aware attention
│   ├── utils.py                  # Utility functions
│   └── main.py                   # Entry point
├── tests/
│   ├── test_partitioning.py      # Partitioning tests
│   └── test_reasoning.py         # Reasoning tests
├── configs/
│   ├── model_config.yaml         # Model configuration
│   └── dataset_config.yaml       # Dataset configuration
├── experiments/
│   └── run_ogbn_mag.sh           # OGBN-MAG experiment
├── pyproject.toml                # Project metadata
└── README.md                     # This file
```

## Technical Details

### Graph Partitioning Objective

Minimize cut edges:
```
minimize cut(V_1, V_2, ..., V_k) = (1/2) * sum_i |E(V_i, V̄_i)|
```

### Graph-Aware Attention

Attention weights modulated by structural similarity:
```
α'_ij = exp(q_i^T k_j / √d_k + β * S(n_i, n_j)) / sum_l exp(...)
```

where S(n_i, n_j) is structural similarity and β is learnable.

### Hierarchical Reasoning Stages

1. **Coarse-Grained**: Identify relevant partitions using summarized graph
2. **Fine-Grained**: Perform detailed reasoning within selected partitions
3. **Synthesis**: Aggregate results into final answer

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use S-GraphLLM in your research, please cite:

```bibtex
@article{s-graphllm2024,
  title={S-GraphLLM: Scalable Graph-Augmented Language Model for Large-Scale Graph Reasoning},
  author={Your Name and Team},
  year={2024}
}
```

## References

- Chai, Z., Zhang, T., Wu, L., et al. (2025). "GraphLLM: Boosting Graph Reasoning Ability of Large Language Model." IEEE Transactions on Big Data.
- [Original GraphLLM Repository](https://github.com/mistyreed63849/Graph-LLM)
- [OGB Benchmarks](https://ogb.stanford.edu/)
- [HotpotQA Dataset](https://hotpotqa.github.io/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on the original GraphLLM framework by Zhejiang University
- Inspired by recent advances in graph neural networks and large language models
- Thanks to the open-source community for excellent tools and datasets

## Support

For issues, questions, or suggestions, please:

1. Check the [GitHub Issues](https://github.com/s-graphllm/s-graphllm/issues)
2. Create a new issue with detailed description
3. Submit a pull request with improvements

---

**Last Updated**: January 2024
**Version**: 0.1.0
**Status**: Active Development
