# S-GraphLLM Documentation

Welcome to the S-GraphLLM theoretical documentation. This directory contains comprehensive explanations of the algorithms and theories underlying the system's architecture.

## 📚 Documentation Index

### Core Theory Documents

1. **[Graph Partitioning Theory](graph_partitioning_theory.md)**
   - METIS algorithm and multilevel partitioning
   - Spectral graph partitioning
   - Complexity analysis and performance considerations
   - Applications in scalable graph processing

2. **[Hierarchical Reasoning Theory](hierarchical_reasoning_theory.md)**
   - Cognitive foundations of hierarchical reasoning
   - Coarse-to-fine reasoning strategies
   - Divide-and-conquer paradigm
   - Applications in knowledge graph reasoning

### Architecture Documentation

3. **[Component Guide](component_guide.md)** ⭐ NEW
   - Step-by-step explanation of every component
   - Theoretical basis with paper citations
   - Mechanism of action for each component
   - Data flow and input/output specifications

4. **[Architecture Diagram](architecture_diagram.md)** ⭐ NEW
   - Complete Mermaid.js architecture diagram
   - Layer-by-layer visualization
   - Stage-by-stage sequence diagrams
   - Implementation file mapping

## 🎯 Quick Navigation

### By Component

| Component | Theory Document | Implementation |
|-----------|----------------|----------------|
| **Graph Partitioning** | [Graph Partitioning Theory](graph_partitioning_theory.md) | `src/graph_processing/partitioner.py` |
| **Graph Coarsening** | [Hierarchical Reasoning Theory](hierarchical_reasoning_theory.md) | `src/graph_processing/coarsener.py` |
| **Hierarchical Orchestrator** | [Hierarchical Reasoning Theory](hierarchical_reasoning_theory.md) | `src/agents/orchestrator.py` |
| **Graph Transformer** | See [GraphLLM Paper](https://arxiv.org/abs/2310.05845) | `src/graph_processing/graph_transformer.py` |
| **Node Encoder-Decoder** | See [GraphLLM Paper](https://arxiv.org/abs/2310.05845) | `src/agents/node_encoder_decoder.py` |

### By Research Area

#### Graph Algorithms
- **Partitioning**: [Graph Partitioning Theory](graph_partitioning_theory.md) § METIS Algorithm
- **Spectral Methods**: [Graph Partitioning Theory](graph_partitioning_theory.md) § Spectral Graph Partitioning
- **Coarsening**: [Hierarchical Reasoning Theory](hierarchical_reasoning_theory.md) § Applications in S-GraphLLM

#### Reasoning Methods
- **Hierarchical Reasoning**: [Hierarchical Reasoning Theory](hierarchical_reasoning_theory.md) § Hierarchical Reasoning Architecture
- **Coarse-to-Fine**: [Hierarchical Reasoning Theory](hierarchical_reasoning_theory.md) § Coarse-to-Fine Reasoning
- **Multi-hop Reasoning**: [Hierarchical Reasoning Theory](hierarchical_reasoning_theory.md) § Stage 2: Fine-Grained Reasoning

#### Graph Neural Networks
- **Graph Transformers**: See [GraphLLM Paper](https://arxiv.org/abs/2310.05845) and [GRIT Paper](https://arxiv.org/abs/2106.05234)
- **Positional Encoding**: [Graph Partitioning Theory](graph_partitioning_theory.md) § Graph Laplacian

## 📖 Reading Guide

### For Newcomers

Start with these sections to understand the high-level architecture:

1. **[Hierarchical Reasoning Theory](hierarchical_reasoning_theory.md) § Overview**
   - Understand why hierarchical reasoning is needed
   - Learn about coarse-to-fine strategies

2. **[Graph Partitioning Theory](graph_partitioning_theory.md) § Problem Definition**
   - Understand the graph partitioning problem
   - Learn why it's essential for scalability

3. **[Hierarchical Reasoning Theory](hierarchical_reasoning_theory.md) § Applications in S-GraphLLM**
   - See how theory translates to implementation
   - Understand the two-stage reasoning process

### For Researchers

Deep dive into the theoretical foundations:

1. **[Graph Partitioning Theory](graph_partitioning_theory.md) § METIS Algorithm**
   - Multilevel paradigm details
   - Coarsening, partitioning, and refinement phases

2. **[Graph Partitioning Theory](graph_partitioning_theory.md) § Spectral Graph Partitioning**
   - Graph Laplacian and eigenvectors
   - Cheeger's inequality and conductance

3. **[Hierarchical Reasoning Theory](hierarchical_reasoning_theory.md) § Theoretical Advantages**
   - Complexity analysis
   - Scalability proofs

### For Developers

Focus on implementation-relevant sections:

1. **[Graph Partitioning Theory](graph_partitioning_theory.md) § Applications in S-GraphLLM**
   - Partitioning strategies used in the code
   - Performance considerations

2. **[Hierarchical Reasoning Theory](hierarchical_reasoning_theory.md) § Applications in S-GraphLLM**
   - Stage-by-stage implementation details
   - Code examples and formulas

## 🔗 External Resources

### Foundational Papers

#### Graph Partitioning
- **METIS**: [Karypis & Kumar (1998)](https://www.cs.utexas.edu/~pingali/CS395T/2009fa/papers/metis.pdf) - "A Fast and High Quality Multilevel Scheme for Partitioning Irregular Graphs"
- **Spectral Methods**: [Pothen et al. (1990)](https://epubs.siam.org/doi/10.1137/0611030) - "Partitioning Sparse Matrices with Eigenvectors of Graphs"
- **Survey**: [Fjällström (1998)](https://www.diva-portal.org/smash/get/diva2:1715376/FULLTEXT01.pdf) - "Algorithms for Graph Partitioning: A Survey"

#### Hierarchical Reasoning
- **HRM**: [Wang et al. (2025)](https://arxiv.org/abs/2506.21734) - "Hierarchical Reasoning Model"
- **Coarse-to-Fine**: [Nguyen et al. (2022)](https://arxiv.org/abs/2110.02526) - "Coarse-to-Fine Reasoning for Visual Question Answering"
- **Divide-and-Conquer**: [Even et al. (2000)](https://dl.acm.org/doi/10.1145/347476.347478) - "Divide-and-Conquer Approximation Algorithms"

#### GraphLLM
- **GraphLLM**: [Chai et al. (2025)](https://arxiv.org/abs/2310.05845) - "GraphLLM: Boosting Graph Reasoning Ability of Large Language Model"
- **GRIT**: [Ma et al. (2021)](https://arxiv.org/abs/2106.05234) - "Graph Transformer Networks with Relative Position Encoding"

### Software and Tools
- **METIS**: [Official Website](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview)
- **PyTorch Geometric**: [Documentation](https://pytorch-geometric.readthedocs.io/)
- **NetworkX**: [Graph Algorithms](https://networkx.org/documentation/stable/reference/algorithms/index.html)

### Textbooks
- **Spectral Graph Theory**: [Fan Chung (1997)](http://www.math.ucsd.edu/~fan/research/revised.html)
- **Introduction to Algorithms**: [CLRS (2009)](https://mitpress.mit.edu/9780262033848/introduction-to-algorithms/) - Chapter 4: Divide-and-Conquer

## 🎓 Citation

If you use these theoretical foundations in your research, please cite the relevant papers:

```bibtex
@article{karypis1998fast,
  title={A fast and high quality multilevel scheme for partitioning irregular graphs},
  author={Karypis, George and Kumar, Vipin},
  journal={SIAM Journal on scientific Computing},
  volume={20},
  number={1},
  pages={359--392},
  year={1998}
}

@article{wang2025hierarchical,
  title={Hierarchical Reasoning Model},
  author={Wang, G and others},
  journal={arXiv preprint arXiv:2506.21734},
  year={2025}
}

@article{chai2025graphllm,
  title={GraphLLM: Boosting Graph Reasoning Ability of Large Language Model},
  author={Chai, Ziwei and Zhang, Tianjie and Wu, Liang and others},
  journal={IEEE Transactions on Big Data},
  year={2025}
}
```

## 📝 Contributing

To contribute to the documentation:

1. Follow the existing structure and formatting
2. Include proper citations with links
3. Provide mathematical formulations where appropriate
4. Add practical examples and code references
5. Update this index when adding new documents

## 📧 Contact

For questions or suggestions about the documentation:
- Open an issue on [GitHub](https://github.com/robguilarr/s-graphllm/issues)
- See the main [README](../README.md) for project information

---

**Last Updated**: January 2026  
**Version**: 1.0
