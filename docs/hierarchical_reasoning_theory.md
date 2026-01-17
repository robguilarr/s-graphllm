# Hierarchical Reasoning Theory

## Overview

Hierarchical reasoning is a cognitive and computational strategy that breaks down complex problems into multiple levels of abstraction, solving them progressively from coarse to fine granularity. This document provides the theoretical foundation for the hierarchical reasoning approach used in S-GraphLLM.

## Table of Contents

1. [Cognitive Foundations](#cognitive-foundations)
2. [Coarse-to-Fine Reasoning](#coarse-to-fine-reasoning)
3. [Divide-and-Conquer Paradigm](#divide-and-conquer-paradigm)
4. [Applications in S-GraphLLM](#applications-in-s-graphllm)
5. [References](#references)

---

## Cognitive Foundations

### Human Hierarchical Reasoning

Human cognition naturally employs hierarchical processing across multiple timescales and levels of abstraction. This is evident in:

1. **Cortical Hierarchy**: Different brain regions process information at different timescales
2. **Abstract-to-Concrete Thinking**: High-level planning followed by detailed execution
3. **Chunking**: Grouping information into meaningful units for efficient processing

**Key Insight**: The human brain doesn't process all details simultaneously but rather uses a hierarchical strategy to manage complexity.

### Hierarchical Reasoning Model (HRM)

Recent research has formalized this approach in AI systems:

**Wang, G., et al. (2025)**. "Hierarchical Reasoning Model." *arXiv preprint arXiv:2506.21734*.
- **Link**: [https://arxiv.org/abs/2506.21734](https://arxiv.org/abs/2506.21734)
- **Key Contribution**: Brain-inspired architecture that organizes computation hierarchically across regions operating at different timescales

**Architecture**:
$$\text{HRM} = \{\text{Layer}_1, \text{Layer}_2, \ldots, \text{Layer}_L\}$$

where each layer $\text{Layer}_i$ operates at a different temporal resolution:
$$\tau_i = \alpha^i \cdot \tau_0, \quad \alpha > 1$$

**Benefits**:
- 100× faster reasoning on complex tasks
- Better performance on ARC-AGI and expert-level puzzles
- More interpretable reasoning traces

---

## Coarse-to-Fine Reasoning

### Definition

**Coarse-to-fine reasoning** is a computational strategy that:
1. First solves a simplified (coarse) version of the problem
2. Uses the coarse solution to guide detailed (fine) problem-solving
3. Iteratively refines the solution at increasing levels of detail

### Mathematical Framework

Given a problem $P$ with solution space $S$, coarse-to-fine reasoning defines a hierarchy:

$$P_0 \rightarrow P_1 \rightarrow \cdots \rightarrow P_n = P$$

where:
- $P_0$ is the coarsest (most abstract) problem
- $P_n$ is the original problem
- Each $P_i$ is a refinement of $P_{i-1}$

**Solution Propagation**:
$$s_i = \text{Refine}(s_{i-1}, P_i), \quad s_0 = \text{Solve}(P_0)$$

### Applications in AI

#### 1. Visual Question Answering

**Nguyen, B. X., et al. (2022)**. "Coarse-to-Fine Reasoning for Visual Question Answering." *CVPR Workshop on Multimodal Learning and Applications*.
- **Link**: [https://arxiv.org/abs/2110.02526](https://arxiv.org/abs/2110.02526)
- **Approach**: 
  - **Coarse Stage**: Identify relevant image regions
  - **Fine Stage**: Detailed reasoning over selected regions

**Architecture**:
$$\text{Answer} = \text{FinePrediction}(\text{CoarseSelection}(\text{Image}, \text{Question}))$$

#### 2. Mathematical Reasoning

**Hu, Y., et al. (2025)**. "Coarse-to-Fine Process Reward Modeling for Mathematical Reasoning." *arXiv preprint arXiv:2501.13622*.
- **Link**: [https://arxiv.org/abs/2501.13622](https://arxiv.org/abs/2501.13622)
- **Approach**:
  - **Coarse**: Evaluate overall solution correctness
  - **Fine**: Identify specific error steps

#### 3. Multi-Agent Reasoning

**Chen, J., et al. (2025)**. "Multi-Agent, Iterative, Coarse-to-Fine Refinement for Reasoning." *EMNLP 2025*.
- **Link**: [https://aclanthology.org/2025.emnlp-main.1660/](https://aclanthology.org/2025.emnlp-main.1660/)
- **Approach**: Multiple agents collaborate in a coarse-to-fine manner with iterative refinement

### Advantages

✅ **Computational Efficiency**: Avoid processing irrelevant details  
✅ **Better Guidance**: Coarse solution provides search direction  
✅ **Robustness**: Less sensitive to local optima  
✅ **Interpretability**: Clear reasoning stages  
✅ **Scalability**: Handles large problem spaces

---

## Divide-and-Conquer Paradigm

### Definition

**Divide-and-conquer** is a fundamental algorithmic paradigm that:
1. **Divide**: Break the problem into smaller subproblems
2. **Conquer**: Solve subproblems recursively
3. **Combine**: Merge subproblem solutions into the final solution

### Mathematical Formulation

For a problem $P$ with input size $n$:

$$T(n) = \begin{cases}
\Theta(1) & \text{if } n \leq c \\
aT(n/b) + D(n) + C(n) & \text{otherwise}
\end{cases}$$

where:
- $a$ = number of subproblems
- $b$ = factor by which problem size is reduced
- $D(n)$ = cost of dividing
- $C(n)$ = cost of combining

**Master Theorem**: For $T(n) = aT(n/b) + f(n)$:
$$T(n) = \begin{cases}
\Theta(n^{\log_b a}) & \text{if } f(n) = O(n^{\log_b a - \epsilon}) \\
\Theta(n^{\log_b a} \log n) & \text{if } f(n) = \Theta(n^{\log_b a}) \\
\Theta(f(n)) & \text{if } f(n) = \Omega(n^{\log_b a + \epsilon})
\end{cases}$$

### Applications to Graphs

#### Graph Separator Theorem

**Lipton, R. J., & Tarjan, R. E. (1979)**. "A Separator Theorem for Planar Graphs." *SIAM Journal on Applied Mathematics*, 36(2), 177-189.

**Theorem**: Any $n$-vertex planar graph has a separator of size $O(\sqrt{n})$ that partitions the graph into two parts, each with at most $2n/3$ vertices.

**Implication**: Enables efficient divide-and-conquer algorithms on planar graphs.

#### Divide-and-Conquer on Graphs

**Even, G., Naor, J. S., Rao, S., & Schieber, B. (2000)**. "Divide-and-Conquer Approximation Algorithms via Spreading Metrics." *Journal of the ACM*, 47(4), 585-616.
- **Link**: [https://dl.acm.org/doi/10.1145/347476.347478](https://dl.acm.org/doi/10.1145/347476.347478)
- **Key Contribution**: Framework for divide-and-conquer approximation algorithms using spreading metrics

**Nowak, A., Folqué, D., & Bruna, J. (2018)**. "Divide and Conquer Networks." *ICLR 2018*.
- **Link**: [https://openreview.net/forum?id=B1jscMbAW](https://openreview.net/forum?id=B1jscMbAW)
- **Key Contribution**: Neural network architecture based on divide-and-conquer for graph problems

---

## Applications in S-GraphLLM

### Hierarchical Reasoning Architecture

S-GraphLLM implements a **two-stage hierarchical reasoning** process for billion-node knowledge graphs:

```
Input Query + Large Knowledge Graph
         ↓
    [COARSE-GRAINED REASONING]
    1. Graph Partitioning
    2. Coarse Graph Construction
    3. Relevant Partition Selection
         ↓
    [FINE-GRAINED REASONING]
    4. Detailed Reasoning per Partition
    5. Multi-hop Path Finding
    6. Entity Relationship Analysis
         ↓
    [ANSWER SYNTHESIS]
    7. Aggregate Results
    8. Generate Final Answer
```

### Stage 1: Coarse-Grained Reasoning

**Goal**: Identify relevant subgraphs without processing all details

**Implementation** (`src/agents/orchestrator.py`):

```python
def coarse_grained_reasoning(self, query: str) -> List[int]:
    """
    Identify relevant partitions using coarse graph.
    
    Returns:
        List of partition IDs to process in fine-grained stage
    """
```

**Process**:
1. **Query Analysis**: Extract key entities and concepts
2. **Coarse Graph Matching**: Find relevant super-nodes in coarse graph
3. **Partition Ranking**: Score partitions by relevance
4. **Top-K Selection**: Select $k$ most relevant partitions

**Mathematical Formulation**:
$$\text{Relevance}(P_i, Q) = \text{Similarity}(\text{Features}(P_i), \text{Embedding}(Q))$$
$$\text{Selected} = \text{TopK}(\{\text{Relevance}(P_i, Q) : i = 1, \ldots, k\}, k=5)$$

### Stage 2: Fine-Grained Reasoning

**Goal**: Perform detailed reasoning within selected partitions

**Implementation**:

```python
def fine_grained_reasoning(
    self, 
    query: str, 
    partition_ids: List[int], 
    llm_agent: LLMAgent
) -> List[ReasoningResult]:
    """
    Detailed multi-hop reasoning within selected partitions.
    """
```

**Process**:
1. **Subgraph Extraction**: Extract full subgraphs for selected partitions
2. **GraphLLM Processing**: Apply graph transformer and encoder-decoder
3. **Multi-hop Reasoning**: Traverse graph to find reasoning paths
4. **Evidence Collection**: Gather supporting facts

**Multi-hop Path Finding**:
$$\text{Path}(s, t) = \{v_0 = s, v_1, \ldots, v_h = t\}$$
$$\text{Score}(\text{Path}) = \prod_{i=0}^{h-1} \text{EdgeWeight}(v_i, v_{i+1})$$

### Stage 3: Answer Synthesis

**Goal**: Combine results from multiple partitions into coherent answer

**Implementation**:

```python
def synthesize_answer(
    self, 
    results: List[ReasoningResult], 
    llm_agent: LLMAgent
) -> str:
    """
    Aggregate reasoning results and generate final answer.
    """
```

**Process**:
1. **Result Aggregation**: Combine evidence from all partitions
2. **Conflict Resolution**: Handle contradictory information
3. **Answer Generation**: Use LLM to generate natural language answer

**Aggregation Strategy**:
$$\text{FinalAnswer} = \text{LLM}\left(\bigcup_{i=1}^{k} \text{Evidence}_i, \text{Query}\right)$$

---

## Theoretical Advantages

### Complexity Analysis

**Without Hierarchical Reasoning**:
- Process entire graph: $O(n \cdot m)$ where $n$ = nodes, $m$ = edges
- Memory: $O(n + m)$
- For billion-node graphs: **Intractable**

**With Hierarchical Reasoning**:
- Coarse-grained: $O(k \cdot m_c)$ where $k$ = partitions, $m_c$ = coarse edges
- Fine-grained: $O(k' \cdot (n/k + m/k))$ where $k'$ = selected partitions
- Memory: $O(n/k + m/k)$ per partition
- **Total**: $O(k \cdot m_c + k' \cdot (n/k + m/k))$ ≪ $O(n \cdot m)$

### Scalability

**Theorem** (Informal): Hierarchical reasoning enables processing of graphs with $n$ nodes in time $O(n \log n)$ instead of $O(n^2)$ by:
1. Reducing search space through coarse-grained filtering
2. Parallel processing of independent partitions
3. Avoiding redundant computation

### Accuracy

**Empirical Observation**: Hierarchical reasoning can improve accuracy by:
- **Focus**: Concentrating computation on relevant subgraphs
- **Context**: Maintaining local graph structure within partitions
- **Robustness**: Reducing noise from irrelevant parts of the graph

---

## Related Work

### Hierarchical Methods in Knowledge Graphs

1. **Ren, H., & Leskovec, J. (2020)**. "Beta Embeddings for Multi-Hop Logical Reasoning in Knowledge Graphs." *NeurIPS 2020*.
   - Uses probabilistic embeddings for hierarchical query answering

2. **Sun, Z., et al. (2019)**. "Recurrent Knowledge Graph Embedding for Effective Recommendation." *RecSys 2019*.
   - Hierarchical reasoning for recommendation systems

3. **Lin, X. V., et al. (2018)**. "Multi-Hop Knowledge Graph Reasoning with Reward Shaping." *EMNLP 2018*.
   - Reinforcement learning with hierarchical reward structure

### Hierarchical Graph Neural Networks

4. **Ying, R., et al. (2018)**. "Hierarchical Graph Representation Learning with Differentiable Pooling." *NeurIPS 2018*.
   - **Link**: [https://arxiv.org/abs/1806.08804](https://arxiv.org/abs/1806.08804)
   - Learns hierarchical graph representations through differentiable pooling

5. **Lee, J., Lee, I., & Kang, J. (2019)**. "Self-Attention Graph Pooling." *ICML 2019*.
   - Hierarchical graph pooling using self-attention

---

## References

### Cognitive Science

1. **Wang, G., et al. (2025)**. "Hierarchical Reasoning Model." *arXiv preprint arXiv:2506.21734*.
   - **Link**: [https://arxiv.org/abs/2506.21734](https://arxiv.org/abs/2506.21734)

2. **Badre, D., & D'Esposito, M. (2009)**. "Is the Rostro-Caudal Axis of the Frontal Lobe Hierarchical?" *Nature Reviews Neuroscience*, 10(9), 659-669.

### Coarse-to-Fine Methods

3. **Nguyen, B. X., et al. (2022)**. "Coarse-to-Fine Reasoning for Visual Question Answering." *CVPR Workshop*.
   - **Link**: [https://arxiv.org/abs/2110.02526](https://arxiv.org/abs/2110.02526)

4. **Hu, Y., et al. (2025)**. "Coarse-to-Fine Process Reward Modeling for Mathematical Reasoning." *arXiv preprint arXiv:2501.13622*.
   - **Link**: [https://arxiv.org/abs/2501.13622](https://arxiv.org/abs/2501.13622)

5. **Chen, J., et al. (2025)**. "Multi-Agent, Iterative, Coarse-to-Fine Refinement for Reasoning." *EMNLP 2025*.
   - **Link**: [https://aclanthology.org/2025.emnlp-main.1660/](https://aclanthology.org/2025.emnlp-main.1660/)

### Divide-and-Conquer Algorithms

6. **Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009)**. *Introduction to Algorithms* (3rd ed.). MIT Press.
   - **Chapter 4**: Divide-and-Conquer

7. **Even, G., Naor, J. S., Rao, S., & Schieber, B. (2000)**. "Divide-and-Conquer Approximation Algorithms via Spreading Metrics." *Journal of the ACM*, 47(4), 585-616.
   - **Link**: [https://dl.acm.org/doi/10.1145/347476.347478](https://dl.acm.org/doi/10.1145/347476.347478)

8. **Lipton, R. J., & Tarjan, R. E. (1979)**. "A Separator Theorem for Planar Graphs." *SIAM Journal on Applied Mathematics*, 36(2), 177-189.

### Neural Networks

9. **Nowak, A., Folqué, D., & Bruna, J. (2018)**. "Divide and Conquer Networks." *ICLR 2018*.
   - **Link**: [https://openreview.net/forum?id=B1jscMbAW](https://openreview.net/forum?id=B1jscMbAW)

10. **Ying, R., et al. (2018)**. "Hierarchical Graph Representation Learning with Differentiable Pooling." *NeurIPS 2018*.
    - **Link**: [https://arxiv.org/abs/1806.08804](https://arxiv.org/abs/1806.08804)

### Knowledge Graphs

11. **Ren, H., & Leskovec, J. (2020)**. "Beta Embeddings for Multi-Hop Logical Reasoning in Knowledge Graphs." *NeurIPS 2020*.
    - **Link**: [https://arxiv.org/abs/2010.11465](https://arxiv.org/abs/2010.11465)

12. **Lin, X. V., et al. (2018)**. "Multi-Hop Knowledge Graph Reasoning with Reward Shaping." *EMNLP 2018*.
    - **Link**: [https://arxiv.org/abs/1808.10568](https://arxiv.org/abs/1808.10568)

---

## Further Reading

### Online Resources

- **Khan Academy**: [Divide and Conquer Algorithms](https://www.khanacademy.org/computing/computer-science/algorithms/merge-sort/a/divide-and-conquer-algorithms)
- **MIT OpenCourseWare**: [Divide-and-Conquer Lecture Notes](https://fanchung.ucsd.edu/teach/202/notes/05divide-and-conquer.pdf)
- **Emergent Mind**: [Coarse-to-Fine Strategy](https://www.emergentmind.com/topics/coarse-to-fine-strategy)

### Implementation Notes

For implementation details specific to S-GraphLLM, see:
- `src/agents/orchestrator.py` - Hierarchical reasoning orchestrator
- `src/graph_processing/coarsener.py` - Graph coarsening for coarse-grained reasoning
- `tests/test_reasoning.py` - Unit tests and examples

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**Maintainer**: S-GraphLLM Team
