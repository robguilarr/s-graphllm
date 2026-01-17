# Graph Partitioning Theory and METIS Algorithm

## Overview

Graph partitioning is a fundamental problem in computer science that involves dividing a graph into smaller subgraphs while minimizing the number of edges cut between partitions. This document provides the theoretical foundation for the graph partitioning techniques used in S-GraphLLM's scalability layer.

## Table of Contents

1. [Problem Definition](#problem-definition)
2. [METIS Algorithm](#metis-algorithm)
3. [Spectral Graph Partitioning](#spectral-graph-partitioning)
4. [Applications in S-GraphLLM](#applications-in-s-graphllm)
5. [References](#references)

---

## Problem Definition

### Graph Partitioning Problem

Given a graph $G = (V, E)$ with $n$ vertices and $m$ edges, the **k-way graph partitioning problem** aims to partition the vertex set $V$ into $k$ disjoint subsets $V_1, V_2, \ldots, V_k$ such that:

1. **Balance Constraint**: Each partition has approximately equal size
   $$\left| |V_i| - \frac{n}{k} \right| \leq \epsilon \cdot \frac{n}{k}, \quad \forall i \in \{1, 2, \ldots, k\}$$

2. **Minimize Edge Cut**: The number of edges crossing partition boundaries is minimized
   $$\text{cut}(V_1, V_2, \ldots, V_k) = \frac{1}{2} \sum_{i=1}^{k} |E(V_i, \bar{V}_i)|$$
   where $E(V_i, \bar{V}_i)$ denotes edges with one endpoint in $V_i$ and the other in $V \setminus V_i$

### Complexity

The graph partitioning problem is **NP-complete** for $k \geq 2$, meaning no polynomial-time exact algorithm is known unless P = NP. Therefore, practical algorithms focus on heuristics and approximations.

---

## METIS Algorithm

### Overview

**METIS** is a family of multilevel graph partitioning algorithms developed by George Karypis and Vipin Kumar at the University of Minnesota. It is one of the most widely used graph partitioning tools in scientific computing.

**Key Paper**: Karypis, G., & Kumar, V. (1998). "A Fast and High Quality Multilevel Scheme for Partitioning Irregular Graphs." *SIAM Journal on Scientific Computing*, 20(1), 359-392.

**Official Repository**: [http://glaros.dtc.umn.edu/gkhome/metis/metis/overview](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview)

### Three-Phase Multilevel Approach

METIS uses a **multilevel paradigm** consisting of three phases:

#### 1. Coarsening Phase

Progressively reduce the graph size by collapsing vertices and edges.

**Matching Strategy**: Find a maximal matching $M \subseteq E$ such that no two edges in $M$ share a vertex.

**Contraction**: Merge matched vertices into super-vertices:
$$G_{i+1} = \text{Contract}(G_i, M_i)$$

**Weight Update**: Super-vertex weight = sum of constituent vertex weights
$$w(v_{\text{super}}) = \sum_{v \in \text{matched}} w(v)$$

This process continues until the coarsest graph has fewer than a threshold number of vertices (typically 100-200).

#### 2. Initial Partitioning Phase

Compute a partition of the coarsest graph $G_m$ using:
- **Recursive Bisection**: Recursively divide into two parts
- **Direct k-way**: Directly partition into $k$ parts using greedy algorithms
- **Spectral Methods**: Use eigenvectors of the Laplacian matrix

#### 3. Uncoarsening and Refinement Phase

Project the partition back to finer graphs and refine at each level.

**Projection**: Map partition $P_i$ of $G_i$ to partition $P_{i-1}$ of $G_{i-1}$

**Refinement**: Use **Kernighan-Lin (KL)** or **Fiduccia-Mattheyses (FM)** algorithms to improve the partition by moving boundary vertices between partitions.

**FM Algorithm** (used in METIS):
- Maintain gain values for moving each vertex
- Use priority queue to select best moves
- Lock moved vertices to avoid cycles
- Accept move sequence if it improves cut

### Advantages of METIS

✅ **Fast**: $O(m)$ complexity for sparse graphs  
✅ **High Quality**: Produces partitions with low edge cuts  
✅ **Scalable**: Handles graphs with millions of vertices  
✅ **Balanced**: Maintains partition size constraints  
✅ **Robust**: Works well across diverse graph types

---

## Spectral Graph Partitioning

### Graph Laplacian

The **Laplacian matrix** of a graph $G = (V, E)$ is defined as:
$$L = D - A$$

where:
- $D$ is the degree matrix (diagonal matrix with $D_{ii} = \deg(v_i)$)
- $A$ is the adjacency matrix

**Normalized Laplacian**:
$$\mathcal{L} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}$$

### Spectral Partitioning Algorithm

**Key Insight**: The eigenvectors of the Laplacian encode graph structure information.

**Algorithm**:
1. Compute the second smallest eigenvector $v_2$ of $L$ (called the **Fiedler vector**)
2. Sort vertices by their values in $v_2$
3. Partition by splitting at the median value

**Theoretical Foundation**: **Cheeger's Inequality** relates the edge cut to eigenvalues:
$$\frac{\lambda_2}{2} \leq h(G) \leq \sqrt{2 \lambda_2}$$

where:
- $\lambda_2$ is the second smallest eigenvalue of $\mathcal{L}$
- $h(G)$ is the **conductance** (normalized edge cut)

**Key Papers**:
- Fiedler, M. (1973). "Algebraic connectivity of graphs." *Czechoslovak Mathematical Journal*, 23(2), 298-305.
- Pothen, A., Simon, H. D., & Liou, K. P. (1990). "Partitioning sparse matrices with eigenvectors of graphs." *SIAM Journal on Matrix Analysis and Applications*, 11(3), 430-452.

**Reference**: [Spectral Graph Theory by Fan Chung](http://www.math.ucsd.edu/~fan/research/revised.html)

---

## Applications in S-GraphLLM

### Why Graph Partitioning?

S-GraphLLM uses graph partitioning to enable **scalable reasoning** over billion-node knowledge graphs:

1. **Memory Constraints**: Modern LLMs cannot process entire large graphs in a single forward pass
2. **Computational Efficiency**: Partitioning enables parallel processing
3. **Hierarchical Reasoning**: Coarse-grained partition selection followed by fine-grained reasoning

### Implementation Strategy

S-GraphLLM implements a **METIS-like algorithm** with the following features:

#### Partitioning Strategies

**1. Spectral Clustering** (`src/graph_processing/partitioner.py`):
```python
def _spectral_partition(self, graph, num_partitions):
    """
    Uses spectral clustering based on graph Laplacian.
    
    Theory: Computes eigenvectors of normalized Laplacian
    and applies k-means clustering.
    """
```

**Mathematical Formulation**:
$$L_{\text{norm}} = I - D^{-1/2} A D^{-1/2}$$
$$\text{Eigenvectors: } L_{\text{norm}} v_i = \lambda_i v_i$$
$$\text{Clustering: } \text{KMeans}([v_2, v_3, \ldots, v_k])$$

**2. Community Detection**:
Uses Louvain algorithm for modularity-based partitioning.

**Modularity**:
$$Q = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

**3. Balanced Partitioning**:
Simple greedy algorithm ensuring size constraints.

### Hierarchical Graph Coarsening

S-GraphLLM also implements **graph coarsening** (`src/graph_processing/coarsener.py`) to create summarized representations:

**Coarsening Process**:
1. **Partition-based Summarization**: Each partition becomes a super-node
2. **Edge Aggregation**: Inter-partition edges become super-edges with aggregated weights
3. **Feature Aggregation**: Node features are aggregated (mean, max, or attention-weighted)

**Mathematical Formulation**:
$$G_{\text{coarse}} = (V_{\text{coarse}}, E_{\text{coarse}})$$
$$V_{\text{coarse}} = \{v_1^*, v_2^*, \ldots, v_k^*\} \quad \text{(one per partition)}$$
$$w(v_i^*) = \text{Aggregate}(\{w(v) : v \in V_i\})$$

---

## Performance Considerations

### Time Complexity

| Algorithm | Complexity | Notes |
|-----------|------------|-------|
| **METIS** | $O(m)$ | For sparse graphs with $m$ edges |
| **Spectral** | $O(n^3)$ or $O(kn^2)$ | Depends on eigenvalue solver |
| **Louvain** | $O(m)$ | Fast in practice |

### Quality Metrics

**Edge Cut Ratio**:
$$\text{ECR} = \frac{\text{cut}(V_1, \ldots, V_k)}{|E|}$$

**Balance Factor**:
$$\text{BF} = \max_i \frac{|V_i|}{\lceil n/k \rceil}$$

**Communication Volume** (for parallel processing):
$$\text{CV} = \sum_{i=1}^{k} |\{v \in V_i : \exists (v,u) \in E, u \notin V_i\}|$$

---

## References

### Foundational Papers

1. **Karypis, G., & Kumar, V. (1998)**. "A Fast and High Quality Multilevel Scheme for Partitioning Irregular Graphs." *SIAM Journal on Scientific Computing*, 20(1), 359-392.
   - **Link**: [https://www.cs.utexas.edu/~pingali/CS395T/2009fa/papers/metis.pdf](https://www.cs.utexas.edu/~pingali/CS395T/2009fa/papers/metis.pdf)
   - **DOI**: 10.1137/S1064827595287997

2. **Karypis, G., & Kumar, V. (1999)**. "Multilevel k-way Partitioning Scheme for Irregular Graphs." *Journal of Parallel and Distributed Computing*, 48(1), 96-129.
   - **Link**: [https://www.sciencedirect.com/science/article/pii/S0743731597914040](https://www.sciencedirect.com/science/article/pii/S0743731597914040)

3. **Kernighan, B. W., & Lin, S. (1970)**. "An Efficient Heuristic Procedure for Partitioning Graphs." *Bell System Technical Journal*, 49(2), 291-307.
   - **DOI**: 10.1002/j.1538-7305.1970.tb01770.x

4. **Fiduccia, C. M., & Mattheyses, R. M. (1982)**. "A Linear-Time Heuristic for Improving Network Partitions." *19th Design Automation Conference*, 175-181.
   - **Link**: [IEEE Xplore](https://ieeexplore.ieee.org/document/1585498)

### Spectral Methods

5. **Fiedler, M. (1973)**. "Algebraic Connectivity of Graphs." *Czechoslovak Mathematical Journal*, 23(2), 298-305.

6. **Pothen, A., Simon, H. D., & Liou, K. P. (1990)**. "Partitioning Sparse Matrices with Eigenvectors of Graphs." *SIAM Journal on Matrix Analysis and Applications*, 11(3), 430-452.
   - **DOI**: 10.1137/0611030

7. **Chung, F. R. K. (1997)**. *Spectral Graph Theory*. American Mathematical Society.
   - **Book**: [http://www.math.ucsd.edu/~fan/research/revised.html](http://www.math.ucsd.edu/~fan/research/revised.html)

### Surveys

8. **Fjällström, P. O. (1998)**. "Algorithms for Graph Partitioning: A Survey." *Linköping Electronic Articles in Computer and Information Science*, 3(10).
   - **Link**: [https://www.diva-portal.org/smash/get/diva2:1715376/FULLTEXT01.pdf](https://www.diva-portal.org/smash/get/diva2:1715376/FULLTEXT01.pdf)

9. **Buluç, A., Meyerhenke, H., Safro, I., Sanders, P., & Schulz, C. (2016)**. "Recent Advances in Graph Partitioning." *Algorithm Engineering*, 117-158.
   - **Link**: [Springer](https://link.springer.com/chapter/10.1007/978-3-319-49487-6_4)

### Software and Tools

10. **METIS Official Website**: [http://glaros.dtc.umn.edu/gkhome/metis/metis/overview](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview)

11. **ParMETIS** (Parallel METIS): [http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview](http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview)

12. **KaHIP** (Karlsruhe High Quality Partitioning): [https://kahip.github.io/](https://kahip.github.io/)

### Related Work in Knowledge Graphs

13. **Chen, L., et al. (2020)**. "Scalable Knowledge Graph Construction over Large Document Collections." *VLDB*, 13(11), 2019-2032.

14. **Zhu, R., et al. (2019)**. "Scalable Graph Embedding for Asymmetric Proximity." *AAAI*, 33(01), 5914-5921.

---

## Further Reading

### Online Resources

- **MIT OpenCourseWare**: [Spectral Graph Theory Lecture Notes](https://www.stat.berkeley.edu/~mmahoney/s15-stat260-cs294/Lectures/lecture06-10feb15.pdf)
- **Stanford CS267**: [Graph Partitioning and Sparse Matrix Ordering](https://people.eecs.berkeley.edu/~demmel/cs267/)
- **Tutorial**: [Graph Partitioning Tutorial by Bruce Hendrickson](https://www.cs.purdue.edu/homes/apothen/Papers/partsurvey.pdf)

### Implementation Notes

For implementation details specific to S-GraphLLM, see:
- `src/graph_processing/partitioner.py` - Graph partitioning implementation
- `src/graph_processing/coarsener.py` - Graph coarsening implementation
- `tests/test_partitioning.py` - Unit tests and examples

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**Maintainer**: S-GraphLLM Team
