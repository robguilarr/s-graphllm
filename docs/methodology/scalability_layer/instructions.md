# Instructions: Create Detailed Scalability Layer Documentation

## Objective

Create **2 separate Markdown files** — one for each of the Scalability Layer components in S-GraphLLM — plus **1 integration file** that connects the dots between them and shows how the Scalability Layer feeds into the Neural and Reasoning Layers. Each file must follow the template structure below while being **significantly more detailed** than the template: every class, method, data structure, and helper function must be cited with its **exact file path and line numbers** in the codebase.

Use `@README.md` and `@docs/` (especially `@docs/component_guide.md`, `@docs/architecture_diagram.md`, and `@docs/graph_partitioning_theory.md`) as additional information sources to enrich descriptions, cross-reference theoretical foundations, and ensure consistency with the existing documentation.

---

## Files to Create

| # | Filename | Component | Primary Source File |
|---|----------|-----------|---------------------|
| 1 | `docs/methodology/scalability_layer/graph_partitioner.md` | **Graph Partitioner — The Divide Engine** | `src/graph_processing/partitioner.py` |
| 2 | `docs/methodology/scalability_layer/graph_coarsener.md` | **Graph Coarsener — The Summarize Engine** | `src/graph_processing/coarsener.py` |
| 3 | `docs/methodology/scalability_layer/layer_integration.md` | **Scalability Layer Integration** | Multiple sources (see § Integration File Guidance) |

> **Note**: All files go under the `docs/methodology/scalability_layer/` directory.

---

## Context: The Scalability Layer in S-GraphLLM

The Scalability Layer is **Layer 2** in the S-GraphLLM four-layer stack (**Scalability → Neural → Reasoning → Output**). It is the entry gate for large knowledge graphs — its job is to break a graph that may contain millions of nodes into manageable partitions, then create a coarsened summary graph that the Reasoning Layer can use for efficient top-level navigation. Without this layer, the downstream Neural and Reasoning layers would be unable to process real-world knowledge graphs within memory and compute budgets.

The two components work in a strict pipeline:

1. **Graph Partitioner** receives the raw knowledge graph $G = (V, E)$ and divides it into $k$ partitions $\{P_1, P_2, \ldots, P_k\}$ using METIS-like spectral clustering, community detection (Louvain), or balanced partitioning.
2. **Graph Coarsener** takes the partitions and creates a coarsened graph $G_\text{coarse} = (V^*, E^*)$ where each super-node represents an entire partition and edges represent inter-partition connectivity.

The Scalability Layer produces three critical outputs consumed downstream:
- **Partitioned subgraphs** → Neural Layer (adjacency matrices $A$ for RRWP and structural similarity)
- **Coarse graph** $G_\text{coarse}$ → Reasoning Layer Stage 1 (coarse-grained partition selection)
- **Partition metadata** → Reasoning Layer Stage 2 (fine-grained subgraph extraction)

---

## Template & Format

Each component file **must** contain the following sections. Expand every section with the level of detail described in the annotations.

````markdown
# [Component Name] — [Subtitle]

> One-paragraph executive summary: what this component does in the overall S-GraphLLM pipeline,
> why it exists, and which theoretical foundations it implements.
> Reference the relevant section of `@README.md`, `@docs/component_guide.md`, and
> `@docs/graph_partitioning_theory.md`.

---

## Architecture Overview

A short prose description of how the component fits into the four-layer architecture
(Scalability → Neural → Reasoning → Output). Include a small Mermaid diagram if helpful.

---

## Components Breakdown

For **every class and standalone function** in the primary source file, create a subsection:

### N. ClassName / function_name — (Role in one phrase)

* **Location**: `src/path/to/file.py`, lines XX–YY
* **Purpose**: 2–3 sentence description.
* **Paper Reference**: Algorithm name, paper citation, and/or section from the referenced papers.

#### The Math

Write out **every formula** the component implements. Use LaTeX notation.
For each formula:
  - State the equation.
  - Define every symbol (e.g., "$A$ — the adjacency matrix, passed as `adj_matrix` argument").
  - Map each symbol back to the exact variable in the code.

**Important**: Do NOT place underscore characters inside `\text{}` blocks in LaTeX.
Use `\text{before}\_\text{after}` to split identifiers with underscores,
keeping the escaped underscore in math mode. Example:
`\text{max}\_\text{nodes}\_\text{per}\_\text{partition}` instead of
`\text{max\_nodes\_per\_partition}`.

#### Plain English Input / Output

* **Input**: Describe shape, dtype, and a concrete human-readable example
  (e.g., "A NetworkX graph with 50,000 nodes representing a subset of Wikidata").
* **Output**: Same level of detail.
* **Side Effects / State**: Note any instance variables updated, caches, or stored results.

#### Python Perspective

Provide a **runnable-style** code snippet showing the component in isolation.
Include shapes/types as comments on every line.

```python
# Example:
import networkx as nx
from src.graph_processing.partitioner import GraphPartitioner

G = nx.erdos_renyi_graph(1000, 0.01)        # 1000-node random graph
partitioner = GraphPartitioner(max_nodes_per_partition=200)
partitions = partitioner.partition_graph(G, method="metis_like")
# partitions: Dict[int, Set[int]] — e.g., {0: {0, 1, ...}, 1: {200, 201, ...}, ...}
```

#### Internal Method Walkthrough

For each method inside the class:
  - **Method name** (line range)
  - Step-by-step description of what it does.
  - Which other classes/functions it calls (with file + line references).

---

## Helper / Utility Functions

List any helper functions used by this component from `src/utils.py` or other modules,
with file paths, line ranges, and a brief description of their role.

---

## Configuration

Describe every relevant field from `configs/model_config.yaml` and the `Config` class
(`src/utils.py`, lines 35–74). State default values and how they affect behavior.

---

## Cross-References

- Link to related docs in `@docs/` (e.g., `graph_partitioning_theory.md`, `component_guide.md`).
- Link to the other component doc in this folder.
- Link to the integration doc.
- Reference `@README.md` sections on architecture and validation.
- Link to the Neural Layer docs that consume this layer's outputs.
- Link to the Reasoning Layer docs that consume this layer's outputs.
````

---

## Detailed Per-File Guidance

### File 1 — `docs/methodology/scalability_layer/graph_partitioner.md`

**Primary source**: `src/graph_processing/partitioner.py` (276 lines)

Document the following components in order:

1. **`GraphPartitioner`** (lines 15–276)
   - `__init__(max_nodes_per_partition)` (lines 23–30): Initialization with configurable partition size.
   - `partition_graph(graph, num_partitions, method)` (lines 35–70): Main entry point. Dispatches to the three strategy methods. Auto-computes `num_partitions` when `None`.
   - `_metis_like_partition(graph, num_partitions)` (lines 72–117): METIS-like spectral clustering using scikit-learn's `SpectralClustering` on the adjacency matrix. Falls back to balanced partition if scikit-learn is unavailable.
   - `_community_partition(graph, num_partitions)` (lines 119–148): Community detection via `networkx.community.greedy_modularity_communities` (Louvain-like).
   - `_balanced_partition(graph, num_partitions)` (lines 150–167): Simple round-robin balanced split by node ordering.
   - `_compute_cut_edges(graph)` (lines 169–187): Computes cross-partition edges and stores them in `self.cut_edges`.
   - `get_partition_subgraph(graph, partition_id)` (lines 189–209): Extracts a NetworkX subgraph for a single partition.
   - `get_cut_edge_ratio()` (lines 211–221): Returns the ratio of cut edges to total edges.
   - `get_partition_stats()` (lines 223–240): Returns a dictionary of partition statistics.
   - `visualize_partitions(output_path)` (lines 242–275): Optional matplotlib visualization.

**Math to include** (from METIS / spectral partitioning theory):
- Cut minimization objective: $\text{minimize} \quad \text{cut}(P_1, P_2, \ldots, P_k) = \frac{1}{2} \sum_{i=1}^{k} |E(P_i, \bar{P}_i)|$
- Spectral partitioning via the graph Laplacian: $L = D - A$, where $D$ is the degree matrix and $A$ is the adjacency matrix. The Fiedler vector (second-smallest eigenvector of $L$) defines the partition boundary.
- Balance constraint: $|P_i| \leq \lceil |V| / k \rceil \cdot (1 + \epsilon)$.
- Cut edge ratio: $r_\text{cut} = |E_\text{cut}| / |E|$.
- Reference `@docs/graph_partitioning_theory.md` extensively for METIS phases (coarsening → initial partitioning → uncoarsening/refinement).

**Paper references to cite**:
- Karypis, G., & Kumar, V. (1998). "A Fast and High Quality Multilevel Scheme for Partitioning Irregular Graphs." SIAM Journal on Scientific Computing, 20(1), 359–392.
- Pothen, A., Simon, H. D., & Liou, K. P. (1990). "Partitioning Sparse Matrices with Eigenvectors of Graphs." SIAM Journal on Matrix Analysis and Applications, 11(3), 430–452.

---

### File 2 — `docs/methodology/scalability_layer/graph_coarsener.md`

**Primary source**: `src/graph_processing/coarsener.py` (283 lines)

Document the following components in order:

1. **`GraphCoarsener`** (lines 14–283)
   - `__init__()` (lines 20–24): Initializes internal stores for coarse graph, node summaries, and partition-to-node mappings.
   - `coarsen_graph(graph, partitions, node_features)` (lines 26–72): Main method — creates $G_\text{coarse}$ with super-nodes (one per partition) and weighted super-edges (inter-partition edge count). Calls `_create_partition_summary()` for each partition.
   - `_find_node_partition(node, partitions)` (lines 74–79): Helper to look up a node's partition ID.
   - `_create_partition_summary(partition_id, nodes, graph, node_features)` (lines 81–124): Computes statistics for a super-node: `num_nodes`, `num_edges`, `avg_degree`, `max_degree`, `density`, `sample_descriptions`.
   - `get_coarse_node_summary(partition_id)` (lines 126–138): Returns the summary dictionary for a partition.
   - `get_fine_nodes_for_coarse_node(partition_id)` (lines 140–152): Returns the set of original nodes belonging to a partition.
   - `get_neighboring_partitions(partition_id)` (lines 154–170): Returns partition IDs adjacent in $G_\text{coarse}$.
   - `get_coarse_graph_description()` (lines 172–198): Generates a natural-language description of $G_\text{coarse}$ suitable for LLM prompts. This is the critical bridge to the Reasoning Layer.
   - `find_relevant_partitions(query_keywords, node_features, top_k)` (lines 200–243): Keyword-based relevance scoring to select the top-$k$ partitions for a query. Uses occurrence counting in node descriptions.
   - `get_subgraph_for_partitions(graph, partition_ids, include_neighbors)` (lines 245–282): Extracts a subgraph containing specified partitions and optionally their neighbors.

**Math to include**:
- Super-node definition: $v^*_i = P_i$ (each partition becomes one node in $G_\text{coarse}$).
- Super-edge weight: $w(v^*_i, v^*_j) = |E(P_i, P_j)|$ — the number of cross-partition edges.
- Partition density: $\rho_i = \frac{2 |E(P_i)|}{|P_i| (|P_i| - 1)}$.
- Relevance score: $\text{score}(P_i, Q) = \sum_{n \in P_i} \mathbb{1}[\text{keyword}(Q) \in \text{desc}(n)]$.
- Reference `@docs/graph_partitioning_theory.md` for the coarsening phase of the multilevel paradigm.

**Paper references to cite**:
- Karypis, G., & Kumar, V. (1998). METIS paper (coarsening phase).
- Hendrickson, B., & Leland, R. (1995). "A Multilevel Algorithm for Partitioning Graphs." Supercomputing.

---

### File 3 — `docs/methodology/scalability_layer/layer_integration.md`

**Primary sources**: `src/graph_processing/partitioner.py`, `src/graph_processing/coarsener.py`, `src/agents/orchestrator.py` (lines 57–92, 124–154), `src/graph_processing/__init__.py`

This file documents **how the two Scalability Layer components connect to each other and to the downstream layers**. Follow the same pattern as `docs/methodology/neural_layer/engine_integration.md`. Specifically:

1. **Architecture Overview** — A Mermaid diagram showing the Scalability Layer within the four-layer stack, with data flows to/from:
   - **Input**: Raw knowledge graph $G = (V, E)$ and query $Q$
   - **Internal**: Partitioner → Coarsener pipeline
   - **Downstream to Neural Layer**: Partitioned adjacency matrices $A$ → RRWP Encoding & Structural Similarity
   - **Downstream to Reasoning Layer**: $G_\text{coarse}$ → Stage 1, partition metadata → Stage 2

2. **Connection A: Partitioner → Coarsener** — The partitions dictionary `Dict[int, Set[int]]` flows from `partition_graph()` to `coarsen_graph()`. Document exact code locations where this handoff occurs in the orchestrator (`src/agents/orchestrator.py`, lines 79–92).

3. **Connection B: Partitioner → Neural Layer** — The partitioned adjacency matrices are consumed by:
   - `compute_rrwp_encoding()` in `src/graph_processing/graph_transformer.py` (line 46)
   - `compute_structural_similarity_matrix()` in `src/attention/graph_aware_attention.py` (line 233)
   Document how `get_partition_subgraph()` produces the subgraphs that are converted to adjacency matrices via `nx.to_numpy_array()`.

4. **Connection C: Coarsener → Reasoning Layer (Stage 1)** — The coarse graph description (`get_coarse_graph_description()`) feeds into `_create_coarse_reasoning_prompt()` in the orchestrator (lines 216–233). Document the text formatting and what information the LLM receives.

5. **Connection D: Coarsener → Reasoning Layer (Stage 2)** — After Stage 1 selects partitions, `find_relevant_partitions()` and `get_subgraph_for_partitions()` extract the fine-grained subgraphs for Stage 2. Document how selected partition IDs flow from the LLM response back into the Coarsener's methods.

6. **Dimension / Type Alignment Map** — A table showing every data exchange with types, shapes, and constraints:
   - `partitions: Dict[int, Set[int]]` — partition ID → node set
   - `G_coarse: nx.Graph` — coarse graph with attributes
   - `adj_matrix: torch.Tensor (N, N)` or `np.ndarray (N, N)`
   - `partition_stats: Dict[str, Any]`

7. **Shared Configuration Surface** — Which `Config` fields are shared between the two components and how they affect the pipeline: `max_nodes_per_partition`, `max_edges_per_partition`, `num_partitions`.

8. **Orchestrator Integration** — Show how `HierarchicalReasoningOrchestrator.__init__()` (lines 57–60) and `setup()` (lines 79–92) wire the two components together. Include the actual code flow with line references.

9. **Cross-References** — Link to:
   - Both component docs in this folder
   - Neural Layer engine docs (`../neural_layer/`)
   - Reasoning Layer docs (`../reasoning_layer/`)
   - `@docs/graph_partitioning_theory.md`
   - `@docs/component_guide.md`
   - `@docs/architecture_diagram.md`
   - `@README.md`

---

## Style & Quality Checklist

Before finalizing each file, verify:

- [ ] Every class and function in the primary source file is documented.
- [ ] Every instance variable, data structure, and stored state is mentioned with its line number.
- [ ] All math uses LaTeX and every symbol is defined with its code counterpart.
- [ ] Underscores in LaTeX `\text{}` blocks are split: use `\text{before}\_\text{after}`, NOT `\text{before\_after}`.
- [ ] Mermaid diagrams do NOT contain underscore characters in node labels.
- [ ] "Plain English Input/Output" includes types/shapes **and** a human-readable example.
- [ ] "Python Perspective" snippets are self-contained and annotated with types/shapes.
- [ ] Internal method walkthroughs reference exact line ranges.
- [ ] Helper functions from other files (`src/utils.py`, etc.) are cross-referenced.
- [ ] Configuration fields from `configs/model_config.yaml` are listed with defaults.
- [ ] Cross-references link to `@README.md`, `@docs/component_guide.md`, `@docs/architecture_diagram.md`, `@docs/graph_partitioning_theory.md`, and the other component docs.
- [ ] The integration doc covers every data flow connection with producer/consumer code locations.

---

## Additional Information Sources

When writing these documents, consult the following for context, theory, and validation details:

- **`@README.md`** — Project overview, architecture summary, performance targets, and paper references.
- **`@docs/README.md`** — Documentation index and navigation guide.
- **`@docs/component_guide.md`** — End-to-end component guide with all layers described. Particularly:
  - Component 2.1: Graph Partitioner (lines 90–137)
  - Component 2.2: Graph Coarsener (lines 141–174)
- **`@docs/architecture_diagram.md`** — Mermaid diagrams and component interaction matrix showing the data flow from Scalability Layer to Neural and Reasoning layers.
- **`@docs/graph_partitioning_theory.md`** — In-depth theoretical background covering:
  - METIS algorithm (3-phase multilevel approach)
  - Spectral graph partitioning (Laplacian, Fiedler vector)
  - Balance and quality metrics
  - Complexity analysis
- **`@docs/hierarchical_reasoning_theory.md`** — Explains how the coarse graph feeds into Stage 1 reasoning (relevant for the integration doc).
- **`@docs/workflow.mermaid`** — Visual workflow of the entire pipeline.
- **`@docs/methodology/neural_layer/`** — Existing Neural Layer docs to reference for downstream connections.
- **`@tests/test_partitioning.py`** — Unit tests for `GraphPartitioner` (118 lines) showing expected behavior and edge cases.
