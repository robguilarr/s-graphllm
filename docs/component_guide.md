# S-GraphLLM: End-to-End Component Guide

## Overview

This document provides a comprehensive step-by-step guide to every component in the S-GraphLLM architecture. Each component is explained with its theoretical foundation, mechanism of action, and source attribution. This guide is designed for bachelor-level computer scientists seeking to understand the system's inner workings.

---

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Layer 1: Input Layer](#layer-1-input-layer)
3. [Layer 2: Scalability Layer](#layer-2-scalability-layer)
4. [Layer 3: GraphLLM Neural Components](#layer-3-graphllm-neural-components)
5. [Layer 4: Hierarchical Reasoning Layer](#layer-4-hierarchical-reasoning-layer)
6. [Layer 5: Output Layer](#layer-5-output-layer)
7. [Data Flow Summary](#data-flow-summary)
8. [References](#references)

---

## System Architecture Overview

S-GraphLLM is a **hybrid architecture** that combines:

1. **Scalability techniques** (graph partitioning, coarsening) for handling billion-node graphs
2. **Neural graph encoding** (GraphLLM components) for learning graph representations
3. **Hierarchical reasoning** (coarse-to-fine strategy) for efficient query answering

The system processes a query over a large knowledge graph through five logical layers, progressively refining the search space and generating accurate answers.

---

## Layer 1: Input Layer

### Component 1.1: Knowledge Graph Input

**Technical Definition**:  
The input layer receives a **knowledge graph** $G = (V, E, X)$ where:
- $V$ = set of nodes (entities)
- $E$ = set of edges (relationships)
- $X$ = node/edge attributes (text descriptions, features)

**Theoretical Basis**:  
Knowledge graphs are a well-established formalism in AI and databases. The foundational work includes:
- **Bollacker, K., et al. (2008)**. "Freebase: A Collaboratively Created Graph Database for Structuring Human Knowledge." *SIGMOD*.
  - [Link](https://dl.acm.org/doi/10.1145/1376616.1376746)

**Mechanism of Action**:
1. Graph is loaded from storage (e.g., RDF triples, adjacency lists, or graph databases)
2. Node attributes are extracted (text descriptions, entity types)
3. Edge attributes are extracted (relationship types, weights)

**Input Format**:
```
G = {
  nodes: [(id, text_description, type), ...],
  edges: [(source_id, target_id, relation_type), ...]
}
```

**Output**: Graph object $G$ passed to Scalability Layer

---

### Component 1.2: Query Input

**Technical Definition**:  
A natural language query $Q$ that requires reasoning over the knowledge graph to produce an answer.

**Theoretical Basis**:  
Query understanding builds on natural language processing (NLP) foundations:
- **Vaswani, A., et al. (2017)**. "Attention Is All You Need." *NeurIPS*.
  - [Link](https://arxiv.org/abs/1706.03762)

**Mechanism of Action**:
1. Query is tokenized and encoded
2. Key entities and relationships are extracted
3. Query embedding is generated for similarity matching

**Input Format**: Natural language string  
**Output**: Query representation $(Q, Q_{emb}, \text{keywords})$

---

## Layer 2: Scalability Layer

This layer enables processing of graphs with billions of nodes by dividing them into manageable subgraphs.

### Component 2.1: Graph Partitioner

**Technical Definition**:  
The Graph Partitioner divides a large graph $G$ into $k$ disjoint subgraphs (partitions) $\{P_1, P_2, \ldots, P_k\}$ while minimizing the number of edges cut between partitions.

**Theoretical Basis**:  
Based on the **METIS algorithm** and **spectral graph theory**:
- **Karypis, G., & Kumar, V. (1998)**. "A Fast and High Quality Multilevel Scheme for Partitioning Irregular Graphs." *SIAM Journal on Scientific Computing*, 20(1), 359-392.
  - [Link](https://www.cs.utexas.edu/~pingali/CS395T/2009fa/papers/metis.pdf)
- **Fiedler, M. (1973)**. "Algebraic Connectivity of Graphs." *Czechoslovak Mathematical Journal*, 23(2), 298-305.

**Mechanism of Action**:

The partitioner implements three strategies:

#### Strategy A: METIS-like / Spectral Clustering

1. **Compute Graph Laplacian**:
   $$L = D - A$$
   where $D$ is the degree matrix and $A$ is the adjacency matrix.

2. **Compute Eigenvectors**: Find the $k$ smallest eigenvectors of the normalized Laplacian:
   $$\mathcal{L} = I - D^{-1/2} A D^{-1/2}$$

3. **Cluster Vertices**: Apply k-means clustering on the eigenvector matrix to assign nodes to partitions.

**Why it works**: The eigenvectors of the Laplacian encode the graph's connectivity structure. Vertices with similar eigenvector values are well-connected and should be in the same partition.

#### Strategy B: Community Detection (Louvain)

1. **Initialize**: Each node starts in its own community
2. **Local Optimization**: Move nodes to neighboring communities to maximize modularity:
   $$Q = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$
3. **Aggregation**: Merge communities into super-nodes and repeat

**Source**: Blondel, V. D., et al. (2008). "Fast unfolding of communities in large networks." *Journal of Statistical Mechanics*.
- [Link](https://arxiv.org/abs/0803.0476)

#### Strategy C: Balanced Partitioning

Simple greedy algorithm that assigns nodes to partitions while maintaining size balance.

**Architectural Addition**: This is a custom heuristic for cases where partition balance is more important than edge cut minimization.

**Input**: Graph $G$, number of partitions $k$  
**Output**: Partition assignment $\{P_1, P_2, \ldots, P_k\}$ where $\bigcup P_i = V$ and $P_i \cap P_j = \emptyset$

**Implementation**: `src/graph_processing/partitioner.py`

---

### Component 2.2: Graph Coarsener

**Technical Definition**:  
The Graph Coarsener creates a **summarized graph** $G_{coarse}$ where each partition becomes a single "super-node" and inter-partition edges become "super-edges."

**Theoretical Basis**:  
Coarsening is a key phase in multilevel graph algorithms:
- **Karypis, G., & Kumar, V. (1998)**. METIS paper (same as above)
- **Hendrickson, B., & Leland, R. (1995)**. "A Multilevel Algorithm for Partitioning Graphs." *Supercomputing*.
  - [Link](https://dl.acm.org/doi/10.1145/224170.224228)

**Mechanism of Action**:

1. **Super-Node Creation**: For each partition $P_i$, create a super-node $v_i^*$:
   $$v_i^* = \text{Aggregate}(\{v : v \in P_i\})$$

2. **Feature Aggregation**: Aggregate node features within each partition:
   - **Mean Pooling**: $f_{v_i^*} = \frac{1}{|P_i|} \sum_{v \in P_i} f_v$
   - **Max Pooling**: $f_{v_i^*} = \max_{v \in P_i} f_v$
   - **Attention-Weighted**: $f_{v_i^*} = \sum_{v \in P_i} \alpha_v f_v$ where $\alpha$ is learned

3. **Super-Edge Creation**: Create edges between super-nodes based on inter-partition edges:
   $$E_{coarse} = \{(v_i^*, v_j^*) : \exists (u, v) \in E, u \in P_i, v \in P_j, i \neq j\}$$

4. **Edge Weight Aggregation**: Weight of super-edge = count or sum of original edge weights:
   $$w(v_i^*, v_j^*) = |\{(u, v) \in E : u \in P_i, v \in P_j\}|$$

**Why it works**: The coarse graph preserves the high-level structure of the original graph while being orders of magnitude smaller. This enables efficient coarse-grained reasoning.

**Input**: Graph $G$, Partitions $\{P_1, \ldots, P_k\}$  
**Output**: Coarse graph $G_{coarse} = (V_{coarse}, E_{coarse})$

**Implementation**: `src/graph_processing/coarsener.py`

---

## Layer 3: GraphLLM Neural Components

These components implement the neural graph encoding from the GraphLLM paper. They learn representations that capture both **node semantics** and **graph structure**.

### Component 3.1: Node Encoder

**Technical Definition**:  
The Node Encoder is a **transformer encoder** that processes textual descriptions of nodes to extract semantic information.

**Theoretical Basis**:  
From the GraphLLM paper, Section 3.2:
- **Chai, Z., et al. (2025)**. "GraphLLM: Boosting Graph Reasoning Ability of Large Language Model." *IEEE Transactions on Big Data*.
  - [Link](https://arxiv.org/abs/2310.05845)

**Mechanism of Action**:

1. **Input**: Node text description $d_i$ (e.g., "Albert Einstein was a physicist...")

2. **Tokenization**: Convert text to token embeddings using a pre-trained tokenizer

3. **Down-Projection**: Project embeddings to hidden dimension:
   $$x_i = W_D \cdot \text{Embed}(d_i)$$

4. **Transformer Encoding**: Apply transformer encoder layers:
   $$c_i = \text{TransformerEncoder}(x_i)$$

5. **Context Vector**: Pool over sequence length to get a single context vector:
   $$c_i = \text{MeanPool}(\text{TransformerEncoder}(x_i))$$

**Mathematical Formula** (Equation 5a from paper):
$$c_i = \text{TransformerEncoder}(d_i, W_D)$$

**Why it works**: The transformer encoder captures semantic relationships within the text description, producing a rich representation of the node's meaning.

**Input**: Node text descriptions $\{d_1, d_2, \ldots, d_n\}$  
**Output**: Context vectors $\{c_1, c_2, \ldots, c_n\}$

**Implementation**: `src/agents/node_encoder_decoder.py` → `NodeEncoder` class

---

### Component 3.2: Node Decoder

**Technical Definition**:  
The Node Decoder is a **transformer decoder** that produces final node representations from context vectors using learnable query embeddings.

**Theoretical Basis**:  
From the GraphLLM paper, Section 3.2:
- **Chai, Z., et al. (2025)**. GraphLLM paper (same as above)

**Mechanism of Action**:

1. **Learnable Query**: Initialize a learnable query embedding $Q$

2. **Cross-Attention**: Decoder attends to the context vector:
   $$H_i = \text{TransformerDecoder}(Q, c_i)$$

3. **Output Projection**: Project to final node representation dimension

**Mathematical Formula** (Equation 5b from paper):
$$H_i = \text{TransformerDecoder}(Q, c_i)$$

**Why it works**: The decoder uses cross-attention to extract the most relevant information from the context vector, guided by the learnable query. This produces a compact, task-relevant node representation.

**Input**: Context vectors $\{c_1, c_2, \ldots, c_n\}$  
**Output**: Node representations $\{H_1, H_2, \ldots, H_n\}$

**Implementation**: `src/agents/node_encoder_decoder.py` → `NodeDecoder` class

---

### Component 3.3: RRWP Encoding (Relative Random Walk Positional Encoding)

**Technical Definition**:  
RRWP encodes the **structural position** of each node pair by computing random walk probabilities at multiple steps.

**Theoretical Basis**:  
From the GraphLLM paper, Section 3.3, and the GRIT paper:
- **Chai, Z., et al. (2025)**. GraphLLM paper
- **Ma, L., et al. (2023)**. "Graph Inductive Biases in Transformers without Message Passing." *ICML*.
  - [Link](https://arxiv.org/abs/2305.17589)

**Mechanism of Action**:

1. **Compute Random Walk Matrix**:
   $$M = D^{-1} A$$
   where $D$ is the degree matrix and $A$ is the adjacency matrix.

2. **Compute Powers of M**: For steps $0, 1, 2, \ldots, C-1$:
   $$M^0 = I, \quad M^1 = M, \quad M^2 = M \cdot M, \ldots$$

3. **Stack into RRWP Tensor**:
   $$R_{i,j} = [I_{i,j}, M_{i,j}, M^2_{i,j}, \ldots, M^{C-1}_{i,j}]$$

**Mathematical Formula** (Equation 6 from paper):
$$R_{i,j} = [I_{i,j}, M_{i,j}, M^2_{i,j}, \ldots, M^{C-1}_{i,j}]$$

**Why it works**: The random walk probability $M^k_{i,j}$ represents the probability of reaching node $j$ from node $i$ in exactly $k$ steps. This captures multi-scale structural relationships:
- $M^1$: Direct neighbors
- $M^2$: 2-hop neighbors
- $M^k$: k-hop connectivity patterns

**Input**: Adjacency matrix $A$, max steps $C$  
**Output**: RRWP tensor $R \in \mathbb{R}^{n \times n \times C}$

**Implementation**: `src/graph_processing/graph_transformer.py` → `compute_rrwp_encoding()`

---

### Component 3.4: Graph Transformer (GRIT)

**Technical Definition**:  
The Graph Transformer applies **self-attention over graph nodes** with edge features derived from RRWP encoding.

**Theoretical Basis**:  
Based on the GRIT (Graph Inductive Bias Transformer) architecture:
- **Ma, L., et al. (2023)**. "Graph Inductive Biases in Transformers without Message Passing." *ICML*.
  - [Link](https://arxiv.org/abs/2305.17589)
- **Chai, Z., et al. (2025)**. GraphLLM paper, Section 3.3

**Mechanism of Action**:

1. **Edge Feature Encoding**: Transform RRWP to edge features:
   $$e_{ij} = \Phi(R_{i,j})$$
   where $\Phi$ is a learnable MLP.

2. **Query, Key, Value Projection**:
   $$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

3. **Sparse Attention with Edge Bias**: Compute attention only over edges:
   $$\alpha_{ij} = \text{softmax}_j\left(\frac{Q_i \cdot K_j^T}{\sqrt{d_k}} + e_{ij}\right)$$

4. **Message Passing**: Aggregate neighbor information:
   $$X' = \sum_{j \in \mathcal{N}(i)} \alpha_{ij} V_j$$

5. **Edge Update**: Update edge features:
   $$e'_{ij} = e_{ij} + \text{MLP}(\alpha_{ij})$$

**Why it works**: Unlike standard transformers that attend to all tokens, the graph transformer attends only along graph edges. The RRWP-derived edge features inject structural information into the attention mechanism.

**Input**: Node features $X$, Edge index, RRWP encoding $R$  
**Output**: Updated node features $X'$, Updated edge features $e'$

**Implementation**: `src/graph_processing/graph_transformer.py` → `GraphTransformer` class

---

### Component 3.5: Graph-Aware Attention

**Technical Definition**:  
Graph-Aware Attention modulates the LLM's attention weights based on **structural similarity** between nodes in the graph.

**Theoretical Basis**:  
**Architectural Addition**: This is a novel contribution of S-GraphLLM, inspired by:
- **Veličković, P., et al. (2018)**. "Graph Attention Networks." *ICLR*.
  - [Link](https://arxiv.org/abs/1710.10903)
- **Chai, Z., et al. (2025)**. GraphLLM paper (general concept of graph-enhanced attention)

**Mechanism of Action**:

1. **Compute Structural Similarity**: For each node pair, compute similarity based on:
   - **Shortest Path Distance**: $S(n_i, n_j) = \frac{1}{1 + d(n_i, n_j)}$
   - **Common Neighbors**: $S(n_i, n_j) = \frac{|\mathcal{N}(n_i) \cap \mathcal{N}(n_j)|}{|\mathcal{N}(n_i) \cup \mathcal{N}(n_j)|}$
   - **Jaccard Similarity**: Similar to common neighbors

2. **Modulate Attention Weights**:
   $$\alpha'_{ij} = \frac{\exp\left(\frac{q_i^T k_j}{\sqrt{d_k}} + \beta \cdot S(n_i, n_j)\right)}{\sum_l \exp\left(\frac{q_i^T k_l}{\sqrt{d_k}} + \beta \cdot S(n_i, n_l)\right)}$$

   where $\beta$ is a learnable parameter controlling the influence of graph structure.

**Why it works**: Standard attention in LLMs treats all tokens equally. By adding a structural bias, nodes that are closer in the graph receive higher attention weights, helping the model focus on relevant entities during reasoning.

**Input**: Query/Key vectors, Structural similarity matrix $S$  
**Output**: Modulated attention weights $\alpha'$

**Implementation**: `src/attention/graph_aware_attention.py`

---

## Layer 4: Hierarchical Reasoning Layer

This layer implements the **coarse-to-fine reasoning strategy** that enables efficient reasoning over large graphs.

### Component 4.1: Hierarchical Reasoning Orchestrator

**Technical Definition**:  
The Orchestrator coordinates the three-stage reasoning process: coarse-grained selection, fine-grained reasoning, and answer synthesis.

**Theoretical Basis**:  
Based on hierarchical and coarse-to-fine reasoning:
- **Wang, G., et al. (2025)**. "Hierarchical Reasoning Model." *arXiv preprint arXiv:2506.21734*.
  - [Link](https://arxiv.org/abs/2506.21734)
- **Nguyen, B. X., et al. (2022)**. "Coarse-to-Fine Reasoning for Visual Question Answering." *CVPR Workshop*.
  - [Link](https://arxiv.org/abs/2110.02526)

**Mechanism of Action**:

The orchestrator manages state and coordinates the three stages:

```python
class HierarchicalReasoningOrchestrator:
    def reason(self, query, llm_agent):
        # Stage 1: Coarse-grained
        relevant_partitions = self.coarse_grained_reasoning(query)
        
        # Stage 2: Fine-grained
        detailed_results = self.fine_grained_reasoning(
            query, relevant_partitions, llm_agent
        )
        
        # Stage 3: Synthesis
        final_answer = self.synthesize_answer(detailed_results, llm_agent)
        
        return final_answer
```

**Implementation**: `src/agents/orchestrator.py`

---

### Component 4.2: Stage 1 - Coarse-Grained Reasoning

**Technical Definition**:  
Coarse-grained reasoning identifies **relevant partitions** by analyzing the coarse graph, avoiding the need to process the entire original graph.

**Theoretical Basis**:  
**Architectural Addition**: Novel contribution combining:
- Coarse-to-fine strategies from computer vision
- Hierarchical search from AI planning

**Mechanism of Action**:

1. **Generate Coarse Graph Context**: Create a textual representation of the coarse graph:
   ```
   Super-node 1: Contains entities [Einstein, Physics, Nobel Prize]
   Super-node 2: Contains entities [Berlin, Germany, Europe]
   Edges: Super-node 1 -- lived_in --> Super-node 2
   ```

2. **Extract Query Keywords**: Identify key entities and concepts from the query

3. **Filter Relevant Partitions**: Match keywords against partition contents:
   $$\text{Relevance}(P_i, Q) = \text{Jaccard}(\text{Keywords}(P_i), \text{Keywords}(Q))$$

4. **LLM Analysis**: Use LLM to rank partitions by relevance:
   ```
   Prompt: "Given the query '{Q}' and the following partitions, 
            rank them by relevance..."
   ```

5. **Select Top-K**: Return the $k$ most relevant partitions

**Input**: Query $Q$, Coarse graph $G_{coarse}$  
**Output**: List of relevant partition IDs $[P_{i_1}, P_{i_2}, \ldots, P_{i_k}]$

---

### Component 4.3: Stage 2 - Fine-Grained Reasoning

**Technical Definition**:  
Fine-grained reasoning performs **detailed multi-hop reasoning** within the selected partitions to find evidence and reasoning paths.

**Theoretical Basis**:  
Multi-hop reasoning in knowledge graphs:
- **Lin, X. V., et al. (2018)**. "Multi-Hop Knowledge Graph Reasoning with Reward Shaping." *EMNLP*.
  - [Link](https://arxiv.org/abs/1808.10568)
- **Ren, H., & Leskovec, J. (2020)**. "Beta Embeddings for Multi-Hop Logical Reasoning in Knowledge Graphs." *NeurIPS*.
  - [Link](https://arxiv.org/abs/2010.11465)

**Mechanism of Action**:

1. **Extract Subgraphs**: For each selected partition, extract the full subgraph:
   $$G_i = (V_i, E_i) \text{ where } V_i = P_i, E_i = \{(u,v) \in E : u,v \in P_i\}$$

2. **Apply GraphLLM Components**: Process subgraph through:
   - Node Encoder-Decoder → Node embeddings
   - Graph Transformer → Structural embeddings
   - Graph-Aware Attention → Attention weights

3. **Format Detailed Context**: Create rich textual representation:
   ```
   Entity: Albert Einstein
   Type: Person
   Relationships:
     - born_in -> Ulm (1879)
     - worked_at -> Princeton University
     - won -> Nobel Prize in Physics (1921)
   ```

4. **LLM Multi-hop Reasoning**: Prompt LLM to reason over the context:
   ```
   Prompt: "Using the following graph context, answer the question 
            by reasoning step-by-step through the relationships..."
   ```

5. **Collect Evidence**: Extract reasoning paths and supporting facts

**Input**: Query $Q$, Selected partitions, LLM agent  
**Output**: List of reasoning results with evidence

---

### Component 4.4: Stage 3 - Answer Synthesis

**Technical Definition**:  
Answer synthesis **aggregates results** from multiple partitions and generates a coherent final answer.

**Theoretical Basis**:  
**Architectural Addition**: Inspired by:
- Ensemble methods in machine learning
- Multi-document summarization in NLP

**Mechanism of Action**:

1. **Aggregate Evidence**: Combine evidence from all partitions:
   $$\text{AllEvidence} = \bigcup_{i=1}^{k} \text{Evidence}_i$$

2. **Resolve Conflicts**: Handle contradictory information:
   - Majority voting for factual claims
   - Confidence-weighted aggregation
   - LLM-based conflict resolution

3. **Generate Final Answer**: Use LLM to synthesize:
   ```
   Prompt: "Based on the following evidence from multiple sources,
            provide a comprehensive answer to: '{Q}'
            
            Evidence:
            1. [From partition 1]: ...
            2. [From partition 2]: ...
            
            Synthesize a final answer."
   ```

**Input**: List of reasoning results, LLM agent  
**Output**: Final answer string

---

### Component 4.5: LLM Agent

**Technical Definition**:  
The LLM Agent provides an interface to large language models for reasoning, analysis, and generation tasks.

**Theoretical Basis**:  
Based on LLM prompting and in-context learning:
- **Brown, T., et al. (2020)**. "Language Models are Few-Shot Learners." *NeurIPS*.
  - [Link](https://arxiv.org/abs/2005.14165)
- **Wei, J., et al. (2022)**. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *NeurIPS*.
  - [Link](https://arxiv.org/abs/2201.11903)

**Mechanism of Action**:

1. **Prompt Construction**: Build structured prompts with:
   - System instructions
   - Context (graph information)
   - Query
   - Output format instructions

2. **API Call**: Send prompt to LLM API (OpenAI, etc.)

3. **Response Parsing**: Extract structured information from response

4. **Error Handling**: Retry on failures, handle rate limits

**Supported Models**:
- GPT-4.1-mini (default)
- GPT-4.1-nano
- Gemini-2.5-flash

**Input**: Prompt string, configuration  
**Output**: LLM response string

**Implementation**: `src/agents/llm_agent.py`

---

## Layer 5: Output Layer

### Component 5.1: Final Answer

**Technical Definition**:  
The output layer formats and delivers the final answer to the user.

**Mechanism of Action**:

1. **Format Answer**: Structure the response with:
   - Main answer
   - Supporting evidence
   - Confidence score
   - Reasoning trace (optional)

2. **Validate Answer**: Check for:
   - Completeness
   - Consistency
   - Factual grounding

**Output Format**:
```json
{
  "answer": "Albert Einstein won the Nobel Prize in Physics in 1921.",
  "confidence": 0.95,
  "evidence": [
    {"source": "partition_3", "fact": "Einstein won Nobel Prize 1921"},
    {"source": "partition_7", "fact": "Nobel Prize in Physics awarded to Einstein"}
  ],
  "reasoning_trace": ["Step 1: ...", "Step 2: ..."]
}
```

---

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INPUT LAYER                                    │
│  Knowledge Graph G = (V, E, X)  +  Natural Language Query Q              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        SCALABILITY LAYER                                 │
│  ┌─────────────────┐      ┌─────────────────┐                           │
│  │ Graph           │      │ Graph           │                           │
│  │ Partitioner     │ ──▶  │ Coarsener       │                           │
│  │                 │      │                 │                           │
│  │ Input: G        │      │ Input: G, P     │                           │
│  │ Output: P₁..Pₖ  │      │ Output: G_coarse│                           │
│  └─────────────────┘      └─────────────────┘                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌─────────────────────────────────┐ ┌─────────────────────────────────────┐
│   GRAPHLLM NEURAL COMPONENTS    │ │   HIERARCHICAL REASONING LAYER      │
│                                 │ │                                     │
│  ┌───────────┐  ┌───────────┐   │ │  ┌─────────────────────────────┐   │
│  │ Node      │  │ Node      │   │ │  │ Stage 1: Coarse-Grained     │   │
│  │ Encoder   │─▶│ Decoder   │   │ │  │ - Analyze G_coarse          │   │
│  └───────────┘  └───────────┘   │ │  │ - Select relevant partitions│   │
│        │              │         │ │  └─────────────────────────────┘   │
│        ▼              ▼         │ │              │                     │
│  Node Embeddings  H₁..Hₙ        │ │              ▼                     │
│                                 │ │  ┌─────────────────────────────┐   │
│  ┌───────────┐  ┌───────────┐   │ │  │ Stage 2: Fine-Grained       │   │
│  │ RRWP      │─▶│ Graph     │   │ │  │ - Extract subgraphs         │   │
│  │ Encoding  │  │ Transformer│  │ │  │ - Multi-hop reasoning       │◀──┼───┐
│  └───────────┘  └───────────┘   │ │  └─────────────────────────────┘   │   │
│        │              │         │ │              │                     │   │
│        ▼              ▼         │ │              ▼                     │   │
│  Structural Embeddings          │ │  ┌─────────────────────────────┐   │   │
│                                 │ │  │ Stage 3: Answer Synthesis   │   │   │
│  ┌───────────────────────────┐  │ │  │ - Aggregate evidence        │   │   │
│  │ Graph-Aware Attention     │──┼─┼─▶│ - Generate final answer     │   │   │
│  │ - Structural similarity   │  │ │  └─────────────────────────────┘   │   │
│  │ - Attention modulation    │  │ │                                     │   │
│  └───────────────────────────┘  │ └─────────────────────────────────────┘   │
│                                 │                 │                         │
│  Embeddings ─────────────────────────────────────┘                         │
└─────────────────────────────────┘                                           │
                                                    │                         │
                                                    ▼                         │
┌─────────────────────────────────────────────────────────────────────────────┘
│                           OUTPUT LAYER                                   │
│  Final Answer + Evidence + Confidence + Reasoning Trace                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## References

### Core Papers

1. **Chai, Z., et al. (2025)**. "GraphLLM: Boosting Graph Reasoning Ability of Large Language Model." *IEEE Transactions on Big Data*.
   - [https://arxiv.org/abs/2310.05845](https://arxiv.org/abs/2310.05845)

2. **Karypis, G., & Kumar, V. (1998)**. "A Fast and High Quality Multilevel Scheme for Partitioning Irregular Graphs." *SIAM Journal on Scientific Computing*.
   - [https://www.cs.utexas.edu/~pingali/CS395T/2009fa/papers/metis.pdf](https://www.cs.utexas.edu/~pingali/CS395T/2009fa/papers/metis.pdf)

3. **Ma, L., et al. (2023)**. "Graph Inductive Biases in Transformers without Message Passing." *ICML*.
   - [https://arxiv.org/abs/2305.17589](https://arxiv.org/abs/2305.17589)

4. **Wang, G., et al. (2025)**. "Hierarchical Reasoning Model." *arXiv preprint*.
   - [https://arxiv.org/abs/2506.21734](https://arxiv.org/abs/2506.21734)

### Supporting Papers

5. **Vaswani, A., et al. (2017)**. "Attention Is All You Need." *NeurIPS*.
   - [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

6. **Veličković, P., et al. (2018)**. "Graph Attention Networks." *ICLR*.
   - [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)

7. **Brown, T., et al. (2020)**. "Language Models are Few-Shot Learners." *NeurIPS*.
   - [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)

8. **Wei, J., et al. (2022)**. "Chain-of-Thought Prompting." *NeurIPS*.
   - [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)

### Additional Resources

- [Graph Partitioning Theory](graph_partitioning_theory.md)
- [Hierarchical Reasoning Theory](hierarchical_reasoning_theory.md)

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**Maintainer**: S-GraphLLM Team
