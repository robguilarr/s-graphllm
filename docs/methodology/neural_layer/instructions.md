# Instructions: Create Detailed Neural Layer Documentation

## Objective

Create **3 separate Markdown files** — one for each of the Neural Layer "engines" in S-GraphLLM — plus **1 integration file** that connects the dots between them and shows how the Neural Layer consumes outputs from the Scalability Layer and delivers its results to the Reasoning Layer. Each file must follow the template structure below while being **significantly more detailed** than the template: every class, method, learnable parameter, and helper function must be cited with its **exact file path and line numbers** in the codebase.

Use `@README.md` and `@docs/` (especially `@docs/component_guide.md`, `@docs/architecture_diagram.md`, and `@docs/hierarchical_reasoning_theory.md`) as additional information sources to enrich descriptions, cross-reference theoretical foundations, and ensure consistency with the existing documentation.

---

## Files to Create

| # | Filename | Engine | Primary Source File |
|---|----------|--------|---------------------|
| 1 | `docs/methodology/neural_layer/engine_node_encoder_decoder.md` | **Node Encoder-Decoder (The Semantic Engine)** | `src/agents/node_encoder_decoder.py` |
| 2 | `docs/methodology/neural_layer/engine_graph_transformer.md` | **Graph Transformer / GRIT (The Structural Engine)** | `src/graph_processing/graph_transformer.py` |
| 3 | `docs/methodology/neural_layer/engine_graph_aware_attention.md` | **Graph-Aware Attention (The Focus Engine)** | `src/attention/graph_aware_attention.py` |
| 4 | `docs/methodology/neural_layer/engine_integration.md` | **Neural Layer Integration** | Multiple sources (see § Integration File Guidance) |

> **Note**: All files go under the `docs/methodology/neural_layer/` directory.

---

## Context: The Neural Layer in S-GraphLLM

The Neural Layer is **Layer 3** in the S-GraphLLM four-layer stack (**Scalability → Neural → Reasoning → Output**). It is the representational core — its job is to transform raw text descriptions and graph topology into dense, information-rich embeddings that the Reasoning Layer can consume for multi-hop question answering. Without this layer, the LLM would have no way to understand graph structure or encode node semantics into a form suitable for reasoning.

The three engines operate in a coordinated, parallel-then-convergent pipeline:

1. **Semantic Engine (Node Encoder-Decoder)** reads textual descriptions for each node and distills them into fixed-dimensional node representations $H_i$ using a Transformer encoder-decoder architecture (GraphLLM paper, Equation 5).
2. **Structural Engine (Graph Transformer / GRIT)** takes the node representations $H_i$ and the adjacency matrix $A$ (from the Scalability Layer), computes Relative Random Walk Positional Encoding (RRWP), and refines node features through sparse graph attention layers (GraphLLM paper, Equation 6 + GRIT architecture).
3. **Focus Engine (Graph-Aware Attention)** computes a structural similarity matrix $S(n_i, n_j)$ from the same adjacency matrix $A$ and uses it to modulate the LLM's attention weights during reasoning, boosting attention between structurally related nodes (S-GraphLLM novel contribution).

The Neural Layer consumes outputs from the **Scalability Layer**:
- **Partitioned adjacency matrices** $A$ → Structural Engine (RRWP) and Focus Engine (structural similarity)
- **Node text descriptions** → Semantic Engine

The Neural Layer delivers outputs to the **Reasoning Layer** (Stage 2):
- **Node embeddings** $H_i$ → Semantic representations for fine-grained context (Connection C)
- **Structural embeddings** $x'$ → Topology-enriched node features (Connection D)
- **Attention weights** $\alpha'$ → Graph-modulated LLM attention (Connection E)

---

## Template & Format

Each engine file **must** contain the following sections. Expand every section with the level of detail described in the annotations.

````markdown
# [Engine Name] — [Subtitle]

> One-paragraph executive summary: what this engine does in the overall S-GraphLLM pipeline,
> why it exists, and which paper equations it implements.
> Reference the relevant section of `@README.md` and `@docs/component_guide.md`.

---

## Architecture Overview

A short prose description of how the engine fits into the four-layer architecture
(Scalability → Neural → Reasoning → Output). Include a small Mermaid diagram if helpful.

---

## Components Breakdown

For **every class and standalone function** in the primary source file, create a subsection:

### N. ClassName / function_name — (Role in one phrase)

* **Location**: `src/path/to/file.py`, lines XX–YY
* **Purpose**: 2–3 sentence description.
* **Paper Reference**: Equation number and/or section from the GraphLLM / GRIT paper.

#### The Math

Write out **every formula** the component implements. Use LaTeX notation.
For each formula:
  - State the equation.
  - Define every symbol (e.g., "$W_D$ — the learnable down-projection matrix, initialized in `__init__` at line XX").
  - Map each symbol back to the exact variable or `nn.Module` in the code.

**Important**: Do NOT place underscore characters inside `\text{}` blocks in LaTeX.
Use `\text{before}\_\text{after}` to split identifiers with underscores,
keeping the escaped underscore in math mode. Example:
`\text{output}\_\text{dim}` instead of `\text{output\_dim}`.

#### Plain English Input / Output

* **Input**: Describe shape, dtype, and a concrete human-readable example
  (e.g., "A list of text strings like `['Albert Einstein was a physicist...', 'Ulm is a city...']`").
* **Output**: Same level of detail.
* **Side Effects / State**: Note any buffers, caches, or `register_buffer` calls.

#### Python Perspective

Provide a **runnable-style** code snippet showing the component in isolation.
Include tensor shapes as comments on every line.

```python
# Example:
import torch
from src.agents.node_encoder_decoder import NodeEncoderDecoder

# Input: (batch_size, sequence_length, embedding_dim)
node_embeddings = torch.randn(4, 32, 768)

enc_dec = NodeEncoderDecoder(input_dim=768, hidden_dim=512, output_dim=256)
node_repr = enc_dec(node_embeddings)
# node_repr.shape == torch.Size([4, 256])              # (batch_size, output_dim)
```

#### Internal Method Walkthrough

For each method inside the class:
  - **Method name** (line range)
  - Step-by-step description of what it does.
  - Which other classes/functions it calls (with file + line references).

---

## Helper / Utility Functions

List any helper functions used by this engine from `src/utils.py` or other modules,
with file paths, line ranges, and a brief description of their role.

---

## Configuration

Describe every relevant field from `configs/model_config.yaml` and any dataclass/config
object (e.g., `GraphAwareAttentionConfig`). State default values and how they affect behavior.

---

## Cross-References

- Link to related docs in `@docs/` (e.g., `graph_partitioning_theory.md`, `hierarchical_reasoning_theory.md`).
- Link to the other two engine docs in this folder.
- Link to the integration doc (`engine_integration.md`).
- Reference `@README.md` sections on architecture and validation.
- Link to the Scalability Layer docs that produce this layer's inputs (`../scalability_layer/`).
- Link to the Reasoning Layer docs that consume this layer's outputs (`../reasoning_layer/`).
````

---

## Detailed Per-File Guidance

### File 1 — `docs/methodology/neural_layer/engine_node_encoder_decoder.md`

**Primary source**: `src/agents/node_encoder_decoder.py` (≈ 340 lines)

Document the following components in order:

1. **`NodeEncoder`** (lines 19–121)
   - The `down_projection` linear layer (`W_D`), line 56.
   - The `transformer_encoder` (Transformer encoder stack), lines 59–69.
   - The `layer_norm`, line 72.
   - The `forward()` method (lines 76–121): down-projection → transformer encoding → masked mean pooling → layer norm.
   - Explain the attention mask handling (lines 96–100, 108–113) and both pooling paths (masked vs. simple mean).

2. **`NodeDecoder`** (lines 124–221)
   - The learnable `query_embedding` (`Q`), line 161.
   - The `transformer_decoder` stack, lines 164–175.
   - The `output_projection` linear layer, line 178.
   - The `forward()` method (lines 185–221): query expansion → cross-attention decoding → projection → layer norm.

3. **`NodeEncoderDecoder`** (lines 224–340)
   - Combines encoder + decoder.
   - `forward()` (lines 274–295): encode → decode pipeline.
   - `encode_batch()` (lines 297–340): tokenization → embedding → forward. Note the placeholder embedding on lines 332–336.

**Math to include** (from GraphLLM paper Section 3.2):
- Encoder: $c_i = \text{TransformerEncoder}(d_i, W_D)$ — Equation 5a.
- Decoder: $H_i = \text{TransformerDecoder}(Q, c_i)$ — Equation 5b.
- Define $d_i$ (description text embeddings), $W_D$ (down-projection), $c_i$ (context vector), $Q$ (learnable query), $H_i$ (final node representation).

**Paper references to cite**:
- Chai, Z., et al. (2025). *GraphLLM: Boosting Graph Reasoning Ability of Large Language Model.* IEEE Transactions on Big Data. Section 3.2 — Node Encoder-Decoder.

---

### File 2 — `docs/methodology/neural_layer/engine_graph_transformer.md`

**Primary source**: `src/graph_processing/graph_transformer.py` (≈ 400 lines)

Document the following components in order:

1. **`pyg_softmax()`** (lines 21–43)
   - Sparse softmax using `scatter_max` and `scatter_add`.
   - Numerical stability via max subtraction.

2. **`compute_rrwp_encoding()`** (lines 46–87)
   - Computes $R_{i,j} = [I_{i,j},\; M_{i,j},\; M^2_{i,j},\; \ldots,\; M^{C-1}_{i,j}]$ — Equation 6.
   - Degree matrix inversion (line 68–69), random walk matrix $M = D^{-1}A$ (line 72), iterative powers (lines 76–82).

3. **`MultiHeadGraphAttention`** (lines 90–255)
   - Q/K/V projections (`wq`, `wk`, `wv`), lines 128–130.
   - Edge bias/weight projections (`w_eb`, `w_ew`), lines 133–134.
   - Output projections (`wo`, `weo`), lines 137–138.
   - Learnable `Aw` parameter, line 141.
   - `propagate_attention()` (lines 150–215): edge bias computation, signed square root, Einstein summation, sparse softmax, scatter aggregation.
   - `forward()` (lines 217–255): project → propagate → reshape → residual → layer norm.

4. **`GraphTransformerLayer`** (lines 258–317)
   - Wraps `MultiHeadGraphAttention` + node FFN + edge FFN.
   - Residual connections and layer norms for both node and edge streams.

5. **`GraphTransformer`** (lines 320–400)
   - Stacks multiple `GraphTransformerLayer`s.
   - `rrwp_projection` linear layer (line 365).
   - `forward()` (lines 368–400): RRWP computation → edge attribute extraction → projection → layer-by-layer processing.

**Math to include** (from GRIT / GraphLLM paper):
- RRWP: $R_{i,j} = [I_{i,j}, M_{i,j}, M^2_{i,j}, \ldots, M^{C-1}_{i,j}] \in \mathbb{R}^C$ — Equation 6.
- Random walk matrix: $M = D^{-1}A$.
- Attention with edge bias: $\text{score} = (K_{src} + Q_{dst}) \odot W_{ew}(R) + W_{eb}(R)$, then signed sqrt, ReLU, Einstein summation with $A_w$.
- Detail every learnable matrix and its `nn.Linear` / `nn.Parameter` counterpart.

**Paper references to cite**:
- Chai, Z., et al. (2025). *GraphLLM: Boosting Graph Reasoning Ability of Large Language Model.* IEEE Transactions on Big Data. Section 3.3 — Graph Transformer.
- Ma, L., et al. (2023). *Graph Inductive Biases in Transformers without Message Passing.* ICML. — GRIT architecture.

---

### File 3 — `docs/methodology/neural_layer/engine_graph_aware_attention.md`

**Primary source**: `src/attention/graph_aware_attention.py` (≈ 311 lines)

Document the following components in order:

1. **`GraphAwareAttentionConfig`** (lines 22–30)
   - Dataclass fields: `beta_init`, `similarity_metric`, `max_distance`, `hidden_dim`, `num_heads`, `dropout`.
   - Corresponding YAML keys in `configs/model_config.yaml` (lines 22–24).

2. **`GraphAwareAttention`** (lines 33–177)
   - Learnable $\beta$ parameter (`self.beta`, line 59).
   - `structural_similarity_matrix` buffer (line 68).
   - `set_structural_similarity_matrix()` (lines 70–82).
   - `forward()` (lines 84–159): standard Q·K attention → add $\beta \cdot S(n_i, n_j)$ at line 132 → softmax → weighted V.
   - `get_attention_focus()` (lines 161–177): diagnostics/statistics.

3. **`MultiHeadGraphAwareAttention`** (lines 180–230)
   - Wrapper creating multiple `GraphAwareAttention` heads.

4. **`compute_structural_similarity_matrix()`** (lines 233–270)
   - "shortest_path" metric: $S(n_i, n_j) = 1 / (1 + d(n_i, n_j))$ — line 261.
   - "common_neighbors" metric: Jaccard similarity — lines 262–268.

5. **`_compute_shortest_path()`** (lines 273–311)
   - BFS-based shortest path for all pairs.

6. **Utility functions in `src/utils.py`**:
   - `compute_structural_similarity()` (lines 91–148): pairwise similarity.
   - `batch_structural_similarity()` (lines 151–171): batch computation.
   - Config field `graph_aware_attention_beta` (line 58).

7. **Module exports**: `src/attention/__init__.py` — `GraphAwareAttention`, `GraphAwareAttentionConfig`, `MultiHeadGraphAwareAttention`, `compute_structural_similarity_matrix`.

**Math to include** (S-GraphLLM enhancement):
- Modified attention: $\alpha'_{ij} = \frac{\exp\!\left(\frac{q_i^T k_j}{\sqrt{d_k}} + \beta \cdot S(n_i, n_j)\right)}{\sum_l \exp\!\left(\frac{q_i^T k_l}{\sqrt{d_k}} + \beta \cdot S(n_i, n_l)\right)}$
- Similarity metrics:
  - Shortest path: $S(n_i, n_j) = \frac{1}{1 + d(n_i, n_j)}$.
  - Common neighbors (Jaccard): $S(n_i, n_j) = \frac{|N(n_i) \cap N(n_j)|}{|N(n_i) \cup N(n_j)|}$.
- $\beta$ is a learnable scalar controlling structural bias strength.

**Paper references to cite**:
- Chai, Z., et al. (2025). *GraphLLM* — S-GraphLLM extends the attention mechanism with structural similarity bias.

---

### File 4 — `docs/methodology/neural_layer/engine_integration.md`

**Primary sources**: `src/agents/node_encoder_decoder.py`, `src/graph_processing/graph_transformer.py`, `src/attention/graph_aware_attention.py`, `src/agents/orchestrator.py` (lines 57–65, 162–185), `src/utils.py` (lines 35–74)

This file documents **how the three Neural Layer engines connect to each other and to the surrounding layers**. Follow the same pattern as the Scalability and Reasoning Layer integration docs. Specifically:

1. **Architecture Overview** — A Mermaid diagram showing the Neural Layer within the four-layer stack, with data flows from:
   - **Scalability Layer**: Partitioned adjacency matrices $A$ → Structural Engine (RRWP) and Focus Engine (similarity); node text descriptions → Semantic Engine
   - **Internal**: Semantic Engine → Structural Engine (Connection A: $H_i$ becomes initial node features $x$)
   - **Downstream to Reasoning Layer**: $H_i$ → Stage 2 (Connection C), $x'$ → Stage 2 (Connection D), $\alpha'$ → Stage 2 (Connection E)

2. **Connection A: Semantic Engine → Structural Engine** — Node representations $H_i$ produced by `NodeEncoderDecoder.forward()` become the initial node features $x$ for `GraphTransformer.forward()`. Document:
   - Producer: `src/agents/node_encoder_decoder.py`, line 293
   - Consumer: `src/graph_processing/graph_transformer.py`, line 368
   - The critical dimension constraint: $\text{NodeEncoderDecoder.output}\_\text{dim} = \text{GraphTransformer.embed}\_\text{dim}$

3. **Connection B: Scalability Layer → Both Structural & Focus Engines** — The shared adjacency matrix $A$ is consumed by:
   - `compute_rrwp_encoding()` in `graph_transformer.py` (line 46) for RRWP
   - `compute_structural_similarity_matrix()` in `graph_aware_attention.py` (line 233) for structural similarity
   Explain how these are complementary encodings of the same topology.

4. **Connection C: Semantic Engine → Reasoning Layer (Stage 2)** — Node embeddings $H_i$ formatted into fine-grained context for the LLM.

5. **Connection D: Structural Engine → Reasoning Layer (Stage 2)** — Structure-enriched embeddings $x'$ formatted into fine-grained context.

6. **Connection E: Focus Engine → Reasoning Layer (Stage 2)** — Graph-aware attention weights $\alpha'$ modulate the LLM's attention during multi-hop reasoning.

7. **Full Pipeline Walkthrough** — A sequence diagram showing the complete execution order with exact data exchanges between all three engines and the surrounding layers.

8. **Dimension Alignment Map** — A table mapping every dimension constraint across the pipeline:
   - `input_dim`, `hidden_dim`, `output_dim / embed_dim`, `rrwp_dim`, `num_heads`, `dropout`
   - Which `Config` field controls each, and the default values

9. **Shared Configuration Surface** — Which `Config` fields are shared across multiple engines and how they affect the pipeline.

10. **Complementary Structural Representations** — Explain why the Structural Engine (RRWP) and Focus Engine (similarity) are not redundant but complementary.

11. **Integration with the Orchestrator** — Show current wiring in `HierarchicalReasoningOrchestrator` and intended future integration with all three engines.

12. **Cross-References** — Link to:
    - All three engine docs in this folder
    - Scalability Layer docs (`../scalability_layer/`)
    - Reasoning Layer docs (`../reasoning_layer/`)
    - `@docs/component_guide.md`
    - `@docs/architecture_diagram.md`
    - `@docs/workflow.mermaid`
    - `@README.md`

---

## Style & Quality Checklist

Before finalizing each file, verify:

- [ ] Every class and function in the primary source file is documented.
- [ ] Every `nn.Parameter`, `nn.Linear`, `nn.LayerNorm`, and `nn.Module` is mentioned with its line number.
- [ ] All math uses LaTeX and every symbol is defined with its code counterpart.
- [ ] Underscores in LaTeX `\text{}` blocks are split: use `\text{before}\_\text{after}`, NOT `\text{before\_after}`.
- [ ] Mermaid diagrams do NOT contain underscore characters in node labels.
- [ ] "Plain English Input/Output" includes tensor shapes **and** a human-readable example.
- [ ] "Python Perspective" snippets are self-contained and annotated with shapes.
- [ ] Internal method walkthroughs reference exact line ranges.
- [ ] Helper functions from other files (`src/utils.py`, etc.) are cross-referenced.
- [ ] Configuration fields from `configs/model_config.yaml` are listed with defaults.
- [ ] Cross-references link to `@README.md`, `@docs/component_guide.md`, `@docs/architecture_diagram.md`, and the other engine docs.
- [ ] The integration doc covers every data flow connection with producer/consumer code locations.

---

## Additional Information Sources

When writing these documents, consult the following for context, theory, and validation details:

- **`@README.md`** — Project overview, architecture summary, performance targets, and paper references.
- **`@docs/README.md`** — Documentation index and navigation guide.
- **`@docs/component_guide.md`** — End-to-end component guide with all layers described. Particularly:
  - Component 3.1: Node Encoder (lines 178–210)
  - Component 3.2: Node Decoder (lines 213–248)
  - Component 3.3: RRWP Encoding (lines 251–290)
  - Component 3.4: Graph Transformer (lines 293–340)
  - Component 3.5: Graph-Aware Attention (lines 343–387)
- **`@docs/architecture_diagram.md`** — Mermaid diagrams and component interaction matrix showing the Neural Layer's internal connections and external data flows.
- **`@docs/graph_partitioning_theory.md`** — Theoretical background on METIS and spectral partitioning (relevant to the Structural Engine's input — the adjacency matrices come from the Scalability Layer).
- **`@docs/hierarchical_reasoning_theory.md`** — Coarse-to-fine reasoning theory (relevant to how engine outputs feed into Stage 2 of the Reasoning Layer).
- **`@docs/workflow.mermaid`** — Visual workflow of the entire pipeline, showing the neural component connections (lines 79–87).
- **`@docs/methodology/scalability_layer/`** — Scalability Layer docs to reference for upstream connections (partitioned adjacency matrices, node text descriptions).
- **`@docs/methodology/reasoning_layer/`** — Reasoning Layer docs to reference for downstream connections (Stage 2 context formatting).
- **`@tests/`** — Unit tests showing expected behavior for the neural components.
