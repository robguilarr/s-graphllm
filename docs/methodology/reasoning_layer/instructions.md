# Instructions: Create Detailed Reasoning Layer Documentation

## Objective

Create **2 separate Markdown files** — one for each of the Reasoning Layer components in S-GraphLLM — plus **1 integration file** that connects the dots between them and shows how the Reasoning Layer consumes outputs from the Scalability and Neural Layers to produce final answers. Each file must follow the template structure below while being **significantly more detailed** than the template: every class, method, data structure, prompt template, and helper function must be cited with its **exact file path and line numbers** in the codebase.

Use `@README.md` and `@docs/` (especially `@docs/component_guide.md`, `@docs/architecture_diagram.md`, and `@docs/hierarchical_reasoning_theory.md`) as additional information sources to enrich descriptions, cross-reference theoretical foundations, and ensure consistency with the existing documentation.

---

## Files to Create

| # | Filename | Component | Primary Source File |
|---|----------|-----------|---------------------|
| 1 | `docs/methodology/reasoning_layer/orchestrator.md` | **Hierarchical Reasoning Orchestrator — The Conductor** | `src/agents/orchestrator.py` |
| 2 | `docs/methodology/reasoning_layer/llm_agent.md` | **LLM Agent — The Reasoner** | `src/agents/llm_agent.py` |
| 3 | `docs/methodology/reasoning_layer/layer_integration.md` | **Reasoning Layer Integration** | Multiple sources (see § Integration File Guidance) |

> **Note**: All files go under the `docs/methodology/reasoning_layer/` directory.

---

## Context: The Reasoning Layer in S-GraphLLM

The Reasoning Layer is **Layer 4** in the S-GraphLLM four-layer stack (**Scalability → Neural → Reasoning → Output**). It is the intelligence core — it orchestrates a **three-stage hierarchical reasoning process** that progressively narrows focus from the entire graph down to specific nodes and relationships, guided by an LLM. This design is inspired by the cognitive science principle of coarse-to-fine processing: first understand the big picture, then zoom into the details, and finally synthesize a coherent answer.

The Reasoning Layer contains two primary components:

1. **Hierarchical Reasoning Orchestrator** (`HierarchicalReasoningOrchestrator`) — The conductor that manages the three-stage pipeline:
   - **Stage 1: Coarse-Grained Reasoning** — Uses the coarse graph $G_\text{coarse}$ from the Scalability Layer to identify which partitions are most relevant to the query. The LLM examines partition summaries and selects the most promising regions.
   - **Stage 2: Fine-Grained Reasoning** — Extracts detailed subgraphs from the selected partitions, formats node features and structural information as context, and has the LLM perform multi-hop reasoning over the detailed evidence.
   - **Stage 3: Answer Synthesis** — Combines coarse-grained context and fine-grained evidence into a final answer with confidence scoring.

2. **LLM Agent** (`LLMAgent`) — The reasoning engine that wraps the OpenAI API, manages conversation history, constructs prompts, and provides specialized methods for entity extraction, relationship extraction, question answering, and multi-hop reasoning.

The Reasoning Layer consumes outputs from **all** upstream layers:
- **From Scalability Layer**: Coarse graph $G_\text{coarse}$ (Stage 1), partitioned subgraphs (Stage 2)
- **From Neural Layer** (intended integration): Node embeddings $H_i$ (Semantic Engine), structural embeddings $x'$ (Structural Engine), attention weights $\alpha'$ (Focus Engine) — all feed into Stage 2 context
- **Produces**: `ReasoningResult` containing `final_answer`, `confidence`, `reasoning_steps`, and intermediate results from all three stages

---

## Template & Format

Each component file **must** contain the following sections. Expand every section with the level of detail described in the annotations.

````markdown
# [Component Name] — [Subtitle]

> One-paragraph executive summary: what this component does in the overall S-GraphLLM pipeline,
> why it exists, and which theoretical foundations it implements.
> Reference the relevant section of `@README.md`, `@docs/component_guide.md`, and
> `@docs/hierarchical_reasoning_theory.md`.

---

## Architecture Overview

A short prose description of how the component fits into the four-layer architecture
(Scalability → Neural → Reasoning → Output). Include a small Mermaid diagram if helpful.

---

## Components Breakdown

For **every class, dataclass, and standalone function** in the primary source file, create a subsection:

### N. ClassName / function_name — (Role in one phrase)

* **Location**: `src/path/to/file.py`, lines XX–YY
* **Purpose**: 2–3 sentence description.
* **Paper Reference**: Section from the GraphLLM paper or related cognitive science / reasoning papers.

#### The Design (or The Math, where applicable)

For the orchestrator, describe the **reasoning flow** and **prompt construction** logic.
For the LLM agent, describe the **API interaction pattern** and **conversation management**.
Where math exists, use LaTeX notation.
For each formula or design pattern:
  - State the approach.
  - Define every key variable (e.g., "$G_\text{coarse}$ — the coarsened graph, produced by `GraphCoarsener.coarsen_graph()`").
  - Map each variable back to the exact code location.

**Important**: Do NOT place underscore characters inside `\text{}` blocks in LaTeX.
Use `\text{before}\_\text{after}` to split identifiers with underscores,
keeping the escaped underscore in math mode. Example:
`\text{coarse}\_\text{reasoning}` instead of `\text{coarse\_reasoning}`.

#### Plain English Input / Output

* **Input**: Describe types, shapes, and a concrete human-readable example
  (e.g., "A natural language question like `'What is the relationship between Albert Einstein and the theory of relativity?'`").
* **Output**: Same level of detail, including the structure of `ReasoningResult`.
* **Side Effects / State**: Note any conversation history, stored reasoning traces, or cached results.

#### Python Perspective

Provide a **runnable-style** code snippet showing the component in isolation.
Include types/shapes as comments on every line.

```python
# Example:
from src.agents.orchestrator import HierarchicalReasoningOrchestrator
from src.agents.llm_agent import LLMAgent

orchestrator = HierarchicalReasoningOrchestrator(graph=G, config=config)
orchestrator.setup()

agent = LLMAgent(model="gpt-4.1-mini", temperature=0.7)
result = orchestrator.reason(query="Who developed relativity?", llm_agent=agent)
# result.final_answer: str — "Albert Einstein developed the theory of relativity..."
# result.confidence: float — 0.92
```

#### Internal Method Walkthrough

For each method inside the class:
  - **Method name** (line range)
  - Step-by-step description of what it does.
  - Which other classes/functions it calls (with file + line references).
  - For prompt-construction methods: include the **full prompt template** or describe its structure.

---

## Helper / Utility Functions

List any helper functions used by this component from `src/utils.py` or other modules,
with file paths, line ranges, and a brief description of their role.

---

## Configuration

Describe every relevant field from `configs/model_config.yaml` and any config objects
(`Config`, `LLMConfig`). State default values and how they affect behavior.

---

## Cross-References

- Link to related docs in `@docs/` (e.g., `hierarchical_reasoning_theory.md`, `component_guide.md`).
- Link to the other component doc in this folder.
- Link to the integration doc.
- Reference `@README.md` sections on architecture and validation.
- Link to the Scalability Layer docs that produce this layer's inputs.
- Link to the Neural Layer docs that produce this layer's inputs.
````

---

## Detailed Per-File Guidance

### File 1 — `docs/methodology/reasoning_layer/orchestrator.md`

**Primary source**: `src/agents/orchestrator.py` (318 lines)

Document the following components in order:

1. **`ReasoningResult`** (lines 21–30)
   - Dataclass with fields: `query`, `coarse_reasoning`, `selected_partitions`, `fine_grained_reasoning`, `final_answer`, `confidence`, `reasoning_steps`.
   - This is the pipeline's final output structure. Document every field, its type, and what populates it during the three stages.

2. **`HierarchicalReasoningOrchestrator`** (lines 33–318)
   - `__init__(graph, config, node_features)` (lines 38–65): Initializes with graph, config, and optional node features. Creates `GraphPartitioner` and `GraphCoarsener` instances. Document every instance variable.
   - `setup()` (lines 67–98): Partitions the graph and creates the coarse graph. This is the **bridge to the Scalability Layer** — document the exact calls to `partition_graph()` and `coarsen_graph()`.
   - `reason(query, llm_agent)` (lines 100–214): The **main reasoning pipeline**. This is the most important method — document each of the three stages in detail:
     - **Stage 1** (lines ~110–135): Coarse-grained reasoning — creates prompt with coarse graph description, calls LLM, extracts selected partitions.
     - **Stage 2** (lines ~137–185): Fine-grained reasoning — extracts subgraph for selected partitions, formats context with `format_graph_context()`, calls LLM for multi-hop reasoning.
     - **Stage 3** (lines ~187–210): Synthesis — creates synthesis prompt combining coarse and fine results, calls LLM for final answer with confidence.
   - `_create_coarse_reasoning_prompt(query, coarse_description)` (lines 216–233): Constructs the Stage 1 prompt. **Include the full prompt template** — this is critical for understanding the reasoning design.
   - `_create_fine_reasoning_prompt(query, context)` (lines 235–254): Constructs the Stage 2 prompt. **Include the full prompt template.**
   - `_create_synthesis_prompt(query, coarse_reasoning, fine_reasoning)` (lines 256–275): Constructs the Stage 3 prompt. **Include the full prompt template.**
   - `get_reasoning_trace()` (lines 277–291): Returns the reasoning history.
   - `save_reasoning_history(filepath)` (lines 293–298): Saves history to JSON.
   - `get_partition_info()` (lines 300–317): Returns partition statistics.

**Design patterns to document**:
- The three-stage coarse-to-fine reasoning pattern and its cognitive science justification.
- How the orchestrator acts as a **facade** that hides the complexity of the Scalability and Neural Layers from the calling code.
- The prompt engineering strategy: each stage has a carefully constructed prompt that provides the right level of detail for the LLM to reason effectively.
- The confidence scoring mechanism in Stage 3.

**Paper references to cite**:
- Chai, Z., et al. (2025). *GraphLLM: Boosting Graph Reasoning Ability of Large Language Model.* IEEE Transactions on Big Data. Section 3 — hierarchical reasoning approach.
- Wang, G., et al. (2025). "Hierarchical Reasoning Model." arXiv:2506.21734.
- Nguyen, B. X., et al. (2022). "Coarse-to-Fine Reasoning for Visual Question Answering." CVPR Workshop.
- Collins, A. M., & Quillian, M. R. (1969). "Retrieval time from semantic memory." — Cognitive science foundation for hierarchical knowledge retrieval.

---

### File 2 — `docs/methodology/reasoning_layer/llm_agent.md`

**Primary source**: `src/agents/llm_agent.py` (280 lines)

Document the following components in order:

1. **`LLMConfig`** (lines 15–22)
   - Dataclass fields: `model`, `temperature`, `max_tokens`, `top_p`, `api_key`.
   - Map each field to `configs/model_config.yaml` keys.

2. **`LLMAgent`** (lines 25–280)
   - `__init__(model, temperature, max_tokens, api_key)` (lines 30–49): Initializes the OpenAI client. Document the conversation history management (`self.conversation_history`).
   - `reason(prompt, system_prompt, include_history)` (lines 51–124): The core reasoning method. This is what the orchestrator calls at each stage. Document:
     - How conversation history is managed (lines ~55–65).
     - The OpenAI API call structure (lines ~70–95).
     - Error handling and fallback behavior (lines ~96–124).
     - How `include_history` affects the messages array.
   - `batch_reason(prompts, system_prompt)` (lines 126–146): Batch processing of multiple prompts.
   - `extract_entities(text)` (lines 148–166): Entity extraction from text using a specialized prompt.
   - `extract_relationships(text)` (lines 168–198): Relationship extraction between entities.
   - `answer_question(question, context)` (lines 200–225): Question answering with provided context.
   - `multi_hop_reasoning(question, evidence_chain)` (lines 227–259): Multi-hop reasoning over a chain of evidence — **critical for Stage 2**.
   - `clear_history()` (lines 261–264): Resets conversation history.
   - `_get_default_system_prompt()` (lines 266–279): Returns the default system prompt. **Include the full prompt text** — it defines the agent's behavior.

**Design patterns to document**:
- The **conversation history** pattern: how the agent maintains context across multiple calls within a single reasoning session.
- The **specialized prompt** pattern: how each method (`extract_entities`, `extract_relationships`, `multi_hop_reasoning`) constructs a task-specific prompt.
- The **error handling** pattern: how API failures are caught and what fallback behavior exists.
- The separation between the `LLMAgent` (stateful wrapper) and the `LLMConfig` (pure configuration).

**Paper references to cite**:
- Brown, T., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS — Foundation for the LLM-as-reasoner paradigm.
- Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." NeurIPS — The prompting strategy for multi-hop reasoning.
- Yao, S., et al. (2023). "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." — Related reasoning approach.

---

### File 3 — `docs/methodology/reasoning_layer/layer_integration.md`

**Primary sources**: `src/agents/orchestrator.py`, `src/agents/llm_agent.py`, `src/utils.py` (lines 190–224), `src/main.py` (lines 134–167)

This file documents **how the two Reasoning Layer components connect to each other and how the Reasoning Layer consumes outputs from both upstream layers**. Follow the same pattern as `docs/methodology/neural_layer/engine_integration.md`. Specifically:

1. **Architecture Overview** — A Mermaid diagram showing the Reasoning Layer within the four-layer stack, with data flows from:
   - **Scalability Layer**: $G_\text{coarse}$ → Stage 1, partition subgraphs → Stage 2
   - **Neural Layer** (intended): $H_i$ → Stage 2, $x'$ → Stage 2, $\alpha'$ → Stage 2
   - **Internal**: Orchestrator ↔ LLM Agent interaction across three stages

2. **Connection A: Orchestrator → LLM Agent (Three-Stage Calls)** — Document the three separate calls from the orchestrator to the LLM agent:
   - Stage 1 call: `_create_coarse_reasoning_prompt()` → `llm_agent.reason()` — partition selection
   - Stage 2 call: `_create_fine_reasoning_prompt()` → `llm_agent.reason()` — multi-hop reasoning
   - Stage 3 call: `_create_synthesis_prompt()` → `llm_agent.reason()` — answer synthesis
   Document exact line numbers for each call site in the orchestrator.

3. **Connection B: Scalability Layer → Stage 1** — How the coarse graph description flows from `GraphCoarsener.get_coarse_graph_description()` through the orchestrator into the Stage 1 prompt. Include the data transformation: `G_coarse (nx.Graph)` → `text description (str)` → `prompt (str)` → `LLM response (str)` → `selected_partitions (List[int])`.

4. **Connection C: Scalability Layer → Stage 2** — How selected partition IDs from Stage 1 are used to extract fine-grained subgraphs:
   - `find_relevant_partitions()` for keyword-based selection
   - `get_subgraph_for_partitions()` for subgraph extraction
   - `format_graph_context()` (`src/utils.py`, lines 190–224) for text formatting
   Document the full data transformation pipeline.

5. **Connection D: Neural Layer → Stage 2 (Intended Integration)** — Document the planned but not yet wired connections:
   - Node embeddings $H_i$ from the Semantic Engine
   - Structural embeddings $x'$ from the Structural Engine
   - Attention weights $\alpha'$ from the Focus Engine
   Reference `docs/methodology/neural_layer/engine_integration.md` (Connections C, D, E) and show the intended code from the orchestrator.

6. **Connection E: Stage 3 → Output** — How the `ReasoningResult` is assembled from the outputs of all three stages and returned to the caller. Document the fields and their provenance.

7. **Data Flow Map** — A comprehensive table showing every data exchange:
   - Input type, shape, producer, consumer, code locations
   - Include both current wiring and intended (future) connections

8. **Prompt Engineering Analysis** — A section dedicated to the three prompt templates:
   - Stage 1 prompt: What information is included, how the coarse graph is presented, what the LLM is asked to do
   - Stage 2 prompt: How the fine-grained context is structured, what multi-hop reasoning looks like
   - Stage 3 prompt: How coarse and fine results are combined, how confidence is elicited
   This section should help anyone modifying the prompts understand the rationale.

9. **Shared Configuration Surface** — Which `Config` and `LLMConfig` fields affect both components:
   - `temperature`, `max_tokens`, `top_p`, `model` (LLM behavior)
   - `max_hops`, `context_window`, `top_k` (reasoning behavior)
   - `max_nodes_per_partition` (affects Stage 2 subgraph size)

10. **Orchestrator as System Facade** — Explain how the orchestrator acts as the single entry point that hides all layer complexity. Show the call chain from `src/main.py` (lines 134–167) through `orchestrator.reason()` to the final result.

11. **Cross-References** — Link to:
    - Both component docs in this folder
    - Scalability Layer docs (`../scalability_layer/`)
    - Neural Layer docs (`../neural_layer/`)
    - `@docs/hierarchical_reasoning_theory.md`
    - `@docs/component_guide.md`
    - `@docs/architecture_diagram.md`
    - `@docs/workflow.mermaid`
    - `@README.md`

---

## Style & Quality Checklist

Before finalizing each file, verify:

- [ ] Every class, dataclass, and function in the primary source file is documented.
- [ ] Every instance variable, conversation state, and stored result is mentioned with its line number.
- [ ] Prompt templates are included verbatim or described in sufficient detail to reconstruct them.
- [ ] All math uses LaTeX and every symbol is defined with its code counterpart.
- [ ] Underscores in LaTeX `\text{}` blocks are split: use `\text{before}\_\text{after}`, NOT `\text{before\_after}`.
- [ ] Mermaid diagrams do NOT contain underscore characters in node labels.
- [ ] "Plain English Input/Output" includes types **and** a human-readable example.
- [ ] "Python Perspective" snippets are self-contained and annotated with types.
- [ ] Internal method walkthroughs reference exact line ranges.
- [ ] Helper functions from other files (`src/utils.py`, etc.) are cross-referenced.
- [ ] Configuration fields from `configs/model_config.yaml` are listed with defaults.
- [ ] Cross-references link to all related documentation.
- [ ] The integration doc covers every data flow connection with producer/consumer code locations.
- [ ] The integration doc includes prompt engineering analysis for all three stages.
- [ ] The `ReasoningResult` dataclass is fully documented with field provenance.

---

## Additional Information Sources

When writing these documents, consult the following for context, theory, and validation details:

- **`@README.md`** — Project overview, architecture summary, performance targets, and paper references.
- **`@docs/README.md`** — Documentation index and navigation guide.
- **`@docs/component_guide.md`** — End-to-end component guide with all layers described. Particularly:
  - Component 4.1: Hierarchical Reasoning Orchestrator (lines 360–393)
  - Component 4.2: Stage 1 — Coarse-Grained Reasoning (lines 397–431)
  - Component 4.3: Stage 2 — Fine-Grained Reasoning (lines 434–476)
  - Component 4.4: Stage 3 — Answer Synthesis (lines 479–512)
  - Component 4.5: LLM Agent (lines 516–551)
- **`@docs/architecture_diagram.md`** — Mermaid diagrams showing hierarchical reasoning layer structure and stage-by-stage data flow.
- **`@docs/hierarchical_reasoning_theory.md`** — In-depth theoretical background (390 lines) covering:
  - Cognitive science foundations (Collins & Quillian, spreading activation)
  - Coarse-to-fine reasoning paradigm
  - Multi-hop reasoning theory
  - Divide-and-conquer over graphs
  - Complexity analysis of hierarchical vs. flat reasoning
- **`@docs/graph_partitioning_theory.md`** — Explains how the Scalability Layer prepares the data this layer consumes.
- **`@docs/workflow.mermaid`** — Visual workflow of the entire pipeline, showing the three reasoning stages (lines 88–97).
- **`@docs/methodology/neural_layer/`** — Existing Neural Layer docs, especially `engine_integration.md` which documents Connections C, D, E (Neural → Reasoning).
- **`@docs/methodology/scalability_layer/`** — Scalability Layer docs to reference for upstream connections.
- **`@src/main.py`** (lines 134–167) — The main entry point that shows how the orchestrator and LLM agent are instantiated and called.
- **`@tests/test_reasoning.py`** — Unit tests for the orchestrator (lines 12–115) and LLM agent (lines 118–163) showing expected behavior.
