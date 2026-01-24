# S-GraphLLM Architecture Diagram

This document contains the validated Mermaid.js diagram representing the complete S-GraphLLM architecture.

## Validation Notes

The original reference diagram had the following issues that have been corrected:

1. **Missing connection from Orchestrator to ReasoningProcess**: The orchestrator should connect to the reasoning stages, not directly to the result
2. **Unclear data flow labels**: Added specific input/output types for all edges
3. **GraphLLM components not properly integrated**: Fixed the flow showing when neural components are applied
4. **Missing LLM Agent component**: Added explicit LLM Agent that serves all reasoning stages
5. **Incorrect stage transitions**: Fixed the flow between stages to show proper data dependencies

---

## Complete Architecture Diagram

### Main Flow

```mermaid
flowchart TD
    Start(["Input: Knowledge Graph + Query"]) --> ScalLayer["Layer 2: Scalability Layer"]
    ScalLayer --> NeuralLayer["Layer 3: Neural Components"]
    ScalLayer --> ReasonLayer["Layer 4: Reasoning Layer"]
    NeuralLayer --> ReasonLayer
    ReasonLayer --> Output(["Output: Final Answer"])
    
    style Start fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style ScalLayer fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style NeuralLayer fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style ReasonLayer fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    style Output fill:#fce4ec,stroke:#c2185b,stroke-width:2px
```

### Layer 2: Scalability Layer

```mermaid
flowchart LR
    Input["Knowledge Graph G"] --> GP["Graph Partitioner"]
    GP --> GC["Graph Coarsener"]
    
    GP -.-> PS1["METIS-like"]
    GP -.-> PS2["Louvain"]
    GP -.-> PS3["Balanced"]
    
    GC --> Out1["Partitions P1...Pk"]
    GC --> Out2["Coarse Graph"]
    
    style Input fill:#e3f2fd,stroke:#1565c0
    style GP fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style GC fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style PS1 fill:#f5f5f5,stroke:#757575,stroke-dasharray: 5 5
    style PS2 fill:#f5f5f5,stroke:#757575,stroke-dasharray: 5 5
    style PS3 fill:#f5f5f5,stroke:#757575,stroke-dasharray: 5 5
```

### Layer 3: GraphLLM Neural Components

```mermaid
flowchart TB
    subgraph NodeComp["Node Understanding"]
        NE["Node Encoder"] --> ND["Node Decoder"]
    end
    
    subgraph StructComp["Structure Understanding"]
        RRWP["RRWP Encoding"] --> GRIT["Graph Transformer"]
    end
    
    subgraph AttnComp["Attention Modulation"]
        SIM["Similarity Metric"] --> GAA["Graph-Aware Attention"]
    end
    
    Input1["Node Text"] --> NE
    Input2["Graph Structure"] --> RRWP
    
    ND --> Output1["Node Embeddings"]
    GRIT --> Output2["Structural Embeddings"]
    GAA --> Output3["Attention Weights"]
    
    style NodeComp fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style StructComp fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style AttnComp fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```

### Layer 4: Hierarchical Reasoning

```mermaid
flowchart TD
    Orch["Hierarchical Orchestrator"] --> Stage1
    
    subgraph Stage1["Stage 1: Coarse-Grained"]
        S1A["Generate Coarse Context"] --> S1B["Extract Keywords"]
        S1B --> S1C["Filter Partitions"]
        S1C --> S1D["LLM: Rank Partitions"]
    end
    
    Stage1 --> Stage2
    
    subgraph Stage2["Stage 2: Fine-Grained"]
        S2A["Extract Subgraphs"] --> S2B["Apply Neural Encoding"]
        S2B --> S2C["Format Context"]
        S2C --> S2D["LLM: Multi-hop Reasoning"]
    end
    
    Stage2 --> Stage3
    
    subgraph Stage3["Stage 3: Synthesis"]
        S3A["Aggregate Evidence"] --> S3B["Resolve Conflicts"]
        S3B --> S3C["LLM: Final Answer"]
    end
    
    LLM["LLM Agent"] -.-> S1D
    LLM -.-> S2D
    LLM -.-> S3C
    
    style Orch fill:#fff3e0,stroke:#ef6c00,stroke-width:3px
    style Stage1 fill:#fff9e6,stroke:#ffa726
    style Stage2 fill:#fff9e6,stroke:#ffa726
    style Stage3 fill:#fff9e6,stroke:#ffa726
    style LLM fill:#e0f7fa,stroke:#00838f,stroke-width:2px
```

### Complete Data Flow

```mermaid
flowchart TD
    KG(["Knowledge Graph"])
    Q(["Query"])
    
    KG --> GP["Graph Partitioner"]
    GP --> GC["Graph Coarsener"]
    
    KG --> NE["Node Encoder"]
    NE --> ND["Node Decoder"]
    
    GP --> RRWP["RRWP"]
    RRWP --> GRIT["Graph Transformer"]
    
    GC --> S1["Stage 1: Coarse"]
    Q --> S1
    
    S1 --> S2["Stage 2: Fine"]
    ND --> S2
    GRIT --> S2
    
    S2 --> S3["Stage 3: Synthesis"]
    
    S3 --> Result(["Final Answer"])
    
    style KG fill:#e3f2fd,stroke:#1565c0
    style Q fill:#e3f2fd,stroke:#1565c0
    style GP fill:#e8f5e9,stroke:#2e7d32
    style GC fill:#e8f5e9,stroke:#2e7d32
    style NE fill:#f3e5f5,stroke:#7b1fa2
    style ND fill:#f3e5f5,stroke:#7b1fa2
    style RRWP fill:#f3e5f5,stroke:#7b1fa2
    style GRIT fill:#f3e5f5,stroke:#7b1fa2
    style S1 fill:#fff3e0,stroke:#ef6c00
    style S2 fill:#fff3e0,stroke:#ef6c00
    style S3 fill:#fff3e0,stroke:#ef6c00
    style Result fill:#fce4ec,stroke:#c2185b
```

---

## Simplified Overview Diagram

For presentations and high-level understanding:

```mermaid
flowchart LR
    subgraph Input ["Input"]
        A[("Graph + Query")]
    end
    
    subgraph Scalability ["Scalability Layer"]
        B["Partition<br/>& Coarsen"]
    end
    
    subgraph Neural ["Neural Layer"]
        C["GraphLLM<br/>Encoding"]
    end
    
    subgraph Reasoning ["Reasoning Layer"]
        D["Coarse →<br/>Fine →<br/>Synthesize"]
    end
    
    subgraph Output ["Output"]
        E[("Answer")]
    end
    
    A --> B --> C --> D --> E
    B --> D
    
    style Input fill:#e3f2fd
    style Scalability fill:#e8f5e9
    style Neural fill:#f3e5f5
    style Reasoning fill:#fff3e0
    style Output fill:#fce4ec
```

---

## Component Interaction Matrix

```mermaid
flowchart TD
    subgraph Legend ["Legend: Data Types"]
        direction LR
        L1["━━━ Primary Data Flow"]
        L2["- - - Strategy Selection"]
        L3["◄──► Bidirectional (LLM)"]
    end
```

| From Component | To Component | Data Type | Format |
|----------------|--------------|-----------|--------|
| Knowledge Graph | Graph Partitioner | Graph | `G = (V, E, X)` |
| Graph Partitioner | Graph Coarsener | Partitions | `List[Set[NodeID]]` |
| Graph Coarsener | Stage 1 | Coarse Graph | `G_coarse = (V*, E*)` |
| Knowledge Graph | Node Encoder | Text | `List[str]` |
| Graph Partitioner | RRWP Encoding | Adjacency | `torch.Tensor` |
| Node Decoder | Stage 2 | Embeddings | `torch.Tensor[n, d]` |
| Graph Transformer | Stage 2 | Embeddings | `torch.Tensor[n, d]` |
| Graph-Aware Attention | Stage 2 LLM | Weights | `torch.Tensor[n, n]` |
| Stage 1 LLM | Stage 2 | Partition IDs | `List[int]` |
| Stage 2 LLM | Stage 3 | Results | `List[ReasoningResult]` |
| Stage 3 LLM | Output | Answer | `str + metadata` |

---

## Stage-by-Stage Data Flow

### Stage 1: Coarse-Grained Reasoning

```mermaid
sequenceDiagram
    participant Q as Query
    participant GC as Coarse Graph
    participant S1 as Stage 1
    participant LLM as LLM Agent
    
    Q->>S1: Query string
    GC->>S1: G_coarse (super-nodes, super-edges)
    S1->>S1: Generate textual context
    S1->>S1: Extract keywords from query
    S1->>S1: Match keywords to partitions
    S1->>LLM: "Rank these partitions by relevance..."
    LLM->>S1: Ranked partition list
    S1->>S1: Select top-k partitions
    Note over S1: Output: [P₃, P₇, P₁₂, P₂₁, P₄₅]
```

### Stage 2: Fine-Grained Reasoning

```mermaid
sequenceDiagram
    participant P as Selected Partitions
    participant NE as Neural Encoder
    participant GT as Graph Transformer
    participant S2 as Stage 2
    participant LLM as LLM Agent
    
    P->>S2: Partition IDs [P₃, P₇, ...]
    S2->>S2: Extract full subgraphs
    S2->>NE: Node text descriptions
    NE->>S2: Node embeddings Hᵢ
    S2->>GT: Subgraph structure
    GT->>S2: Structural embeddings
    S2->>S2: Format detailed context
    S2->>LLM: "Reason step-by-step over this graph..."
    LLM->>S2: Reasoning paths + evidence
    Note over S2: Output: List[ReasoningResult]
```

### Stage 3: Answer Synthesis

```mermaid
sequenceDiagram
    participant R as Reasoning Results
    participant S3 as Stage 3
    participant LLM as LLM Agent
    participant O as Output
    
    R->>S3: Results from all partitions
    S3->>S3: Aggregate evidence
    S3->>S3: Detect conflicts
    S3->>S3: Resolve by confidence/voting
    S3->>LLM: "Synthesize final answer from evidence..."
    LLM->>S3: Coherent answer
    S3->>O: Answer + Evidence + Confidence
    Note over O: Final structured response
```

---

## Implementation File Mapping

```mermaid
flowchart TD
    subgraph Files ["Implementation Files"]
        direction TB
        
        F1["src/graph_processing/partitioner.py"]
        F2["src/graph_processing/coarsener.py"]
        F3["src/graph_processing/graph_transformer.py"]
        F4["src/agents/node_encoder_decoder.py"]
        F5["src/attention/graph_aware_attention.py"]
        F6["src/agents/orchestrator.py"]
        F7["src/agents/llm_agent.py"]
    end
    
    subgraph Components ["Components"]
        direction TB
        
        C1["Graph Partitioner"]
        C2["Graph Coarsener"]
        C3["RRWP + Graph Transformer"]
        C4["Node Encoder-Decoder"]
        C5["Graph-Aware Attention"]
        C6["Hierarchical Orchestrator"]
        C7["LLM Agent"]
    end
    
    F1 --> C1
    F2 --> C2
    F3 --> C3
    F4 --> C4
    F5 --> C5
    F6 --> C6
    F7 --> C7
    
    style Files fill:#f5f5f5
    style Components fill:#e3f2fd
```

---

## Rendering Instructions

To render these diagrams:

1. **GitHub**: Diagrams render automatically in `.md` files
2. **VS Code**: Install "Markdown Preview Mermaid Support" extension
3. **Online**: Use [Mermaid Live Editor](https://mermaid.live/)
4. **CLI**: Use `manus-render-diagram` utility:
   ```bash
   manus-render-diagram docs/architecture_diagram.md output.png
   ```

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**Maintainer**: S-GraphLLM Team
