# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**神经拟态图记忆系统 (Neuro-Gated Graph Memory System)** - A next-generation retrieval mechanism for AI Agents inspired by neuroscience, implementing a dynamic memory network based on graph structures and spreading activation models.

This project aims to go beyond traditional static vector retrieval by building a graph-based memory system where:
- **Memory is a Graph**: Information stored as nodes connected by typed edges
- **Query is a Modulation Kernel**: User queries dynamically adjust edge weights at runtime
- **Retrieval is Energy Flow**: Retrieval is an iterative energy diffusion process across the graph
- **Plasticity**: The graph structure evolves through Hebbian learning (LTP) and time-based forgetting (LTD)

---

## Key Design Documents

1. **DESIGN.md** - Complete system architecture specification (Chinese)
   - Defines all components: Storage, Ingestion, Retrieval, Plasticity layers
   - Specifies data structures: `MemoryNode`, `MemoryEdge`
   - Details algorithms: Spreading Activation, Kernel Generation, LTP/LTD

2. **refer/REFER.md** - Implementation reference extracted from HippoRAG
   - Architectural patterns for LLM integration, graph construction, retrieval
   - Code examples and design patterns to follow
   - Mapping between HippoRAG components and our system

3. **refer/HippoRAG/** - Reference implementation to study
   - Production-quality RAG system with graph-based retrieval
   - Study for: dataset integration, LLM interfaces, embedding management, graph construction

---

## Development Commands

### Environment Setup

```bash
# Create conda environment
conda create -n neurogated python=3.10
conda activate neurogated

# Install dependencies (once requirements.txt is created)
pip install -r requirements.txt

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=<path to Huggingface home>
export OPENAI_API_KEY=<your openai api key>
```

### Running Tests

```bash
# Run unit tests
pytest tests/

# Run specific test file
pytest tests/test_graph_store.py

# Run with coverage
pytest --cov=src tests/
```

### Running the System

```bash
# Basic usage with OpenAI
python main.py --dataset sample --llm_name gpt-4o-mini --embedding_name nvidia/NV-Embed-v2

# With local vLLM deployment
python main.py --dataset sample --llm_base_url http://localhost:8000/v1 --llm_name meta-llama/Llama-3.3-70B-Instruct

# Force rebuild from scratch
python main.py --dataset sample --force_index_from_scratch true
```

---

## Architecture Overview

### Core Components (4 Layers)

```
src/
├── config/
│   └── memory_config.py          # MemoryConfig class with all hyperparameters
├── storage/
│   ├── graph_store.py            # Graph data structure (nodes + edges)
│   └── embedding_store.py        # Vector storage (reuse from HippoRAG)
├── ingestion/
│   ├── ingestion_engine.py       # Document → Graph pipeline
│   ├── chunker.py                # Text chunking
│   └── edge_builder.py           # SEQ/SIM/CAUSE edge construction
├── retrieval/
│   ├── neuro_retriever.py        # Main retrieval engine
│   ├── kernel_generator.py       # LLM-driven kernel generation
│   └── spreading_activation.py   # Activation diffusion algorithm
├── plasticity/
│   └── plasticity_engine.py      # LTP/LTD weight updates
├── llm/
│   ├── base.py                   # BaseLLM interface
│   ├── openai_llm.py             # OpenAI implementation
│   └── factory.py                # LLM factory function
├── prompts/
│   ├── prompt_manager.py         # Prompt template manager
│   └── templates/                # All prompt templates
│       ├── causal_extraction.py
│       ├── kernel_generation.py
│       └── ner.py
└── utils/
    ├── hash_utils.py             # compute_mdhash_id
    └── vector_utils.py           # cosine_similarity, retrieve_knn
```

### Main Entry Point

```python
# main.py
class NeuroGraphMemory:
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.store = GraphStore()
        self.retriever = NeuroRetriever(self.store, self.config)
        self.plasticity = PlasticityEngine(self.store, self.config)

    def add_document(self, text: str):
        """Pipeline: Ingestion → Graph Construction"""
        pass

    def retrieve(self, query: str) -> List[str]:
        """Pipeline: LLM Kernel → Vector Priming → Spreading Activation → Return Content"""
        pass

    def feedback(self, relevant_node_ids: List[UUID]):
        """Trigger Hebbian Learning (LTP)"""
        pass

    def maintenance(self):
        """Trigger Passive Decay (LTD)"""
        pass
```

---

## Configuration System

All "magic numbers" must be extracted into `MemoryConfig`:

```python
# src/config/memory_config.py
from dataclasses import dataclass

@dataclass
class MemoryConfig:
    # Retrieval parameters
    TOP_K_ANCHORS: int = 5           # Number of anchor nodes from vector search
    TOP_N_RETRIEVAL: int = 3         # Final number of memory nodes to return
    MAX_HOPS: int = 2                # Maximum hops for energy diffusion
    ENERGY_DECAY_RATE: float = 0.5   # Energy decay per hop

    # Graph construction
    MAX_SIM_NEIGHBORS: int = 5       # Max similarity edges per node

    # Plasticity
    HEBBIAN_LEARNING_RATE: float = 0.1  # Weight increment for LTP
    TIME_DECAY_FACTOR: float = 0.99     # Natural forgetting coefficient
    MIN_EDGE_WEIGHT: float = 0.1        # Minimum edge weight before pruning

    # LLM configuration
    llm_name: str = "gpt-4o-mini"
    llm_base_url: str = None
    temperature: float = 0.0
    max_new_tokens: int = 2048

    # Storage
    save_dir: str = "outputs"
    force_index_from_scratch: bool = False
```

---

## Data Structures

### MemoryNode
```python
@dataclass
class MemoryNode:
    id: str                    # UUID or hash ID
    content: str               # Text content
    embedding: List[float]     # Dense vector
    base_energy: float = 1.0   # Base energy level
    last_accessed: datetime    # For forgetting curve
    metadata: Dict = field(default_factory=dict)
```

### MemoryEdge
```python
from enum import Enum

class EdgeType(Enum):
    SEQ = "sequential"         # Temporal sequence
    SIM = "similarity"         # Semantic similarity
    CAUSE = "causal"           # Causal logic

@dataclass
class MemoryEdge:
    source_id: str
    target_id: str
    type: EdgeType
    weight: float              # Dynamic strength (0.0 - 1.0)
    created_at: datetime
```

---

## Key Algorithms

### 1. Spreading Activation (Retrieval Core)

```python
def spreading_activation(query: str, graph: GraphStore, config: MemoryConfig):
    # Step 1: Generate modulation kernel via LLM
    kernel = kernel_generator.generate(query)
    # kernel = {'weights': {SEQ: 0.5, SIM: 1.5, CAUSE: 2.0}, 'justification': '...'}

    # Step 2: Vector search for anchor nodes
    query_embedding = embedding_model.encode(query)
    anchors = graph.vector_search(query_embedding, k=config.TOP_K_ANCHORS)

    # Step 3: Initialize energy pool
    activations = {node.id: similarity_score for node, similarity_score in anchors}

    # Step 4: Diffusion loop (iterate MAX_HOPS times)
    for hop in range(config.MAX_HOPS):
        new_activations = defaultdict(float)

        for node_id, energy in activations.items():
            neighbors = graph.get_neighbors(node_id)

            for neighbor_id, edge in neighbors:
                # Apply kernel modulation
                kernel_mod = kernel['weights'][edge.type]

                # Calculate energy flow
                flow = energy * edge.weight * kernel_mod * config.ENERGY_DECAY_RATE

                # Accumulate energy
                new_activations[neighbor_id] += flow

        # Merge activations
        activations = merge(activations, new_activations)

    # Step 5: Return top-N nodes
    top_nodes = sorted(activations.items(), key=lambda x: x[1], reverse=True)[:config.TOP_N_RETRIEVAL]
    return [graph.get_node(node_id) for node_id, _ in top_nodes]
```

### 2. Edge Construction

**SEQ Edges (Sequential)**:
```python
# Connect consecutive chunks
for i in range(len(chunks) - 1):
    graph.add_edge(chunks[i].id, chunks[i+1].id, EdgeType.SEQ, weight=1.0)
```

**SIM Edges (Similarity)**:
```python
# NO hard threshold - use top-K neighbors
for node in new_nodes:
    similarities = cosine_similarity(node.embedding, existing_embeddings)
    top_k_indices = np.argsort(similarities)[::-1][:config.MAX_SIM_NEIGHBORS]

    for idx in top_k_indices:
        neighbor = existing_nodes[idx]
        weight = similarities[idx]  # Use similarity as weight directly
        graph.add_edge(node.id, neighbor.id, EdgeType.SIM, weight=weight)
```

**CAUSE Edges (Causal - LLM-driven)**:
```python
# Sliding window over recent neighbors
for node_a, node_b in sliding_window(nodes, window_size=5):
    # Call LLM to judge causality
    result = llm.infer(prompt=f"Is there a causal relationship between:\nA: {node_a.content}\nB: {node_b.content}")

    if result['has_causality']:
        weight = result['confidence']  # 0.0 - 1.0
        # Bidirectional edges
        graph.add_edge(node_a.id, node_b.id, EdgeType.CAUSE, weight=weight)
        graph.add_edge(node_b.id, node_a.id, EdgeType.CAUSE, weight=weight)
```

### 3. Plasticity (LTP/LTD)

**LTP (Long-Term Potentiation)**:
```python
def reinforce_path(retrieved_nodes: List[MemoryNode]):
    """Strengthen edges in the retrieval path"""
    for i in range(len(retrieved_nodes) - 1):
        edge = graph.get_edge(retrieved_nodes[i].id, retrieved_nodes[i+1].id)
        edge.weight = min(1.0, edge.weight + config.HEBBIAN_LEARNING_RATE)
```

**LTD (Long-Term Depression)**:
```python
def decay_unused():
    """Passive forgetting for unused edges"""
    for edge in graph.all_edges():
        # IMPORTANT: SIM edges do NOT decay (objective similarity)
        if edge.type == EdgeType.SIM:
            continue

        # Decay SEQ and CAUSE edges
        if not recently_activated(edge):
            edge.weight *= config.TIME_DECAY_FACTOR

            # Prune weak edges
            if edge.weight < config.MIN_EDGE_WEIGHT:
                graph.remove_edge(edge)
```

---

## LLM Integration Patterns

### Factory Pattern
```python
# src/llm/factory.py
def get_llm(config: MemoryConfig) -> BaseLLM:
    if config.llm_name.startswith("gpt"):
        return OpenAILLM(config)
    elif config.llm_name.startswith("claude"):
        return AnthropicLLM(config)
    else:
        return LocalLLM(config)
```

### Base Interface
```python
# src/llm/base.py
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def infer(self, messages: List[Dict]) -> Tuple[str, Dict]:
        """Synchronous inference"""
        pass

    @abstractmethod
    async def ainfer(self, messages: List[Dict]) -> Tuple[str, Dict]:
        """Asynchronous inference"""
        pass
```

### Response Caching
```python
# Use SHA256 hash as cache key
cache_key = hashlib.sha256(
    json.dumps({
        "messages": messages,
        "model": self.llm_name,
        "temperature": self.temperature
    }).encode()
).hexdigest()
```

---

## Prompt Management

Use `PromptTemplateManager` for centralized prompt management:

```python
# src/prompts/templates/kernel_generation.py
KERNEL_GENERATION_TEMPLATE = [
    ("system", "You are an expert at analyzing query intent and assigning edge type weights."),
    ("user", """Analyze the following query and assign weights (0.0-2.0) to each edge type:
- SEQ (sequential/temporal): For queries about order, timeline, process
- SIM (similarity/semantic): For queries about related concepts, analogies
- CAUSE (causal/logical): For queries about reasons, consequences, explanations

Query: {query}

Return JSON:
{{
    "weights": {{"SEQ": <float>, "SIM": <float>, "CAUSE": <float>}},
    "justification": "<explanation>"
}}""")
]
```

---

## Dataset Integration

### Dataset Format
Follow HippoRAG's format:

**Corpus** (`dataset/{name}_corpus.json`):
```json
[
  {"title": "Document Title", "text": "Document content...", "idx": 0}
]
```

**Queries** (`dataset/{name}.json`):
```json
[
  {
    "id": "q1",
    "question": "Query text?",
    "answer": ["Answer 1", "Answer 2"],
    "paragraphs": [
      {"title": "Supporting Doc", "text": "...", "is_supporting": true, "idx": 0}
    ]
  }
]
```

### Loading Code
```python
# Load corpus
with open(f"dataset/{dataset_name}_corpus.json") as f:
    corpus = json.load(f)
docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]

# Load queries
with open(f"dataset/{dataset_name}.json") as f:
    samples = json.load(f)
queries = [s['question'] for s in samples]
```

---

## Testing Strategy

### Unit Tests
```python
# tests/test_graph_store.py
def test_add_node():
    store = GraphStore()
    node = MemoryNode(id="test-1", content="Test", embedding=[0.1, 0.2])
    store.add_node(node)
    assert store.get_node("test-1") == node

def test_spreading_activation():
    # Test energy diffusion with known graph structure
    pass
```

### Integration Tests
```python
# tests/test_end_to_end.py
def test_full_pipeline():
    memory = NeuroGraphMemory(config)
    memory.add_document("Test document")
    results = memory.retrieve("Test query")
    assert len(results) > 0
```

---

## Success Criteria

1. **Parameterization**: No hard-coded thresholds or magic numbers in code
2. **Dynamic Behavior**: Same graph returns different results for different query types (e.g., "why" vs "what")
3. **Stability**: SIM edge weights remain constant, while unused SEQ edges decay over time
4. **Modularity**: Each layer (Storage/Ingestion/Retrieval/Plasticity) can be tested independently

---

## Implementation Order

1. **Phase 1: Foundation**
   - Create `MemoryConfig` class
   - Implement `GraphStore` with node/edge management
   - Reuse `EmbeddingStore` from HippoRAG (copy and adapt)
   - Implement `BaseLLM` interface and OpenAI implementation

2. **Phase 2: Ingestion**
   - Implement text chunking
   - Build SEQ edge construction
   - Build SIM edge construction (top-K neighbors, no threshold)
   - Build CAUSE edge construction (LLM-driven)

3. **Phase 3: Retrieval**
   - Implement `KernelGenerator` (LLM-driven)
   - Implement `SpreadingActivation` algorithm
   - Integrate vector search for anchor nodes

4. **Phase 4: Plasticity**
   - Implement `reinforce_path` (LTP)
   - Implement `decay_unused` (LTD with SIM exemption)

5. **Phase 5: Integration & Testing**
   - Create `NeuroGraphMemory` main class
   - Write unit tests for each component
   - Run end-to-end tests with sample dataset
   - Benchmark and optimize

---

## Code Style Guidelines

- Use type hints for all function signatures
- Use dataclasses for data structures
- Use enums for categorical values (EdgeType)
- Follow PEP 8 naming conventions
- Add docstrings to all public methods
- Keep functions focused and single-purpose
- Prefer composition over inheritance

---

## Dependencies

```
# requirements.txt
numpy>=1.24.0
scikit-learn>=1.3.0
openai>=1.0.0
python-igraph>=0.11.0
pandas>=2.0.0
tqdm>=4.65.0
pytest>=7.4.0
```

---

## Reference Materials

- **DESIGN.md**: Complete system specification (authoritative source)
- **refer/REFER.md**: Implementation patterns from HippoRAG
- **refer/HippoRAG/**: Production codebase to study

When implementing, always refer to DESIGN.md for requirements and refer/REFER.md for implementation patterns.
