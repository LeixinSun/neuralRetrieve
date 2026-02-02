# Configuration System Summary

## Changes Made

### 1. Updated `src/neurogated/__init__.py`
- Added `config_from_yaml` to exports
- Now users can import: `from neurogated import config_from_yaml`

### 2. Updated `main.py`
- **Now loads from config.yaml by default**
- Added `--config` argument to specify config file path
- Command-line arguments now **override** config.yaml values
- Priority: Command-line args > config.yaml > defaults

### 3. Updated `config.yaml`
- Enhanced comment for `base_url` field with example
- Already contains `base_url` setting at line 12

## How to Use

### Method 1: Use config.yaml (Recommended)

Edit `config.yaml` and set your base_url:

```yaml
api:
  llm:
    name: "gpt-4o-mini"
    base_url: "http://localhost:8000/v1"  # Your custom endpoint
```

Then run:
```bash
# Using uv (recommended for this project)
uv run python main.py --dataset sample

# Or using regular python
python main.py --dataset sample
```

### Method 2: Override via Command Line

```bash
# Using uv
uv run python main.py --dataset sample --llm_base_url http://localhost:8000/v1

# Or using regular python
python main.py --dataset sample --llm_base_url http://localhost:8000/v1
```

### Method 3: Use Custom Config File

```bash
# Using uv
uv run python main.py --config my_config.yaml --dataset sample

# Or using regular python
python main.py --config my_config.yaml --dataset sample
```

### Method 4: Programmatic Usage

```python
from neurogated import config_from_yaml, MemoryConfig

# Load from YAML
config = config_from_yaml("config.yaml")

# Override specific values
config.llm_base_url = "http://localhost:8000/v1"
config.llm_name = "meta-llama/Llama-3.3-70B-Instruct"

# Use the config
memory = NeuroGraphMemory(config)
```

## Configuration Priority

1. **Command-line arguments** (highest priority)
2. **config.yaml** values
3. **Default values** in MemoryConfig (lowest priority)

## All Available Settings in config.yaml

### API Configuration
- `api.openai_api_key`: Your OpenAI API key
- `api.llm.name`: LLM model name
- `api.llm.base_url`: Custom LLM endpoint (null = use OpenAI default)
- `api.llm.temperature`: Sampling temperature
- `api.llm.max_new_tokens`: Max tokens to generate
- `api.embedding.model_name`: Embedding model name
- `api.embedding.base_url`: Custom embedding endpoint

### Retrieval Parameters
- `retrieval.top_k_anchors`: Number of initial anchor nodes (default: 5)
- `retrieval.top_n_retrieval`: Final results to return (default: 3)
- `retrieval.max_hops`: Maximum diffusion hops (default: 2)
- `retrieval.energy_decay_rate`: Energy decay per hop (default: 0.5)
- `retrieval.anchor_energy_distribution`: Energy distribution method
- `retrieval.merge_strategy`: Activation merge strategy
- `retrieval.merge_old_weight`: Weight for old activations

### Graph Construction
- `graph.sim_edges.max_neighbors_intra_doc`: Max SIM edges within doc
- `graph.sim_edges.max_neighbors_inter_doc`: Max SIM edges across docs
- `graph.cause_edges.window_size`: Sliding window for CAUSE detection
- `graph.cause_edges.neighbor_hops`: Graph neighbor expansion hops
- `graph.nodes.use_entity_nodes`: Enable entity nodes
- `graph.nodes.use_chunk_nodes`: Enable chunk nodes

### Plasticity
- `plasticity.ltp.learning_rate`: Hebbian learning rate
- `plasticity.ltp.activation_threshold`: Min energy for LTP
- `plasticity.ltd.time_decay_factor`: Time decay coefficient
- `plasticity.ltd.min_edge_weight`: Min weight before pruning

### Storage
- `storage.save_dir`: Output directory
- `storage.force_index_from_scratch`: Force rebuild
- `storage.cache_llm_responses`: Enable LLM response caching

### System
- `system.log_level`: Logging level (DEBUG/INFO/WARNING/ERROR)
- `system.verbose`: Detailed output
- `system.hf_home`: Hugging Face cache directory
- `system.cuda_visible_devices`: CUDA device IDs

## Verification

To verify the configuration is loaded correctly:

```bash
# Check if config.yaml exists
ls -la config.yaml

# Run test script using uv
uv run python test_config_structure.py

# Run with verbose logging
uv run python main.py --dataset sample --verbose
```

The system will log: "Loaded configuration from config.yaml" if successful.

## Example: Using Local vLLM

Edit `config.yaml`:

```yaml
api:
  llm:
    name: "meta-llama/Llama-3.3-70B-Instruct"
    base_url: "http://localhost:8000/v1"
```

Or use command line:

```bash
# Using uv
uv run python main.py \
  --dataset sample \
  --llm_name meta-llama/Llama-3.3-70B-Instruct \
  --llm_base_url http://localhost:8000/v1

# Or using regular python
python main.py \
  --dataset sample \
  --llm_name meta-llama/Llama-3.3-70B-Instruct \
  --llm_base_url http://localhost:8000/v1
```

## Files Modified

1. `src/neurogated/__init__.py` - Added config_from_yaml export
2. `main.py` - Now loads from config.yaml with CLI override support
3. `config.yaml` - Enhanced base_url documentation

## Files Already Existing

- `src/neurogated/utils/yaml_loader.py` - YAML loading logic (already implemented)
- `src/neurogated/config/memory_config.py` - Configuration dataclass (already implemented)
- `config.yaml` - Configuration file with all parameters (already created)
