# 配置文件使用指南

本项目支持**3种配置方式**，按优先级从高到低：

## 📋 配置方式

### 方式1: YAML配置文件（推荐）⭐

**优点**:
- 所有参数集中管理
- 支持注释
- 易于版本控制
- 结构清晰

**使用方法**:

1. 复制示例配置文件：
```bash
cp config.yaml my_config.yaml
```

2. 编辑配置文件，填入你的参数

3. 使用配置文件：
```python
from neurogated.utils.yaml_loader import config_from_yaml

# 加载配置
config = config_from_yaml("my_config.yaml")

# 使用配置
from neurogated import NeuroGraphMemory
memory = NeuroGraphMemory(config)
```

**命令行使用**:
```bash
# 修改 main.py 支持 --config 参数
uv run python main.py --config my_config.yaml
```

---

### 方式2: 环境变量 + .env文件

**优点**:
- 适合CI/CD环境
- 敏感信息（API Key）不入代码
- 灵活覆盖默认值

**使用方法**:

1. 复制示例文件：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，填入你的配置

3. 环境变量会自动加载：
```python
from neurogated import NeuroGraphMemory, MemoryConfig

# 自动从环境变量加载
config = MemoryConfig()
memory = NeuroGraphMemory(config)
```

**或者手动export**:
```bash
export OPENAI_API_KEY="your-key"
export TOP_K_ANCHORS=5
export MAX_HOPS=3

uv run python main.py
```

---

### 方式3: 代码直接配置

**优点**:
- 最直接
- 适合脚本和Jupyter Notebook
- 完全控制

**使用方法**:
```python
from neurogated import NeuroGraphMemory, MemoryConfig

config = MemoryConfig(
    llm_name="gpt-4o-mini",
    llm_base_url="https://your-api.com/v1",
    TOP_K_ANCHORS=5,
    TOP_N_RETRIEVAL=3,
    MAX_HOPS=2,
    save_dir="outputs/my_experiment"
)

memory = NeuroGraphMemory(config)
```

---

## 🔄 配置优先级

当多种配置方式同时存在时，优先级为：

```
代码参数 > 环境变量 > YAML配置文件 > 默认值
```

**示例**:
```python
# config.yaml 中设置 TOP_K_ANCHORS=5
# 环境变量设置 export TOP_K_ANCHORS=7
# 代码中设置 TOP_K_ANCHORS=10

config = config_from_yaml("config.yaml")  # 读取YAML: 5
# 但环境变量会覆盖: 7

config = MemoryConfig(TOP_K_ANCHORS=10)   # 代码参数最高优先级: 10
```

---

## 📝 所有可配置参数

### API配置
- `OPENAI_API_KEY` - OpenAI API密钥（必需）
- `LLM_NAME` - LLM模型名称
- `LLM_BASE_URL` - 自定义API URL
- `LLM_TEMPERATURE` - 采样温度
- `LLM_MAX_NEW_TOKENS` - 最大生成token数
- `EMBEDDING_MODEL_NAME` - Embedding模型名称
- `EMBEDDING_BATCH_SIZE` - 批处理大小

### 检索参数
- `TOP_K_ANCHORS` - 初始锚点数量（默认5）
- `TOP_N_RETRIEVAL` - 最终返回数量（默认3）
- `MAX_HOPS` - 最大扩散跳数（默认2）
- `ENERGY_DECAY_RATE` - 能量衰减率（默认0.5）
- `ANCHOR_ENERGY_DISTRIBUTION` - 初始能量分配（默认softmax）
- `MERGE_STRATEGY` - 合并策略（默认weighted_average）
- `MERGE_OLD_WEIGHT` - 旧能量权重（默认0.7）

### 图构建参数
- `MAX_SIM_NEIGHBORS_INTRA_DOC` - 文档内相似边数（默认3）
- `MAX_SIM_NEIGHBORS_INTER_DOC` - 跨文档相似边数（默认2）
- `CAUSE_WINDOW_SIZE` - 因果窗口大小（默认5）
- `CAUSE_NEIGHBOR_HOPS` - 因果邻居跳数（默认1）
- `USE_ENTITY_NODES` - 是否使用Entity节点（默认true）
- `USE_CHUNK_NODES` - 是否使用Chunk节点（默认true）

### 可塑性参数
- `HEBBIAN_LEARNING_RATE` - LTP学习率（默认0.1）
- `TIME_DECAY_FACTOR` - LTD衰减系数（默认0.99）
- `MIN_EDGE_WEIGHT` - 最小边权重（默认0.1）
- `LTP_ACTIVATION_THRESHOLD` - LTP激活阈值（默认0.05）

### 文本处理参数
- `CHUNK_SIZE` - 切片大小（默认512）
- `CHUNK_OVERLAP` - 切片重叠（默认128）

### 存储参数
- `SAVE_DIR` - 输出目录（默认outputs）
- `FORCE_INDEX_FROM_SCRATCH` - 强制重建（默认false）
- `CACHE_LLM_RESPONSES` - 缓存LLM响应（默认true）

### 系统参数
- `LOG_LEVEL` - 日志级别（默认INFO）
- `HF_HOME` - Hugging Face缓存目录
- `CUDA_VISIBLE_DEVICES` - CUDA设备ID

---

## 💡 最佳实践

### 开发环境
```yaml
# dev_config.yaml
api:
  llm:
    name: "gpt-4o-mini"
    temperature: 0.0

retrieval:
  top_k_anchors: 3
  max_hops: 1

storage:
  save_dir: "outputs/dev"
  cache_llm_responses: true

system:
  log_level: "DEBUG"
```

### 生产环境
```yaml
# prod_config.yaml
api:
  llm:
    name: "gpt-4o"
    temperature: 0.0

retrieval:
  top_k_anchors: 5
  max_hops: 3

storage:
  save_dir: "outputs/prod"
  cache_llm_responses: true

system:
  log_level: "INFO"
```

### 实验环境
```yaml
# experiment_config.yaml
api:
  llm:
    name: "gpt-4o-mini"

retrieval:
  top_k_anchors: 10      # 实验不同的值
  max_hops: 5
  energy_decay_rate: 0.3

storage:
  save_dir: "outputs/experiment_001"
```

---

## 🔒 安全建议

1. **不要提交敏感信息**:
```bash
# .gitignore 已包含
.env
config.yaml
my_config.yaml
*_config.yaml
```

2. **使用环境变量存储API Key**:
```bash
# 不要在YAML中硬编码API Key
export OPENAI_API_KEY="your-key"
```

3. **使用不同配置文件**:
```bash
config.yaml.example  # 提交到git（不含敏感信息）
config.yaml          # 本地使用（不提交）
```

---

## 📚 相关文档

- [RUN_GUIDE.md](RUN_GUIDE.md) - 运行指南
- [CUSTOM_API_GUIDE.md](CUSTOM_API_GUIDE.md) - 自定义API配置

---

**推荐使用 YAML 配置文件进行参数管理！** 🎯
