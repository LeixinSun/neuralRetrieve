# 运行指南 - 如何启动系统和运行数据集

本文档说明运行系统所需的电脑配置、环境设置和运行步骤。

---

## 💻 电脑配置要求

### 最低配置

| 组件 | 要求 | 说明 |
|------|------|------|
| **CPU** | 4核心 | 用于文本处理和图操作 |
| **内存** | 8GB | 小数据集（sample）可运行 |
| **硬盘** | 5GB可用空间 | 存储模型、数据集和输出 |
| **网络** | 稳定互联网连接 | 调用OpenAI API |
| **操作系统** | macOS / Linux / Windows | 支持Python 3.10+ |

### 推荐配置

| 组件 | 推荐 | 说明 |
|------|------|------|
| **CPU** | 8核心+ | 加速并行处理 |
| **内存** | 16GB+ | 运行大数据集（MuSiQue, HotpotQA） |
| **硬盘** | 20GB+ SSD | 更快的I/O |
| **GPU** | 可选 | 如果使用本地embedding模型 |

### 成本估算

**OpenAI API 成本**（以sample数据集为例）:
- **Embedding**: text-embedding-3-small
  - 约100个chunk × $0.00002/1K tokens ≈ $0.01
- **LLM调用**: gpt-4o-mini
  - NER: 约100次 × $0.00015/1K tokens ≈ $0.05
  - Kernel生成: 约10次 × $0.00015/1K tokens ≈ $0.01
  - 因果检测: 约50次 × $0.00015/1K tokens ≈ $0.03
- **总计**: 约 $0.10（sample数据集）

**大数据集**（MuSiQue, 2000+文档）:
- 预计成本: $5-10

**优化建议**:
- 使用LLM缓存（已实现）避免重复调用
- 跳过CAUSE边构建（可选）减少LLM调用
- 使用本地embedding模型（需要GPU）

---

## 🔧 环境设置

### 1. 安装 Python 和 uv

```bash
# 检查Python版本（需要3.10+）
python --version

# 安装uv（如果未安装）
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 验证安装
uv --version
```

### 2. 克隆/下载项目

```bash
cd /path/to/your/workspace
# 如果是git仓库
git clone <repository-url>
cd referHippoNeural

# 或者直接使用现有目录
cd /Users/lx/Documents/referHippoNeural
```

### 3. 安装依赖

```bash
# 使用uv安装所有依赖
uv sync

# 这会自动：
# 1. 创建虚拟环境
# 2. 安装Python 3.10
# 3. 安装所有依赖包
```

### 4. 配置系统

**方式1: 使用 config.yaml（推荐）**

编辑项目根目录的 `config.yaml` 文件：

```yaml
# config.yaml
api:
  # 设置OpenAI API Key（必需）
  openai_api_key: "your-api-key-here"

  # LLM配置
  llm:
    name: "gpt-4o-mini"
    base_url: null  # 使用自定义API时设置，例如: http://localhost:8000/v1
    temperature: 0.0
    max_new_tokens: 2048

  # Embedding配置
  embedding:
    model_name: "text-embedding-3-small"
    base_url: null  # 使用自定义embedding API时设置

# 其他配置参数...
```

**方式2: 使用环境变量**

```bash
# 设置OpenAI API Key（必需）
export OPENAI_API_KEY="your-api-key-here"

# 可选：设置其他环境变量
export HF_HOME="/path/to/huggingface/cache"  # Hugging Face缓存目录
export CUDA_VISIBLE_DEVICES="0"              # 如果有GPU
```

**永久设置环境变量**:

```bash
# macOS/Linux - 添加到 ~/.bashrc 或 ~/.zshrc
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.zshrc
source ~/.zshrc

# Windows - 系统环境变量
setx OPENAI_API_KEY "your-api-key-here"
```

**使用自定义 API**:

如果您想使用 Azure OpenAI、本地部署的模型或其他兼容 OpenAI 的服务：
1. 在 `config.yaml` 中设置 `api.llm.base_url`
2. 或使用命令行参数 `--llm_base_url`
3. 详细说明请参考 [CUSTOM_API_GUIDE.md](CUSTOM_API_GUIDE.md)

**配置优先级**:
1. 命令行参数（最高优先级）
2. config.yaml 配置
3. 默认值（最低优先级）

### 5. 准备数据集

```bash
# 示例数据集已包含
ls dataset/
# 应该看到：
# sample_corpus.json
# sample.json

# 如果要使用其他数据集，从HippoRAG复制
cp refer/HippoRAG/reproduce/dataset/musique* dataset/
```

---

## 🚀 运行步骤

### 方式1: 基础测试（推荐首次运行）

```bash
# 运行基础测试，验证系统工作
uv run python test_basic.py
```

**预期输出**:
```
INFO - Testing Neuro-Gated Graph Memory System
INFO - 1. Initializing system...
INFO - 2. Adding document...
INFO -    Document added: {'chunk_nodes_created': 3, 'entity_nodes_created': 5, ...}
INFO - 3. Graph statistics:
INFO -    total_nodes: 8
INFO -    chunk_nodes: 3
INFO -    entity_nodes: 5
INFO - 4. Testing retrieval...
INFO -    Query: What is the capital of France?
INFO -    Retrieved 2 results
INFO - 5. Saving system...
INFO - ✅ Test passed!
```

**运行时间**: 约30-60秒
**成本**: 约$0.02

### 方式2: 运行示例数据集

```bash
# 运行sample数据集（10个文档，1个查询）
# 系统会自动加载 config.yaml 中的配置
uv run python main.py --dataset sample

# 或者使用自定义配置文件
uv run python main.py --dataset sample --config my_config.yaml

# 或者通过命令行覆盖config.yaml中的参数
uv run python main.py --dataset sample --llm_name gpt-4o --top_k_anchors 10
```

**预期输出**:
```
================================================================================
Neuro-Gated Graph Memory System - Experiment Runner
================================================================================
Loaded configuration from config.yaml

1. Initializing system...
2. Loading dataset...
   Loaded 10 documents from dataset/sample_corpus.json
   Loaded 1 queries from dataset/sample.json

3. Indexing documents...
   Indexing document 1/10: doc_0
   ...
   Indexing document 10/10: doc_9

4. Graph statistics:
   total_nodes: 45
   chunk_nodes: 30
   entity_nodes: 15
   total_edges: 120

5. Running retrieval...
   Query 1/1: Which Stanford University professor works on Alzheimer's?
   Retrieved 3 chunks

6. Evaluating retrieval...
   Retrieval Recall: 0.8500

7. Running maintenance...
8. Saving system...

================================================================================
Experiment completed!
================================================================================
```

**运行时间**: 约5-10分钟
**成本**: 约$0.10

### 方式3: 自定义配置运行

有三种方式自定义配置：

**方式A: 编辑 config.yaml（推荐）**

```yaml
# config.yaml
retrieval:
  top_k_anchors: 10
  top_n_retrieval: 5
  max_hops: 3

storage:
  save_dir: "outputs/my_experiment"
```

然后运行：
```bash
uv run python main.py --dataset sample
```

**方式B: 使用命令行参数覆盖**

```bash
# 命令行参数会覆盖 config.yaml 中的设置
uv run python main.py \
    --dataset sample \
    --llm_name gpt-4o-mini \
    --embedding_name text-embedding-3-small \
    --top_k_anchors 5 \
    --top_n_retrieval 3 \
    --max_hops 2 \
    --save_dir outputs/my_experiment
```

**方式C: 使用自定义配置文件**

```bash
# 创建自定义配置文件
cp config.yaml my_config.yaml
# 编辑 my_config.yaml...

# 使用自定义配置
uv run python main.py --config my_config.yaml --dataset sample
```

### 方式4: 使用本地vLLM或自定义API

**方式A: 在 config.yaml 中配置**

```yaml
# config.yaml
api:
  llm:
    name: "meta-llama/Llama-3.3-70B-Instruct"
    base_url: "http://localhost:8000/v1"
```

然后运行：
```bash
uv run python main.py --dataset sample
```

**方式B: 使用命令行参数**

```bash
# 确保数据集文件存在
ls dataset/musique*

# 运行（需要更多时间和成本）
uv run python main.py \
    --dataset musique \
    --llm_name meta-llama/Llama-3.3-70B-Instruct \
    --llm_base_url http://localhost:8000/v1 \
    --top_n_retrieval 5 \
    --max_hops 3
```

详细的自定义API配置说明请参考 [CUSTOM_API_GUIDE.md](CUSTOM_API_GUIDE.md)。

**运行时间**: 约1-2小时
**成本**: 约$5-10

---

## 📊 输出说明

### 输出目录结构

```
outputs/
└── sample/                          # 数据集名称
    ├── chunk_embeddings/            # Chunk向量存储
    │   └── vdb_chunk.parquet
    ├── entity_embeddings/           # Entity向量存储
    │   └── vdb_entity.parquet
    ├── graph.pickle                 # 图结构
    ├── mappings.pkl                 # 节点和边映射
    └── llm_cache.db                 # LLM响应缓存
```

### 日志说明

**INFO级别**: 正常流程信息
```
INFO - Indexing document 1/10: doc_0
INFO - Retrieved 3 chunks
```

**DEBUG级别**: 详细调试信息（需要设置log_level=DEBUG）
```
DEBUG - Extracted 5 entities from text
DEBUG - Built 12 SIM edges
```

**WARNING级别**: 警告信息
```
WARNING - Failed to parse NER response
WARNING - Cache read error
```

**ERROR级别**: 错误信息
```
ERROR - Failed to index document: ...
ERROR - Retrieval failed: ...
```

---

## 🔧 常见问题排查

### 问题1: OpenAI API错误

**错误信息**:
```
ERROR - OpenAI API error: Incorrect API key provided
```

**解决方案**:

**方式A: 在 config.yaml 中设置**
```yaml
# config.yaml
api:
  openai_api_key: "sk-your-actual-key-here"
```

**方式B: 使用环境变量**
```bash
# 检查环境变量
echo $OPENAI_API_KEY

# 重新设置
export OPENAI_API_KEY="sk-..."

# 验证
uv run python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
```

### 问题2: 找不到数据集

**错误信息**:
```
ERROR - Dataset not found: dataset/sample_corpus.json
```

**解决方案**:
```bash
# 检查数据集目录
ls dataset/

# 复制示例数据集
cp refer/HippoRAG/reproduce/dataset/sample* dataset/

# 或创建软链接
ln -s refer/HippoRAG/reproduce/dataset dataset
```

### 问题3: 内存不足

**错误信息**:
```
MemoryError: Unable to allocate array
```

**解决方案**:

**方式A: 在 config.yaml 中调整**
```yaml
# config.yaml
api:
  embedding:
    batch_size: 8  # 默认32，减小到8

text_processing:
  chunking:
    chunk_size: 256  # 默认512，减小到256
```

**方式B: 使用命令行参数**
```bash
# 暂不支持通过命令行调整这些参数
# 请使用方式A修改 config.yaml
```

### 问题4: 依赖安装失败

**错误信息**:
```
ERROR - Failed to install python-igraph
```

**解决方案**:
```bash
# macOS - 安装系统依赖
brew install igraph

# Ubuntu/Debian
sudo apt-get install libigraph-dev

# 然后重新安装
uv sync
```

### 问题5: 运行速度慢

**优化建议**:

1. **跳过CAUSE边构建**（最耗时）:
   ```yaml
   # config.yaml
   advanced:
     optimization:
       skip_cause_edges: true  # 跳过因果边构建
   ```

2. **减少LLM调用**:
   ```yaml
   # config.yaml
   graph:
     nodes:
       use_entity_nodes: false  # 跳过实体提取
   ```

3. **使用缓存**:
   - 第二次运行会使用LLM缓存，速度更快
   - 不要设置 `force_index_from_scratch=True`
   - 缓存文件位于 `outputs/{dataset}/llm_cache.db`

4. **并行处理**（待实现）:
   - 目前是串行处理
   - 可以修改代码支持多进程

---

## 📈 性能基准

### Sample数据集（10文档）

| 阶段 | 时间 | LLM调用 | 成本 |
|------|------|---------|------|
| 索引 | 3-5分钟 | ~150次 | $0.08 |
| 检索 | 10-20秒 | ~10次 | $0.02 |
| 总计 | 5-10分钟 | ~160次 | $0.10 |

### MuSiQue数据集（2000+文档）

| 阶段 | 时间 | LLM调用 | 成本 |
|------|------|---------|------|
| 索引 | 1-2小时 | ~30000次 | $8-10 |
| 检索 | 5-10分钟 | ~500次 | $1-2 |
| 总计 | 1.5-2.5小时 | ~30500次 | $9-12 |

**注意**:
- 使用缓存后，重复运行几乎无成本
- 跳过CAUSE边可减少50%的LLM调用

---

## 🎯 快速开始检查清单

- [ ] Python 3.10+ 已安装
- [ ] uv 已安装
- [ ] 项目依赖已安装 (`uv sync`)
- [ ] config.yaml 已配置（特别是 API key）
- [ ] 数据集文件已准备
- [ ] 运行基础测试成功
- [ ] 运行示例数据集成功

---

## ⚙️ 配置文件说明

系统使用 `config.yaml` 作为主配置文件，包含所有可调节的参数。

### 配置文件结构

```yaml
api:                    # API配置
  openai_api_key: "..."
  llm:                  # LLM配置
    name: "gpt-4o-mini"
    base_url: null      # 自定义API URL
  embedding:            # Embedding配置
    model_name: "text-embedding-3-small"

retrieval:              # 检索参数
  top_k_anchors: 5
  top_n_retrieval: 3
  max_hops: 2

graph:                  # 图构建参数
  sim_edges:
    max_neighbors_intra_doc: 3
  cause_edges:
    window_size: 5

storage:                # 存储配置
  save_dir: "outputs"
  force_index_from_scratch: false

system:                 # 系统配置
  log_level: "INFO"
  verbose: false
```

### 查看完整配置

```bash
# 查看配置文件
cat config.yaml

# 查看配置说明
cat CONFIG_GUIDE.md
```

### 配置优先级

1. **命令行参数** - 最高优先级
   ```bash
   uv run python main.py --llm_name gpt-4o --top_k_anchors 10
   ```

2. **config.yaml** - 中等优先级
   ```yaml
   api:
     llm:
       name: "gpt-4o-mini"
   ```

3. **默认值** - 最低优先级（在 `src/neurogated/config/memory_config.py` 中定义）

---

## 📞 获取帮助

如果遇到问题：

1. **检查日志**: 查看详细的错误信息
2. **查看文档**: README.md, USAGE.md, FULFILL.md
3. **检查配置**: 确认所有参数正确
4. **验证环境**: 确认依赖和API key正确

---

## 🚀 下一步

成功运行后，可以：

1. **尝试不同配置**: 调整超参数，观察效果
2. **运行大数据集**: MuSiQue, HotpotQA
3. **分析结果**: 查看图统计、检索结果
4. **优化性能**: 调整批处理大小、跳过CAUSE边
5. **扩展功能**: 添加新的边类型、评估指标

---

**祝运行顺利！** 🎉
