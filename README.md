# Neuro-Gated Graph Memory System

神经拟态图记忆系统 - 一个受神经科学启发的检索机制，实现激活扩散和动态边权重调制。

---

## 📚 快速导航

| 文档 | 用途 | 适合人群 |
|------|------|----------|
| **[README.md](README.md)** | 项目概述和快速开始 | 所有人 |
| **[RUN_GUIDE.md](RUN_GUIDE.md)** | 完整运行指南 | 想运行系统的用户 |
| **[CUSTOM_API_GUIDE.md](CUSTOM_API_GUIDE.md)** | 自定义API配置 | 使用Azure/本地模型的用户 |
| **[FULFILL.md](FULFILL.md)** | 代码实现清单 | 想了解实现的开发者 |
| **[DESIGN.md](DESIGN.md)** | 系统设计规范 | 想了解设计的开发者 |
| **[CLAUDE.md](CLAUDE.md)** | AI开发指南 | AI助手和开发者 |

---

## 🎉 项目状态

**✅ 核心系统已完成** - 所有模块已实现，系统完全可运行。

### 已实现的功能

- ✅ 配置系统 - 完整的超参数配置
- ✅ 数据结构 - MemoryNode + MemoryEdge
- ✅ 存储层 - GraphStore + EmbeddingStore
- ✅ LLM接口 - OpenAI实现（带缓存，支持自定义API）
- ✅ 摄入层 - 文档切片、NER、边构建
- ✅ 检索层 - 激活扩散算法
- ✅ 可塑性层 - LTP/LTD实现
- ✅ 评估层 - 数据集加载和指标计算
- ✅ 主类 - 完整的系统集成

---

## 🚀 快速开始

### 1. 安装

```bash
# 使用 uv 管理项目
uv sync
```

### 2. 配置

**方式A: 编辑 config.yaml（推荐）**

```yaml
# config.yaml
api:
  openai_api_key: "your-api-key-here"
  llm:
    name: "gpt-4o-mini"
    base_url: null  # 可选：自定义API URL
```

**方式B: 使用环境变量**

```bash
export OPENAI_API_KEY=your_api_key_here
```

### 3. 验证安装

```bash
# 验证环境配置
uv run python verify_setup.py

# 测试导入（无需API key）
uv run python test_imports.py

# 完整测试（需要API key）
uv run python test_basic.py
```

### 4. 运行

```bash
# 运行数据集实验
uv run python main.py --dataset sample

# 使用自定义 API（Azure OpenAI, 本地 vLLM 等）
uv run python main.py \
    --dataset sample \
    --llm_base_url "http://localhost:8000/v1" \
    --llm_name "meta-llama/Llama-3.3-70B-Instruct"
```

**详细说明**:
- 运行指南 → [RUN_GUIDE.md](RUN_GUIDE.md)
- API配置 → [CUSTOM_API_GUIDE.md](CUSTOM_API_GUIDE.md)
- 配置说明 → [CONFIG_GUIDE.md](CONFIG_GUIDE.md)

---

## 💡 核心特性

### 1. 记忆即图谱
信息存储为节点（Chunk + Entity），通过有类型的边（SEQ/SIM/CAUSE）相互连接。

### 2. Query 即卷积核
用户查询生成动态的"核"，在运行时重新调制不同类型边的权重：
- **SEQ** (时序): 用于"接下来发生了什么"类问题
- **SIM** (语义): 用于"什么与X相似"类问题
- **CAUSE** (因果): 用于"为什么"、"如何影响"类问题

### 3. 检索即能量流动
检索是能量从初始锚点在图谱中迭代扩散的过程：
```
Query → Softmax初始化 → 多跳扩散 → 加权合并 → Top-N结果
```

### 4. 可塑性
图谱结构通过 Hebbian 学习（LTP）自我强化，同时支持基于时间的被动遗忘（LTD）。

---

## 📖 使用示例

### 基础使用

```python
from neurogated import NeuroGraphMemory, MemoryConfig, config_from_yaml

# 方式1: 从 config.yaml 加载（推荐）
config = config_from_yaml("config.yaml")

# 方式2: 直接创建配置
config = MemoryConfig(
    save_dir="outputs/demo",
    llm_name="gpt-4o-mini",
    TOP_K_ANCHORS=5,
    TOP_N_RETRIEVAL=3,
    MAX_HOPS=2
)

# 初始化系统
memory = NeuroGraphMemory(config)

# 添加文档
memory.add_document(
    "Paris is the capital of France. The Eiffel Tower was built in 1889.",
    document_id="doc_1"
)

# 检索
results = memory.retrieve("What is the capital of France?")
print(results)

# 保存
memory.save()
```

### 使用 config.yaml 配置

```yaml
# config.yaml
api:
  openai_api_key: "your-key"
  llm:
    name: "gpt-4o-mini"
    base_url: "http://localhost:8000/v1"  # 可选：自定义API

retrieval:
  top_k_anchors: 5
  top_n_retrieval: 3
  max_hops: 2
```

```python
from neurogated import config_from_yaml, NeuroGraphMemory

config = config_from_yaml("config.yaml")
memory = NeuroGraphMemory(config)
```

### 运行数据集实验

```bash
# 运行 sample 数据集（使用 config.yaml 配置）
uv run python main.py --dataset sample

# 使用命令行覆盖配置
uv run python main.py \
    --dataset sample \
    --top_k_anchors 5 \
    --top_n_retrieval 3 \
    --max_hops 2

# 使用自定义配置文件
uv run python main.py --config my_config.yaml --dataset sample
```

---

## 🏗️ 架构

```
src/neurogated/
├── config/           # 配置管理（MemoryConfig）
├── storage/          # 图存储 + 向量存储
├── llm/              # LLM 接口（OpenAI + 缓存）
├── prompts/          # 提示词管理
├── retrieval/        # 检索引擎（KernelGenerator + SpreadingActivation）
├── plasticity/       # 可塑性引擎（LTP/LTD）
├── ingestion/        # 摄入层（Chunking + NER + 边构建）
├── evaluation/       # 评估指标
└── core.py           # 主类（NeuroGraphMemory）
```

**详细实现**: 参考 [FULFILL.md](FULFILL.md)

---

## 💻 系统要求

### 最低配置
- **CPU**: 4核心
- **内存**: 8GB
- **硬盘**: 5GB
- **网络**: 稳定连接（OpenAI API）

### 推荐配置
- **CPU**: 8核心+
- **内存**: 16GB+
- **硬盘**: 20GB+ SSD

### 成本估算
- **Sample数据集**: $0.10, 5-10分钟
- **MuSiQue数据集**: $9-12, 1.5-2.5小时

**详细要求**: 参考 [RUN_GUIDE.md](RUN_GUIDE.md)

---

## 🆚 与 HippoRAG 的区别

| 特性 | HippoRAG | 神经拟态系统 |
|------|----------|-------------|
| 检索算法 | PPR (PageRank) | 激活扩散 |
| 边权重 | 静态 | 动态调制 |
| Query 处理 | 固定策略 | LLM 生成核 |
| 可塑性 | 无 | LTP/LTD |
| 节点类型 | Chunk + Entity + Fact | Chunk + Entity |

---

## 📊 性能特点

- **LLM 缓存**: SQLite 缓存避免重复调用
- **批量处理**: 向量编码和 LLM 调用批处理
- **增量索引**: 哈希ID去重，支持增量添加
- **稀疏图**: 限制邻居数，保持图稀疏

---

## 🔄 更新日志

### v1.0.0 (2024)
- ✅ 完整实现所有核心模块
- ✅ 支持自定义 OpenAI API Base URL
- ✅ 完整的文档体系
- ✅ 数据集集成和评估

---

## 📚 参考

本项目参考了 [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) 的实现模式，但采用了不同的检索机制和可塑性设计。

---

## 📄 License

MIT

---

## 🎯 下一步

成功运行后，可以：

1. **尝试不同配置**: 调整超参数，观察效果
2. **运行大数据集**: MuSiQue, HotpotQA
3. **分析结果**: 查看图统计、检索结果
4. **优化性能**: 调整批处理大小、跳过CAUSE边
5. **扩展功能**: 添加新的边类型、评估指标

---

**开始使用**: 参考 [RUN_GUIDE.md](RUN_GUIDE.md) 🚀
