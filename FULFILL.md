# FULFILL.md - 代码实现清单

本文档基于实际代码结构，说明系统真实实现了什么功能。

---

## 📁 项目结构

```
referHippoNeural/
├── src/neurogated/              # 核心代码
│   ├── config/                  # 配置模块
│   ├── storage/                 # 存储模块
│   ├── llm/                     # LLM接口模块
│   ├── prompts/                 # 提示词模块
│   ├── ingestion/               # 摄入模块
│   ├── retrieval/               # 检索模块
│   ├── plasticity/              # 可塑性模块
│   ├── evaluation/              # 评估模块
│   ├── embedding_model/         # 嵌入模型（复用HippoRAG）
│   ├── utils/                   # 工具函数
│   ├── data_structures.py       # 数据结构定义
│   ├── core.py                  # 主类
│   └── __init__.py              # 包初始化
├── dataset/                     # 数据集目录
├── outputs/                     # 输出目录
├── main.py                      # 主程序入口
├── demo.py                      # 演示脚本
├── test_basic.py                # 基础测试
└── pyproject.toml               # 项目配置

文档：
├── README.md                    # 项目说明
├── USAGE.md                     # 使用指南
├── DESIGN.md                    # 设计文档
├── CLAUDE.md                    # 开发指南
├── PROGRESS.md                  # 进度追踪
├── COMPLETION_SUMMARY.md        # 完成总结
└── refer/REFER.md               # HippoRAG参考
```

---

## 🔧 已实现的功能模块

### 1. 配置模块 (`config/memory_config.py`)

**实现内容**:
- `MemoryConfig` 类：包含所有系统超参数
  - 检索参数：TOP_K_ANCHORS, TOP_N_RETRIEVAL, MAX_HOPS, ENERGY_DECAY_RATE
  - 图构建参数：MAX_SIM_NEIGHBORS_INTRA_DOC, MAX_SIM_NEIGHBORS_INTER_DOC, CAUSE_WINDOW_SIZE
  - 可塑性参数：HEBBIAN_LEARNING_RATE, TIME_DECAY_FACTOR, MIN_EDGE_WEIGHT
  - LLM参数：llm_name, llm_base_url, llm_temperature, llm_max_new_tokens
  - 嵌入参数：embedding_model_name, embedding_batch_size, embedding_dimension
  - 存储参数：save_dir, force_index_from_scratch, cache_llm_responses
  - 文本处理参数：chunk_size, chunk_overlap
- 参数验证：`__post_init__` 方法验证参数合法性
- 配置导出：`to_dict()` 方法

**真实功能**:
- 提供统一的配置接口
- 所有"魔术数字"都可配置
- 支持参数验证，防止非法值

---

### 2. 数据结构模块 (`data_structures.py`)

**实现内容**:
- `NodeType` 枚举：CHUNK（文档块）、ENTITY（实体）
- `EdgeType` 枚举：SEQ（时序）、SIM（相似）、CAUSE（因果）
- `MemoryNode` 类：
  - 字段：id, node_type, content, embedding, base_energy, last_accessed, metadata
  - 方法：to_dict(), from_dict()（序列化）
- `MemoryEdge` 类：
  - 字段：source_id, target_id, edge_type, weight, created_at, last_activated, activation_count, metadata
  - 方法：to_dict(), from_dict(), activate()
- `RetrievalResult` 类：
  - 字段：node_ids, nodes, scores, metadata
  - 方法：get_chunks(), get_entities(), to_dict()

**真实功能**:
- 定义图中节点和边的数据结构
- 支持两种节点类型和三种边类型
- 提供序列化/反序列化功能
- 追踪边的激活历史

---

### 3. 存储模块 (`storage/`)

#### 3.1 GraphStore (`storage/graph_store.py`)

**实现内容**:
- 基于 `python-igraph` 的图存储
- 方法：
  - `add_node(node)`: 添加节点到图
  - `add_edge(edge)`: 添加边到图
  - `get_node(node_id)`: 获取节点
  - `get_neighbors(node_id, edge_type)`: 获取邻居节点
  - `vector_search(query_embedding, node_type, top_k)`: 向量搜索
  - `get_all_edges(edge_type)`: 获取所有边
  - `remove_edge(source_id, target_id, edge_type)`: 删除边
  - `save_graph()`: 保存图到磁盘
  - `_load_graph()`: 从磁盘加载图
  - `get_stats()`: 获取图统计信息
- 数据结构：
  - `self.graph`: igraph.Graph 对象
  - `self.nodes`: Dict[str, MemoryNode]
  - `self.edges`: Dict[Tuple, MemoryEdge]

**真实功能**:
- 管理图的节点和边
- 支持向量搜索（KNN）
- 持久化到磁盘（Pickle格式）
- 提供图统计信息

#### 3.2 EmbeddingStore (`storage/embedding_store.py`)

**实现内容**:
- 复用自 HippoRAG
- 使用 Parquet 格式存储向量
- 方法：
  - `insert_strings(texts)`: 批量插入文本并编码
  - `get_missing_string_hash_ids(texts)`: 获取缺失的文本
  - `delete(hash_ids)`: 删除向量
  - `get_row(hash_id)`: 获取向量行
  - `get_hash_id(text)`: 获取文本的哈希ID
- 数据结构：
  - `self.hash_ids`: 哈希ID列表
  - `self.texts`: 文本列表
  - `self.embeddings`: 向量列表
  - `self.hash_id_to_idx`: 哈希到索引的映射

**真实功能**:
- 存储和管理向量
- 使用MD5哈希去重
- 支持批量编码
- Parquet格式高效存储

---

### 4. LLM接口模块 (`llm/`)

#### 4.1 BaseLLM (`llm/base.py`)

**实现内容**:
- 抽象基类，定义LLM接口
- 方法：
  - `infer(messages, **kwargs)`: 同步推理（抽象方法）
  - `ainfer(messages, **kwargs)`: 异步推理（抽象方法）
  - `batch_infer(messages_list, **kwargs)`: 批量推理

**真实功能**:
- 定义统一的LLM接口
- 支持同步/异步/批量调用

#### 4.2 OpenAILLM (`llm/openai_llm.py`)

**实现内容**:
- OpenAI API 实现
- SQLite 缓存机制：
  - `_init_cache()`: 初始化缓存数据库
  - `_compute_cache_key()`: 计算缓存键（SHA256）
  - `_get_from_cache()`: 从缓存读取
  - `_save_to_cache()`: 保存到缓存
- 方法：
  - `infer(messages, **kwargs)`: 调用OpenAI API，带缓存
  - `ainfer(messages, **kwargs)`: 异步调用（待实现）

**真实功能**:
- 调用OpenAI API
- 使用SQLite缓存响应，避免重复调用
- 缓存键基于消息、模型、温度等参数
- 返回响应文本和元数据

#### 4.3 工厂函数 (`llm/__init__.py`)

**实现内容**:
- `get_llm(config)`: 根据配置返回LLM实例

**真实功能**:
- 根据llm_name选择LLM实现
- 目前支持OpenAI

---

### 5. 提示词模块 (`prompts/prompt_manager.py`)

**实现内容**:
- `PromptTemplateManager` 类
- 模板：
  - `_ner_template(passage)`: 命名实体识别
  - `_kernel_generation_template(query)`: 生成调制核
  - `_causal_detection_template(text_a, text_b)`: 因果关系检测
- 方法：
  - `render(name, **kwargs)`: 渲染模板

**真实功能**:
- 管理所有提示词模板
- 返回格式化的消息列表
- 支持变量替换

---

### 6. 摄入模块 (`ingestion/`)

#### 6.1 TextChunker (`ingestion/chunker.py`)

**实现内容**:
- 基于 tiktoken 的文本切片
- 方法：
  - `chunk(text, document_id)`: 切分文本
- 参数：
  - chunk_size: 每块token数
  - chunk_overlap: 重叠token数

**真实功能**:
- 将长文本切分为小块
- 支持重叠（保持上下文连续性）
- 返回(chunk_text, metadata)列表

#### 6.2 EntityExtractor (`ingestion/entity_extractor.py`)

**实现内容**:
- LLM-based 命名实体识别
- 方法：
  - `extract(text)`: 提取实体
  - `batch_extract(texts)`: 批量提取
  - `_parse_ner_response(response)`: 解析LLM响应

**真实功能**:
- 调用LLM提取命名实体
- 返回格式："EntityName (TYPE)"
- 支持批量处理
- 自动去重

#### 6.3 EdgeBuilder (`ingestion/edge_builder.py`)

**实现内容**:
- 构建三种类型的边
- 方法：
  - `build_seq_edges(chunk_nodes)`: 构建SEQ边
  - `build_sim_edges(nodes, existing_nodes)`: 构建SIM边
  - `build_cause_edges(nodes, existing_nodes)`: 构建CAUSE边
  - `_build_sim_edges_for_candidates()`: SIM边构建辅助
  - `_build_cause_edges_window()`: 滑动窗口CAUSE边
  - `_detect_causality()`: LLM检测因果关系
  - `_parse_causality_response()`: 解析因果响应

**真实功能**:
- **SEQ边**: 连接相邻的chunk，权重1.0
- **SIM边**:
  - 分层策略：文档内top-K1，跨文档top-K2
  - 权重=余弦相似度
  - 只连接相同类型节点
- **CAUSE边**:
  - 滑动窗口策略
  - LLM判断因果关系
  - 双向边，权重=置信度

#### 6.4 IngestionEngine (`ingestion/ingestion_engine.py`)

**实现内容**:
- 完整的摄入流程
- 方法：
  - `ingest_document(text, document_id)`: 摄入文档
  - `_extract_and_create_entity_nodes()`: 提取并创建实体节点
- 流程：
  1. 文档切片
  2. 创建chunk节点
  3. 批量编码chunk
  4. 提取实体
  5. 创建entity节点
  6. 批量编码entity
  7. 构建SEQ边
  8. 构建SIM边
  9. （可选）构建CAUSE边

**真实功能**:
- 将文档转换为图结构
- 自动去重（基于哈希ID）
- 返回摄入统计信息

---

### 7. 检索模块 (`retrieval/`)

#### 7.1 KernelGenerator (`retrieval/kernel_generator.py`)

**实现内容**:
- LLM生成边权重调制核
- 方法：
  - `generate(query)`: 生成核
  - `_parse_kernel_response(response)`: 解析响应
  - `_get_default_kernel()`: 默认核（失败时）

**真实功能**:
- 分析查询意图
- 为SEQ/SIM/CAUSE分配权重（0.0-2.0）
- 返回权重字典和理由
- 失败时返回均匀权重

#### 7.2 SpreadingActivation (`retrieval/spreading_activation.py`)

**实现内容**:
- 激活扩散算法
- 方法：
  - `retrieve(query_embedding, kernel, return_chunks_only)`: 执行检索
  - `_get_anchor_nodes(query_embedding)`: 获取初始锚点
  - `_initialize_activations(anchors)`: 初始化能量
  - `_propagate_energy(activations, kernel, hop)`: 能量传播
  - `_harvest_results(activations, return_chunks_only)`: 收集结果
  - `get_activated_edges()`: 获取激活的边
- 核心算法：
  ```python
  # 初始化
  activations = softmax([sim for _, sim in anchors])

  # 扩散
  for hop in range(MAX_HOPS):
      for node_id, energy in activations.items():
          for neighbor_id, edge in get_neighbors(node_id):
              flow = energy * edge.weight * kernel[edge.type] * DECAY_RATE
              new_activations[neighbor_id] += flow

      # 合并
      activations = merge(activations, new_activations)

  # 返回
  return top_n(activations)
  ```

**真实功能**:
- 混合起点：从Chunk和Entity都搜索锚点
- Softmax初始能量分配
- 多跳迭代扩散
- 加权平均合并（可配置）
- 追踪激活的边（用于LTP）
- 智能返回：top-k chunk + entity路由的chunk，去重

#### 7.3 NeuroRetriever (`retrieval/__init__.py`)

**实现内容**:
- 检索引擎主类
- 方法：
  - `retrieve(query, query_embedding)`: 执行检索
  - `get_activated_edges()`: 获取激活边

**真实功能**:
- 集成KernelGenerator和SpreadingActivation
- 提供统一的检索接口

---

### 8. 可塑性模块 (`plasticity/plasticity_engine.py`)

**实现内容**:
- LTP/LTD实现
- 方法：
  - `reinforce_path(activated_edges)`: LTP强化
  - `decay_unused()`: LTD衰减
  - `maintenance()`: 维护

**真实功能**:
- **LTP**:
  - 强化激活路径上的边
  - weight = min(1.0, weight + LEARNING_RATE)
- **LTD**:
  - 衰减未使用的边
  - **特殊规则**: SIM边不衰减
  - weight *= DECAY_FACTOR
  - 剪枝：weight < MIN_WEIGHT时删除边

---

### 9. 评估模块 (`evaluation/`)

#### 9.1 DatasetLoader (`evaluation/dataset_loader.py`)

**实现内容**:
- 加载HippoRAG格式数据集
- 方法：
  - `load_corpus(dataset_name)`: 加载语料库
  - `load_queries(dataset_name)`: 加载查询
  - `format_documents(corpus)`: 格式化文档

**真实功能**:
- 读取JSON格式数据集
- 提取查询、金标准答案、金标准文档
- 格式化为系统可用的格式

#### 9.2 Metrics (`evaluation/metrics.py`)

**实现内容**:
- `RetrievalRecall`: 检索召回率
- `QAExactMatch`: 问答精确匹配
- `QAF1Score`: 问答F1分数

**真实功能**:
- 计算检索和问答的评估指标
- 支持文本归一化
- 返回平均分数

---

### 10. 主类 (`core.py`)

**实现内容**:
- `NeuroGraphMemory` 类
- 方法：
  - `__init__(config)`: 初始化系统
  - `add_document(text, document_id)`: 添加文档
  - `retrieve(query)`: 检索
  - `feedback(relevant_node_ids)`: 反馈
  - `maintenance()`: 维护
  - `save()`: 保存
  - `get_stats()`: 获取统计

**真实功能**:
- 集成所有模块
- 提供高层API
- 自动管理LLM、embedding model、存储
- 支持保存和加载

---

### 11. 主程序 (`main.py`)

**实现内容**:
- 命令行参数解析
- 完整的实验流程：
  1. 初始化系统
  2. 加载数据集
  3. 索引文档
  4. 运行检索
  5. 评估结果
  6. 维护和保存

**真实功能**:
- 支持运行数据集实验
- 命令行配置所有参数
- 输出详细日志
- 计算评估指标

---

## 🔍 核心算法实现

### 激活扩散算法

**代码位置**: `src/neurogated/retrieval/spreading_activation.py`

**实现细节**:
1. **初始化** (`_initialize_activations`):
   - 向量搜索找到top-K锚点（Chunk + Entity混合）
   - 使用Softmax分配初始能量

2. **扩散** (`_propagate_energy`):
   - 遍历所有激活节点
   - 对每个邻居计算能量流：`flow = energy * edge.weight * kernel[edge.type] * DECAY_RATE`
   - 累积到新激活字典
   - 追踪超过阈值的边（用于LTP）

3. **合并** (`_propagate_energy`):
   - 加权平均：`merged = 0.7 * old + 0.3 * new`
   - 支持其他策略：replace, accumulate, max

4. **收集** (`_harvest_results`):
   - 分离Chunk和Entity激活
   - 收集top-k Chunk
   - 收集Entity路由的Chunk
   - 合并去重
   - 返回top-N

### 可塑性算法

**代码位置**: `src/neurogated/plasticity/plasticity_engine.py`

**实现细节**:
1. **LTP** (`reinforce_path`):
   - 遍历激活的边
   - 增加权重：`weight = min(1.0, weight + LEARNING_RATE)`

2. **LTD** (`decay_unused`):
   - 遍历所有边
   - 跳过SIM边（不衰减）
   - 检查最后激活时间
   - 衰减：`weight *= DECAY_FACTOR`
   - 剪枝：`if weight < MIN_WEIGHT: remove_edge()`

---

## 📊 数据流

```
文档输入
  ↓
TextChunker (切片)
  ↓
EntityExtractor (NER)
  ↓
创建节点 (Chunk + Entity)
  ↓
EmbeddingStore (批量编码)
  ↓
EdgeBuilder (构建边)
  ├── SEQ边 (相邻chunk)
  ├── SIM边 (相似节点)
  └── CAUSE边 (因果关系)
  ↓
GraphStore (存储图)
  ↓
保存到磁盘

查询输入
  ↓
编码查询向量
  ↓
KernelGenerator (生成调制核)
  ↓
SpreadingActivation (激活扩散)
  ├── 向量搜索锚点
  ├── Softmax初始化
  ├── 多跳扩散
  └── 收集结果
  ↓
PlasticityEngine (LTP强化)
  ↓
返回结果
```

---

## 🎯 关键特性

### 1. 完全参数化
- 所有超参数都在MemoryConfig中
- 无硬编码的"魔术数字"

### 2. 模块化设计
- 每个模块独立
- 清晰的接口
- 易于测试和扩展

### 3. 高效存储
- igraph: 高效的图操作
- Parquet: 压缩的向量存储
- SQLite: LLM响应缓存

### 4. 智能检索
- 动态边权重调制
- 混合起点
- Entity路由

### 5. 自适应学习
- LTP强化常用路径
- LTD遗忘不用的边
- SIM边保持稳定

---

## 📈 代码统计

- **Python文件数**: 约30个
- **核心代码行数**: 约2500+行
- **模块数**: 10个主要模块
- **类数**: 约20个
- **函数数**: 约100+个

---

## ✅ 功能完整性

### 已实现
- ✅ 配置管理
- ✅ 数据结构
- ✅ 图存储
- ✅ 向量存储
- ✅ LLM接口
- ✅ 提示词管理
- ✅ 文档切片
- ✅ 实体提取
- ✅ 边构建（SEQ/SIM/CAUSE）
- ✅ 激活扩散检索
- ✅ 动态核生成
- ✅ 可塑性学习（LTP/LTD）
- ✅ 数据集加载
- ✅ 评估指标
- ✅ 主程序

### 可选/待优化
- ⏳ CAUSE边构建（目前简化，可优化）
- ⏳ 异步LLM调用
- ⏳ QA模块
- ⏳ 可视化工具

---

## 🚀 可运行性

系统**完全可运行**，支持：
1. 添加文档并构建图
2. 执行查询并检索
3. 运行数据集实验
4. 计算评估指标
5. 保存和加载状态

---

**总结**: 这是一个**完整实现的、可运行的神经拟态图记忆系统**，所有核心功能都已实现并可用。
