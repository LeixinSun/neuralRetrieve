# REFER.md - HippoRAG 实现参考指南

本文档从 HippoRAG 项目中抽象出关键实现模式，供实现神经拟态图记忆系统时参考。

---

## 1. 整体架构模式

### 1.1 配置驱动设计
**参考文件**: `src/hipporag/utils/config_utils.py`

HippoRAG 使用单一配置类 `BaseConfig` 管理所有超参数：

```python
@dataclass
class BaseConfig:
    # LLM 配置
    llm_name: str = "gpt-4o-mini"
    llm_base_url: str = None
    max_new_tokens: int = 2048
    temperature: float = 0

    # 存储配置
    force_index_from_scratch: bool = False
    save_dir: str = "outputs"

    # 检索配置
    retrieval_top_k: int = 200
    linking_top_k: int = 5
    max_qa_steps: int = 3
```

**借鉴要点**:
- 使用 `dataclass` 定义配置类，带默认值和类型注解
- 所有"魔术数字"都应该是配置参数
- 配置对象在初始化时传递给所有模块

---

## 2. LLM 接口抽象

### 2.1 工厂模式 + 基类抽象
**参考文件**: `src/hipporag/llm/base.py`, `src/hipporag/llm/__init__.py`

```python
# 基类定义
class BaseLLM(ABC):
    def __init__(self, global_config: BaseConfig):
        self.global_config = global_config
        self.llm_name = global_config.llm_name
        self._init_llm_config()

    @abstractmethod
    def infer(self, messages: List[Dict]) -> Tuple[str, Dict, bool]:
        """同步推理"""
        pass

    @abstractmethod
    async def ainfer(self, messages: List[Dict]) -> Tuple[str, Dict]:
        """异步推理"""
        pass

# 工厂函数
def _get_llm_class(config: BaseConfig):
    if config.llm_name.startswith("bedrock"):
        return BedrockLLM(config)
    elif "Transformers" in config.llm_name:
        return TransformersLLM(config)
    else:
        return CacheOpenAI(config)
```

**借鉴要点**:
- 定义统一的 LLM 接口（同步/异步/批量）
- 使用工厂函数根据配置选择实现
- 返回值包含：响应文本、元数据、缓存命中标志

### 2.2 响应缓存机制
**参考文件**: `src/hipporag/llm/openai_gpt.py`

HippoRAG 使用 SQLite 缓存 LLM 响应：

```python
def cache_response(func):
    def wrapper(self, messages, *args, **kwargs):
        # 生成缓存键
        cache_key = hashlib.sha256(
            json.dumps({
                "messages": messages,
                "model": self.llm_name,
                "seed": self.seed,
                "temperature": self.temperature
            }).encode()
        ).hexdigest()

        # 查询缓存
        if cache_key in cache_db:
            return cache_db[cache_key], metadata, True

        # 调用 API
        response = func(self, messages, *args, **kwargs)

        # 存入缓存
        cache_db[cache_key] = response
        return response, metadata, False
```

**借鉴要点**:
- 使用 SHA256 哈希作为缓存键
- 缓存键包含：消息、模型、种子、温度
- 返回缓存命中标志，便于调试

---

## 3. 向量存储与嵌入管理

### 3.1 EmbeddingStore 设计
**参考文件**: `src/hipporag/embedding_store.py`

HippoRAG 使用 Parquet 文件存储向量：

```python
class EmbeddingStore:
    def __init__(self, embedding_model, db_filename, batch_size, namespace):
        self.embedding_model = embedding_model
        self.namespace = namespace  # 用于区分不同类型的向量
        self.filename = f"{db_filename}/vdb_{namespace}.parquet"

        # 内存索引
        self.hash_ids = []          # 哈希ID列表
        self.texts = []             # 文本内容列表
        self.embeddings = []        # 向量列表
        self.hash_id_to_idx = {}    # 哈希到索引的映射
        self.hash_id_to_text = {}   # 哈希到文本的映射

        self._load_data()

    def insert_strings(self, texts: List[str]):
        # 1. 计算哈希ID
        nodes_dict = {
            compute_mdhash_id(text, prefix=f"{self.namespace}-"): {'content': text}
            for text in texts
        }

        # 2. 过滤已存在的
        missing_ids = [h for h in nodes_dict.keys() if h not in self.hash_id_to_row]

        # 3. 批量编码
        texts_to_encode = [nodes_dict[h]["content"] for h in missing_ids]
        embeddings = self.embedding_model.batch_encode(texts_to_encode)

        # 4. 更新内存和磁盘
        self._upsert(missing_ids, texts_to_encode, embeddings)
```

**借鉴要点**:
- 使用 MD5 哈希作为节点ID（去重 + 幂等性）
- 使用 `namespace` 区分不同类型的向量（entity/fact/chunk）
- Parquet 格式高效存储：`{hash_id, content, embedding}`
- 维护多个内存索引加速查询

### 3.2 哈希ID生成
**参考文件**: `src/hipporag/utils/misc_utils.py`

```python
def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + hashlib.md5(content.encode()).hexdigest()
```

**借鉴要点**:
- 使用内容哈希作为ID，天然去重
- 添加前缀区分不同类型节点

---

## 4. 知识图谱构建

### 4.1 图结构选择
**参考文件**: `src/hipporag/HippoRAG.py`

HippoRAG 使用 `python-igraph` 库：

```python
import igraph as ig

# 初始化图
self.graph = ig.Graph(directed=False)

# 添加节点
self.graph.add_vertices(node_ids)
self.graph.vs["name"] = node_ids

# 添加边
edges = [(source_id, target_id) for source_id, target_id in edge_list]
weights = [weight for _, _, weight in edge_list]
self.graph.add_edges(edges)
self.graph.es["weight"] = weights
self.graph.es["type"] = edge_types

# 保存图
self.graph.write_pickle("graph.pickle")
```

**借鉴要点**:
- `igraph` 性能优于 `networkx`（C 实现）
- 使用顶点属性 `vs["name"]` 存储节点ID
- 使用边属性 `es["weight"]`, `es["type"]` 存储权重和类型
- Pickle 序列化保存图结构

### 4.2 OpenIE 信息抽取
**参考文件**: `src/hipporag/information_extraction/openie_openai.py`

HippoRAG 分两步抽取：

**步骤1: 命名实体识别 (NER)**
```python
def ner(self, chunk_key: str, passage: str) -> NerRawOutput:
    # 1. 渲染提示词
    messages = self.prompt_template_manager.render(
        name='ner',
        passage=passage
    )

    # 2. 调用 LLM
    raw_response, metadata, cache_hit = self.llm_model.infer(messages)

    # 3. 解析响应（正则提取JSON）
    pattern = r'\{[^{}]*"named_entities"\s*:\s*\[[^\]]*\][^{}]*\}'
    match = re.search(pattern, raw_response, re.DOTALL)
    entities = eval(match.group())["named_entities"]

    # 4. 去重
    unique_entities = list(dict.fromkeys(entities))

    return NerRawOutput(chunk_id=chunk_key, unique_entities=unique_entities)
```

**步骤2: 三元组抽取**
```python
def triple_extraction(self, chunk_key: str, passage: str, named_entities: List[str]):
    # 1. 渲染提示词（包含已识别的实体）
    messages = self.prompt_template_manager.render(
        name='triple_extraction',
        passage=passage,
        named_entity_json=json.dumps({"named_entities": named_entities})
    )

    # 2. 调用 LLM
    raw_response, metadata, cache_hit = self.llm_model.infer(messages)

    # 3. 解析三元组
    pattern = r'\{[^{}]*"triples"\s*:\s*\[[^\]]*\][^{}]*\}'
    triples = eval(match.group())["triples"]

    # 4. 过滤无效三元组
    valid_triples = filter_invalid_triples(triples)

    return TripleRawOutput(chunk_id=chunk_key, triples=valid_triples)
```

**借鉴要点**:
- 两阶段抽取：先识别实体，再抽取关系
- 使用正则表达式容错解析 JSON
- 返回结构化对象（`NerRawOutput`, `TripleRawOutput`）
- 包含元数据（缓存命中、完成原因等）

### 4.3 边的构建策略

**Fact Edges（事实边）**:
```python
def add_fact_edges(self):
    for triple in all_triples:
        source_id = compute_mdhash_id(triple.subject, prefix="entity-")
        target_id = compute_mdhash_id(triple.object, prefix="entity-")

        # 累积权重（多次出现增加权重）
        if (source_id, target_id) in node_to_node_stats:
            node_to_node_stats[(source_id, target_id)] += 1
        else:
            node_to_node_stats[(source_id, target_id)] = 1
```

**Passage Edges（文档-实体边）**:
```python
def add_passage_edges(self):
    for entity_id, chunk_ids in ent_node_to_chunk_ids.items():
        for chunk_id in chunk_ids:
            edges.append((entity_id, chunk_id, 1.0))  # 固定权重
```

**Synonymy Edges（同义边）**:
```python
def add_synonymy_edges(self):
    # 1. 获取所有实体向量
    entity_embeddings = np.array([emb for emb in entity_embedding_store.embeddings])

    # 2. KNN 搜索
    for i, entity_id in enumerate(entity_ids):
        similarities = cosine_similarity([entity_embeddings[i]], entity_embeddings)[0]

        # 3. 阈值过滤
        similar_indices = np.where(similarities > synonymy_threshold)[0]

        # 4. 建边
        for j in similar_indices:
            if i != j:
                edges.append((entity_ids[i], entity_ids[j], similarities[j]))
```

**借鉴要点**:
- **Fact Edges**: 权重 = 共现次数
- **Passage Edges**: 固定权重 1.0
- **Synonymy Edges**: 权重 = 余弦相似度，使用阈值过滤

---

## 5. 检索实现

### 5.1 HippoRAG 检索流程
**参考文件**: `src/hipporag/HippoRAG.py` (retrieve 方法)

```python
def retrieve(self, queries: List[str], num_to_retrieve: int = 5):
    # 1. 编码查询（两种指令）
    query_to_fact_embeddings = self.embedding_model.batch_encode(
        queries,
        instruction=get_query_instruction("query_to_fact")
    )
    query_to_passage_embeddings = self.embedding_model.batch_encode(
        queries,
        instruction=get_query_instruction("query_to_passage")
    )

    results = []
    for i, query in enumerate(queries):
        # 2. 计算 Fact 分数（点积）
        fact_scores = np.dot(
            query_to_fact_embeddings[i],
            np.array(self.fact_embedding_store.embeddings).T
        )

        # 3. Rerank Facts（使用 DSPy 过滤器）
        reranked_facts = self.rerank_filter.forward(query, fact_scores)

        # 4. 图遍历（从 Fact 到 Passage）
        activated_passages = set()
        for fact_id in reranked_facts[:linking_top_k]:
            # 获取 Fact 连接的实体
            entities = self.graph.neighbors(fact_id)

            # 获取实体连接的文档
            for entity_id in entities:
                passages = self.graph.neighbors(entity_id, mode="out")
                activated_passages.update(passages)

        # 5. Passage 重排序
        passage_scores = np.dot(
            query_to_passage_embeddings[i],
            np.array([self.chunk_embedding_store.embeddings[p] for p in activated_passages]).T
        )

        # 6. 返回 Top-N
        top_passages = sorted(zip(activated_passages, passage_scores),
                             key=lambda x: x[1], reverse=True)[:num_to_retrieve]

        results.append([self.chunk_embedding_store.hash_id_to_text[p] for p, _ in top_passages])

    return results
```

**借鉴要点**:
- **双向量编码**: Query 分别编码为 fact-query 和 passage-query
- **两阶段检索**: 先检索 Fact，再通过图遍历到 Passage
- **图遍历**: Fact → Entity → Passage
- **重排序**: 使用 LLM 过滤器（DSPy）重排 Fact

### 5.2 向量检索工具函数
**参考文件**: `src/hipporag/utils/embed_utils.py`

```python
def retrieve_knn(query_embedding, embeddings, top_k):
    """KNN 检索"""
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return top_indices, similarities[top_indices]
```

---

## 6. 数据集集成

### 6.1 数据集格式
**参考文件**: `reproduce/dataset/sample_corpus.json`, `reproduce/dataset/sample.json`

**Corpus 格式**:
```json
[
  {
    "title": "文档标题",
    "text": "文档内容",
    "idx": 0
  }
]
```

**Query 格式**:
```json
[
  {
    "id": "sample/question_1",
    "question": "问题文本",
    "answer": ["答案1", "答案2"],
    "answerable": true,
    "paragraphs": [
      {
        "title": "支持文档标题",
        "text": "支持文档内容",
        "is_supporting": true,
        "idx": 0
      }
    ]
  }
]
```

### 6.2 数据加载
**参考文件**: `main.py`

```python
# 加载语料库
corpus_path = f"reproduce/dataset/{dataset_name}_corpus.json"
with open(corpus_path, "r") as f:
    corpus = json.load(f)
docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]

# 加载查询
samples = json.load(open(f"reproduce/dataset/{dataset_name}.json", "r"))
queries = [s['question'] for s in samples]

# 提取金标准答案
gold_answers = get_gold_answers(samples)
gold_docs = get_gold_docs(samples, dataset_name)
```

**借鉴要点**:
- 文档格式：`title\n\ntext`
- 支持多种数据集格式（HotpotQA, MuSiQue, 2WikiMultiHopQA）
- 提取金标准用于评估

---

## 7. 提示词管理

### 7.1 PromptTemplateManager
**参考文件**: `src/hipporag/prompts/prompt_template_manager.py`

```python
class PromptTemplateManager:
    def __init__(self, role_mapping: Dict[str, str]):
        self.role_mapping = role_mapping  # {"system": "system", "user": "user"}
        self.templates = self._load_templates()

    def render(self, name: str, **kwargs) -> List[Dict]:
        """渲染提示词模板"""
        template = self.templates[name]
        messages = []

        for role, content_template in template:
            content = content_template.format(**kwargs)
            messages.append({
                "role": self.role_mapping[role],
                "content": content
            })

        return messages
```

**模板示例** (`src/hipporag/prompts/templates/ner.py`):
```python
NER_TEMPLATE = [
    ("system", "You are a helpful assistant that extracts named entities."),
    ("user", "Extract all named entities from the following passage:\n\n{passage}")
]
```

**借鉴要点**:
- 集中管理所有提示词模板
- 使用 `role_mapping` 适配不同 LLM 的角色名称
- 模板使用 Python 字符串格式化

---

## 8. 评估指标

### 8.1 检索评估
**参考文件**: `src/hipporag/evaluation/retrieval_eval.py`

```python
class RetrievalRecall:
    def compute(self, retrieved_docs: List[List[str]], gold_docs: List[List[str]]):
        recalls = []
        for retrieved, gold in zip(retrieved_docs, gold_docs):
            retrieved_set = set(retrieved)
            gold_set = set(gold)
            recall = len(retrieved_set & gold_set) / len(gold_set)
            recalls.append(recall)
        return np.mean(recalls)
```

### 8.2 QA 评估
**参考文件**: `src/hipporag/evaluation/qa_eval.py`

```python
class QAExactMatch:
    def compute(self, predictions: List[str], gold_answers: List[List[str]]):
        em_scores = []
        for pred, golds in zip(predictions, gold_answers):
            normalized_pred = self._normalize(pred)
            normalized_golds = [self._normalize(g) for g in golds]
            em = int(normalized_pred in normalized_golds)
            em_scores.append(em)
        return np.mean(em_scores)
```

**借鉴要点**:
- Recall@K: 检索到的文档中有多少是金标准
- Exact Match: 预测答案是否完全匹配金标准
- F1 Score: 预测答案与金标准的词级别重叠

---

## 9. 关键设计模式总结

### 9.1 模块化设计
```
HippoRAG (主类)
├── GraphStore (图存储)
├── EmbeddingStore × 3 (chunk/entity/fact)
├── BaseLLM (LLM 接口)
├── BaseEmbeddingModel (嵌入模型)
├── OpenIE (信息抽取)
├── PromptTemplateManager (提示词管理)
└── Metrics (评估指标)
```

### 9.2 数据流
```
文档输入
  ↓
Chunking (切片)
  ↓
OpenIE (NER + Triple Extraction)
  ↓
Graph Construction (建图)
  ├── Fact Edges (三元组)
  ├── Passage Edges (文档-实体)
  └── Synonymy Edges (实体相似)
  ↓
Embedding (向量化)
  ├── Chunk Embeddings
  ├── Entity Embeddings
  └── Fact Embeddings
  ↓
Retrieval (检索)
  ├── Vector Search (向量搜索)
  ├── Graph Traversal (图遍历)
  └── Reranking (重排序)
  ↓
QA (问答)
```

### 9.3 持久化策略
- **向量**: Parquet 文件 (`vdb_{namespace}.parquet`)
- **图**: Pickle 文件 (`graph.pickle`)
- **LLM 缓存**: SQLite 数据库
- **OpenIE 结果**: JSON 文件

---

## 10. 实现神经拟态系统的映射建议

| HippoRAG 组件 | 神经拟态系统对应组件 | 修改建议 |
|--------------|---------------------|---------|
| `BaseConfig` | `MemoryConfig` | 添加 `ENERGY_DECAY_RATE`, `HEBBIAN_LEARNING_RATE` 等参数 |
| `GraphStore` | `GraphStore` | 添加边类型 `SEQ/SIM/CAUSE`，支持动态权重更新 |
| `EmbeddingStore` | 保持不变 | 继续使用 Parquet + 哈希ID |
| `OpenIE` | `IngestionEngine` | 添加因果关系抽取（LLM 驱动） |
| `retrieve()` | `NeuroRetriever` | 替换为激活扩散算法 + 卷积核调制 |
| N/A | `KernelGenerator` | 新增：LLM 生成边权重调制核 |
| N/A | `PlasticityEngine` | 新增：LTP/LTD 权重更新 |

---

## 11. 代码复用建议

**可直接复用**:
- `EmbeddingStore` 类（向量存储）
- `compute_mdhash_id` 函数（哈希ID生成）
- `PromptTemplateManager` 类（提示词管理）
- `BaseLLM` 接口和工厂模式
- 数据集加载逻辑

**需要修改**:
- 图构建逻辑（添加 `SEQ/SIM/CAUSE` 边类型）
- 检索算法（从图遍历改为激活扩散）
- 添加权重更新机制（LTP/LTD）

**需要新增**:
- `KernelGenerator`（LLM 生成调制核）
- `SpreadingActivation`（激活扩散算法）
- `PlasticityEngine`（可塑性引擎）

---

## 12. 性能优化技巧

1. **批量处理**: 所有 LLM 调用和向量编码都使用批量接口
2. **缓存**: LLM 响应缓存避免重复调用
3. **增量更新**: 使用哈希ID检测已存在的节点，避免重复计算
4. **稀疏图**: 限制每个节点的最大邻居数（`MAX_SIM_NEIGHBORS`）
5. **异步调用**: 使用 `asyncio` 并发调用 LLM API

---

**参考完毕。开始实现时，建议先搭建配置和存储层，再实现摄入层，最后实现检索和进化层。**
