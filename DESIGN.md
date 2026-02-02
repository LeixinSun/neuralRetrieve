这是一个经过精细化修订的 `DESIGN.md`。针对你提出的两点要求（相似度边逻辑修改、超参数全面参数化），进行了针对性的调整。

这份文档现在可以直接提供给 Claude/Cursor 等代码助手，作为标准化的开发规范。

```markdown
# DESIGN.md - 神经拟态图记忆系统 (Neuro-Gated Graph Memory System)

## 1. 系统概述 (System Overview)

本文档详细描述了 **神经拟态图记忆系统** 的架构设计。这是一个面向 AI Agent 的下一代检索机制，旨在超越传统的静态向量检索 (Vector Search)，构建一个基于图结构、受神经科学启发、采用 **“激活扩散 (Spreading Activation)”** 模型的动态记忆网络。

**核心设计哲学:**
*   **记忆即图谱 (Memory as a Graph):** 信息存储为节点 (Nodes)，通过有类型的边 (Typed Edges) 相互连接。
*   **Query 即卷积核 (Query as a Modulation Kernel):** 用户的 Query 会生成一个动态的“核”，在运行时重新调制不同类型边的权重 (例如：针对“为什么”的问题，放大“因果”边的权重)。
*   **检索即能量流动 (Retrieval as Energy Flow):** 检索不是查字典，而是能量从初始锚点在图谱中迭代扩散的过程。
*   **可塑性 (Plasticity):** 图谱结构不是静态的，它会根据使用情况（赫布学习 Hebbian Learning）进行自我强化，同时支持基于时间的被动遗忘（LTD），但特定的结构（如语义关联）保持稳定。

---

## 2. 系统配置与超参数 (Configuration & Hyperparameters)

**所有硬编码数字必须提取为以下可配置参数：**

```python
class MemoryConfig:
    # 检索相关
    TOP_K_ANCHORS = 5           # 向量搜索初筛的锚点数量
    TOP_N_RETRIEVAL = 3         # 最终返回的记忆节点数量
    MAX_HOPS = 2                # 能量扩散的最大跳数 (迭代深度)
    ENERGY_DECAY_RATE = 0.5     # 扩散时的能量衰减率 (每跳一步衰减多少)
    
    # 建图相关
    MAX_SIM_NEIGHBORS = 5       # 每个节点允许建立的最大相似边数量 (保持图稀疏)
    
    # 进化与学习
    HEBBIAN_LEARNING_RATE = 0.1 # 触发强化时的权重增量
    TIME_DECAY_FACTOR = 0.99    # 随时间推移的自然遗忘/衰减系数
    MIN_EDGE_WEIGHT = 0.1       # 边的最小权重，低于此值可能被剪枝
```

---

## 3. 架构组件 (Architecture Components)

系统由四个核心模组解耦组成：
1.  **存储层 (The Cortex):** 管理图数据结构（节点+边）与向量索引。
2.  **摄入层 (The Senses):** 将原始文本转化为图中的节点，并建立初始连接（时序、语义、因果）。
3.  **检索层 (The Hippocampus):** 核心引擎。负责调用 LLM 生成调制核，并执行激活扩散算法。
4.  **进化层 (The Plasticity):** 反馈闭环。负责权重的动态调整（LTP 强化 / LTD 衰减）。

---

## 4. 数据结构设计 (Data Structures)

为了方便实现与演示，我们采用内存对象模型 (In-memory Object Model) 模拟数据库行为。

### 4.1 记忆节点 (`MemoryNode`)
代表一个信息单元（神经元）。

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `id` | UUID | 唯一标识符。 |
| `content` | String | 文本内容。 |
| `embedding` | List[Float] | 稠密向量。 |
| `base_energy` | Float | 基础能量值 (默认 1.0)。长期不用会衰减。 |
| `last_accessed` | Timestamp | 用于计算遗忘曲线。 |
| `metadata` | Dict | 元数据。 |

### 4.2 突触连接 (`MemoryEdge`)
代表神经元之间的连接通道。

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `source_id` | UUID | 起始节点。 |
| `target_id` | UUID | 目标节点。 |
| `type` | Enum | `SEQ` (时序), `SIM` (语义相似), `CAUSE` (因果逻辑)。 |
| `weight` | Float | 动态强度 (0.0 - 1.0)。 |
| `created_at` | Timestamp | 创建时间。 |

---

## 5. 模块实现细节 (Module Implementation)

### 5.1 存储层 (`GraphStore`)
*   **职责:** 维护节点列表和邻接表。
*   **关键方法:**
    *   `add_node(node)`
    *   `add_edge(source, target, type, weight)`
    *   `get_neighbors(node_id) -> List[Edge]`
    *   `vector_search(query_vec, k)`: 计算 Query 与所有节点的余弦相似度，返回 `TOP_K_ANCHORS` 个锚点。

### 5.2 摄入层 (`IngestionEngine`)
*   **职责:** 将文档转化为图。
*   **逻辑流程:**
    1.  **切片 (Chunking):** 将文本切分为块。
    2.  **建点:** 为每个块创建 `MemoryNode`。
    3.  **建边 - SEQ (时序):** 块 `i` 与 `i+1` 之间建立 `SEQ` 边，初始权重 1.0。
    4.  **建边 - SIM (语义):** 
        *   **不使用硬阈值。**
        *   计算当前节点与图中现有节点的相似度。
        *   选取相似度最高的 `MAX_SIM_NEIGHBORS` 个节点。
        *   **权重映射:** 直接将相似度映射为权重 (e.g., `weight = cosine_similarity`)。
        *   建立 `SIM` 边。
    5.  **建边 - CAUSE (因果 - LLM 驱动):**
        *   **机制:** 遍历最近的邻居或滑动窗口内的节点对。
        *   **调用 LLM:** 发送 Prompt 请求判断因果关系。
        *   **建边:** 如果存在因果，建立 **双向边 (Bidirectional)** `A<->B`，权重由 LLM 指定 (0.0 - 1.0)。

### 5.3 检索层 (`NeuroRetriever`)
**核心大脑。**

#### 子组件: 卷积核生成器 (`KernelGenerator`)
*   **输入:** 用户 Query (文本)。
*   **输出:** 字典 `{'weights': {SEQ: float, SIM: float, CAUSE: float}, 'justification': str}`。
*   **机制 (LLM 驱动):**
    *   **调用 LLM API:** 发送 Prompt，要求分析意图并分配权重 (0.0-2.0)。
    *   **理由 (Justification):** LLM 必须解释为什么这样分配权重（便于 Debug 和展示）。

#### 子组件: 激活扩散算法 (`SpreadingActivation`)
*   **算法步骤:**
    1.  **注入 (Priming):**
        *   调用 `vector_search` 找到 `TOP_K_ANCHORS` 个锚点。
        *   初始化能量池: `activations = {node.id: similarity_score}`。
    
    2.  **扩散循环 (Propagation Loop) - 迭代 `MAX_HOPS` 次:**
        *   创建 `new_activations` 缓冲区。
        *   遍历当前所有 `active_node`:
            *   获取邻居节点。
            *   对于每个邻居:
                *   `kernel_mod = kernel['weights'][edge.type]` (从 LLM 生成的核中获取调节系数)
                *   `flow = current_energy * edge.weight * kernel_mod * ENERGY_DECAY_RATE`
                *   `new_activations[neighbor] += flow` (能量累积)
        *   更新: `activations = merge(activations, new_activations)`。
    
    3.  **涌现 (Harvesting):**
        *   按最终能量值排序。
        *   返回 `TOP_N_RETRIEVAL` 个节点。

### 5.4 进化层 (`PlasticityEngine`)
*   **方法:** `reinforce_path(retrieved_nodes)` (LTP)
    *   对于检索结果中连接的边，执行 `weight += HEBBIAN_LEARNING_RATE`。
*   **方法:** `decay_unused()` (LTD)
    *   **被动衰减逻辑:**
        *   遍历全图边。
        *   **特殊规则:** 如果边类型是 `SIM` (语义相似)，**不执行时间衰减** (相似度是客观存在的，不随时间改变)。
        *   对于其他类型 (`SEQ`, `CAUSE`)，如果长期未被激活，执行 `weight *= TIME_DECAY_FACTOR`。
        *   如果 `weight < MIN_EDGE_WEIGHT`，则移除该边。

---

## 6. 编码规范与接口定义 (Coding Guidelines)

**语言:** Python 3.9+
**依赖:** `numpy`, `scikit-learn`, `openai` (或兼容接口)。

### 6.1 主类接口 (`NeuroGraphMemory`)
```python
class NeuroGraphMemory:
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.store = GraphStore()
        self.retriever = NeuroRetriever(self.store, self.config)
        self.plasticity = PlasticityEngine(self.store, self.config)
    
    def add_document(self, text: str):
        """Pipeline: Ingestion -> Graph Construction"""
        pass
        
    def retrieve(self, query: str) -> List[str]:
        """Pipeline: LLM Kernel -> Vector Priming -> Spreading Activation -> Return Content"""
        pass
        
    def feedback(self, relevant_node_ids: List[UUID]):
        """Trigger Hebbian Learning"""
        pass
        
    def maintenance(self):
        """Trigger Passive Decay (LTD)"""
        pass
```

---

## 7. 给代码实现的具体指引 (Implementation Guide)

1.  **配置先行:** 首先定义 `MemoryConfig` 类，将所有“魔术数字”移入其中。
2.  **模拟 LLM:**
    *   `extract_causal_edges`: 输入两段文本，输出 `(bool, weight)`。
    *   `generate_kernel`: 输入 Query，输出 `{'weights': {...}, 'justification': ...}`。
3.  **建图细节:**
    *   在建立 `SIM` 边时，不要使用 `if sim > 0.8`。而是对每个新节点，找到 `MAX_SIM_NEIGHBORS` 个最相似的旧节点，直接连接，权重设为 `sim` 值。
4.  **因果边双向性:** 确保 `CAUSE` 边总是成对创建 (A->B, B->A)，权重一致。
5.  **核心扩散:** 实现 `SpreadingActivation` 时，严格遵循公式：`Flow = Energy * EdgeWeight * KernelMod * Decay`。
6.  **遗忘逻辑:** 在 `decay_unused` 中增加判断：`if edge.type == EdgeType.SIM: continue`。

## 8. 成功标准 (Success Criteria)
1.  **参数化:** 代码中不应出现硬编码的阈值或数量。
2.  **动态性:** 展示同一个图，针对 Query A ("原因") 和 Query B ("结果")，`KernelGenerator` 生成了显著不同的权重，导致召回了不同的节点。
3.  **稳定性:** 展示多次调用后，`SIM` 类型的边权重保持不变，而未使用的 `SEQ` 边权重逐渐降低。
```
