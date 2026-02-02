# 自定义 OpenAI API Base URL 使用指南

本文档说明如何配置自定义的 OpenAI API Base URL，以支持兼容 OpenAI 的 API 服务（如 Azure OpenAI、本地部署的模型等）。

---

## 🔧 配置方法

### 方式1: 命令行参数

```bash
# 使用自定义 base_url
uv run python main.py \
    --dataset sample \
    --llm_base_url "https://your-custom-api.com/v1" \
    --llm_name "gpt-4o-mini"
```

### 方式2: 代码配置

```python
from neurogated import NeuroGraphMemory, MemoryConfig

config = MemoryConfig(
    llm_name="gpt-4o-mini",
    llm_base_url="https://your-custom-api.com/v1",  # 自定义 base URL
    save_dir="outputs/custom"
)

memory = NeuroGraphMemory(config)
```

### 方式3: 环境变量（可选）

```bash
# 设置自定义 base URL
export OPENAI_BASE_URL="https://your-custom-api.com/v1"

# 在代码中读取
import os
config = MemoryConfig(
    llm_base_url=os.getenv("OPENAI_BASE_URL")
)
```

---

## 📋 支持的服务

### 1. Azure OpenAI

```bash
uv run python main.py \
    --llm_base_url "https://your-resource.openai.azure.com/openai/deployments/your-deployment" \
    --llm_name "gpt-4o-mini"

# 需要设置 Azure API Key
export OPENAI_API_KEY="your-azure-api-key"
```

### 2. 本地部署的 vLLM

```bash
# 启动 vLLM 服务
vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --port 8000 \
    --tensor-parallel-size 2

# 使用本地服务
uv run python main.py \
    --llm_base_url "http://localhost:8000/v1" \
    --llm_name "meta-llama/Llama-3.3-70B-Instruct"
```

### 3. OpenAI 兼容的第三方服务

```bash
# 例如：OneAPI, FastChat, LocalAI 等
uv run python main.py \
    --llm_base_url "http://your-service:8080/v1" \
    --llm_name "gpt-4o-mini"
```

### 4. 代理服务

```bash
# 通过代理访问 OpenAI
uv run python main.py \
    --llm_base_url "https://your-proxy.com/v1" \
    --llm_name "gpt-4o-mini"
```

---

## 🔍 验证配置

### 测试连接

创建测试脚本 `test_custom_api.py`:

```python
from neurogated import MemoryConfig
from neurogated.llm import get_llm

# 配置
config = MemoryConfig(
    llm_name="gpt-4o-mini",
    llm_base_url="https://your-custom-api.com/v1",
    save_dir="outputs/test"
)

# 初始化 LLM
llm = get_llm(config)

# 测试调用
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Say hello!"}
]

try:
    response, metadata = llm.infer(messages)
    print(f"✅ Success! Response: {response}")
    print(f"Metadata: {metadata}")
except Exception as e:
    print(f"❌ Error: {e}")
```

运行测试：

```bash
uv run python test_custom_api.py
```

---

## 📝 配置示例

### 完整配置示例

```python
from neurogated import NeuroGraphMemory, MemoryConfig

config = MemoryConfig(
    # LLM 配置
    llm_name="gpt-4o-mini",
    llm_base_url="https://your-custom-api.com/v1",  # 自定义 URL
    llm_temperature=0.0,
    llm_max_new_tokens=2048,

    # Embedding 配置（也可以自定义）
    embedding_model_name="text-embedding-3-small",
    embedding_base_url=None,  # 如果需要也可以自定义

    # 其他配置
    save_dir="outputs/custom",
    cache_llm_responses=True,  # 启用缓存
    TOP_K_ANCHORS=5,
    TOP_N_RETRIEVAL=3,
    MAX_HOPS=2
)

memory = NeuroGraphMemory(config)
```

---

## 🔐 API Key 管理

### OpenAI API Key

```bash
# 标准 OpenAI
export OPENAI_API_KEY="sk-..."

# Azure OpenAI
export OPENAI_API_KEY="your-azure-key"

# 自定义服务
export OPENAI_API_KEY="your-custom-key"
```

### 多个 API Key

如果需要同时使用多个服务：

```python
import os

# 方式1: 在代码中动态设置
os.environ["OPENAI_API_KEY"] = "your-key-for-this-service"

# 方式2: 修改 OpenAILLM 类支持传入 api_key
# （需要修改代码）
```

---

## 🚨 常见问题

### 问题1: 连接超时

```
ERROR - OpenAI API error: Connection timeout
```

**解决方案**:
- 检查 base_url 是否正确
- 检查网络连接
- 检查服务是否运行

### 问题2: 认证失败

```
ERROR - OpenAI API error: Incorrect API key
```

**解决方案**:
- 检查 OPENAI_API_KEY 环境变量
- 确认 API key 对应正确的服务

### 问题3: 不兼容的响应格式

```
ERROR - Failed to parse response
```

**解决方案**:
- 确认服务完全兼容 OpenAI API
- 检查返回的 JSON 格式
- 可能需要修改解析代码

### 问题4: base_url 格式错误

```
ERROR - Invalid URL
```

**解决方案**:
- 确保 URL 以 http:// 或 https:// 开头
- 通常以 /v1 结尾
- 不要包含尾部斜杠

**正确格式**:
```
✅ https://api.openai.com/v1
✅ http://localhost:8000/v1
✅ https://your-resource.openai.azure.com/openai/deployments/your-deployment

❌ https://api.openai.com/v1/
❌ api.openai.com/v1
❌ https://api.openai.com
```

---

## 📊 性能对比

### OpenAI vs 本地部署

| 特性 | OpenAI API | 本地 vLLM |
|------|-----------|-----------|
| 延迟 | 1-3秒 | 0.5-1秒 |
| 成本 | 按使用付费 | 硬件成本 |
| 质量 | 高 | 取决于模型 |
| 隐私 | 数据上传 | 完全本地 |
| 维护 | 无需维护 | 需要维护 |

---

## 🎯 推荐配置

### 开发/测试环境

```python
config = MemoryConfig(
    llm_base_url="http://localhost:8000/v1",  # 本地 vLLM
    llm_name="meta-llama/Llama-3.1-8B-Instruct",  # 小模型
    cache_llm_responses=True  # 启用缓存
)
```

### 生产环境

```python
config = MemoryConfig(
    llm_base_url="https://api.openai.com/v1",  # 官方 API
    llm_name="gpt-4o-mini",  # 高质量模型
    cache_llm_responses=True  # 启用缓存
)
```

### 成本优化

```python
config = MemoryConfig(
    llm_base_url="https://your-proxy.com/v1",  # 代理服务
    llm_name="gpt-4o-mini",
    cache_llm_responses=True,  # 必须启用缓存
    USE_ENTITY_NODES=False,  # 减少 LLM 调用
)
```

---

## 📚 相关文档

- [RUN_GUIDE.md](RUN_GUIDE.md) - 完整运行指南
- [USAGE.md](USAGE.md) - 使用说明
- [MemoryConfig 文档](src/neurogated/config/memory_config.py) - 所有配置参数

---

## ✅ 配置检查清单

- [ ] base_url 格式正确（http(s)://...）
- [ ] API key 已设置
- [ ] 服务正在运行（如果是本地）
- [ ] 网络连接正常
- [ ] 模型名称正确
- [ ] 测试连接成功

---

**现在您可以使用任何兼容 OpenAI 的 API 服务了！** 🎉
