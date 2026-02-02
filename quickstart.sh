#!/bin/bash
# 快速开始脚本 - 帮助新用户快速设置和验证环境

set -e  # 遇到错误立即退出

echo "========================================"
echo "神经拟态图记忆系统 - 快速开始"
echo "========================================"
echo ""

# 检查 uv 是否安装
echo "1. 检查 uv..."
if ! command -v uv &> /dev/null; then
    echo "❌ uv 未安装"
    echo ""
    echo "请安装 uv:"
    echo "  macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  Windows: powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\""
    exit 1
fi
echo "✅ uv 已安装: $(uv --version)"
echo ""

# 安装依赖
echo "2. 安装依赖..."
uv sync
echo "✅ 依赖安装完成"
echo ""

# 检查配置
echo "3. 检查配置..."
if [ ! -f "config.yaml" ]; then
    echo "❌ config.yaml 不存在"
    exit 1
fi
echo "✅ config.yaml 存在"
echo ""

# 检查 API key
echo "4. 检查 API Key..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  环境变量 OPENAI_API_KEY 未设置"
    echo ""
    echo "请设置 API Key:"
    echo "  方式1: export OPENAI_API_KEY='your-key'"
    echo "  方式2: 在 config.yaml 中设置 api.openai_api_key"
    echo ""
    echo "继续验证其他项目..."
else
    echo "✅ OPENAI_API_KEY 已设置"
fi
echo ""

# 运行验证脚本
echo "5. 运行环境验证..."
uv run python verify_setup.py
echo ""

# 提示下一步
echo "========================================"
echo "✅ 环境设置完成！"
echo "========================================"
echo ""
echo "下一步:"
echo "  1. 如果还没设置 API Key，请设置:"
echo "     export OPENAI_API_KEY='your-key'"
echo ""
echo "  2. 运行测试:"
echo "     uv run python test_imports.py    # 测试导入（无需API key）"
echo "     uv run python test_basic.py      # 完整测试（需要API key）"
echo ""
echo "  3. 运行示例:"
echo "     uv run python main.py --dataset sample"
echo ""
echo "  4. 查看文档:"
echo "     cat README.md"
echo "     cat RUN_GUIDE.md"
echo ""
