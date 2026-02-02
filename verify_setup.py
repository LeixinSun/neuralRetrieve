#!/usr/bin/env python3
"""
验证 RUN_GUIDE.md 中的步骤是否可行
"""

import os
import sys
from pathlib import Path

def check_step(step_name, check_func):
    """检查单个步骤"""
    print(f"\n{'='*60}")
    print(f"检查: {step_name}")
    print('='*60)
    try:
        result = check_func()
        if result:
            print(f"✅ {step_name} - 通过")
            return True
        else:
            print(f"❌ {step_name} - 失败")
            return False
    except Exception as e:
        print(f"❌ {step_name} - 错误: {e}")
        return False

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 10:
        return True
    else:
        print("需要Python 3.10+")
        return False

def check_config_yaml():
    """检查config.yaml是否存在"""
    config_path = Path("config.yaml")
    if config_path.exists():
        print(f"找到配置文件: {config_path}")

        # 检查是否包含必要的字段
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        required_sections = ['api', 'retrieval', 'graph', 'storage']
        for section in required_sections:
            if section in config:
                print(f"  ✓ 包含 '{section}' 配置")
            else:
                print(f"  ✗ 缺少 '{section}' 配置")
                return False

        return True
    else:
        print("未找到config.yaml文件")
        return False

def check_imports():
    """检查能否导入neurogated包"""
    try:
        from neurogated import NeuroGraphMemory, MemoryConfig, config_from_yaml
        print("✓ 成功导入 NeuroGraphMemory")
        print("✓ 成功导入 MemoryConfig")
        print("✓ 成功导入 config_from_yaml")
        return True
    except ImportError as e:
        print(f"导入失败: {e}")
        print("请运行: uv sync")
        return False

def check_config_loading():
    """检查config.yaml加载功能"""
    try:
        from neurogated import config_from_yaml
        config = config_from_yaml("config.yaml")
        print(f"✓ 成功加载config.yaml")
        print(f"  LLM: {config.llm_name}")
        print(f"  Embedding: {config.embedding_model_name}")
        print(f"  Top K Anchors: {config.TOP_K_ANCHORS}")
        return True
    except Exception as e:
        print(f"加载配置失败: {e}")
        return False

def check_dataset():
    """检查数据集是否存在"""
    dataset_dir = Path("dataset")
    if not dataset_dir.exists():
        print("未找到dataset目录")
        return False

    required_files = ["sample_corpus.json", "sample.json"]
    for filename in required_files:
        filepath = dataset_dir / filename
        if filepath.exists():
            print(f"✓ 找到 {filename}")
        else:
            print(f"✗ 缺少 {filename}")
            return False

    return True

def check_api_key():
    """检查API key是否设置"""
    # 先检查环境变量
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key and env_key != "your-api-key-here":
        print(f"✓ 环境变量中找到 OPENAI_API_KEY")
        return True

    # 检查config.yaml
    try:
        import yaml
        with open("config.yaml") as f:
            config = yaml.safe_load(f)

        yaml_key = config.get('api', {}).get('openai_api_key')
        if yaml_key and yaml_key != "your-openai-api-key-here":
            print(f"✓ config.yaml中找到 openai_api_key")
            return True
        else:
            print("⚠️  未设置有效的API key")
            print("   请在config.yaml中设置 api.openai_api_key")
            print("   或设置环境变量 OPENAI_API_KEY")
            return False
    except Exception as e:
        print(f"检查API key失败: {e}")
        return False

def main():
    """运行所有检查"""
    print("="*60)
    print("RUN_GUIDE.md 步骤验证")
    print("="*60)

    checks = [
        ("Python 3.10+", check_python_version),
        ("config.yaml 存在", check_config_yaml),
        ("neurogated 包导入", check_imports),
        ("config.yaml 加载", check_config_loading),
        ("数据集文件", check_dataset),
        ("API Key 设置", check_api_key),
    ]

    results = []
    for name, func in checks:
        results.append(check_step(name, func))

    # 总结
    print("\n" + "="*60)
    print("验证总结")
    print("="*60)

    passed = sum(results)
    total = len(results)

    for i, (name, _) in enumerate(checks):
        status = "✅" if results[i] else "❌"
        print(f"{status} {name}")

    print(f"\n通过: {passed}/{total}")

    if passed == total:
        print("\n🎉 所有检查通过！可以开始运行系统了。")
        print("\n下一步:")
        print("  uv run python test_basic.py")
        print("  uv run python main.py --dataset sample")
        return True
    else:
        print("\n⚠️  部分检查未通过，请根据上述提示修复问题。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
