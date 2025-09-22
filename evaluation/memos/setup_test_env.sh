#!/bin/bash

# LoCoMo 测试评估环境一键安装脚本
# 适用于 nemori 项目的测试评估框架

# 启用调试模式和错误处理
set -euo pipefail  # 遇到错误立即退出，未定义变量报错，管道失败报错

# 错误处理函数
error_exit() {
    echo "❌ 脚本在第 $1 行发生错误，退出码: $2"
    echo "💡 请检查上面的错误信息并重新运行脚本"
    exit $2
}

# 设置错误陷阱
trap 'error_exit $LINENO $?' ERR

echo "🚀 开始设置 LoCoMo 测试评估环境..."
echo "========================================================="

# 检查是否在正确的目录
if [ ! -f "run.sh" ] || [ ! -d "scripts" ]; then
    echo "❌ 错误：请在 evaluationV2/memos 目录下运行此脚本"
    exit 1
fi

# 检查 Python 版本
echo "📋 检查 Python 环境..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ 错误：需要 Python $required_version 或更高版本，当前版本：$python_version"
    exit 1
fi
echo "✅ Python 版本检查通过：$python_version"

# 检查 uv 是否安装
echo "📋 检查 uv 包管理器..."
if ! command -v uv &> /dev/null; then
    echo "❌ 错误：未找到 uv 包管理器"
    echo "请先安装 uv：curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "✅ uv 包管理器已安装：$(uv --version)"

# 返回项目根目录安装主要依赖
echo "📦 安装 nemori 主项目依赖..."
cd ../../
uv sync
echo "✅ 主项目依赖安装完成"

# 返回测试目录
cd evaluationV2/memos/

# 安装测试专用依赖
echo "📦 安装测试评估专用依赖..."

# 分步安装依赖，避免一次性安装过多导致失败
echo "正在安装基础依赖..."
if ! uv add --group test pandas python-dotenv openai pydantic; then
    echo "⚠️ 基础依赖安装失败，尝试逐个安装..."
    uv add --group test pandas || echo "pandas 安装失败"
    uv add --group test python-dotenv || echo "python-dotenv 安装失败"
    uv add --group test openai || echo "openai 安装失败"
    uv add --group test pydantic || echo "pydantic 安装失败"
fi

echo "正在安装NLP依赖..."
if ! uv add --group test nltk transformers sentence-transformers tokenizers; then
    echo "⚠️ NLP依赖安装失败，尝试逐个安装..."
    uv add --group test nltk || echo "nltk 安装失败"
    uv add --group test transformers || echo "transformers 安装失败"
    uv add --group test sentence-transformers || echo "sentence-transformers 安装失败"
    uv add --group test tokenizers || echo "tokenizers 安装失败"
fi

echo "正在安装评估指标依赖..."
if ! uv add --group test bert-score rouge-score scipy tqdm numpy asyncio-throttle; then
    echo "⚠️ 评估指标依赖安装失败，尝试逐个安装..."
    uv add --group test bert-score || echo "bert-score 安装失败"
    uv add --group test rouge-score || echo "rouge-score 安装失败"
    uv add --group test scipy || echo "scipy 安装失败"
    uv add --group test tqdm || echo "tqdm 安装失败"
    uv add --group test numpy || echo "numpy 安装失败"
    uv add --group test asyncio-throttle || echo "asyncio-throttle 安装失败"
fi

echo "正在安装记忆框架和LLM依赖..."
if ! uv add --group test litellm; then
    echo "⚠️ litellm 安装失败，继续安装其他依赖..."
fi

if ! uv add --group test mem0-ai; then
    echo "⚠️ mem0-ai 安装失败，继续安装其他依赖..."
fi

if ! uv add --group test zep-cloud; then
    echo "⚠️ zep-cloud 安装失败，继续安装其他依赖..."
fi

echo "✅ 测试依赖安装完成"

# 下载必要的 NLTK 数据
echo "📚 下载 NLTK 数据..."
uv run python -c "
import nltk
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print('✅ NLTK 数据下载完成')
except Exception as e:
    print(f'⚠️ NLTK 数据下载失败: {e}')
"

# 创建必要的目录结构
echo "📁 创建目录结构..."
mkdir -p logs
mkdir -p results/locomo
mkdir -p data/locomo
echo "✅ 目录结构创建完成"

# 检查数据文件
echo "📋 检查数据文件..."
if [ ! -f "data/locomo/locomo10.json" ]; then
    echo "⚠️ 警告：未找到 data/locomo/locomo10.json 数据文件"
    echo "   请确保将 LoCoMo 数据集放置在 data/locomo/ 目录下"
else
    echo "✅ 数据文件检查通过"
fi

# 创建环境配置文件
echo "⚙️ 设置环境配置..."
if [ ! -f ".env" ]; then
    if [ -f "env.template" ]; then
        cp env.template .env
        echo "✅ 已创建 .env 文件，请编辑其中的 API 密钥配置"
        echo "   主要需要配置："
        echo "   - OPENAI_API_KEY: 你的 OpenAI API 密钥"
        echo "   - CHAT_MODEL: 使用的聊天模型 (如 gpt-4o-mini)"
        echo "   - EVAL_VERSION: 评估版本标识符"
    else
        echo "⚠️ 警告：未找到 env.template 文件"
    fi
else
    echo "✅ .env 文件已存在"
fi

# 验证安装
echo "🔍 验证安装..."
echo "检查核心依赖..."

# 分别检查各个包，避免一个失败导致整个验证失败
packages_to_check=(
    "pandas"
    "nltk" 
    "transformers"
    "openai"
    "rouge_score"
    "bert_score"
    "sentence_transformers"
    "pydantic"
    "tqdm"
    "numpy"
)

failed_packages=()

for package in "${packages_to_check[@]}"; do
    if uv run python -c "import $package" 2>/dev/null; then
        echo "✅ $package 导入成功"
    else
        echo "❌ $package 导入失败"
        failed_packages+=("$package")
    fi
done

# 检查可选包（不影响主要功能）
optional_packages=("litellm" "mem0" "zep_cloud")
echo "检查可选依赖..."

for package in "${optional_packages[@]}"; do
    if uv run python -c "import $package" 2>/dev/null; then
        echo "✅ $package (可选) 导入成功"
    else
        echo "⚠️ $package (可选) 导入失败，但不影响核心功能"
    fi
done

if [ ${#failed_packages[@]} -eq 0 ]; then
    echo "✅ 所有核心 Python 包都已正确安装"
else
    echo "⚠️ 以下核心包安装失败: ${failed_packages[*]}"
    echo "💡 可以尝试手动安装失败的包：uv add --group test ${failed_packages[*]}"
fi

echo "========================================================="
echo "🎉 LoCoMo 测试评估环境安装完成！"
echo ""
echo "📋 下一步操作："
echo "1. 编辑 .env 文件，配置你的 API 密钥"
echo "2. 确保 data/locomo/locomo10.json 数据文件存在"
echo "3. 运行测试：./run.sh"
echo ""
echo "🔧 常用命令："
echo "- 运行完整评估：./run.sh"
echo "- 单独运行摄取：uv run python scripts/locomo/locomo_ingestion_emb.py --lib nemori --version test"
echo "- 单独运行搜索：uv run python scripts/locomo/locomo_search.py --lib nemori --version test"
echo "- 单独运行响应：uv run python scripts/locomo/locomo_responses.py --lib nemori --version test"
echo "- 单独运行评估：uv run python scripts/locomo/locomo_eval.py --lib nemori --version test"
echo ""
echo "📖 更多信息请查看 TEST_SETUP_README.md"
echo "========================================================="
