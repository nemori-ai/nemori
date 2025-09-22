#!/bin/bash

# LoCoMo Results Analyzer 启动脚本

echo "=========================================="
echo "  LoCoMo 结果分析器 - React 版本"
echo "=========================================="

# 检查Node.js是否已安装
if ! command -v node &> /dev/null; then
    echo "❌ Node.js 未安装，请先安装 Node.js (https://nodejs.org/)"
    exit 1
fi

# 检查npm是否已安装
if ! command -v npm &> /dev/null; then
    echo "❌ npm 未安装，请先安装 npm"
    exit 1
fi

echo "✅ Node.js 版本: $(node --version)"
echo "✅ npm 版本: $(npm --version)"
echo ""

# 检查package.json是否存在
if [ ! -f "package.json" ]; then
    echo "❌ 找不到 package.json 文件，请确保在正确的目录中运行此脚本"
    exit 1
fi

# 安装依赖
echo "📦 正在安装依赖..."
if npm install; then
    echo "✅ 依赖安装完成"
else
    echo "❌ 依赖安装失败"
    exit 1
fi

echo ""
echo "🚀 启动开发服务器..."
echo "📝 应用将在 http://localhost:3000 打开"
echo "⏹️  按 Ctrl+C 停止服务器"
echo ""

# 启动开发服务器
npm start
