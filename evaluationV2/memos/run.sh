#!/bin/bash

# --- 配置 ---
export OPENAI_API_KEY="sk-xxxxx"
export OPENAI_BASE_URL="https://xxxx/v1"
export CHAT_MODEL="gpt-4.1-mini"
export EVAL_VERSION="episode_semantic_4.1miniv2"

# 创建日志目录
LOG_DIR="logs/${EVAL_VERSION}"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "========================================================="
echo "开始评估流程，版本: ${EVAL_VERSION}"
echo "所有日志将保存在目录: ${LOG_DIR}"
echo "========================================================="

# --- 运行评估脚本并记录日志 ---
cd evaluation

echo "--> 正在运行 Ingestion 脚本..."
python scripts/locomo/locomo_ingestion_emb.py --lib nemori --version $EVAL_VERSION > "${LOG_DIR}/ingestion_${TIMESTAMP}.log" 2>&1

echo "--> 正在运行 Search 脚本..."
python scripts/locomo/locomo_search.py --lib nemori --version $EVAL_VERSION > "${LOG_DIR}/search_${TIMESTAMP}.log" 2>&1

echo "--> 正在运行 Responses 脚本..."
python scripts/locomo/locomo_responses.py --lib nemori --version $EVAL_VERSION > "${LOG_DIR}/responses_${TIMESTAMP}.log" 2>&1

echo "--> 正在运行 Eval 脚本..."
python scripts/locomo/locomo_eval.py --lib nemori --workers 10 --version $EVAL_VERSION > "${LOG_DIR}/eval_${TIMESTAMP}.log" 2>&1

echo "========================================================="
echo "评估流程已完成！"
echo "========================================================="