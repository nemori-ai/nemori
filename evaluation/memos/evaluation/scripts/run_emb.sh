# 确保路径和权限正确
#MODEL_PATH="/data/jcxy/llm_model/PsyLLM-test3-o1-7B-Qwen"
MODEL_PATH="/data2/jcxy/llm_model/Qwen3-Embedding-0.6B"
LOG_FILE="/data/jcxy/haolu/workspace/tools/vllm3-embed-qwen3.log"
SERVED_MODEL_NAME="qwen3-emb"
# export VLLM_ATTENTION_BACKEND=FLASHINFER
export CUDA_VISIBLE_DEVICES=0
# export VLLM_USE_V1=1
# export VLLM_CUDA_MEM_ALIGN_KV_CACHE=1
# export VLLM_WORKER_MULTIPROC_METHOD="spawn"
# 运行命令
nohup  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --trust-remote-code \
    --task embed \
    --port 6007 \
    --host localhost \
    --dtype auto \
    --max-model-len 32768 \
    --gpu_memory_utilization 0.1 \
    --tensor_parallel_size 1 \
    >"$LOG_FILE" 2>&1 &
