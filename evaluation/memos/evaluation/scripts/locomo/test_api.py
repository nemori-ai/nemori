import os
import asyncio
import threading
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import logging

from nemori_eval.experiment import NemoriExperiment, RetrievalStrategy
from nemori_eval.search import get_nemori_unified_client, nemori_unified_search
from nemori.core.data_types import ConversationData, DataType, RawEventData, TemporalInfo
from datetime import datetime

# --- 全局配置和初始化 ---

# 加载环境变量
load_dotenv()

# 创建 Flask 应用
app = Flask(__name__)
# 配置日志
logging.basicConfig(level=logging.INFO)


# Nemori 服务类，用于封装核心逻辑
class NemoriService:
    def __init__(self):
        """
        服务初始化。
        - 创建并启动一个独立的事件循环线程，用于运行所有异步任务。
        - 初始化用于缓存 Experiment 和 Client 实例的字典。
        """
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.start_loop, daemon=True)
        self.thread.start()
        # 使用字典来缓存已初始化的实例，键为 version 字符串
        self.experiments = {}
        self.clients = {}
        app.logger.info("NemoriService initialized with caching enabled.")

    def start_loop(self):
        """启动并运行异步事件循环"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run_async(self, coro):
        """在服务的主事件循环中提交并运行一个协程"""
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    # --- 初始化函数 ---

    async def initialize_experiment_async(self, version: str, embed_config: dict, llm_config: dict):
        """
        [新增] 初始化并缓存一个 NemoriExperiment 实例。
        如果指定 version 的实例已存在，它将被新的实例覆盖。
        """
        if version in self.experiments:
            app.logger.warning(f"Re-initializing experiment for version '{version}'.")
        
        app.logger.info(f"Initializing experiment for version '{version}'...")
        experiment = NemoriExperiment(
            version=version,
            episode_mode="speaker",
            retrievalstrategy=RetrievalStrategy.EMBEDDING,
            max_concurrency=1
        )
        
        # 设置 LLM 和 Embedding
        llm_available = await experiment.setup_llm_provider(**llm_config)
        if not llm_available:
            raise RuntimeError("LLM provider is not available.")
            
        await experiment.setup_storage_and_retrieval(**embed_config)
        
        # 缓存实例
        self.experiments[version] = experiment
        app.logger.info(f"Successfully initialized and cached experiment for version '{version}'.")
        return {"status": "success", "message": f"Experiment for version '{version}' initialized."}

    async def initialize_client_async(self, version: str, embed_config: dict):
        """
        [新增] 初始化并缓存一个 Nemori 搜索客户端。
        """
        if version in self.clients:
            app.logger.warning(f"Re-initializing client for version '{version}'.")
            
        app.logger.info(f"Initializing search client for version '{version}'...")
        unified_retrieval, retrieval_service, _, _ = await get_nemori_unified_client(
            version=version,
            retrievalstrategy=RetrievalStrategy.EMBEDDING,
            emb_api_key=embed_config.get("emb_api_key", "EMPTY"),
            emb_base_url=embed_config.get("emb_base_url"),
            embed_model=embed_config.get("embed_model")
        )
        
        # 缓存客户端元组
        self.clients[version] = (unified_retrieval, retrieval_service)
        app.logger.info(f"Successfully initialized and cached client for version '{version}'.")
        return {"status": "success", "message": f"Client for version '{version}' initialized."}


    # --- 核心业务逻辑函数 (已重构) ---
    
    def _get_experiment(self, version: str) -> NemoriExperiment:
        """从缓存中获取 Experiment 实例，如果不存在则抛出异常"""
        experiment = self.experiments.get(version)
        if not experiment:
            raise ValueError(f"Experiment for version '{version}' is not initialized. Please call /api/initialize/experiment first.")
        return experiment

    def _get_client(self, version: str) -> tuple:
        """从缓存中获取 Client 实例，如果不存在则抛出异常"""
        client = self.clients.get(version)
        if not client:
            raise ValueError(f"Client for version '{version}' is not initialized. Please call /api/initialize/client first.")
        return client

    async def update_memory_asyncV2(self, version: str, conversations: list, boundaries: list):
        """异步处理记忆更新 V2 - 使用预初始化的 experiment"""
        experiment = self._get_experiment(version)
        user_id = conversations["user_id"]
        
        raw_data = experiment.convert_conversation_to_nemori(conversations, user_id)
        speakers = {msg["speaker_id"] for msg in raw_data.content if isinstance(msg, dict) and "speaker_id" in msg}

        print(f"   👥 Speakers: {list(speakers)}")

        episode_boundaries = boundaries
        all_episodes = []
        all_semantic_nodes = []
        # 注意：这里我们收集的是一个列表的列表
        #all_semantic_nodes_nested = [] 
        for speaker_id in speakers:
            episodes, semantic_nodes = await experiment._build_episodes_semantic_for_speaker(
                raw_data, speaker_id, episode_boundaries
            )
            all_semantic_nodes.extend(semantic_nodes)
            #all_semantic_nodes_nested.append(semantic_nodes)
            all_episodes.extend(episodes)
            speaker_discoveries = len(semantic_nodes) if semantic_nodes else 0
            print(f"   ✅ {speaker_id}: {len(episodes)} episodes, {speaker_discoveries} semantic_discoveries")

        total_discoveries = len(all_semantic_nodes)
        print(f"   📊 Total: {len(all_episodes)} episodes, {total_discoveries} semantic_discoveries from {len(speakers)} speakers")
        # 1. 转换 Episode 对象列表
        episodes_as_dicts = [ep.to_dict() for ep in all_episodes]
        
        # 2. 扁平化并转换嵌套的 SemanticNode 对象列表
        #    - `for sublist in all_semantic_nodes_nested` 遍历外层列表
        #    - `for node in sublist` 遍历内层列表中的每个 SemanticNode 对象
        #    - `node.to_dict()` 将每个对象转换为字典
        #    假设 SemanticNode 类也有一个 .to_dict() 方法

        nodes_as_dicts = [
            node.to_dict() 
            for sublist in all_semantic_nodes
            for node in sublist
        ]
        return {
            "status": "completed",
            "user_id": user_id,
            "episodes": episodes_as_dicts,
            "semantic_nodes": nodes_as_dicts
        }

    async def detect_boundaries_async(self, version: str, conversations: list):
        """异步处理边界检测 - 使用预初始化的 experiment"""
        experiment = self._get_experiment(version)
        
        raw_data = experiment.convert_conversation_to_nemori(conversations, conversations["user_id"])
        conversation_data = ConversationData(raw_data)
        episode_boundaries = await experiment._detect_conversation_boundaries(conversation_data.messages)
        
        return {
            "status": "completed",
            "episode_boundaries": episode_boundaries
        }

    async def update_memory_async(self, version: str, conversations: list):
        """异步处理记忆更新 - 使用预初始化的 experiment"""
        experiment = self._get_experiment(version)
        
        experiment.conversations = conversations
        app.logger.info(f"[{version}] Loaded {len(conversations)} conversations for memory update.")

        await experiment.build_episodes_semantic()
        app.logger.info(f"[{version}] Memory update complete. Created {len(experiment.episodes)} episodes.")
        
        return {
            "status": "completed", 
            "episodes_created": len(experiment.episodes),
            "semantic_concepts": getattr(experiment, 'actual_semantic_count', 0)
        }

    async def query_memory_async(self, version: str, user_id: str, query: str, speaker_a: str, speaker_b: str, top_k: int):
        """异步处理记忆查询 - 使用预初始化的 client"""
        client = self._get_client(version)
        
        speaker_a_user_id = f"{speaker_a.lower().replace(' ', '_')}_{user_id}"
        speaker_b_user_id = f"{speaker_b.lower().replace(' ', '_')}_{user_id}"
        
        app.logger.info(f"[{version}] Performing search for query: '{query}'")
        context, duration_ms = await nemori_unified_search(
            unified_retrieval_service=client[0],
            retrieval_service=client[1],
            query=query,
            speaker_a_user_id=speaker_a_user_id,
            speaker_b_user_id=speaker_b_user_id,
            top_k=top_k
        )
        return context, duration_ms

# 实例化服务
nemori_service = NemoriService()

# --- API 路由定义 ---

# --- [新增] 初始化路由 ---
@app.route('/api/initialize/experiment', methods=['POST'])
def initialize_experiment():
    """初始化并缓存一个 Experiment 实例"""
    data = request.json
    version = data['version']
    llm_config = data['llm_config']
    embed_config = data['embed_config']
    
    result = nemori_service.run_async(
        nemori_service.initialize_experiment_async(version, embed_config, llm_config)
    )
    return jsonify(result), 200


@app.route('/api/initialize/client', methods=['POST'])
def initialize_client():
    """初始化并缓存一个搜索客户端实例"""
    data = request.json
    version = data['version']
    embed_config = data['embed_config']
    
    result = nemori_service.run_async(
        nemori_service.initialize_client_async(version, embed_config)
    )
    return jsonify(result), 200



# --- 业务逻辑路由 (已修改) ---

@app.route('/api/boundaries/detect', methods=['POST'])
def detect_boundaries():
    """检测对话边界"""
    data = request.json
    version = data['version']
    messages = data['messages']
    result = nemori_service.run_async(
        nemori_service.detect_boundaries_async(version, messages)
    )
    return jsonify(result), 200


@app.route('/api/memory/update-v2', methods=['POST'])
def update_memory_v2():
    """使用预分割的消息直接更新记忆库"""
    data = request.json
    version = data['version']
    messages = data['messages']
    boundaries = data['boundaries']
    result = nemori_service.run_async(
        nemori_service.update_memory_asyncV2(version, messages, boundaries)
    )
    return jsonify(result), 200



@app.route('/api/memory/update', methods=['POST'])
def update_memory():
    """接收对话数据，更新记忆库"""
    data = request.json
    version = data['version']
    conversations = data['conversations']
    
    # 检查 experiment 是否已初始化 (在同步代码中)
    nemori_service._get_experiment(version)

    # 提交到后台执行
    asyncio.run_coroutine_threadsafe(
        nemori_service.update_memory_async(version, conversations),
        nemori_service.loop
    )
    return jsonify({
        "status": "success",
        "message": "Memory update process started successfully.",
        "version": version
    }), 202
  


@app.route('/api/memory/query', methods=['POST'])
def query_memory():
    """查询记忆库"""
    data = request.json
    version = data['version']
    # ... (其余参数)
    context, duration_ms = nemori_service.run_async(
        nemori_service.query_memory_async(
            version, data['user_id'], data['query'], 
            data['speaker_a'], data['speaker_b'], data.get('top_k', 20)
        )
    )
    return jsonify({
        "status": "success",
        "query": data['query'],
        "context": context,
        "duration_ms": duration_ms
    }), 200



@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)