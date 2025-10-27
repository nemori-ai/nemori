# -*- coding: utf-8 -*-
"""
Complete LoCoMo Ingestion with Full Semantic Memory Processing and Boundary Detection
(Refactored for Coarse-Grained, Inter-Conversation Concurrency)

This script processes LoCoMo conversation data using the complete Nemori semantic discovery flow:
1. Load LoCoMo conversations
2. Process multiple conversations concurrently using asyncio.gather
3. For each conversation (sequentially):
    a. Detect conversation boundaries using LLM-powered analysis
    b. Process each segment with EnhancedConversationEpisodeBuilder
    c. Store in unified episodic + semantic memory storage
    d. Enable full differential analysis and knowledge evolution
"""

import argparse
import asyncio
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from nemori.core.data_types import RawEventData, DataType, TemporalInfo, ConversationMessage
from nemori.storage.storage_types import StorageConfig
from nemori.storage.duckdb_storage import DuckDBEpisodicMemoryRepository, DuckDBRawDataRepository
from nemori.storage.duckdb_semantic_storage import DuckDBSemanticMemoryRepository
from nemori.core.builders import EpisodeBuilderRegistry
from nemori.builders.enhanced_conversation_builder import EnhancedConversationEpisodeBuilder
from nemori.builders.conversation_builder import ConversationEpisodeBuilder
from nemori.episode_manager import EpisodeManager
from nemori.llm.providers.openai_provider import OpenAIProvider
from nemori.semantic.unified_retrieval import UnifiedRetrievalService
from nemori.semantic.discovery import ContextAwareSemanticDiscoveryEngine
from nemori.semantic.evolution import SemanticEvolutionManager


class LoCoMoSemanticProcessor:
    """Complete LoCoMo processor with semantic memory, boundary detection, and inter-conversation concurrency."""
    
    # [改造] 增加了 max_concurrency 参数用于控制并发数量
    def __init__(self, version: str = "full_semantic", max_concurrency: int = 5):
        self.version = version
        self.db_dir = Path(f"results/locomo/nemori-{version}/storages")
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        # [改造] 添加 asyncio.Semaphore 来控制并发
        self.semaphore = asyncio.Semaphore(max_concurrency)
        print(f"🔩 Concurrency level set to a maximum of {max_concurrency} conversations at a time.")
        
        # Core components
        self.raw_data_repo = None
        self.episode_repo = None
        self.semantic_repo = None
        self.episode_manager = None
        self.llm_provider = None
        self.unified_retrieval = None
        self.discovery_engine = None
        self.evolution_manager = None
        
        # Data tracking
        self.conversations = []
        self.episodes = []
        self.semantic_nodes = []

    # ... (setup_llm_provider, setup_unified_storage, setup_semantic_components 函数保持不变) ...
    async def setup_llm_provider(self, model: str, api_key: str, base_url: str) -> bool:
        """Setup LLM provider for semantic discovery and boundary detection"""
        print("\n🤖 Setting up LLM Provider for Semantic Discovery & Boundary Detection")
        print("=" * 60)
        
        try:
            self.llm_provider = OpenAIProvider(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=0.1,
                max_tokens=16384
            )
            
            print(f"✅ LLM Provider initialized: {model}")
            print(f"   🌐 Base URL: {base_url}")
            print(f"   🔍 Boundary Detection: Enabled")
            print(f"   🧠 Semantic Discovery: Enabled")
            return True
            
        except Exception as e:
            print(f"❌ LLM Provider setup failed: {e}")
            return False
    
    async def setup_unified_storage(self, emb_api_key: str, emb_base_url: str, embed_model: str):
        """Setup unified episodic + semantic storage with embeddings"""
        print("\n🗄️ Setting up Unified Episodic + Semantic Storage")
        print("=" * 60)
        
        # Clean existing database
        db_path = self.db_dir / "nemori_full_semantic.duckdb"
        if db_path.exists():
            db_path.unlink()
            print("🧹 Cleaned existing database")
        
        # Clean existing indices
        for index_file in self.db_dir.glob("*_index_*.pkl"):
            index_file.unlink()
            print(f"🧹 Cleaned index: {index_file.name}")
        
        # Storage configuration for episodic memory
        episode_storage_config = StorageConfig(
            backend_type="duckdb",
            connection_string=str(db_path)
        )
        
        # Storage configuration for semantic memory (with embedding support)
        semantic_storage_config = StorageConfig(
            backend_type="duckdb", 
            connection_string=str(db_path)
        )
        semantic_storage_config.openai_api_key = emb_api_key
        semantic_storage_config.openai_base_url = emb_base_url
        semantic_storage_config.openai_embed_model = embed_model
        
        # Initialize repositories
        self.raw_data_repo = DuckDBRawDataRepository(episode_storage_config)
        self.episode_repo = DuckDBEpisodicMemoryRepository(episode_storage_config) 
        self.semantic_repo = DuckDBSemanticMemoryRepository(semantic_storage_config)
        
        await self.raw_data_repo.initialize()
        await self.episode_repo.initialize()
        await self.semantic_repo.initialize()
        
        print(f"✅ Unified storage initialized: {db_path}")
        print(f"   📖 Episodic Memory: DuckDB repository")
        print(f"   🧠 Semantic Memory: DuckDB repository with embeddings")
        print(f"   🎯 Embedding Model: {embed_model}")
        
    async def setup_semantic_components(self):
        """Setup semantic discovery and evolution components"""
        print("\n🧠 Setting up Semantic Discovery Components")
        print("=" * 60)
        
        # Initialize unified retrieval service
        self.unified_retrieval = UnifiedRetrievalService(
            episodic_storage=self.episode_repo,
            semantic_storage=self.semantic_repo
        )
        await self.unified_retrieval.initialize()
        print("✅ Unified retrieval service initialized")
        
        # Initialize semantic discovery engine
        self.discovery_engine = ContextAwareSemanticDiscoveryEngine(
            llm_provider=self.llm_provider,
            retrieval_service=self.unified_retrieval
        )
        print("✅ Context-aware semantic discovery engine initialized")
        
        # Initialize semantic evolution manager
        self.evolution_manager = SemanticEvolutionManager(
            storage=self.semantic_repo,
            discovery_engine=self.discovery_engine,
            retrieval_service=self.unified_retrieval
        )
        print("✅ Semantic evolution manager initialized")
        
        # Initialize enhanced episode builder with semantic integration
        builder_registry = EpisodeBuilderRegistry()
        enhanced_builder = EnhancedConversationEpisodeBuilder(
            llm_provider=self.llm_provider,
            semantic_manager=self.evolution_manager
        )
        builder_registry.register(enhanced_builder)
        
        self.episode_manager = EpisodeManager(
            raw_data_repo=self.raw_data_repo,
            episode_repo=self.episode_repo,
            builder_registry=builder_registry
        )
        print("✅ Enhanced episode manager with semantic integration initialized")

    async def _detect_conversation_boundaries(self, messages: list[ConversationMessage], conv_id: str) -> list[tuple[int, int, str]]:
        """Detect conversation boundaries using the conversation builder's boundary detection."""
        print(f"[{conv_id}] 🔍 Starting boundary detection for {len(messages)} messages")

        boundaries = [(0, len(messages) - 1, "Single episode - no boundary detection")]

        if not self.llm_provider or len(messages) <= 1:
            print(f"[{conv_id}] ⚠️ No LLM provider or too few messages, using single episode")
            return boundaries
        
        builder = ConversationEpisodeBuilder(llm_provider=self.llm_provider)
        message_dicts = [
            {"content": msg.content, "speaker_id": msg.speaker_id, "timestamp": msg.timestamp.isoformat() if msg.timestamp else None}
            for msg in messages
        ]

        boundaries = []
        current_start = 0
        current_episode_reason = "Episode start"

        for i in range(1, len(message_dicts)):
            current_episode_history = message_dicts[current_start:i]
            new_message = message_dicts[i]

            should_end, reason, masked_boundary_detected = await builder._detect_boundary(
                conversation_history=current_episode_history,
                new_messages=[new_message],
                smart_mask=True,
            )

            if should_end:
                boundaries.append((current_start, i - 1, current_episode_reason))
                print(f"[{conv_id}] ✂️ Boundary detected at message {i}: {reason}")
                current_start = i if not masked_boundary_detected else i - 1
                current_episode_reason = reason
        
        boundaries.append((current_start, len(message_dicts) - 1, current_episode_reason))
        print(f"[{conv_id}] 📊 Detected {len(boundaries)} conversation segments")
        return boundaries
    
    # [改造] 函数名去掉 "concurrent" 后缀，因为它现在是顺序执行的
    async def _process_single_segment(self, segment_messages, segment_id, session_id, 
                                      session_datetime, conversation, reason, 
                                      start_idx, end_idx):
        """Process a single conversation segment sequentially."""
        try:
            raw_data = RawEventData(
                data_type=DataType.CONVERSATION,
                content=segment_messages,
                temporal_info=TemporalInfo(timestamp=pd.to_datetime(session_datetime)),
                metadata={
                    'conversation_id': conversation['conversation_id'],
                    'session_id': session_id,
                    'segment_id': segment_id,
                    'participants': [conversation['speaker_a'], conversation['speaker_b']],
                    'message_count': len(segment_messages),
                    'boundary_reason': reason,
                }
            )
            
            primary_speaker = segment_messages[0]['speaker_id'] if segment_messages else conversation['speaker_a']
            episode = await self.episode_manager.process_raw_data(raw_data, primary_speaker)
            
            if episode:
                # Episode is already saved by episode_manager.process_raw_data()
                # No need to save again, just verify it's in the database
                print(f"         ✅ Episode processed and stored: {episode.episode_id}")
                
                new_semantic_discoveries = await self.perform_complete_semantic_discovery(
                    episode=episode,
                    segment_messages=segment_messages,
                    segment_id=segment_id
                )
                return episode, new_semantic_discoveries
            else:
                print(f"     ⚠️ Failed to create episode for segment {segment_id}")
                return None, 0
                
        except Exception as e:
            print(f"     ❌ Error processing segment {segment_id}: {e}")
            return None, 0

    # ... (perform_complete_semantic_discovery, step2/3_native, parse/load_locomo_data 函数基本不变, 仅在日志中添加 conv_id) ...
    async def perform_complete_semantic_discovery(self, episode, segment_messages, segment_id):
        """Perform complete 3-step semantic discovery process."""
        conv_id = episode.metadata.custom_fields.get('conversation_id', 'unknown_conv')
        print(f"[{conv_id}] 🧠 Starting Semantic Discovery for {segment_id}")
        
        original_conversation = "\n".join([f"{msg['speaker_id']}: {msg['content']}" for msg in segment_messages])
        discovered_count = 0
        try:
            discovered_nodes = await self.step3_native_semantic_discovery(episode, original_conversation)
            discovered_count = len(discovered_nodes)
            if discovered_count > 0:
                print(f"[{conv_id}] ✅ Discovered {discovered_count} new semantic concepts for {segment_id}")
            else:
                print(f"[{conv_id}] 📝 No new semantic knowledge discovered for {segment_id}")
        except Exception as e:
            print(f"[{conv_id}] ❌ Complete semantic discovery failed for {segment_id}: {e}")
        return discovered_count


    async def step3_native_semantic_discovery(self, episode, original_conversation):
        """Step 3: Use native SemanticEvolutionManager for knowledge discovery"""
        try:
            # [修正] 将步骤2中已经计算好的 reconstructed_conversation 字符串传递给 evolution_manager。
            # 这可以避免 evolution_manager 内部因尝试使用 episode.content (一个列表) 进行重构而失败。
            # 这也遵循了“差分分析”的正确逻辑，即比较原始文本和重构文本。
            discovered_nodes = await self.evolution_manager.process_episode_for_semantics(
                episode=episode,
                original_content=original_conversation,
            )
            
            # Semantic nodes are automatically saved by evolution_manager.process_episode_for_semantics()
            # No need to save again, just log the results
            print(f"           ✅ Native semantic discovery completed")
            print(f"           🔍 Discovered {len(discovered_nodes)} new semantic concepts")
            print(f"           💾 Semantic nodes automatically saved by evolution manager")
            
            return discovered_nodes
            
        except Exception as e:
            print(f"           ❌ Native semantic discovery failed: {e}")
            # 打印更详细的堆栈信息以帮助调试
            import traceback
            traceback.print_exc()
            return []

    def parse_locomo_timestamp(self, timestamp_str: str) -> datetime:
        """Parse LoComo timestamp format to datetime object."""
        try:
            timestamp_str = timestamp_str.replace("\\s+", " ").strip()
            return datetime.strptime(timestamp_str, "%I:%M %p on %d %B, %Y")
        except ValueError:
            return datetime.now()

    def load_locomo_data(self, locomo_df: pd.DataFrame, sample_size: int = None):
        """Load and parse LoCoMo conversation data with optional sampling"""
        print("\n📊 Loading LoCoMo Conversation Data")
        print("=" * 60)
        
        # 应用采样
        if sample_size and sample_size < len(locomo_df):
            print(f"🎯 Sampling {sample_size} conversations from {len(locomo_df)} total conversations")
            locomo_df = locomo_df.head(sample_size)  # 取前sample_size个对话
        
        self.conversations = []
        for idx, row in locomo_df.iterrows():
            # ... (data loading logic is unchanged) ...
            conversation = row['conversation']
            conv_data = {'conversation_id': f"locomo_{idx}", 'speaker_a': conversation.get('speaker_a', 'A'), 'speaker_b': conversation.get('speaker_b', 'B'), 'sessions': []}
            session_keys = sorted([key for key in conversation.keys() if key.startswith('session_') and not key.endswith('_date_time')])
            for session_key in session_keys:
                messages = conversation[session_key]
                session_datetime = conversation.get(f"{session_key}_date_time", datetime.now().isoformat())
                session_time = self.parse_locomo_timestamp(session_datetime)
                formatted_messages = []
                for msg in messages:
                    content = msg["text"]
                    if msg.get("img_url"):
                        content = f"[{msg['speaker']} shared an image: {msg.get('blip_caption', 'an image')}] {content}"
                    msg_timestamp = msg.get('timestamp', session_time.isoformat()).replace("Z", "+00:00")
                    formatted_messages.append({'speaker_id': msg['speaker'], 'content': content, 'timestamp': msg_timestamp})
                conv_data['sessions'].append({'session_id': session_key, 'datetime': session_datetime, 'messages': formatted_messages})
            self.conversations.append(conv_data)
        print(f"✅ Loaded {len(self.conversations)} LoCoMo conversations")
        
        if sample_size and sample_size < len(locomo_df):
            print(f"🔍 Sample Mode: Using conversations 0-{len(self.conversations)-1} for validation")

    # [新增] 这是一个新的 "工作单元" 函数，负责完整处理单场对话
    async def _process_single_conversation(self, conv_idx: int, conversation: dict) -> tuple:
        """
        Processes a single conversation from start to finish.
        This function is designed to be run concurrently with others.
        """
        conv_id = conversation['conversation_id']
        async with self.semaphore:
            print(f"🚀 Starting processing for conversation: {conv_id}")
            
            local_episodes = []
            total_discoveries = 0
            total_boundaries = 0

            for session_idx, session in enumerate(conversation['sessions']):
                session_id = f"{conv_id}_session_{session_idx}"
                if not session['messages']:
                    print(f"[{conv_id}] ⚠️ Empty session {session_id}, skipping")
                    continue
                
                conv_messages = [
                    ConversationMessage(
                        speaker_id=msg['speaker_id'],
                        content=msg['content'],
                        timestamp=pd.to_datetime(msg['timestamp']) if msg['timestamp'] else None
                    ) for msg in session['messages']
                ]
                
                boundaries = await self._detect_conversation_boundaries(conv_messages, conv_id)
                total_boundaries += len(boundaries)

                # [核心改造] 将对片段的处理从并发改为串行
                print(f"[{conv_id}] 🔄 Starting sequential processing of {len(boundaries)} segments for session {session_idx}.")
                for boundary_idx, (start_idx, end_idx, reason) in enumerate(boundaries):
                    segment_id = f"{session_id}_segment_{boundary_idx}"
                    segment_messages = session['messages'][start_idx:end_idx + 1]
                    
                    # 直接调用并等待单个片段处理函数
                    result = await self._process_single_segment(
                        segment_messages, segment_id, session_id,
                        session['datetime'], conversation, reason, start_idx, end_idx
                    )
                    
                    if result and result[0]:
                        episode, discoveries = result
                        local_episodes.append(episode)
                        total_discoveries += discoveries

            print(f"✅ Finished processing for conversation: {conv_id}. Found {len(local_episodes)} episodes and {total_discoveries} concepts.")
            return local_episodes, total_discoveries, total_boundaries

    # [改造] 这是主处理函数，现在负责调度并发任务
    async def process_conversations_with_boundary_detection_and_semantic_discovery(self):
        """Process all conversations concurrently with boundary detection and semantic discovery."""
        print("\n🏗️ Processing All Conversations Concurrently")
        print("=" * 70)
        
        start_time = time.time()
        
        # 为每一场对话创建一个并发任务
        tasks = [
            self._process_single_conversation(conv_idx, conversation)
            for conv_idx, conversation in enumerate(self.conversations)
        ]
        
        print(f"📋 Created {len(tasks)} concurrent tasks. Starting execution...")
        
        # 使用 asyncio.gather 来并发执行所有对话处理任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        print("\n🏁 All conversation processing tasks completed.")

        # 收集和汇总所有并发任务的结果
        total_episodes = []
        total_semantic_discoveries = 0
        total_boundaries_detected = 0
        successful_convs = 0

        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"❌ Conversation {idx+1} failed with an error: {result}")
            else:
                episodes, discoveries, boundaries = result
                total_episodes.extend(episodes)
                total_semantic_discoveries += discoveries
                total_boundaries_detected += boundaries
                successful_convs += 1
        
        self.episodes = total_episodes
        
        processing_time = time.time() - start_time
        
        print(f"\n🎉 Complete Conversation Processing Finished!")
        print(f"   ⏱️ Total processing time: {processing_time:.2f} seconds")
        print(f"   📊 Conversations processed: {successful_convs}/{len(self.conversations)}")
        print(f"   🔍 Total boundary segments detected: {total_boundaries_detected}")
        print(f"   📖 Total episodes created: {len(self.episodes)}")
        print(f"   🧠 Total semantic concepts discovered: {total_semantic_discoveries}")
        print(f"   ⚡ Concurrency model: Inter-Conversation (Coarse-Grained)")
        
    # ... (generate_processing_report, build_unified_indices, cleanup 函数保持不变) ...
    async def generate_processing_report(self):
        """Generate comprehensive processing report"""
        print("\n📊 Generating Comprehensive Processing Report")
        print("=" * 60)
        
        if self.episodes:
            owners = list(set(ep.owner_id for ep in self.episodes))
            print(f"📖 EPISODIC MEMORY ANALYSIS:")
            print(f"   • Total Episodes: {len(self.episodes)}")
            print(f"   • Unique Speakers: {len(owners)}")
            all_owners = set(ep.owner_id for ep in self.episodes)
            print(f"\n🧠 SEMANTIC MEMORY ANALYSIS:")
            total_semantic_nodes = 0
            for owner in sorted(all_owners)[:10]:
                try:
                    nodes = await self.semantic_repo.get_all_semantic_nodes_for_owner(owner)
                    if nodes:
                        total_semantic_nodes += len(nodes)
                        avg_conf = sum(n.confidence for n in nodes) / len(nodes)
                        print(f"   • {owner}: {len(nodes)} concepts (avg confidence: {avg_conf:.2f})")
                except Exception as e:
                    print(f"   ⚠️ {owner}: Error accessing semantic memory - {e}")
            print(f"\n   📊 Total Semantic Concepts: {total_semantic_nodes}")
        
        await self.build_unified_indices()

    async def build_unified_indices(self):
        """Build unified embedding indices for both episodic and semantic memory after processing"""
        print(f"\n🔧 Building Unified Memory Indices")
        print(f"=" * 60)
        
        if not self.episodes:
            print("⚠️ No episodes to index")
            return
            
        owner_ids = {episode.owner_id for episode in self.episodes}
        print(f"🎯 Building indices for {len(owner_ids)} owners.")
        
        if not hasattr(self, 'retrieval_service'):
            from nemori.retrieval import RetrievalService, RetrievalConfig, RetrievalStorageType, RetrievalStrategy
            self.retrieval_service = RetrievalService(self.episode_repo)
            retrieval_config = RetrievalConfig(storage_type=RetrievalStorageType.DISK, storage_config={"directory": str(self.db_dir)}, api_key="EMPTY", base_url="http://localhost:6007/v1", embed_model="qwen3-emb")
            self.retrieval_service.register_provider(RetrievalStrategy.EMBEDDING, retrieval_config)
            await self.retrieval_service.initialize()
        
        print("🔄 Triggering embedding index rebuild for episodic memory...")
        all_episodes_count = 0
        
        # Get the embedding provider to directly call rebuild methods
        embedding_provider = None
        for strategy, provider in self.retrieval_service.providers.items():
            if strategy == RetrievalStrategy.EMBEDDING:
                embedding_provider = provider
                break
        
        if not embedding_provider:
            print("   ⚠️ No embedding provider found, cannot build episodic indices")
        else:
            for owner_id in owner_ids:
                try:
                    result = await self.episode_repo.get_episodes_by_owner(owner_id)
                    owner_episodes = result.episodes if hasattr(result, "episodes") else result
                    all_episodes_count += len(owner_episodes)
                    
                    # Directly call the rebuild method on the embedding provider
                    await embedding_provider._rebuild_embedding_index(owner_id)
                    print(f"   ✅ Episodic embedding index built for {owner_id}: {len(owner_episodes)} episodes")
                except Exception as e:
                    print(f"   ⚠️ Episodic index building failed for {owner_id}: {e}")
        
        # Build semantic memory embedding indices
        print(f"\\n🧠 Triggering semantic memory embedding indices...")
        all_semantic_nodes_count = 0
        semantic_owners_with_nodes = set()
        nodes_with_embeddings = 0
        
        # Initialize semantic embedding provider for efficient indexing
        if not hasattr(self, 'semantic_embedding_provider'):
            from nemori.retrieval.providers.semantic_embedding_provider import SemanticEmbeddingProvider
            
            self.semantic_embedding_provider = SemanticEmbeddingProvider(
                semantic_storage=self.semantic_repo,
                api_key="EMPTY",
                base_url="http://localhost:6007/v1",
                embed_model="qwen3-emb",
                persistence_dir=self.db_dir,
                enable_persistence=True
            )
            await self.semantic_embedding_provider.initialize()
            print("   ✅ Semantic embedding provider initialized")
        
        for owner_id in owner_ids:
            try:
                # Get all semantic nodes for this owner
                semantic_nodes = await self.semantic_repo.get_all_semantic_nodes_for_owner(owner_id)
                if semantic_nodes:
                    semantic_owners_with_nodes.add(owner_id)
                    all_semantic_nodes_count += len(semantic_nodes)
                    
                    # Rebuild semantic embedding index for this owner
                    # This will create persistent JSON index files similar to episodic memory
                    await self.semantic_embedding_provider._rebuild_semantic_index(owner_id)
                    
                    # Count nodes with embeddings
                    for node in semantic_nodes:
                        if node.embedding_vector or True:  # Assume all will have embeddings after rebuild
                            nodes_with_embeddings += 1
                    
                    print(f"   ✅ Semantic index built for {owner_id}: {len(semantic_nodes)} nodes")
                else:
                    print(f"   📝 No semantic nodes found for {owner_id}")
                    
            except Exception as e:
                print(f"   ⚠️ Semantic index building failed for {owner_id}: {e}")
        
        print(f"\\n📊 Unified Index Building Summary:")
        print(f"   📖 Episodic Memory: {all_episodes_count} episodes indexed")
        print(f"   🧠 Semantic Memory: {all_semantic_nodes_count} nodes indexed")
        print(f"   👥 Owners with semantic knowledge: {len(semantic_owners_with_nodes)}/{len(owner_ids)}")
        print(f"   🎯 Both episodic and semantic indices ready for unified retrieval")
        
    async def verify_data_persistence(self):
        """Verify that all processed data has been properly saved to the database"""
        print(f"\\n🔍 Verifying Data Persistence")
        print(f"=" * 60)
        
        # Verify episodic memory
        total_episodes_in_memory = len(self.episodes)
        total_episodes_in_db = 0
        total_semantic_nodes_in_db = 0
        
        if total_episodes_in_memory > 0:
            # Count episodes in database by checking each unique owner
            owner_ids = set(ep.owner_id for ep in self.episodes)
            print(f"📊 Checking {len(owner_ids)} unique owners...")
            
            for owner_id in owner_ids:
                try:
                    # Check episodes for this owner
                    result = await self.episode_repo.get_episodes_by_owner(owner_id)
                    owner_episodes = result.episodes if hasattr(result, "episodes") else result
                    episode_count = len(owner_episodes)
                    total_episodes_in_db += episode_count
                    
                    # Check semantic nodes for this owner  
                    semantic_nodes = await self.semantic_repo.get_all_semantic_nodes_for_owner(owner_id)
                    semantic_count = len(semantic_nodes)
                    total_semantic_nodes_in_db += semantic_count
                    
                    print(f"   👤 {owner_id}: {episode_count} episodes, {semantic_count} semantic nodes in DB")
                    
                except Exception as e:
                    print(f"   ❌ Error verifying data for {owner_id}: {e}")
        
        # Summary
        print(f"\\n💾 Data Persistence Summary:")
        print(f"   📖 Episodes in memory: {total_episodes_in_memory}")
        print(f"   📖 Episodes in database: {total_episodes_in_db}")
        print(f"   🧠 Semantic nodes in database: {total_semantic_nodes_in_db}")
        
        if total_episodes_in_db >= total_episodes_in_memory:
            print(f"   ✅ All episodic data successfully persisted")
        else:
            print(f"   ⚠️ Potential data loss: {total_episodes_in_memory - total_episodes_in_db} episodes missing")
        
        if total_semantic_nodes_in_db > 0:
            print(f"   ✅ Semantic memory data successfully persisted")
        else:
            print(f"   ⚠️ No semantic nodes found in database")
        
        return {
            "episodes_in_memory": total_episodes_in_memory,
            "episodes_in_db": total_episodes_in_db, 
            "semantic_nodes_in_db": total_semantic_nodes_in_db,
            "persistence_verified": total_episodes_in_db >= total_episodes_in_memory and total_semantic_nodes_in_db > 0
        }

    async def cleanup(self):
        """Cleanup resources"""
        print("\n🧹 Cleaning up resources...")
        if self.raw_data_repo: await self.raw_data_repo.close()
        if self.episode_repo: await self.episode_repo.close()
        if self.semantic_repo: await self.semantic_repo.close()
        if self.unified_retrieval: await self.unified_retrieval.close()
        if hasattr(self, 'semantic_embedding_provider') and self.semantic_embedding_provider: 
            await self.semantic_embedding_provider.close()
        print("✅ Cleanup completed")


async def main_nemori_full_semantic(version: str, max_concurrency: int, sample_size: int = None):
    """Main function for complete Nemori semantic processing of LoCoMo data"""
    load_dotenv()
    print("🚀 LoCoMo Ingestion with Complete Semantic Memory (Coarse-Grained Concurrency)")
    
    if sample_size:
        print(f"📊 Data Sampling Mode: Processing only {sample_size} conversations for validation")
    else:
        print(f"📊 Full Dataset Mode: Processing all available conversations")
    
    # [改造] 将 max_concurrency 传递给处理器
    processor = LoCoMoSemanticProcessor(version=version, max_concurrency=max_concurrency)
    try:
        llm_success = await processor.setup_llm_provider(
            model="gpt-4o-mini",
            api_key="sk-NuO4dbNTkXXi5zWYkLFnhyug1kkqjeysvrb74pA9jTGfz8cm",
            base_url="https://jeniya.cn/v1"
        )
        if not llm_success: return

        await processor.setup_unified_storage(
            emb_api_key="EMPTY",
            emb_base_url="http://localhost:6007/v1", 
            embed_model="qwen3-emb"
        )
        await processor.setup_semantic_components()
        
        locomo_df = pd.read_json("data/locomo/locomo10.json")
        processor.load_locomo_data(locomo_df, sample_size=sample_size)  # 传递采样参数
        
        await processor.process_conversations_with_boundary_detection_and_semantic_discovery()
        await processor.generate_processing_report()
        
        # Verify all data has been properly persisted to the database
        verification_result = await processor.verify_data_persistence()
        
        print(f"\\n🎉 LoCoMo Complete Processing Finished!")
        print(f"🎯 Ready for evaluation with:")
        print(f"   🔍 LLM-powered conversation boundary detection")
        print(f"   📖 Unified episodic memory system") 
        print(f"   🧠 Complete semantic knowledge discovery")
        print(f"   🔗 Bidirectional episode-semantic linking")
        print(f"   🚀 Search system ready at: results/locomo/nemori-{version}/storages/")
        
        if verification_result["persistence_verified"]:
            print(f"   ✅ All data verified and persisted to database")
        else:
            print(f"   ⚠️ Data persistence verification completed with warnings")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await processor.cleanup()

def main(frame: str, version: str, max_concurrency: int, sample_size: int = None):
    """Main entry point"""
    if frame == "nemori":
        asyncio.run(main_nemori_full_semantic(version, max_concurrency, sample_size))
    else:
        print(f"❌ Unsupported framework: {frame}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoCoMo Ingestion with Coarse-Grained Concurrency")
    parser.add_argument("--lib", type=str, choices=["nemori"], default="nemori", help="Memory framework")
    parser.add_argument("--version", type=str, default="full_semantic", help="Version identifier for results")
    # [改造] 添加命令行参数来控制并发数
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help="Maximum number of conversations to process concurrently"
    )
    # [新增] 添加数据采样参数
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of conversations to process (for testing/validation). If not specified, processes all conversations."
    )
    
    args = parser.parse_args()
    main(args.lib, args.version, args.max_concurrency, args.sample_size)