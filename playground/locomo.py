#!/usr/bin/env python3
"""
LoComo Dataset Episode Building and Retrieval Experiment
基于 LoComo 数据集的情景构建和检索实验

This script demonstrates the complete workflow of:
1. Loading and preprocessing LoComo conversation data
2. Building episodes using EpisodeManager + ConversationEpisodeBuilder with boundary detection
3. Supporting two episode construction modes: agent perspective and speaker perspective
4. Storing episodes in DuckDB storage
5. Setting up BM25 retrieval provider
6. Performing retrieval experiments

本脚本演示了以下完整工作流程：
1. 加载和预处理 LoComo 对话数据
2. 使用 EpisodeManager + ConversationEpisodeBuilder 构建情景，包含边界检测
3. 支持两种情景构建模式：智能体视角和说话者视角
4. 在 DuckDB 存储中存储情景
5. 设置 BM25 检索提供者
6. 执行检索实验
"""

import asyncio
import json
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Import nemori modules after setting up the path
from nemori.builders.conversation_builder import ConversationEpisodeBuilder
from nemori.core.builders import EpisodeBuilderRegistry
from nemori.core.data_types import ConversationData, DataType, RawEventData, TemporalInfo
from nemori.episode_manager import EpisodeManager
from nemori.llm.providers.openai_provider import OpenAIProvider
from nemori.retrieval import (
    RetrievalConfig,
    RetrievalQuery,
    RetrievalService,
    RetrievalStorageType,
    RetrievalStrategy,
)
from nemori.storage.duckdb_storage import (
    DuckDBEpisodicMemoryRepository,
    DuckDBRawDataRepository,
)
from nemori.storage.storage_types import StorageConfig

# Add the parent directory to the path so we can import nemori
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class LoCoMoExperiment:
    """
    Complete LoComo dataset episode building and retrieval experiment.
    完整的 LoComo 数据集情景构建和检索实验。
    """

    def __init__(
        self,
        data_dir: str = "dataset",
        output_dir: str = "output",
        db_dir: str = ".tmp",
        episode_mode: str = "agent",
        max_concurrency: int = 3,
    ):
        """
        Initialize the experiment with directories.

        Args:
            data_dir: Directory containing LoComo dataset
            output_dir: Directory for output files
            db_dir: Directory for database files
            episode_mode: Episode construction mode ('agent' or 'speaker')
            max_concurrency: Maximum number of concurrent speaker processing tasks (default: 3)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.db_dir = Path(db_dir)
        self.episode_mode = episode_mode
        self.max_concurrency = max_concurrency

        # Validate episode mode
        if self.episode_mode not in ["agent", "speaker"]:
            raise ValueError("episode_mode must be 'agent' or 'speaker'")

        # Validate concurrency limit
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")

        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.db_dir.mkdir(exist_ok=True)

        # Storage components
        self.raw_data_repo = None
        self.episode_repo = None
        self.retrieval_service = None
        self.bm25_provider = None
        self.episode_manager = None

        # Experiment data
        self.conversations = []
        self.episodes = []
        self.llm_provider = None

        print("🚀 LoComo Experiment Initialized | LoComo 实验初始化完成")
        print(f"📁 Data directory: {self.data_dir.absolute()}")
        print(f"📁 Output directory: {self.output_dir.absolute()}")
        print(f"📁 Database directory: {self.db_dir.absolute()}")
        print(f"🎭 Episode mode: {self.episode_mode} perspective | 情景模式: {self.episode_mode} 视角")
        print(f"🔄 Max concurrency: {self.max_concurrency} | 最大并发数: {self.max_concurrency}")

    async def setup_llm_provider(self) -> bool:
        """Setup OpenAI LLM provider if API key is available."""
        print("\n🤖 Setting up LLM Provider | 设置 LLM 提供者")
        print("=" * 50)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️ OPENAI_API_KEY not found in environment")
            print("⚠️ 环境中未找到 OPENAI_API_KEY")
            print("💡 Option 1: Add to .env file in project root: OPENAI_API_KEY=your_key_here")
            print("💡 Option 2: Set environment variable: export OPENAI_API_KEY=your_key_here")
            print("💡 选项1: 在项目根目录的 .env 文件中添加: OPENAI_API_KEY=your_key_here")
            print("💡 选项2: 设置环境变量: export OPENAI_API_KEY=your_key_here")
            return False

        try:
            # self.llm_provider = GeminiProvider(model="gemini-2.5-flash", temperature=0.3, max_tokens=16 * 1024)
            self.llm_provider = OpenAIProvider(model="gpt-4o-mini", temperature=0.3, max_tokens=16 * 1024)

            if await self.llm_provider.test_connection():
                print("✅ OpenAI connection successful!")
                print("✅ OpenAI 连接成功！")
                print(f"🎯 Model: {self.llm_provider.model}")
                print(f"🌡️ Temperature: {self.llm_provider.temperature}")
                print(f"📊 Max tokens: {self.llm_provider.max_tokens}")
                return True
            else:
                print("❌ OpenAI connection failed!")
                print("❌ OpenAI 连接失败！")
                return False

        except Exception as e:
            print(f"❌ Error setting up OpenAI provider: {e}")
            print(f"❌ 设置 OpenAI 提供者时出错: {e}")
            return False

    def load_locomo_data(self, filename: str = "locomo10.json", sample_size: int = 10) -> list[dict]:
        """Load and sample LoComo dataset."""
        print("\n📚 Loading LoComo Data | 加载 LoComo 数据")
        print("=" * 50)

        file_path = self.data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        # Sample random conversations
        if sample_size < len(data):
            sampled_indices = random.sample(range(len(data)), sample_size)
            sampled_conversations = [data[i] for i in sampled_indices]
        else:
            sampled_conversations = data

        self.conversations = sampled_conversations

        print(f"📊 Total conversations available: {len(data)}")
        print(f"📊 可用对话总数: {len(data)}")
        print(f"🎯 Sampled for experiment: {len(sampled_conversations)}")
        print(f"🎯 实验采样数: {len(sampled_conversations)}")

        # Show first conversation structure
        if sampled_conversations:
            first_conv = sampled_conversations[0]
            conv_data = first_conv["conversation"]
            session_keys = [
                key for key in conv_data.keys() if key.startswith("session_") and not key.endswith("_date_time")
            ]
            total_messages = sum(len(conv_data[session_key]) for session_key in session_keys)

            print("📝 Sample conversation structure:")
            print(f"   Speaker A: {conv_data.get('speaker_a', 'Unknown')}")
            print(f"   Speaker B: {conv_data.get('speaker_b', 'Unknown')}")
            print(f"   Sessions: {len(session_keys)}")
            print(f"   Total messages: {total_messages}")

        return sampled_conversations

    def parse_locomo_timestamp(self, timestamp_str: str) -> datetime:
        """Parse LoComo timestamp format to datetime object."""
        try:
            # Clean and normalize timestamp
            timestamp_str = re.sub(r"\s+", " ", timestamp_str.strip())

            # Parse: "1:56 pm on 8 May, 2023"
            dt = datetime.strptime(timestamp_str, "%I:%M %p on %d %B, %Y")
            return dt
        except ValueError as e:
            print(f"⚠️ Warning: Could not parse timestamp '{timestamp_str}': {e}")
            return datetime.now()

    def convert_locomo_to_nemori(self, conversation_data: dict, conversation_id: str) -> RawEventData:
        """Convert LoComo conversation format to Nemori RawEventData format."""
        print(f"\n   🔄 Converting LoComo conversation {conversation_id}...")
        messages = []
        conv = conversation_data["conversation"]

        # Get all session keys in order
        session_keys = sorted(
            [key for key in conv.keys() if key.startswith("session_") and not key.endswith("_date_time")]
        )

        print(f"   📅 Found {len(session_keys)} sessions")
        print(f"   🎭 Speakers: {conv.get('speaker_a', 'Unknown')} & {conv.get('speaker_b', 'Unknown')}")

        # Generate unique speaker IDs for this conversation
        speaker_name_to_id = {}
        for session_key in session_keys:
            session_messages = conv[session_key]
            session_time_key = f"{session_key}_date_time"

            if session_time_key in conv:
                # Parse session timestamp
                session_time = self.parse_locomo_timestamp(conv[session_time_key])

                # Process each message in this session
                for i, msg in enumerate(session_messages):
                    # Generate timestamp for this message (session time + message offset)
                    msg_timestamp = session_time + timedelta(seconds=i * 30)  # 30 seconds between messages
                    iso_timestamp = msg_timestamp.isoformat()

                    # Generate unique speaker_id for this conversation
                    speaker_name = msg["speaker"]
                    if speaker_name not in speaker_name_to_id:
                        # Generate unique ID: {name}_{conversation_index}
                        unique_id = f"{speaker_name.lower().replace(' ', '_')}_{conversation_id}"
                        speaker_name_to_id[speaker_name] = unique_id

                    # Process content with image information if present
                    content = msg["text"]
                    if "img_url" in msg and msg["img_url"]:
                        blip_caption = msg.get("blip_caption", "an image")
                        content = f"[{speaker_name} shared an image: {blip_caption}] {content}"

                    message = {
                        "speaker_id": speaker_name_to_id[speaker_name],
                        "user_name": speaker_name,
                        "content": content,
                        "timestamp": iso_timestamp,
                        "original_timestamp": conv[session_time_key],
                        "dia_id": msg["dia_id"],
                        "session": session_key,
                    }

                    # Add optional fields if present
                    for optional_field in ["img_url", "blip_caption", "query"]:
                        if optional_field in msg:
                            message[optional_field] = msg[optional_field]

                    messages.append(message)

        print(f"   ✅ Converted {len(messages)} messages from {len(session_keys)} sessions")

        # Calculate total duration based on session lengths (more realistic)
        if messages:
            first_time = datetime.fromisoformat(messages[0]["timestamp"])

            # Calculate duration as total session time rather than span across all sessions
            # Each session gets estimated duration based on message count
            session_durations = {}
            for msg in messages:
                session = msg["session"]
                if session not in session_durations:
                    session_durations[session] = 0
                session_durations[session] += 1  # Count messages per session

            # Estimate duration: 30 seconds per message + 5 minutes setup per session
            total_duration = sum(msg_count * 30 + 300 for msg_count in session_durations.values())
            duration = total_duration
        else:
            duration = 0.0
            first_time = datetime.now()

        temporal_info = TemporalInfo(timestamp=first_time, duration=duration, timezone="UTC")

        return RawEventData(
            data_type=DataType.CONVERSATION,
            content=messages,
            source="locomo_dataset",
            temporal_info=temporal_info,
            metadata={
                "conversation_id": conversation_id,
                "sample_id": conversation_data.get("sample_id", "unknown"),
                "speaker_a": conv.get("speaker_a"),
                "speaker_b": conv.get("speaker_b"),
                "participant_count": 2,
                "session_count": len(session_keys),
                "message_count": len(messages),
                "has_images": any("img_url" in msg for msg in messages),
                "original_format": "locomo_multi_session",
                "episode_mode": self.episode_mode,
            },
        )

    async def setup_storage_and_retrieval(self):
        """Setup DuckDB storage and BM25 retrieval service."""
        print("\n🗄️ Setting up Storage and Retrieval | 设置存储和检索")
        print("=" * 50)

        # Setup DuckDB storage
        db_path = self.db_dir / "nemori_memory.duckdb"

        # Remove existing database to start fresh
        if db_path.exists():
            db_path.unlink()
            print("🧹 Cleaned existing database")

        # Create storage configurations
        storage_config = StorageConfig(
            backend_type="duckdb",
            connection_string=str(db_path),
            batch_size=100,
            cache_size=1000,
            enable_semantic_search=False,
        )

        # Initialize repositories
        self.raw_data_repo = DuckDBRawDataRepository(storage_config)
        self.episode_repo = DuckDBEpisodicMemoryRepository(storage_config)

        await self.raw_data_repo.initialize()
        await self.episode_repo.initialize()

        print(f"✅ DuckDB storage initialized: {db_path}")
        print(f"✅ DuckDB 存储初始化完成: {db_path}")

        # Setup BM25 retrieval service
        self.retrieval_service = RetrievalService(self.episode_repo)

        # Create BM25 retrieval provider configuration with disk storage
        retrieval_config = RetrievalConfig(
            storage_type=RetrievalStorageType.DISK,
            storage_config={"directory": str(self.db_dir)},  # Use same .tmp directory
        )

        # Register the provider with the service (service creates provider internally)
        self.retrieval_service.register_provider(RetrievalStrategy.BM25, retrieval_config)

        # Initialize the retrieval service
        await self.retrieval_service.initialize()

        # Get the provider instance for direct access
        self.bm25_provider = self.retrieval_service.get_provider(RetrievalStrategy.BM25)

        print("✅ BM25 retrieval service configured")
        print("✅ BM25 检索服务配置完成")

        # Setup episode manager
        builder_registry = EpisodeBuilderRegistry()
        if self.llm_provider:
            conversation_builder = ConversationEpisodeBuilder(llm_provider=self.llm_provider)
        else:
            conversation_builder = ConversationEpisodeBuilder()

        builder_registry.register(conversation_builder)

        self.episode_manager = EpisodeManager(
            raw_data_repo=self.raw_data_repo,
            episode_repo=self.episode_repo,
            builder_registry=builder_registry,
            retrieval_service=self.retrieval_service,
        )

        # Note: DuckDB handles concurrent writes internally, no additional semaphore needed

        print("✅ Episode manager initialized")
        print("✅ 情景管理器初始化完成")

    async def _build_episodes_agent_mode(self, raw_data: RawEventData) -> list:
        """Build episodes using agent perspective (unified observer)."""
        owner_id = "agent"

        print(f"   👤 Owner: {owner_id}")
        print(f"   💬 Messages: {len(raw_data.content)}")
        print(f"   🕐 Duration: {raw_data.temporal_info.duration:.0f} seconds")

        # Use boundary detection to split conversation into episodes
        episodes = await self._build_episodes_with_boundary_detection(raw_data, owner_id)

        return episodes

    async def _build_episodes_speaker_mode(self, raw_data: RawEventData) -> list:
        """Build episodes using speaker perspective (each speaker gets their own episodes)."""
        # Get unique speakers from the conversation
        speakers = {msg["speaker_id"] for msg in raw_data.content if isinstance(msg, dict) and "speaker_id" in msg}

        print(f"   👥 Speakers: {list(speakers)}")
        print(f"   💬 Messages: {len(raw_data.content)}")
        print(f"   🕐 Duration: {raw_data.temporal_info.duration:.0f} seconds")

        # Detect conversation boundaries once for all speakers
        conversation_data = ConversationData(raw_data)
        episode_boundaries = await self._detect_conversation_boundaries(conversation_data.messages)

        all_episodes = []

        # Process speakers sequentially (only 2 speakers, concurrency not needed)
        for speaker_id in speakers:
            episodes = await self._build_episodes_for_speaker(raw_data, speaker_id, episode_boundaries)
            all_episodes.extend(episodes)
            print(f"   ✅ {speaker_id}: {len(episodes)} episodes")

        print(f"   📊 Total: {len(all_episodes)} episodes from {len(speakers)} speakers")
        return all_episodes

    async def _build_episodes_with_boundary_detection(self, raw_data: RawEventData, owner_id: str) -> list:
        """Build episodes using boundary detection to split conversations."""
        # Convert to ConversationData for boundary detection
        conversation_data = ConversationData(raw_data)

        # If no LLM provider, fallback to single episode per conversation
        if not self.llm_provider:
            episode = await self.episode_manager.process_raw_data(raw_data, owner_id)
            return [episode] if episode else []

        # Detect conversation boundaries once
        episode_boundaries = await self._detect_conversation_boundaries(conversation_data.messages)

        # Create episodes for this owner using the detected boundaries
        return await self._build_episodes_for_speaker(raw_data, owner_id, episode_boundaries)

    async def _detect_conversation_boundaries(self, messages: list) -> list[tuple[int, int, str]]:
        """Detect conversation boundaries using the conversation builder's boundary detection."""
        print(f"\n     🔍 Starting boundary detection for {len(messages)} messages")

        boundaries = [(0, len(messages) - 1, "Single episode - no boundary detection")]  # Default: single episode

        if not self.llm_provider or len(messages) <= 1:
            print("     ⚠️ No LLM provider or too few messages, using single episode")
            return boundaries

        print("     ⚠️ Note: Boundary detection uses sequential LLM calls - this may appear slower than expected")

        # Create a conversation builder for boundary detection
        builder = ConversationEpisodeBuilder(llm_provider=self.llm_provider)

        # Convert messages to the format expected by boundary detection
        message_dicts = []
        for msg in messages:
            message_dict = {
                "content": msg.content,
                "speaker_id": msg.speaker_id,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
            }
            message_dicts.append(message_dict)

        print("\n     🔍 Starting boundary detection analysis...")

        # Detect boundaries by checking each message against conversation history
        boundaries = []
        current_start = 0
        current_episode_reason = "conversation start"

        for i in range(1, len(message_dicts)):
            # Check if we should end the current episode at this message
            # Use only the current episode's history (from current_start to i-1)
            current_episode_history = message_dicts[current_start:i]
            new_message = message_dicts[i]

            # Use async boundary detection
            should_end, reason, masked_boundary_detected = await builder._detect_boundary(
                conversation_history=current_episode_history,
                new_messages=[new_message],
                smart_mask=True,
            )

            if should_end:
                # End current episode and start new one
                boundaries.append((current_start, i - 1, current_episode_reason))
                print(f"     ✂️ Boundary at message {i}: {reason}, masked_boundary_detected: {masked_boundary_detected}")
                current_start = i if not masked_boundary_detected else i - 1
                current_episode_reason = reason  # The reason becomes the context for the next episode

        # Add the final segment
        boundaries.append((current_start, len(message_dicts) - 1, "conversation end"))

        print(f"     📊 Detected {len(boundaries)} conversation segments")

        return boundaries

    def _calculate_segment_duration(self, messages: list) -> float:
        """Calculate the duration of a message segment."""
        if len(messages) < 2:
            return 300.0  # Default 5 minutes for single message

        first_msg = messages[0]
        last_msg = messages[-1]

        if first_msg.timestamp and last_msg.timestamp:
            duration = (last_msg.timestamp - first_msg.timestamp).total_seconds()
            return max(duration, 60.0)  # Minimum 1 minute
        else:
            # Estimate based on message count (30 seconds per message)
            return len(messages) * 30.0

    async def _build_episodes_for_speaker(
        self, raw_data: RawEventData, owner_id: str, episode_boundaries: list[tuple[int, int, str]]
    ) -> list:
        """Build episodes for a specific speaker using pre-detected boundaries."""
        conversation_data = ConversationData(raw_data)
        messages = conversation_data.messages
        episodes = []

        if not messages:
            return episodes

        # Create episodes for each boundary segment
        for start_idx, end_idx, boundary_reason in episode_boundaries:
            segment_messages = messages[start_idx : end_idx + 1]

            # Create a new RawEventData for this segment
            segment_raw_data = RawEventData(
                data_type=DataType.CONVERSATION,
                content=[
                    {
                        "speaker_id": msg.speaker_id,
                        "user_name": msg.user_name,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                        "metadata": msg.metadata,
                    }
                    for msg in segment_messages
                ],
                source=raw_data.source,
                temporal_info=TemporalInfo(
                    timestamp=segment_messages[0].timestamp or raw_data.temporal_info.timestamp,
                    duration=self._calculate_segment_duration(segment_messages),
                    timezone=raw_data.temporal_info.timezone,
                ),
                metadata={
                    **raw_data.metadata,
                    "segment_start": start_idx,
                    "segment_end": end_idx,
                    "total_segments": len(episode_boundaries),
                    "owner_id": owner_id,
                    "boundary_reason": boundary_reason,
                },
            )

            # Process the segment through episode manager
            episode = await self.episode_manager.process_raw_data(segment_raw_data, owner_id)
            if episode:
                episodes.append(episode)

        return episodes

    async def build_episodes(self):
        """Build episodes from LoComo conversations using boundary detection."""
        print("\n🏗️ Building Episodes with Boundary Detection | 构建情景（含边界检测）")
        print("=" * 50)
        print(f"🎭 Mode: {self.episode_mode} perspective | 模式: {self.episode_mode} 视角")
        print(f"🔄 Processing {len(self.conversations)} conversations with max concurrency: {self.max_concurrency}")

        self.episodes = []

        async def process_conversation(conv_index: int, conv_data: dict) -> tuple[str, list]:
            """Process a single conversation with concurrency control."""
            conv_id = str(conv_index)

            start_time = time.time()
            print(f"   🚀 Starting conversation {conv_id} processing... [{start_time:.1f}]")
            try:
                # Convert to Nemori format (can be concurrent)
                raw_data = self.convert_locomo_to_nemori(conv_data, conv_id)

                if self.episode_mode == "agent":
                    # Agent perspective: single owner observes all conversations
                    episodes = await self._build_episodes_agent_mode(raw_data)
                elif self.episode_mode == "speaker":
                    # Speaker perspective: each speaker has their own episodes
                    episodes = await self._build_episodes_speaker_mode(raw_data)

                end_time = time.time()
                duration = end_time - start_time
                print(f"   ✅ Conversation {conv_id}: {len(episodes)} episodes [{end_time:.1f}, took {duration:.1f}s]")
                return conv_id, episodes

            except Exception as e:
                print(f"   ❌ Error processing conversation {conv_id}: {e}")
                return conv_id, []

        # Create tasks for all conversations
        tasks = [process_conversation(i, conv_data) for i, conv_data in enumerate(self.conversations)]
        print(f"   📋 Created {len(tasks)} concurrent tasks")

        # Wait for all tasks to complete
        print("   ⏳ Starting concurrent execution...")
        results = await asyncio.gather(*tasks)
        print("   🏁 All conversations completed")

        # Collect all episodes
        for _, episodes in results:
            self.episodes.extend(episodes)

        print("\n📊 Episode Building Complete | 情景构建完成")
        print(f"✅ Successfully created {len(self.episodes)} episodes")
        print(f"✅ 成功创建 {len(self.episodes)} 个情景")

    async def perform_retrieval_experiments(self):
        """Perform comprehensive retrieval experiments."""
        print("\n🔍 Retrieval Experiments | 检索实验")
        print("=" * 50)

        if not self.episodes:
            print("❌ No episodes available for retrieval experiments")
            return

        # Test queries for different scenarios
        test_queries = [
            "What creative achievements and personal reflections did Nate and Joanna share during their conversation on March 10, 2024?",
            "travel and adventure",
            "health and wellness",
            "friends and relationships",
        ]

        retrieval_results = {}

        # Get all unique owners based on episode mode
        owners = {episode.owner_id for episode in self.episodes}
        print(f"🎭 Testing retrieval for {len(owners)} owners: {list(owners)}")

        for query_text in test_queries:
            print(f"\n🎯 Query: '{query_text}'")
            print(f"🎯 查询: '{query_text}'")

            for owner_id in owners:
                try:
                    # Create retrieval query for each owner
                    query = RetrievalQuery(text=query_text, owner_id=owner_id, limit=5, strategy=RetrievalStrategy.BM25)

                    # Use BM25 provider directly for more detailed results
                    if self.bm25_provider:
                        result = await self.bm25_provider.search(query)
                        print(
                            f"   🔍 Raw result for {owner_id}: {len(result.episodes)} episodes, {len(result.scores)} scores"
                        )

                        # Set relevance threshold for meaningful matches
                        relevance_threshold = 1.0  # Only scores > 1.0 are considered relevant hits

                        # Filter episodes by relevance score
                        relevant_episodes = []
                        relevant_scores = []

                        # Ensure both arrays have the same length
                        min_length = min(len(result.episodes), len(result.scores))
                        for i in range(min_length):
                            episode = result.episodes[i]
                            score = result.scores[i]
                            if score > relevance_threshold:
                                relevant_episodes.append(episode)
                                relevant_scores.append(score)

                        if relevant_episodes:
                            print(
                                f"   👤 Owner {owner_id}: Found {len(relevant_episodes)} relevant matches (threshold > {relevance_threshold})"
                            )
                            print(f"      Query time: {result.query_time_ms:.1f}ms")
                            print(f"      Total candidates: {result.total_candidates}")

                            for j in range(min(3, len(relevant_episodes), len(relevant_scores))):
                                episode = relevant_episodes[j]
                                score = relevant_scores[j]
                                print(f"      {j + 1}. {episode.title[:50]}... (BM25 score: {score:.3f})")

                            # Store successful retrieval
                            retrieval_results[f"{query_text}_{owner_id}"] = {
                                "query": query_text,
                                "owner_id": owner_id,
                                "relevant_episodes": len(relevant_episodes),
                                "scores": relevant_scores[:3],
                                "total_candidates": result.total_candidates,
                                "query_time_ms": result.query_time_ms,
                            }
                        else:
                            print(
                                f"   👤 Owner {owner_id}: No relevant matches found (threshold > {relevance_threshold})"
                            )
                            print(f"      Query time: {result.query_time_ms:.1f}ms")
                            print(f"      Total candidates: {result.total_candidates}")
                            if result.episodes:
                                best_score = max(result.scores) if result.scores else 0.0
                                print(f"      Best score: {best_score:.3f} (below threshold)")

                except Exception as e:
                    print(f"   ❌ Error in retrieval for '{query_text}' with owner {owner_id}: {e}")

        # Test BM25 provider statistics
        if self.bm25_provider:
            print("\n📊 BM25 Provider Statistics | BM25 提供者统计")
            print("─" * 40)

            try:
                stats = await self.bm25_provider.get_stats()
                print(f"Total episodes indexed: {stats.total_episodes}")
                print(f"Total documents: {stats.total_documents}")
                print(f"Index size: {stats.index_size_mb:.2f} MB")
                print(f"Last updated: {stats.last_updated}")

                if hasattr(stats, "provider_stats") and stats.provider_stats:
                    print("Provider-specific stats:")
                    for key, value in stats.provider_stats.items():
                        print(f"  {key}: {value}")

            except Exception as e:
                print(f"❌ Error getting BM25 stats: {e}")

        # Summary of retrieval performance
        print("\n📈 Retrieval Summary | 检索总结")
        print("─" * 30)

        total_queries = len(test_queries) * len(owners)
        successful_queries = len(retrieval_results)  # Only queries with relevant results are stored

        print(f"Total queries tested: {total_queries}")
        print(f"Successful retrievals (score > 1.0): {successful_queries}")
        print(f"Success rate: {successful_queries / total_queries * 100:.1f}%")

        if retrieval_results:
            print("\n🎯 Successful Query Details:")
            for result in retrieval_results.values():
                print(
                    f"   '{result['query']}' ({result['owner_id']}): {result['relevant_episodes']} episodes, avg score: {sum(result['scores'])/len(result['scores']):.2f}"
                )

        return retrieval_results

    async def analyze_results(self):
        """Analyze and visualize experiment results."""
        print("\n📊 Results Analysis | 结果分析")
        print("=" * 50)

        if not self.episodes:
            print("❌ No episodes to analyze")
            return

        # Basic statistics
        total_episodes = len(self.episodes)
        total_content_length = sum(len(ep.content) for ep in self.episodes)
        avg_content_length = total_content_length / total_episodes

        # Level distribution
        level_counts = Counter([ep.level.name for ep in self.episodes])

        # Owner distribution
        owner_counts = Counter([ep.owner_id for ep in self.episodes])

        print("📈 Episode Statistics | 情景统计:")
        print(f"   Total episodes: {total_episodes}")
        print(f"   Average content length: {avg_content_length:,.0f} characters")
        print(f"   Unique owners: {len(owner_counts)}")

        print("\n🏷️ Episode Levels | 情景级别:")
        for level, count in level_counts.items():
            percentage = (count / total_episodes) * 100
            print(f"   {level}: {count} ({percentage:.1f}%)")

        print("\n👥 Top Owners | 主要用户:")
        for owner, count in owner_counts.most_common(20):
            print(f"   {owner}: {count} episodes")

        # Storage statistics
        raw_stats = await self.raw_data_repo.get_stats()
        episode_stats = await self.episode_repo.get_stats()

        print("\n💾 Storage Statistics | 存储统计:")
        print(f"   Raw data entries: {raw_stats.total_raw_data}")
        print(f"   Processed entries: {raw_stats.processed_raw_data}")
        print(f"   Episodes stored: {episode_stats.total_episodes}")
        print(f"   Total storage: {raw_stats.storage_size_mb + episode_stats.storage_size_mb:.2f} MB")

        # Retrieval index statistics
        if self.retrieval_service:
            retrieval_stats = await self.retrieval_service.get_all_stats()
            print("\n🔍 Retrieval Statistics | 检索统计:")
            for strategy, stats in retrieval_stats.items():
                print(f"   {strategy}: {stats}")

    def create_visualization(self):
        """Create visualization of experiment results."""
        print("\n🎨 Creating Visualization | 创建可视化")
        print("=" * 50)

        if not self.episodes:
            print("❌ No episodes to visualize")
            return

        # Set up the plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            "LoComo Episode Building & Retrieval Experiment Results\nLoComo 情景构建和检索实验结果",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Episode Level Distribution
        levels = [ep.level.name for ep in self.episodes]
        level_counts = Counter(levels)

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
        ax1.pie(
            level_counts.values(),
            labels=level_counts.keys(),
            autopct="%1.1f%%",
            colors=colors[: len(level_counts)],
            startangle=90,
        )
        ax1.set_title("Episode Level Distribution\n情景级别分布")

        # 2. Content Length Distribution
        content_lengths = [len(ep.content) for ep in self.episodes]
        ax2.hist(content_lengths, bins=10, color="#45B7D1", alpha=0.7, edgecolor="black")
        ax2.set_title("Content Length Distribution\n内容长度分布")
        ax2.set_xlabel("Characters\n字符数")
        ax2.set_ylabel("Frequency\n频率")

        # 3. Episodes per Owner
        owner_counts = Counter([ep.owner_id for ep in self.episodes])
        owners = list(owner_counts.keys())[:20]  # Top 20 owners
        counts = [owner_counts[owner] for owner in owners]

        ax3.bar(range(len(owners)), counts, color=colors * (len(owners) // len(colors) + 1))
        ax3.set_title("Episodes per Owner\n每个用户的情景数")
        ax3.set_xlabel("Owner\n用户")
        ax3.set_ylabel("Episode Count\n情景数量")
        ax3.set_xticks(range(len(owners)))
        ax3.set_xticklabels(owners, rotation=45, ha="right")

        # 4. Timeline of Episodes
        timestamps = [ep.temporal_info.timestamp for ep in self.episodes]
        durations = [ep.temporal_info.duration / 3600 for ep in self.episodes]  # Convert to hours

        ax4.scatter(timestamps, durations, c=range(len(self.episodes)), cmap="viridis", alpha=0.7, s=50)
        ax4.set_title("Episode Timeline\n情景时间线")
        ax4.set_xlabel("Date\n日期")
        ax4.set_ylabel("Duration (hours)\n持续时间（小时）")

        # Rotate x-axis labels for better readability
        for ax in [ax4]:
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        # Save visualization
        viz_path = self.output_dir / "locomo_experiment_results.png"
        plt.savefig(viz_path, dpi=300, bbox_inches="tight")

        print(f"✅ Visualization saved to: {viz_path}")
        print(f"✅ 可视化已保存到: {viz_path}")

    async def cleanup(self):
        """Clean up resources."""
        print("\n🧹 Cleaning up | 清理资源")

        if self.bm25_provider:
            await self.bm25_provider.close()
        if self.raw_data_repo:
            await self.raw_data_repo.close()
        if self.episode_repo:
            await self.episode_repo.close()

        print("✅ Cleanup complete | 清理完成")

    async def run_complete_experiment(self, data_file: str = "locomo10.json", sample_size: int = 10):
        """Run the complete experiment workflow."""
        print("🎯" * 50)
        print("🚀 STARTING LOCOMO EXPERIMENT | 开始 LOCOMO 实验 🚀")
        print(f"🎭 Episode Mode: {self.episode_mode} | 情景模式: {self.episode_mode}")
        print("🎯" * 50)

        try:
            # Step 1: Setup LLM provider
            llm_available = await self.setup_llm_provider()
            if not llm_available:
                print("⚠️ Continuing with fallback mode (no LLM)")
                print("⚠️ 继续使用回退模式（无 LLM）")

            # Step 2: Load data
            self.load_locomo_data(data_file, sample_size)

            # Step 3: Setup storage and retrieval
            await self.setup_storage_and_retrieval()

            # Step 4: Build episodes
            await self.build_episodes()

            # Step 5: Perform retrieval experiments
            await self.perform_retrieval_experiments()

            # Step 6: Analyze results
            await self.analyze_results()

            # Step 7: Create visualization
            self.create_visualization()

            print("\n🎉 EXPERIMENT COMPLETE | 实验完成 🎉")
            print(f"✅ Successfully processed {len(self.conversations)} conversations")
            print(f"✅ Created {len(self.episodes)} episodes")
            print(f"✅ Files saved to: {self.output_dir.absolute()}")
            print(f"✅ Database stored in: {self.db_dir.absolute()}")

        except Exception as e:
            print(f"\n❌ Experiment failed: {e}")
            print(f"❌ 实验失败: {e}")
            raise
        finally:
            await self.cleanup()


async def main():
    """Main function to run the LoComo experiment."""
    print("🎯 LoComo Episode Building and Retrieval Experiment")
    print("🎯 LoComo 情景构建和检索实验")
    print("=" * 60)

    # Load environment variables from .env file in project root
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"🔧 Loaded environment variables from: {env_path}")
    else:
        print(f"⚠️ No .env file found at: {env_path}")
        print("💡 You can create a .env file in the project root with OPENAI_API_KEY=your_key_here")

    # Check if dataset exists
    dataset_path = Path("dataset/locomo10.json")
    if not dataset_path.exists():
        print(f"❌ Dataset file not found: {dataset_path}")
        print("💡 Please ensure the LoComo dataset is available in the dataset/ directory")
        return

    # Initialize and run experiment
    import sys

    # Parse command line arguments for episode mode
    episode_mode = "agent"  # Default mode
    if len(sys.argv) > 1:
        if sys.argv[1] in ["agent", "speaker"]:
            episode_mode = sys.argv[1]
        else:
            print("Usage: python locomo.py [agent|speaker] [max_concurrency]")
            print("Default: agent mode, max_concurrency=3")

    print(f"🎭 Using episode mode: {episode_mode}")

    # Check for concurrency parameter
    max_concurrency = 3  # Default
    if len(sys.argv) > 2:
        try:
            max_concurrency = int(sys.argv[2])
            if max_concurrency < 1:
                raise ValueError("Concurrency must be at least 1")
        except ValueError as e:
            print(f"❌ Invalid concurrency parameter: {sys.argv[2]}")
            print(f"   Error: {e}")
            print("   Using default concurrency: 3")
            max_concurrency = 3

    print(f"🔄 Using max concurrency: {max_concurrency}")

    experiment = LoCoMoExperiment(episode_mode=episode_mode, max_concurrency=max_concurrency)
    await experiment.run_complete_experiment(
        data_file="locomo10-m.json",
        sample_size=1,  # Process all 10 samples
    )

    print("\n👋 Experiment complete! | 实验完成！")
    print("💡 To query the episodes interactively:")
    print("   - Agent mode: python locomo_interactive.py agent")
    print("   - Speaker mode: python locomo_interactive.py speaker")
    print("💡 要交互式查询情景：")
    print("   - 智能体模式: python locomo_interactive.py agent")
    print("   - 说话者模式: python locomo_interactive.py speaker")


if __name__ == "__main__":
    # Set up matplotlib for better Chinese font support
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # Run the experiment
    asyncio.run(main())
