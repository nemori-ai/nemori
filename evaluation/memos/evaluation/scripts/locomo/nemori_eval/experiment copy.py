"""
Nemori experiment implementation extracted from wait_for_refactor.

This module contains the core NemoriExperiment class that handles ingestion
and episodic memory building for the LoCoMo evaluation.
"""

import asyncio
import os
import time

from datetime import datetime, timedelta
from pathlib import Path

from litellm import api_base
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

class NemoriExperiment:
    """Nemori experiment adapted from LoCoMoExperiment."""

    def __init__(
        self, version: str = "default", episode_mode: str = "speaker", db_dir: Path = Path(f"results/locomo/nemori-default/storages"), retrievalstrategy: RetrievalStrategy = RetrievalStrategy.BM25, max_concurrency: int = 3
    ):
        self.version = version
        self.episode_mode = episode_mode
        self.max_concurrency = max_concurrency

        # Paths
        self.db_dir = Path(f"results/locomo/nemori-{version}/storages")
        self.db_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.raw_data_repo = None
        self.episode_repo = None
        self.retrieval_service = None
        self.episode_manager = None
        self.llm_provider = None
        self.retrievalstrategy = retrievalstrategy

        # Data
        self.conversations = []
        self.episodes = []

    async def setup_llm_provider(self, model="", api_key="", base_url="") -> bool:
        """Setup OpenAI LLM provider if API key is available."""
        print("\n🤖 Setting up LLM Provider")
        print("=" * 50)

        self.llm_provider = OpenAIProvider(
            model=model,
            temperature=0.1,
            max_tokens=16 * 1024,
            api_key=api_key,
            base_url=base_url,
        )

        if await self.llm_provider.test_connection():
            print("✅ OpenAI connection successful!")
            print(f"🎯 Model: {self.llm_provider.model}")
            return True
        else:
            print("❌ OpenAI connection failed!")
            return False

    async def setup_storage_and_retrieval(self, emb_api_key = "", emb_base_url = "", embed_model = ""):
        """Setup DuckDB storage and BM25 retrieval service."""
        print("\n🗄️ Setting up Storage and Retrieval")
        print("=" * 50)

        # Setup DuckDB storage
        db_path = self.db_dir / "nemori_memory.duckdb"

        # Remove existing database to start fresh
        if db_path.exists():
            db_path.unlink()
            print("🧹 Cleaned existing database")

        # Also clean any existing BM25 indices
        for index_file in self.db_dir.glob("*_index_*.pkl"):
            index_file.unlink()
            print(f"🧹 Cleaned existing index: {index_file.name}")

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
        #self.episode_repo = DuckDBSemanticMemoryRepository
        await self.raw_data_repo.initialize()
        await self.episode_repo.initialize()

        print(f"✅ DuckDB storage initialized: {db_path}")

        # Setup BM25 retrieval service
        self.retrieval_service = RetrievalService(self.episode_repo)

        # Create BM25 retrieval provider configuration with disk storage
        retrieval_config = RetrievalConfig(
            storage_type=RetrievalStorageType.DISK,
            storage_config={"directory": str(self.db_dir)},
            api_key=emb_api_key,
            base_url=emb_base_url,
            embed_model=embed_model,
        )

        # Register the provider with the service
        self.retrieval_service.register_provider(self.retrievalstrategy, retrieval_config)

        # Initialize the retrieval service
        await self.retrieval_service.initialize()

        print("✅ Embed retrieval service configured")

        # Setup episode manager
        builder_registry = EpisodeBuilderRegistry()
        if self.llm_provider:
            conversation_builder = ConversationEpisodeBuilder(llm_provider=self.llm_provider)
        else:
            raise ValueError("LLM provider not found")

        builder_registry.register(conversation_builder)

        self.episode_manager = EpisodeManager(
            raw_data_repo=self.raw_data_repo,
            episode_repo=self.episode_repo,
            builder_registry=builder_registry,
            retrieval_service=self.retrieval_service,
        )

        print("✅ Episode manager initialized")

    def load_locomo_data(self, locomo_df):
        """Load LoComo data from DataFrame."""
        self.conversations = [locomo_df.iloc[i].to_dict() for i in range(len(locomo_df))]
        print(f"📊 Loaded {len(self.conversations)} conversations")

    # [新增] 这是您要求的新函数，用于加载自定义对话数据
    def load_conversation_data(self, conversation_list: list[dict]):
        """
        Loads a list of conversations into the experiment.
        Each item in the list should be a dictionary representing a conversation,
        e.g., {"conversation_id": "conv1", "messages": [...]}.
        """
        # if "conversation" in conversation_list:
        #     self.conversations = conversation_list["conversation"]
        # else:
        self.conversations = conversation_list
        print(f"📊 Loaded {len(self.conversations)} custom conversations")

    # [新增] 辅助函数，将我们的简单对话格式转换为 Nemori 的 RawEventData
    def convert_conversation_to_nemori(
        self, conversation_data: dict
    ) -> RawEventData:
        """Converts our simple conversation format to Nemori's RawEventData."""
        conversation_id = conversation_data["conversation_id"]
        print(f"   🔄 Converting conversation '{conversation_id}'...")
        
        nemori_messages = []
        for msg in conversation_data["messages"]:
            speaker_name = msg["speaker"]
            # 为每个说话人生成唯一的 owner_id
            speaker_id = f"{speaker_name.lower()}_{conversation_id}"
            
            # 兼容 Python < 3.11 的时间戳格式
            iso_timestamp = msg["timestamp"].replace("Z", "+00:00")
            
            nemori_messages.append({
                "speaker_id": speaker_id,
                "user_name": speaker_name,
                "content": msg["text"],
                "timestamp": iso_timestamp,
            })

        if not nemori_messages:
            raise ValueError("Conversation has no messages.")

        first_timestamp = datetime.fromisoformat(nemori_messages[0]["timestamp"])
        last_timestamp = datetime.fromisoformat(nemori_messages[-1]["timestamp"])
        duration = (last_timestamp - first_timestamp).total_seconds()

        return RawEventData(
            data_type=DataType.CONVERSATION,
            content=nemori_messages,
            source="custom_loader",
            temporal_info=TemporalInfo(
                timestamp=first_timestamp,
                duration=duration,
                timezone="UTC",
            ),
            metadata={"conversation_id": conversation_id},
        )
    

    def parse_locomo_timestamp(self, timestamp_str: str) -> datetime:
        """Parse LoComo timestamp format to datetime object."""
        try:
            timestamp_str = timestamp_str.replace("\\s+", " ").strip()
            dt = datetime.strptime(timestamp_str, "%I:%M %p on %d %B, %Y")
            return dt
        except ValueError as e:
            print(f"⚠️ Warning: Could not parse timestamp '{timestamp_str}': {e}")
            return datetime.now()

    def convert_locomo_to_nemori(
        self, conversation_data: dict, conversation_id: str
    ) -> RawEventData:
        """Convert LoComo conversation format to Nemori RawEventData format."""
        print(f"   🔄 Converting LoComo conversation {conversation_id}...")
        messages = []
        conv = conversation_data["conversation"]

        # Get all session keys in order
        session_keys = sorted(
            [
                key
                for key in conv
                if key.startswith("session_") and not key.endswith("_date_time")
            ]
        )

        print(f"   📅 Found {len(session_keys)} sessions")
        print(
            f"   🎭 Speakers: {conv.get('speaker_a', 'Unknown')} & {conv.get('speaker_b', 'Unknown')}"
        )

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
                    # msg_timestamp = session_time + timedelta(
                    #     seconds=i * 30
                    # )  # 30 seconds between messages
                    # iso_timestamp = msg_timestamp.isoformat()
                    iso_timestamp = msg["timestamp"].replace("Z", "+00:00")
                    # Generate unique speaker_id for this conversation
                    speaker_name = msg["speaker"]
                    if speaker_name not in speaker_name_to_id:
                        # Generate unique ID: {name}_{conversation_index}
                        unique_id = f"{speaker_name.lower().replace(' ', '_')}_{conversation_id}"
                        speaker_name_to_id[speaker_name] = unique_id

                    # Process content with image information if present
                    content = msg["text"]
                    if msg.get("img_url"):
                        blip_caption = msg.get("blip_caption", "an image")
                        content = f"[{speaker_name} shared an image: {blip_caption}] {content}"

                    message = {
                        "speaker_id": speaker_name_to_id[speaker_name],
                        "user_name": speaker_name,
                        "content": content,
                        "timestamp": iso_timestamp,
                        "original_timestamp": conv[session_time_key],
                        "session": session_key,
                    }

                    # Add optional fields if present
                    for optional_field in ["img_url", "blip_caption", "query"]:
                        if optional_field in msg:
                            message[optional_field] = msg[optional_field]

                    messages.append(message)

        print(f"   ✅ Converted {len(messages)} messages from {len(session_keys)} sessions")

        # Calculate total duration based on session lengths
        if messages:
            first_time = datetime.fromisoformat(messages[0]["timestamp"])

            # Calculate duration as total session time rather than span across all sessions
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

    def convert_conversation_to_nemori(
        self, conversation_data: dict, conversation_id: str
    ) -> RawEventData:
        """Convert LoComo conversation format to Nemori RawEventData format."""
        print(f"   🔄 Converting LoComo conversation {conversation_id}...")
        messages = []
        conv = conversation_data["conversation"]

        # Get all session keys in order
        session_keys = sorted(
            [
                key
                for key in conv
                if key.startswith("session_") and not key.endswith("_date_time")
            ]
        )

        print(f"   📅 Found {len(session_keys)} sessions")
        print(
            f"   🎭 Speakers: {conv.get('speaker_a', 'Unknown')} & {conv.get('speaker_b', 'Unknown')}"
        )

        # Generate unique speaker IDs for this conversation
        speaker_name_to_id = {}
        for session_key in session_keys:
            session_messages = conv[session_key]
            session_time_key = f"{session_key}_date_time"

            if session_time_key in conv:
                session_time = self.parse_locomo_timestamp(conv[session_time_key])
                # Parse session timestamp
                for i, msg in enumerate(session_messages):
                    if "timestamp" in msg:
                        iso_timestamp = msg["timestamp"].replace("Z", "+00:00")
                    else:
                        # Generate timestamp for this message (session time + message offset)
                        msg_timestamp = session_time + timedelta(
                            seconds=i * 30
                        )  # 30 seconds between messages
                        iso_timestamp = msg_timestamp.isoformat()
                    # Generate unique speaker_id for this conversation
                    speaker_name = msg["speaker"]
                    if speaker_name not in speaker_name_to_id:
                        # Generate unique ID: {name}_{conversation_index}
                        unique_id = f"{speaker_name.lower().replace(' ', '_')}_{conversation_id}"
                        # if unique_id != "deborah_7":
                        #     #print("skip this user:",unique_id)
                        #     continue
                        speaker_name_to_id[speaker_name] = unique_id

                    # Process content with image information if present
                    content = msg["text"]
                    if msg.get("img_url"):
                        blip_caption = msg.get("blip_caption", "an image")
                        content = f"[{speaker_name} shared an image: {blip_caption}] {content}"

                    message = {
                        "speaker_id": speaker_name_to_id[speaker_name],
                        "user_name": speaker_name,
                        "content": content,
                        "timestamp": iso_timestamp,
                        "original_timestamp": conv[session_time_key],
                        "session": session_key,
                    }

                    # Add optional fields if present
                    for optional_field in ["img_url", "blip_caption", "query"]:
                        if optional_field in msg:
                            message[optional_field] = msg[optional_field]

                    messages.append(message)

        print(f"   ✅ Converted {len(messages)} messages from {len(session_keys)} sessions")

        # Calculate total duration based on session lengths
        if messages:
            first_time = datetime.fromisoformat(messages[0]["timestamp"])

            # Calculate duration as total session time rather than span across all sessions
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

    async def _detect_conversation_boundaries(self, messages: list) -> list[tuple[int, int, str]]:
        """Detect conversation boundaries using the conversation builder's boundary detection."""
        print(f"\n     🔍 Starting boundary detection for {len(messages)} messages")

        boundaries = [(0, len(messages) - 1, "Single episode - no boundary detection")]  # Default: single episode

        if not self.llm_provider or len(messages) <= 1:
            print("     ⚠️ No LLM provider or too few messages, using single episode")
            return boundaries

        print("     ⚠️ Note: Boundary detection uses sequential LLM calls")

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
        current_episode_reason = "Episode start"

        for i in range(1, len(message_dicts)):
            # Check if we should end the current episode at this message
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
        boundaries.append((current_start, len(message_dicts) - 1, current_episode_reason))

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

    async def _build_episodes_speaker_mode(self, raw_data: RawEventData) -> list:
        """Build episodes using speaker perspective (each speaker gets their own episodes)."""
        # Get unique speakers from the conversation
        speakers = {
            msg["speaker_id"]
            for msg in raw_data.content
            if isinstance(msg, dict) and "speaker_id" in msg
        }

        print(f"   👥 Speakers: {list(speakers)}")
        print(f"   💬 Messages: {len(raw_data.content)}")
        print(f"   🕐 Duration: {raw_data.temporal_info.duration:.0f} seconds")

        # Detect conversation boundaries once for all speakers
        conversation_data = ConversationData(raw_data)
        episode_boundaries = await self._detect_conversation_boundaries(conversation_data.messages)

        all_episodes = []

        # Process speakers sequentially (only 2 speakers, concurrency not needed)
        for speaker_id in speakers:
            episodes = await self._build_episodes_for_speaker(
                raw_data, speaker_id, episode_boundaries
            )
            all_episodes.extend(episodes)
            print(f"   ✅ {speaker_id}: {len(episodes)} episodes")

        print(f"   📊 Total: {len(all_episodes)} episodes from {len(speakers)} speakers")
        return all_episodes

    async def process_conversation(self, conv_index: int, conv_data: dict) -> tuple[str, list]:
        """Process a single conversation with concurrency control."""
        # Use the conversation_id from the data itself if available, otherwise use index
        conv_id = conv_data.get("conversation_id", str(conv_index))

        start_time = time.time()
        print(f"   🚀 Starting conversation '{conv_id}' processing... [{start_time:.1f}]")
        # ✅ CORRECTED: Call the converter designed for your simple data format.
        raw_data = self.convert_conversation_to_nemori(conv_data)

        # Use speaker perspective
        episodes = await self._build_episodes_speaker_mode(raw_data)

        end_time = time.time()
        duration = end_time - start_time
        print(
            f"   ✅ Conversation '{conv_id}': {len(episodes)} episodes [{end_time:.1f}, took {duration:.1f}s]"
        )
        return conv_id, episodes


    async def build_episodes(self):
        """Build episodes from LoComo conversations using boundary detection."""
        print("\n🏗️ Building Episodes with Boundary Detection")
        print("=" * 50)
        print(f"🎭 Mode: {self.episode_mode} perspective")
        print(
            f"🔄 Processing {len(self.conversations)} conversations with max concurrency: {self.max_concurrency}"
        )

        self.episodes = []

        async def process_conversation(conv_index: int, conv_data: dict) -> tuple[str, list]:
            """Process a single conversation with concurrency control."""
            if "user_id" in conv_data:
                conv_id = conv_data.get("user_id","0")
            else:
                conv_id = str(conv_index) #conv_data.get("user_id","0")#str(conv_index)

            start_time = time.time()
            print(f"   🚀 Starting conversation {conv_id} processing... [{start_time:.1f}]")

            # Convert to Nemori format (can be concurrent)
            #raw_data = self.convert_locomo_to_nemori(conv_data, conv_id)
            raw_data = self.convert_conversation_to_nemori(conv_data, conv_id)
            # Use speaker perspective
            episodes = await self._build_episodes_speaker_mode(raw_data)

            end_time = time.time()
            duration = end_time - start_time
            print(
                f"   ✅ Conversation {conv_id}: {len(episodes)} episodes [{end_time:.1f}, took {duration:.1f}s]"
            )
            return conv_id, episodes

        # Create tasks for all conversations
        tasks = [
            process_conversation(i, conv_data) for i, conv_data in enumerate(self.conversations)
        ]
        print(f"   📋 Created {len(tasks)} concurrent tasks")

        # Wait for all tasks to complete
        print("   ⏳ Starting concurrent execution...")
        results = await asyncio.gather(*tasks)
        print("   🏁 All conversations completed")

        # Collect all episodes
        for _, episodes in results:
            self.episodes.extend(episodes)

        print("\n📊 Episode Building Complete")
        print(f"✅ Successfully created {len(self.episodes)} episodes")
        if self.retrievalstrategy == RetrievalStrategy.BM25:
            # Build BM25 indices for all episodes
            await self.build_bm25_indices()
        elif self.retrievalstrategy == RetrievalStrategy.EMBEDDING:
            # Build BM25 indices for all episodes
            await self.build_embedding_indices()

    async def build_embedding_indices(self):
        """Build embedding indices for all episodes after they are created."""
        print("\n🔧 Building embedding Indices")
        print("=" * 50)

        if not self.episodes:
            print("⚠️ No episodes to index")
            return

        # Get all unique owner_ids from episodes
        owner_ids = {episode.owner_id for episode in self.episodes}
        print(f"🎯 Building indices for {len(owner_ids)} owners: {list(owner_ids)}")

        # Force refresh of embedding indices
        embedding_provider = self.retrieval_service.get_provider(RetrievalStrategy.EMBEDDING)
        if embedding_provider:
            print("🔄 Triggering embedding index rebuild...")

            # Get all episodes from repository
            all_episodes = []
            for owner_id in owner_ids:
                result = await self.episode_repo.get_episodes_by_owner(owner_id)
                # Handle EpisodeSearchResult object
                owner_episodes = result.episodes if hasattr(result, "episodes") else result
                all_episodes.extend(owner_episodes)
                print(f"   📊 Owner {owner_id}: {len(owner_episodes)} episodes")

            if all_episodes:
                print(f"🏗️ Rebuilding indices for {len(all_episodes)} total episodes...")

                # Trigger EMBEDDING index building by performing dummy searches for each owner
                for owner_id in owner_ids:
                    result = await self.episode_repo.get_episodes_by_owner(owner_id)
                    # Handle EpisodeSearchResult object
                    if hasattr(result, "episodes"):
                        owner_episodes = result.episodes
                    else:
                        owner_episodes = result

                    if owner_episodes:
                        # Trigger index building by performing a search
                        dummy_query = RetrievalQuery(
                            text=".",
                            owner_id=owner_id,
                            limit=1,
                            strategy=RetrievalStrategy.EMBEDDING,
                        )
                        await self.retrieval_service.search(dummy_query)
                        print(
                            f"   ✅ Triggered index build for {owner_id}: {len(owner_episodes)} episodes"
                        )
                    else:
                        print(f"   ⚠️ No episodes found for {owner_id}")

            print("✅ EMBEDDING index building completed")
        else:
            print("❌ EMBEDDING provider not found")

    async def build_bm25_indices(self):
        """Build BM25 indices for all episodes after they are created."""
        print("\n🔧 Building BM25 Indices")
        print("=" * 50)

        if not self.episodes:
            print("⚠️ No episodes to index")
            return

        # Get all unique owner_ids from episodes
        owner_ids = {episode.owner_id for episode in self.episodes}
        print(f"🎯 Building indices for {len(owner_ids)} owners: {list(owner_ids)}")

        # Force refresh of BM25 indices
        bm25_provider = self.retrieval_service.get_provider(RetrievalStrategy.BM25)
        if bm25_provider:
            print("🔄 Triggering BM25 index rebuild...")

            # Get all episodes from repository
            all_episodes = []
            for owner_id in owner_ids:
                result = await self.episode_repo.get_episodes_by_owner(owner_id)
                # Handle EpisodeSearchResult object
                owner_episodes = result.episodes if hasattr(result, "episodes") else result
                all_episodes.extend(owner_episodes)
                print(f"   📊 Owner {owner_id}: {len(owner_episodes)} episodes")

            if all_episodes:
                print(f"🏗️ Rebuilding indices for {len(all_episodes)} total episodes...")

                # Trigger BM25 index building by performing dummy searches for each owner
                for owner_id in owner_ids:
                    result = await self.episode_repo.get_episodes_by_owner(owner_id)
                    # Handle EpisodeSearchResult object
                    if hasattr(result, "episodes"):
                        owner_episodes = result.episodes
                    else:
                        owner_episodes = result

                    if owner_episodes:
                        # Trigger index building by performing a search
                        dummy_query = RetrievalQuery(
                            text=".",
                            owner_id=owner_id,
                            limit=1,
                            strategy=RetrievalStrategy.BM25,
                        )
                        await self.retrieval_service.search(dummy_query)
                        print(
                            f"   ✅ Triggered index build for {owner_id}: {len(owner_episodes)} episodes"
                        )
                    else:
                        print(f"   ⚠️ No episodes found for {owner_id}")

            print("✅ BM25 index building completed")
        else:
            print("❌ BM25 provider not found")


    async def cleanup(self):
        """Clean up resources."""
        print("\n🧹 Cleaning up")

        if self.retrieval_service:
            await self.retrieval_service.close()
        if self.raw_data_repo:
            await self.raw_data_repo.close()
        if self.episode_repo:
            await self.episode_repo.close()

        print("✅ Cleanup complete")