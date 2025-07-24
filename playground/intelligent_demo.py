#!/usr/bin/env python3
"""
Nemori Intelligent Demo - Real LLM-Powered Episode Creation

This demo showcases Nemori's full intelligence capabilities:
- REQUIRES OpenAI API key for authentic episode generation
- Performs real conversation boundary detection  
- Uses LLM to create meaningful episodic memories
- Supports multiple storage backends (JSONL/DuckDB/PostgreSQL)
- Demonstrates intelligent search with actual semantic understanding

Environment variables:
- OPENAI_API_KEY: Required for LLM processing
- NEMORI_STORAGE: Storage type (jsonl/duckdb/postgresql, default: jsonl)
- POSTGRESQL_TEST_URL: Required if using postgresql storage
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path

# Nemori imports
from nemori.builders.conversation_builder import ConversationEpisodeBuilder
from nemori.core.builders import EpisodeBuilderRegistry
from nemori.core.data_types import DataType, RawEventData, TemporalInfo
from nemori.episode_manager import EpisodeManager
from nemori.llm.providers.openai_provider import OpenAIProvider
from nemori.retrieval import RetrievalConfig, RetrievalQuery, RetrievalService, RetrievalStorageType, RetrievalStrategy
from nemori.storage import create_jsonl_config, create_duckdb_config, create_postgresql_config, create_repositories


class NemoriIntelligentDemo:
    """Full intelligence demonstration with real LLM processing"""

    def __init__(self, storage_type="jsonl"):
        self.storage_type = storage_type.lower()
        self.demo_dir = Path("playground/intelligent_demo_results")
        self.demo_dir.mkdir(exist_ok=True)

        # Core components
        self.llm_provider = None
        self.raw_data_repo = None
        self.episode_repo = None
        self.episode_manager = None
        self.retrieval_service = None

        # Demo data
        self.conversations = []
        self.generated_episodes = []

    def create_realistic_conversations(self):
        """Create rich, realistic conversation data for LLM processing"""
        print("📊 Creating realistic conversation data for LLM analysis...")

        # Long-form, natural conversations with topic transitions
        self.conversations = [
            {
                "id": "startup_strategy_discussion",
                "title": "Startup Strategy and Market Analysis",
                "messages": [
                    {"speaker": "emma", "content": "I've been thinking about our go-to-market strategy. We need to decide between focusing on enterprise clients or starting with SMBs.", "timestamp": "2024-02-01T10:00:00"},
                    {"speaker": "james", "content": "That's a crucial decision. What's your gut feeling? I've seen startups succeed with both approaches.", "timestamp": "2024-02-01T10:01:30"},
                    {"speaker": "emma", "content": "Honestly, I'm leaning toward SMBs first. They're easier to reach, have shorter sales cycles, and can give us faster feedback for product iteration.", "timestamp": "2024-02-01T10:02:45"},
                    {"speaker": "james", "content": "Makes sense. Plus, we can build case studies with SMBs and use those to approach enterprise later. What about pricing strategy?", "timestamp": "2024-02-01T10:04:00"},
                    {"speaker": "emma", "content": "I've been researching competitors. Most charge $99-299 per month for similar features. I think we should start at $79 to be aggressive.", "timestamp": "2024-02-01T10:05:15"},
                    {"speaker": "james", "content": "Aggressive pricing could work, but we need to make sure our unit economics still work out. Have you run the numbers on customer acquisition cost?", "timestamp": "2024-02-01T10:06:30"},
                    {"speaker": "emma", "content": "Not yet, but I estimate we'll need about 3-6 months to break even on each customer if we can keep churn below 5% monthly.", "timestamp": "2024-02-01T10:07:45"},
                    {"speaker": "james", "content": "That sounds reasonable. What about the technical implementation? I'm worried about scaling if we get a sudden influx of users.", "timestamp": "2024-02-01T10:09:00"},
                    {"speaker": "emma", "content": "Good point. Maybe we should implement some basic rate limiting and database optimization before launch. Better safe than sorry.", "timestamp": "2024-02-01T10:10:15"},
                    {"speaker": "james", "content": "Absolutely. I'd rather delay launch by two weeks than have our servers crash when TechCrunch writes about us.", "timestamp": "2024-02-01T10:11:30"},
                ]
            },
            {
                "id": "ai_research_collaboration",
                "title": "AI Research Project Collaboration",
                "messages": [
                    {"speaker": "sarah", "content": "I just finished reading your paper on attention mechanisms. The results on long-sequence modeling are impressive!", "timestamp": "2024-02-05T14:00:00"},
                    {"speaker": "alex", "content": "Thanks! It took months to get those experiments right. The key breakthrough was realizing we needed sparse attention patterns for sequences over 10k tokens.", "timestamp": "2024-02-05T14:01:20"},
                    {"speaker": "sarah", "content": "That's fascinating. Have you considered how this might apply to episodic memory systems? I'm working on something similar for conversation understanding.", "timestamp": "2024-02-05T14:02:40"},
                    {"speaker": "alex", "content": "Actually, yes! I think there's a natural connection. Episodes could be treated as attention patterns over temporal sequences of events.", "timestamp": "2024-02-05T14:04:00"},
                    {"speaker": "sarah", "content": "Exactly what I was thinking! In my current project, I'm trying to identify natural conversation boundaries using transformer models.", "timestamp": "2024-02-05T14:05:20"},
                    {"speaker": "alex", "content": "Interesting approach. Are you using pre-trained models or training from scratch? I imagine domain-specific conversation data would be crucial.", "timestamp": "2024-02-05T14:06:40"},
                    {"speaker": "sarah", "content": "I started with BERT but found that fine-tuning on conversational data gave much better boundary detection. The model learns to recognize topic shifts and speaker intention changes.", "timestamp": "2024-02-05T14:08:00"},
                    {"speaker": "alex", "content": "That makes perfect sense. Have you thought about incorporating temporal dynamics? Conversation flow has natural rhythms that might help with segmentation.", "timestamp": "2024-02-05T14:09:20"},
                    {"speaker": "sarah", "content": "Yes! I'm experimenting with RNNs to capture the temporal dependencies between messages. Early results show 15% improvement in boundary detection accuracy.", "timestamp": "2024-02-05T14:10:40"},
                    {"speaker": "alex", "content": "Wow, that's significant! We should definitely collaborate on this. My attention work might complement your temporal modeling perfectly.", "timestamp": "2024-02-05T14:12:00"},
                    {"speaker": "sarah", "content": "I'd love that! Let's set up a research meeting next week to explore how we can combine our approaches.", "timestamp": "2024-02-05T14:13:20"},
                ]
            },
            {
                "id": "travel_documentary_planning",
                "title": "Travel Documentary Production Planning",
                "messages": [
                    {"speaker": "mike", "content": "I've been reviewing the footage from our Japan trip. The material is incredible, but we need to figure out the narrative structure.", "timestamp": "2024-02-10T16:00:00"},
                    {"speaker": "lisa", "content": "What's your vision for the story arc? Are we focusing on cultural immersion, personal transformation, or the journey itself?", "timestamp": "2024-02-10T16:01:30"},
                    {"speaker": "mike", "content": "I think personal transformation works best. The way local people changed our perspective on minimalism and mindfulness was profound.", "timestamp": "2024-02-10T16:03:00"},
                    {"speaker": "lisa", "content": "Perfect angle! We can structure it chronologically - arrival expectations, cultural shock, gradual understanding, and final transformation.", "timestamp": "2024-02-10T16:04:30"},
                    {"speaker": "mike", "content": "Yes, and we have amazing footage of that tea ceremony in Kyoto where everything clicked for us. That could be the turning point.", "timestamp": "2024-02-10T16:06:00"},
                    {"speaker": "lisa", "content": "That scene is pure gold! The old tea master's words about 'finding beauty in imperfection' - so powerful. We should build toward that moment.", "timestamp": "2024-02-10T16:07:30"},
                    {"speaker": "mike", "content": "Absolutely. And the contrast with our frantic first days in Tokyo will make the Kyoto transformation even more impactful.", "timestamp": "2024-02-10T16:09:00"},
                    {"speaker": "lisa", "content": "What about music? I'm thinking traditional Japanese instruments mixed with contemporary ambient sounds to bridge old and new.", "timestamp": "2024-02-10T16:10:30"},
                    {"speaker": "mike", "content": "Brilliant! I know a composer who specializes in cultural fusion music. She could create something that evolves with our narrative arc.", "timestamp": "2024-02-10T16:12:00"},
                    {"speaker": "lisa", "content": "This is coming together beautifully. When do you think we can have a rough cut ready for feedback?", "timestamp": "2024-02-10T16:13:30"},
                    {"speaker": "mike", "content": "Given the complexity, I'd say 6-8 weeks for a solid first cut. Quality storytelling takes time, but it'll be worth it.", "timestamp": "2024-02-10T16:15:00"},
                ]
            }
        ]

        print(f"✅ Created {len(self.conversations)} realistic conversation scenarios")
        total_messages = sum(len(conv['messages']) for conv in self.conversations)
        print(f"📝 Total messages for LLM processing: {total_messages}")

    async def setup_llm_provider(self) -> bool:
        """Setup OpenAI LLM provider - REQUIRED for this demo"""
        print("\n🤖 Setting up LLM provider (OpenAI)...")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ ERROR: OPENAI_API_KEY environment variable is required!")
            print("   This demo requires real LLM processing to demonstrate intelligence.")
            print("   Please set your OpenAI API key and try again:")
            print("   export OPENAI_API_KEY='your-api-key-here'")
            return False

        try:
            self.llm_provider = OpenAIProvider(
                model="gpt-4o-mini",  # Cost-effective but intelligent model
                temperature=0.1,      # Low temperature for consistent results
                max_tokens=3000       # Enough for detailed episodes
            )

            # Test connection
            if await self.llm_provider.test_connection():
                print("✅ OpenAI connection successful!")
                print(f"   Model: gpt-4o-mini (optimized for cost and quality)")
                return True
            else:
                print("❌ OpenAI connection failed - please check your API key")
                return False

        except Exception as e:
            print(f"❌ LLM setup error: {e}")
            return False

    async def setup_storage(self):
        """Setup storage backend based on configuration""" 
        print(f"\n🗄️ Setting up {self.storage_type.upper()} storage...")

        try:
            if self.storage_type == "jsonl":
                # Clean up old JSONL files
                for file_path in self.demo_dir.glob("*.jsonl"):
                    file_path.unlink()
                    print(f"🧹 Cleaned up: {file_path.name}")
                
                config = create_jsonl_config(str(self.demo_dir))
                
            elif self.storage_type == "duckdb":
                # Clean up old database
                db_path = self.demo_dir / "intelligent_demo.duckdb"
                if db_path.exists():
                    db_path.unlink()
                    print(f"🧹 Cleaned up: {db_path.name}")
                
                config = create_duckdb_config(str(db_path))
                
            elif self.storage_type == "postgresql":
                # Check PostgreSQL availability
                pg_url = os.getenv("POSTGRESQL_TEST_URL")
                if not pg_url:
                    print("❌ ERROR: POSTGRESQL_TEST_URL required for PostgreSQL storage!")
                    print("   Example: export POSTGRESQL_TEST_URL='postgresql+asyncpg://postgres:password@localhost/nemori_demo'")
                    print("   Falling back to JSONL storage...")
                    self.storage_type = "jsonl"
                    config = create_jsonl_config(str(self.demo_dir))
                else:
                    # Parse database name from URL
                    import re
                    db_match = re.search(r'/([^/]+)(?:\?|$)', pg_url)
                    db_name = db_match.group(1) if db_match else "nemori_demo"
                    config = create_postgresql_config(
                        host="localhost", database=db_name, username="postgres", password="postgres"
                    )
            else:
                print(f"⚠️ Unsupported storage type '{self.storage_type}', using JSONL")
                self.storage_type = "jsonl"
                config = create_jsonl_config(str(self.demo_dir))

            # Initialize repositories
            self.raw_data_repo, self.episode_repo = create_repositories(config)
            await self.raw_data_repo.initialize()
            await self.episode_repo.initialize()

            # Clean up existing demo data for fresh runs
            if self.storage_type == "postgresql":
                print("🧹 Cleaning up existing PostgreSQL demo data...")
                try:
                    # Clear existing demo data by deleting all records from this demo
                    await self._cleanup_postgresql_demo_data()
                    print("✅ PostgreSQL demo data cleaned up")
                except Exception as e:
                    print(f"⚠️ PostgreSQL cleanup warning: {e}")

            print(f"✅ {self.storage_type.upper()} storage initialized successfully")

        except Exception as e:
            print(f"⚠️ Storage setup failed: {e}")
            if self.storage_type != "jsonl":
                print("🔄 Falling back to JSONL storage...")
                self.storage_type = "jsonl"
                config = create_jsonl_config(str(self.demo_dir))
                self.raw_data_repo, self.episode_repo = create_repositories(config)
                await self.raw_data_repo.initialize()
                await self.episode_repo.initialize()

    async def _cleanup_postgresql_demo_data(self):
        """Clean up existing PostgreSQL demo data for fresh runs"""
        try:
            # Get all episodes from intelligent demo owners
            demo_owners = ["emma_intelligent", "james_intelligent", "sarah_intelligent", 
                          "alex_intelligent", "mike_intelligent", "lisa_intelligent"]
            
            for owner_id in demo_owners:
                try:
                    # Get episodes for this owner
                    episodes_result = await self.episode_repo.get_episodes_by_owner(owner_id)
                    episodes = episodes_result.episodes if hasattr(episodes_result, 'episodes') else (episodes_result or [])
                    
                    # Delete episodes
                    for episode in episodes:
                        await self.episode_repo.delete_episode(episode.episode_id)
                except Exception:
                    # Owner might not exist, continue
                    pass
            
            # Clean up raw data from intelligent demo
            demo_conversation_ids = ["startup_strategy_discussion", "ai_research_collaboration", "travel_documentary_planning"]
            for conversation_id in demo_conversation_ids:
                try:
                    await self.raw_data_repo.delete_raw_data(conversation_id)
                except Exception:
                    # Data might not exist, continue
                    pass
                    
        except Exception as e:
            # If cleanup fails, it's not critical - just warn
            print(f"⚠️ Demo data cleanup had issues: {e}")

    async def setup_episode_manager(self):
        """Setup the intelligent episode manager"""
        print("\n🏗️ Setting up intelligent episode manager...")

        # Create conversation builder with LLM
        conversation_builder = ConversationEpisodeBuilder(llm_provider=self.llm_provider)

        # Create builder registry
        builder_registry = EpisodeBuilderRegistry()
        builder_registry.register(conversation_builder)

        # Create episode manager
        self.episode_manager = EpisodeManager(
            raw_data_repo=self.raw_data_repo,
            episode_repo=self.episode_repo,
            builder_registry=builder_registry,
            retrieval_service=None  # Will be set up later
        )

        print("✅ Episode manager ready for intelligent processing")

    async def setup_retrieval_service(self):
        """Setup intelligent retrieval service"""
        print("\n🔍 Setting up intelligent retrieval service...")

        # Clean up old indices  
        for index_file in self.demo_dir.glob("bm25_index_*.pkl"):
            index_file.unlink()

        self.retrieval_service = RetrievalService(self.episode_repo)

        # Configure BM25 retrieval
        retrieval_config = RetrievalConfig(
            storage_type=RetrievalStorageType.DISK,
            storage_config={"directory": str(self.demo_dir)}
        )

        self.retrieval_service.register_provider(RetrievalStrategy.BM25, retrieval_config)
        await self.retrieval_service.initialize()

        # Update episode manager with retrieval service
        self.episode_manager.retrieval_service = self.retrieval_service

        print("✅ Intelligent retrieval service ready")

    async def ingest_and_process_conversations(self):
        """Ingest conversations and generate intelligent episodes"""
        print("\n📥 Ingesting conversations and generating intelligent episodes...")
        print("🧠 Using real LLM to analyze conversations and detect boundaries...")

        for i, conversation in enumerate(self.conversations, 1):
            print(f"\n  Processing conversation {i}/{len(self.conversations)}: {conversation['title']}")

            # Convert to RawEventData
            messages = conversation["messages"]
            start_time = datetime.fromisoformat(messages[0]["timestamp"])
            end_time = datetime.fromisoformat(messages[-1]["timestamp"])
            duration = (end_time - start_time).total_seconds()

            raw_data = RawEventData(
                data_id=conversation["id"],
                data_type=DataType.CONVERSATION,
                content=messages,
                source="intelligent_demo",
                temporal_info=TemporalInfo(
                    timestamp=start_time,
                    duration=duration,
                    timezone="UTC",
                    precision="second"
                ),
                metadata={
                    "title": conversation["title"],
                    "speakers": list(set(msg["speaker"] for msg in messages)),
                    "message_count": len(messages),
                    "demo_type": "intelligent"
                },
                processed=False,
                processing_version="1.0"
            )

            # Store raw data (once per conversation)
            await self.raw_data_repo.store_raw_data(raw_data)
            print(f"    ✅ Stored raw conversation ({len(messages)} messages)")

            # Generate intelligent episodes for each participant
            speakers = list(set(msg["speaker"] for msg in messages))
            for speaker in speakers:
                owner_id = f"{speaker}_intelligent"
                
                print(f"    🧠 LLM analyzing conversation for {speaker}...")
                try:
                    # Use episode manager to generate intelligent episode (without re-storing raw_data)
                    episode = await self.episode_manager.process_raw_data_to_episode(raw_data, owner_id)
                    
                    if episode:
                        self.generated_episodes.append(episode)
                        print(f"    ✅ Generated episode: {episode.title[:60]}...")
                    else:
                        print(f"    ⚠️ No episode generated for {speaker}")
                        
                except Exception as e:
                    print(f"    ❌ Error processing {speaker}: {e}")

            # Note: raw data is automatically marked as processed by episode_manager

        print(f"\n✅ Generated {len(self.generated_episodes)} intelligent episodes")

    async def build_search_indices(self):
        """Build search indices for intelligent retrieval"""
        print("\n🔧 Building intelligent search indices...")

        if not self.generated_episodes:
            print("⚠️ No episodes to index")
            return

        # Get all unique owners
        owners = list(set(ep.owner_id for ep in self.generated_episodes))
        print(f"🎯 Building indices for {len(owners)} participants")

        # Trigger index building by performing dummy searches
        for owner_id in owners:
            try:
                dummy_query = RetrievalQuery(
                    text="test",
                    owner_id=owner_id,
                    limit=1,
                    strategy=RetrievalStrategy.BM25
                )
                await self.retrieval_service.search(dummy_query)
                print(f"  ✅ Built search index for {owner_id}")
            except Exception as e:
                print(f"  ❌ Index building failed for {owner_id}: {e}")

        print("✅ Intelligent search indices ready")

    async def demonstrate_intelligent_search(self):
        """Demonstrate intelligent search capabilities"""
        print("\n🔍 Demonstrating intelligent search capabilities...")
        print("🧠 These searches use LLM-generated episode content for semantic understanding")

        # Define meaningful search queries that should find relevant episodes
        search_queries = [
            "startup strategy and market analysis",
            "artificial intelligence research collaboration", 
            "travel documentary storytelling",
            "customer acquisition and pricing",
            "attention mechanisms and deep learning",
            "cultural transformation through travel",
            "business model optimization",
            "machine learning model architecture"
        ]

        # Get available owners
        owners = list(set(ep.owner_id for ep in self.generated_episodes))

        for query in search_queries[:6]:  # Limit to 6 queries for demo
            print(f"\n  🔎 Intelligent search: '{query}'")
            
            total_results = 0
            for owner_id in owners[:3]:  # Show results for first 3 owners
                try:
                    search_query = RetrievalQuery(
                        text=query,
                        owner_id=owner_id,
                        limit=2,
                        strategy=RetrievalStrategy.BM25
                    )

                    result = await self.retrieval_service.search(search_query)

                    if result.episodes:
                        print(f"    👤 {owner_id}: Found {len(result.episodes)} relevant episode(s)")
                        for i, episode in enumerate(result.episodes):
                            print(f"      {i+1}. {episode.title}")
                            print(f"         {episode.summary}")
                            print(f"         Keywords: {', '.join(episode.search_keywords[:4])}")
                        total_results += len(result.episodes)
                    else:
                        print(f"    👤 {owner_id}: No relevant results")

                except Exception as e:
                    print(f"    ❌ Search failed for {owner_id}: {e}")

            print(f"    📊 Total relevant episodes found: {total_results}")

    async def show_intelligence_comparison(self):
        """Show the difference between mock and LLM-generated content"""
        print("\n🧠 Intelligence Comparison: LLM vs Mock Episodes")
        print("=" * 60)

        if not self.generated_episodes:
            print("⚠️ No LLM episodes to compare")
            return

        # Show a sample LLM-generated episode  
        sample_episode = self.generated_episodes[0]
        print(f"🤖 LLM-Generated Episode Example:")
        print(f"   Title: {sample_episode.title}")
        print(f"   Owner: {sample_episode.owner_id}")
        print(f"   Summary: {sample_episode.summary}")
        print(f"   Content: {sample_episode.content[:200]}...")
        print(f"   Keywords: {', '.join(sample_episode.search_keywords[:6])}")
        print(f"   Key Points: {sample_episode.metadata.key_points[:3]}")

        print(f"\n✨ Key Differences from Mock Episodes:")
        print(f"   • Real semantic understanding of conversation context")
        print(f"   • Natural language boundary detection")
        print(f"   • Intelligent keyword extraction")
        print(f"   • Contextual summary generation")
        print(f"   • Meaningful topic identification")

    async def generate_final_report(self):
        """Generate comprehensive demo report"""
        print("\n📊 Generating intelligent demo report...")

        # Collect statistics
        raw_stats = await self.raw_data_repo.get_stats()
        episode_stats = await self.episode_repo.get_stats()

        print(f"\n📈 Demo Results:")
        print(f"   Conversations processed: {raw_stats.total_raw_data}")
        print(f"   Intelligent episodes generated: {episode_stats.total_episodes}")
        print(f"   Participants analyzed: {len(set(ep.owner_id for ep in self.generated_episodes))}")
        print(f"   Storage backend used: {self.storage_type.upper()}")
        print(f"   LLM model: gpt-4o-mini")

        # Show episode distribution by owner
        episode_by_owner = {}
        for episode in self.generated_episodes:
            owner = episode.owner_id
            if owner not in episode_by_owner:
                episode_by_owner[owner] = 0
            episode_by_owner[owner] += 1

        print(f"\n👥 Episodes by participant:")
        for owner, count in episode_by_owner.items():
            print(f"   {owner}: {count} episodes")

    async def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up resources...")

        if self.retrieval_service:
            await self.retrieval_service.close()
        if self.raw_data_repo:
            await self.raw_data_repo.close()
        if self.episode_repo:
            await self.episode_repo.close()

        print("✅ Cleanup completed")

    async def run_intelligent_demo(self):
        """Run the complete intelligent demo workflow"""
        print("🚀 Nemori Intelligent Demo - Real LLM Processing")
        print("=" * 60)
        print("🧠 This demo uses OpenAI's LLM for authentic intelligence")
        print(f"💾 Storage backend: {self.storage_type.upper()}")
        print()

        try:
            # 1. Create realistic conversation data
            self.create_realistic_conversations()

            # 2. Setup LLM provider (required!)
            llm_ready = await self.setup_llm_provider()
            if not llm_ready:
                print("\n❌ Demo aborted: OpenAI API key required")
                return

            # 3. Setup storage backend
            await self.setup_storage()

            # 4. Setup intelligent processing components
            await self.setup_episode_manager()
            await self.setup_retrieval_service()

            # 5. Process conversations with real intelligence
            await self.ingest_and_process_conversations()

            # 6. Build search indices
            await self.build_search_indices()

            # 7. Demonstrate intelligent search
            await self.demonstrate_intelligent_search()

            # 8. Show intelligence comparison
            await self.show_intelligence_comparison()

            # 9. Generate final report
            await self.generate_final_report()

            print(f"\n🎉 Intelligent demo completed successfully!")
            print(f"✅ Demonstrated full LLM-powered episodic memory system")
            
            if self.storage_type == "jsonl":
                print(f"📁 Intelligent episodes saved at: {self.demo_dir}/")
                print(f"💡 Inspect LLM-generated content:")
                print(f"   cat {self.demo_dir}/episodes.jsonl")

        except Exception as e:
            print(f"❌ Demo error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.cleanup()


async def main():
    """Main function"""
    # Check for required OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("🚀 Nemori Intelligent Demo")
        print("=" * 60)
        print("❌ ERROR: This demo requires an OpenAI API key")
        print("\n🔑 To run this demo:")
        print("1. Get an OpenAI API key from https://platform.openai.com/api-keys")
        print("2. Set the environment variable:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("3. Run this demo again")
        print("\n💡 For a demo without API requirements, try:")
        print("   python playground/basic_demo.py")
        return

    # Get storage type from environment
    storage_type = os.getenv("NEMORI_STORAGE", "jsonl").lower()
    
    # Validate storage type
    supported_types = ["jsonl", "duckdb", "postgresql"]
    if storage_type not in supported_types:
        print(f"⚠️ Unsupported storage type '{storage_type}', using JSONL")
        storage_type = "jsonl"

    # Create and run demo
    demo = NemoriIntelligentDemo(storage_type=storage_type)
    await demo.run_intelligent_demo()


if __name__ == "__main__":
    asyncio.run(main())