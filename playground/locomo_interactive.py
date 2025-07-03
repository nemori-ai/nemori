#!/usr/bin/env python3
"""
Interactive Query Tool for LoComo Episodic Memory
交互式查询工具 - LoComo 情景记忆系统

This script allows you to query previously saved episodic memories
without running the full experiment.

该脚本允许你查询之前保存的情景记忆，无需运行完整实验。
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from nemori.retrieval import (
    RetrievalConfig,
    RetrievalQuery,
    RetrievalService,
    RetrievalStorageType,
    RetrievalStrategy,
)
from nemori.storage.duckdb_storage import DuckDBEpisodicMemoryRepository
from nemori.storage.storage_types import StorageConfig

# Add the parent directory to sys.path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class InteractiveQueryTool:
    """Interactive tool for querying saved episodic memories."""

    def __init__(self, episode_mode: str = "agent"):
        """Initialize the query tool."""
        # Check if we're in the project root or playground directory
        current_dir = Path.cwd()
        if current_dir.name == "playground":
            self.db_dir = Path(".tmp")
        elif (current_dir / "playground" / ".tmp").exists():
            self.db_dir = Path("playground") / ".tmp"
        else:
            # Default fallback
            self.db_dir = Path(".tmp")
        self.episode_repo = None
        self.retrieval_service = None
        self.bm25_provider = None
        self.episode_mode = episode_mode
        self.available_owners = set()

    async def initialize(self):
        """Initialize storage and retrieval components."""
        print("🎮 Interactive Query Tool | 交互式查询工具")
        print("=" * 60)
        print(f"🎭 Episode mode: {self.episode_mode} | 情景模式: {self.episode_mode}")
        print("🔧 Initializing components | 初始化组件...")

        # Check if database exists
        db_path = self.db_dir / "nemori_memory.duckdb"
        if not db_path.exists():
            print(f"❌ No database found at: {db_path}")
            print("💡 Please run the main experiment first to create episodes")
            print("💡 请先运行主实验以创建情景")
            return False

        # Check if any BM25 index exists for the current mode
        # In speaker mode, there might be multiple index files for different speakers
        # In agent mode, there should be bm25_index_agent.pkl
        index_files = list(self.db_dir.glob("bm25_index_*.pkl"))

        if not index_files:
            print(f"❌ No BM25 index found in: {self.db_dir}")
            print("💡 Please run the main experiment first to create the index")
            print("💡 请先运行主实验以创建索引")
            print("💡 Expected pattern: bm25_index_*.pkl")
            return False

        print(f"✅ Found {len(index_files)} BM25 index file(s)")
        for idx_file in index_files:
            print(f"   📁 {idx_file.name}")

        # Initialize storage
        storage_config = StorageConfig(
            backend_type="duckdb",
            connection_string=str(db_path),
            batch_size=100,
            cache_size=1000,
            enable_semantic_search=False,
        )

        self.episode_repo = DuckDBEpisodicMemoryRepository(storage_config)
        await self.episode_repo.initialize()

        # Initialize retrieval service
        self.retrieval_service = RetrievalService(self.episode_repo)

        # Configure BM25 with disk storage and persistence
        retrieval_config = RetrievalConfig(
            storage_type=RetrievalStorageType.DISK, storage_config={"directory": str(self.db_dir)}
        )

        # Register provider
        self.retrieval_service.register_provider(RetrievalStrategy.BM25, retrieval_config)
        await self.retrieval_service.initialize()

        # Get provider instance
        self.bm25_provider = self.retrieval_service.get_provider(RetrievalStrategy.BM25)

        print("✅ Components initialized successfully | 组件初始化成功")
        print(f"📁 Database: {db_path}")

        # Get stats and available owners
        if self.bm25_provider:
            stats = await self.bm25_provider.get_stats()
            print(f"📊 Episodes indexed: {stats.total_episodes}")
            print(f"📄 Documents: {stats.total_documents}")
            print(f"💾 Index size: {stats.index_size_mb:.1f} MB")

            self.available_owners = set(self.bm25_provider.user_indices.keys())

        # Get available owners from episode repository
        if not self.available_owners:
            await self._discover_available_owners()

        return True

    async def _discover_available_owners(self):
        """Discover available owners from the episode repository."""
        try:
            # This is a simplified approach - we'll try to get episodes for common owners
            test_owners = ["agent"]

            # If speaker mode, also check for speaker IDs
            if self.episode_mode == "speaker":
                # Try common speaker patterns from LoComo dataset
                test_owners.extend(["speaker_a", "speaker_b", "nate", "joanna"])

            for owner in test_owners:
                try:
                    result = await self.episode_repo.get_episodes_by_owner(owner, limit=1)
                    if result.episodes:
                        self.available_owners.add(owner)
                except Exception:
                    pass  # Owner not found

            # If no owners found, default to agent
            if not self.available_owners:
                self.available_owners.add("agent")

            print(f"📋 Discovered owners: {list(self.available_owners)}")

        except Exception as e:
            print(f"⚠️ Error discovering owners: {e}")
            self.available_owners.add("agent")  # Fallback

    async def interactive_query_mode(self):
        """Interactive mode for querying episodes."""
        print("\n🎮 Interactive Query Mode | 交互式查询模式")
        print("=" * 60)
        print("💡 You can now query your episodic memories!")
        print("💡 现在可以查询你的情景记忆了！")
        print(f"👥 Available owners: {list(self.available_owners)}")
        print(f"👥 可用用户: {list(self.available_owners)}")
        print("🔍 Type your query, or 'quit'/'exit' to stop")
        print("🔍 输入查询内容，或输入 'quit'/'exit' 退出")
        print("💡 Commands: 'owners' to list owners, 'mode:<owner>' to switch perspective")
        print("💡 Examples: 'basketball', 'music and concerts', 'health', 'programming'")
        print("-" * 60)

        current_owner = "agent" if "agent" in self.available_owners else next(iter(self.available_owners))

        while True:
            try:
                # Get user input
                query_text = input("\n🔎 Query: ").strip()

                if not query_text:
                    continue

                if query_text.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye! | 再见！")
                    break

                # Handle special commands
                if query_text.lower() == "owners":
                    print(f"👥 Available owners: {list(self.available_owners)}")
                    print(f"👤 Current owner: {current_owner}")
                    continue

                if query_text.lower().startswith("mode:"):
                    new_owner = query_text[5:].strip()
                    if new_owner in self.available_owners:
                        current_owner = new_owner
                        print(f"👤 Switched to owner: {current_owner}")
                    else:
                        print(f"❌ Owner '{new_owner}' not found. Available: {list(self.available_owners)}")
                    continue

                # Perform search
                print(f"\n🔍 Searching for: '{query_text}' (owner: {current_owner})...")
                print(f"🔍 搜索: '{query_text}' (用户: {current_owner})...")

                query = RetrievalQuery(
                    text=query_text, owner_id=current_owner, limit=10, strategy=RetrievalStrategy.BM25
                )

                if self.bm25_provider:
                    result = await self.bm25_provider.search(query)

                    # Format and display results
                    self._display_search_results(query_text, result)
                else:
                    print("❌ BM25 provider not available")

            except KeyboardInterrupt:
                print("\n👋 Goodbye! | 再见！")
                break
            except Exception as e:
                print(f"❌ Error during search: {e}")

    def _display_search_results(self, query_text: str, result):
        """Display search results in a pretty formatted way."""
        print("\n" + "=" * 80)
        print(f"📊 Search Results for: '{query_text}'")
        print(f"📊 搜索结果: '{query_text}'")
        print("=" * 80)

        if not result.episodes:
            print("🔍 No episodes found | 未找到相关情景")
            print(f"📈 Query time: {result.query_time_ms:.1f}ms")
            print(f"📊 Total candidates: {result.total_candidates}")
            return

        # Filter by relevance threshold
        relevance_threshold = 1.0
        relevant_results = []

        min_length = min(len(result.episodes), len(result.scores))
        for i in range(min_length):
            episode = result.episodes[i]
            score = result.scores[i]
            if score > relevance_threshold:
                relevant_results.append((episode, score))

        if not relevant_results:
            print(f"🔍 No highly relevant episodes found (threshold > {relevance_threshold})")
            print(f"🔍 未找到高相关性情景 (阈值 > {relevance_threshold})")
            print(f"📈 Query time: {result.query_time_ms:.1f}ms")
            print(f"📊 Total candidates: {result.total_candidates}")

            # Show top 3 results anyway with lower scores
            print(f"\n📋 Top {min(3, len(result.episodes))} results with any score:")
            print(f"📋 得分最高的 {min(3, len(result.episodes))} 个结果:")
            for i in range(min(3, len(result.episodes))):
                episode = result.episodes[i]
                score = result.scores[i]
                self._format_episode_result(i + 1, episode, score, show_content=False)
            return

        # Display metadata
        print(f"✅ Found {len(relevant_results)} relevant episodes | 找到 {len(relevant_results)} 个相关情景")
        print(f"📈 Query time: {result.query_time_ms:.1f}ms")
        print(f"📊 Total candidates: {result.total_candidates}")

        # Display results
        print(f"\n📋 Top {min(5, len(relevant_results))} Results:")
        print(f"📋 前 {min(5, len(relevant_results))} 个结果:")
        print("-" * 80)

        for i, (episode, score) in enumerate(relevant_results[:5]):
            self._format_episode_result(i + 1, episode, score, show_content=True)

        # Ask if user wants to see more details
        if len(relevant_results) > 5:
            show_more = input(f"\n🔍 Show {len(relevant_results) - 5} more results? (y/n): ").strip().lower()
            if show_more in ["y", "yes", "ya", "da"]:
                print("\n📋 Additional Results:")
                print("📋 更多结果:")
                print("-" * 80)
                for i, (episode, score) in enumerate(relevant_results[5:]):
                    self._format_episode_result(i + 6, episode, score, show_content=True)

    def _format_episode_result(self, rank: int, episode, score: float, show_content: bool = True):
        """Format a single episode result for display."""
        print(f"\n🏆 #{rank} | BM25 Score: {score:.3f}")
        print(f"📝 Title: {episode.title}")
        print(f"👤 Owner: {episode.owner_id}")
        print(f"🏷️  Level: {episode.level.name}")
        print(f"📅 Date: {episode.temporal_info.timestamp.strftime('%Y-%m-%d %H:%M')}")
        print(f"⏱️  Duration: {episode.temporal_info.duration/60:.1f} minutes")
        print(f"⭐ Importance: {episode.importance_score:.2f}")
        print(f"🔢 Recall Count: {episode.recall_count}")

        if episode.search_keywords:
            keywords = ", ".join(episode.search_keywords[:5])  # Show first 5 keywords
            print(f"🔍 Keywords: {keywords}")

        if show_content:
            # Show summary or content preview
            if episode.summary:
                summary = episode.summary[:200] + "..." if len(episode.summary) > 200 else episode.summary
                print(f"📄 Summary: {summary}")
            elif episode.content:
                content = episode.content[:200] + "..." if len(episode.content) > 200 else episode.content
                print(f"📄 Content: {content}")

        print("─" * 60)

    async def cleanup(self):
        """Cleanup resources."""
        if self.retrieval_service:
            await self.retrieval_service.close()
        if self.episode_repo:
            await self.episode_repo.close()


async def main():
    """Main function for interactive query tool."""
    # Parse command line arguments for episode mode
    import sys

    episode_mode = "agent"  # Default mode
    if len(sys.argv) > 1:
        if sys.argv[1] in ["agent", "speaker"]:
            episode_mode = sys.argv[1]
        else:
            print("Usage: python locomo_interactive.py [agent|speaker]")
            print("Default: agent mode")

    print(f"🎭 Using episode mode: {episode_mode}")

    # Load environment variables
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"🔧 Loaded environment variables from: {env_path}")

    # Initialize and run query tool
    query_tool = InteractiveQueryTool(episode_mode=episode_mode)

    try:
        success = await query_tool.initialize()
        if success:
            await query_tool.interactive_query_mode()
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await query_tool.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
