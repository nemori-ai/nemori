#!/usr/bin/env python3
"""
Nemori Basic Demo - Simulated Episode Creation

This demo shows Nemori's core functionality without requiring external dependencies:
- Uses JSONL storage (human-readable files)
- Creates mock episodes without LLM dependency
- Demonstrates storage, retrieval, and search capabilities
- Perfect for understanding the system without API keys

Generated files: raw_data.jsonl, episodes.jsonl, episode_links.jsonl
"""

import asyncio
from datetime import datetime
from pathlib import Path

# Nemori core imports
from nemori.core.data_types import DataType, RawEventData, TemporalInfo
from nemori.core.episode import Episode, EpisodeLevel, EpisodeMetadata, EpisodeType
from nemori.storage import create_jsonl_config, create_repositories


async def basic_demo():
    """Basic demo: Mock episode creation with JSONL storage"""
    print("🚀 Nemori Basic Demo - Simulated Episode Creation")
    print("=" * 60)
    print("📋 This demo uses mock data and simulated episodes (no OpenAI required)")
    print("💾 All data will be saved as human-readable JSONL files")
    print()

    # Setup demo directory
    demo_dir = Path("playground/basic_demo_results")
    demo_dir.mkdir(exist_ok=True)

    # Clean up old files
    for file_path in demo_dir.glob("*.jsonl"):
        file_path.unlink()
        print(f"🧹 Cleaned up old file: {file_path.name}")

    # Setup JSONL storage (fixed, no options)
    print("🗄️ Setting up JSONL storage...")
    config = create_jsonl_config(str(demo_dir))
    raw_data_repo, episode_repo = create_repositories(config)

    await raw_data_repo.initialize()
    await episode_repo.initialize()
    print(f"✅ JSONL storage initialized: {demo_dir}/")

    print("\n📊 Creating rich conversation data...")

    # Create comprehensive conversation scenarios
    conversations = [
        {
            "id": "work_discussion",
            "title": "AI Project Planning Meeting",
            "messages": [
                {"speaker": "sarah", "content": "Let's review our AI project milestone. We need to finalize the model architecture this week.", "timestamp": "2024-01-15T09:00:00"},
                {"speaker": "mike", "content": "I've been working on the transformer implementation. The attention mechanism is showing promising results.", "timestamp": "2024-01-15T09:02:00"},
                {"speaker": "sarah", "content": "Great! What about the training data pipeline? Are we ready for the next phase?", "timestamp": "2024-01-15T09:03:00"},
                {"speaker": "mike", "content": "Almost there. I need another day to optimize the data loading. The current batch processing is too slow.", "timestamp": "2024-01-15T09:05:00"},
                {"speaker": "sarah", "content": "No problem. Quality over speed. Let's also discuss the evaluation metrics for next week.", "timestamp": "2024-01-15T09:06:00"},
            ]
        },
        {
            "id": "travel_planning",
            "title": "European Vacation Planning",
            "messages": [
                {"speaker": "alice", "content": "I'm so excited about our Europe trip next month! Have you booked the flights to Paris yet?", "timestamp": "2024-01-20T14:00:00"},
                {"speaker": "bob", "content": "Not yet, but I've been monitoring the prices. They seem reasonable now. Should I go ahead?", "timestamp": "2024-01-20T14:02:00"},
                {"speaker": "alice", "content": "Yes! And don't forget travel insurance. I found a good policy that covers medical and trip cancellation.", "timestamp": "2024-01-20T14:03:00"},
                {"speaker": "bob", "content": "Perfect. What about accommodations in Rome? I saw some great options near the Colosseum.", "timestamp": "2024-01-20T14:05:00"},
                {"speaker": "alice", "content": "That sounds amazing! I've always wanted to see the Colosseum at sunrise. Book it!", "timestamp": "2024-01-20T14:06:00"},
            ]
        },
        {
            "id": "tech_architecture",
            "title": "Database Architecture Decision",
            "messages": [
                {"speaker": "charlie", "content": "For our new memory system, should we use PostgreSQL or stick with DuckDB?", "timestamp": "2024-01-25T16:00:00"},
                {"speaker": "diana", "content": "It depends on our use case. PostgreSQL is better for concurrent access and production workloads.", "timestamp": "2024-01-25T16:02:00"},
                {"speaker": "charlie", "content": "True, but DuckDB excels at analytical queries. Our memory system needs both operational and analytical capabilities.", "timestamp": "2024-01-25T16:03:00"},
                {"speaker": "diana", "content": "Why not implement a hybrid approach? PostgreSQL for operations and DuckDB for analytics?", "timestamp": "2024-01-25T16:05:00"},
                {"speaker": "charlie", "content": "Brilliant idea! That gives us the best of both worlds. Let's prototype this architecture.", "timestamp": "2024-01-25T16:06:00"},
            ]
        }
    ]

    print(f"✅ Created {len(conversations)} conversation scenarios")

    # Store raw conversation data
    print("\n📥 Storing raw conversation data...")
    for conv in conversations:
        # Calculate conversation duration
        start_time = datetime.fromisoformat(conv["messages"][0]["timestamp"])
        end_time = datetime.fromisoformat(conv["messages"][-1]["timestamp"])
        duration = (end_time - start_time).total_seconds()

        raw_data = RawEventData(
            data_id=conv["id"],
            data_type=DataType.CONVERSATION,
            content=conv["messages"],
            source="demo_conversations",
            temporal_info=TemporalInfo(
                timestamp=start_time,
                duration=duration,
                timezone="UTC",
                precision="second"
            ),
            metadata={
                "title": conv["title"],
                "speakers": list(set(msg["speaker"] for msg in conv["messages"])),
                "message_count": len(conv["messages"])
            },
            processed=False,
            processing_version="1.0"
        )

        await raw_data_repo.store_raw_data(raw_data)
        print(f"  ✅ Stored: {conv['title']} ({len(conv['messages'])} messages)")

    # Create mock episodes (simulating LLM-generated episodes)
    print("\n🏗️ Creating mock episodes (simulating LLM processing)...")

    episode_data = [
        {
            "conversation_id": "work_discussion",
            "episodes": [
                {
                    "owner": "sarah",
                    "title": "AI Project Milestone Review and Architecture Planning",
                    "content": "Sarah led a discussion about the AI project milestone, focusing on finalizing model architecture. She emphasized the importance of quality over speed and suggested planning evaluation metrics for the following week.",
                    "summary": "Project milestone review with focus on model architecture finalization",
                    "keywords": ["AI project", "milestone", "model architecture", "evaluation metrics", "quality"],
                    "entities": ["Sarah", "Mike", "AI project", "transformer", "attention mechanism"],
                    "key_points": ["finalize model architecture", "quality over speed", "evaluation metrics planning"]
                },
                {
                    "owner": "mike",
                    "title": "Transformer Implementation Progress and Data Pipeline Optimization",
                    "content": "Mike reported progress on transformer implementation with promising attention mechanism results. He identified data loading optimization as the next critical task, noting current batch processing speed issues.",
                    "summary": "Technical progress report on transformer implementation and data pipeline challenges",
                    "keywords": ["transformer", "attention mechanism", "data pipeline", "batch processing", "optimization"],
                    "entities": ["Mike", "Sarah", "transformer", "attention mechanism", "data pipeline"],
                    "key_points": ["attention mechanism progress", "data loading optimization needed", "batch processing improvements"]
                }
            ]
        },
        {
            "conversation_id": "travel_planning",
            "episodes": [
                {
                    "owner": "alice",
                    "title": "European Trip Excitement and Travel Insurance Planning",
                    "content": "Alice expressed excitement about the upcoming Europe trip and took initiative in travel planning. She researched and found comprehensive travel insurance covering medical and trip cancellation, and showed enthusiasm for visiting the Colosseum at sunrise.",
                    "summary": "Enthusiastic travel planning with focus on insurance and Rome attractions",
                    "keywords": ["Europe trip", "travel insurance", "medical coverage", "Colosseum", "sunrise"],
                    "entities": ["Alice", "Bob", "Europe", "Paris", "Rome", "Colosseum"],
                    "key_points": ["excited about Europe trip", "found travel insurance", "wants Colosseum sunrise visit"]
                },
                {
                    "owner": "bob",
                    "title": "Flight Monitoring and Rome Accommodation Research",
                    "content": "Bob took responsibility for monitoring flight prices to Paris and found them reasonable. He also researched Rome accommodations, specifically identifying attractive options near the Colosseum for Alice's interest.",
                    "summary": "Practical travel arrangements focusing on flights and accommodations",
                    "keywords": ["flight prices", "Paris", "Rome accommodation", "Colosseum proximity", "monitoring"],
                    "entities": ["Bob", "Alice", "Paris", "Rome", "Colosseum"],
                    "key_points": ["monitoring flight prices", "found reasonable prices", "researched Rome accommodations"]
                }
            ]
        },
        {
            "conversation_id": "tech_architecture",
            "episodes": [
                {
                    "owner": "charlie",
                    "title": "Database Architecture Analysis for Memory System",
                    "content": "Charlie initiated a technical discussion about database selection for a new memory system, comparing PostgreSQL and DuckDB capabilities. He recognized the need for both operational and analytical capabilities in the system design.",
                    "summary": "Technical analysis of database options for memory system architecture",
                    "keywords": ["database architecture", "PostgreSQL", "DuckDB", "memory system", "analytical queries"],
                    "entities": ["Charlie", "Diana", "PostgreSQL", "DuckDB", "memory system"],
                    "key_points": ["database selection analysis", "operational vs analytical needs", "hybrid approach consideration"]
                },
                {
                    "owner": "diana",
                    "title": "Hybrid Database Architecture Proposal",
                    "content": "Diana provided expert analysis of database trade-offs and proposed an innovative hybrid approach. She suggested using PostgreSQL for operational workloads and DuckDB for analytical processing, creating a best-of-both-worlds solution.",
                    "summary": "Expert database consultation leading to hybrid architecture proposal",
                    "keywords": ["hybrid architecture", "PostgreSQL operations", "DuckDB analytics", "concurrent access", "workload optimization"],
                    "entities": ["Diana", "Charlie", "PostgreSQL", "DuckDB", "hybrid architecture"],
                    "key_points": ["analyzed database trade-offs", "proposed hybrid approach", "operations/analytics separation"]
                }
            ]
        }
    ]

    all_episodes = []
    for conv_data in episode_data:
        conv_id = conv_data["conversation_id"]
        # Find the original conversation
        original_conv = next(c for c in conversations if c["id"] == conv_id)

        for ep_data in conv_data["episodes"]:
            # Create episode with realistic timestamp
            first_msg_time = datetime.fromisoformat(original_conv["messages"][0]["timestamp"])
            last_msg_time = datetime.fromisoformat(original_conv["messages"][-1]["timestamp"])
            duration = (last_msg_time - first_msg_time).total_seconds()

            episode = Episode(
                episode_id=f"ep_{conv_id}_{ep_data['owner']}",
                owner_id=f"{ep_data['owner']}_demo",
                episode_type=EpisodeType.CONVERSATIONAL,
                level=EpisodeLevel.ATOMIC,
                title=ep_data["title"],
                content=ep_data["content"],
                summary=ep_data["summary"],
                temporal_info=TemporalInfo(
                    timestamp=first_msg_time,
                    duration=duration,
                    timezone="UTC",
                    precision="second"
                ),
                metadata=EpisodeMetadata(
                    source_data_ids=[conv_id],
                    source_types={DataType.CONVERSATION},
                    entities=ep_data["entities"],
                    topics=[original_conv["title"]],
                    key_points=ep_data["key_points"]
                ),
                search_keywords=ep_data["keywords"],
                importance_score=0.8
            )

            await episode_repo.store_episode(episode)
            all_episodes.append(episode)
            print(f"  ✅ Created episode: {ep_data['owner']} - {ep_data['title'][:50]}...")

    print(f"✅ Created {len(all_episodes)} mock episodes")

    # Demonstrate search and retrieval
    print("\n🔍 Demonstrating search and retrieval capabilities...")

    # Get all unique owners
    owners = list(set(ep.owner_id for ep in all_episodes))
    print(f"📊 Found {len(owners)} users: {', '.join(owners)}")

    # Show episodes by owner
    print("\n👥 Episodes by owner:")
    for owner in owners:
        owner_episodes = await episode_repo.get_episodes_by_owner(owner)
        episodes = owner_episodes.episodes if hasattr(owner_episodes, 'episodes') else owner_episodes
        print(f"\n  {owner}:")
        for ep in episodes:
            print(f"    • {ep.title}")
            print(f"      Keywords: {', '.join(ep.search_keywords[:5])}")

    # Demonstrate keyword-based retrieval
    print("\n🔎 Keyword-based episode search:")
    test_keywords = ["AI project", "travel", "database", "architecture", "Europe"]

    for keyword in test_keywords:
        print(f"\n  🔍 Searching for: '{keyword}'")
        found_episodes = []

        # Simple keyword search across all episodes
        for episode in all_episodes:
            searchable_text = f"{episode.title} {episode.content} {episode.summary} {' '.join(episode.search_keywords)}"
            if keyword.lower() in searchable_text.lower():
                found_episodes.append(episode)

        if found_episodes:
            print(f"    Found {len(found_episodes)} relevant episode(s):")
            for ep in found_episodes[:2]:  # Show top 2 results
                print(f"      • {ep.title[:60]}...")
                print(f"        Owner: {ep.owner_id} | Score: {ep.importance_score}")
        else:
            print("    No matching episodes found")

    # Show storage statistics
    print("\n📊 Storage statistics:")
    raw_stats = await raw_data_repo.get_stats()
    episode_stats = await episode_repo.get_stats()

    print(f"  Raw conversations: {raw_stats.total_raw_data}")
    print(f"  Generated episodes: {episode_stats.total_episodes}")
    print(f"  Episode types: {dict(episode_stats.episodes_by_type)}")

    # Cleanup
    await raw_data_repo.close()
    await episode_repo.close()

    print("\n🎉 Basic demo completed successfully!")
    print("✅ All functionality demonstrated without external dependencies")
    print(f"📁 Data files saved at: {demo_dir}/")
    print("\n💡 Inspect the generated files:")
    print(f"  cat {demo_dir}/raw_data.jsonl      # Original conversations")
    print(f"  cat {demo_dir}/episodes.jsonl     # Generated episodes")
    print(f"  cat {demo_dir}/episode_links.jsonl # Episode-data relationships")
    print("\n🚀 Ready to try the intelligent demo with real LLM processing!")


if __name__ == "__main__":
    asyncio.run(basic_demo())
