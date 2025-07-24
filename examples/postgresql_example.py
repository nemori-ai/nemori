"""
Example usage of Nemori with PostgreSQL backend.

This example demonstrates how to:
1. Configure PostgreSQL storage
2. Store and retrieve raw event data
3. Create and search episodes
4. Use different PostgreSQL connection options
"""

import asyncio
import os
from datetime import datetime

from nemori.core.data_types import DataType, RawEventData, TemporalInfo
from nemori.core.episode import Episode, EpisodeLevel, EpisodeMetadata, EpisodeType
from nemori.storage import (
    create_postgresql_config,
    create_repositories,
    validate_config,
)
from nemori.storage.storage_types import EpisodeQuery, RawDataQuery


async def main():
    """Main example function."""
    # Configuration option 1: Using environment variable (example)
    # postgresql_url = os.getenv("POSTGRESQL_URL", "postgresql+asyncpg://postgres@localhost/nemori")

    # Configuration option 2: Using helper function
    config = create_postgresql_config(
        host="localhost",
        port=5432,
        database="nemori_demo",
        username="postgres",
        password=os.getenv("POSTGRES_PASSWORD"),  # Optional password from env
        batch_size=500,
        cache_size=5000,
        enable_full_text_search=True,
    )

    # Validate configuration
    try:
        validate_config(config)
        print("✓ Configuration is valid")
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return

    # Create repositories
    raw_repo, episode_repo = create_repositories(config)

    try:
        # Initialize repositories
        print("Initializing repositories...")
        await raw_repo.initialize()
        await episode_repo.initialize()

        # Health check
        raw_healthy = await raw_repo.health_check()
        episode_healthy = await episode_repo.health_check()
        print(f"Raw data repository health: {'✓' if raw_healthy else '✗'}")
        print(f"Episode repository health: {'✓' if episode_healthy else '✗'}")

        if not (raw_healthy and episode_healthy):
            print("Repository health check failed")
            return

        # Example 1: Store raw event data
        print("\n=== Storing Raw Event Data ===")

        raw_data = RawEventData(
            data_id="example_conversation_1",
            data_type=DataType.CONVERSATION,
            content={
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"},
                ],
                "participants": ["user", "assistant"],
            },
            source="chat_application",
            temporal_info=TemporalInfo(timestamp=datetime.now(), duration=30.0, timezone="UTC", precision="second"),
            metadata={"session_id": "session_123", "channel": "web"},
            processed=False,
            processing_version="1.0",
        )

        stored_id = await raw_repo.store_raw_data(raw_data)
        print(f"Stored raw data with ID: {stored_id}")

        # Example 2: Retrieve raw data
        print("\n=== Retrieving Raw Event Data ===")

        retrieved_data = await raw_repo.get_raw_data(stored_id)
        if retrieved_data:
            print(f"Retrieved data: {retrieved_data.data_id}")
            print(f"Content: {retrieved_data.content}")

        # Example 3: Search raw data
        print("\n=== Searching Raw Event Data ===")

        query = RawDataQuery(data_types=[DataType.CONVERSATION], sources=["chat_application"], limit=10)

        search_result = await raw_repo.search_raw_data(query)
        print(f"Found {search_result.total_count} raw data items")
        print(f"Query time: {search_result.query_time_ms:.2f}ms")

        # Example 4: Create and store an episode
        print("\n=== Creating and Storing Episode ===")

        episode = Episode(
            episode_id="example_episode_1",
            owner_id="user_123",
            episode_type=EpisodeType.CONVERSATION,
            level=EpisodeLevel.ATOMIC,
            title="Friendly Greeting Exchange",
            content="The user greeted the assistant and asked how they were doing. The assistant responded positively and offered to help.",
            summary="A polite greeting exchange between user and assistant",
            temporal_info=TemporalInfo(timestamp=datetime.now(), duration=30.0, timezone="UTC", precision="second"),
            metadata=EpisodeMetadata(
                source_data_ids=[stored_id],
                source_types={DataType.CONVERSATION},
                processing_timestamp=datetime.now(),
                processing_version="1.0",
                entities=["user", "assistant"],
                topics=["greeting", "help_offering"],
                emotions=["friendly", "helpful"],
                key_points=[
                    "User asked how assistant is doing",
                    "Assistant responded positively",
                    "Assistant offered to help",
                ],
                confidence_score=0.95,
                completeness_score=0.90,
                relevance_score=0.85,
            ),
            structured_data={
                "greeting_type": "formal",
                "sentiment": "positive",
                "interaction_type": "help_request_initiation",
            },
            search_keywords=["greeting", "hello", "help", "assistant", "conversation"],
            importance_score=0.7,
        )

        episode_id = await episode_repo.store_episode(episode)
        print(f"Stored episode with ID: {episode_id}")

        # Example 5: Search episodes
        print("\n=== Searching Episodes ===")

        # Search by text
        text_query = EpisodeQuery(text_search="greeting", limit=5)

        episode_results = await episode_repo.search_episodes(text_query)
        print(f"Found {episode_results.total_count} episodes with 'greeting'")

        for ep in episode_results.episodes:
            print(f"  - {ep.title} (importance: {ep.importance_score})")

        # Search by owner
        owner_query = EpisodeQuery(owner_ids=["user_123"], limit=10)

        owner_results = await episode_repo.search_episodes(owner_query)
        print(f"Found {owner_results.total_count} episodes for user_123")

        # Example 6: Link episode to raw data
        print("\n=== Linking Episode to Raw Data ===")

        link_success = await episode_repo.link_episode_to_raw_data(episode_id, [stored_id])
        print(f"Link created: {'✓' if link_success else '✗'}")

        # Verify the link
        linked_raw_data_ids = await episode_repo.get_raw_data_for_episode(episode_id)
        print(f"Raw data linked to episode: {linked_raw_data_ids}")

        episodes_for_raw_data = await episode_repo.get_episodes_for_raw_data(stored_id)
        print(f"Episodes created from raw data: {[ep.episode_id for ep in episodes_for_raw_data]}")

        # Example 7: Update episode importance and mark as accessed
        print("\n=== Updating Episode ===")

        # Mark as accessed (increases recall count)
        await episode_repo.mark_episode_accessed(episode_id)

        # Update importance score
        await episode_repo.update_episode_importance(episode_id, 0.9)

        # Retrieve updated episode
        updated_episode = await episode_repo.get_episode(episode_id)
        if updated_episode:
            print(f"Updated importance: {updated_episode.importance_score}")
            print(f"Recall count: {updated_episode.recall_count}")
            print(f"Last accessed: {updated_episode.last_accessed}")

        # Example 8: Get storage statistics
        print("\n=== Storage Statistics ===")

        raw_stats = await raw_repo.get_stats()
        episode_stats = await episode_repo.get_stats()

        print(f"Raw data count: {raw_stats.total_raw_data}")
        print(f"Processed raw data: {raw_stats.processed_raw_data}")
        print(f"Episode count: {episode_stats.total_episodes}")
        print(f"Storage size: {episode_stats.storage_size_mb:.2f} MB")

        # Example 9: Batch operations
        print("\n=== Batch Operations ===")

        # Create multiple raw data items
        batch_raw_data = []
        for i in range(3):
            data = RawEventData(
                data_id=f"batch_data_{i}",
                data_type=DataType.CONVERSATION,
                content={"message": f"Batch message {i}"},
                source="batch_example",
                temporal_info=TemporalInfo(timestamp=datetime.now()),
                metadata={},
                processed=False,
            )
            batch_raw_data.append(data)

        batch_ids = await raw_repo.store_raw_data_batch(batch_raw_data)
        print(f"Stored {len(batch_ids)} raw data items in batch")

        # Retrieve batch
        batch_retrieved = await raw_repo.get_raw_data_batch(batch_ids)
        print(f"Retrieved {len([item for item in batch_retrieved if item])} items from batch")

        print("\n=== Example completed successfully! ===")

    except Exception as e:
        print(f"Error during example execution: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up
        print("\nClosing repositories...")
        await raw_repo.close()
        await episode_repo.close()


if __name__ == "__main__":
    """
    To run this example:

    1. Make sure PostgreSQL is running and accessible
    2. Set environment variables (optional):
       export POSTGRESQL_URL="postgresql+asyncpg://user:password@localhost/nemori"
       export POSTGRES_PASSWORD="your_password"
    3. Run: python examples/postgresql_example.py

    Note: The example will create tables automatically in the specified database.
    Make sure the database exists and the user has appropriate permissions.
    """
    asyncio.run(main())
