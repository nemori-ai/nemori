"""Nemori quickstart example."""
import asyncio
from nemori import NemoriMemory, MemoryConfig


async def main():
    # DSN, API keys, and base URLs are resolved from environment variables.
    # Only model names need to be specified explicitly.
    config = MemoryConfig(
        llm_model="openai/gpt-4.1-mini",
        embedding_model="google/gemini-embedding-001",
    )

    async with NemoriMemory(config=config) as memory:
        health = await memory.health()
        print(f"System healthy: {health.healthy}")

        await memory.add_messages("alice", [
            {"role": "user", "content": "I just moved to Tokyo last month"},
            {"role": "assistant", "content": "How exciting! How are you finding life in Tokyo?"},
            {"role": "user", "content": "Love it! The food is amazing, especially ramen"},
        ])

        episodes = await memory.flush("alice")
        print(f"Created {len(episodes)} episodes")

        results = await memory.search("alice", "Where does Alice live?")
        print(results)


if __name__ == "__main__":
    asyncio.run(main())
