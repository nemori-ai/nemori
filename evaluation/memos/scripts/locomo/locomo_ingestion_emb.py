import argparse
import asyncio
import os
import pandas as pd

from dotenv import load_dotenv
from nemori_eval.experiment import NemoriExperiment, RetrievalStrategy

from nemori_eval import NemoriExperiment

NEMORI_AVAILABLE = True


async def main_nemori(version="default"):
    """Main function for Nemori processing."""
    load_dotenv()
    locomo_df = pd.read_json("data/locomo/locomo10.json")

    print("🚀 Starting Nemori Ingestion")
    print("=" * 50)

    # Create Nemori experiment
    experiment = NemoriExperiment(
        version=version, episode_mode="speaker", retrievalstrategy=RetrievalStrategy.EMBEDDING, max_concurrency=64
    )

    # Step 1: Setup LLM provider
    api_key = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('OPENAI_BASE_URL')
    model = "gpt-4.1-mini"
    llm_available = await experiment.setup_llm_provider(model=model, api_key=api_key, base_url=base_url)
    if not llm_available:
        print("⚠️ Continuing with fallback mode (no LLM)")

    # Step 2: Load data
    experiment.load_locomo_data(locomo_df)
    api_key = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('OPENAI_BASE_URL')
    model = "text-embedding-3-small"
    # Step 3: Setup storage and retrieval
    await experiment.setup_storage_and_retrieval(emb_api_key=api_key, emb_base_url=base_url, embed_model=model)

    # Step 4: Build episodes
    await experiment.build_episodes_semantic()

    print("\n🎉 Nemori Ingestion Complete")
    print(f"✅ Successfully processed {len(experiment.conversations)} conversations")
    print(f"✅ Created {len(experiment.episodes)} episodes")
    semantic_count = getattr(experiment, 'actual_semantic_count', 0)
    print(f"✅ Discovered {semantic_count} semantic concepts")
    if semantic_count > 0 and len(experiment.episodes) > 0:
        print(f"📊 Average semantic concepts per episode: {semantic_count/len(experiment.episodes):.1f}")


def main(frame, version="default"):
    load_dotenv()
    if frame == "nemori":
        # Run async main for nemori
        asyncio.run(main_nemori(version))
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lib",
        type=str,
        choices=["zep", "memos", "mem0", "mem0_graph", "nemori"],
        help="Specify the memory framework (zep or memos or mem0 or mem0_graph or nemori)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="default",
        help="Version identifier for saving results (e.g., 1010)",
    )
    args = parser.parse_args()
    lib = args.lib
    version = args.version

    main(lib, version)
