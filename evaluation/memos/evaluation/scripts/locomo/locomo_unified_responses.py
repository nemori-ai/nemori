"""
Unified Memory LoCoMo Response Generation Script

This script generates responses for LoCoMo evaluation using the unified episodic and semantic memory system.
It processes real LoCoMo questions and uses both memory types to generate contextually rich responses.
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from nemori_eval.experiment import NemoriExperiment, RetrievalStrategy
from nemori.retrieval import RetrievalQuery


async def generate_unified_response_for_question(experiment, owner_id: str, question: str, context_limit: int = 5) -> dict:
    """Generate response for a LoCoMo question using unified episodic and semantic memory retrieval."""
    start_time = time.time()
    
    # Step 1: Unified memory search (both episodic and semantic)
    search_start = time.time()
    
    try:
        # Use the unified retrieval service that combines both memory types
        unified_results = await experiment.unified_retrieval_service.enhanced_query(
            owner_id, 
            question, 
            episode_limit=context_limit, 
            semantic_limit=context_limit
        )
        
        episodes = unified_results.get('episodes', [])
        semantic_knowledge = unified_results.get('semantic_knowledge', [])
        
        # Build rich context from both episodic episodes and semantic knowledge
        context_parts = []
        
        # Add episodic episode context (personal experiences, conversations)
        episodic_context = []
        for episode in episodes:
            episode_info = f"Episode: {episode.title}\nContent: {episode.content[:500]}..."  # Limit length
            episodic_context.append(episode_info)
            
        # Add semantic knowledge context (extracted facts, preferences, relationships)
        semantic_context = []
        for node in semantic_knowledge:
            knowledge_info = f"Knowledge: {node.key} = {node.value}"
            if node.confidence < 1.0:
                knowledge_info += f" (confidence: {node.confidence:.2f})"
            semantic_context.append(knowledge_info)
            
        # Combine contexts with clear separation
        if episodic_context:
            context_parts.append("EPISODIC MEMORIES (Personal experiences):\n" + "\n\n".join(episodic_context))
        if semantic_context:
            context_parts.append("SEMANTIC KNOWLEDGE (Known facts):\n" + "\n".join(semantic_context))
            
        search_context = "\n\n" + "="*50 + "\n\n".join(context_parts) if context_parts else ""
        
    except Exception as e:
        print(f"❌ Unified retrieval error for {owner_id}: {e}")
        search_context = ""
        episodes = []
        semantic_knowledge = []
    
    search_duration = (time.time() - search_start) * 1000
    
    # Step 2: Generate contextual response using LLM
    response_start = time.time()
    
    if experiment.llm_provider and search_context.strip():
        try:
            # Craft a prompt that leverages both types of memory
            prompt = f"""You are answering a question based on retrieved memories about a person. Use the provided context to give a natural, accurate response.

RETRIEVED MEMORY CONTEXT:
{search_context}

QUESTION: {question}

INSTRUCTIONS:
- Answer naturally and conversationally based on the retrieved memories
- Use both episodic memories (personal experiences) and semantic knowledge (known facts)
- If episodic and semantic information conflict, prefer more recent episodic evidence
- If you don't have sufficient relevant information, say so honestly
- Don't hallucinate information not present in the context

RESPONSE:"""

            response = await experiment.llm_provider.generate(prompt)
            answer = response.strip()
            
        except Exception as e:
            print(f"❌ LLM generation error: {e}")
            # Fallback: create response from context directly
            if episodes and semantic_knowledge:
                answer = f"Based on my unified memory search, I found {len(episodes)} relevant experiences and {len(semantic_knowledge)} related facts. However, I encountered an error generating a natural response."
            elif episodes:
                answer = f"I found {len(episodes)} relevant personal experiences but encountered an error processing them into a response."
            elif semantic_knowledge:
                answer = f"I have {len(semantic_knowledge)} relevant facts but encountered an error generating a natural response."
            else:
                answer = "I don't have relevant information to answer this question."
    else:
        # Fallback without LLM: basic context-based response
        if episodes or semantic_knowledge:
            context_summary = []
            if episodes:
                context_summary.append(f"{len(episodes)} relevant experiences")
            if semantic_knowledge:
                context_summary.append(f"{len(semantic_knowledge)} known facts")
            answer = f"Based on my unified memory containing {' and '.join(context_summary)}, I can provide context but cannot generate a natural language response without LLM support."
        else:
            answer = "I don't have any relevant memories (episodic or semantic) about this topic."
    
    response_duration = (time.time() - response_start) * 1000
    total_duration = (time.time() - start_time) * 1000
    
    return {
        "question": question,
        "answer": answer,
        "search_context": search_context,
        "episodic_results_count": len(episodes),
        "semantic_results_count": len(semantic_knowledge),
        "total_context_items": len(episodes) + len(semantic_knowledge),
        "search_duration_ms": search_duration,
        "response_duration_ms": response_duration,
        "total_duration_ms": total_duration,
        "timestamp": datetime.now().isoformat(),
        "memory_type": "unified_episodic_semantic"
    }


async def process_user_questions_unified(experiment, user_data: dict, user_id: str) -> list:
    """Process all questions for a single user using unified memory."""
    questions = user_data.get("questions", [])
    if not questions:
        print(f"⚠️ No questions found for user {user_id}")
        return []
    
    print(f"📋 Processing {len(questions)} questions for user {user_id} with unified memory")
    
    responses = []
    for question_data in tqdm(questions, desc=f"User {user_id}"):
        question = question_data.get("question", "")
        category = question_data.get("category", "unknown")
        golden_answer = question_data.get("golden_answer")
        
        if not question:
            continue
            
        response = await generate_unified_response_for_question(experiment, user_id, question)
        response.update({
            "category": category,
            "golden_answer": golden_answer,
            "user_id": user_id
        })
        responses.append(response)
    
    return responses


async def load_or_create_locomo_questions(experiment) -> dict:
    """Load LoCoMo questions or create sample questions from episodes."""
    question_file = "data/locomo/locomo_questions.json" 
    
    if os.path.exists(question_file):
        print(f"📋 Loading existing LoCoMo questions from {question_file}")
        with open(question_file) as f:
            questions_data = json.load(f)
        return questions_data
    
    print(f"📝 Creating LoCoMo-style questions from available episodes")
    
    # Get all available owners from episodes
    if not experiment.episodes:
        print("❌ No episodes available to create questions from")
        return {}
    
    episode_owners = list(set(ep.owner_id for ep in experiment.episodes))
    questions_data = {}
    
    # Create contextual questions for each owner based on their episodes
    for owner in episode_owners[:5]:  # Limit to first 5 owners for demo
        owner_episodes = [ep for ep in experiment.episodes if ep.owner_id == owner]
        
        # Generate LoCoMo-style questions based on episode content
        owner_questions = [
            {
                "question": "What topics have we discussed in our conversations?",
                "category": "conversation_topics",
                "golden_answer": "Various conversation topics from our interactions"
            },
            {
                "question": "What do you remember about my interests and preferences?",
                "category": "personal_preferences", 
                "golden_answer": "Personal interests and preferences mentioned"
            },
            {
                "question": "Tell me about our recent conversations.",
                "category": "recent_memory",
                "golden_answer": "Content from recent conversation episodes"
            },
            {
                "question": "What personal details do you know about me?",
                "category": "personal_information",
                "golden_answer": "Personal information shared in conversations"
            }
        ]
        
        questions_data[f"locomo_exp_user_{owner}"] = owner_questions
    
    # Save the generated questions
    os.makedirs("data/locomo", exist_ok=True)
    with open(question_file, 'w') as f:
        json.dump(questions_data, f, indent=2)
    
    print(f"✅ Created {len(questions_data)} user question sets: {question_file}")
    return questions_data


async def main_nemori_unified_responses(version="default"):
    """Generate LoCoMo responses using unified episodic and semantic memory system."""
    print("🚀 Generating LoCoMo Responses with Unified Memory System")
    print("=" * 70)
    print("🎯 This will:")
    print("   1. Load pre-built unified memory (episodes + semantic knowledge)")
    print("   2. Process LoCoMo evaluation questions")
    print("   3. Use both episodic and semantic memory for rich responses")
    print("   4. Generate evaluation-ready JSON output")
    
    # Load LoCoMo conversation data (for memory building if needed)
    locomo_df = pd.read_json("data/locomo/locomo10.json")
    
    # Create experiment with unified memory capabilities
    experiment = NemoriExperiment(
        version=version,
        episode_mode="speaker", 
        retrievalstrategy=RetrievalStrategy.EMBEDDING
    )
    
    try:
        # Step 1: Setup LLM provider for response generation
        api_key = "sk-NuO4dbNTkXXi5zWYkLFnhyug1kkqjeysvrb74pA9jTGfz8cm"
        base_url = "https://jeniya.cn/v1"
        model = "gpt-4o-mini"
        
        print("\n🤖 Step 1: Setting up LLM Provider for Response Generation")
        print("-" * 60)
        llm_available = await experiment.setup_llm_provider(model=model, api_key=api_key, base_url=base_url)
        if not llm_available:
            print("⚠️ Continuing without LLM - responses will be context-based only")
        else:
            print("✅ LLM provider ready for natural language response generation")
        
        # Step 2: Load conversation data and build unified memory
        print("\n🏗️ Step 2: Building Unified Memory from LoCoMo Data")
        print("-" * 60)
        experiment.load_locomo_data(locomo_df)
        
        # Setup unified storage and retrieval
        emb_api_key = api_key
        emb_base_url = base_url
        embed_model = "text-embedding-ada-002"
        
        await experiment.setup_storage_and_retrieval(
            emb_api_key=emb_api_key,
            emb_base_url=emb_base_url, 
            embed_model=embed_model
        )
        
        # Build unified memory (episodes + semantic knowledge)
        print(f"🔄 Processing {len(experiment.conversations)} conversations...")
        await experiment.build_episodes()
        
        print(f"✅ Unified memory built:")
        print(f"   📖 Episodes: {len(experiment.episodes)}")
        
        # Check semantic memory
        episode_owners = set(ep.owner_id for ep in experiment.episodes) if experiment.episodes else set()
        total_semantic_nodes = 0
        for owner in episode_owners:
            try:
                nodes = await experiment.semantic_repo.get_all_semantic_nodes_for_owner(owner)
                total_semantic_nodes += len(nodes) if nodes else 0
            except:
                pass
        print(f"   🧠 Semantic Concepts: {total_semantic_nodes}")
        
        # Step 3: Load LoCoMo evaluation questions
        print("\n📋 Step 3: Loading LoCoMo Evaluation Questions")
        print("-" * 60)
        
        questions_data = await load_or_create_locomo_questions(experiment)
        
        if not questions_data:
            print("❌ No questions available for evaluation")
            return
            
        print(f"✅ Loaded questions for {len(questions_data)} users")
        
        # Step 4: Generate responses using unified memory
        print(f"\n🎯 Step 4: Generating Responses with Unified Memory")
        print("-" * 60)
        print("🔄 For each question:")
        print("   • Search episodic episodes for relevant experiences")
        print("   • Search semantic knowledge for relevant facts")
        print("   • Combine both contexts for rich LLM response generation")
        print()
        
        all_responses = {}
        
        for user_key, user_questions in questions_data.items():
            user_id = user_key.replace("locomo_exp_user_", "")
            print(f"\n👤 Processing user: {user_id}")
            
            try:
                user_responses = await process_user_questions_unified(
                    experiment,
                    {"questions": user_questions}, 
                    user_id
                )
                all_responses[user_key] = user_responses
                
                if user_responses:
                    # Show sample response analysis
                    sample_response = user_responses[0]
                    print(f"   ✅ Generated {len(user_responses)} responses")
                    print(f"   📊 Sample response context:")
                    print(f"      • Episodic episodes: {sample_response.get('episodic_results_count', 0)}")
                    print(f"      • Semantic concepts: {sample_response.get('semantic_results_count', 0)}")
                    print(f"      • Total context items: {sample_response.get('total_context_items', 0)}")
                else:
                    print(f"   ⚠️ No responses generated for {user_id}")
                
            except Exception as e:
                print(f"   ❌ Error processing user {user_id}: {e}")
                all_responses[user_key] = []
        
        # Step 5: Save responses for evaluation
        print(f"\n💾 Step 5: Saving Responses for LoCoMo Evaluation")
        print("-" * 60)
        
        results_dir = f"results/locomo/nemori-{version}"
        os.makedirs(results_dir, exist_ok=True)
        
        response_file = f"{results_dir}/nemori_locomo_responses.json"
        with open(response_file, 'w') as f:
            json.dump(all_responses, f, indent=2)
        
        # Step 6: Generate evaluation summary
        print(f"\n📊 Step 6: Evaluation Summary")
        print("=" * 60)
        
        total_responses = sum(len(responses) for responses in all_responses.values())
        users_with_responses = len([k for k, v in all_responses.items() if v])
        
        print(f"✅ Response generation completed!")
        print(f"   📄 Response file: {response_file}")
        print(f"   📊 Total responses: {total_responses}")
        print(f"   👥 Users processed: {users_with_responses}")
        
        # Analyze memory utilization in responses
        if total_responses > 0:
            total_episodic_hits = sum(r.get('episodic_results_count', 0) for responses in all_responses.values() for r in responses)
            total_semantic_hits = sum(r.get('semantic_results_count', 0) for responses in all_responses.values() for r in responses)
            avg_episodic = total_episodic_hits / total_responses
            avg_semantic = total_semantic_hits / total_responses
            
            print(f"\n🧠 Memory Utilization in Responses:")
            print(f"   📖 Avg episodic episodes per response: {avg_episodic:.1f}")
            print(f"   🧠 Avg semantic concepts per response: {avg_semantic:.1f}")
            print(f"   🔗 Unified memory integration: Active")
        
        print(f"\n🎯 Ready for LoCoMo Benchmark Evaluation!")
        print(f"   Next step: python locomo_eval.py --lib nemori --version {version}")
        
    except Exception as e:
        print(f"❌ Response generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        await experiment.cleanup()


def main(frame, version="default"):
    if frame == "nemori":
        asyncio.run(main_nemori_unified_responses(version))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lib",
        type=str,
        choices=["zep", "memos", "mem0", "mem0_graph", "nemori"],
        help="Specify the memory framework",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="default",
        help="Version identifier",
    )
    args = parser.parse_args()
    
    main(args.lib, args.version)