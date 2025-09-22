import os
import sys
import json
import argparse

# 允许从仓库根目录导入 evaluation 与 src
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(REPO_ROOT)

from evaluation.locomo.search import MemorySystemSearch  # noqa: E402


def run_once(user_id: str, query: str, k_sem: int = 20, k_epi: int = 10):
    """仿照 evaluation/locomo/search.py 的写法做单次检索，并打印Top语义结果"""
    # 与 evaluation 目录相同的相对目录设置
    storage_path = os.path.join('evaluation', 'memories')

    searcher = MemorySystemSearch(
        output_path=os.path.join('evaluation', 'locomo', 'results_test_single.json'),
        storage_path=storage_path,
        model='gpt-4o-mini',
        language='en',
        top_k_episodes=k_epi,
        top_k_semantic=k_sem,
        include_original_messages_top_k=2,
        max_workers=8,
        save_batch_size=50,
        enable_memory_cleanup=False,
        search_method='vector',
    )

    results, t = searcher.search_memory(user_id, query)

    # 只看语义记忆
    semantic = [m for m in results if m.get('memory_type') == 'semantic']
    print(f"Search time: {t:.3f}s, semantic_top={len(semantic)}")

    for i, m in enumerate(semantic[:k_sem], 1):
        score = m.get('score')
        mid = m.get('episode_id')
        content = m.get('memory', '')
        preview = content if len(content) <= 200 else content[:200] + '...'
        print(f"#{i}\tscore={score}\tmemory_id={mid}\t{preview}")

    return semantic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', default='Caroline_0')
    parser.add_argument('--query', default="What inspired Caroline's painting for the art show?")
    parser.add_argument('--k_sem', type=int, default=20)
    parser.add_argument('--k_epi', type=int, default=10)
    args = parser.parse_args()

    sem = run_once(args.user, args.query, args.k_sem, args.k_epi)

    # 目标语义记忆子串检测（便于快速定位）
    target_sub = "Caroline's painting for the art show was inspired by her visit to an LGBTQ center"
    hit = next((m for m in sem if target_sub.lower() in m.get('memory', '').lower()), None)
    print("\nContains target semantic? ", bool(hit))
    if hit:
        print("Target memory_id:", hit.get('episode_id'))


if __name__ == '__main__':
    main()


