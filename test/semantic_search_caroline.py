import os
import sys
import json
import argparse
import numpy as np

# 将项目 src 加入路径，复用内置 EmbeddingClient
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(REPO_ROOT, 'src'))
from utils.embedding_client import EmbeddingClient  # noqa: E402


# 注意：你要求将密钥写入代码，以下常量即为你提供的密钥。
# 使用后建议删除此文件或将密钥改为从环境变量读取，避免泄漏风险。
OPENAI_API_KEY_HARDCODED = "***REMOVED***"


def load_semantic_corpus(user_key: str):
    base = os.path.join(REPO_ROOT, 'evaluation', 'memories', 'semantic')
    emb_path = os.path.join(base, 'vector_db', f'{user_key}_embeddings.npy')
    jsonl_path = os.path.join(base, f'{user_key}_semantic.jsonl')

    if not os.path.exists(emb_path):
        raise FileNotFoundError(f'Embeddings not found: {emb_path}')
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f'Semantic JSONL not found: {jsonl_path}')

    E = np.load(emb_path, mmap_mode='r')  # (N, D) 已归一化
    rows = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if len(rows) != E.shape[0]:
        print(f'[WARN] embeddings({E.shape[0]}) != jsonl({len(rows)})')
    return E, rows


def search(user_key: str, query: str, k: int = 10, model: str = 'text-embedding-3-small'):
    E, rows = load_semantic_corpus(user_key)
    client = EmbeddingClient(api_key=OPENAI_API_KEY_HARDCODED, model=model)
    qv = client.embed_text(query)
    if not qv:
        raise RuntimeError('Empty embedding returned for query')
    q = np.asarray(qv, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-12)

    # 余弦相似度：库向量已归一化，点乘即可
    scores = E @ q
    idx = np.argsort(-scores)[:k]

    results = []
    for i in idx:
        o = rows[int(i)]
        results.append({
            'rank': len(results) + 1,
            'score': float(scores[i]),
            'memory_id': o.get('memory_id', ''),
            'content': o.get('content', '')
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', default='Caroline_0', help='User key, e.g., Caroline_0')
    parser.add_argument('--k', type=int, default=10, help='Top-K')
    parser.add_argument('--query', default="What inspired Caroline's painting for the art show?", help='Query text')
    args = parser.parse_args()

    try:
        results = search(args.user, args.query, args.k)
        for r in results:
            content = r['content']
            preview = content if len(content) <= 200 else content[:200] + '...'
            print(f"#{r['rank']}\tscore={r['score']:.4f}\tmemory_id={r['memory_id']}\t{preview}")
    except Exception as e:
        # 避免打印密钥，错误仅输出类型与消息
        print(f'[ERROR] {e.__class__.__name__}: {e}')


if __name__ == '__main__':
    main()


