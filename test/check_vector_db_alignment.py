import os
import re
import sys
import json
import argparse
from typing import Dict, Tuple, List

import numpy as np
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    faiss = None  # type: ignore
    FAISS_AVAILABLE = False


def count_jsonl_records(jsonl_path: str) -> int:
    count = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError:
                # 跳过坏行，但仍然报告
                print(f"[WARN] Bad JSON line skipped: {jsonl_path}", file=sys.stderr)
    return count


def build_expected_counts(group_dir: str, suffix: str) -> Dict[str, int]:
    """
    遍历目录下的 *{suffix}.jsonl 文件，构建 {user_key: 记录数}
    例如：Caroline_0_episodes.jsonl -> user_key = Caroline_0
    """
    expected: Dict[str, int] = {}
    for name in os.listdir(group_dir):
        if not name.endswith('.jsonl'):
            continue
        if not name.endswith(suffix + '.jsonl'):
            continue
        path = os.path.join(group_dir, name)
        user_key = name[: -len(suffix + '.jsonl')]
        expected[user_key] = count_jsonl_records(path)
    return expected


def read_embeddings_count(npy_path: str) -> int:
    arr = np.load(npy_path, mmap_mode='r')
    if arr.ndim == 1:
        return int(arr.shape[0] // 1)
    return int(arr.shape[0])


def read_faiss_count(index_path: str):
    if not FAISS_AVAILABLE:
        return None
    index = faiss.read_index(index_path)  # type: ignore
    return int(index.ntotal)


def check_group(group_name: str, base_dir: str, suffix: str) -> Tuple[int, int]:
    group_dir = os.path.join(base_dir, group_name)
    vec_dir = os.path.join(group_dir, 'vector_db')
    expected = build_expected_counts(group_dir, suffix)

    total = 0
    mismatches = 0
    print(f"\n=== {group_name.upper()} ===")
    if not expected:
        print(f"[WARN] No *{suffix}.jsonl found under {group_dir}")
        return 0, 0

    for user_key, exp_count in sorted(expected.items()):
        total += 1
        faiss_path = os.path.join(vec_dir, f"{user_key}.faiss")
        npy_path = os.path.join(vec_dir, f"{user_key}_embeddings.npy")

        faiss_count = None
        emb_count = None
        status: List[str] = []

        if os.path.exists(npy_path):
            try:
                emb_count = read_embeddings_count(npy_path)
                status.append(f"emb={emb_count}")
            except Exception as e:
                status.append(f"emb=ERR({e.__class__.__name__})")
        else:
            status.append("emb=MISSING")

        if os.path.exists(faiss_path):
            try:
                faiss_count = read_faiss_count(faiss_path)
                if faiss_count is None:
                    status.append("faiss=SKIP(no_module)")
                else:
                    status.append(f"faiss={faiss_count}")
            except Exception as e:
                status.append(f"faiss=ERR({e.__class__.__name__})")
        else:
            status.append("faiss=MISSING")

        ok = True
        # 至少需要校验 embeddings；faiss 若不可用则不参与判断
        if emb_count is None:
            ok = False
        else:
            if emb_count != exp_count:
                ok = False
            if faiss_count is not None:
                if faiss_count != exp_count or emb_count != faiss_count:
                    ok = False

        result = "OK" if ok else "FAIL"
        if not ok:
            mismatches += 1

        print(f"- {user_key:20s} | jsonl={exp_count:5d} | {'; '.join(status):30s} | {result}")

    print(f"Summary: checked={total}, mismatches={mismatches}")
    return total, mismatches


def main():
    parser = argparse.ArgumentParser(description='Check vector DB alignment with JSONL counts')
    parser.add_argument('--base', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'evaluation', 'memories'),
                        help='Base directory of memories (default: ../evaluation/memories)')
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base)
    print(f"Base: {base_dir}")

    total_a, mism_a = check_group('episodes', base_dir, suffix='_episodes')
    total_b, mism_b = check_group('semantic', base_dir, suffix='_semantic')

    mismatches = mism_a + mism_b
    if mismatches > 0:
        print(f"\n[FAIL] Found {mismatches} users with misaligned counts.")
        sys.exit(1)
    else:
        print("\n[OK] All vector DB indices align with JSONL counts.")


if __name__ == '__main__':
    main()


