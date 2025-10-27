import os
import requests
import json
import numpy as np
import faiss  # 需要安装 faiss-cpu 或 faiss-gpu
from typing import List, Dict, Any, Optional

class FaissRAG:
    """
    一个使用 FAISS 和 JSON 文件存储的简单RAG系统。
    - FAISS 用于存储和搜索文档键 (key) 的嵌入向量。
    - JSON 用于存储文档的 ID、键 (key)、值 (value) 和元数据。
    """
    
    def __init__(
        self, 
        json_path: str = "rag_data.json", 
        faiss_index_path: str = "rag_index.faiss",
        api_url: str = "http://localhost:6007/v1",
        model_name: str = "qwen3-emb" # 嵌入模型名称
    ):
        self.api_url = api_url
        self.model_name = model_name
        self.json_path = json_path
        self.faiss_index_path = faiss_index_path
        
        self.documents: Dict[int, Dict[str, Any]] = {}
        self.index: Optional[faiss.IndexIDMap] = None
        self.dimension: Optional[int] = None

        self._load()
    
    def _load(self):
        """从文件加载文档和FAISS索引"""
        # 加载 JSON 数据
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r', encoding='utf-8') as f:
                # 将json的字符串key转为int
                self.documents = {int(k): v for k, v in json.load(f).items()}
            print(f"从 {self.json_path} 加载了 {len(self.documents)} 个文档。")
        
        # 加载 FAISS 索引
        if os.path.exists(self.faiss_index_path):
            self.index = faiss.read_index(self.faiss_index_path)
            self.dimension = self.index.d
            print(f"从 {self.faiss_index_path} 加载了 FAISS 索引，维度: {self.dimension}。")
        
    def _save(self):
        """将文档和FAISS索引保存到文件"""
        # 保存 JSON 数据
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=4)
        
        # 保存 FAISS 索引
        if self.index:
            faiss.write_index(self.index, self.faiss_index_path)
        print("数据和索引已保存。")

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """使用API获取文本嵌入"""
        try:
            response = requests.post(
                f"{self.api_url}/embeddings",
                json={
                    "model": self.model_name,
                    "input": text
                },
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()
            embedding = result["data"][0]["embedding"]
            
            # 归一化嵌入，这对于使用内积计算余弦相似度至关重要
            embedding_np = np.array(embedding, dtype='float32')
            faiss.normalize_L2(embedding_np.reshape(1, -1))
            return embedding_np.flatten().tolist()
            
        except Exception as e:
            print(f"获取嵌入失败: {e}")
            return None

    def add_document(self, key: str, value: str, metadata: Dict[str, Any] = None) -> int:
        """
        添加一个键值对文档到RAG系统。
        系统会对'key'进行嵌入并用于后续的相似度搜索。
        """
        if metadata is None:
            metadata = {}
        
        # 1. 获取 'key' 的嵌入
        embedding = self.get_embedding(key)
        if embedding is None:
            print("无法添加文档，因为获取嵌入失败。")
            return None
        
        embedding_np = np.array([embedding], dtype='float32')
        
        # 2. 初始化 FAISS 索引 (如果尚未存在)
        if self.index is None:
            self.dimension = embedding_np.shape[1]
            # 使用内积 (IP) 索引，因为归一化向量的内积等于余弦相似度
            index_flat = faiss.IndexFlatIP(self.dimension)
            # 使用 IndexIDMap 来支持通过ID删除
            self.index = faiss.IndexIDMap(index_flat)
            print(f"已初始化 FAISS 索引，维度: {self.dimension}")

        # 检查维度是否一致
        if embedding_np.shape[1] != self.dimension:
            print(f"错误: 嵌入维度 ({embedding_np.shape[1]}) 与索引维度 ({self.dimension}) 不匹配。")
            return None

        # 3. 生成唯一的文档ID
        doc_id = max(self.documents.keys()) + 1 if self.documents else 1
        
        # 4. 将嵌入和ID添加到FAISS索引
        self.index.add_with_ids(embedding_np, np.array([doc_id]))
        
        # 5. 将文档内容存入字典
        self.documents[doc_id] = {
            "key": key,
            "value": value,
            "metadata": metadata
        }
        
        # 6. 保存更改
        self._save()
        
        return doc_id

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """根据查询与存储的'key'进行相似度搜索，返回对应的'value'和元数据。"""
        if self.index is None or not self.documents:
            print("系统中没有文档，无法搜索。")
            return []
            
        # 1. 获取查询的嵌入
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            return []
        
        query_embedding_np = np.array([query_embedding], dtype='float32')
        
        # 2. 在FAISS中搜索
        # 返回的 'distances' 是内积值，即余弦相似度
        similarities, doc_ids = self.index.search(query_embedding_np, top_k)
        
        # 3. 格式化返回结果
        results = []
        for i in range(len(doc_ids[0])):
            doc_id = doc_ids[0][i]
            if doc_id in self.documents:
                doc = self.documents[doc_id]
                results.append({
                    "id": int(doc_id),
                    "key": doc["key"],
                    "value": doc["value"],
                    "metadata": doc["metadata"],
                    "similarity": float(similarities[0][i])
                })
        
        return results

    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """通过ID获取单个文档"""
        return self.documents.get(doc_id)

    def list_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """列出所有文档"""
        return list(self.documents.values())[:limit]

    def delete_document(self, doc_id: int) -> bool:
        """通过ID删除文档"""
        if doc_id not in self.documents:
            print(f"文档ID {doc_id} 不存在。")
            return False
            
        try:
            # 1. 从FAISS索引中删除
            self.index.remove_ids(np.array([doc_id]))
            
            # 2. 从字典中删除
            del self.documents[doc_id]
            
            # 3. 保存更改
            self._save()
            print(f"文档ID {doc_id} 已成功删除。")
            return True
        except Exception as e:
            print(f"删除文档ID {doc_id} 失败: {e}")
            return False

def main():
    """示例使用"""
    # 清理旧文件以便重新演示
    if os.path.exists("rag_data.json"):
        os.remove("rag_data.json")
    if os.path.exists("rag_index.faiss"):
        os.remove("rag_index.faiss")
        
    # 初始化RAG系统
    rag = FaissRAG()
    
    print("\n=== FAISS + JSON RAG系统演示 ===")
    
    # 添加示例文档 (key-value 形式)
    sample_docs = [
        {
            "key": "人工智能的定义",
            "value": "人工智能（AI）是计算机科学的一个前沿分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
        },
        {
            "key": "机器学习是什么",
            "value": "机器学习是人工智能的一个核心子领域，它专注于研究计算机如何模拟或实现人类的学习行为，以获取新的知识或技能，并重新组织已有的知识结构，使之不断改善自身的性能。",
        },
        {
            "key": "深度学习的应用领域",
            "value": "深度学习通过构建深层神经网络，在许多领域取得了巨大成功，例如在计算机视觉中的图像识别、在自然语言处理中的机器翻译和语音识别等。",
        }
    ]
    
    print("\n添加文档...")
    for doc in sample_docs:
        doc_id = rag.add_document(doc["key"], doc["value"])
        if doc_id:
            print(f"文档已添加，ID: {doc_id}, Key: '{doc['key']}'")
    
    # 搜索测试
    queries = [
        "介绍一下机器学习",
        "深度学习用在什么地方？",
        "AI的全称是什么"
    ]
    
    print("\n--- 搜索测试 ---")
    for query in queries:
        print(f"\n[查询]: {query}")
        results = rag.search(query, top_k=2)
        if not results:
            print("  未找到相关结果。")
            continue
        for i, result in enumerate(results, 1):
            print(f"  {i}. 相似度: {result['similarity']:.4f} (与Key: '{result['key']}')")
            print(f"     返回内容(Value): {result['value'][:100]}...")
            print(f"     元数据: {result['metadata']}")
            
    # 删除一个文档并验证
    print("\n--- 删除测试 ---")
    doc_to_delete_id = 2
    print(f"尝试删除文档ID: {doc_to_delete_id}...")
    rag.delete_document(doc_to_delete_id)
    
    print(f"\n再次搜索 '介绍一下机器学习' (之前最相关的文档已被删除):")
    results = rag.search("介绍一下机器学习", top_k=2)
    for i, result in enumerate(results, 1):
        print(f"  {i}. 相似度: {result['similarity']:.4f} (与Key: '{result['key']}')")
        print(f"     返回内容(Value): {result['value'][:100]}...")

    print("\n演示完成！")

if __name__ == "__main__":
    # 确保你的嵌入API服务正在运行
    # 例如，使用 LocalAI 或 vLLM 部署 qwen2-emb 模型
    main()