"""
ChromaDB向量搜索引擎
"""

import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from ..models import Episode, SemanticMemory
from ..utils import EmbeddingClient
from ..config import MemoryConfig

logger = logging.getLogger(__name__)


class ChromaSearchEngine:
    """基于ChromaDB的简化向量搜索引擎"""
    
    def __init__(self, embedding_client: EmbeddingClient, config: MemoryConfig):
        """
        初始化ChromaDB搜索引擎
        
        Args:
            embedding_client: 嵌入生成客户端
            config: 内存系统配置
        """
        self.embedding_client = embedding_client
        self.config = config
        
        # 初始化ChromaDB客户端
        self.persist_directory = config.chroma_persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # 创建ChromaDB客户端，启用持久化
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,  # 禁用遥测
                allow_reset=True
            )
        )
        
        # 集合名称前缀
        self.collection_prefix = config.chroma_collection_prefix
        
        # 缓存集合对象
        self._episode_collections: Dict[str, Any] = {}
        self._semantic_collections: Dict[str, Any] = {}
        
        logger.info(f"ChromaDB搜索引擎已初始化，持久化目录: {self.persist_directory}")
    
    def _get_episode_collection_name(self, user_id: str) -> str:
        """获取用户Episode集合名称"""
        return f"{self.collection_prefix}_{user_id}_episodes"
    
    def _get_semantic_collection_name(self, user_id: str) -> str:
        """获取用户SemanticMemory集合名称"""
        return f"{self.collection_prefix}_{user_id}_semantic"
    
    def _get_episode_collection(self, user_id: str):
        """获取或创建用户Episode集合"""
        if user_id not in self._episode_collections:
            collection_name = self._get_episode_collection_name(user_id)
            try:
                # 尝试获取现有集合
                collection = self.client.get_collection(name=collection_name)
                logger.debug(f"获取到现有Episode集合: {collection_name}")
            except (ValueError, Exception) as e:
                # 集合不存在或其他错误，创建新集合
                try:
                    collection = self.client.create_collection(
                        name=collection_name,
                        metadata={"user_id": user_id, "type": "episodes"}
                    )
                    logger.info(f"创建新Episode集合: {collection_name}")
                except Exception as create_error:
                    logger.error(f"创建Episode集合失败: {create_error}")
                    # 尝试删除可能存在的损坏集合并重新创建
                    try:
                        self.client.delete_collection(name=collection_name)
                    except:
                        pass
                    collection = self.client.create_collection(
                        name=collection_name,
                        metadata={"user_id": user_id, "type": "episodes"}
                    )
                    logger.info(f"重新创建Episode集合: {collection_name}")
            
            self._episode_collections[user_id] = collection
        
        return self._episode_collections[user_id]
    
    def _get_semantic_collection(self, user_id: str):
        """获取或创建用户SemanticMemory集合"""
        if user_id not in self._semantic_collections:
            collection_name = self._get_semantic_collection_name(user_id)
            try:
                # 尝试获取现有集合
                collection = self.client.get_collection(name=collection_name)
                logger.debug(f"获取到现有Semantic集合: {collection_name}")
            except (ValueError, Exception) as e:
                # 集合不存在或其他错误，创建新集合
                try:
                    collection = self.client.create_collection(
                        name=collection_name,
                        metadata={"user_id": user_id, "type": "semantic"}
                    )
                    logger.info(f"创建新Semantic集合: {collection_name}")
                except Exception as create_error:
                    logger.error(f"创建Semantic集合失败: {create_error}")
                    # 尝试删除可能存在的损坏集合并重新创建
                    try:
                        self.client.delete_collection(name=collection_name)
                    except:
                        pass
                    collection = self.client.create_collection(
                        name=collection_name,
                        metadata={"user_id": user_id, "type": "semantic"}
                    )
                    logger.info(f"重新创建Semantic集合: {collection_name}")
            
            self._semantic_collections[user_id] = collection
        
        return self._semantic_collections[user_id]
    
    def load_user_indices(self, user_id: str, episodes: List[Episode], semantic_memories: List[SemanticMemory]):
        """
        为用户加载索引（批量初始化）
        
        Args:
            user_id: 用户ID
            episodes: Episode列表
            semantic_memories: SemanticMemory列表
        """
        try:
            # 获取集合
            episode_collection = self._get_episode_collection(user_id)
            semantic_collection = self._get_semantic_collection(user_id)
            
            # 检查集合是否已有数据
            episode_count = episode_collection.count()
            semantic_count = semantic_collection.count()
            
            logger.debug(f"用户 {user_id} - Episode集合现有 {episode_count} 条记录, Semantic集合现有 {semantic_count} 条记录")
            
            # 如果集合为空或数据不匹配，重新索引
            if episode_count == 0 and episodes:
                self.index_episodes(user_id, episodes)
            
            if semantic_count == 0 and semantic_memories:
                self.index_semantic_memories(user_id, semantic_memories)
            
            logger.info(f"用户 {user_id} 的索引加载完成")
            
        except Exception as e:
            logger.error(f"加载用户 {user_id} 索引失败: {e}")
            raise
    
    def index_episodes(self, user_id: str, episodes: List[Episode]):
        """
        为用户索引Episodes
        
        Args:
            user_id: 用户ID
            episodes: Episode列表
        """
        try:
            if not episodes:
                logger.debug(f"用户 {user_id} 没有Episode需要索引")
                return
            
            collection = self._get_episode_collection(user_id)
            
            # 准备数据
            ids = []
            documents = []
            metadatas = []
            embeddings = []
            
            for episode in episodes:
                # 检查是否已存在
                existing = collection.get(ids=[episode.episode_id])
                if existing['ids']:
                    logger.debug(f"Episode {episode.episode_id} 已存在，跳过")
                    continue
                
                # 准备文档内容（标题 + 内容）
                document_text = f"{episode.title}. {episode.content}"
                
                # 准备元数据
                metadata = {
                    "episode_id": episode.episode_id,
                    "user_id": episode.user_id,
                    "title": episode.title,
                    "message_count": episode.message_count,
                    "boundary_reason": episode.boundary_reason,
                    "created_at": episode.created_at.isoformat(),
                    "timestamp": episode.timestamp.isoformat(),
                    "type": "episode"
                }
                
                ids.append(episode.episode_id)
                documents.append(document_text)
                metadatas.append(metadata)
            
            if not ids:
                logger.debug(f"用户 {user_id} 所有Episode都已存在")
                return
            
            # 生成embeddings
            embedding_response = self.embedding_client.embed_texts(documents)
            embeddings = embedding_response.embeddings
            
            # 添加到ChromaDB
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            logger.info(f"为用户 {user_id} 索引了 {len(ids)} 个Episode")
            
        except Exception as e:
            logger.error(f"索引用户 {user_id} 的Episode失败: {e}")
            raise
    
    def index_semantic_memories(self, user_id: str, memories: List[SemanticMemory]):
        """
        为用户索引SemanticMemory
        
        Args:
            user_id: 用户ID
            memories: SemanticMemory列表
        """
        try:
            if not memories:
                logger.debug(f"用户 {user_id} 没有SemanticMemory需要索引")
                return
            
            collection = self._get_semantic_collection(user_id)
            
            # 准备数据
            ids = []
            documents = []
            metadatas = []
            embeddings = []
            
            for memory in memories:
                # 检查是否已存在
                existing = collection.get(ids=[memory.memory_id])
                if existing['ids']:
                    logger.debug(f"SemanticMemory {memory.memory_id} 已存在，跳过")
                    continue
                
                # 准备元数据
                metadata = {
                    "memory_id": memory.memory_id,
                    "user_id": memory.user_id,
                    "knowledge_type": memory.knowledge_type,
                    "confidence": memory.confidence,
                    "created_at": memory.created_at.isoformat(),
                    "source_episodes": ",".join(memory.source_episodes) if memory.source_episodes else "",  # 转换列表为字符串
                    "revision_count": memory.revision_count,
                    "type": "semantic"
                }
                
                if memory.updated_at:
                    metadata["updated_at"] = memory.updated_at.isoformat()
                
                ids.append(memory.memory_id)
                documents.append(memory.content)
                metadatas.append(metadata)
            
            if not ids:
                logger.debug(f"用户 {user_id} 所有SemanticMemory都已存在")
                return
            
            # 生成embeddings
            embedding_response = self.embedding_client.embed_texts(documents)
            embeddings = embedding_response.embeddings
            
            # 添加到ChromaDB
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            logger.info(f"为用户 {user_id} 索引了 {len(ids)} 个SemanticMemory")
            
        except Exception as e:
            logger.error(f"索引用户 {user_id} 的SemanticMemory失败: {e}")
            raise
    
    def add_episode(self, user_id: str, episode: Episode):
        """
        增量添加单个Episode
        
        Args:
            user_id: 用户ID
            episode: Episode对象
        """
        try:
            collection = self._get_episode_collection(user_id)
            
            # 检查是否已存在
            existing = collection.get(ids=[episode.episode_id])
            if existing['ids']:
                logger.debug(f"Episode {episode.episode_id} 已存在，跳过添加")
                return
            
            # 准备数据
            document_text = f"{episode.title}. {episode.content}"
            metadata = {
                "episode_id": episode.episode_id,
                "user_id": episode.user_id,
                "title": episode.title,
                "message_count": episode.message_count,
                "boundary_reason": episode.boundary_reason,
                "created_at": episode.created_at.isoformat(),
                "timestamp": episode.timestamp.isoformat(),
                "type": "episode"
            }
            
            # 生成embedding
            embedding_response = self.embedding_client.embed_texts([document_text])
            embedding = embedding_response.embeddings[0]
            
            # 添加到ChromaDB
            collection.add(
                ids=[episode.episode_id],
                documents=[document_text],
                metadatas=[metadata],
                embeddings=[embedding]
            )
            
            logger.debug(f"成功添加Episode {episode.episode_id} 到用户 {user_id}")
            
        except Exception as e:
            logger.error(f"添加Episode {episode.episode_id} 到用户 {user_id} 失败: {e}")
            raise
    
    def add_semantic_memory(self, user_id: str, memory: SemanticMemory, embedding: Optional[List[float]] = None):
        """
        增量添加单个SemanticMemory
        
        Args:
            user_id: 用户ID
            memory: SemanticMemory对象
            embedding: 已计算的嵌入向量（可选）
        """
        try:
            collection = self._get_semantic_collection(user_id)
            
            # 检查是否已存在
            existing = collection.get(ids=[memory.memory_id])
            if existing['ids']:
                logger.debug(f"SemanticMemory {memory.memory_id} 已存在，跳过添加")
                return
            
            # 准备元数据
            metadata = {
                "memory_id": memory.memory_id,
                "user_id": memory.user_id,
                "knowledge_type": memory.knowledge_type,
                "confidence": memory.confidence,
                "created_at": memory.created_at.isoformat(),
                "source_episodes": ",".join(memory.source_episodes) if memory.source_episodes else "",  # 转换列表为字符串
                "revision_count": memory.revision_count,
                "type": "semantic"
            }
            
            if memory.updated_at:
                metadata["updated_at"] = memory.updated_at.isoformat()
            
            # 生成embedding（如未提供）
            if embedding is None:
                embedding_response = self.embedding_client.embed_texts([memory.content])
                embedding = embedding_response.embeddings[0]

            # 添加到ChromaDB
            collection.add(
                ids=[memory.memory_id],
                documents=[memory.content],
                metadatas=[metadata],
                embeddings=[embedding]
            )

            logger.debug(f"成功添加SemanticMemory {memory.memory_id} 到用户 {user_id}")
            
        except Exception as e:
            logger.error(f"添加SemanticMemory {memory.memory_id} 到用户 {user_id} 失败: {e}")
            raise

    def get_semantic_embeddings(self, user_id: str, memory_ids: List[str]) -> Dict[str, Optional[List[float]]]:
        """批量获取指定语义记忆的嵌入向量"""
        if not memory_ids:
            return {}

        collection = self._get_semantic_collection(user_id)
        try:
            results = collection.get(ids=memory_ids, include=["embeddings"])
        except Exception as e:
            logger.warning(f"获取用户 {user_id} 语义向量失败: {e}")
            return {}

        embeddings_map: Dict[str, Optional[List[float]]] = {}
        result_ids = results.get("ids", [])
        result_embeddings = results.get("embeddings", [])

        for idx, mem_id in enumerate(result_ids):
            if idx < len(result_embeddings) and result_embeddings[idx] is not None:
                embeddings_map[mem_id] = list(result_embeddings[idx])
            else:
                embeddings_map[mem_id] = None

        return embeddings_map
    
    def search_episodes(self, user_id: str, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        搜索Episodes
        
        Args:
            user_id: 用户ID
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        try:
            collection = self._get_episode_collection(user_id)
            
            # 检查集合是否为空
            if collection.count() == 0:
                logger.debug(f"用户 {user_id} 的Episode集合为空")
                return []
            
            # 生成查询embedding
            query_embedding = self.embedding_client.embed_text(query)
            
            # 执行搜索
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection.count())
            )
            
            # 处理结果
            search_results = []
            for i, doc_id in enumerate(results['ids'][0]):
                result = {
                    "episode_id": doc_id,
                    "title": results['metadatas'][0][i].get('title', ''),
                    "content": results['documents'][0][i],
                    "distance": results['distances'][0][i],
                    "score": 1 - results['distances'][0][i],  # 转换为相似度分数
                    "metadata": results['metadatas'][0][i],
                    "type": "episode"
                }
                search_results.append(result)
            
            logger.debug(f"为用户 {user_id} 搜索Episode返回 {len(search_results)} 个结果")
            return search_results
            
        except Exception as e:
            logger.error(f"搜索用户 {user_id} 的Episode失败: {e}")
            return []
    
    def search_semantic_memories(self, user_id: str, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        搜索SemanticMemory
        
        Args:
            user_id: 用户ID
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        try:
            collection = self._get_semantic_collection(user_id)
            
            # 检查集合是否为空
            if collection.count() == 0:
                logger.debug(f"用户 {user_id} 的SemanticMemory集合为空")
                return []
            
            # 生成查询embedding
            query_embedding = self.embedding_client.embed_text(query)
            
            # 执行搜索
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection.count())
            )
            
            # 处理结果
            search_results = []
            for i, doc_id in enumerate(results['ids'][0]):
                result = {
                    "memory_id": doc_id,
                    "content": results['documents'][0][i],
                    "knowledge_type": results['metadatas'][0][i].get('knowledge_type', ''),
                    "distance": results['distances'][0][i],
                    "score": 1 - results['distances'][0][i],  # 转换为相似度分数
                    "metadata": results['metadatas'][0][i],
                    "type": "semantic"
                }
                search_results.append(result)
            
            logger.debug(f"为用户 {user_id} 搜索SemanticMemory返回 {len(search_results)} 个结果")
            return search_results
            
        except Exception as e:
            logger.error(f"搜索用户 {user_id} 的SemanticMemory失败: {e}")
            return []
    
    def clear_user_index(self, user_id: str) -> bool:
        """
        清除用户的所有索引
        
        Args:
            user_id: 用户ID
            
        Returns:
            是否成功清除
        """
        try:
            success = True
            
            # 删除Episode集合
            episode_collection_name = self._get_episode_collection_name(user_id)
            try:
                self.client.delete_collection(name=episode_collection_name)
                logger.info(f"删除用户 {user_id} 的Episode集合: {episode_collection_name}")
            except (ValueError, Exception) as e:
                logger.debug(f"Episode集合 {episode_collection_name} 删除失败或不存在: {e}")
            
            # 删除SemanticMemory集合
            semantic_collection_name = self._get_semantic_collection_name(user_id)
            try:
                self.client.delete_collection(name=semantic_collection_name)
                logger.info(f"删除用户 {user_id} 的SemanticMemory集合: {semantic_collection_name}")
            except (ValueError, Exception) as e:
                logger.debug(f"SemanticMemory集合 {semantic_collection_name} 删除失败或不存在: {e}")
            
            # 清除缓存
            self._episode_collections.pop(user_id, None)
            self._semantic_collections.pop(user_id, None)
            
            return success
            
        except Exception as e:
            logger.error(f"清除用户 {user_id} 索引失败: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取搜索引擎统计信息"""
        try:
            collections = self.client.list_collections()
            
            stats = {
                "engine_type": "ChromaDB",
                "persist_directory": self.persist_directory,
                "total_collections": len(collections),
                "collections": []
            }
            
            for collection in collections:
                collection_info = {
                    "name": collection.name,
                    "count": collection.count(),
                    "metadata": collection.metadata
                }
                stats["collections"].append(collection_info)
            
            return stats
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"error": str(e)}
