"""
PostgreSQL implementation for semantic memory storage.

This module provides semantic memory storage using PostgreSQL with full-text search
and proper semantic node/relationship management.
"""

import json
import time
from typing import Any

import numpy as np
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine, async_sessionmaker
from sqlmodel import and_, delete, func, or_, select
import asyncio

from ..core.data_types import RelationshipType, SemanticNode, SemanticRelationship
from .repository import SemanticMemoryRepository
from .sql_models import BaseSQLRepository, SemanticNodeTable, SemanticRelationshipTable
from .storage_types import (
    DuplicateKeyError,
    NotFoundError,
    SemanticNodeQuery,
    SemanticRelationshipQuery,
    SemanticSearchResult,
    SortOrder,
    StorageConfig,
    StorageStats,
)


class PostgreSQLSemanticMemoryRepository(SemanticMemoryRepository, BaseSQLRepository):
    """PostgreSQL implementation of semantic memory repository."""

    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self.connection_string = config.connection_string
        self.engine = None
        self.session_factory = None
        self._initialized = False
        # Initialize OpenAI client for embeddings
        self.openai_client = None
        self._setup_embedding_client(config)
        # Embedding query cache
        self._embedding_cache = {}
        self._cache_ttl = 3600

    async def initialize(self) -> None:
        """Initialize the PostgreSQL semantic memory storage."""
        # Create async engine with larger pool for high concurrency
        self.engine = create_async_engine(
            self.connection_string,
            echo=False,
            pool_size=50,
            max_overflow=50,
            pool_recycle=3600,
            pool_pre_ping=True,
        )

        # Create session factory for connection pooling
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Initialize base class
        BaseSQLRepository.__init__(self, self.engine)

        # Create semantic memory tables
        async with self.engine.begin() as conn:
            await conn.run_sync(
                lambda sync_conn: SemanticNodeTable.metadata.create_all(sync_conn, checkfirst=True)
            )
            await conn.run_sync(
                lambda sync_conn: SemanticRelationshipTable.metadata.create_all(sync_conn, checkfirst=True)
            )

        self._initialized = True

    async def close(self) -> None:
        """Close the database connection."""
        if self.engine:
            await self.engine.dispose()
            self.engine = None
        self._initialized = False

    async def health_check(self) -> bool:
        """Check if storage is healthy."""
        if not self._initialized or not self.engine:
            return False
        try:
            async with self.session_factory() as session:
                await session.execute(select(func.count(SemanticNodeTable.node_id)))
                return True
        except Exception:
            return False

    async def get_stats(self) -> StorageStats:
        """Get semantic memory statistics."""
        stats = StorageStats()

        async with self.session_factory() as session:
            # Count semantic nodes and relationships
            node_count_result = await session.execute(select(func.count(SemanticNodeTable.node_id)))
            node_count = node_count_result.scalar()
            
            relationship_count_result = await session.execute(select(func.count(SemanticRelationshipTable.relationship_id)))
            relationship_count = relationship_count_result.scalar()

            # Use available stats fields appropriately
            stats.total_raw_data = node_count  # Repurpose for semantic node count
            stats.total_episodes = relationship_count  # Repurpose for relationship count

        return stats

    async def backup(self, destination: str) -> bool:
        """Create a backup using pg_dump."""
        try:
            import subprocess
            import os
            from urllib.parse import urlparse
            
            # Parse connection string to get connection details
            parsed = urlparse(self.connection_string)
            
            env = os.environ.copy()
            env['PGPASSWORD'] = parsed.password
            
            cmd = [
                'pg_dump',
                '-h', parsed.hostname,
                '-p', str(parsed.port),
                '-U', parsed.username,
                '-d', parsed.path[1:],  # Remove leading '/'
                '-f', destination
            ]
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            return False

    async def restore(self, source: str) -> bool:
        """Restore from backup using psql."""
        try:
            import subprocess
            import os
            from urllib.parse import urlparse
            
            # Parse connection string to get connection details
            parsed = urlparse(self.connection_string)
            
            env = os.environ.copy()
            env['PGPASSWORD'] = parsed.password
            
            cmd = [
                'psql',
                '-h', parsed.hostname,
                '-p', str(parsed.port),
                '-U', parsed.username,
                '-d', parsed.path[1:],  # Remove leading '/'
                '-f', source
            ]
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            # Reinitialize after restore
            if result.returncode == 0:
                await self.initialize()
            
            return result.returncode == 0
        except Exception:
            return False

    # === Semantic Node Operations ===

    async def store_semantic_node(self, node: SemanticNode) -> None:
        """Store a semantic node."""
        node_id = self.validate_id(node.node_id)

        # Check for duplicate key
        existing = await self.find_semantic_node_by_key(node.owner_id, node.key)
        if existing and existing.node_id != node.node_id:
            print(f"⚠️ Duplicate semantic node key '{node.key}' for owner '{node.owner_id}' - updating existing node")
            # Update existing node instead of raising error
            updated_node = SemanticNode(
                node_id=existing.node_id,
                owner_id=node.owner_id,
                key=node.key,
                value=node.value,
                context=node.context,
                confidence=max(node.confidence, existing.confidence),  # Keep higher confidence
                version=node.version,
                evolution_history=existing.evolution_history + node.evolution_history,
                created_at=existing.created_at,  # Keep original creation time
                last_updated=node.last_updated,
                last_accessed=node.last_accessed,
                discovery_episode_id=node.discovery_episode_id,
                discovery_method=node.discovery_method,
                linked_episode_ids=list(set(existing.linked_episode_ids + node.linked_episode_ids)),
                evolution_episode_ids=list(set(existing.evolution_episode_ids + node.evolution_episode_ids)),
                search_keywords=list(set(existing.search_keywords + node.search_keywords)),
                embedding_vector=node.embedding_vector or existing.embedding_vector,
                access_count=existing.access_count + 1,
                relevance_score=max(node.relevance_score, existing.relevance_score),
                importance_score=max(node.importance_score, existing.importance_score),
            )
            await self.update_semantic_node(updated_node)
            return

        # Generate embedding if not present
        embedding_vector = node.embedding_vector
        if not embedding_vector and self.openai_client:
            # Generate embedding from semantic node content
            embedding_text = f"{node.key}: {node.value} {node.context}"
            embedding_vector = await self._generate_query_embedding(embedding_text)
            if embedding_vector:
                print(f"✅ Generated embedding for semantic node: {node.key[:50]}...")

        try:
            node_row = SemanticNodeTable(
                node_id=node_id,
                owner_id=node.owner_id,
                key=node.key,
                value=node.value,
                context=node.context,
                confidence=node.confidence,
                version=node.version,
                evolution_history=json.dumps(node.evolution_history, ensure_ascii=False),
                created_at=node.created_at,
                last_updated=node.last_updated,
                last_accessed=node.last_accessed,
                discovery_episode_id=node.discovery_episode_id,
                discovery_method=node.discovery_method,
                linked_episode_ids=json.dumps(node.linked_episode_ids, ensure_ascii=False),
                evolution_episode_ids=json.dumps(node.evolution_episode_ids, ensure_ascii=False),
                search_keywords=json.dumps(node.search_keywords, ensure_ascii=False),
                embedding_vector=embedding_vector if embedding_vector is not None else None,
                access_count=node.access_count,
                relevance_score=node.relevance_score,
                importance_score=node.importance_score,
            )

            async with self.session_factory() as session:
                session.add(node_row)
                await session.commit()
                print(f"✅ Successfully stored semantic node '{node.key}' for owner '{node.owner_id}'")
                
        except Exception as e:
            print(f"❌ Error storing semantic node '{node.key}' for owner '{node.owner_id}': {e}")
            raise

    async def get_semantic_node_by_id(self, node_id: str) -> SemanticNode | None:
        """Retrieve a semantic node by its ID."""
        node_id = self.validate_id(node_id)

        async with self.session_factory() as session:
            node_result = await session.execute(select(SemanticNodeTable).where(SemanticNodeTable.node_id == node_id))
            node_row = node_result.first()

            if not node_row:
                return None

            return self._row_to_semantic_node(node_row[0])

    async def find_semantic_node_by_key(self, owner_id: str, key: str) -> SemanticNode | None:
        """Find semantic node by owner and key combination."""
        async with self.session_factory() as session:
            node_result = await session.execute(
                select(SemanticNodeTable).where(
                    and_(SemanticNodeTable.owner_id == owner_id, SemanticNodeTable.key == key)
                )
            )
            node_row = node_result.first()

            if not node_row:
                return None

            return self._row_to_semantic_node(node_row[0])

    async def update_semantic_node(self, node: SemanticNode) -> None:
        """Update an existing semantic node."""
        node_id = self.validate_id(node.node_id)

        async with self.session_factory() as session:
            # Check if node exists
            existing_result = await session.execute(select(SemanticNodeTable).where(SemanticNodeTable.node_id == node_id))
            existing_row = existing_result.first()

            if not existing_row:
                raise NotFoundError(f"Semantic node with ID '{node_id}' not found")

            existing = existing_row[0]
            # Update all fields
            existing.owner_id = node.owner_id
            existing.key = node.key
            existing.value = node.value
            existing.context = node.context
            existing.confidence = node.confidence
            existing.version = node.version
            existing.evolution_history = json.dumps(node.evolution_history, ensure_ascii=False)
            existing.last_updated = node.last_updated
            existing.last_accessed = node.last_accessed
            existing.discovery_episode_id = node.discovery_episode_id
            existing.discovery_method = node.discovery_method
            existing.linked_episode_ids = json.dumps(node.linked_episode_ids, ensure_ascii=False)
            existing.evolution_episode_ids = json.dumps(node.evolution_episode_ids, ensure_ascii=False)
            existing.search_keywords = json.dumps(node.search_keywords, ensure_ascii=False)
            existing.embedding_vector = node.embedding_vector if node.embedding_vector is not None else None
            existing.access_count = node.access_count
            existing.relevance_score = node.relevance_score
            existing.importance_score = node.importance_score

            session.add(existing)
            await session.commit()

    async def delete_semantic_node(self, node_id: str) -> bool:
        """Delete a semantic node by ID."""
        node_id = self.validate_id(node_id)

        async with self.session_factory() as session:
            # Delete related relationships first
            await session.execute(
                delete(SemanticRelationshipTable).where(
                    or_(
                        SemanticRelationshipTable.source_node_id == node_id,
                        SemanticRelationshipTable.target_node_id == node_id,
                    )
                )
            )

            # Delete the node
            result = await session.execute(delete(SemanticNodeTable).where(SemanticNodeTable.node_id == node_id))
            await session.commit()

            return result.rowcount > 0

    async def search_semantic_nodes(self, query: SemanticNodeQuery, _skip_embedding_search: bool = False) -> SemanticSearchResult:
        """Search semantic nodes with complex query parameters.
        
        Args:
            query: The search query parameters
            _skip_embedding_search: Internal flag to prevent recursion in fallback scenarios
        """
        start_time = time.time()
        
        # Priority: Use embedding search if text_search is provided (unless we're in a fallback)
        if query.text_search and self.openai_client and not _skip_embedding_search:
            print(f"🔄 Using embedding-based similarity search for text: '{query.text_search[:50]}...'")
            nodes = await self.similarity_search_semantic_nodes(
                owner_id=query.owner_id,
                query=query.text_search,
                limit=query.limit
            )
            return SemanticSearchResult(
                semantic_nodes=nodes,
                total_nodes=len(nodes),
                has_more_nodes=False,
                query_time_ms=(time.time() - start_time) * 1000,
                query_info={"text_search": query.text_search, "search_type": "embedding"},
            )

        async with self.session_factory() as session:
            stmt = select(SemanticNodeTable).where(SemanticNodeTable.owner_id == query.owner_id)

            # Apply filters
            if query.key_pattern:
                stmt = stmt.where(SemanticNodeTable.key.like(f"%{query.key_pattern}%"))

            if query.value_pattern:
                stmt = stmt.where(SemanticNodeTable.value.like(f"%{query.value_pattern}%"))

            if query.text_search:
                # Simple text search across key, value, and context
                search_term = f"%{query.text_search}%"
                stmt = stmt.where(
                    or_(
                        SemanticNodeTable.key.like(search_term),
                        SemanticNodeTable.value.like(search_term),
                        SemanticNodeTable.context.like(search_term),
                    )
                )

            if query.min_confidence:
                stmt = stmt.where(SemanticNodeTable.confidence >= query.min_confidence)

            if query.discovery_episode_id:
                stmt = stmt.where(SemanticNodeTable.discovery_episode_id == query.discovery_episode_id)

            if query.created_after:
                stmt = stmt.where(SemanticNodeTable.created_at >= query.created_after)

            if query.updated_after:
                stmt = stmt.where(SemanticNodeTable.last_updated >= query.updated_after)

            # Count total before pagination - fix the cartesian product warning
            count_stmt = select(func.count(SemanticNodeTable.node_id)).where(SemanticNodeTable.owner_id == query.owner_id)
            
            # Apply the same filters to the count query
            if query.key_pattern:
                count_stmt = count_stmt.where(SemanticNodeTable.key.like(f"%{query.key_pattern}%"))
            if query.value_pattern:
                count_stmt = count_stmt.where(SemanticNodeTable.value.like(f"%{query.value_pattern}%"))
            if query.text_search:
                search_term = f"%{query.text_search}%"
                count_stmt = count_stmt.where(
                    or_(
                        SemanticNodeTable.key.like(search_term),
                        SemanticNodeTable.value.like(search_term),
                        SemanticNodeTable.context.like(search_term),
                    )
                )
            if query.min_confidence:
                count_stmt = count_stmt.where(SemanticNodeTable.confidence >= query.min_confidence)
            if query.discovery_episode_id:
                count_stmt = count_stmt.where(SemanticNodeTable.discovery_episode_id == query.discovery_episode_id)
            if query.created_after:
                count_stmt = count_stmt.where(SemanticNodeTable.created_at >= query.created_after)
            if query.updated_after:
                count_stmt = count_stmt.where(SemanticNodeTable.last_updated >= query.updated_after)
                
            total_count_result = await session.execute(count_stmt)
            total_count = total_count_result.scalar()

            # Apply sorting
            if query.sort_by == "created_at":
                sort_field = SemanticNodeTable.created_at
            elif query.sort_by == "updated_at":
                sort_field = SemanticNodeTable.last_updated
            elif query.sort_by == "confidence":
                sort_field = SemanticNodeTable.confidence
            elif query.sort_by == "importance_score":
                sort_field = SemanticNodeTable.importance_score
            elif query.sort_by == "access_count":
                sort_field = SemanticNodeTable.access_count
            else:
                sort_field = SemanticNodeTable.created_at

            if query.sort_order == SortOrder.DESC:
                sort_field = sort_field.desc()

            stmt = stmt.order_by(sort_field)

            # Apply pagination
            stmt = stmt.offset(query.offset).limit(query.limit)

            # Execute query
            node_result = await session.execute(stmt)
            node_rows = node_result.fetchall()
            nodes = [self._row_to_semantic_node(row[0]) for row in node_rows]

            query_time_ms = (time.time() - start_time) * 1000

            return SemanticSearchResult(
                semantic_nodes=nodes,
                total_nodes=total_count,
                has_more_nodes=total_count > (query.offset + len(nodes)),
                query_time_ms=query_time_ms,
                query_info={"text_search": query.text_search, "total_results": total_count},
            )

    async def similarity_search_semantic_nodes(self, owner_id: str, query: str, limit: int = 10) -> list[SemanticNode]:
        """Search semantic nodes by embedding similarity to query text."""
        
        async with self.session_factory() as session:
            total_nodes_result = await session.execute(
                select(func.count(SemanticNodeTable.node_id)).where(SemanticNodeTable.owner_id == owner_id)
            )
            total_nodes = total_nodes_result.scalar()
            
            if total_nodes == 0:
                return []
        
        try:
            query_embedding = await self._generate_query_embedding(query)
            
            if query_embedding:
                return await self._embedding_similarity_search(owner_id, query_embedding, limit)
            else:
                return await self._enhanced_text_search_fallback(owner_id, query, limit)
                
        except Exception as e:
            return await self._enhanced_text_search_fallback(owner_id, query, limit)
    
    async def _enhanced_text_search_fallback(self, owner_id: str, query: str, limit: int) -> list[SemanticNode]:
        """Enhanced text search fallback with multiple strategies."""
        print(f"   📝 Starting enhanced text search fallback...")
        
        # Strategy 1: Try direct text search first (skip embedding to prevent recursion)
        search_query = SemanticNodeQuery(
            owner_id=owner_id, text_search=query, limit=limit, sort_by="confidence", sort_order=SortOrder.DESC
        )
        result = await self.search_semantic_nodes(search_query, _skip_embedding_search=True)
        
        if result.semantic_nodes:
            print(f"   ✅ Direct text search found {len(result.semantic_nodes)} nodes")
            return result.semantic_nodes
        
        print(f"   ⚠️ Direct text search found 0 nodes, trying keyword extraction...")
        
        # Strategy 2: Extract individual keywords and search
        keywords = self._extract_search_keywords(query)
        print(f"   🔤 Extracted keywords: {keywords}")
        
        all_results = {}  # Use dict to avoid duplicates
        
        for keyword in keywords:
            if len(keyword) > 2:  # Skip very short keywords
                keyword_query = SemanticNodeQuery(
                    owner_id=owner_id, text_search=keyword, limit=limit*2, sort_by="confidence", sort_order=SortOrder.DESC
                )
                keyword_result = await self.search_semantic_nodes(keyword_query, _skip_embedding_search=True)
                
                # Add results, avoiding duplicates
                for node in keyword_result.semantic_nodes:
                    all_results[node.node_id] = node
                    
                if keyword_result.semantic_nodes:
                    print(f"     - Keyword '{keyword}': found {len(keyword_result.semantic_nodes)} nodes")
        
        # Convert back to list and limit results
        final_results = list(all_results.values())[:limit]
        print(f"   📊 Keyword search found {len(final_results)} unique nodes total")
        
        if not final_results:
            print(f"   ⚠️ No results found with any search strategy, returning all nodes for debugging...")
            # Last resort: return some nodes for debugging (skip embedding)
            all_nodes_query = SemanticNodeQuery(owner_id=owner_id, limit=limit)
            debug_result = await self.search_semantic_nodes(all_nodes_query, _skip_embedding_search=True)
            return debug_result.semantic_nodes
        
        return final_results
    
    def _extract_search_keywords(self, query: str) -> list[str]:
        """Extract meaningful keywords from search query."""
        import re
        
        # Split by common delimiters and clean
        keywords = re.split(r'[,，\s\-_:：]+', query)
        
        # Filter out common stop words and short words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
            'discussion', 'libraries', 'library', 'about', 'python', 'machine', 'learning',
            '的', '是', '在', '有', '和', '用', '对', '关于', '讨论'
        }
        
        filtered_keywords = []
        for kw in keywords:
            cleaned = kw.strip().lower()
            if len(cleaned) > 2 and cleaned not in stop_words:
                # Keep original case for better matching
                filtered_keywords.append(kw.strip())
        
        return filtered_keywords[:10]  # Limit to 10 most important keywords

    async def find_semantic_nodes_by_episode(self, episode_id: str) -> list[SemanticNode]:
        """Find all semantic nodes discovered from a specific episode."""
        async with self.session_factory() as session:
            node_result = await session.execute(
                select(SemanticNodeTable).where(SemanticNodeTable.discovery_episode_id == episode_id)
            )
            node_rows = node_result.fetchall()

            return [self._row_to_semantic_node(row[0]) for row in node_rows]

    async def find_semantic_nodes_by_linked_episode(self, episode_id: str) -> list[SemanticNode]:
        """Find all semantic nodes that have the episode in their linked_episode_ids."""
        async with self.session_factory() as session:
            # Use JSON search for linked episode IDs
            node_result = await session.execute(
                select(SemanticNodeTable).where(SemanticNodeTable.linked_episode_ids.like(f'%"{episode_id}"%'))
            )
            node_rows = node_result.fetchall()

            # Filter results to ensure exact match (not substring)
            matching_nodes = []
            for row_tuple in node_rows:
                row = row_tuple[0]
                linked_ids = json.loads(row.linked_episode_ids)
                if episode_id in linked_ids:
                    matching_nodes.append(self._row_to_semantic_node(row))

            return matching_nodes

    # === Embedding-based Similarity Search ===

    def _setup_embedding_client(self, config: StorageConfig) -> None:
        """Setup OpenAI client for embedding generation."""
        try:
            # Try to get API key from config or environment
            api_key = getattr(config, 'openai_api_key', None)
            base_url = getattr(config, 'openai_base_url', None)
            embed_model = getattr(config, 'openai_embed_model', None)
            if not api_key:
                import os
                api_key = os.getenv('OPENAI_API_KEY')
                base_url = os.getenv('OPENAI_BASE_URL')
                embed_model = os.getenv('OPENAI_EMB_MODEL')
            
            # Set default embed model if none provided
            if not embed_model:
                embed_model = "text-embedding-3-small"
            
            if api_key or base_url:
                from openai import AsyncOpenAI
                self.openai_client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url
                )
                self.embed_model = embed_model
                print(f"✅ Initialized OpenAI client for embeddings with model: {embed_model}")
            else:
                print("⚠️ No OpenAI API key or base_url provided - semantic embedding search disabled")
                self.openai_client = None
                self.embed_model = None
        except Exception as e:
            print(f"❌ Warning: Could not initialize OpenAI client: {e}")
            self.openai_client = None
            self.embed_model = None

    async def _generate_query_embedding(self, query: str) -> list[float] | None:
        """Generate embedding for query text with caching."""
        if not self.openai_client:
            return None
        
        if not query.strip():
            return None
        
        if not self.embed_model:
            return None
        
        import hashlib
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        if query_hash in self._embedding_cache:
            embedding, timestamp = self._embedding_cache[query_hash]
            if time.time() - timestamp < self._cache_ttl:
                return embedding
        
        try:
            response = await self.openai_client.embeddings.create(
                model=self.embed_model,
                input=query
            )
            embedding = response.data[0].embedding
            self._embedding_cache[query_hash] = (embedding, time.time())
            return embedding
        except Exception as e:
            return None

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return float(dot_product / (norm1 * norm2))
        except Exception:
            return 0.0

    async def _embedding_similarity_search(self, owner_id: str, query_embedding: list[float], limit: int) -> list[SemanticNode]:
        """Perform embedding-based similarity search using pgvector."""
        async with self.session_factory() as session:
            # Use pgvector's native cosine similarity operator (<=>)
            # ORDER BY embedding_vector <=> query_embedding gives us cosine distance (lower is more similar)
            stmt = (
                select(SemanticNodeTable)
                .where(
                    and_(
                        SemanticNodeTable.owner_id == owner_id,
                        SemanticNodeTable.embedding_vector.isnot(None)
                    )
                )
                .order_by(SemanticNodeTable.embedding_vector.op('<=>')(query_embedding))
                .limit(limit)
            )
            
            result = await session.execute(stmt)
            rows = result.scalars().all()
            
            # Convert to SemanticNode objects
            return [self._row_to_semantic_node(row) for row in rows]

    async def store_semantic_node_with_embedding(self, node: SemanticNode, content_for_embedding: str | None = None) -> None:
        """Store semantic node and generate embedding if possible."""
        # Generate embedding if client is available and no embedding exists
        if self.openai_client and not node.embedding_vector and content_for_embedding:
            try:
                embedding_text = content_for_embedding or f"{node.key} {node.value} {node.context}"
                embedding = await self._generate_query_embedding(embedding_text)
                if embedding:
                    # Create new node with embedding
                    from dataclasses import replace
                    node = replace(node, embedding_vector=embedding)
            except Exception as e:
                print(f"Warning: Could not generate embedding for node {node.node_id}: {e}")
        
        # Store the node (with or without embedding)
        await self.store_semantic_node(node)

    # === Semantic Relationship Operations ===

    async def store_semantic_relationship(self, relationship: SemanticRelationship) -> None:
        """Store a semantic relationship."""
        relationship_id = self.validate_id(relationship.relationship_id)

        relationship_row = SemanticRelationshipTable(
            relationship_id=relationship_id,
            source_node_id=relationship.source_node_id,
            target_node_id=relationship.target_node_id,
            relationship_type=relationship.relationship_type.value,
            strength=relationship.strength,
            description=relationship.description,
            created_at=relationship.created_at,
            last_reinforced=relationship.last_reinforced,
            discovery_episode_id=relationship.discovery_episode_id,
        )

        async with self.session_factory() as session:
            session.add(relationship_row)
            await session.commit()

    async def get_semantic_relationship_by_id(self, relationship_id: str) -> SemanticRelationship | None:
        """Retrieve a semantic relationship by its ID."""
        relationship_id = self.validate_id(relationship_id)

        async with self.session_factory() as session:
            relationship_result = await session.execute(
                select(SemanticRelationshipTable).where(SemanticRelationshipTable.relationship_id == relationship_id)
            )
            relationship_row = relationship_result.first()

            if not relationship_row:
                return None

            return self._row_to_semantic_relationship(relationship_row[0])

    async def find_relationships_for_node(self, node_id: str) -> list[tuple[SemanticNode, SemanticRelationship]]:
        """Find all relationships and related nodes for a given semantic node."""
        node_id = self.validate_id(node_id)

        async with self.session_factory() as session:
            # Find relationships where this node is source or target
            relationships_result = await session.execute(
                select(SemanticRelationshipTable).where(
                    or_(
                        SemanticRelationshipTable.source_node_id == node_id,
                        SemanticRelationshipTable.target_node_id == node_id,
                    )
                )
            )
            relationships = relationships_result.fetchall()

            results = []
            for rel_row_tuple in relationships:
                rel_row = rel_row_tuple[0]
                # Determine the related node ID
                related_node_id = (
                    rel_row.target_node_id if rel_row.source_node_id == node_id else rel_row.source_node_id
                )

                # Get the related node
                related_node_result = await session.execute(
                    select(SemanticNodeTable).where(SemanticNodeTable.node_id == related_node_id)
                )
                related_node_row = related_node_result.first()

                if related_node_row:
                    related_node = self._row_to_semantic_node(related_node_row[0])
                    relationship = self._row_to_semantic_relationship(rel_row)
                    results.append((related_node, relationship))

            return results

    async def search_semantic_relationships(self, query: SemanticRelationshipQuery) -> SemanticSearchResult:
        """Search semantic relationships with complex query parameters."""
        start_time = time.time()

        async with self.session_factory() as session:
            stmt = select(SemanticRelationshipTable)

            # Apply filters
            if query.source_node_id:
                stmt = stmt.where(SemanticRelationshipTable.source_node_id == query.source_node_id)

            if query.target_node_id:
                stmt = stmt.where(SemanticRelationshipTable.target_node_id == query.target_node_id)

            if query.involves_node_id:
                stmt = stmt.where(
                    or_(
                        SemanticRelationshipTable.source_node_id == query.involves_node_id,
                        SemanticRelationshipTable.target_node_id == query.involves_node_id,
                    )
                )

            if query.relationship_types:
                stmt = stmt.where(SemanticRelationshipTable.relationship_type.in_(query.relationship_types))

            if query.min_strength:
                stmt = stmt.where(SemanticRelationshipTable.strength >= query.min_strength)

            if query.discovery_episode_id:
                stmt = stmt.where(SemanticRelationshipTable.discovery_episode_id == query.discovery_episode_id)

            # Count total before pagination
            count_stmt = select(func.count(SemanticRelationshipTable.relationship_id)).select_from(stmt.subquery())
            total_count_result = await session.execute(count_stmt)
            total_count = total_count_result.scalar()

            # Apply sorting
            if query.sort_by == "created_at":
                sort_field = SemanticRelationshipTable.created_at
            elif query.sort_by == "last_reinforced":
                sort_field = SemanticRelationshipTable.last_reinforced
            elif query.sort_by == "strength":
                sort_field = SemanticRelationshipTable.strength
            else:
                sort_field = SemanticRelationshipTable.created_at

            if query.sort_order == SortOrder.DESC:
                sort_field = sort_field.desc()

            stmt = stmt.order_by(sort_field)

            # Apply pagination
            stmt = stmt.offset(query.offset).limit(query.limit)

            # Execute query
            relationship_result = await session.execute(stmt)
            relationship_rows = relationship_result.fetchall()
            relationships = [self._row_to_semantic_relationship(row[0]) for row in relationship_rows]

            query_time_ms = (time.time() - start_time) * 1000

            return SemanticSearchResult(
                semantic_relationships=relationships,
                total_relationships=total_count,
                has_more_relationships=total_count > (query.offset + len(relationships)),
                query_time_ms=query_time_ms,
            )

    async def update_semantic_relationship(self, relationship: SemanticRelationship) -> None:
        """Update an existing semantic relationship."""
        relationship_id = self.validate_id(relationship.relationship_id)

        async with self.session_factory() as session:
            existing_result = await session.execute(
                select(SemanticRelationshipTable).where(SemanticRelationshipTable.relationship_id == relationship_id)
            )
            existing_row = existing_result.first()

            if not existing_row:
                raise NotFoundError(f"Semantic relationship with ID '{relationship_id}' not found")

            existing = existing_row[0]
            # Update fields
            existing.source_node_id = relationship.source_node_id
            existing.target_node_id = relationship.target_node_id
            existing.relationship_type = relationship.relationship_type.value
            existing.strength = relationship.strength
            existing.description = relationship.description
            existing.last_reinforced = relationship.last_reinforced
            existing.discovery_episode_id = relationship.discovery_episode_id

            session.add(existing)
            await session.commit()

    async def delete_semantic_relationship(self, relationship_id: str) -> bool:
        """Delete a semantic relationship by ID."""
        relationship_id = self.validate_id(relationship_id)

        async with self.session_factory() as session:
            result = await session.execute(
                delete(SemanticRelationshipTable).where(SemanticRelationshipTable.relationship_id == relationship_id)
            )
            await session.commit()

            return result.rowcount > 0

    # === Bulk Operations ===

    async def get_semantic_nodes_by_ids(self, node_ids: list[str]) -> list[SemanticNode]:
        """Retrieve multiple semantic nodes by their IDs."""
        if not node_ids:
            return []

        validated_ids = [self.validate_id(node_id) for node_id in node_ids]

        async with self.session_factory() as session:
            node_result = await session.execute(
                select(SemanticNodeTable).where(SemanticNodeTable.node_id.in_(validated_ids))
            )
            node_rows = node_result.fetchall()

            return [self._row_to_semantic_node(row[0]) for row in node_rows]

    async def get_all_semantic_nodes_for_owner(self, owner_id: str) -> list[SemanticNode]:
        """Retrieve all semantic nodes for a specific owner."""
        async with self.session_factory() as session:
            node_result = await session.execute(
                select(SemanticNodeTable)
                .where(SemanticNodeTable.owner_id == owner_id)
                .order_by(SemanticNodeTable.created_at.desc())
            )
            node_rows = node_result.fetchall()

            return [self._row_to_semantic_node(row[0]) for row in node_rows]

    # === Statistics and Maintenance ===

    async def get_semantic_statistics(self, owner_id: str) -> dict[str, Any]:
        """Get statistics about semantic memory for an owner."""
        async with self.session_factory() as session:
            # Count nodes
            node_count_result = await session.execute(
                select(func.count(SemanticNodeTable.node_id)).where(SemanticNodeTable.owner_id == owner_id)
            )
            node_count = node_count_result.scalar()

            # Count relationships (involving nodes owned by this user)
            node_ids_subquery = select(SemanticNodeTable.node_id).where(SemanticNodeTable.owner_id == owner_id)
            relationship_count_result = await session.execute(
                select(func.count(SemanticRelationshipTable.relationship_id)).where(
                    or_(
                        SemanticRelationshipTable.source_node_id.in_(node_ids_subquery),
                        SemanticRelationshipTable.target_node_id.in_(node_ids_subquery),
                    )
                )
            )
            relationship_count = relationship_count_result.scalar()

            # Average confidence
            avg_confidence_result = await session.execute(
                select(func.avg(SemanticNodeTable.confidence)).where(SemanticNodeTable.owner_id == owner_id)
            )
            avg_confidence = avg_confidence_result.scalar() or 0.0

            # Version distribution
            version_stats_result = await session.execute(
                select(SemanticNodeTable.version, func.count(SemanticNodeTable.node_id))
                .where(SemanticNodeTable.owner_id == owner_id)
                .group_by(SemanticNodeTable.version)
            )
            version_stats = version_stats_result.fetchall()

            return {
                "node_count": node_count,
                "relationship_count": relationship_count,
                "average_confidence": float(avg_confidence),
                "version_distribution": {version: count for version, count in version_stats},
                "owner_id": owner_id,
            }

    async def cleanup_orphaned_relationships(self) -> int:
        """Clean up relationships that reference non-existent nodes."""
        async with self.session_factory() as session:
            # Find relationships with non-existent source nodes
            orphaned_source_result = await session.execute(
                select(SemanticRelationshipTable.relationship_id)
                .outerjoin(SemanticNodeTable, SemanticNodeTable.node_id == SemanticRelationshipTable.source_node_id)
                .where(SemanticNodeTable.node_id.is_(None))
            )
            orphaned_source_rows = orphaned_source_result.fetchall()
            orphaned_source = [row[0] for row in orphaned_source_rows]

            # Find relationships with non-existent target nodes
            orphaned_target_result = await session.execute(
                select(SemanticRelationshipTable.relationship_id)
                .outerjoin(SemanticNodeTable, SemanticNodeTable.node_id == SemanticRelationshipTable.target_node_id)
                .where(SemanticNodeTable.node_id.is_(None))
            )
            orphaned_target_rows = orphaned_target_result.fetchall()
            orphaned_target = [row[0] for row in orphaned_target_rows]

            # Combine and deduplicate
            orphaned_ids = list(set(orphaned_source + orphaned_target))

            if orphaned_ids:
                result = await session.execute(
                    delete(SemanticRelationshipTable).where(SemanticRelationshipTable.relationship_id.in_(orphaned_ids))
                )
                await session.commit()
                return result.rowcount

            return 0

    # === Helper Methods ===

    def _row_to_semantic_node(self, row: SemanticNodeTable) -> SemanticNode:
        """Convert database row to SemanticNode."""
        return SemanticNode(
            node_id=row.node_id,
            owner_id=row.owner_id,
            key=row.key,
            value=row.value,
            context=row.context,
            confidence=row.confidence,
            version=row.version,
            evolution_history=json.loads(row.evolution_history),
            created_at=row.created_at,
            last_updated=row.last_updated,
            last_accessed=row.last_accessed,
            discovery_episode_id=row.discovery_episode_id,
            discovery_method=row.discovery_method,
            linked_episode_ids=json.loads(row.linked_episode_ids),
            evolution_episode_ids=json.loads(row.evolution_episode_ids),
            search_keywords=json.loads(row.search_keywords),
            embedding_vector=row.embedding_vector if row.embedding_vector is not None else None,
            access_count=row.access_count,
            relevance_score=row.relevance_score,
            importance_score=row.importance_score,
        )

    def _row_to_semantic_relationship(self, row: SemanticRelationshipTable) -> SemanticRelationship:
        """Convert database row to SemanticRelationship."""
        return SemanticRelationship(
            relationship_id=row.relationship_id,
            source_node_id=row.source_node_id,
            target_node_id=row.target_node_id,
            relationship_type=RelationshipType(row.relationship_type),
            strength=row.strength,
            description=row.description,
            created_at=row.created_at,
            last_reinforced=row.last_reinforced,
            discovery_episode_id=row.discovery_episode_id,
        )