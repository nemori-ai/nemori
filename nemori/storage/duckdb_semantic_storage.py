"""
DuckDB implementation for semantic memory storage.

This module provides semantic memory storage using DuckDB with full-text search
and proper semantic node/relationship management.
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sqlmodel import Session, SQLModel, and_, create_engine, delete, func, or_, select

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


class DuckDBSemanticMemoryRepository(SemanticMemoryRepository, BaseSQLRepository):
    """DuckDB implementation of semantic memory repository."""

    def __init__(self, config: StorageConfig):
        super().__init__(config)
        self.db_path = config.connection_string or "nemori_semantic.duckdb"
        self.engine = None
        self._initialized = False
        # Initialize OpenAI client for embeddings
        self.openai_client = None
        self._setup_embedding_client(config)

    async def initialize(self) -> None:
        """Initialize the DuckDB semantic memory storage."""
        # Create database directory if needed
        db_path = Path(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create engine
        self.engine = create_engine(f"duckdb:///{self.db_path}", echo=False)

        # Initialize base class
        BaseSQLRepository.__init__(self, self.engine)

        # Create semantic memory tables
        SQLModel.metadata.create_all(
            self.engine, tables=[SemanticNodeTable.__table__, SemanticRelationshipTable.__table__]
        )

        self._initialized = True

    async def close(self) -> None:
        """Close the database connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
        self._initialized = False

    async def health_check(self) -> bool:
        """Check if storage is healthy."""
        if not self._initialized or not self.engine:
            return False
        try:
            with Session(self.engine) as session:
                session.exec(select(func.count(SemanticNodeTable.node_id))).one()
                return True
        except Exception:
            return False

    async def get_stats(self) -> StorageStats:
        """Get semantic memory statistics."""
        stats = StorageStats()

        with Session(self.engine) as session:
            # Count semantic nodes and relationships
            node_count = session.exec(select(func.count(SemanticNodeTable.node_id))).one()
            relationship_count = session.exec(select(func.count(SemanticRelationshipTable.relationship_id))).one()

            # Use available stats fields appropriately
            stats.total_raw_data = node_count  # Repurpose for semantic node count
            stats.total_episodes = relationship_count  # Repurpose for relationship count

            # Storage size
            db_path = Path(self.db_path)
            if db_path.exists():
                stats.storage_size_mb = db_path.stat().st_size / (1024 * 1024)

        return stats

    async def backup(self, destination: str) -> bool:
        """Create a backup by copying the database file."""
        try:
            import shutil

            # Dispose engine to ensure data is written
            if self.engine:
                self.engine.dispose()

            # Copy the database file
            shutil.copy2(self.db_path, destination)

            # Recreate engine
            self.engine = create_engine(f"duckdb:///{self.db_path}")
            return True
        except Exception:
            return False

    async def restore(self, source: str) -> bool:
        """Restore from backup by copying the database file."""
        try:
            import shutil

            # Dispose current engine
            if self.engine:
                self.engine.dispose()

            # Copy backup file
            shutil.copy2(source, self.db_path)

            # Recreate engine and initialize
            await self.initialize()
            return True
        except Exception:
            return False

    # === Semantic Node Operations ===

    async def store_semantic_node(self, node: SemanticNode) -> None:
        """Store a semantic node."""
        node_id = self.validate_id(node.node_id)

        # Check for duplicate key
        existing = await self.find_semantic_node_by_key(node.owner_id, node.key)
        if existing and existing.node_id != node.node_id:
            raise DuplicateKeyError(f"Node with key '{node.key}' already exists for owner '{node.owner_id}'")

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
            embedding_vector=json.dumps(node.embedding_vector, ensure_ascii=False) if node.embedding_vector else None,
            access_count=node.access_count,
            relevance_score=node.relevance_score,
            importance_score=node.importance_score,
        )

        with Session(self.engine) as session:
            session.add(node_row)
            session.commit()

    async def get_semantic_node_by_id(self, node_id: str) -> SemanticNode | None:
        """Retrieve a semantic node by its ID."""
        node_id = self.validate_id(node_id)

        with Session(self.engine) as session:
            node_row = session.exec(select(SemanticNodeTable).where(SemanticNodeTable.node_id == node_id)).first()

            if not node_row:
                return None

            return self._row_to_semantic_node(node_row)

    async def find_semantic_node_by_key(self, owner_id: str, key: str) -> SemanticNode | None:
        """Find semantic node by owner and key combination."""
        with Session(self.engine) as session:
            node_row = session.exec(
                select(SemanticNodeTable).where(
                    and_(SemanticNodeTable.owner_id == owner_id, SemanticNodeTable.key == key)
                )
            ).first()

            if not node_row:
                return None

            return self._row_to_semantic_node(node_row)

    async def update_semantic_node(self, node: SemanticNode) -> None:
        """Update an existing semantic node."""
        node_id = self.validate_id(node.node_id)

        with Session(self.engine) as session:
            # Check if node exists
            existing = session.exec(select(SemanticNodeTable).where(SemanticNodeTable.node_id == node_id)).first()

            if not existing:
                raise NotFoundError(f"Semantic node with ID '{node_id}' not found")

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
            existing.embedding_vector = (
                json.dumps(node.embedding_vector, ensure_ascii=False) if node.embedding_vector else None
            )
            existing.access_count = node.access_count
            existing.relevance_score = node.relevance_score
            existing.importance_score = node.importance_score

            session.add(existing)
            session.commit()

    async def delete_semantic_node(self, node_id: str) -> bool:
        """Delete a semantic node by ID."""
        node_id = self.validate_id(node_id)

        with Session(self.engine) as session:
            # Delete related relationships first
            session.exec(
                delete(SemanticRelationshipTable).where(
                    or_(
                        SemanticRelationshipTable.source_node_id == node_id,
                        SemanticRelationshipTable.target_node_id == node_id,
                    )
                )
            )

            # Delete the node
            result = session.exec(delete(SemanticNodeTable).where(SemanticNodeTable.node_id == node_id))
            session.commit()

            return result.rowcount > 0

    async def search_semantic_nodes(self, query: SemanticNodeQuery) -> SemanticSearchResult:
        """Search semantic nodes with complex query parameters."""
        start_time = time.time()

        with Session(self.engine) as session:
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

            # Count total before pagination
            count_stmt = select(func.count(SemanticNodeTable.node_id)).select_from(stmt.subquery())
            total_count = session.exec(count_stmt).one()

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
            node_rows = session.exec(stmt).all()
            nodes = [self._row_to_semantic_node(row) for row in node_rows]

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
        print(f"🔍 Starting semantic similarity search for owner '{owner_id}' with query: '{query[:100]}...'")
        
        # First, check if we have any semantic nodes at all
        with Session(self.engine) as session:
            total_nodes = session.exec(
                select(func.count(SemanticNodeTable.node_id)).where(SemanticNodeTable.owner_id == owner_id)
            ).one()
            print(f"   📊 Total semantic nodes for owner '{owner_id}': {total_nodes}")
            
            if total_nodes == 0:
                print(f"   ❌ No semantic nodes found for owner '{owner_id}' - returning empty list")
                return []
        
        try:
            # First, try embedding-based similarity search
            print(f"   🧠 Attempting embedding generation for query...")
            query_embedding = await self._generate_query_embedding(query)
            
            if query_embedding:
                print(f"   ✅ Generated embedding with {len(query_embedding)} dimensions")
                return await self._embedding_similarity_search(owner_id, query_embedding, limit)
            else:
                print(f"   ⚠️ Embedding generation failed, falling back to enhanced text search")
                # Enhanced fallback strategy with multiple search approaches
                return await self._enhanced_text_search_fallback(owner_id, query, limit)
                
        except Exception as e:
            print(f"   ❌ Error in embedding similarity search, falling back to enhanced text search: {e}")
            # Enhanced fallback strategy
            return await self._enhanced_text_search_fallback(owner_id, query, limit)
    
    async def _enhanced_text_search_fallback(self, owner_id: str, query: str, limit: int) -> list[SemanticNode]:
        """Enhanced text search fallback with multiple strategies."""
        print(f"   📝 Starting enhanced text search fallback...")
        
        # Strategy 1: Try direct text search first
        search_query = SemanticNodeQuery(
            owner_id=owner_id, text_search=query, limit=limit, sort_by="confidence", sort_order=SortOrder.DESC
        )
        result = await self.search_semantic_nodes(search_query)
        
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
                keyword_result = await self.search_semantic_nodes(keyword_query)
                
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
            # Last resort: return some nodes for debugging
            all_nodes_query = SemanticNodeQuery(owner_id=owner_id, limit=limit)
            debug_result = await self.search_semantic_nodes(all_nodes_query)
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
        with Session(self.engine) as session:
            node_rows = session.exec(
                select(SemanticNodeTable).where(SemanticNodeTable.discovery_episode_id == episode_id)
            ).all()

            return [self._row_to_semantic_node(row) for row in node_rows]

    async def find_semantic_nodes_by_linked_episode(self, episode_id: str) -> list[SemanticNode]:
        """Find all semantic nodes that have the episode in their linked_episode_ids."""
        with Session(self.engine) as session:
            # Use JSON search for linked episode IDs
            node_rows = session.exec(
                select(SemanticNodeTable).where(SemanticNodeTable.linked_episode_ids.like(f'%"{episode_id}"%'))
            ).all()

            # Filter results to ensure exact match (not substring)
            matching_nodes = []
            for row in node_rows:
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
        """Generate embedding for query text."""
        if not self.openai_client:
            print(f"      ❌ No OpenAI client available for embedding generation")
            return None
        
        if not query.strip():
            print(f"      ❌ Empty query provided for embedding generation")
            return None
        
        if not self.embed_model:
            print(f"      ❌ No embed_model configured for embedding generation")
            return None
        
        try:
            print(f"      🔄 Generating embedding using model '{self.embed_model}'...")
            response = await self.openai_client.embeddings.create(
                model=self.embed_model,
                input=query
            )
            embedding = response.data[0].embedding
            print(f"      ✅ Successfully generated embedding with {len(embedding)} dimensions")
            return embedding
        except Exception as e:
            print(f"      ❌ Error generating query embedding: {e}")
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
        """Perform embedding-based similarity search."""
        with Session(self.engine) as session:
            # Get all nodes for the owner
            node_rows = session.exec(
                select(SemanticNodeTable).where(SemanticNodeTable.owner_id == owner_id)
            ).all()
            
            if not node_rows:
                return []
            
            # Calculate similarities for nodes with embeddings
            similarities = []
            for node_row in node_rows:
                if node_row.embedding_vector:
                    try:
                        node_embedding = json.loads(node_row.embedding_vector)
                        similarity = self._cosine_similarity(query_embedding, node_embedding)
                        similarities.append((node_row, similarity))
                    except Exception as e:
                        print(f"Error processing embedding for node {node_row.node_id}: {e}")
                        continue
            
            # Sort by similarity (descending) and take top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_similarities = similarities[:limit]
            
            # Convert to SemanticNode objects
            return [self._row_to_semantic_node(row) for row, _ in top_similarities]

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

        with Session(self.engine) as session:
            session.add(relationship_row)
            session.commit()

    async def get_semantic_relationship_by_id(self, relationship_id: str) -> SemanticRelationship | None:
        """Retrieve a semantic relationship by its ID."""
        relationship_id = self.validate_id(relationship_id)

        with Session(self.engine) as session:
            relationship_row = session.exec(
                select(SemanticRelationshipTable).where(SemanticRelationshipTable.relationship_id == relationship_id)
            ).first()

            if not relationship_row:
                return None

            return self._row_to_semantic_relationship(relationship_row)

    async def find_relationships_for_node(self, node_id: str) -> list[tuple[SemanticNode, SemanticRelationship]]:
        """Find all relationships and related nodes for a given semantic node."""
        node_id = self.validate_id(node_id)

        with Session(self.engine) as session:
            # Find relationships where this node is source or target
            relationships = session.exec(
                select(SemanticRelationshipTable).where(
                    or_(
                        SemanticRelationshipTable.source_node_id == node_id,
                        SemanticRelationshipTable.target_node_id == node_id,
                    )
                )
            ).all()

            results = []
            for rel_row in relationships:
                # Determine the related node ID
                related_node_id = (
                    rel_row.target_node_id if rel_row.source_node_id == node_id else rel_row.source_node_id
                )

                # Get the related node
                related_node_row = session.exec(
                    select(SemanticNodeTable).where(SemanticNodeTable.node_id == related_node_id)
                ).first()

                if related_node_row:
                    related_node = self._row_to_semantic_node(related_node_row)
                    relationship = self._row_to_semantic_relationship(rel_row)
                    results.append((related_node, relationship))

            return results

    async def search_semantic_relationships(self, query: SemanticRelationshipQuery) -> SemanticSearchResult:
        """Search semantic relationships with complex query parameters."""
        start_time = time.time()

        with Session(self.engine) as session:
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
            total_count = session.exec(count_stmt).one()

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
            relationship_rows = session.exec(stmt).all()
            relationships = [self._row_to_semantic_relationship(row) for row in relationship_rows]

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

        with Session(self.engine) as session:
            existing = session.exec(
                select(SemanticRelationshipTable).where(SemanticRelationshipTable.relationship_id == relationship_id)
            ).first()

            if not existing:
                raise NotFoundError(f"Semantic relationship with ID '{relationship_id}' not found")

            # Update fields
            existing.source_node_id = relationship.source_node_id
            existing.target_node_id = relationship.target_node_id
            existing.relationship_type = relationship.relationship_type.value
            existing.strength = relationship.strength
            existing.description = relationship.description
            existing.last_reinforced = relationship.last_reinforced
            existing.discovery_episode_id = relationship.discovery_episode_id

            session.add(existing)
            session.commit()

    async def delete_semantic_relationship(self, relationship_id: str) -> bool:
        """Delete a semantic relationship by ID."""
        relationship_id = self.validate_id(relationship_id)

        with Session(self.engine) as session:
            result = session.exec(
                delete(SemanticRelationshipTable).where(SemanticRelationshipTable.relationship_id == relationship_id)
            )
            session.commit()

            return result.rowcount > 0

    # === Bulk Operations ===

    async def get_semantic_nodes_by_ids(self, node_ids: list[str]) -> list[SemanticNode]:
        """Retrieve multiple semantic nodes by their IDs."""
        if not node_ids:
            return []

        validated_ids = [self.validate_id(node_id) for node_id in node_ids]

        with Session(self.engine) as session:
            node_rows = session.exec(
                select(SemanticNodeTable).where(SemanticNodeTable.node_id.in_(validated_ids))
            ).all()

            return [self._row_to_semantic_node(row) for row in node_rows]

    async def get_all_semantic_nodes_for_owner(self, owner_id: str) -> list[SemanticNode]:
        """Retrieve all semantic nodes for a specific owner."""
        with Session(self.engine) as session:
            node_rows = session.exec(
                select(SemanticNodeTable)
                .where(SemanticNodeTable.owner_id == owner_id)
                .order_by(SemanticNodeTable.created_at.desc())
            ).all()

            return [self._row_to_semantic_node(row) for row in node_rows]

    # === Statistics and Maintenance ===

    async def get_semantic_statistics(self, owner_id: str) -> dict[str, Any]:
        """Get statistics about semantic memory for an owner."""
        with Session(self.engine) as session:
            # Count nodes
            node_count = session.exec(
                select(func.count(SemanticNodeTable.node_id)).where(SemanticNodeTable.owner_id == owner_id)
            ).one()

            # Count relationships (involving nodes owned by this user)
            node_ids_subquery = select(SemanticNodeTable.node_id).where(SemanticNodeTable.owner_id == owner_id)
            relationship_count = session.exec(
                select(func.count(SemanticRelationshipTable.relationship_id)).where(
                    or_(
                        SemanticRelationshipTable.source_node_id.in_(node_ids_subquery),
                        SemanticRelationshipTable.target_node_id.in_(node_ids_subquery),
                    )
                )
            ).one()

            # Average confidence
            avg_confidence = (
                session.exec(
                    select(func.avg(SemanticNodeTable.confidence)).where(SemanticNodeTable.owner_id == owner_id)
                ).one()
                or 0.0
            )

            # Version distribution
            version_stats = session.exec(
                select(SemanticNodeTable.version, func.count(SemanticNodeTable.node_id))
                .where(SemanticNodeTable.owner_id == owner_id)
                .group_by(SemanticNodeTable.version)
            ).all()

            return {
                "node_count": node_count,
                "relationship_count": relationship_count,
                "average_confidence": float(avg_confidence),
                "version_distribution": {version: count for version, count in version_stats},
                "owner_id": owner_id,
            }

    async def cleanup_orphaned_relationships(self) -> int:
        """Clean up relationships that reference non-existent nodes."""
        with Session(self.engine) as session:
            # Find relationships with non-existent source nodes
            orphaned_source = session.exec(
                select(SemanticRelationshipTable.relationship_id)
                .outerjoin(SemanticNodeTable, SemanticNodeTable.node_id == SemanticRelationshipTable.source_node_id)
                .where(SemanticNodeTable.node_id.is_(None))
            ).all()

            # Find relationships with non-existent target nodes
            orphaned_target = session.exec(
                select(SemanticRelationshipTable.relationship_id)
                .outerjoin(SemanticNodeTable, SemanticNodeTable.node_id == SemanticRelationshipTable.target_node_id)
                .where(SemanticNodeTable.node_id.is_(None))
            ).all()

            # Combine and deduplicate
            orphaned_ids = list(set(orphaned_source + orphaned_target))

            if orphaned_ids:
                result = session.exec(
                    delete(SemanticRelationshipTable).where(SemanticRelationshipTable.relationship_id.in_(orphaned_ids))
                )
                session.commit()
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
            embedding_vector=json.loads(row.embedding_vector) if row.embedding_vector else None,
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
