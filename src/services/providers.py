"""Factory helpers that wire the default infrastructure components."""

from __future__ import annotations

from typing import Optional

from ..config import MemoryConfig
from ..domain.interfaces import (
    BoundaryDetector,
    EpisodeGenerator,
    EpisodeRepository,
    LexicalIndex,
    SemanticGenerator,
    SemanticRepository,
    VectorIndex,
)
from ..core.boundary_detector import BoundaryDetector as BoundaryDetectorImpl
from ..generation.episode_generator import EpisodeGenerator as EpisodeGeneratorImpl
from ..generation.semantic_generator import SemanticGenerator as SemanticGeneratorImpl
from ..infrastructure.indices import (
    Bm25Index,
    ChromaVectorIndex,
    InMemoryLexicalIndex,
    InMemoryVectorIndex,
)
from ..infrastructure.repositories import (
    EpisodeStorageRepository,
    InMemoryEpisodeRepository,
    InMemorySemanticRepository,
    SemanticStorageRepository,
)
from ..search.bm25_search import BM25Search
from ..search.chroma_search import ChromaSearchEngine
from ..storage.episode_storage import EpisodeStorage
from ..storage.semantic_storage import SemanticStorage
from ..utils import EmbeddingClient, LLMClient


class DefaultProviders:
    """Factory collection for building default service graph."""

    def __init__(
        self,
        config: MemoryConfig,
        llm_client: Optional[LLMClient] = None,
        embedding_client: Optional[EmbeddingClient] = None,
    ) -> None:
        self.config = config
        self.llm_client = llm_client or LLMClient(
            api_key=config.openai_api_key,
            model=config.llm_model,
        )
        self.embedding_client = embedding_client or EmbeddingClient(
            api_key=config.openai_api_key,
            model=config.embedding_model,
        )
        self._episode_repo: Optional[EpisodeRepository] = None
        self._semantic_repo: Optional[SemanticRepository] = None
        self._bm25: Optional[LexicalIndex] = None
        self._chroma_engine: Optional[ChromaSearchEngine] = None
        self._vector_index: Optional[VectorIndex] = None
        self._boundary: Optional[BoundaryDetector] = None
        self._episode_gen: Optional[EpisodeGenerator] = None
        self._semantic_gen: Optional[SemanticGenerator] = None
        self._storage_backend = getattr(config, "storage_backend", "filesystem")
        self._vector_backend = getattr(config, "vector_index_backend", "chroma")
        self._lexical_backend = getattr(config, "lexical_index_backend", "bm25")

    # Storage -----------------------------------------------------------------
    def episode_repository(self) -> EpisodeRepository:
        if self._episode_repo is None:
            if self._storage_backend == "memory":
                self._episode_repo = InMemoryEpisodeRepository()
            else:
                self._episode_repo = EpisodeStorageRepository(EpisodeStorage(self.config.storage_path))
        return self._episode_repo

    def semantic_repository(self) -> SemanticRepository:
        if self._semantic_repo is None:
            if self._storage_backend == "memory":
                self._semantic_repo = InMemorySemanticRepository()
            else:
                self._semantic_repo = SemanticStorageRepository(SemanticStorage(self.config.storage_path))
        return self._semantic_repo

    # Indices -----------------------------------------------------------------
    def lexical_index(self) -> LexicalIndex:
        if self._bm25 is None:
            if self._lexical_backend == "memory":
                self._bm25 = InMemoryLexicalIndex()
            else:
                self._bm25 = Bm25Index(BM25Search(language=self.config.language))
        return self._bm25

    def vector_index(self) -> VectorIndex:
        if self._vector_index is None:
            if self._vector_backend == "memory":
                self._vector_index = InMemoryVectorIndex()
            else:
                self._chroma_engine = ChromaSearchEngine(self.embedding_client, self.config)
                self._vector_index = ChromaVectorIndex(self._chroma_engine)
        return self._vector_index  # type: ignore[return-value]

    # Generators ---------------------------------------------------------------
    def boundary_detector(self) -> BoundaryDetector:
        if self._boundary is None:
            self._boundary = BoundaryDetectorImpl(self.llm_client, self.config)
        return self._boundary

    def episode_generator(self) -> EpisodeGenerator:
        if self._episode_gen is None:
            self._episode_gen = EpisodeGeneratorImpl(self.llm_client, self.config)
        return self._episode_gen

    def semantic_generator(self) -> SemanticGenerator:
        if self._semantic_gen is None:
            vector_search = getattr(self.vector_index(), "_backend", None)
            self._semantic_gen = SemanticGeneratorImpl(
                self.llm_client,
                self.embedding_client,
                self.config,
                vector_search=vector_search,
            )
        return self._semantic_gen
