"""
Episode builders for transforming raw user data into unified episodes.

This module provides the abstract base class and interfaces for converting
different types of user experiences into standardized Episode objects.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from ..llm.protocol import LLMProvider
from .data_types import DataType, RawEventData, TypedEventData
from .episode import Episode, EpisodeLevel, EpisodeMetadata, EpisodeType


class EpisodeBuilder(ABC):
    """
    Abstract base class for episode builders.

    Each data type should have its own specialized builder that understands
    how to extract meaningful episodic memories from that type of data.
    """

    def __init__(self, llm_provider: LLMProvider | None = None):
        """
        Initialize the episode builder.

        Args:
            llm_provider: Optional LLM provider for content generation
        """
        self.llm_provider = llm_provider

    @property
    @abstractmethod
    def supported_data_type(self) -> DataType:
        """The data type this builder supports."""
        pass

    @property
    @abstractmethod
    def default_episode_type(self) -> EpisodeType:
        """The default episode type this builder produces."""
        pass

    def can_build(self, data: RawEventData) -> bool:
        """Check if this builder can process the given data."""
        return data.data_type == self.supported_data_type

    def build_episode(self, data: RawEventData, for_owner: str) -> Episode:
        """
        Build an episode from raw event data for a specific owner.

        Args:
            data: The raw event data to process
            for_owner: The ID of the entity who owns this episode

        This is the main entry point that orchestrates the episode creation process.
        """
        if not self.can_build(data):
            raise ValueError(f"Builder for {self.supported_data_type} cannot process {data.data_type} data")

        # Pre-process the data
        processed_data = self._preprocess_data(data)

        # Extract core content
        title, content, summary = self._extract_content(processed_data)

        # Generate metadata
        metadata = self._generate_metadata(processed_data)

        # Extract structured data
        structured_data = self._extract_structured_data(processed_data)

        # Determine episode level
        level = self._determine_episode_level(processed_data)

        # Generate search keywords
        keywords = self._generate_keywords(title, content, summary)

        # Create the episode
        episode = Episode(
            owner_id=for_owner,
            episode_type=self.default_episode_type,
            level=level,
            title=title,
            content=content,
            summary=summary,
            temporal_info=data.temporal_info,
            metadata=metadata,
            structured_data=structured_data,
            search_keywords=keywords,
        )

        # Post-process the episode
        episode = self._postprocess_episode(episode, processed_data)

        return episode

    def _preprocess_data(self, data: RawEventData) -> TypedEventData:
        """
        Preprocess raw data into typed data.

        Override this method to add type-specific preprocessing.
        """
        from .data_types import create_typed_data

        return create_typed_data(data)

    @abstractmethod
    def _extract_content(self, data: TypedEventData) -> tuple[str, str, str]:
        """
        Extract title, content, and summary from the data.

        Returns:
            tuple: (title, content, summary)
        """
        pass

    def _generate_metadata(self, data: TypedEventData) -> EpisodeMetadata:
        """
        Generate episode metadata from the data.

        Override this method to add type-specific metadata extraction.
        """
        return EpisodeMetadata(
            source_data_ids=[data.data_id],
            source_types={data.raw_data.data_type},
            processing_timestamp=datetime.now(),
            custom_fields=data.raw_data.metadata.copy(),
        )

    def _extract_structured_data(self, data: TypedEventData) -> dict[str, Any]:
        """
        Extract structured data specific to this data type.

        Override this method to add type-specific structured data extraction.
        """
        return {
            "original_source": data.raw_data.source,
            "data_type": data.raw_data.data_type.value,
            "processing_metadata": data.raw_data.metadata,
        }

    def _determine_episode_level(self, data: TypedEventData) -> EpisodeLevel:
        """
        Determine the appropriate episode level for this data.

        Override this method to implement type-specific level determination logic.
        """
        return EpisodeLevel.ATOMIC

    def _generate_keywords(self, title: str, content: str, summary: str) -> list[str]:
        """
        Generate search keywords from the episode content.

        This is a simple implementation - override for more sophisticated keyword extraction.
        """
        import re

        # Combine all text
        all_text = f"{title} {content} {summary}".lower()

        # Simple keyword extraction - get words longer than 3 characters
        words = re.findall(r"\b\w{4,}\b", all_text)

        # Remove duplicates and return top keywords
        unique_words = list(set(words))
        return unique_words[:20]  # Limit to top 20 keywords

    def _postprocess_episode(self, episode: Episode, data: TypedEventData) -> Episode:
        """
        Post-process the episode after creation.

        Override this method to add type-specific post-processing.
        """
        return episode


class BatchEpisodeBuilder:
    """
    Utility class for building episodes from multiple raw data items.

    This can be useful for creating compound episodes from related data.
    """

    def __init__(self, builders: dict[DataType, EpisodeBuilder]):
        """
        Initialize with a mapping of data types to their builders.

        Args:
            builders: Dictionary mapping data types to their respective builders
        """
        self.builders = builders

    def build_episodes(self, data_items: list[RawEventData], for_owner: str) -> list[Episode]:
        """Build episodes from a list of raw data items."""
        episodes = []

        for data in data_items:
            builder = self.builders.get(data.data_type)
            if builder:
                try:
                    episode = builder.build_episode(data, for_owner)
                    episodes.append(episode)
                except Exception as e:
                    print(f"Failed to build episode from {data.data_id}: {e}")
            else:
                print(f"No builder available for data type: {data.data_type}")

        return episodes

    def build_compound_episode(
        self, data_items: list[RawEventData], for_owner: str, title: str, compound_type: EpisodeType = EpisodeType.MIXED
    ) -> Episode:
        """
        Build a compound episode from multiple related data items.

        This creates a higher-level episode that synthesizes multiple
        raw data sources into a single narrative.
        """
        if not data_items:
            raise ValueError("Cannot create compound episode from empty data list")

        # Use the first item's temporal info as base, owner is specified
        base_data = data_items[0]

        # Build individual episodes first
        individual_episodes = self.build_episodes(data_items, for_owner)

        # Combine content
        combined_content = "\n\n".join([ep.content for ep in individual_episodes])
        combined_summary = " | ".join([ep.summary for ep in individual_episodes])

        # Combine metadata
        all_source_ids = []
        all_source_types = set()
        all_entities = []
        all_topics = []
        all_emotions = []
        all_keywords = []

        for episode in individual_episodes:
            all_source_ids.extend(episode.metadata.source_data_ids)
            all_source_types.update(episode.metadata.source_types)
            all_entities.extend(episode.metadata.entities)
            all_topics.extend(episode.metadata.topics)
            all_emotions.extend(episode.metadata.emotions)
            all_keywords.extend(episode.search_keywords)

        # Create compound metadata
        compound_metadata = EpisodeMetadata(
            source_data_ids=all_source_ids,
            source_types=all_source_types,
            entities=list(set(all_entities)),
            topics=list(set(all_topics)),
            emotions=list(set(all_emotions)),
            related_episode_ids=[ep.episode_id for ep in individual_episodes],
        )

        # Create compound episode
        compound_episode = Episode(
            owner_id=for_owner,
            episode_type=compound_type,
            level=EpisodeLevel.COMPOUND,
            title=title,
            content=combined_content,
            summary=combined_summary,
            temporal_info=base_data.temporal_info,
            metadata=compound_metadata,
            search_keywords=list(set(all_keywords)),
        )

        return compound_episode


class EpisodeBuilderRegistry:
    """Registry for managing episode builders."""

    def __init__(self):
        self._builders: dict[DataType, EpisodeBuilder] = {}

    def register(self, builder: EpisodeBuilder) -> None:
        """Register a builder for its supported data type."""
        data_type = builder.supported_data_type
        if data_type in self._builders:
            print(f"Warning: Overriding existing builder for {data_type}")
        self._builders[data_type] = builder

    def get_builder(self, data_type: DataType) -> EpisodeBuilder | None:
        """Get the builder for a specific data type."""
        return self._builders.get(data_type)

    def get_all_builders(self) -> dict[DataType, EpisodeBuilder]:
        """Get all registered builders."""
        return self._builders.copy()

    def can_process(self, data_type: DataType) -> bool:
        """Check if we have a builder for the given data type."""
        return data_type in self._builders

    def build_episode(self, data: RawEventData, for_owner: str) -> Episode | None:
        """Build an episode using the appropriate registered builder."""
        builder = self.get_builder(data.data_type)
        if builder:
            return builder.build_episode(data, for_owner)
        return None


# Global registry instance
default_registry = EpisodeBuilderRegistry()
