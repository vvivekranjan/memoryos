from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Protocol,
)
from uuid import UUID

import numpy as np

from memory.models import (
    BaseMemory,
    SessionScope,
)

from retrieval.engine import RetrievalCandidate

@dataclass(slots=True)
class VectorSearchResult:
    """
    Vector retrieval result.
    """

    memory_id: UUID
    score: float
    metadata: dict[str, Any]


@dataclass(slots=True)
class GraphSearchResult:
    """
    Graph traversal retrieval result.
    """

    memory_id: UUID
    score: float
    traversal_depth: int
    path: list[str]


@dataclass(slots=True)
class CacheResult:
    """
    Cache retrieval wrapper.
    """

    hit: bool
    value: Any | None = None


class BaseRetriever(ABC):
    @abstractmethod
    async def retrieve(
        self,
        query_embedding: List[float],
        query_text: str,
        k: int,
        agent_id: str,
        session_scope: Optional[SessionScope] = None,
    ) -> List[RetrievalCandidate]:
        """Must return candidates with score + trace fragment."""
        ...


class VectorStore(ABC):
    """
    Vector index abstraction.

    Examples:
    - FAISS
    - HNSW
    - ScaNN
    """

    @abstractmethod
    def add(
        self,
        *,
        memory_id: UUID,
        embedding: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        """
        Inserts vector into index.
        """

    @abstractmethod
    def search(
        self,
        *,
        embedding: np.ndarray,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """
        Vector similarity search.
        """

    @abstractmethod
    def delete(
        self,
        *,
        memory_id: UUID,
    ) -> None:
        """
        Removes vector from index.
        """

    @abstractmethod
    def persist(self) -> None:
        """
        Persists vector index.
        """

    @abstractmethod
    def count(self) -> int:
        """
        Returns vector count.
        """


class GraphStoreProtocol(Protocol):
    async def bfs_traversal_async(
        self,
        *,
        seeds: list[str],
        max_hops: int = 2,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        ...


class DocStoreProtocol(Protocol):
    async def get_memories_by_ids(
        self,
        *,
        agent_id: str,
        memory_ids: list[UUID],
    ) -> list[BaseMemory]:
        ...


class MemoryStoreProtocol(Protocol):
    async def get_memories_by_ids(
        self,
        *,
        agent_id: str,
        memory_ids: list[UUID],
    ) -> list[BaseMemory]:
        ...


class CacheStore(ABC):
    """
    Cache abstraction.

    Examples:
    - in-memory LRU
    - Redis (future)
    """

    @abstractmethod
    def get(
        self,
        key: str,
    ) -> CacheResult:
        """
        Retrieves cached value.
        """

    @abstractmethod
    def set(
        self,
        *,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> None:
        """
        Stores cache entry.
        """

    @abstractmethod
    def invalidate(
        self,
        key: str,
    ) -> None:
        """
        Invalidates cache key.
        """


class Embedder(ABC):
    """
    Embedding model abstraction.
    """

    @abstractmethod
    async def embed_text(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Generates text embedding.
        """

    @abstractmethod
    async def embed_batch(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """
        Batch embedding generation.
        """

    @abstractmethod
    def dimension(self) -> int:
        """
        Returns embedding dimension.
        """


class EventBus(Protocol):
    """
    Async subsystem event bus.
    """

    async def publish(
        self,
        *,
        topic: str,
        payload: dict,
    ) -> None:
        ...

    async def subscribe(
        self,
        *,
        topic: str,
        handler,
    ) -> None:
        ...


class HealthCheckable(Protocol):
    """
    Health-aware subsystem.
    """

    async def health_check(
        self,
    ) -> dict[str, Any]:
        ...


class Schedulable(Protocol):
    """
    Background schedulable task.
    """

    async def run(self) -> None:
        ...


class DocStore(Protocol):
    async def get_memories_by_ids(
        self,
        *,
        agent_id: str,
        memory_ids: list[UUID],
    ) -> list[BaseMemory]:
        ...


class MemoryStore(Protocol):
    async def get_memories_by_ids(
        self,
        *,
        agent_id: str,
        memory_ids: list[UUID],
    ) -> list[BaseMemory]:
        ...
