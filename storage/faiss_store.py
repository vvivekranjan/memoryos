from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, TypedDict
from uuid import UUID
from memory.models import LifecycleStateEnum, MemoryTypeEnum
from vector.embedder import DEFAULT_DIMENSION

import json
import numpy as np
import faiss
import asyncio

VECTOR_DIMENSION = DEFAULT_DIMENSION
FAISS_ROOT = Path("data/faiss")
INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"
INDEX_CONFIG = {
    MemoryTypeEnum.EPISODIC: "episodic",
    MemoryTypeEnum.SEMANTIC: "semantic",
    MemoryTypeEnum.PROCEDURAL: "procedural",
    MemoryTypeEnum.WORKING: "working",
}


class VectorStoreError(Exception):
    """Base vector store error."""


class VectorDimensionError(VectorStoreError):
    """Raised when embedding dimension mismatch occurs."""


class IndexNotInitialisedError(VectorStoreError):
    """Raised when index missing."""

class VectorStore(Protocol):

    def initialise(self) -> None:
        ...
    
    def add_embedding(
        self,
        *,
        memory_id: UUID,
        agent_id: str,
        memory_type: MemoryTypeEnum,
        embedding: list[float],
        lifecycle_state: LifecycleStateEnum = LifecycleStateEnum.ACTIVE,
    ) -> None:
        ...
    
    def remove_embedding(
        self,
        *,
        memory_id: UUID,
        memory_type: MemoryTypeEnum,
    ) -> None:
        ...
    
    def persist(self) -> None:
        ...
    
    def load(self) -> None:
        ...


def normalise_embedding(embedding: list[float]) -> np.ndarray:
    """Converts embedding to float32 normalized vector."""

    array = np.array(embedding, dtype=np.float32)

    if array.ndim != 1:
        raise VectorDimensionError("Embedding must be 1-dimensional")

    if len(array) != VECTOR_DIMENSION:
        raise VectorDimensionError(
            f"Expected dimension={VECTOR_DIMENSION}, got={len(array)}"
        )

    norm = np.linalg.norm(array)
    if norm == 0:
        raise VectorDimensionError("Zero-norm embedding")

    return (array / norm).reshape(1, -1)

def uuid_to_int64(value: UUID) -> int:
    """
    FAISS requires int64 IDs.
    """
    return value.int % (2**63 - 1)


def _coerce_memory_type(value: MemoryTypeEnum | str) -> MemoryTypeEnum:
    if isinstance(value, MemoryTypeEnum):
        return value
    return MemoryTypeEnum(str(value))


def _coerce_lifecycle_state(
    value: LifecycleStateEnum | str,
) -> LifecycleStateEnum:
    if isinstance(value, LifecycleStateEnum):
        return value
    return LifecycleStateEnum(str(value))

class SearchResult(TypedDict):
    vector_id: int
    score: float
    metadata: dict[str, Any]


class FAISSStore:
    """Persistent FAISS index with metadata mapping."""

    def __init__(self, root_path: Path = FAISS_ROOT):
        self.root_path = root_path
        # legacy single-index for compatibility
        self.index: faiss.IndexFlatIP | None = None
        # named indices for advanced usage
        self.indices: dict[str, faiss.IndexFlatIP] = {}
        # global metadata map: vector_id -> metadata
        self.metadata: dict[int, dict[str, Any]] = {}

    @property
    def index_path(self) -> Path:
        return self.root_path / INDEX_FILENAME

    @property
    def metadata_path(self) -> Path:
        return self.root_path / METADATA_FILENAME

    def initialise(self) -> None:
        """Creates empty index and metadata if absent."""

        self.root_path.mkdir(parents=True, exist_ok=True)

        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
            except Exception:
                self.index = faiss.IndexFlatIP(VECTOR_DIMENSION)
        else:
            self.index = faiss.IndexFlatIP(VECTOR_DIMENSION)

        for idx_name in INDEX_CONFIG.values():
            index_path = self.root_path / f"{idx_name}.faiss"
            if index_path.exists():
                try:
                    self.indices[idx_name] = faiss.read_index(str(index_path))
                except Exception:
                    self.indices[idx_name] = self.indices.get(idx_name, faiss.IndexFlatIP(VECTOR_DIMENSION))
            else:
                self.indices[idx_name] = faiss.IndexFlatIP(VECTOR_DIMENSION)

        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.metadata = {int(k): v for k, v in raw.items()}
        else:
            self.metadata = {}

    def load(self) -> None:
        """Loads persisted index and ID mappings, or initializes empty state."""

        self.initialise()

    def persist(self) -> None:
        """Persists index and metadata to disk."""

        if self.index is None:
            raise IndexNotInitialisedError("FAISS index is not initialised")

        self.root_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(self.index_path))

        for name, index in self.indices.items():
            try:
                faiss.write_index(index, str(self.root_path / f"{name}.faiss"))
            except Exception:
                pass

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=True, indent=2)

    def add_embedding(
        self,
        *,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        # backwards-compatible optional args
        memory_id: UUID | None = None,
        agent_id: str | None = None,
        memory_type: MemoryTypeEnum | str | None = None,
        lifecycle_state: LifecycleStateEnum = LifecycleStateEnum.ACTIVE,
    ) -> int:
        """Inserts a single embedding and returns its vector id."""

        if self.index is None:
            raise IndexNotInitialisedError("Call load() or initialise() before adding")

        embedding_array = normalise_embedding(embedding)
        vector_id = int(self.index.ntotal)
        self.index.add(embedding_array)
        self.metadata[vector_id] = metadata or {}

        # If memory_type provided, also add to named index (best-effort)
        if memory_type is not None:
            try:
                mtype = _coerce_memory_type(memory_type)
                name = INDEX_CONFIG.get(mtype)
                if name and name in self.indices:
                    self.indices[name].add(embedding_array)
            except Exception:
                # don't fail the entire write for index partitioning
                pass

        return vector_id

    def add_embeddings(
        self,
        *,
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Batch inserts embeddings with optional metadata list."""
        if metadatas is not None and len(metadatas) != len(embeddings):
            raise VectorStoreError("metadatas length must match embeddings length")

        if self.index is None:
            raise IndexNotInitialisedError("Call load() or initialise() before adding")

        if not embeddings:
            return []

        # Convert to numpy array and normalize all at once
        arr = np.asarray(embeddings, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.shape[1] != VECTOR_DIMENSION:
            raise VectorDimensionError(
                f"Expected dimension={VECTOR_DIMENSION}, got={arr.shape[1]}"
            )

        norms = np.linalg.norm(arr, axis=1)
        if np.any(norms == 0):
            raise VectorDimensionError("Zero-norm embedding in batch")

        arr = (arr / norms.reshape(-1, 1)).astype(np.float32)

        start_id = int(self.index.ntotal)
        self.index.add(arr)

        vector_ids: list[int] = []
        for i in range(arr.shape[0]):
            vid = start_id + i
            vector_ids.append(int(vid))
            self.metadata[int(vid)] = (metadatas[i] if metadatas is not None else {})

        return vector_ids

    def search(
        self,
        *,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Returns top_k nearest vectors with score and metadata."""

        if self.index is None:
            raise IndexNotInitialisedError("Call load() or initialise() before searching")

        if top_k <= 0 or self.index.ntotal == 0:
            return []

        query = normalise_embedding(query_embedding)
        k = min(top_k, int(self.index.ntotal))
        distances, ids = self.index.search(query, k)

        results: list[SearchResult] = []
        for score, vector_id in zip(distances[0], ids[0]):
            if vector_id < 0:
                continue
            results.append(
                SearchResult(
                    vector_id=int(vector_id),
                    score=float(score),
                    metadata=self.metadata.get(int(vector_id), {}),
                )
            )

        return results

    async def search_async(
        self,
        *,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Async wrapper around search for retrieval callers."""

        return await asyncio.to_thread(
            self.search,
            query_embedding=query_embedding,
            top_k=top_k,
        )

    async def add_embedding_async(
        self,
        *,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
        memory_id: UUID | None = None,
        agent_id: str | None = None,
        memory_type: MemoryTypeEnum | str | None = None,
        lifecycle_state: LifecycleStateEnum = LifecycleStateEnum.ACTIVE,
    ) -> int:
        """Async wrapper around add_embedding for ingestion callers."""

        return await asyncio.to_thread(
            self.add_embedding,
            embedding=embedding,
            metadata=metadata,
            memory_id=memory_id,
            agent_id=agent_id,
            memory_type=memory_type,
            lifecycle_state=lifecycle_state,
        )

    async def add_embeddings_async(
        self,
        *,
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Async wrapper around add_embeddings for batch ingestion."""

        return await asyncio.to_thread(
            self.add_embeddings,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    
    # Rebuild
    def clear(self) -> None:
        """
        Clears all indices.
        """

        self.indices.clear()
        self.metadata.clear()

        self.initialise()
    
    def stats(self) -> dict:
        """
        Basic operational stats.
        """

        return {
            "total_vectors": len(self.metadata),
            "indices": {
                name: index.ntotal
                for name, index in self.indices.items()
            },
        }

