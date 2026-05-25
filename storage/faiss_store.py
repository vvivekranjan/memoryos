from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict
from vector.embedder import DEFAULT_DIMENSION

import json
import numpy as np
import faiss
import asyncio

VECTOR_DIMENSION = DEFAULT_DIMENSION
FAISS_ROOT = Path("data/faiss")
INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"


class VectorStoreError(Exception):
    """Base vector store error."""


class VectorDimensionError(VectorStoreError):
    """Raised when embedding dimension mismatch occurs."""


class IndexNotInitialisedError(VectorStoreError):
    """Raised when index missing."""


class SearchResult(TypedDict):
    vector_id: int
    score: float
    metadata: dict[str, Any]


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


class FAISSStore:
    """Persistent FAISS index with metadata mapping."""

    def __init__(self, root_path: Path = FAISS_ROOT):
        self.root_path = root_path
        self.index: faiss.IndexFlatIP | None = None
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
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = faiss.IndexFlatIP(VECTOR_DIMENSION)

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

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=True, indent=2)

    def add_embedding(
        self,
        *,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Inserts a single embedding and returns its vector id."""

        if self.index is None:
            raise IndexNotInitialisedError("Call load() or initialise() before adding")

        embedding_array = normalise_embedding(embedding)
        vector_id = int(self.index.ntotal)
        self.index.add(embedding_array)
        self.metadata[vector_id] = metadata or {}
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

    # Async wrappers -----------------------------------------------------
    async def add_embeddings_async(
        self,
        *,
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Async wrapper around `add_embeddings` using a thread executor."""

        return await asyncio.to_thread(
            self.add_embeddings, embeddings=embeddings, metadatas=metadatas
        )

    async def add_embedding_async(
        self, *, embedding: list[float], metadata: dict[str, Any] | None = None
    ) -> int:
        """Async wrapper around `add_embedding` using a thread executor."""

        return await asyncio.to_thread(self.add_embedding, embedding=embedding, metadata=metadata)

    async def search_async(self, *, query_embedding: list[float], top_k: int = 10) -> list[SearchResult]:
        """Async wrapper around `search` using a thread executor."""

        return await asyncio.to_thread(self.search, query_embedding=query_embedding, top_k=top_k)
