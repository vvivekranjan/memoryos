from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
import re

import duckdb
import numpy as np

from aimemoryos.vector.embedder import (
    Embedder,
)

DEFAULT_CACHE_TABLE = "embedding_cache"

@dataclass(slots=True)
class CacheEmbeddingResult:
    """
    Cached embedding lookup result.
    """

    hit: bool
    embedding: np.ndarray | None

class CacheEmbedder:
    """
    DuckDB-backed embedding cache.

    Responsibilities:
    - SHA-256 text hashing
    - embedding persistence
    - cache lookup
    - duplicate embedding prevention

    Flow:
    text
        ↓
    SHA-256
        ↓
    DuckDB cache lookup
        ↓
    embedder fallback
        ↓
    cache persist

    Does NOT:
    - perform ANN retrieval
    - own vector indices
    - own chunking
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        db_path: str,
        table_name: str = (
            DEFAULT_CACHE_TABLE
        ),
    ):
        self.embedder = embedder
        self.db_path = Path(db_path)
        self.table_name = table_name
        # validate table name to avoid SQL injection via identifier
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", self.table_name):
            raise ValueError("Invalid table_name; must be alphanumeric or underscore and not start with a digit")
        self.connection = duckdb.connect(str(self.db_path))
        self._initialize()

    def _initialize(self) -> None:
        """
        Initializes cache table.
        """

        self.connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS
            {self.table_name} (
                cache_key TEXT PRIMARY KEY,
                source_text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dimension INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


    @staticmethod
    def hash_text(
        text: str,
    ) -> str:
        """
        Stable SHA-256 cache key.
        """

        normalized = text.strip().encode("utf-8")

        return hashlib.sha256(
            normalized
        ).hexdigest()


    async def embed_text(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Cached embedding generation.
        """

        cache_key = self.hash_text(text)
        cached = self._lookup(cache_key)

        if cached.hit:

            return cached.embedding

        embedding = await self.embedder.embed_text(text)

        self._store(
            cache_key=cache_key,
            source_text=text,
            embedding=embedding,
        )

        return embedding

    async def embed_batch(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """
        Batch cached embeddings.
        """

        embeddings = []

        for text in texts:

            embedding = await self.embed_text(text)
            embeddings.append(embedding)

        return np.vstack(embeddings)

    def _lookup(
        self,
        cache_key: str,
    ) -> CacheEmbeddingResult:
        """
        Cache lookup by SHA key.
        """

        result = (
            self.connection.execute(
                f"""
                SELECT embedding
                FROM {self.table_name}
                WHERE cache_key = ?
                """,
                [cache_key],
            ).fetchone()
        )

        if result is None:

            return (
                CacheEmbeddingResult(
                    hit=False,
                    embedding=None,
                )
            )

        embedding = pickle.loads(result[0])

        return CacheEmbeddingResult(
            hit=True,
            embedding=embedding,
        )

    def _store(
        self,
        *,
        cache_key: str,
        source_text: str,
        embedding: np.ndarray,
    ) -> None:
        """
        Persists embedding cache.
        """

        serialized = pickle.dumps(embedding)
        # DuckDB does not support SQLite's `INSERT OR REPLACE` syntax reliably.
        # Do a safe upsert: delete any existing row with the same primary key
        # then insert the new row. Table name is validated in __init__.
        self.connection.execute(
            f"DELETE FROM {self.table_name} WHERE cache_key = ?",
            [cache_key],
        )

        self.connection.execute(
            f"INSERT INTO {self.table_name} (cache_key, source_text, embedding, dimension) VALUES (?, ?, ?, ?)",
            [
                cache_key,
                source_text,
                serialized,
                int(embedding.shape[0]),
            ],
        )

    def exists(
        self,
        text: str,
    ) -> bool:
        """
        Cache existence check.
        """

        cache_key = self.hash_text(text)

        result = (
            self.connection.execute(
                f"""
                SELECT 1
                FROM {self.table_name}
                WHERE cache_key = ?
                LIMIT 1
                """,
                [cache_key],
            ).fetchone()
        )

        return result is not None

    def count(self) -> int:
        """
        Total cached embeddings.
        """

        result = (
            self.connection.execute(
                f"""
                SELECT COUNT(*)
                FROM {self.table_name}
                """
            ).fetchone()
        )

        return int(result[0])

    def clear(self) -> None:
        """
        Clears embedding cache.
        """

        self.connection.execute(
            f"""
            DELETE FROM
            {self.table_name}
            """
        )

    def close(self) -> None:
        """
        Closes DuckDB connection.
        """

        self.connection.close()

