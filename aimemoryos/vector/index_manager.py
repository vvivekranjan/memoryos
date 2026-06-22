from __future__ import annotations

from dataclasses import dataclass
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from enum import Enum
from pathlib import Path
import pickle
import os
from uuid import UUID
from typing import Any
from uuid import UUID

import faiss
import numpy as np

from aimemoryos.memory.models import (
    BaseMemory,
)

class RecencyBucketEnum(
    str,
    Enum,
):
    """
    Temporal partition buckets.
    """

    HOT = "HOT"
    WARM = "WARM"
    COLD = "COLD"

@dataclass(slots=True, frozen=True)
class PartitionKey:
    """
    FAISS partition identifier.
    """

    memory_type: str
    recency_bucket: RecencyBucketEnum

    @property
    def key(self) -> str:
        return (
            f"{self.memory_type}:"
            f"{self.recency_bucket.value}"
        )

@dataclass(slots=True)
class PartitionSearchHit:
    """
    Unified partition search result.
    """

    memory_id: UUID
    score: float
    partition: str
    metadata: dict[str, Any]

@dataclass(slots=True)
class IndexPartition:
    """
    FAISS partition wrapper.
    """

    key: PartitionKey
    index: faiss.IndexFlatIP
    memory_ids: list[UUID]
    metadata: dict[
        UUID,
        dict[str, Any],
    ]

class IndexManager:
    """
    FAISS partition coordinator.

    Responsibilities:
    - partition routing
    - recency bucketing
    - index isolation
    - multi-partition search
    - partition persistence

    Partition strategy:
    memory_type + recency_bucket

    Does NOT:
    - generate embeddings
    - rerank results
    - access replay
    """

    def __init__(
        self,
        *,
        dimension: int,
        base_path: str,
    ):
        self.dimension = dimension
        self.base_path = Path(base_path)
        self.base_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._partitions: dict[
            str,
            IndexPartition,
        ] = {}

        # attempt to load persisted partitions from disk
        self._load_partitions()

    def add(
        self,
        *,
        memory: BaseMemory,
        embedding: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Routes vector into partition.
        """

        partition_key = self._build_partition_key(memory)
        partition = self._get_or_create_partition(partition_key)
        vector = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(1, -1)

        # validate embedding dimension
        if vector.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {vector.shape[1]} does not match index dimension {self.dimension}"
            )

        faiss.normalize_L2(vector)
        partition.index.add(vector)
        partition.memory_ids.append(memory.memory_id)
        partition.metadata[memory.memory_id] = metadata or {}

        # persist updated partition to disk
        try:
            self._persist_partition(partition)
        except Exception:
            # persistence failure should not interrupt runtime flow
            pass

    def search(
        self,
        *,
        embedding: np.ndarray,
        top_k: int = 10,
        memory_type: str | None = None,
        include_cold: bool = True,
    ) -> list[
        PartitionSearchHit
    ]:
        """
        Multi-partition ANN search.
        """

        vector = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(1, -1)

        faiss.normalize_L2(vector)

        candidates: list[PartitionSearchHit] = []

        for partition in (
            self._eligible_partitions(
                memory_type=memory_type,
                include_cold=(
                    include_cold
                ),
            )
        ):

            if partition.index.ntotal == 0:
                continue

            scores, indices = (
                partition.index.search(
                    vector,
                    top_k,
                )
            )

            for score, idx in zip(
                scores[0],
                indices[0],
            ):

                if idx < 0:
                    continue

                # faiss may return indices that are out of range if
                # the index changed; guard against IndexError
                if idx >= len(partition.memory_ids):
                    continue

                memory_id = partition.memory_ids[idx]

                candidates.append(
                    PartitionSearchHit(
                        memory_id=memory_id,
                        score=float(score),
                        partition=(
                            partition
                            .key
                            .key
                        ),
                        metadata=(
                            partition
                            .metadata
                            .get(
                                memory_id,
                                {},
                            )
                        ),
                    )
                )

        return sorted(
            candidates,
            key=lambda x: x.score,
            reverse=True,
        )[:top_k]

    def _build_partition_key(
        self,
        memory: BaseMemory,
    ) -> PartitionKey:
        """
        Assigns temporal partition.
        """

        now = datetime.now(timezone.utc)

        created_at = getattr(memory, "created_at")

        # ensure created_at is timezone-aware; assume UTC if naive
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        age = now - created_at

        if age <= timedelta(days=1):

            bucket = RecencyBucketEnum.HOT

        elif age <= timedelta(days=30):

            bucket = RecencyBucketEnum.WARM

        else:

            bucket = RecencyBucketEnum.COLD

        return PartitionKey(
            memory_type=str(memory.memory_type),
            recency_bucket=bucket,
        )

    def _get_or_create_partition(
        self,
        key: PartitionKey,
    ) -> IndexPartition:
        """
        Lazy partition creation.
        """

        if key.key in self._partitions:

            return self._partitions[key.key]

        index = faiss.IndexFlatIP(self.dimension)

        partition = (
            IndexPartition(
                key=key,
                index=index,
                memory_ids=[],
                metadata={},
            )
        )

        self._partitions[key.key] = partition

        # persist newly created empty partition so it can be reloaded
        try:
            self._persist_partition(partition)
        except Exception:
            pass
        return partition

    def _eligible_partitions(
        self,
        *,
        memory_type: str | None,
        include_cold: bool,
    ) -> list[IndexPartition]:
        """
        Runtime partition filtering.
        """

        partitions = []

        for partition in (
            self._partitions.values()
        ):

            if (
                memory_type
                and partition.key
                .memory_type
                != memory_type
            ):
                continue

            if (
                not include_cold
                and partition.key
                .recency_bucket
                == RecencyBucketEnum
                .COLD
            ):
                continue

            partitions.append(partition)

        return partitions

    def _partition_safe_name(self, key: str) -> str:
        # replace characters not safe for filenames (':' on Windows)
        return key.replace(":", "__")

    def _persist_partition(self, partition: IndexPartition) -> None:
        """Persist FAISS index and partition metadata to disk."""

        safe_name = self._partition_safe_name(partition.key.key)
        index_path = self.base_path / f"{safe_name}.index"
        meta_path = self.base_path / f"{safe_name}.meta"

        # write faiss index
        faiss.write_index(partition.index, str(index_path))

        # prepare metadata serializable form
        serial_meta = {
            "memory_ids": [str(m) for m in partition.memory_ids],
            "metadata": {str(k): v for k, v in partition.metadata.items()},
            "key": partition.key.key,
        }

        with open(meta_path, "wb") as fh:
            pickle.dump(serial_meta, fh)

    def _load_partitions(self) -> None:
        """Load persisted partitions from disk if present."""

        for meta_file in self.base_path.glob("*.meta"):
            try:
                with open(meta_file, "rb") as fh:
                    serial_meta = pickle.load(fh)

                key_str = serial_meta.get("key")
                if not key_str:
                    # attempt to reconstruct from filename
                    key_str = meta_file.stem.replace("__", ":")

                safe_name = self._partition_safe_name(key_str)
                index_path = self.base_path / f"{safe_name}.index"

                if not index_path.exists():
                    continue

                index = faiss.read_index(str(index_path))

                memory_ids = [UUID(s) for s in serial_meta.get("memory_ids", [])]
                metadata = {UUID(k): v for k, v in serial_meta.get("metadata", {}).items()}

                mem_type, bucket_str = key_str.split(":", 1)

                partition_key = PartitionKey(
                    memory_type=mem_type,
                    recency_bucket=RecencyBucketEnum(bucket_str),
                )

                partition = IndexPartition(
                    key=partition_key,
                    index=index,
                    memory_ids=memory_ids,
                    metadata=metadata,
                )

                self._partitions[partition_key.key] = partition
            except Exception:
                # ignore corrupted partition files
                continue

    def partition_count(
        self,
    ) -> int:
        """
        Active partition count.
        """

        return len(self._partitions)

    def vector_count(
        self,
    ) -> int:
        """
        Total indexed vectors.
        """

        return sum(
            partition.index.ntotal
            for partition
            in self._partitions
            .values()
        )

    def stats(
        self,
    ) -> dict[str, Any]:
        """
        Partition runtime metrics.
        """

        return {
            "partitions": {
                key: {
                    "vectors": int(
                        partition
                        .index
                        .ntotal
                    ),
                    "memory_type": (
                        partition
                        .key
                        .memory_type
                    ),
                    "bucket": (
                        partition
                        .key
                        .recency_bucket
                        .value
                    ),
                }
                for key, partition
                in self._partitions.items()
            },
            "total_vectors": (
                self.vector_count()
            ),
        }

