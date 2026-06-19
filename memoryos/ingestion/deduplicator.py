from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from memoryos.core.exceptions import InvalidContentError

class DedupStore(Protocol):
    """
    Storage abstraction for dedup checks.
    """

    async def content_hash_exists(
        self,
        *,
        agent_id: str,
        content_hash: str,
    ) -> bool:
        ...

@dataclass(slots=True)
class DeduplicationResult:
    """
    Result of deduplication check.
    """

    content_hash: str
    is_duplicate: bool
    chars_checked: int

class Deduplicator:
    """
    SHA-256 deduplication stage.

    ACS HARD REQUIREMENT:
    dedup MUST occur before embedding.

    Responsibilities:
    - deterministic hashing
    - duplicate detection
    - duplicate reporting

    Does NOT:
    - embed content
    - write storage
    - orchestrate ingestion
    """

    def __init__(
        self,
        *,
        store: DedupStore,
    ):
        self.store = store
    
    async def check_content(
        self,
        *,
        agent_id: str,
        content: str,
    ) -> DeduplicationResult:
        """
        Full deduplication check.
        """

        self._validate_content(content)

        content_hash = self.compute_hash(content)

        is_duplicate = await self.is_duplicate(
            agent_id=agent_id,
            content_hash=content_hash,
        )

        return DeduplicationResult(
            content_hash=content_hash,
            is_duplicate=is_duplicate,
            chars_checked=len(content),
        )
    
    @staticmethod
    def compute_hash(
        content: str,
    ) -> str:
        """
        Deterministic SHA-256 hash.

        IMPORTANT:
        Hash processed_content,
        NOT raw content.
        """

        if not isinstance(content, str):
            content = str(content)

        content = content.strip()

        encoded = content.encode("utf-8")

        return hashlib.sha256(
            encoded
        ).hexdigest()
    
    async def is_duplicate(
        self,
        *,
        agent_id: str,
        content_hash: str,
    ) -> bool:
        """
        Queries storage layer for existing hash.
        """

        return await self.store.content_hash_exists(
            agent_id=agent_id,
            content_hash=content_hash,
        )
    
    @staticmethod
    def _validate_content(
        content: str,
    ) -> None:

        if not isinstance(content, str):
            raise InvalidContentError(
                "Content must be string"
            )

        if not content.strip():
            raise InvalidContentError(
                "Content cannot be empty"
            )

