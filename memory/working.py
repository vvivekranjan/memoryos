from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from memory.models import (
    BaseMemory,
    MemoryTypeEnum,
)

DEFAULT_TTL_MINUTES = 30

DEFAULT_PRIORITY = 0.5

def _content_sha256(content: str) -> str:
    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


class WorkingMemory(BaseMemory):
    """
    Short-lived active cognition memory.

    Working memories represent:
    - active context
    - recent conversational state
    - temporary reasoning artifacts
    - short-term cognitive focus

    Scope:
    - volatile state
    - rapid retrieval priority
    - TTL expiration
    - replay-safe structure
    """

    memory_type: MemoryTypeEnum = MemoryTypeEnum.WORKING
    expires_at: datetime | None = None
    ttl_minutes: int = DEFAULT_TTL_MINUTES
    working_priority: float = DEFAULT_PRIORITY
    session_id: UUID | None = None
    conversation_turn: int = 0
    attention_weight: float = 1.0
    volatile: bool = True
    pinned: bool = False
    access_count: int = 0

    def mark_accessed(self) -> None:
        """
        Updates working-memory access metadata.
        """

        self.access_count += 1
        self.last_accessed_at = datetime.now(timezone.utc)

    def refresh_ttl(self) -> None:
        """
        Extends expiration window.
        """

        self.expires_at = (
            datetime.now(
                timezone.utc
            )
            + timedelta(
                minutes=self.ttl_minutes
            )
        )

    def is_expired(self) -> bool:
        """
        TTL expiration check.
        """

        if self.pinned:
            return False

        if self.expires_at is None:
            return False

        return (
            datetime.now(
                timezone.utc
            )
            >= self.expires_at
        )

    def pin(self) -> None:
        """
        Prevents expiration eviction.
        """

        self.pinned = True

    def unpin(self) -> None:
        """
        Re-enables expiration.
        """

        self.pinned = False

    def to_context_dict(
        self,
    ) -> dict[str, Any]:
        """
        Replay-safe context serialization.
        """

        return {
            "memory_id": str(self.memory_id),
            "agent_id": self.agent_id,
            "memory_type": _enum_value(self.memory_type),
            "content": self.content,
            "importance_score": self.importance_score,
            "created_at": (
                self.created_at
                .isoformat()
            ),
            "last_accessed_at": (
                self.last_accessed_at
                .isoformat()
                if self.last_accessed_at
                else None
            ),
            "expires_at": (
                self.expires_at
                .isoformat()
                if self.expires_at
                else None
            ),
            "ttl_minutes": self.ttl_minutes,
            "working_priority": self.working_priority,
            "session_id": (
                str(self.session_id)
                if self.session_id
                else None
            ),
            "conversation_turn": self.conversation_turn,
            "attention_weight": self.attention_weight,
            "volatile": self.volatile,
            "pinned": self.pinned,
            "access_count": self.access_count,
            "metadata": self.metadata,
        }

    @classmethod
    def create(
        cls,
        *,
        memory_id: UUID,
        agent_id: str,
        content: str,
        importance_score: float,
        session_id: (
            UUID | None
        ) = None,
        conversation_turn: int = 0,
        ttl_minutes: int = (
            DEFAULT_TTL_MINUTES
        ),
        metadata: (
            dict[str, Any]
            | None
        ) = None,
    ) -> "WorkingMemory":
        """
        Deterministic working-memory
        constructor.
        """

        now = datetime.now(timezone.utc)

        return cls(
            memory_id=memory_id,
            agent_id=agent_id,
            memory_type=MemoryTypeEnum.WORKING,
            content=content,
            sha256=_content_sha256(content),
            importance_score=importance_score,
            created_at=now,
            last_accessed_at=now,
            expires_at=(
                now
                + timedelta(
                    minutes=ttl_minutes
                )
            ),
            ttl_minutes=ttl_minutes,
            session_id=session_id,
            conversation_turn=conversation_turn,
            metadata=metadata or {},
        )

