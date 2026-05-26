from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pydantic import Field

from memory.models import (
    BaseMemory,
    MemoryTypeEnum,
)

def _content_sha256(content: str) -> str:
    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


class EpisodicMemory(BaseMemory):
    """
    Canonical autobiographical memory.

    Episodic memories represent:
    - experiences
    - observations
    - interactions
    - temporally grounded events
    """

    memory_type: MemoryTypeEnum = MemoryTypeEnum.EPISODIC
    experienced_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    source: str | None = None
    participants: list[str] = Field(default_factory=list)
    context_window_id: UUID | None = None
    access_count: int = 0
    reinforcement_count: int = 0

    def reinforce(self) -> None:
        """
        Reinforces episodic memory.
        """

        self.reinforcement_count += 1
        self.access_count += 1
        self.last_accessed_at = datetime.now(timezone.utc)

    def mark_accessed(self) -> None:
        """
        Updates retrieval metadata.
        """

        self.access_count += 1
        self.last_accessed_at = datetime.now(timezone.utc)

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
            "experienced_at": (
                self.experienced_at
                .isoformat()
            ),
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
            "source": self.source,
            "participants": self.participants,
            "access_count": self.access_count,
            "reinforcement_count": self.reinforcement_count,
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
        metadata: (
            dict[str, Any]
            | None
        ) = None,
        source: str | None = None,
    ) -> "EpisodicMemory":
        """
        Deterministic episodic constructor.
        """

        now = datetime.now(timezone.utc)

        return cls(
            memory_id=memory_id,
            agent_id=agent_id,
            memory_type=MemoryTypeEnum.EPISODIC,
            content=content,
            sha256=_content_sha256(content),
            importance_score=importance_score,
            created_at=now,
            last_accessed_at=now,
            experienced_at=now,
            source=source,
            metadata=metadata or {},
        )

