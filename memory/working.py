from __future__ import annotations
from datetime import datetime, timedelta
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import Field, field_validator

from memory.models import BaseMemory, MemoryTypeEnum

class WorkingMemory(BaseMemory):
    """
    Session-scoped scratch memory.

    WorkingMemory MUST NEVER be written to FAISS
    unless promoted_to is set.

    Enforcement occurs in:
        storage/orchestrator.py

    Not in the schema layer.
    """

    memory_type: Literal[MemoryTypeEnum.WORKING] = (MemoryTypeEnum.WORKING)
    session_id: UUID
    ttl_seconds: int = 3600 # cleared at session end or TTL expiry
    promoted_to: Optional[UUID] = None # set on explicit promotion
    scratch_data: dict[str, Any] = Field(default_factory=dict) # unstructured in-session state
    expires_at: datetime | None = None # created_at + ttl_seconds

    @field_validator("ttl_seconds")
    @classmethod
    def validate_ttl_seconds(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("ttl_seconds must be > 0")

        return value
    
    def model_post_init(self, __context: Any) -> None:
        if self.expires_at is None:
            self.expires_at = (
                self.created_at
                + timedelta(seconds=self.ttl_seconds)
            )

