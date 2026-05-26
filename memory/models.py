from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator
from uuid import UUID, uuid4
from enum import Enum
from datetime import datetime, timezone, timedelta
from typing import Optional, Any, Literal

import re

class MemoryTypeEnum(Enum):
    EPISODIC = "EPISODIC"
    WORKING = "WORKING"

class SpeakerRoleEnum(str, Enum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    SYSTEM = "SYSTEM"
    TOOL = "TOOL"

class EventTypeEnum(str, Enum):
    MEMORY_INGESTED = "MEMORY_INGESTED"
    MEMORY_RETRIEVED = "MEMORY_RETRIEVED"

class BaseMemory(BaseModel):
    """
    Root schema for all memory types.

    MemoryOS v1.0
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
    )

    # Identity
    memory_id: UUID = Field(default_factory=uuid4)
    schema_version: str = "1.0.0"
    memory_type: MemoryTypeEnum
    agent_id: str

    # Content
    content: str # normalised UTF-8 text
    sha256: str # SHA-256 of normalised content

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc)) # updated on every retrieval
    access_count: int = 0 # monotonically increasing

    # Importance
    importance_score: float # [0.0, 1.0]; recomputed on retrieval

    # Extensible
    metadata: dict[str, Any] = Field(default_factory=dict) # schema_version-gated bag

    # Validators
    @field_validator("agent_id")
    @classmethod
    def validate_agent_id(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("agent_id cannot be empty")

        return value
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("content cannot be empty")

        return value
    
    @field_validator('sha256')
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        value = value.lower().strip()

        if not re.fullmatch(r"^[a-f0-9]{64}$", value):
            raise ValueError("sha256 must be exactly 64 lowercase hex characters")
        
        return value
    
    @field_validator("importance_score")
    @classmethod
    def validate_importance_score(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "importance_score must be between 0.0 and 1.0"
            )

        return value
    
    @field_validator("access_count")
    @classmethod
    def validate_access_count(cls, value: int) -> int:
        if value < 0:
            raise ValueError(
                "access_count cannot be negative"
            )

        return value

class EpisodicMemory(BaseMemory):
    """
    Canonical replay memory.

    Used for:
    - reconstruct_state_at()
    - session replay
    - conversational chronology
    - reflection clustering
    """

    memory_type: Literal[MemoryTypeEnum.EPISODIC] = (MemoryTypeEnum.EPISODIC)
    session_id: UUID # groups turns in one session
    turn_index: int # 0-based; ordered within session
    speaker_role: SpeakerRoleEnum
    referenced_memory_ids: list[UUID] = Field(default_factory=list) # explicit cross-refs in this turn
    is_system_message: bool = False
    tool_call_id: Optional[str] = None # if speaker_role=TOOL

    @field_validator("turn_index")
    @classmethod
    def validate_turn_index(cls, value: int) -> int:
        if value < 0:
            raise ValueError("turn_index cannot be negative")
        
        return value

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

class IngestionPayload(BaseModel):

    model_config = ConfigDict(extra="forbid")
    memory_id: UUID
    memory_type: str
    sha256: str
    chunks_created: int
    pipeline_stages_ms: dict[str, float] # stage_name -> latency
    entity_count: int = 0
    relation_count: int = 0

class RetrievalPayload(BaseModel):
    
    model_config = ConfigDict(extra="forbid")
    memory_id: UUID
    importance_before: float
    importance_after: float
    access_count_after: int
    retrieval_score: float

# Utilites
def utc_now() -> datetime:
    return datetime.now(timezone.utc)
