from __future__ import annotations
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator
from uuid import UUID, uuid4
from enum import Enum
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Any, Literal

import re

class MemoryTypeEnum(str, Enum):
    EPISODIC = "EPISODIC"
    WORKING = "WORKING"
    SEMANTIC = "SEMANTIC"
    PROCEDURAL = "PROCEDURAL"

class LifecycleStateEnum(str, Enum):
    ACTIVE = "ACTIVE"
    STALE = "STALE"
    ARCHIVED = "ARCHIVED"
    PRUNED = "PRUNED"

class ModalityEnum(str, Enum):
    TEXT = "TEXT"
    DOC = "DOC"

class ProvenanceEnum(str, Enum):
    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    HYPOTHESISED = "HYPOTHESISED"
    IMAGINED = "IMAGINED"
    RECEIVED_TRUSTED = "RECEIVED_TRUSTED"
    RECEIVED_UNTRUSTED = "RECEIVED_UNTRUSTED"

class SpeakerRoleEnum(str, Enum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    SYSTEM = "SYSTEM"
    TOOL = "TOOL"

class EventTypeEnum(str, Enum):
    MEMORY_INGESTED = "MEMORY_INGESTED"
    MEMORY_RETRIEVED = "MEMORY_RETRIEVED"
    FEEDBACK_RECEIVED = "FEEDBACK_RECEIVED"
    MEMORY_LIFECYCLE_TRANSITION = "MEMORY_LIFECYCLE_TRANSITION"
    MEMORY_PRUNED = "MEMORY_PRUNED"

class FeedbackTypeEnum(str, Enum):
    CONFIRMED = "CONFIRMED"
    CORRECTION = "CORRECTION"
    IGNORED = "IGNORED"

class TriggerEnum(str, Enum):
    DECAY_SCHEDULAR = "DECAY_SCHEDULAR"
    REFLECTION_MERGE = "REFLECTION_MERGE"
    MANUAL = "MANUAL"
    COMPRESSION = "COMPRESSION"

class TransactionStateEnum(str, Enum):
    PENDING = "PENDING"
    COMMITTED = "COMMITTED"
    CORRUPTED = "CORRUPTED"
    FAILED = "FAILED"

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
    modality: ModalityEnum = ModalityEnum.TEXT

    # Lifecycle
    lifecycle_state: LifecycleStateEnum = LifecycleStateEnum.ACTIVE
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc)) # updated on every retrieval
    access_count: int = 0 # monotonically increasing
    decay_anchor: datetime = Field(default_factory=lambda: datetime.now(timezone.utc)) # reset on access; used by decay functions
    decay_multiplier: float = 1.0 # 1.0=normal; >1.0=interference-accelerated
    forward_ref: Optional[UUID] = None

    # Importance
    importance_score: float # [0.0, 1.0]; recomputed on retrieval
    salience_score: float = 0.5 # static salience from ingestion context

    # Affect (Stub until M6)
    vad_v: Optional[float] = None # valence   [-1, 1]
    vad_a: Optional[float] = None # arousal   [-1, 1]
    vad_d: Optional[float] = None # dominance [-1, 1]

    emotional_class: Optional[str] = None # GoEmotions 28-class label

    # Provenance
    provenance: ProvenanceEnum = ProvenanceEnum.OBSERVED
    provenance_confidence: float = 1.0 # [0.0, 1.0]

    source_ids: list[UUID] = Field(default_factory=list) # parent memory IDs

    # Graph
    graph_node_id: Optional[str] = None # KuzuDB node identifier

    # Extensible
    tags: list[str] = Field(default_factory=list)
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
    
    @field_validator("salience_score")
    @classmethod
    def validate_salience_score(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "salience_score must be between 0.0 and 1.0"
            )

        return value
    
    @field_validator("provenance_confidence")
    @classmethod
    def validate_provenance_confidence(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "provenance_confidence must be between 0.0 and 1.0"
            )

        return value
    
    @field_validator("decay_multiplier")
    @classmethod
    def validate_decay_multiplier(cls, value: float) -> float:
        if value < 1.0:
            raise ValueError(
                "decay_multiplier must be >= 1.0"
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
    
    @field_validator("vad_v", "vad_a", "vad_d")
    @classmethod
    def validate_vad_range(
        cls,
        value: Optional[float],
    ) -> Optional[float]:
        if value is None:
            return value

        if value < -1.0:
            return -1.0

        if value > 1.0:
            return 1.0

        return value

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        import re
        if not re.fullmatch(r"^\d+\.\d+\.\d+$", value):
            raise ValueError("schema_version must follow semantic versioning (e.g. 1.0.0)")
        return value

class BaseEvent(BaseModel):
    """
    Immutable append-only event.

    SQLite is the single source of truth.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )

    event_id: UUID = Field(default_factory=uuid4)
    schema_version: str = "1.0.0"
    event_type: EventTypeEnum
    agent_id: str
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    payload: dict[str, Any]
    checksum: str  # SHA-256(event_id + schema_version + event_type + occurred_at + sequence_num + payload)
    sequence_num: int # monotonically increasing per agent_id

    @field_validator("sequence_num")
    @classmethod
    def validate_sequence_num(cls, value: int) -> int:
        if value < 0:
            raise ValueError("sequence_num cannot be negative")
        
        return value

    @field_validator("event_id")
    @classmethod
    def validate_event_id(cls, value: UUID) -> UUID:
        if not isinstance(value, UUID):
            raise ValueError("event_id must be a UUID")
        return value

    @field_validator("agent_id")
    @classmethod
    def validate_base_agent_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("agent_id cannot be empty")
        return value

    @field_validator("payload")
    @classmethod
    def validate_base_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("payload must be a dictionary")
        return value

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        import re
        if not re.fullmatch(r"^\d+\.\d+\.\d+$", value):
            raise ValueError("schema_version must follow semantic versioning (e.g. 1.0.0)")
        return value

class IngestionPayload(BaseModel):

    model_config = ConfigDict(extra="forbid")
    memory_id: UUID
    memory_type: str
    sha256: str
    chunks_created: int
    modality: str
    pipeline_stages_ms: dict[str, float] # stage_name -> latency
    entity_count: int = 0
    relation_count: int = 0
    emotion_class: Optional[str] = None
    provenance: str = "OBSERVED"

    @field_validator("memory_type")
    @classmethod
    def validate_memory_type(cls, value: str) -> str:
        value = value.strip().upper()
        if value not in MemoryTypeEnum._value2member_map_:
            raise ValueError(f"memory_type must be one of {list({e.value for e in MemoryTypeEnum}.keys())}")
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        import re
        value = value.lower().strip()
        if not re.fullmatch(r"^[a-f0-9]{64}$", value):
            raise ValueError("sha256 must be exactly 64 lowercase hex characters")
        return value

    @field_validator("chunks_created")
    @classmethod
    def validate_chunks_created(cls, value: int) -> int:
        if value < 0:
            raise ValueError("chunks_created must be >= 0")
        return value

    @field_validator("entity_count", "relation_count")
    @classmethod
    def validate_non_negative_counts(cls, value: int) -> int:
        if value < 0:
            raise ValueError("count must be >= 0")
        return value

    @field_validator("pipeline_stages_ms")
    @classmethod
    def validate_pipeline_stages(cls, value: dict[str, float]) -> dict[str, float]:
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError("pipeline_stages_ms keys must be strings")
            if v < 0:
                raise ValueError("pipeline_stages_ms values must be >= 0")
        return value

    @field_validator("provenance")
    @classmethod
    def validate_provenance(cls, value: str) -> str:
        return value.strip().upper()

class RetrievalPayload(BaseModel):
    
    model_config = ConfigDict(extra="forbid")
    memory_id: UUID
    importance_before: float
    importance_after: float
    access_count_after: int
    retrieval_score: float

class FeedbackPayload(BaseModel):

    model_config = ConfigDict(extra="forbid")
    memory_id: UUID
    feedback_type: FeedbackTypeEnum
    feedback_text: Optional[str] = None
    submitted_by: Optional[str] = None

class LifecycleTransitionPayload(BaseModel):

    model_config = ConfigDict(extra="forbid")
    memory_id: UUID
    old_state: LifecycleStateEnum
    new_state: LifecycleStateEnum
    trigger: TriggerEnum
    importance_at_transition: float

class SessionScope(BaseModel):
    model_config = ConfigDict(strict=True)
    
    agent_id: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    working_memory_ids: List[str] = Field(default_factory=list)
    spreading_activation_seeds: List[str] = Field(default_factory=list)

# Utilites
def utc_now() -> datetime:
    return datetime.now(timezone.utc)
